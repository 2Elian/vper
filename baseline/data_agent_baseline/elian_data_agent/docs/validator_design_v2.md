# Validator Agent 设计方案（修订版）

## 1. 核心定位修正

**Validator 在工作流中的位置**：
- 在 `run()` 方法的 `for iteration` 循环**之外**执行
- 等价于：`plan.is_complete() == True` 或 `plan.all_steps_failed() == True` 之后
- 在 Phase 4 (`_synthesize_answer`) **之前**执行

```
Phase 1: Workspace 冷启动
    ↓
Phase 2: Planner 生成计划
    ↓
Phase 3: Executor ↔ Replanner 循环
         (直到 plan.is_complete() 或 all_steps_failed())
    ↓
┌──────────────────────────────────────────────┐
│ Phase 3.5: Validator 验证                   │
│ - 提取 plan 最终结果作为候选答案              │
│ - 调用 OpenJudge Grader 进行 TOT 评分        │
│ - 通过 → Phase 4                            │
│ - 失败 → 触发 Replanner 修正 → 回到 Phase 3  │
└──────────────────────────────────────────────┘
    ↓
Phase 4: 合成最终答案
```

**现有代码中的正确插入点**（在 `workflow.py` 的 `run()` 方法中）：

```python
# Phase 3 循环结束后 (line 263 附近)
# plan.is_complete() 或 plan.all_steps_failed() 退出循环后

# ========== 这里开始是 Validator 逻辑 ==========

# Phase 3.5: Validator 验证
# 提取候选答案
candidate_result = self._extract_candidate_result(plan)

if self.config.enable_validation and self.validator and candidate_result:
    validation_passed = False
    for val_round in range(self.config.max_validation_replans):
        # 调用 Validator (内嵌 OpenJudge TOT 评分)
        validation_result = self.validator.validate(
            question=question,
            candidate_result=candidate_result,
            knowledge=session.get("knowledge", ""),
            execution_trace=...,  # 完整历史
        )

        if validation_result.decision == "accept":
            validation_passed = True
            break

        # 验证失败 → 触发 Replanner 修正
        # ... Replanner logic ...

        # 修正后重新执行 Phase 3 循环
        # ...

# ========== 插入点结束 ==========

# Phase 4: 合成最终答案
final_result = self._synthesize_answer(task_id, plan, session, step_results)
```

---

## 2. 参考 OpenJudge 的设计

### 2.1 OpenJudge 核心组件映射

| OpenJudge 组件 | 本项目对应 | 说明 |
|--------------|----------|------|
| `BaseGrader` | 继承/组合 | 评分器基类 |
| `LLMGrader` | 组合 | 使用 LLM 进行评分 |
| `GraderScore(score, reason, metadata)` | `ValidationScore` | 评分结果数据结构 |
| `AverageEvaluationStrategy` | `TOTEvaluationStrategy` | 多轨迹评分策略 |
| `DirectEvaluationStrategy` | 单轨迹模式 | 可选降级 |
| Rubrics (prompt 模板) | 动态生成 | 基于 knowledge.md |

### 2.2 ValidatorAgent 与 OpenJudge 的关系

**两种方案对比**：

| 方案 | 优点 | 缺点 |
|-----|------|------|
| A: 直接继承 `LLMGrader` | 完全复用 OpenJudge 逻辑 | 需要适配异步接口（OpenJudge 全异步） |
| B: 组合 OpenJudge 组件 | 灵活，适配成本低 | 需要写适配层 |
| C: 参考 OpenJudge 思想自己实现 | 完全可控 | 造轮子（用户明确反对） |

**推荐方案 B**：组合 OpenJudge 的核心思想 + 适配层

- `ValidationScorer` 组合 `LLMGrader` 模式（prompt 模板 + 结构化输出）
- `TOTEvaluationStrategy` 实现多轨迹评分（复用 AverageEvaluationStrategy 的聚合思路）
- 不直接继承 `BaseGrader`（避免异步适配成本），但继承其**评分模式**

---

## 3. 详细设计

### 3.1 数据结构

```python
@dataclass
class ValidationScore:
    """评分结果（参考 OpenJudge GraderScore）"""
    score: float              # 0.0 - 1.0
    reason: str              # 评分理由
    metadata: Dict[str, Any] = field(default_factory=dict)

    # TOT 特有字段
    all_scores: List[float] = field(default_factory=list)  # 原始 N 个分数
    trimmed_scores: List[float] = field(default_factory=list)  # 去极值后的分数


@dataclass
class ValidationDecision:
    """验证决策"""
    decision: str   # "accept" 或 "reject"
    score: ValidationScore
    suggestion: str  # 修正建议（reject 时有）
```

### 3.2 ValidationScorer（参考 OpenJudge LLMGrader）

```python
class ValidationScorer:
    """
    参考 OpenJudge LLMGrader 的评分器

    核心职责：
    1. 构建评分 prompt（使用 RubricTemplate）
    2. 调用 LLM 获取结构化评分
    3. 解析 JSON 输出为 ValidationScore
    """

    def __init__(self, model, rubric_template: str):
        self.model = model
        self.rubric_template = rubric_template

    def score(
        self,
        question: str,
        candidate_result: Dict[str, Any],
        knowledge: str,
        execution_trace: str,
    ) -> ValidationScore:
        """
        单次评分（对应 OpenJudge LLMGrader._aevaluate）
        """
        prompt = self._build_prompt(question, candidate_result, knowledge, execution_trace)
        raw_response = self.model.sync_generate_answer([{"role": "user", "content": prompt}])

        parsed = self._parse_json_response(raw_response)

        return ValidationScore(
            score=parsed.get("score", 0.0),
            reason=parsed.get("reason", ""),
            metadata=parsed.get("dimension_scores", {}),
        )

    def _build_prompt(self, ...) -> str:
        """构建评分 prompt"""
        # 参考 OpenJudge 的 prompt 模板结构
        return f"""
=== 待评分内容 ===
问题: {question}
候选答案: {json.dumps(candidate_result, ensure_ascii=False)}
领域知识: {knowledge or '无'}
执行历史: {execution_trace}

=== 评分标准 ===
{self.rubric_template}

=== 输出要求 ===
请输出 JSON: {{"score": 0.0-1.0, "reason": "...", "dimension_scores": {{"correctness": 0.0, "completeness": 0.0}}}}
"""
```

### 3.3 TOTEvaluationStrategy（参考 OpenJudge AverageEvaluationStrategy）

```python
class TOTEvaluationStrategy:
    """
    Tree of Thought 评分策略
    参考 OpenJudge AverageEvaluationStrategy 的多轨迹 + 聚合思路
    """

    def __init__(self, num_trajectories: int = 5, trim_extremes: bool = True):
        self.num_trajectories = num_trajectories
        self.trim_extremes = trim_extremes

    def evaluate(
        self,
        scorer: ValidationScorer,
        **context,
    ) -> ValidationDecision:
        """
        执行 TOT 评分
        1. 生成 N 个评分轨迹
        2. 去极值（可选）
        3. 平均
        4. 决策
        """
        # 并行生成 N 个评分轨迹
        scores = []
        for i in range(self.num_trajectories):
            # 每次轨迹略有不同（参考 OpenJudge 的 temperature 变体）
            score = scorer.score(**context, trajectory_id=i)
            scores.append(score)

        # 聚合
        all_values = [s.score for s in scores]

        if self.trim_extremes and len(all_values) >= 3:
            sorted_vals = sorted(all_values)
            trimmed = sorted_vals[1:-1]
        else:
            trimmed = all_values

        final_score = sum(trimmed) / len(trimmed)

        # 决策
        best_reason = max(scores, key=lambda s: s.score).reason
        worst_suggestion = min(scores, key=lambda s: s.score).metadata.get("suggestion", "")

        return ValidationDecision(
            decision="accept" if final_score >= 0.7 else "reject",
            score=ValidationScore(
                score=final_score,
                reason=best_reason,
                metadata={"dimension_scores": scores[0].metadata},
                all_scores=all_values,
                trimmed_scores=trimmed,
            ),
            suggestion=worst_suggestion,
        )
```

### 3.4 ValidatorAgent

```python
class ValidatorAgent:
    """
    Validator Agent - 基于 OpenJudge + TOT 的答案验证器

    组合使用:
    - ValidationScorer (参考 LLMGrader)
    - TOTEvaluationStrategy (参考 AverageEvaluationStrategy)
    """

    def __init__(
        self,
        model,
        num_trajectories: int = 5,
        validation_threshold: float = 0.7,
    ):
        self.model = model
        self.num_trajectories = num_trajectories
        self.validation_threshold = validation_threshold

        # 初始化评分组件（参考 OpenJudge LLMGrader）
        self.rubric = self._build_default_rubric()
        self.scorer = ValidationScorer(model=model, rubric_template=self.rubric)
        self.strategy = TOTEvaluationStrategy(
            num_trajectories=num_trajectories,
            trim_extremes=True,
        )

    def validate(
        self,
        question: str,
        candidate_result: Dict[str, Any],
        knowledge: str,
        execution_trace: str,
    ) -> ValidationDecision:
        """
        执行验证
        """
        return self.strategy.evaluate(
            scorer=self.scorer,
            question=question,
            candidate_result=candidate_result,
            knowledge=knowledge,
            execution_trace=execution_trace,
        )

    def _build_default_rubric(self) -> str:
        """构建默认评分标准（参考 OpenJudge Rubrics Generation）"""
        return """评估维度（每项 0.0-1.0）：
1. correctness: 答案是否与数据实际内容一致？
2. completeness: 是否回答了问题的所有方面？
3. reasoning_quality: 分析过程是否逻辑严密？

总分 = 三项平均

评分标准：
- >= 0.8: 优秀，答案准确完整
- 0.6-0.8: 良好，有小瑕疵
- 0.4-0.6: 一般，存在明显问题
- < 0.4: 较差，答案有重大错误"""
```

---

## 4. 与 Workflow 的集成

### 4.1 `run()` 方法中的集成

```python
def run(self, task_id, question, context_dir, difficulty):
    # ... Phase 1-3 保持不变 ...

    # Phase 3 循环结束后
    # 已完成: plan.is_complete() 或 plan.all_steps_failed()

    # Phase 3.5: Validator 验证
    if self.config.enable_validation and self.validator:
        candidate_result = self._extract_candidate_result(plan)
        execution_trace = self._build_execution_trace(history, plan)

        if candidate_result:
            validation_passed = False
            for val_round in range(self.config.max_validation_replans):
                # 调用 Validator
                validation_decision = self.validator.validate(
                    question=question,
                    candidate_result=candidate_result,
                    knowledge=session.get("knowledge", ""),
                    execution_trace=execution_trace,
                )

                if validation_decision.decision == "accept":
                    validation_passed = True
                    break

                # 验证失败 → 注入反馈到 Session
                session.set("validation_feedback", {
                    "round": val_round + 1,
                    "score": validation_decision.score.score,
                    "reason": validation_decision.score.reason,
                    "suggestion": validation_decision.suggestion,
                })

                # 触发 Replanner 修正计划
                if self.config.enable_replanning:
                    replanner_input = AgentInput(
                        task_id=task_id,
                        question=question,
                        context={"validation_reject": True, "feedback": validation_decision.suggestion},
                    )
                    replanner_event = self.replanner.run(replanner_input, session, history)

                    if replanner_event.is_terminal():
                        break

                    # 更新 plan，重新执行
                    plan = session.get("plan")
                    if plan:
                        self._execute_pending_steps(plan, workspace, ...)  # 复用执行逻辑
                        candidate_result = self._extract_candidate_result(plan)

    # Phase 4: 合成最终答案
    final_result = self._synthesize_answer(task_id, plan, session, step_results)
    return final_result
```

### 4.2 配置参数

```python
@dataclass
class WorkflowConfig:
    # ... 现有字段 ...
    enable_validation: bool = True
    validation_threshold: float = 0.7       # 通过阈值
    num_validation_trajectories: int = 5    # TOT 分支数
    max_validation_replans: int = 3         # 验证失败最大重试次数
```

---

## 5. 关键设计决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| 验证时机 | Phase 3 循环结束后 | 确保已完整执行所有步骤 |
| 验证失败处理 | 触发 Replanner 修正 + 重新执行 | 复用现有 Replanner 能力 |
| OpenJudge 集成方式 | 组合而非继承 | 避免异步接口适配成本 |
| 多轨迹数 | 5 | 平衡多样性与成本 |
| 聚合策略 | Trimmed Mean (去极值) | 更鲁棒，减少极端评分影响 |
| 评分维度 | correctness, completeness, reasoning | 覆盖答案质量核心要素 |
| 通过阈值 | 0.7 (可配置) | 有一定容错，不过于严格 |

---

## 6. 待确认问题

1. **验证失败时的重试逻辑**：重新执行时，是否只执行 `status == PENDING` 的步骤？还是清除所有结果重新执行？
2. **execution_trace 的粒度**：是否需要包含每个步骤的完整 tool_calls？还是只包含最终结果摘要？
3. **rubric 是否支持自定义**：是否需要从 `knowledge.md` 中提取/生成评分标准？
4. **Validator 是否作为独立 Agent**：`ValidatorAgent` 是否需要有 `run()` 方法（实现 `Agent` 接口）？还是作为 Workflow 内部组件更合适？

---

## 7. 文件变更计划

| 文件 | 操作 | 说明 |
|-----|------|------|
| `agents/validator.py` | **重写** | 移除之前错误实现，改为组合 OpenJudge 思想 |
| `orchestration/workflow.py` | 修改 | Phase 3 循环后插入 Validator 调用 |
| `core/types.py` | 修改 | 已添加 `has_successful_steps()` ✓ |
| `runner.py` | 修改 | 实例化 ValidatorAgent 并传入 Workflow |

**新增文件**：
| 文件 | 说明 |
|-----|------|
| `validation/scorer.py` | `ValidationScorer` 类 |
| `validation/strategy.py` | `TOTEvaluationStrategy` 类 |
| `validation/types.py` | `ValidationScore`, `ValidationDecision` 数据类 |
