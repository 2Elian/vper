# OpenJudge 架构分析与 Validator Agent 设计方案

## 1. OpenJudge 框架核心概念

### 1.1 整体架构

OpenJudge 是一个 LLM-as-Judge 评估框架，核心设计遵循**策略模式（Strategy Pattern）**，将"评估逻辑"与"评估执行方式"分离：

```
┌─────────────────────────────────────────────────────────┐
│                      Grader (评分器)                     │
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ BaseGrader      │──│ LLMGrader                   │  │
│  │ - _aevaluate()  │  │ - template (PromptTemplate) │  │
│  │ - strategy      │  │ - model (BaseChatModel)     │  │
│  │ - mode          │  │ - structured_model          │  │
│  │   (POINTWISE/   │  │ - language                  │  │
│  │    LISTWISE)    │  │                             │  │
│  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │ strategy
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Evaluation Strategy (评估策略)              │
│  ┌─────────────┐ ┌─────────────┐ ┌────────────────────┐ │
│  │ Direct      │ │ Average     │ │ Voting             │ │
│  │ (单次评估)   │ │ (多次平均)   │ │ (多数投票)          │ │
│  └─────────────┘ └─────────────┘ └────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐   │
│  │ GRPO Tournament (成对比较 tournament)              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 评分模型 (GraderScore)

```python
class GraderScore(BaseModel):
    name: str           # 评分器名称
    score: float        # 数值分数 (0.0-1.0 或 1-5)
    reason: str         # 评分理由
    metadata: Dict      # 附加元数据
```

**设计要点**：
- 强制包含 `score` 和 `reason`，确保可解释性
- `metadata` 用于存储置信度、模型信息等额外数据
- 支持 Pydantic 验证，便于结构化输出解析

### 1.3 LLMGrader 核心机制

LLMGrader 是主要的 LLM-as-Judge 实现：

```python
class LLMGrader(BaseGrader):
    def __init__(self, model, template, structured_model, ...):
        # template: PromptTemplate - 可配置的多语言模板
        # structured_model: Pydantic BaseModel - 结构化输出模式
        # model: BaseChatModel - LLM 接口

    async def _aevaluate(self, **kwargs) -> GraderScore | GraderRank:
        messages = self.template.format(language=self.language, **kwargs)
        chat_response = await self.model.achat(
            messages=messages,
            structured_model=self.structured_model,  # 强制结构化输出
        )
        parsed = chat_response.parsed
        return GraderScore(
            name=self.name,
            score=parsed["score"],
            reason=parsed["reason"],
            metadata=parsed  # 保留所有解析字段
        )
```

**关键设计模式**：
1. **结构化输出**：使用 Pydantic 模型强制 LLM 返回 `{"score": N, "reason": "..."}` 格式
2. **多语言模板**：支持中英文等不同语言的评估提示词
3. **Template 渲染**：变量通过 `**kwargs` 动态注入

### 1.4 评估策略详解

#### DirectEvaluationStrategy
- **用途**：单次评估，无聚合
- **适用**：确定性强的评估场景
- **实现**：直接转发调用

#### AverageEvaluationStrategy（核心参考）
- **用途**：执行 N 次评估，取平均分
- **关键代码**：
```python
async def execute(self, call_fn, **kwargs):
    coroutines = [call_fn(**kwargs) for _ in range(self.num_evaluations)]
    results = await asyncio.gather(*coroutines)  # 并行执行
    avg_score = sum(r.score for r in results) / len(results)
    return GraderScore(
        score=avg_score,
        reason=f"Averaged from {len(results)} evaluations"
    )
```
- **适用**：减少单次评估的随机性噪声

#### VotingEvaluationStrategy
- **用途**：多次评估，取众数
- **tie_breaker**：支持 MIN/MAX/CLOSEST_TO_MEAN 解决平局
- **适用**：离散评分（如 1-5 分制）

#### GRPOTournamentEvaluationStrategy
- **用途**：成对比较多个回答，计算净胜率
- **算法**：N*(N-1)/2 次 pairwise 比较
- **reward**：r_i = (wins_i - losses_i) / (N - 1) ∈ [-1.0, 1.0]
- **debiased**：双向比较减少位置偏见

## 2. 与 Tree of Thought (TOT) 的映射关系

### 2.1 TOT 核心思想

Tree of Thought 论文提出的核心模式：
1. **Generate**：生成多个思考轨迹（推理路径）
2. **Evaluate**：对每个轨迹进行评分
3. **Aggregate**：聚合分数（去除极端值、平均、投票等）
4. **Decide**：基于聚合结果做决策

### 2.2 OpenJudge → TOT 的映射

| TOT 阶段 | OpenJudge 组件 | 在 Validator 中的应用 |
|---------|---------------|---------------------|
| Generate | 多次调用 LLMGrader | 生成 N 个独立的评分轨迹 |
| Evaluate | LLMGrader._aevaluate() | 每个轨迹返回 score + reason |
| Aggregate | Average/Voting Strategy | 去极值后平均 |
| Decide | 应用层逻辑 | score > threshold ? 通过 : 重新规划 |

**关键洞察**：
- OpenJudge 的 `AverageEvaluationStrategy` 已实现 TOT 的"多轨迹生成+平均聚合"模式
- 我们的 TOT 变体需要**扩展**：在平均前去除最高/最低分（trimmed mean）

## 3. Validator Agent 设计方案

### 3.1 定位与职责

在 Plan-Execute-Replan 工作流中，Validator 位于 Phase 3 (Executor-Replanner 循环) 和 Phase 4 (答案合成) 之间：

```
Phase 1: Workspace 冷启动
    ↓
Phase 2: Planner 生成计划
    ↓
Phase 3: Executor ↔ Replanner 循环 (直到计划完成)
    ↓
╔════════════════════════════════════════╗
║ Phase 3.5: Validator Agent (新增)       ║
║ - 深度分析执行结果                      ║
║ - TOT 多轨迹评分                        ║
║ - 决策：通过 / 触发重新规划              ║
╚════════════════════════════════════════╝
    ↓ (高分) → Phase 4: 合成最终答案
    ↓ (低分) → 触发 Replanner 修正计划
```

### 3.2 输入数据源

Validator 需要访问：

1. **knowledge.md** (来自 Session)
   - `session.get("knowledge")`
   - 领域知识、评估标准

2. **完整 ReAct 交互历史** (来自 History)
   - `history.get_messages()` - LLM 对话历史
   - `history.get_events()` - Agent 事件序列
   - 每个步骤的 thought → action → observation 链条

3. **计划执行结果** (来自 Plan)
   - `plan.steps` - 所有步骤及状态
   - 每个步骤的 `step.result` - 执行输出
   - `step.tool_calls` - 工具调用记录

4. **原始问题**
   - `question` - 用户原始查询

### 3.3 TOT 评分实现

参考 OpenJudge 的 `AverageEvaluationStrategy`，实现 **TrimmedMeanEvaluationStrategy**：

```python
class TrimmedMeanEvaluationStrategy:
    """
    TOT 评分策略：
    1. 生成 N 个独立的评分轨迹（并行）
    2. 收集所有 (score, reason) 结果
    3. 去除最高分和最低分
    4. 剩余分数取平均
    5. 合并 reasons 提供可解释性
    """

    def __init__(self, num_trajectories: int = 5):
        self.num_trajectories = num_trajectories  # TOT 分支数

    def execute(self, evaluate_fn, **kwargs) -> ValidationResult:
        # 1. 生成 N 个评分轨迹（并行调用）
        results = [evaluate_fn(**kwargs) for _ in range(self.num_trajectories)]

        # 2. 提取分数
        scores = [r.score for r in results]

        # 3. 去极值（Trimmed Mean）
        if len(scores) > 2:
            scores_sorted = sorted(scores)
            trimmed_scores = scores_sorted[1:-1]  # 去头去尾
        else:
            trimmed_scores = scores

        final_score = sum(trimmed_scores) / len(trimmed_scores)

        # 4. 聚合 reasons
        all_reasons = [r.reason for r in results]
        aggregated_reason = self._merge_reasons(all_reasons, scores)

        return ValidationResult(
            score=final_score,
            reason=aggregated_reason,
            metadata={
                "all_scores": scores,
                "trimmed_scores": trimmed_scores,
                "num_trajectories": self.num_trajectories,
            }
        )
```

### 3.4 ValidatorAgent 类设计

继承 `ChatModelAgent`，遵循现有 Agent 模式：

```python
class ValidatorAgent(ChatModelAgent):
    """
    Validator Agent - 基于 TOT + LLM-as-Judge 的结果验证器

    核心流程：
    1. 收集完整上下文（knowledge + history + plan results）
    2. 使用 TOT 生成多个评分轨迹
    3. 去极值后计算最终分数
    4. 决策：高分通过 / 低分触发 replan
    """

    def __init__(self, model: BaseLLMClient, num_trajectories: int = 5):
        super().__init__(model=model)
        self.num_trajectories = num_trajectories
        self.validation_threshold = 0.7  # 可配置

    @property
    def name(self) -> str:
        return "Validator"

    def _build_system_prompt(self) -> str:
        return VALIDATOR_SYSTEM_PROMPT  # 见下文

    def run(self, agent_input: AgentInput, session: Session, history: History) -> AgentEvent:
        # 1. 收集上下文
        knowledge = session.get("knowledge", "")
        plan = session.get("plan")
        question = agent_input.question

        # 2. 构建完整历史追踪
        execution_trace = self._build_execution_trace(history, plan)

        # 3. TOT 评分
        validation_result = self._tot_validate(
            question=question,
            knowledge=knowledge,
            execution_trace=execution_trace,
            final_result=agent_input.context.get("candidate_result"),
        )

        # 4. 决策
        if validation_result.score >= self.validation_threshold:
            return AgentEvent(
                agent_name=self.name,
                action=AgentAction.EXIT,  # 通过，进入 Phase 4
                output={
                    "validation_score": validation_result.score,
                    "validation_reason": validation_result.reason,
                    "decision": "accept",
                }
            )
        else:
            return AgentEvent(
                agent_name=self.name,
                action=AgentAction.CONTINUE,  # 触发 Replanner 修正
                output={
                    "validation_score": validation_result.score,
                    "validation_reason": validation_result.reason,
                    "decision": "reject",
                    "suggestion": validation_result.suggestion,
                }
            )

    def _tot_validate(self, **context) -> ValidationResult:
        """Tree of Thought 评分核心逻辑"""
        trajectories = []

        # 并行生成 N 个评分轨迹
        for i in range(self.num_trajectories):
            # 每次使用稍微不同的温度或提示词变体增加多样性
            messages = self._build_validation_messages(
                trajectory_id=i,
                **context
            )
            raw_response = self._call_model(messages)
            parsed = self._parse_json_response(raw_response)

            if parsed:
                trajectories.append(ValidationTrajectory(
                    score=parsed.get("score", 0.0),
                    reason=parsed.get("reason", ""),
                    suggestion=parsed.get("suggestion", ""),
                ))

        # 去极值平均
        return self._aggregate_trajectories(trajectories)
```

### 3.5 提示词设计

参考 OpenJudge 的 LLMGrader 模板结构：

```python
VALIDATOR_SYSTEM_PROMPT = """You are a rigorous data analysis validator.

Your task is to evaluate whether the execution result correctly answers the original question.

You will be given:
1. The original question
2. Domain knowledge (from knowledge.md)
3. Complete execution history (ReAct traces for all steps)
4. The candidate answer

Evaluate on these dimensions (each 0.0-1.0):
- correctness: Does the answer match the data?
- completeness: Does it address all aspects of the question?
- reasoning_quality: Is the analysis process sound?

Output format (JSON):
```json
{
  "score": 0.85,
  "reason": "Detailed explanation of the score...",
  "dimension_scores": {
    "correctness": 0.9,
    "completeness": 0.8,
    "reasoning_quality": 0.85
  },
  "suggestion": "If score is low, suggest how to fix..."
}
```

Rules:
1. Be critical - a score above 0.8 requires strong evidence
2. Reference specific data points from the execution trace
3. If tools failed or data is missing, score must be below 0.5
"""
```

### 3.6 与 Workflow 集成

在 `workflow.py` 的 Phase 3 和 Phase 4 之间插入 Validator：

```python
class PlanExecuteReplanWorkflow:
    def __init__(self, planner, executor, replanner, validator, config):
        self.planner = planner
        self.executor = executor
        self.replanner = replanner
        self.validator = validator  # 新增
        self.config = config

    def run(self, task_id, question, context_dir, difficulty):
        # ... Phase 1-3 保持不变 ...

        # Phase 3.5: Validator 验证
        if self.validator and plan.has_successful_steps():
            candidate_result = self._extract_candidate_result(plan)

            validator_input = AgentInput(
                task_id=task_id,
                question=question,
                context={"candidate_result": candidate_result},
            )
            validator_event = self.validator.run(validator_input, session, history)
            events.append(validator_event.to_dict())
            history.add_event(validator_event)

            output = validator_event.output or {}

            # 低分触发 Replanner 修正
            if output.get("decision") == "reject":
                # 将验证结果注入 Session，Replanner 可以看到
                session.set("validation_feedback", output)

                # 触发 Replanner 进行修正
                replanner_input = AgentInput(
                    task_id=task_id,
                    question=question,
                    context={"validation_reject": True, "feedback": output},
                )
                replanner_event = self.replanner.run(replanner_input, session, history)
                events.append(replanner_event.to_dict())

                # 如果 Replanner 决定继续，循环会重新执行
                if not replanner_event.is_terminal():
                    continue  # 回到 Phase 3 循环

            # 高分通过，进入 Phase 4
            elif output.get("decision") == "accept":
                session.set("validation_passed", True)

        # Phase 4: 合成最终答案
        final_result = self._synthesize_answer(task_id, plan, session, step_results)
        return final_result
```

### 3.7 关键设计决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| TOT 分支数 | 5 | 平衡多样性与成本；可配置 |
| 聚合方式 | Trimmed Mean | 去除极端值，比纯平均更鲁棒 |
| 评分维度 | 3 维度 + 总分 | 提供细粒度反馈给 Replanner |
| 通过阈值 | 0.7 (可配置) | 允许一定容错，避免过于严格 |
| 低分处理 | 触发 Replanner | 复用现有 replan 机制 |
| 历史使用 | 完整 ReAct trace | 让 Validator 能看到推理过程 |

## 4. 实施计划

1. **创建 `agents/validator.py`**
   - 实现 `ValidatorAgent` 类
   - 实现 `TrimmedMeanEvaluationStrategy`（内嵌或独立）

2. **修改 `orchestration/workflow.py`**
   - `WorkflowConfig` 添加 `enable_validation` 和 `validation_threshold`
   - `PlanExecuteReplanWorkflow.__init__` 接受 `validator` 参数
   - 在 Phase 3 和 Phase 4 之间插入 Validator 调用

3. **修改 `runner.py`**
   - 实例化 `ValidatorAgent`
   - 传入 `PlanExecuteReplanWorkflow`

4. **添加配置选项**
   - `num_validation_trajectories`: TOT 分支数 (默认 5)
   - `validation_threshold`: 通过阈值 (默认 0.7)
   - `trim_extremes`: 是否去极值 (默认 True)

## 5. 与 OpenJudge 的对比总结

| 特性 | OpenJudge | Validator Agent (本项目) |
|-----|-----------|-------------------------|
| 架构 | 策略模式分离 Grader/Strategy | 单 Agent 内嵌 TOT 逻辑 |
| 执行 | 异步 asyncio | 同步（匹配现有代码） |
| 评分模型 | GraderScore | 内嵌 dict，适配现有类型 |
| 多轨迹 | Average/Voting Strategy | Trimmed Mean（去极值） |
| 上下文 | 灵活 kwargs | 固定结构（knowledge+history+plan） |
| 输出 | 纯评分 | 评分 + 修正建议 |
| 集成 | 独立框架 | 嵌入 Plan-Execute-Replan 工作流 |

**核心借鉴点**：
1. LLM 结构化输出模式（score + reason）
2. 多轨迹并行评估思想
3. 策略模式的可配置性
4. Prompt Template 的多语言支持（可选）
