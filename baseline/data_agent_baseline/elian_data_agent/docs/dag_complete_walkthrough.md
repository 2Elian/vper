# DAG 完整链路文档

## 1. 两条入口路径

`runner.py` 中根据 `dag_enabled` 选择入口：

```python
# runner.py run_task() 末尾
if dag_enabled:
    result = workflow.run_with_dag(...)   # 路径 A: DAG 批量执行
else:
    result = workflow.run(...)            # 路径 B: 循环逐步执行
```

**两条路径都使用 DAG 的拓扑排序和依赖检测**，区别在于执行引擎不同。

---

## 2. 路径 B：`run()` — 循环逐步执行

### 2.1 完整流程

```
用户问题 + context_dir
    │
    ▼
Phase 1: Workspace 冷启动
    │  - 扫描 context_dir 下所有文件
    │  - 加载 knowledge.md → workspace.knowledge
    │  - 提取 schema_summary（每个文件的列名/类型摘要）
    │  - 全部写入 Session
    ▼
Phase 2: PlannerAgent.run()
    │  - 构建 prompt: schema_summary + knowledge + question
    │  - 调用 LLM → 返回 JSON 计划
    │  - 解析为 Plan(steps=[PlanStep, PlanStep, ...])
    │  - 写入 session.set("plan", plan)
    ▼
Phase 3: _execute_plan_loop()  ←─ DAG 在这里发挥作用
    │
    │  ┌─────────────────────────────────────────────────┐
    │  │ for iteration in range(1, max_iterations + 1): │
    │  │                                                  │
    │  │   ① plan.is_complete()? → YES → break           │
    │  │      plan.all_steps_failed()? → YES → break      │
    │  │                                                  │
    │  │   ② 获取就绪步骤 (DAG 调度)                      │
    │  │      dag_scheduler.get_next_ready_steps(         │
    │  │          plan, completed_steps                   │
    │  │      )                                           │
    │  │      ↓                                           │
    │  │      内部: 从 plan.steps 重建 DAGGraph           │
    │  │      → graph.get_ready_nodes(completed)          │
    │  │      → 返回所有依赖已满足的 PENDING 步骤          │
    │  │                                                  │
    │  │   ③ 执行就绪步骤                                 │
    │  │      for step in ready_steps:                    │
    │  │        plan.mark_step_running(step.step_id)      │
    │  │        收集依赖结果 dep_results                   │
    │  │        executor.run(step, dep_results, ...)      │
    │  │        → 成功: mark_step_done + completed_steps  │
    │  │        → 失败: mark_step_failed (可能重置PENDING)│
    │  │                                                  │
    │  │   ④ Replanner 审查                               │
    │  │      replanner.run()                             │
    │  │      → "continue": 继续循环                      │
    │  │      → "complete"/"fail": break 退出循环          │
    │  │      → "replan": 修改 plan.steps, 继续           │
    │  │                                                  │
    │  │   ⑤ 回到 ①                                      │
    │  └─────────────────────────────────────────────────┘
    ▼
Phase 3.5: Validator 验证（如果启用）
    │  - 提取候选答案
    │  - TOT 评分
    │  - 通过 → 继续
    │  - 失败 → Replanner 修正 → 回到 Phase 3 重新执行
    ▼
Phase 4: _synthesize_answer()
    │  - 从 session["final_result"] 或 plan.steps 中提取
    │  - 构建 WorkflowResult(answer={columns, rows})
    ▼
返回 WorkflowResult
```

### 2.2 DAG 调度细节（`get_next_ready_steps`）

每次循环迭代都调用此方法。展开内部调用链：

```
dag_scheduler.get_next_ready_steps(plan, completed_steps)
    │
    ▼
DAGGraph.from_steps(plan.steps)
    │  遍历 plan.steps:
    │    graph.add_node(step.step_id, data=step)
    │  遍历 plan.steps:
    │    for dep_id in step.depends_on:
    │      graph.add_edge(dep_id, step.step_id)
    │
    │  举例: step_1 无依赖, step_2 依赖 step_1, step_3 依赖 step_1
    │  结果:
    │    nodes: {step_1, step_2, step_3}
    │    edges: step_1→step_2, step_1→step_3
    │
    ▼
graph.get_ready_nodes(completed, running)
    │  遍历所有节点:
    │    排除: node_id in completed
    │    排除: node_id in running
    │    就绪条件: node.dependencies.issubset(completed)
    │      即: 该节点的所有依赖节点都在 completed 集合中
    │  排序: 按依赖数少的优先（更容易并行）
    │
    ▼
返回 [PlanStep, ...] 就绪步骤列表
```

### 2.3 各阶段状态演变示例

假设 Plan 有 3 个步骤：`step_1 → step_2, step_1 → step_3`

```
iteration=1:
  completed_steps = {}
  ready = [step_1]               ← step_1 无依赖
  执行 step_1 → 成功
  completed_steps = {step_1}
  Replanner: continue

iteration=2:
  completed_steps = {step_1}
  ready = [step_2, step_3]       ← step_2, step_3 都只依赖 step_1（已完成）
  执行 step_2 → 成功
  执行 step_3 → 成功
  completed_steps = {step_1, step_2, step_3}
  Replanner: continue

iteration=3:
  plan.is_complete() == True     ← 所有步骤都是 DONE
  break

→ 进入 Phase 3.5 Validator
```

---

## 3. 路径 A：`run_with_dag()` — DAG 批量执行

### 3.1 完整流程

```
用户问题 + context_dir
    │
    ▼
Phase 1: Workspace 冷启动                    ← 同路径 B
Phase 2: PlannerAgent.run()                  ← 同路径 B
    ▼
Phase 3: DAG 批量执行
    │
    │  ① 选择执行策略
    │     strategy = dag_scheduler.select_execution_strategy(plan)
    │
    │  ② 构建 DAG 并执行
    │     dag_result = dag_scheduler.execute_plan(plan, step_executor_fn, strategy)
    │
    │  ③ Replanner 审查（一次性，不在循环中）
    │
    ▼
Phase 3.5: Validator 验证（如果启用）
    ▼
Phase 4: _synthesize_answer()
    ▼
返回 WorkflowResult
```

### 3.2 执行策略选择（`select_execution_strategy`）

```
dag_scheduler.select_execution_strategy(plan)
    │
    ├─ 任何 step 有 depends_on? 
    │    YES → return "hybrid"
    │
    ├─ plan.plan_type == "sequential"?
    │    YES → return "sequential"
    │
    ├─ plan.plan_type == "dag"?
    │    YES → return "hybrid"
    │
    └─ 默认 → return "parallel"
```

| 策略 | 条件 | 执行引擎 |
|------|------|---------|
| `parallel` | 无依赖 + plan_type 非 sequential | `ThreadPoolExecutor` 全部并行 |
| `sequential` | plan_type=sequential 或无依赖但指定顺序 | 拓扑排序后逐个执行 |
| `hybrid` | 有依赖关系 | 线程池 + 依赖等待 |

### 3.3 `execute_plan` 内部调用链

```
dag_scheduler.execute_plan(plan, step_executor_fn, strategy)
    │
    ▼
    graph = DAGGraph.from_steps(plan.steps)       ← 同路径 B，构建图
    │
    ▼
    graph.validate()                              ← Kahn 算法环检测
    │  有环? → return DAGExecutionResult(error="DAG has cycle: ...")
    │
    ▼  (根据 strategy 分支)

    ┌─ strategy == "sequential" ─────────────────────────────────────┐
    │                                                                  │
    │  executor.execute_sequential(graph, step_executor_fn)           │
    │    sorted_order = graph.topological_sort()  ← Kahn 算法排序     │
    │    for node_id in sorted_order:                                 │
    │      收集依赖结果 dep_results                                    │
    │      step_result = step_executor_fn(step, dep_results)          │
    │      失败? → retry (最多 max_retries 次)                        │
    │                                                                  │
    ├─ strategy == "parallel" ────────────────────────────────────────┤
    │                                                                  │
    │  executor._execute_parallel(graph, step_executor_fn)            │
    │    ThreadPoolExecutor(max_workers=5):                            │
    │      所有节点同时提交: executor.submit(step_executor, step, {})  │
    │      as_completed 收集结果                                       │
    │      无依赖结果传递（dep_results = {}）                           │
    │                                                                  │
    ├─ strategy == "hybrid" ──────────────────────────────────────────┤
    │                                                                  │
    │  executor._execute_hybrid(graph, step_executor_fn)              │
    │    ThreadPoolExecutor(max_workers=5):                            │
    │      所有节点同时提交 execute_node()                              │
    │      ↓ 每个线程内部:                                             │
    │        _wait_for_dependencies(node.dependencies, completed, failed)│
    │          ↓ 轮询检查:                                             │
    │            所有依赖 in completed? → 就绪, 返回 True             │
    │            任何依赖 in failed?   → 跳过, 返回 False             │
    │            超时(600s)?           → 返回 False                    │
    │            否则 sleep(5s) 继续检查                               │
    │        依赖就绪后:                                               │
    │          收集依赖结果 dep_results                                │
    │          step_result = step_executor_fn(step, dep_results)       │
    │          失败? → retry                                           │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    │
    ▼
    _update_plan_status(plan, result)
    │  遍历 result.results:
    │    success → step.status = DONE, step.result = data
    │    failed  → step.status = FAILED, step.result = {error: ...}
    │
    ▼
    return DAGExecutionResult(
        total_steps=3,
        completed_steps=2,
        failed_steps=1,
        results={step_1: StepResult, step_2: StepResult, step_3: StepResult},
        success=False,     ← 有失败的步骤
    )
```

### 3.4 Hybrid 模式线程执行时序示例

假设 Plan: `step_1 → step_2, step_1 → step_3, step_2+step_3 → step_4`

```
时间 →

线程1: [step_1 执行...][完成 ✓]
线程2: [等待 step_1....][step_2 执行...][完成 ✓]
线程3: [等待 step_1....][step_3 执行...][完成 ✓]
线程4: [等待 step_2,step_3..............][step_4 执行...][完成 ✓]

completed:  {}          {step_1}           {step_1,step_2}    {step_1,step_2,step_3}   {全部}
```

如果 step_2 失败：

```
线程1: [step_1 执行...][完成 ✓]
线程2: [等待 step_1....][step_2 执行...][失败 ✗]
线程3: [等待 step_1....][step_3 执行...][完成 ✓]
线程4: [等待............][检测到 step_2 在 failed 中 → 跳过, 返回 "Dependency failed"]

结果: step_1 ✓, step_2 ✗, step_3 ✓, step_4 跳过
```

---

## 4. 两条路径对比

| 维度 | `run()` 路径 B | `run_with_dag()` 路径 A |
|------|---------------|----------------------|
| 执行引擎 | 手动 for 循环 | DAGExecutor (ThreadPoolExecutor) |
| 并行能力 | 同一轮 ready_steps 顺序执行 | Hybrid 线程并行执行 |
| DAG 构建 | 每次迭代重建 DAGGraph | 一次构建，整图执行 |
| Replanner | 每轮迭代都调用 | 执行完成后调用一次 |
| 失败重试 | mark_step_failed → 可能重置 PENDING | executor 内部 retry |
| 依赖传递 | 手动收集 dep_results | 自动从 results 字典获取 |
| 适用场景 | 需要 Replanner 频繁干预 | 步骤独立、可并行 |

---

## 5. 所有可能的状态和退出条件

### 5.1 PlanStep 状态机

```
PENDING ──→ RUNNING ──→ DONE
  │              │
  │              └──→ FAILED (retry_count >= max_retries)
  │                        │
  └──→ mark_step_failed    │
       (retry_count < max_retries)
       → 重置为 PENDING ──→ (下一轮重新执行)
```

### 5.2 Phase 3 循环退出条件

| 条件 | 位置 | 说明 |
|------|------|------|
| `plan.is_complete()` | 循环开头 | 所有步骤 DONE 或 SKIPPED |
| `plan.all_steps_failed()` | 循环开头 | 所有步骤 FAILED |
| `not ready_steps` + `plan.has_failures()` | 获取就绪步骤后 | 死锁：有失败且无就绪步骤 |
| `replanner_event.is_terminal()` | Replanner 后 | Replanner 决定 "complete" 或 "fail" |
| `iteration > max_iterations` | 循环上限 | 达到最大迭代次数 |

### 5.3 DAG 执行退出条件（Hybrid 模式）

| 条件 | 说明 |
|------|------|
| 所有节点执行完成 | 正常退出 |
| 依赖失败 | 下游节点标记 "Dependency failed" 并跳过 |
| 依赖超时 (600s) | `_wait_for_dependencies` 返回 False |
| 线程异常 | 捕获异常，标记 FAILED |
| 环检测 | `graph.validate()` 发现环，直接返回错误 |

---

## 6. 完整数据流

```
输入
  task_id: "task_19"
  question: "哪个城市的销售额最高？"
  context_dir: /data/task_19/context/
  difficulty: "medium"

        │
        ▼  Phase 1

Session {
  schema_summary: "sales.csv: [city, amount, date]\nproducts.db: table products [...]",
  knowledge: "销售额单位为万元，日期格式为 YYYY-MM-DD...",
  context_dir: /data/task_19/context/
}

        │
        ▼  Phase 2

Plan {
  question: "哪个城市的销售额最高？",
  plan_type: "sequential",
  steps: [
    PlanStep(step_1, "读取数据", depends_on=[]),
    PlanStep(step_2, "分析计算", depends_on=["step_1"]),
  ]
}

        │
        ▼  Phase 3 (DAG 调度)

iteration 1:
  ready_steps = [step_1]
  Executor.run(step_1) →
    ReAct: list_context → read_csv → execute_python → return_result
  step_1.result = {"columns":["city","total"],"rows":[["北京",1500],["上海",1200]]}
  step_1.status = DONE
  completed_steps = {step_1}

iteration 2:
  ready_steps = [step_2]           ← step_2.depends_on=[step_1] ⊆ completed_steps
  Executor.run(step_2) →
    ReAct: 接收 dep_results → return_result
  step_2.result = {"columns":["city"],"rows":[["北京"]]}
  step_2.status = DONE
  completed_steps = {step_1, step_2}

iteration 3:
  plan.is_complete() == True → break

        │
        ▼  Phase 3.5 (Validator)

validate(
  question="哪个城市的销售额最高？",
  candidate_result={"columns":["city"],"rows":[["北京"]]},
  knowledge="销售额单位为万元...",
  execution_trace="=== Plan Steps & Results ===\n[Step: step_1] ..."
)
→ TOT 5 条轨迹评分 → 去极值平均 → 0.85 ≥ 0.7 → accept

        │
        ▼  Phase 4

_synthesize_answer():
  从 plan.steps 找最后一个 DONE 且有 result 的步骤
  → step_2.result = {"columns":["city"],"rows":[["北京"]]}

        │
        ▼
输出
WorkflowResult {
  task_id: "task_19",
  answer: {"columns": ["city"], "rows": [["北京"]]},
  success: True,
  plan: { ... },
  events: [ ... ],
  execution_time: 12.5
}
```
