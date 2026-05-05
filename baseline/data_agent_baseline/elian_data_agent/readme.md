  -   基于 Eino ADK 的 Plan-Execute-Replan 模式 + DAG 工作流，用 Python 实现的多 Agent 数据处理系统。

      elian_data_agent/
      ├── core/                    # 核心层
      │   ├── types.py             # 核心类型：Plan, PlanStep, Session, History, AgentEvent
      │   └── workspace.py         # 共享工作空间（冷启动、缓存、Schema地图）
      ├── agents/                  # Agent层
      │   ├── base.py              # Agent接口 + ChatModelAgent基类（ReAct）
      │   ├── model_adapter.py     # LLM模型适配器（兼容openai 0.x/1.x）
      │   ├── planner.py           # Planner Agent：分析任务，创建执行计划
      │   ├── executor.py          # Executor Agent：ReAct循环执行步骤
      │   └── replanner.py         # Replanner Agent：审查结果，调整计划
      ├── dag/                     # DAG层（参考Shannon实现）
      │   ├── graph.py             # DAG数据结构（Kahn算法拓扑排序+环检测）
      │   ├── executor.py          # DAG执行引擎（Parallel/Sequential/Hybrid）
      │   └── scheduler.py         # DAG调度器（策略选择+Plan转DAG）
      ├── orchestration/           # 编排层（参考Eino Workflow Agents）
      │   └── workflow.py          # PlanExecuteReplanWorkflow主工作流
      ├── tools/                   # 工具层
      │   └── registry.py          # 工具注册表（8个工具，适配baseline）
      └── runner.py                # 运行入口（兼容baseline配置）

      核心设计模式

      ┌─────────────────────┬─────────────────────┬────────────────────────────────┐
      │        模式         │        来源         │              实现              │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ Plan-Execute-Replan │ Eino ADK            │ Planner→Executor↔Replanner循环 │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ ReAct循环           │ Eino ChatModelAgent │ Think-Act-Observe在Executor中  │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ Session状态共享     │ Eino Session        │ 跨Agent的KV存储                │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ History数据传递     │ Eino History        │ Agent事件→LLM消息转换          │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ DAG拓扑排序         │ Shannon (Kahn算法)  │ 环检测+执行层级                │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ 三种执行模式        │ DAG文章             │ Parallel/Sequential/Hybrid     │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ 依赖等待            │ Shannon             │ 增量检查+超时机制              │
      ├─────────────────────┼─────────────────────┼────────────────────────────────┤
      │ 工具桥接            │ baseline tools      │ 复用8个现有工具                │
      └─────────────────────┴─────────────────────┴────────────────────────────────┘

      DAG增强

      系统在Eino的Plan-Execute-Replan基础上增加了DAG工作流：
      - Planner 生成的计划步骤支持 depends_on 依赖声明
      - DAGScheduler 自动选择执行策略（有依赖→Hybrid，无依赖→Parallel）
      - DAGExecutor 使用 ThreadPoolExecutor 并行执行独立步骤
      - 依赖等待使用增量检查（5秒间隔），而非死等
      - 结果通过 produces/consumes 主题在步骤间传递
