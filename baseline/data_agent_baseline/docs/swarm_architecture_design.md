---
name: swarm-architecture-design
description: Swarm架构设计文档，解决baseline核心问题
type: project
---

# Swarm 架构设计文档

## 一、Baseline 问题诊断总结

### 当前架构缺陷
| 问题类别 | 影响范围 | 核心表现 |
|----------|----------|----------|
| **非结构化数据处理** | Hard/Extreme 5+2任务 | 只读文档前4000字符，无法提取叙述性文本中的结构化字段 |
| **列名匹配错误** | Easy 7/9 | Agent自创列名而非匹配Gold模式 |
| **SQL过滤条件歧义** | 跨所有难度 | 大小写不匹配、时间格式歧义、字段混淆 |
| **跨数据源集成** | Medium主导 | 无法正确关联CSV/SQLite/JSON/markdown |
| **步数浪费** | task_350等 | 第一步总是`list_context`，逐个读文件浪费时间 |

### 当前 ReAct Agent 流程
```
Step 1: list_context (固定动作，浪费)
Step 2: 选择文件读取 (_read_csv/_read_json/_read_doc)
Step 3-16: SQL查询/Python执行/错误恢复循环
最终: answer 或 步数耗尽失败
```

**效率问题**: 每次任务独立运行，无上下文共享，无并行能力。

---

## 二、Swarm 架构设计

### 2.1 核心概念

```
Swarm = Lead Agent (事件驱动) + Worker Agent (独立ReAct循环)
```

**职责分离**:
- **Lead Agent**: 规划、分发、验证、合成 — 不干具体活
- **Worker Agent**: 执行具体任务 — 高度自主，完成后进入 idle

### 2.2 三层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         🔷 Lead Agent                           │
│                   规划·事件循环·质量验证                          │
│                     spawn/assign/shutdown                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         🟢 Worker Agents                        │
├───────────────────┬───────────────────┬─────────────────────────┤
│   Worker A        │   Worker B        │     Worker C            │
│  researcher       │   analyst         │     writer              │
│  (ReAct循环)      │   (ReAct循环)     │     (ReAct循环)          │
└─────────┬─────────┴─────────┬─────────┴───────────┬─────────────┘
          │                   │                     │
          └───────────────┬─────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    📦 SHARED INFRASTRUCTURE                     │
├───────────────────┬───────────────────┬─────────────────────────┤
│   📁 Workspace    │  📧 P2PMailbox    │    📋 Task List          │
│    共享数据读写    │   Agent间消息      │   任务队列与状态          │
└───────────────────┴───────────────────┴─────────────────────────┘
```

### 2.3 Lead Agent 生命周期

```python
# Lead Agent 三阶段
Phase 1: INITIAL_PLANNING
    - 接收任务
    - 分析问题类型和难度
    - 制定执行计划（DAG或顺序）
    - spawn Worker Agents

Phase 2: EVENT_LOOP
    - 监听事件: idle | completed | checkpoint | human_input | error
    - 事件驱动决策:
        - idle → reassign 新任务 或 shutdown
        - completed → 验证结果，决定是否需要重试/补充
        - checkpoint → 保存进度，允许中断恢复
        - human_input → 等待人类反馈
        - error → 错误处理，可能 spawn 修复 Worker

Phase 3: SHUTDOWN_SYNTHESIS
    - 收集所有 Worker 结果
    - 合成最终答案
    - 调用 answer 工具提交
```

### 2.4 Worker Agent 状态机

```
┌─────────┐    assigned     ┌─────────┐
│  IDLE   │ ───────────────▶│ RUNNING │
└─────────┘                 └─────────┘
     ▲                           │
     │           completed       │
     │      ┌────────────────────┘
     │      │
     │      ▼
     │  ┌─────────┐
     └──│ DONE    │──▶ (可被 shutdown)
        └─────────┘
```

**Worker 完成任务后进入 idle，Lead 可以**:
1. `reassign` - 分配新任务
2. `shutdown` - 关闭该 Worker

---

## 三、子Agent 执行模式

### 3.1 Plan 模式

#### 顺序执行 (Sequential)
```
Task → [Step1 → Step2 → Step3 → ...] → Result
```
适用场景: 简单任务，依赖关系明确

#### DAG 执行 (Directed Acyclic Graph)
```
        ┌─── Worker A ───┐
Task ──▶│                 │──▶ Synthesis
        └─── Worker B ───┘
```
适用场景: 可并行子任务，无循环依赖

**DAG 关键约束**:
- Worker A 和 B 可同时执行
- 结果在合成节点汇总
- Lead 监控 DAG 节点状态

### 3.2 其他模式

| 模式 | 适用场景 | 执行特点 |
|------|----------|----------|
| **COT** (Chain of Thought) | 逻辑推理型任务 | 单Agent逐步推理 |
| **TOT** (Tree of Thought) | 多方案探索型任务 | 分支探索+剪枝 |
| **Reflection** | 需要质量把关的任务 | 执行→反思→改进循环 |

---

## 四、共享基础设施设计

### 4.1 Workspace (共享数据读写)

```python
class Workspace:
    """
    共享工作空间，解决跨Agent数据传递问题

    解决的问题:
    - 避免重复读取同一文件（baseline浪费步数）
    - 支持中间结果缓存
    - 跨数据源集成（CSV/DB/JSON/markdown）
    """

    def __init__(self, task_context_path: str):
        self.context_path = task_context_path
        self.cache: dict[str, Any] = {}  # 文件缓存
        self.schemas: dict[str, SchemaInfo] = {}  # 跨源数据地图
        self.intermediate_results: dict[str, Any] = {}  # Worker中间结果

    def list_context(self) -> list[FileInfo]:
        """冷启动：一次性扫描所有可用文件"""
        # 替代baseline的逐步list_context

    def read_file(self, path: str, full: bool = False) -> Any:
        """带缓存的文件读取"""
        if path in self.cache:
            return self.cache[path]
        # 实际读取并缓存

    def get_schema_map(self) -> dict[str, SchemaInfo]:
        """建立跨源数据地图"""
        # 解决 task_350 的"假设表存在"问题
```

### 4.2 P2PMailbox (Agent间消息)

```python
class P2PMailbox:
    """
    点对点消息系统，支持Agent间协作

    消息类型:
    - request_data: 请求数据
    - share_result: 共享结果
    - notify_status: 状态通知
    - handoff: 任务交接
    """

    def __init__(self):
        self.mailboxes: dict[str, list[Message]] = {}  # agent_id → messages

    def send(self, from_agent: str, to_agent: str, message: Message):
        """发送消息"""

    def receive(self, agent_id: str) -> list[Message]:
        """接收消息"""

    def broadcast(self, from_agent: str, message: Message):
        """广播消息给所有Worker"""
```

### 4.3 Task List (任务队列与状态)

```python
class TaskList:
    """
    任务队列管理

    状态流转:
    pending → assigned → running → completed → verified
                         ↓
                       failed → retrying
    """

    def __init__(self):
        self.tasks: dict[str, TaskState] = {}

    def assign(self, task_id: str, agent_id: str):
        """分配任务给Worker"""

    def update_status(self, task_id: str, status: TaskStatus):
        """更新任务状态"""

    def get_pending(self) -> list[str]:
        """获取待分配任务"""
```

---

## 五、解决Baseline问题的对应设计

| Baseline问题 | Swarm解决方案 |
|--------------|---------------|
| 第一步总是`list_context` | Workspace冷启动，一次性扫描 |
| 逐个读文件浪费时间 | Workspace缓存，并行读取 |
| 只读文档前4000字符 | ReadWorker支持分块/全文读取 |
| 跨源集成失败 | Schema Map + 专门的IntegrationWorker |
| 列名匹配错误 | Lead Agent质量验证环节 |
| SQL过滤条件歧义 | AnalystWorker增加值探索步骤 |
| 步数耗尽 | Lead Agent监控进度，早停+重新规划 |

---

## 六、Agent 角色分工建议

### 6.1 Worker Agent 角色

| Agent角色 | 负责领域 | 核心能力 |
|-----------|----------|----------|
| **ReaderAgent** | 文件读取 | 并行读取、全文解析、分块处理 |
| **AnalystAgent** | 数据分析 | SQL执行、Python计算、聚合查询 |
| **ResearcherAgent** | 非结构化解析 | Markdown提取、文本结构化、字段抽取 |
| **IntegrationAgent** | 跨源集成 | CSV+DB+JSON关联、Schema匹配 |
| **VerifierAgent** | 结果验证 | 列名匹配、格式检查、答案正确性预检 |

### 6.2 Lead Agent 核心能力

```python
class LeadAgent:
    """主控Agent，事件驱动"""

    def plan(self, task: PublicTask) -> ExecutionPlan:
        """分析任务，制定执行计划"""
        # 决定使用哪种模式：Sequential/DAG/COT/TOT/Reflection

    def spawn(self, agent_type: str) -> str:
        """创建Worker Agent"""

    def assign(self, worker_id: str, subtask: SubTask):
        """分配任务给Worker"""

    def verify(self, result: Any) -> VerificationResult:
        """验证Worker结果质量"""

    def synthesize(self, results: list[Any]) -> AnswerTable:
        """合成最终答案"""

    def shutdown(self, worker_id: str):
        """关闭Worker"""

    def run_event_loop(self):
        """事件循环，处理Worker状态变化"""
```

---

## 七、Handoff 机制设计

### 7.1 简单 Handoff (上下文注入)

```python
# Worker A 完成，将结果注入 Worker B 的上下文
handoff(
    from="analyst_1",
    to="verifier_1",
    context={
        "query_result": result_table,
        "original_question": question,
        "gold_column_names": expected_columns  # 可选
    }
)
```

### 7.2 复杂 Handoff (P2P消息协议)

```python
# Worker A 请求 Worker B 提供数据
mailbox.send(
    from_agent="analyst_1",
    to_agent="researcher_1",
    message=Message(
        type="request_data",
        payload={"need": "patient_gender_from_markdown"}
    )
)

# Worker B 响应
mailbox.send(
    from_agent="researcher_1",
    to_agent="analyst_1",
    message=Message(
        type="share_result",
        payload={"gender_mapping": {"patient_1": "M", "patient_2": "F"}}
    )
)
```

---

## 八、关键技术点（需要确认）

### 🔴 知识点1: LangGraph 多Agent实现
**问题**: LangGraph 中如何实现 Lead Agent 监控多个 Worker Agent 的状态？
- 是否需要 SubGraph？
- 状态共享机制是什么？
- interrupt() 如何在多Agent场景使用？

**建议**: 是否需要调用 `langgraph` skill 获取详细指导？

### 🔴 知识点2: Worker Agent 的生命周期管理
**问题**: spawn/assign/shutdown 在代码层面如何实现？
- Worker是独立线程/进程还是状态节点？
- 资源如何回收？
- 并发限制如何管理？

### 🔴 知识点3: P2P Mailbox 与 LangGraph Store 的关系
**问题**: LangGraph 的 Store 是否可以替代自定义 P2PMailbox？
- Store 的适用场景是什么？
- 是否支持 Agent 间定向通信？

### 🔴 知识点4: DAG执行的具体实现
**问题**: 在 LangGraph 中如何实现 DAG 并行执行？
- 使用 Send() 批量分发？
- 如何等待多个节点完成？
- 如何处理 DAG 中的失败节点？

### 🔴 知识点5: Reflection模式的具体实现
**问题**: Reflection 循环如何避免无限循环？
- 何时终止？
- 如何传递反思结果？

---

## 九、下一步行动建议

1. **确认技术栈**: 是否使用 LangGraph 作为底层框架？
2. **调用相关Skill**: 建议调用 `langgraph-agent-skill` 或 `langgraph-fundamentals` 获取实现细节
3. **原型验证**: 先实现最小Swarm（Lead + 1 Reader + 1 Analyst）
4. **逐步扩展**: 根据问题类型添加更多 Worker 角色