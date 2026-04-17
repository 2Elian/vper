from typing import Any, Dict, List, Optional

from data_agent.agents.base import ChatModelAgent
from data_agent.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Plan,
    PlanStep,
    Session,
    StepStatus,
)
from data_agent.llms import BaseLLMClient

PLANNER_SYSTEM_PROMPT = """You are an expert data analysis planning agent.

Your job is to analyze a question about data and create a structured execution plan.

You must break down the task into clear, actionable steps. Each step should:
1. Have a clear description of what data to analyze or what operation to perform
2. Specify which tools to use (if obvious)
3. Declare dependencies on other steps (if any)
4. Specify what data it produces (topic name) for downstream steps
5. Specify what data it needs (topic name) from upstream steps

Plan types:
- "sequential": Steps must run one after another (default for simple tasks)
- "dag": Steps have dependencies but some can run in parallel
- "hybrid": Mix of sequential and parallel execution

Output format (JSON in code block):
```json
{
  "plan_type": "sequential",
  "steps": [
    {
      "step_id": "step_1",
      "description": "Read and inspect the CSV file structure",
      "hint": "Use read_csv to preview the data",
      "depends_on": [],
      "produces": ["file_schema"],
      "consumes": [],
      "suggested_tools": ["read_csv", "read_json"],
      "max_steps": 4,
      "priority": 10
    },
    {
      "step_id": "step_2",
      "description": "Execute the analysis query",
      "hint": "Use execute_python with pandas for complex queries",
      "depends_on": ["step_1"],
      "produces": ["query_result"],
      "consumes": ["file_schema"],
      "suggested_tools": ["execute_python", "execute_context_sql"],
      "max_steps": 8,
      "priority": 5
    }
  ]
}
```

Rules:
1. Keep steps focused and achievable within 8 LLM calls each.
2. For simple queries (single source, simple aggregation), use "sequential" with 1-2 steps.
3. For complex queries (multiple sources, data transformations), use "dag" with clear dependencies.
4. The last step should produce the final answer.
5. Always include step_id, description, and depends_on for each step.
6. Priority: higher number = higher priority when multiple steps are ready.
"""
# TODO 限制一下，看看在哪个计划中提交answer 并且能自主选择子Agent进行操作

class PlannerAgent(ChatModelAgent):
    """参考 Eino 的 Planner：
        1. 分析用户问题和数据源
        2. 将任务分解为可执行的步骤（PlanStep）
        3. 确定步骤之间的依赖关系（DAG）
        4. 选择执行策略（sequential / dag / hybrid）

    核心流程：
        - 接收用户问题 + Schema 信息 + Knowledge
        - LLM 分析后生成结构化的执行计划
        - 计划写入 Session 供 Executor 和 Replanner 使用
    """
    def __init__(self, model: BaseLLMClient, schema_summary: str = "", knowledge: str = ""):
        super().__init__(model=model)
        self._schema_summary = schema_summary
        self._knowledge = knowledge

    @property
    def name(self) -> str:
        return "Planner"

    @property
    def description(self) -> str:
        return "Analyzes tasks and creates structured execution plans with step dependencies."

    def _build_system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    def run(
        self,
        agent_input: AgentInput,
        session: Session,
        history: History,
    ) -> AgentEvent:
        # 从 session 获取 workspace 信息
        schema_summary = session.get("schema_summary", self._schema_summary)
        knowledge = session.get("knowledge", self._knowledge)

        # 构建规划 prompt
        planning_prompt = self._build_planning_prompt(
            question=agent_input.question,
            schema_summary=schema_summary,
            knowledge=knowledge,
            difficulty=agent_input.context.get("difficulty", ""),
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": planning_prompt},
        ]

        try:
            raw_response = self._call_model(messages)
            plan = self._parse_plan(raw_response, agent_input)
            self.logger.info(f"the plan init: {plan}")
            if plan is None:
                self.logger.error(f"parse plan error, please check its json struct of `raw_response`, in planer.py line-125.")
                plan = self._create_default_plan(agent_input)

            # 写入 Session --> 参考 Eino Session 机制
            session.set("plan", plan)
            session.set("current_plan_json", plan.to_dict())

            return AgentEvent(
                agent_name=self.name,
                output={"plan": plan.to_dict()},
                action=AgentAction.CONTINUE,
            )

        except Exception as exc:
            # # 解析失败时使用默认计划
            plan = self._create_default_plan(agent_input)
            # session.set("plan", plan)
            self.logger.error(f"planer.run error: {exc}")
            return AgentEvent(
                agent_name=self.name,
                output={"plan": plan.to_dict(), "fallback": True},
                action=AgentAction.CONTINUE,
                error=str(exc),
            )

    def _build_planning_prompt(self, question: str, schema_summary: str, knowledge: str, difficulty: str,) -> str:
        """构建规划提示词"""
        prompt_parts = [
            "=== Available Data Sources ===",
            schema_summary or "No schema information available.",
            "",
        ]
        if knowledge:
            prompt_parts.extend([
                "=== Knowledge Guide ===",
                knowledge[:3000],
                "",
            ])
        prompt_parts.extend([
            "=== Question ===",
            question,
            "",
            f"=== Difficulty ===\n{difficulty}",
            "",
            "=== Task ===",
            "Analyze this question and create an execution plan. "
            "Break it down into clear steps with dependencies.",
            "Output a JSON plan in a code block.",
        ])

        return "\n".join(prompt_parts)

    def _parse_plan(self, raw_response: str, agent_input: AgentInput) -> Optional[Plan]:
        """parse output of LLM to Plan"""
        parsed = self._parse_json_response(raw_response)
        if not parsed:
            return None

        plan_type = parsed.get("plan_type", "sequential")
        steps_raw = parsed.get("steps", [])

        if not steps_raw:
            return None

        steps = []
        for i, st in enumerate(steps_raw):
            depends_on = st.get("depends_on", [])
            # 验证 depends_on 引用有效
            valid_deps = []
            for dep in depends_on:
                if isinstance(dep, str):
                    valid_deps.append(dep)

            step = PlanStep(
                step_id=st.get("step_id", f"step_{i + 1}"),
                description=st.get("description", ""),
                hint=st.get("hint", ""),
                depends_on=valid_deps,
                produces=st.get("produces", []),
                consumes=st.get("consumes", []),
                suggested_tools=st.get("suggested_tools", []),
                max_steps=int(st.get("max_steps", 8)),
                priority=int(st.get("priority", 0)),
                status=StepStatus.PENDING,
            )
            steps.append(step)

        # 验证步骤ID唯一
        step_ids = [s.step_id for s in steps]
        if len(step_ids) != len(set(step_ids)):
            return None

        # 验证 depends_on 引用存在的步骤
        step_id_set = set(step_ids)
        for step in steps:
            step.depends_on = [d for d in step.depends_on if d in step_id_set]

        return Plan(
            task_id=agent_input.task_id,
            question=agent_input.question,
            difficulty=agent_input.context.get("difficulty", ""),
            plan_type=plan_type,
            steps=steps,
            context_dir=agent_input.context.get("context_dir"),
        )

    def _create_default_plan(self, agent_input: AgentInput) -> Plan:
        """创建默认计划"""
        steps = [
            PlanStep(
                step_id="step_1",
                description="读取相关数据文件，理解数据结构",
                hint="查看数据文件列名和结构",
                produces=["file_schema"],
                suggested_tools=["read_csv", "read_json", "inspect_sqlite_schema"],
                max_steps=4,
                priority=10,
            ),
            PlanStep(
                step_id="step_2",
                description="执行查询或分析，回答问题",
                hint="使用 Python/pandas 或 SQL 执行查询",
                depends_on=["step_1"],
                consumes=["file_schema"],
                suggested_tools=["execute_python", "execute_context_sql"],
                max_steps=8,
                priority=5,
            ),
        ]

        return Plan(
            task_id=agent_input.task_id,
            question=agent_input.question,
            difficulty=agent_input.context.get("difficulty", ""),
            plan_type="sequential",
            steps=steps,
        )