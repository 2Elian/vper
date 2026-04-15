"""
Replanner Agent - 重规划器

参考 Eino ADK 的 Replanner，职责：
1. 审查 Executor 的执行结果
2. 决定是否继续执行、调整计划或完成
3. 更新 Session 中的 Plan

核心决策逻辑：
- 如果步骤成功，继续执行下一步
- 如果步骤失败，决定重试还是调整
- 如果计划完成，生成最终结果
"""


import json
from typing import Any, Dict, List, Optional
from data_agent.llms import BaseLLMClient
from data_agent.agents.base import ChatModelAgent
from data_agent.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Plan,
    PlanStep,
    Session,
    StepResult,
    StepStatus,
)


REPLANNER_SYSTEM_PROMPT = """You are a replanning agent that reviews execution results and decides what to do next.

Given the current plan and execution results, you must decide:
1. "continue" - Continue executing the next steps in the plan
2. "replan" - Adjust the plan (add/remove/modify steps)
3. "complete" - The task is complete, generate final result
4. "fail" - The task cannot be completed

Output format (JSON in code block):
```json
{
  "decision": "continue",
  "reason": "Step 1 completed successfully, step 2 is ready to execute",
  "plan_updates": {
    "add_steps": [],
    "remove_steps": [],
    "modify_steps": []
  },
  "final_result": null
}
```

For "complete" decision:
```json
{
  "decision": "complete",
  "reason": "All steps completed successfully",
  "plan_updates": null,
  "final_result": {
    "columns": ["col1", "col2"],
    "rows": [["val1", "val2"]]
  }
}
```

Rules:
1. If a step failed but can be retried with different approach, use "replan".
2. If all steps are done and results are valid, use "complete".
3. If remaining steps don't need execution, use "complete" with available results.
4. If the task is fundamentally impossible, use "fail".
"""


class ReplannerAgent(ChatModelAgent):
    """
    Replanner Agent: 审查结果，调整计划

    参考 Eino 的 Replanner：
    - 审查执行结果
    - 决定下一步动作
    - 更新执行计划
    """

    def __init__(self, model: BaseLLMClient):
        super().__init__(model=model)

    @property
    def name(self) -> str:
        return "Replanner"

    @property
    def description(self) -> str:
        return "Reviews execution results and decides to continue, replan, or complete."

    def _build_system_prompt(self) -> str:
        return REPLANNER_SYSTEM_PROMPT

    def run(
        self,
        agent_input: AgentInput,
        session: Session,
        history: History,
    ) -> AgentEvent:
        """审查执行结果，决定下一步"""

        # 从 Session 获取当前计划
        plan = session.get("plan")
        if not plan:
            return AgentEvent(
                agent_name=self.name,
                action=AgentAction.ERROR,
                error="No plan found in session",
            )

        # 构建审查 prompt
        review_prompt = self._build_review_prompt(plan, history)

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": review_prompt},
        ]

        try:
            raw_response = self._call_model(messages)
            decision = self._parse_decision(raw_response)

            if decision is None:
                # 默认：如果计划没完成则继续
                if plan.is_complete():
                    decision = {"decision": "complete", "reason": "Plan completed"}
                else:
                    decision = {"decision": "continue", "reason": "Continue execution"}

            action_type = decision.get("decision", "continue")

            if action_type == "complete":
                final_result = decision.get("final_result")
                return self._handle_complete(plan, final_result, session)

            elif action_type == "replan":
                return self._handle_replan(plan, decision, session)

            elif action_type == "fail":
                return AgentEvent(
                    agent_name=self.name,
                    action=AgentAction.EXIT,
                    error=decision.get("reason", "Task failed"),
                )

            else:  # continue
                return AgentEvent(
                    agent_name=self.name,
                    output={"decision": "continue", "reason": decision.get("reason", "")},
                    action=AgentAction.CONTINUE,
                )

        except Exception as exc:
            return AgentEvent(
                agent_name=self.name,
                action=AgentAction.CONTINUE,
                error=f"Replanner error: {exc}",
            )

    def _build_review_prompt(self, plan: Plan, history: History) -> str:
        """构建审查提示词"""
        lines = [
            "=== Current Plan ===",
            f"Task: {plan.question}",
            f"Plan type: {plan.plan_type}",
            "",
            "=== Steps Status ===",
        ]

        for step in plan.steps:
            status = step.status.value
            result_str = ""
            if step.result:
                result_str = f"\n  Result: {json.dumps(step.result, ensure_ascii=False)[:200]}"
            lines.append(f"- [{status}] {step.step_id}: {step.description}{result_str}")

        lines.extend([
            "",
            "=== Recent History ===",
        ])

        # 添加最近的历史
        recent_messages = history.get_messages()[-10:]
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:300]
            lines.append(f"[{role}] {content}")

        lines.extend([
            "",
            "=== Decision Required ===",
            "Based on the plan status and results, decide what to do next.",
        ])

        return "\n".join(lines)

    def _parse_decision(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """解析决策"""
        return self._parse_json_response(raw_response)

    def _handle_complete(
        self,
        plan: Plan,
        final_result: Optional[Dict],
        session: Session,
    ) -> AgentEvent:
        """处理完成决策"""

        # 尝试从最后一个成功步骤获取结果
        if final_result is None:
            for step in reversed(plan.steps):
                if step.status == StepStatus.DONE and step.result:
                    final_result = step.result
                    break

        session.set("final_result", final_result)

        return AgentEvent(
            agent_name=self.name,
            output={"decision": "complete", "final_result": final_result},
            action=AgentAction.EXIT,
        )

    def _handle_replan(
        self,
        plan: Plan,
        decision: Dict,
        session: Session,
    ) -> AgentEvent:
        """处理重规划"""
        plan_updates = decision.get("plan_updates", {})

        # 添加新步骤
        for st in plan_updates.get("add_steps", []):
            new_step = PlanStep(
                step_id=st.get("step_id", f"step_{len(plan.steps) + 1}"),
                description=st.get("description", ""),
                hint=st.get("hint", ""),
                depends_on=st.get("depends_on", []),
                suggested_tools=st.get("suggested_tools", []),
                max_steps=int(st.get("max_steps", 8)),
            )
            plan.steps.append(new_step)

        # 移除步骤
        remove_ids = set(plan_updates.get("remove_steps", []))
        plan.steps = [s for s in plan.steps if s.step_id not in remove_ids]

        # 更新 Session
        session.set("plan", plan)

        return AgentEvent(
            agent_name=self.name,
            output={"decision": "replan", "reason": decision.get("reason", ""), "plan": plan.to_dict()},
            action=AgentAction.CONTINUE,
        )