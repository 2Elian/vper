"""
Planning Tools — 细粒度计划管理工具（参考 DeepEye）

提供 create_plan / update_plan / mark_step_done 三个工具，
让 agent 可以在 ReAct 循环中显式管理自己的执行计划。

与 replan 工具的互补关系：
  - planning tools: agent 直接管理计划（不调用 LLM）
  - replan tool: 调用 LLM 分析 trajectory 后修订计划
"""

from __future__ import annotations

import json
from typing import Any, Callable

from data_agent_baseline.tools.registry import ToolExecutionResult, ToolSpec

# --------------- spec 定义 ---------------

PLANNING_TOOL_SPECS: dict[str, ToolSpec] = {
    "create_plan": ToolSpec(
        name="create_plan",
        description=(
            "Create a new execution plan with a list of steps. "
            "Use this at the beginning to set up your plan, or when you need to "
            "completely replace the current plan. Each step should have: "
            "step (number), description (what to do), action (tool name), details (specific instructions)."
        ),
        input_schema={
            "steps": [
                {
                    "step": 1,
                    "description": "Read knowledge.md to understand data schema",
                    "action": "read_doc",
                    "details": "Target: knowledge.md",
                }
            ]
        },
    ),
    "update_plan": ToolSpec(
        name="update_plan",
        description=(
            "Update the current plan with revised steps. "
            "Use this when you discover new information that changes the approach, "
            "or when a step fails and you need to adjust. "
            "Provide the complete revised plan (not just the changes)."
        ),
        input_schema={
            "steps": [
                {
                    "step": 1,
                    "description": "Revised step description",
                    "action": "tool_name",
                    "details": "Specific instructions",
                }
            ]
        },
    ),
    "mark_step_done": ToolSpec(
        name="mark_step_done",
        description=(
            "Mark a step in the plan as completed. "
            "Call this after successfully finishing a step. "
            "This helps track progress through the plan."
        ),
        input_schema={"step_index": 1},
    ),
}

# --------------- plan 格式化 ---------------

def _format_plan_from_steps(steps: list[dict]) -> str:
    """将步骤列表格式化为 prompt 中展示的计划文本。"""
    lines = ["Current Plan:"]
    for item in steps:
        if not isinstance(item, dict):
            continue
        step_num = item.get("step", item.get("step_id", "?"))
        description = item.get("description", "")
        action = item.get("action", item.get("tool", ""))
        details = item.get("details", item.get("hint", ""))
        lines.append(
            f"Step {step_num}: Description: {description} | Action: {action} | Details: {details}"
        )
    return "\n".join(lines)


# --------------- handler 工厂 ---------------

def create_planning_tool_handlers(
    plan_state: dict,
) -> dict[str, Callable[..., ToolExecutionResult]]:
    """
    创建 planning tools 的 handler 集合。

    Args:
        plan_state: 可变 dict，包含:
            - "current_plan": str — 当前计划的文本表示
            - "completed_steps": set[int] — 已完成的步骤编号

    Returns:
        {"create_plan": handler, "update_plan": handler, "mark_step_done": handler}
    """

    def handle_create_plan(task: Any, action_input: dict) -> ToolExecutionResult:
        steps = action_input.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return ToolExecutionResult(
                ok=False,
                content={"error": "steps must be a non-empty list of step objects"},
            )
        plan_text = _format_plan_from_steps(steps)
        plan_state["current_plan"] = plan_text
        plan_state["completed_steps"] = set()
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "plan_created",
                "step_count": len(steps),
                "plan": plan_text,
            },
        )

    def handle_update_plan(task: Any, action_input: dict) -> ToolExecutionResult:
        steps = action_input.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return ToolExecutionResult(
                ok=False,
                content={"error": "steps must be a non-empty list of step objects"},
            )
        plan_text = _format_plan_from_steps(steps)
        plan_state["current_plan"] = plan_text
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "plan_updated",
                "step_count": len(steps),
                "plan": plan_text,
            },
        )

    def handle_mark_step_done(task: Any, action_input: dict) -> ToolExecutionResult:
        step_index = action_input.get("step_index")
        if step_index is None:
            return ToolExecutionResult(
                ok=False,
                content={"error": "step_index is required"},
            )
        completed = plan_state.setdefault("completed_steps", set())
        completed.add(int(step_index))
        all_steps = plan_state.get("all_step_count", 0)
        progress = f"{len(completed)}/{all_steps}" if all_steps else f"{len(completed)}"
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "step_marked_done",
                "step_index": int(step_index),
                "completed_count": len(completed),
                "progress": progress,
            },
        )

    return {
        "create_plan": handle_create_plan,
        "update_plan": handle_update_plan,
        "mark_step_done": handle_mark_step_done,
    }
