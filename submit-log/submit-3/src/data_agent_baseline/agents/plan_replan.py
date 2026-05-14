from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.llm_extractor import LLMExtractor
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_init_plan_prompt,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.planning_tools import (
    PLANNING_TOOL_SPECS,
    create_planning_tool_handlers,
)
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry, ToolSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("PlanReplanAgent")

# ============================================================
# Replan system prompt — 参考 vper / DeepEye
# ============================================================

REPLAN_SYSTEM_PROMPT = """You are a replanning agent that reviews execution progress and revises the plan.

Given the current plan and execution trajectory, decide what to do next:

Decision types:
- "continue": The plan is fine, keep following it. Revised plan should be the original plan.
- "adjust": Modify the plan based on findings — add, remove, reorder, or rewrite steps.
- "complete": All needed work is done but answer hasn't been submitted — add an "answer" step.

Output format (JSON in code block):
```json
{
  "decision": "adjust",
  "reason": "Step 2 failed because the SQL query had a syntax error, switching to Python",
  "revised_plan": [
    {
      "step": 1,
      "description": "Read knowledge.md to understand available data sources",
      "action": "read_doc",
      "details": "Target: knowledge.md"
    },
    {
      "step": 2,
      "description": "Execute the analysis using Python",
      "action": "execute_python",
      "details": "Use pandas to read and query the data"
    },
    {
      "step": 3,
      "description": "Submit the final answer",
      "action": "answer",
      "details": "Format as columns and rows"
    }
  ]
}
```

Rules:
1. If a tool call failed, suggest an alternative approach (different tool or different logic).
2. If the same action was repeated without progress, suggest a different strategy.
3. Keep the revised plan concise — 3 to 7 steps.
4. Mark steps that are already done as completed so the executor can skip them.
5. The last step should always be "answer" to submit results.
6. Include enough detail in "details" so the executor knows exactly what to do.
"""


# ============================================================
# JSON / parse utilities
# ============================================================

def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _save_trace(path: Path, run_result: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_result, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass


def _repair_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()

    closing = []
    for ch in reversed(stack):
        closing.append("}" if ch == "{" else "]")
    return text + "".join(closing)


def _load_single_json_object(text: str) -> dict[str, object]:
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        text = _repair_json(text)
        payload, end = json.JSONDecoder().raw_decode(text)

    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    """Parse a model response into a ModelStep. Used as rule_parser for LLMExtractor."""
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


# ============================================================
# Trigger detection
# ============================================================

def _detect_replan_triggers(
    error_streak: int,
    last_actions: list[tuple[str, str]],
    error_threshold: int = 2,
    repeat_threshold: int = 2,
) -> str | None:
    """Check if replan should be auto-triggered. Returns reason or None."""
    if error_streak >= error_threshold:
        return f"auto: {error_streak} consecutive errors"

    if len(last_actions) >= repeat_threshold:
        recent = last_actions[-repeat_threshold:]
        if len(set(recent)) == 1:
            action, action_input = recent[0]
            return f"auto: repeated action '{action}' {repeat_threshold} times with same input"

    return None


# ============================================================
# Trajectory summary builder
# ============================================================

def _build_trajectory_summary(state: AgentRuntimeState) -> str:
    lines: list[str] = []
    for step in state.steps:
        status = "OK" if step.ok else "FAIL"
        action_input_str = json.dumps(step.action_input, ensure_ascii=False)[:120]
        lines.append(f"[{status}] Step {step.step_index}: {step.action} | input={action_input_str}")
        if not step.ok:
            error = step.observation.get("error", step.observation.get("content", {}).get("error", "unknown"))
            lines.append(f"  -> error: {error}")
        elif "content" in step.observation:
            content_preview = json.dumps(step.observation["content"], ensure_ascii=False)[:200]
            lines.append(f"  -> result: {content_preview}")
    return "\n".join(lines) if lines else "(no steps executed yet)"


# ============================================================
# Replan response parsing & plan formatting
# ============================================================

def _parse_replan_response(raw_response: str) -> dict[str, Any] | None:
    try:
        text = _strip_json_fence(raw_response)
        return _load_single_json_object(text)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse replan response, raw: %.300s", raw_response)
        return None


def _parse_plan_response(raw_response: str) -> list[dict] | None:
    """Parse initial plan LLM response into a list of step dicts."""
    normalized = _strip_json_fence(raw_response)
    try:
        payload = _load_single_json_object(normalized)
    except (ValueError, json.JSONDecodeError):
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict) and "steps" in payload:
        return payload["steps"]
    if isinstance(payload, list):
        return payload
    return None


def _format_plan_from_json(plan_items: list[dict]) -> str:
    """Format a JSON plan array into the prompt-friendly string format."""
    lines = ["Current Plan:"]
    for item in plan_items:
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


# ============================================================
# Replan tool handler factory
# ============================================================

def create_replan_tool_handler(
    model: ModelAdapter,
    plan_state: dict,
    state: AgentRuntimeState,
) -> Any:
    """Create a replan tool handler — closure capturing model + plan_state + trajectory.

    When called, invokes LLM to analyze the current trajectory and revise the plan.
    """

    def replan_handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        reason = action_input.get("reason", "manual trigger")
        logger.info("Replan triggered — reason: %s", reason)

        current_plan = plan_state.get("current_plan", "")
        trajectory = _build_trajectory_summary(state)

        replan_prompt = f"""=== Current Plan ===
{current_plan}

=== Execution Trajectory ===
{trajectory}

=== Trigger Reason ===
{reason}

=== Task Question ===
{task.question}

=== Instruction ===
Review the plan and trajectory above. Decide whether to continue, adjust the plan, or wrap up.
Output your decision as a single JSON object in a ```json fenced block."""

        messages = [
            ModelMessage(role="system", content=REPLAN_SYSTEM_PROMPT),
            ModelMessage(role="user", content=replan_prompt),
        ]

        try:
            raw_response = model.complete(messages)
            logger.info("Replan LLM response (first 500 chars):\n%.500s", raw_response)
            decision = _parse_replan_response(raw_response)

            if decision is None:
                return ToolExecutionResult(
                    ok=False,
                    content={"error": "Failed to parse replan response", "plan_unchanged": True},
                )

            decision_type = decision.get("decision", "continue")
            decision_reason = decision.get("reason", "")

            if decision_type == "adjust" and decision.get("revised_plan"):
                revised_plan = decision["revised_plan"]
                if isinstance(revised_plan, list) and len(revised_plan) > 0:
                    new_plan = _format_plan_from_json(revised_plan)
                    plan_state["current_plan"] = new_plan
                    plan_state["all_step_count"] = len(revised_plan)
                    return ToolExecutionResult(
                        ok=True,
                        content={
                            "decision": "adjust",
                            "reason": decision_reason,
                            "revised_plan": new_plan,
                            "step_count": len(revised_plan),
                        },
                    )

            if decision_type == "continue":
                return ToolExecutionResult(
                    ok=True,
                    content={
                        "decision": "continue",
                        "reason": decision_reason,
                        "message": "Plan is fine, continue execution.",
                    },
                )

            if decision_type == "complete":
                return ToolExecutionResult(
                    ok=True,
                    content={
                        "decision": "complete",
                        "reason": decision_reason,
                        "message": "Work appears done, consider submitting the answer.",
                    },
                )

            return ToolExecutionResult(
                ok=True,
                content={"decision": decision_type, "reason": decision_reason},
            )

        except Exception as exc:
            logger.error("Replan LLM call failed: %s", exc)
            return ToolExecutionResult(
                ok=False,
                content={"error": f"Replan failed: {exc}", "plan_unchanged": True},
            )

    return replan_handler


# ============================================================
# Agent config & Agent
# ============================================================

@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_steps: int = 16
    trace_save_path: Path | None = None
    enable_auto_replan: bool = True
    enable_planning_tools: bool = True
    error_streak_threshold: int = 2
    repeat_streak_threshold: int = 2
    llm_extractor_max_retries: int = 3


class Agent:
    """Plan + ReAct agent with planning tools, replan, and auto-trigger.

    Integrates patterns from DeepEye:
    - Planning tools (create_plan, update_plan, mark_step_done)
    - Replan tool (LLM-driven plan revision)
    - LLMExtractor for robust structured parsing with retry
    - Auto-trigger on consecutive errors or repeated steps
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self._plan_state: dict = {"current_plan": "", "completed_steps": set(), "all_step_count": 0}
        self._extractor = LLMExtractor(max_retries=self.config.llm_extractor_max_retries)

    # ---- message builders ----

    def _build_messages(
        self, task: PublicTask, state: AgentRuntimeState, plan: str
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        messages.append(ModelMessage(role="user", content=plan))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def _build_plan_messages(self, task: PublicTask) -> list[ModelMessage]:
        init_plan_prompt = build_init_plan_prompt(
            tool_descriptions=self.tools.describe_for_prompt(),
        )
        messages = [ModelMessage(role="system", content=init_plan_prompt)]
        user_prompt = f"The INPUT Question: {task.question}"
        messages.append(ModelMessage(role="user", content=user_prompt))
        return messages

    def _parse_plan_to_react(self, plan_raw: str) -> str:
        """Parse initial plan raw response into formatted plan string."""
        parsed = _parse_plan_response(plan_raw)
        if parsed is not None:
            return _format_plan_from_json(parsed)
        # Fallback: use raw text
        return f"Current Plan:\n{plan_raw[:2000]}"

    def _build_partial_result(self, task: PublicTask, state: AgentRuntimeState) -> dict:
        result = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
        return result.to_dict()

    # ---- inject runtime tools ----

    def _inject_runtime_tools(self, state: AgentRuntimeState) -> dict:
        """Inject planning tools + replan tool into the registry.

        Returns the replan_handler for direct use in auto-triggers.
        """
        # Planning tools
        if self.config.enable_planning_tools:
            planning_handlers = create_planning_tool_handlers(self._plan_state)
            for name, spec in PLANNING_TOOL_SPECS.items():
                self.tools.specs[name] = spec
                self.tools.handlers[name] = planning_handlers[name]

        # Replan tool
        replan_handler = create_replan_tool_handler(
            self.model, self._plan_state, state,
        )
        self.tools.specs["replan"] = ToolSpec(
            name="replan",
            description=(
                "Revise the current execution plan when you are stuck. "
                "Use this when: (a) a tool call keeps failing, (b) you tried the same thing "
                "multiple times without progress, or (c) you discover new information that "
                "changes the approach. The tool will return a revised plan."
            ),
            input_schema={"reason": "why you need to revise the plan"},
        )
        self.tools.handlers["replan"] = replan_handler

        return replan_handler

    # ---- main run loop ----

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        trace_path = self.config.trace_save_path

        # -- Phase 0: inject runtime tools (planning + replan) --
        replan_handler = self._inject_runtime_tools(state)

        # -- Phase 1: initial plan (use LLMExtractor for retry) --
        plan_messages = self._build_plan_messages(task)
        plan_results, _ = self._extractor.extract_with_retry(
            model=self.model,
            messages=plan_messages,
            rule_parser=_parse_plan_response,
            n=1,
        )
        if plan_results:
            init_plan_format = _format_plan_from_json(plan_results[0])
            self._plan_state["all_step_count"] = len(plan_results[0])
        else:
            # Fallback
            init_plan_raw = self.model.complete(plan_messages)
            init_plan_format = self._parse_plan_to_react(init_plan_raw)
        self._plan_state["current_plan"] = init_plan_format
        logger.info("Initial plan:\n%s", init_plan_format)

        # -- tracking state for auto-triggers --
        error_streak = 0
        last_actions: list[tuple[str, str]] = []

        # -- Phase 2: ReAct loop --
        for step_index in range(1, self.config.max_steps + 1):
            current_plan = self._plan_state["current_plan"]

            # ---- auto-trigger check ----
            if self.config.enable_auto_replan:
                trigger_reason = _detect_replan_triggers(
                    error_streak,
                    last_actions,
                    error_threshold=self.config.error_streak_threshold,
                    repeat_threshold=self.config.repeat_streak_threshold,
                )
                if trigger_reason is not None:
                    logger.warning(
                        "[%d/%d] Auto-trigger replan: %s",
                        step_index, self.config.max_steps, trigger_reason,
                    )
                    replan_result = replan_handler(task, {"reason": trigger_reason})

                    virtual_obs = {
                        "ok": replan_result.ok,
                        "tool": "__auto_replan__",
                        "content": replan_result.content,
                    }
                    state.steps.append(StepRecord(
                        step_index=step_index,
                        thought="Auto-replan triggered",
                        action="__auto_replan__",
                        action_input={"reason": trigger_reason},
                        raw_response='{"thought":"Auto-replan triggered","action":"__auto_replan__","action_input":{}}',
                        observation=virtual_obs,
                        ok=replan_result.ok,
                    ))

                    if trace_path:
                        _save_trace(trace_path, self._build_partial_result(task, state))

                    error_streak = 0
                    last_actions.clear()
                    continue

            # ---- model call (with LLMExtractor for retry on parse failure) ----
            call_messages = self._build_messages(task, state, current_plan)
            step_results_list, _ = self._extractor.extract_with_retry(
                model=self.model,
                messages=call_messages,
                rule_parser=parse_model_step,
                n=1,
            )

            if not step_results_list:
                # All retries exhausted — record as error
                raw_response = "LLMExtractor: all retries exhausted"
                logger.error("[%d/%d] All LLM retries exhausted", step_index, self.config.max_steps)
                error_streak += 1
                observation = {"ok": False, "error": "Model response parsing failed after retries"}
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__parse_error__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                ))
                if trace_path:
                    _save_trace(trace_path, self._build_partial_result(task, state))
                continue

            model_step = step_results_list[0]
            raw_response = model_step.raw_response

            try:
                logger.info(
                    "[%d/%d] Thought: %s | Action: %s | Input: %s",
                    step_index, self.config.max_steps,
                    model_step.thought[:150], model_step.action,
                    json.dumps(model_step.action_input, ensure_ascii=False)[:150],
                )

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                logger.info(
                    "[%d/%d] Observation ok=%s tool=%s",
                    step_index, self.config.max_steps,
                    tool_result.ok, model_step.action,
                )

                # -- update trigger tracking --
                if tool_result.ok:
                    error_streak = 0
                else:
                    error_streak += 1
                    logger.warning(
                        "[%d/%d] Tool error (streak=%d): %s",
                        step_index, self.config.max_steps, error_streak,
                        tool_result.content.get("error", "unknown"),
                    )

                action_key = (
                    model_step.action,
                    json.dumps(model_step.action_input, sort_keys=True, ensure_ascii=False),
                )
                last_actions.append(action_key)
                if len(last_actions) > 3:
                    last_actions.pop(0)

                # -- record step --
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)

                if trace_path:
                    _save_trace(trace_path, self._build_partial_result(task, state))

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    logger.info(
                        "[%d/%d] Terminal — answer submitted with %d columns",
                        step_index, self.config.max_steps,
                        len(tool_result.answer.columns) if tool_result.answer else 0,
                    )
                    break

            except Exception as exc:
                logger.error("[%d/%d] Exception: %s", step_index, self.config.max_steps, exc)
                error_streak += 1
                observation = {"ok": False, "error": str(exc)}
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__error__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                ))
                if trace_path:
                    _save_trace(trace_path, self._build_partial_result(task, state))

        # -- finalize --
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        result = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )

        if trace_path:
            _save_trace(trace_path, result.to_dict())

        return result
