from __future__ import annotations
import logging
import json
import re
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
    build_init_plan_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("PlanReactAgent")


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    trace_save_path: Path | None = None


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
    """Save trace to disk (for timeout recovery)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_result, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass


def _repair_json(text: str) -> str:
    """Auto-repair common LLM JSON errors such as missing closing braces/brackets."""
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


class PlanAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState, plan: str) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        # plan inject
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
        normalized = _strip_json_fence(plan_raw)
        try:
            payload = _load_single_json_object(normalized)
        except ValueError:
            # Try parsing as JSON array first
            try:
                payload = json.loads(normalized)
            except json.JSONDecodeError:
                # If all parsing fails, use raw response as plan
                return f"Current Plan:\n{normalized[:2000]}"
        
        # If payload is a dict with "steps" key, extract steps
        if isinstance(payload, dict) and "steps" in payload:
            items = payload["steps"]
        elif isinstance(payload, list):
            items = payload
        else:
            # Can't parse as structured plan, return raw
            return f"Current Plan:\n{normalized[:2000]}"
        
        plan_format_lines = ["Current Plan:"]
        for item in items:
            if not isinstance(item, dict):
                continue
            step_num = item.get("step", item.get("step_id", "?"))
            description = item.get("description", "")
            action = item.get("action", item.get("tool", ""))
            details = item.get("details", item.get("hint", ""))
            step_prompt = f"Step {step_num}: Description: {description} | Action: {action} | Details: {details}"
            plan_format_lines.append(step_prompt)
        return "\n".join(plan_format_lines)

    def _build_partial_result(self, task: PublicTask, state: AgentRuntimeState) -> dict:
        result = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
        return result.to_dict()

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        trace_path = self.config.trace_save_path

        # init the plan
        init_plan_raw = self.model.complete(self._build_plan_messages(task))
        init_plan_format = self._parse_plan_to_react(init_plan_raw)
        logger.info(f"the init plan:\n{init_plan_format}")

        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state, init_plan_format))
            try:
                model_step = parse_model_step(raw_response)
                logger.info(f"[{step_index} / {self.config.max_steps}] Model Step:\nThink:{model_step.thought}\nAction:{model_step.action}\nAction-Input:{model_step.action_input}")
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                logger.info(f"[{step_index} / {self.config.max_steps}] Observation:\n{observation}")
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

                # Save trace after each step (incremental trace saving)
                if trace_path:
                    _save_trace(trace_path, self._build_partial_result(task, state))

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                logger.error(f"[{step_index} / {self.config.max_steps}] Error: {exc}")
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                # Save trace after error (incremental trace saving)
                if trace_path:
                    _save_trace(trace_path, self._build_partial_result(task, state))

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        result = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )

        # Final trace save
        if trace_path:
            _save_trace(trace_path, result.to_dict())

        return result
