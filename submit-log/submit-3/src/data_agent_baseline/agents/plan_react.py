from __future__ import annotations

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
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class PlanReActAgentConfig:
    max_steps: int = 16
    explore_steps: int = 3  # Minimum exploration steps (adaptive)
    verify_before_answer: bool = True
    trace_save_path: Path | None = None


ADAPTIVE_SYSTEM_PROMPT = """You are a data analysis agent. You solve data tasks by exploring data and producing answers.

## APPROACH: EXPLORE FIRST, THEN ANSWER

1. FIRST: Understand the data
   - Read knowledge.md to understand the data schema
   - List context files to see what's available
   - Explore the data structure (tables, columns, sample data)

2. THEN: Answer the question
   - Use SQL or Python to get the answer
   - Verify your answer before submitting

## CRITICAL: COLUMN NAMES DON'T MATTER - ONLY VALUES MATTER!

The evaluation system IGNORES column names and only compares VALUES.
Focus on getting the CORRECT VALUES, not the perfect column names.

## RULES FOR VALUES

### 1. Get the RIGHT answer
- Read the question carefully
- Filter data correctly (don't return all rows!)
- Use the correct aggregation (COUNT, SUM, AVG, etc.)
- Don't round numbers unnecessarily

### 2. Don't return extra rows
- If the question asks for "the" answer, return ONE row
- If the question asks for "all" items matching criteria, return those items
- Don't return all rows from the table

### 3. Use correct aggregation
- COUNT = count rows
- SUM = add values
- AVG = average values
- Don't use SUM when AVG is needed
- Don't use COUNT when you need to list individual rows

### 4. Preserve numerical precision
- Don't round numbers unless necessary
- Use full precision from the data
- Example: 60.77956989247312, not 60.78

### 5. Date format
- Use ISO format: YYYY-MM-DD
- Example: 2024-03-01, not 2024-3-1

### 6. Multiple columns
- If the question asks for multiple things, include all columns
- Example: "name and age" -> two columns

## ANSWERING

1. Call the `answer` tool when you have the final result
2. The `answer` tool needs `columns` and `rows`
3. Each row must have the same number of elements as columns
4. Return exactly one JSON object with keys `thought`, `action`, and `action_input`
5. Wrap it in a ```json fenced code block
""".strip()


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
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


def _save_trace(path: Path, run_result: dict) -> None:
    """Save trace to disk (for timeout recovery)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_result, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass


class PlanReActAgent:
    """Adaptive two-phase ReAct agent: explore first, then answer."""
    # this is 1004-v1
    # idea16

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: PlanReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or PlanReActAgentConfig()
        self.system_prompt = system_prompt or ADAPTIVE_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))

        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def _validate_answer(self, action_input: dict) -> str | None:
        """Validate answer before submission. Returns error message or None if valid."""
        columns = action_input.get("columns", [])
        rows = action_input.get("rows", [])

        if not columns:
            return "Answer must have at least one column."
        if not rows:
            return "Answer must have at least one row."
        if not all(isinstance(c, str) for c in columns):
            return "All column names must be strings."
        if not all(isinstance(r, list) for r in rows):
            return "Each row must be a list."

        bad_rows = [i for i, row in enumerate(rows) if len(row) != len(columns)]
        if bad_rows:
            return f"Rows {bad_rows} have wrong number of columns (expected {len(columns)})."

        return None

    def _has_explored(self, state: AgentRuntimeState) -> bool:
        """Check if the agent has done enough exploration."""
        explore_tools = {"list_context", "read_doc", "read_csv", "read_json", "inspect_sqlite_schema"}
        explore_count = sum(1 for s in state.steps if s.action in explore_tools)
        return explore_count >= self.config.explore_steps

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

        for step_index in range(1, self.config.max_steps + 1):
            # Check if agent has explored enough
            explored = self._has_explored(state)
            
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)

                # If not explored enough, discourage answering
                if not explored and model_step.action == "answer":
                    observation = {
                        "ok": False,
                        "tool": "__explore_first__",
                        "content": {"error": "Please explore the data first (read knowledge.md, list files, understand schema) before answering."},
                    }
                    step_record = StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action="__explore_first__",
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                    state.steps.append(step_record)
                    if trace_path:
                        _save_trace(trace_path, self._build_partial_result(task, state))
                    continue

                # Validate answer before submitting
                if model_step.action == "answer" and self.config.verify_before_answer:
                    validation_error = self._validate_answer(model_step.action_input)
                    if validation_error:
                        observation = {
                            "ok": False,
                            "tool": "__validation__",
                            "content": {"error": validation_error, "hint": "Please fix and try again."},
                        }
                        step_record = StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action="__validation_error__",
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                        state.steps.append(step_record)
                        if trace_path:
                            _save_trace(trace_path, self._build_partial_result(task, state))
                        continue

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
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

                # Save trace after each step
                if trace_path:
                    _save_trace(trace_path, self._build_partial_result(task, state))

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break

            except Exception as exc:
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
