"""
SupervisorAgent — 参考 DeepEye Supervisor + ExecutionEngine

编排模式：
  1. Planner LLM 生成带依赖关系的 DAG 计划
  2. DAGExecutor 按拓扑顺序并行执行就绪步骤
  3. 每个步骤通过 ToolRegistry 执行（ReAct 子循环）
  4. 步骤失败 → replan 修订计划 → 继续执行
  5. 最终合成答案
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
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
from data_agent_baseline.dag.executor import (
    DAGExecutor,
    DAGExecutionConfig,
    DAGPlan,
    DAGStep,
    build_dag_plan_from_json,
)
from data_agent_baseline.nl2sql.pipeline import NL2SQLPipeline
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

logger = logging.getLogger("SupervisorAgent")

# ============================================================
# Supervisor plan prompt — 生成 DAG 计划
# ============================================================

SUPERVISOR_PLAN_PROMPT = """You are a supervisor agent that creates DAG execution plans for data analysis tasks.

## Task
Given a data analysis question, create a structured plan where each step declares its dependencies.
Steps without dependencies can run in parallel.

## Output Format (JSON in code block):
```json
{
  "plan_type": "dag",
  "steps": [
    {
      "step": 1,
      "description": "Read knowledge.md to understand data schema",
      "action": "read_doc",
      "details": "Target: knowledge.md",
      "depends_on": []
    },
    {
      "step": 2,
      "description": "List all context files",
      "action": "list_context",
      "details": "See available data files",
      "depends_on": []
    },
    {
      "step": 3,
      "description": "Execute analysis query using SQL",
      "action": "execute_context_sql",
      "details": "SELECT ... FROM ... WHERE ...",
      "depends_on": [1, 2]
    },
    {
      "step": 4,
      "description": "Submit the final answer",
      "action": "answer",
      "details": "Format as columns and rows",
      "depends_on": [3]
    }
  ]
}
```

## DAG Planning Rules:
1. `depends_on` lists the step numbers that must complete BEFORE this step
2. Steps with empty `depends_on` can execute in parallel
3. Data exploration steps (read_doc, list_context, read_csv) usually have no dependencies
4. Analysis steps depend on exploration steps
5. The final `answer` step depends on all analysis steps
6. Use `execute_context_sql` for SQL queries, `execute_python` for complex transformations
7. Keep the plan to 4-7 steps
"""

REPLAN_PROMPT = """You are a supervisor replanner. Review the execution results and revise the plan.

Current Plan:
{PLAN}

Execution Results:
{RESULTS}

Failed Steps:
{FAILURES}

Decide: "continue" (keep going), "adjust" (revise plan), or "complete" (done).

Output format (JSON in code block):
```json
{
  "decision": "adjust",
  "reason": "...",
  "revised_plan": [...]
}
```
"""


# ============================================================
# Utilities
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


def _save_trace(path: Path, run_result: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_result, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass


def _parse_plan_response(raw_response: str) -> list[dict] | None:
    """Parse supervisor plan LLM response into step dicts."""
    try:
        text = _strip_json_fence(raw_response)
        payload = _load_single_json_object(text)
    except Exception:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict) and "steps" in payload:
        return payload["steps"]
    if isinstance(payload, list):
        return payload
    return None


# ============================================================
# Agent config & SupervisorAgent
# ============================================================

@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    max_steps: int = 20
    trace_save_path: Path | None = None
    dag_max_concurrency: int = 4
    dag_fail_fast: bool = False
    enable_nl2sql: bool = True
    enable_auto_replan: bool = True
    max_replan_rounds: int = 3
    llm_extractor_max_retries: int = 3
    react_sub_steps: int = 10  # ReAct sub-steps per DAG step (should be small — each step is one action)


class SupervisorAgent:
    """Supervisor orchestrator — DAG planner + ExecutionEngine + sub-agent delegation.

    参考 DeepEye 的 SupervisorAgent + ExecutionEngine 设计：
    - 生成 DAG 计划，声明步骤依赖
    - 用 DAGExecutor 并行执行无依赖步骤
    - 每个步骤内部是简短的 ReAct 循环
    - 步骤失败时自动 replan
    - 可选：对 SQL 步骤使用 NL2SQL pipeline
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: SupervisorConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or SupervisorConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.extractor = LLMExtractor(max_retries=self.config.llm_extractor_max_retries)
        self.dag_executor = DAGExecutor(DAGExecutionConfig(
            max_concurrency=self.config.dag_max_concurrency,
            fail_fast=self.config.dag_fail_fast,
        ))
        self._plan_state: dict = {"current_plan": "", "completed_steps": set(), "all_step_count": 0}

    # ---- message builders ----

    def _build_plan_messages(self, task: PublicTask) -> list[ModelMessage]:
        prompt = SUPERVISOR_PLAN_PROMPT + "\n\n## Available tools:\n" + self.tools.describe_for_prompt()
        return [
            ModelMessage(role="system", content=prompt),
            ModelMessage(role="user", content=f"Question: {task.question}\nCreate a DAG execution plan."),
        ]

    def _build_react_messages(
        self, task: PublicTask, state: AgentRuntimeState, step: DAGStep, dep_results: dict
    ) -> list[ModelMessage]:
        # Direct prompt: just execute the assigned action, no extra exploration
        direct_prompt = f"""You are executing ONE specific step in a data analysis plan.

## Question: {task.question}

## Your Task: {step.description}
Use the '{step.action}' tool with these instructions: {step.action_input}

## Results from Previous Steps:
{json.dumps(dep_results, ensure_ascii=False)[:800]}

## Tools Available
{self.tools.describe_for_prompt()}

## Rules
1. Call the '{step.action}' tool NOW with appropriate parameters based on the results above
2. If the tool returns data, analyze it and call the NEXT needed tool
3. If you have the final answer, call 'answer' with columns and rows
4. Return exactly one JSON object in a ```json fenced block with thought, action, action_input
5. Be concise — do not explore further than needed
"""
        messages = [ModelMessage(role="system", content=direct_prompt)]
        for record in state.steps:
            messages.append(ModelMessage(role="assistant", content=record.raw_response))
            messages.append(ModelMessage(
                role="user", content=build_observation_prompt(record.observation)
            ))
        return messages

    # ---- step executor (sub-agent ReAct loop) ----

    def _execute_step_react(
        self, task: PublicTask, step: DAGStep, dep_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single DAG step with a short ReAct loop."""
        state = AgentRuntimeState()

        for sub_step in range(1, self.config.react_sub_steps + 1):
            messages = self._build_react_messages(task, state, step, dep_results)
            model_results, _ = self.extractor.extract_with_retry(
                model=self.model,
                messages=messages,
                rule_parser=self._parse_model_step,
                n=1,
            )

            if not model_results:
                error_obs = {"ok": False, "error": "Model parse failed after retries"}
                state.steps.append(StepRecord(
                    step_index=sub_step, thought="", action="__parse_error__",
                    action_input={}, raw_response="", observation=error_obs, ok=False,
                ))
                break

            model_step = model_results[0]
            try:
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(StepRecord(
                    step_index=sub_step, thought=model_step.thought,
                    action=model_step.action, action_input=model_step.action_input,
                    raw_response=model_step.raw_response, observation=observation,
                    ok=tool_result.ok,
                ))
                if tool_result.is_terminal:
                    answer_val = tool_result.answer
                    if answer_val is not None and hasattr(answer_val, 'to_dict'):
                        answer_val = answer_val.to_dict()
                    return {
                        "done": True,
                        "answer": answer_val,
                        "sub_steps": [s.to_dict() for s in state.steps],
                    }
                if sub_step == self.config.react_sub_steps:
                    # Max sub-steps reached, return last observation
                    return {
                        "done": False,
                        "last_result": tool_result.content,
                        "sub_steps": [s.to_dict() for s in state.steps],
                    }
            except Exception as exc:
                state.steps.append(StepRecord(
                    step_index=sub_step, thought="", action="__error__",
                    action_input={}, raw_response=model_step.raw_response,
                    observation={"ok": False, "error": str(exc)}, ok=False,
                ))

        return {
            "done": False,
            "last_result": None,
            "sub_steps": [s.to_dict() for s in state.steps],
            "error": "Max sub-steps reached without terminal action",
        }

    def _parse_model_step(self, raw_response: str) -> ModelStep | None:
        """Parse a model response into ModelStep. Used as rule_parser for LLMExtractor."""
        from data_agent_baseline.agents.plan_replan import _repair_json
        try:
            text = _strip_json_fence(raw_response)
            try:
                payload, end = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                text = _repair_json(text)
                payload, end = json.JSONDecoder().raw_decode(text)
            if not isinstance(payload, dict):
                return None
            thought = payload.get("thought", "")
            action = payload.get("action")
            action_input = payload.get("action_input", {})
            if not isinstance(action, str) or not action:
                return None
            if not isinstance(action_input, dict):
                return None
            return ModelStep(
                thought=str(thought), action=action,
                action_input=action_input, raw_response=raw_response,
            )
        except Exception:
            return None

    # ---- step handler for DAGExecutor ----

    def _make_step_handler(self, task: PublicTask):
        """Create a step handler closure for the DAGExecutor."""

        def handler(step: DAGStep, dep_results: dict[str, Any]) -> dict[str, Any]:
            logger.info("Supervisor executing step: %s — %s", step.step_id, step.description[:80])

            # If NL2SQL enabled and step is SQL-related, try pipeline
            if (self.config.enable_nl2sql and
                    step.action == "execute_context_sql" and
                    isinstance(step.action_input, str) and
                    "SELECT" not in str(step.action_input).upper()):
                # step.action_input has a description, not actual SQL
                nl2sql_result = self._try_nl2sql(task, step, dep_results)
                if nl2sql_result:
                    return nl2sql_result

            # Default: ReAct sub-loop
            result = self._execute_step_react(task, step, dep_results)
            return result

        return handler

    def _try_nl2sql(
        self, task: PublicTask, step: DAGStep, dep_results: dict[str, Any]
    ) -> dict | None:
        """Try using NL2SQL pipeline for SQL-heavy steps."""
        try:
            # Get db path from dep_results or step hints
            db_path = ""
            for key, val in dep_results.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        if "sqlite" in str(v).lower() or ".db" in str(v).lower():
                            db_path = str(v)
                            break

            if not db_path:
                # Try to get from task context
                from pathlib import Path as P
                context = task.context_dir
                for f in context.iterdir():
                    if f.suffix in (".sqlite", ".db"):
                        db_path = str(f)
                        break

            if not db_path:
                return None

            # Build schema summary from dependency results
            schema_summary = json.dumps(dep_results, ensure_ascii=False)[:3000]

            pipeline = NL2SQLPipeline(
                model=self.model,
                db_path=db_path,
                schema_summary=schema_summary,
                sql_generation_budget=2,
            )
            nl2sql_result = pipeline.run(task.question)
            if nl2sql_result.get("success"):
                sql = nl2sql_result["sql"]
                # Execute the SQL using the tool
                from data_agent_baseline.tools.sqlite import execute_read_only_sql
                raw = execute_read_only_sql(P(db_path), sql, limit=200)
                return {
                    "done": True,
                    "sql": sql,
                    "result": raw,
                    "method": "nl2sql_pipeline",
                }
        except Exception as exc:
            logger.warning("NL2SQL pipeline failed: %s", exc)
        return None

    # ---- answer extraction ----

    @staticmethod
    def _extract_answer(step_results: dict[str, Any]) -> Any:
        """Search step results for an answer (checks done=True + sub_steps)."""
        from data_agent_baseline.benchmark.schema import AnswerTable
        for step_id, result in step_results.items():
            if not isinstance(result, dict):
                continue
            # Direct answer from terminal action
            ans = result.get("answer")
            if ans is not None:
                if isinstance(ans, dict) and "columns" in ans and "rows" in ans:
                    return AnswerTable(columns=ans["columns"], rows=ans["rows"])
            # Search sub_steps for answer tool call
            for sub in result.get("sub_steps", []):
                if not isinstance(sub, dict):
                    continue
                # Check if answer action was called
                if sub.get("action") == "answer" or sub.get("action") == "__validation_error__":
                    pass  # continue searching
                obs = sub.get("observation", {})
                if isinstance(obs, dict):
                    cont = obs.get("content", {})
                    if isinstance(cont, dict):
                        if "columns" in cont and "rows" in cont:
                            return AnswerTable(columns=list(cont["columns"]), rows=[list(r) for r in cont["rows"]])
                        # Check for nested status/result
                        for key in ("result", "data"):
                            nested = cont.get(key, {})
                            if isinstance(nested, dict) and "columns" in nested and "rows" in nested:
                                return AnswerTable(columns=list(nested["columns"]), rows=[list(r) for r in nested["rows"]])
        return None

    # ---- main run ----

    def run(self, task: PublicTask) -> AgentRunResult:
        trace_path = self.config.trace_save_path
        all_step_records: list[StepRecord] = []
        answer = None
        failure_reason = None

        # -- Inject planning tools --
        planning_handlers = create_planning_tool_handlers(self._plan_state)
        for name, spec in PLANNING_TOOL_SPECS.items():
            self.tools.specs[name] = spec
            self.tools.handlers[name] = planning_handlers[name]

        # -- Phase 1: Generate DAG plan --
        plan_messages = self._build_plan_messages(task)
        plan_results, _ = self.extractor.extract_with_retry(
            model=self.model, messages=plan_messages,
            rule_parser=_parse_plan_response, n=1,
        )

        if not plan_results:
            # Fallback: simple sequential plan
            plan_steps = [
                {"step": 1, "description": "Read knowledge.md", "action": "read_doc",
                 "details": "knowledge.md", "depends_on": []},
                {"step": 2, "description": "List context", "action": "list_context",
                 "details": "", "depends_on": []},
                {"step": 3, "description": "Answer question", "action": "answer",
                 "details": "Submit result", "depends_on": [1, 2]},
            ]
        else:
            plan_steps = plan_results[0]

        dag_plan = build_dag_plan_from_json(plan_steps)
        logger.info("Supervisor DAG plan: %d steps", len(dag_plan.steps))
        for s in dag_plan.steps:
            logger.info("  Step %s: %s (depends_on=%s, action=%s)",
                         s.step_id, s.description[:60], s.depends_on, s.action)

        # -- Phase 2: Execute via DAGExecutor --
        replan_rounds = 0
        final_answer = None

        while replan_rounds <= self.config.max_replan_rounds:
            # Reset pending steps for re-execution
            for step in dag_plan.steps:
                if step.status in ("failed",):
                    step.status = "pending"  # type: ignore[assignment]

            step_handler = self._make_step_handler(task)
            dag_result = self.dag_executor.execute(dag_plan, step_handler)

            logger.info(
                "DAG execution done: %d completed, %d failed",
                dag_result.steps_completed, dag_result.steps_failed,
            )

            # Check if answer was produced (search all results + sub_steps)
            final_answer = self._extract_answer(dag_result.step_results)

            if final_answer:
                break

            if dag_result.success:
                # All steps done but no answer found — add a final answer step
                logger.info("All DAG steps succeeded but no answer — adding answer step")
                answer_step = DAGStep(
                    step_id="final_answer", description="Submit answer",
                    action="answer", depends_on=[],
                )
                answer_result = step_handler(answer_step, dag_result.step_results)
                if isinstance(answer_result, dict):
                    final_answer = answer_result.get("answer")
                    # Also search sub_steps
                    for sub in answer_result.get("sub_steps", []):
                        if isinstance(sub, dict) and sub.get("action") == "answer":
                            content = sub.get("observation", {}).get("content", {})
                            if "columns" in content and "rows" in content:
                                from data_agent_baseline.benchmark.schema import AnswerTable
                                final_answer = AnswerTable(
                                    columns=content["columns"], rows=content["rows"],
                                )
                if final_answer:
                    break

            if not dag_result.success and self.config.enable_auto_replan:
                replan_rounds += 1
                if replan_rounds > self.config.max_replan_rounds:
                    break

                logger.info("Replanning round %d/%d", replan_rounds, self.config.max_replan_rounds)
                new_steps = self._replan(task, dag_plan, dag_result)
                if new_steps:
                    dag_plan = build_dag_plan_from_json(new_steps)
            else:
                break

        # -- Phase 3: Synthesize answer --
        if final_answer:
            answer = final_answer
        else:
            answer = self._extract_answer(dag_result.step_results)

        if answer is None:
            failure_reason = "Supervisor: no answer produced from any step"

        return AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=all_step_records,
            failure_reason=failure_reason,
        )

    def _replan(
        self, task: PublicTask, dag_plan: DAGPlan, dag_result: Any
    ) -> list[dict] | None:
        """Replan: analyze failures and generate revised plan."""
        failures = {k: v for k, v in dag_result.errors.items()}
        results_summary = {
            k: str(v)[:200] for k, v in dag_result.step_results.items()
        }

        prompt = REPLAN_PROMPT.format(
            PLAN=json.dumps([
                {"step": s.step_id, "description": s.description,
                 "action": s.action, "depends_on": s.depends_on}
                for s in dag_plan.steps
            ], ensure_ascii=False, indent=2),
            RESULTS=json.dumps(results_summary, ensure_ascii=False, indent=2),
            FAILURES=json.dumps(failures, ensure_ascii=False, indent=2),
        )

        messages = [ModelMessage(role="user", content=prompt)]

        def parse_replan(response: str) -> list[dict] | None:
            try:
                text = _strip_json_fence(response)
                payload = _load_single_json_object(text)
                if payload.get("decision") == "adjust" and payload.get("revised_plan"):
                    return payload["revised_plan"]
            except Exception:
                pass
            return None

        results, _ = self.extractor.extract_with_retry(
            model=self.model, messages=messages, rule_parser=parse_replan, n=1,
        )

        if results:
            logger.info("Replan produced %d steps", len(results[0]))
            return results[0]
        return None
