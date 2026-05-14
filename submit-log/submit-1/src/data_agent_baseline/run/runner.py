from __future__ import annotations
import csv
import json
import logging
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.plan_replan import Agent
from data_agent_baseline.agents.plan_replan import AgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig, DatasetConfig, RunConfig, load_app_config
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        return ""
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    if effective_run_id:
        run_output_dir = output_root / effective_run_id
    else:
        run_output_dir = output_root
    run_output_dir.mkdir(parents=True, exist_ok=not bool(effective_run_id))
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=os.environ.get("MODEL_NAME", config.agent.model),
        api_base=os.environ.get("MODEL_API_URL", config.agent.api_base),
        api_key=os.environ.get("MODEL_API_KEY", config.agent.api_key),
        temperature=float(os.environ.get("MODEL_TEMPERATURE", config.agent.temperature)),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    trace_save_path: Path | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    # Use PlanReActAgent with trace saving
    agent = Agent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=AgentConfig(
            max_steps=config.agent.max_steps,
            trace_save_path=trace_save_path,
        ),
    )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any], trace_save_path: str | None = None) -> None:
    try:
        from pathlib import Path
        trace_path = Path(trace_save_path) if trace_save_path else None
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config, trace_save_path=trace_path),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _try_recover_from_trace(task_id: str, config: AppConfig) -> dict[str, Any] | None:
    """Try to recover a successful result from an existing trace.json file."""
    try:
        run_output_dir = config.run.output_dir
        run_id = config.run.run_id
        if run_id:
            trace_path = run_output_dir / run_id / task_id / "trace.json"
        else:
            # Try to find the latest run
            runs = sorted(run_output_dir.iterdir(), reverse=True)
            for run_dir in runs:
                if run_dir.is_dir():
                    trace_path = run_dir / task_id / "trace.json"
                    if trace_path.exists():
                        break
            else:
                return None
        
        if not trace_path.exists():
            return None
        
        with open(trace_path) as f:
            trace = json.load(f)
        
        # Check if the trace shows success
        if trace.get("succeeded") and trace.get("answer"):
            return trace
    except Exception:
        pass
    return None


def _try_recover_from_specific_trace(trace_path_str: str) -> dict[str, Any] | None:
    """Try to recover a result from a specific trace.json file (including partial/incremental)."""
    try:
        trace_path = Path(trace_path_str)
        if not trace_path.exists():
            return None
        
        with open(trace_path) as f:
            trace = json.load(f)
        
        # If the trace has a successful answer, return it
        if trace.get("succeeded") and trace.get("answer"):
            return trace
        
        # If the trace has steps but no answer, return it as a partial result
        # (the steps are valuable even if the agent didn't finish)
        if trace.get("steps"):
            return trace
    except Exception:
        pass
    return None


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    
    # Compute trace save path for incremental saving
    run_output_dir = config.run.output_dir
    run_id = config.run.run_id
    if run_id:
        trace_save_dir = run_output_dir / run_id / task_id
    else:
        trace_save_dir = run_output_dir / task_id
    trace_save_dir.mkdir(parents=True, exist_ok=True)
    trace_save_path = str(trace_save_dir / "trace.json")
    
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config, trace_save_path=Path(trace_save_path))

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue, trace_save_path),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        
        # Try to recover from the trace file that was saved incrementally
        recovered = _try_recover_from_specific_trace(trace_save_path)
        if recovered:
            return recovered
        
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            recovered = _try_recover_from_specific_trace(trace_save_path)
            if recovered:
                return recovered
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        
        recovered = _try_recover_from_specific_trace(trace_save_path)
        if recovered:
            return recovered
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    result = queue.get()
    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        # Compute trace save path for direct execution
        trace_save_path = run_output_dir / task_id / "trace.json"
        trace_save_path.parent.mkdir(parents=True, exist_ok=True)
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools, trace_save_path=trace_save_path)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts


def run_submission(config_path: str | None = None) -> None:
    """Run in evaluation/submission mode. Reuses run_benchmark internally.

    When /input exists (Docker eval env): uses /input, /output, /logs.
    Otherwise (local testing): uses paths from config or defaults.
    Model API config is always read from env vars first, falling back to YAML.
    """
    eval_mode = Path("/input").is_dir()

    # --- Logging ---
    log_dir = Path("/logs") if eval_mode else Path("artifacts/eval_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runtime.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    # Suppress verbose httpx/openai HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    logger.info("=== DABench Submission Started ===")

    # --- Load config (env vars override model settings in load_app_config) ---
    if config_path:
        app_config = load_app_config(Path(config_path))
        logger.info("Loaded config from %s", config_path)
    else:
        app_config = AppConfig()
        logger.info("No config provided, using defaults.")

    # --- Override paths for eval mode ---
    if eval_mode:
        app_config = AppConfig(
            dataset=DatasetConfig(root_path=Path("/input")),
            agent=app_config.agent,
            run=RunConfig(
                output_dir=Path("/output"),
                run_id="",  # flat output: /output/task_<id>/prediction.csv
                max_workers=app_config.run.max_workers,
                task_timeout_seconds=app_config.run.task_timeout_seconds,
            ),
        )

    logger.info("MODEL_API_URL: %s", app_config.agent.api_base)
    logger.info("MODEL_NAME: %s", app_config.agent.model)
    logger.info("max_steps: %d", app_config.agent.max_steps)
    logger.info("temperature: %.2f", app_config.agent.temperature)
    logger.info("max_workers: %d", app_config.run.max_workers)
    logger.info("task_timeout_seconds: %d", app_config.run.task_timeout_seconds)
    logger.info("input_dir: %s", app_config.dataset.root_path)
    logger.info("output_dir: %s", app_config.run.output_dir)

    # --- Run (reuses all existing logic) ---
    run_output_dir, artifacts = run_benchmark(config=app_config)

    succeeded = sum(1 for a in artifacts if a.succeeded)
    logger.info(
        "[%s] All tasks completed. %d/%d succeeded.",
        time.strftime("%H:%M:%S"),
        succeeded,
        len(artifacts),
    )
