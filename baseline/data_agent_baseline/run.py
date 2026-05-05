#!/usr/bin/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
from data_agent_baseline.config import load_app_config, AppConfig
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task
)
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
console = Console()
def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"

def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"
def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }

def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )

def main():
    config_path = Path(
        r"G:\项目成果打包\kbbcup_dataAgent\baseline\configs\react_baseline.local.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    try:
        config = load_app_config(config_path)
    except UnicodeDecodeError as e:
        import yaml
        raw_text = config_path.read_text(encoding='utf-8')
        payload = yaml.safe_load(raw_text)
        from data_agent_baseline.config import DatasetConfig, AgentConfig, RunConfig
        config = AppConfig(
            dataset=DatasetConfig(root_path=Path(payload['dataset']['root_path'])),
            agent=AgentConfig(**payload['agent']),
            run=RunConfig(**payload.get('run', {}))
        )
    dataset_root = config.dataset.root_path
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_root}")
    try:
        dataset = DABenchPublicDataset(dataset_root)
        task_total = len(dataset.iter_tasks())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    if config.run.test_single_task:
        test_task_id = config.run.task_id
        try:
            # shared_model = build_model_adapter(config)
            # shared_tools = create_default_tool_registry()
            # import os
            # raw_data_paths = os.listdir(r"G:\项目成果打包\kbbcup_dataAgent\demo_samples\input")
            # processed_data_paths = os.listdir(r"G:\项目成果打包\kbbcup_dataAgent\artifacts\runs\baseline_task")
            # for proed_path in processed_data_paths:
            #     if proed_path in raw_data_paths:
            #         raw_data_paths.remove(proed_path)
            # from tqdm import tqdm
            # for process_id in tqdm(raw_data_paths, desc="Processing tasks", unit="task"):
            #     run_id = config.run.run_id or "debug_run"
            #     effective_run_id, run_output_dir = create_run_output_dir(
            #         config.run.output_dir,
            #         run_id=run_id
            #     )
            #     artifact = run_single_task(
            #         task_id=process_id,
            #         config=config,
            #         run_output_dir=run_output_dir,
            #         model=shared_model,
            #         tools=shared_tools
            #     )
            # 单任务
            shared_model = build_model_adapter(config)
            shared_tools = create_default_tool_registry()
            run_output_dir = Path(r"G:\项目成果打包\kbbcup_dataAgent\baseline\baseline_test_res\task_11")
            artifact = run_single_task(
                task_id=test_task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return
    else:
        print("batch推理")
        effective_workers = config.run.max_workers
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]|[/dim]"),
            TextColumn("[green]ok={task.fields[ok]}[/green]"),
            TextColumn("[red]fail={task.fields[fail]}[/red]"),
            TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
            TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
            TextColumn("[dim]|[/dim]"),
            TextColumn("{task.fields[speed]}"),
            TextColumn("[dim]| elapsed[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]| eta[/dim]"),
            TimeRemainingColumn(),
            TextColumn("[dim]|[/dim]"),
            TextColumn("{task.fields[last]}"),
        ]
        with Progress(*progress_columns, console=console) as progress:
            progress_task_id = progress.add_task(
                "Benchmark",
                total=task_total,
                completed=0,
                **_build_compact_progress_fields(
                    completed_count=0,
                    succeeded_count=0,
                    failed_count=0,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=0.0,
                    last_artifact=None,
                ),
            )

            completion_count = 0
            succeeded_count = 0
            failed_count = 0
            start_time = perf_counter()

            def on_task_complete(artifact) -> None:
                nonlocal completion_count, succeeded_count, failed_count
                completion_count += 1
                if artifact.succeeded:
                    succeeded_count += 1
                else:
                    failed_count += 1
                progress.update(
                    progress_task_id,
                    completed=completion_count,
                    description="Benchmark",
                    refresh=True,
                    **_build_compact_progress_fields(
                        completed_count=completion_count,
                        succeeded_count=succeeded_count,
                        failed_count=failed_count,
                        task_total=task_total,
                        max_workers=effective_workers,
                        elapsed_seconds=perf_counter() - start_time,
                        last_artifact=artifact,
                    ),
                )

            try:
                run_output_dir, artifacts = run_benchmark(
                    config=config,
                    progress_callback=on_task_complete,
                )
            except (ValueError, FileExistsError) as exc:
                import typer
                raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
            progress.update(
                progress_task_id,
                completed=task_total,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=task_total,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifacts[-1] if artifacts else None,
                ),
            )
        console.print(f"Run output: {run_output_dir}")
        console.print(f"Tasks attempted: {len(artifacts)}")
        console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


if __name__ == "__main__":
    main()