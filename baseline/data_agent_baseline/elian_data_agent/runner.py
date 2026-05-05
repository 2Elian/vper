import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from elian_data_agent.utils import get_logger
from elian_data_agent.llms import OpenAIClient, Tokenizer
from elian_data_agent.agents.planner import PlannerAgent
from elian_data_agent.agents.executor import ExecutorAgent
from elian_data_agent.agents.replanner import ReplannerAgent
from elian_data_agent.agents.validator import ValidatorAgent
from elian_data_agent.orchestration.workflow import (
    PlanExecuteReplanWorkflow,
    WorkflowConfig,
)
from elian_data_agent.tools.registry import create_tool_registry

logger = get_logger(name="run")

def _ensure_path():
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

_ensure_path()

def create_model_adapter(config: Dict[str, Any]):
    tokenizer_instanece = Tokenizer(
        model_name=config.get("model", "gpt-4.1-mini")
    )
    model_instance = OpenAIClient(
        model=config.get("model", "gpt-4.1-mini"),
        api_base=config.get("api_base", "https://api.openai.com/v1"),
        api_key=config.get("api_key", ""),
        temperature=config.get("temperature", 0.0),
    )
    return tokenizer_instanece, model_instance

def run_task(task_id: str,question: str, context_dir: Path, difficulty: str = "", model_config: Optional[Dict[str, Any]] = None, dag_enabled: bool = True, max_iterations: int = 20, validation_threshold: float = 0.7, num_validation_trajectories: int = 5, enable_validation: bool = True) -> Dict[str, Any]:
    if model_config:
        tokenizer, model = create_model_adapter(model_config)
    else:
        raise ValueError("model_config is required")
    # 创建工具注册表
    tools = create_tool_registry(context_dir)

    # 创建 Agent
    planner = PlannerAgent(model=model)
    executor = ExecutorAgent(model=model, tools=tools)
    replanner = ReplannerAgent(model=model)
    validator = ValidatorAgent(
        model=model,
        model_config=model_config,
        num_trajectories=num_validation_trajectories,
        validation_threshold=validation_threshold,
    ) if enable_validation else None

    # 创建工作流
    workflow_config = WorkflowConfig(
        max_iterations=max_iterations,
        enable_dag=dag_enabled,
        enable_validation=enable_validation,
        validation_threshold=validation_threshold,
        num_validation_trajectories=num_validation_trajectories,
    )

    workflow = PlanExecuteReplanWorkflow(
        planner=planner,
        executor=executor,
        replanner=replanner,
        validator=validator,
        config=workflow_config,
    )

    # 运行
    if dag_enabled:
        result = workflow.run_with_dag(
            task_id=task_id,
            question=question,
            context_dir=context_dir,
            difficulty=difficulty,
        )
    else:
        result = workflow.run(
            task_id=task_id,
            question=question,
            context_dir=context_dir,
            difficulty=difficulty,
        )

    return result.to_dict()


def run_benchmark_task(
    task: Any,
    model: Any = None,
    model_config: Optional[Dict[str, Any]] = None,
    dag_enabled: bool = True,
    enable_validation: bool = True,
    validation_threshold: float = 0.7,
    num_validation_trajectories: int = 5,
) -> Dict[str, Any]:
    # 获取模型
    if model is None and model_config:
        model = create_model_adapter(model_config)
    elif model is None:
        raise ValueError("Either model or model_config must be provided")

    # 创建工具
    tools = create_tool_registry(task.context_dir)

    # 创建 Agent
    planner = PlannerAgent(model=model)
    executor = ExecutorAgent(model=model, tools=tools)
    replanner = ReplannerAgent(model=model)
    validator = ValidatorAgent(
        model=model,
        model_config=model_config,
        num_trajectories=num_validation_trajectories,
        validation_threshold=validation_threshold,
    ) if enable_validation else None

    # 工作流
    workflow = PlanExecuteReplanWorkflow(
        planner=planner,
        executor=executor,
        replanner=replanner,
        validator=validator,
        config=WorkflowConfig(
            enable_dag=dag_enabled,
            enable_validation=enable_validation,
            validation_threshold=validation_threshold,
            num_validation_trajectories=num_validation_trajectories,
        ),
    )

    # 运行
    result = workflow.run(
        task_id=task.task_id,
        question=task.question,
        context_dir=task.context_dir,
        difficulty=task.difficulty,
    )

    return result.to_dict()

def run_from_baseline_config(config_path: str, task_id: Optional[str] = None) -> None:
    import yaml
    from pathlib import Path

    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_config = {
        "model": payload.get("agent", {}).get("model", "gpt-4.1-mini"),
        "api_base": payload.get("agent", {}).get("api_base", "https://api.openai.com/v1"),
        "api_key": payload.get("agent", {}).get("api_key", ""),
        "temperature": float(payload.get("agent", {}).get("temperature", 0.0)),
    }

    dataset_root = Path(payload.get("dataset", {}).get("root_path", ""))
    run_task_id = task_id or payload.get("run", {}).get("task_id", "task_19")

    # 查找任务
    context_dir = dataset_root / run_task_id / "context"
    if not context_dir.exists():
        logger.error(f"Context directory not found: {context_dir}")
        return
    question = ""
    difficulty = ""
    task_file = context_dir.parent / "task.json"
    if task_file.exists():
        task_data = json.loads(task_file.read_text(encoding="utf-8"))
        question = task_data.get("question", "")
        difficulty = task_data.get("difficulty", "")

    if not question:
        print(f"No question found for task {run_task_id}")
        return
    logger.info("=" * 200)
    logger.info(f"Running task: {run_task_id}")
    logger.info(f"Question: {question}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"Context: {context_dir}")
    logger.info(f"DAG enabled: True")
    logger.info("=" * 200)

    result = run_task(
        task_id=run_task_id,
        question=question,
        context_dir=context_dir,
        difficulty=difficulty,
        model_config=model_config,
        dag_enabled=True,
    )

    logger.info("\n" + "=" * 60)
    logger.info("RESULT:")
    logger.info(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elian Data Agent Runner")
    parser.add_argument("--config", type=str, help="Path to baseline config YAML")
    parser.add_argument("--task-id", type=str, help="Task ID to run")
    parser.add_argument("--context-dir", type=str, help="Context directory")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--no-dag", action="store_true", help="Disable DAG parallelism")

    args = parser.parse_args()

    if args.config:
        run_from_baseline_config(args.config, args.task_id)
    elif args.context_dir and args.question:
        result = run_task(
            task_id=args.task_id or "test",
            question=args.question,
            context_dir=Path(args.context_dir),
            model_config={
                "model": args.model,
                "api_key": args.api_key,
                "api_base": args.api_base,
            },
            dag_enabled=not args.no_dag,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()