import os
import json
import sys
import asyncio
from tqdm import tqdm
import csv
from pathlib import Path
from typing import Any, Dict, Optional
from openjudge.models.openai_chat_model import OpenAIChatModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from vper.utils import get_logger
from vper.llms import OpenAIClient, Tokenizer
from vper.agents import (
    PlannerAgent,
    ExecutorAgent,
    ReplannerAgent,
    ValidatorAgent,
)
from vper.orchestration.workflow import (
    PlanExecuteReplanWorkflow,
    WorkflowConfig,
)
from vper.tools.registry import create_tool_registry
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

logger = get_logger(name="run")

def _ensure_path():
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
5
_ensure_path()

def create_model_adapter(config: Dict[str, Any]):
    tokenizer_instance = Tokenizer(
        model_name=config.get("model", "gpt-4.1-mini")
    )
    model_instance = OpenAIClient(
        model_name=config.get("model", "gpt-4.1-mini"),
        base_url=config.get("base_url", "https://api.openai.com/v1"),
        api_key=config.get("api_key", ""),
        temperature=config.get("temperature", 0.0),
        tokenizer=tokenizer_instance
    )
    slow_model = OpenAIChatModel(
        model=config.get("slow_model", "deepseek-r1"),
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )
    return tokenizer_instance, model_instance, slow_model

async def run_task(task_id: str,question: str, context_dir: Path, difficulty: str = "", model_config: Optional[Dict[str, Any]] = None, dag_enabled: bool = True, max_iterations: int = 20) -> Dict[str, Any]:
    if model_config:
        tokenizer, model, slow_model = create_model_adapter(model_config)
    else:
        raise ValueError("model_config is not required")
    # 创建工具注册表
    tools = create_tool_registry(context_dir)

    # 创建 Agent
    planner = PlannerAgent(model=model)
    executor = ExecutorAgent(model=model, tools=tools)
    replanner = ReplannerAgent(model=model)
    enable_validation = True
    validator = ValidatorAgent(
        model=model,
        slow_model=slow_model,
        num_trajectories=5, # TODO to config
        validation_threshold=0.7, # TODO to config
    ) if enable_validation else None # TODO to config

    # 创建工作流
    workflow_config = WorkflowConfig(
        max_iterations=max_iterations,
        enable_dag=dag_enabled,
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
        result = await workflow.run(
            task_id=task_id,
            question=question,
            context_dir=context_dir,
            difficulty=difficulty,
        )

    return result.to_dict()

async def run_from_baseline_config(config_path: str, output_name: str = "") -> None:
    import yaml
    from pathlib import Path

    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_name = output_name
    ourput_dir = os.path.join(r"G:\项目成果打包\kbbcup_dataAgent\elian\output", output_name)
    Path(ourput_dir).mkdir(parents=True, exist_ok=True)
    model_config = {
        "model": payload.get("agent", {}).get("model", "gpt-4.1-mini"),
        "slow_model": payload.get("agent", {}).get("slow_model", "deepseek-r1-0528"),
        "base_url": payload.get("agent", {}).get("api_base", "https://api.openai.com/v1"),
        "api_key": payload.get("agent", {}).get("api_key", ""),
        "temperature": float(payload.get("agent", {}).get("temperature", 0.0)),
    }

    dataset_root = Path(payload.get("dataset", {}).get("root_path", ""))
    if not payload.get("run", {}).get("test_single_task", True):
        logger.info("开始批量处理")
        list_tasks = os.listdir(dataset_root)
        for task in tqdm(list_tasks):
            context_dir = dataset_root / task / "context"
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
                print(f"No question found for task {task}")
                return
            logger.info("🔍" * 200)
            logger.info(f"Running task: {task}")
            logger.info(f"Question: {question}")
            logger.info(f"Difficulty: {difficulty}")
            logger.info(f"Context: {context_dir}")

            result = await run_task(
                task_id=task,
                question=question,
                context_dir=context_dir,
                difficulty=difficulty,
                model_config=model_config,
                dag_enabled=False,
            )
            # output
            task_output_path = os.path.join(ourput_dir, task)
            os.makedirs(task_output_path, exist_ok=True)
            csv_file_path = os.path.join(task_output_path, "prediction.csv")
            if result.get("success", False):
                answer = result.get("answer", {})
                if "columns" in answer:
                    columns = answer.get("columns", [])
                    rows = answer.get("rows", [])
                elif "submit" in answer:
                    submit = answer["submit"]
                    columns = submit.get("columns", [])
                    rows = submit.get("rows", [])
                else:
                    columns, rows = [], []

                if columns or rows:
                    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(columns)
                        writer.writerows(rows)
                    logger.info(f"CSV created: {csv_file_path}, rows: {len(rows)}")
                else:
                    logger.warning(f"Task {task} invalid data: columns={len(columns)}, rows={len(rows)}")
            else:
                logger.warning(f"Task {task} result success is False")
            json_file_path = os.path.join(task_output_path, "trace.json")
            with open(json_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info(f"Result JSON created: {json_file_path}")
    else:
        run_task_id = payload.get("run", {}).get("task_id", "task_19")
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
        logger.info("🔍" * 200)
        logger.info(f"Running task: {run_task_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Difficulty: {difficulty}")
        logger.info(f"Context: {context_dir}")
        logger.info(f"DAG enabled: True")

        result = await run_task(
            task_id=run_task_id,
            question=question,
            context_dir=context_dir,
            difficulty=difficulty,
            model_config=model_config,
            dag_enabled=False,
        )
        # output
        task_output_path = os.path.join(ourput_dir, run_task_id)
        os.makedirs(task_output_path, exist_ok=True)
        csv_file_path = os.path.join(task_output_path, "prediction.csv")
        if result.get("success", False):
            answer = result.get("answer", {})
            if "columns" in answer:
                columns = answer.get("columns", [])
                rows = answer.get("rows", [])
            elif "submit" in answer:
                submit = answer["submit"]
                columns = submit.get("columns", [])
                rows = submit.get("rows", [])
            else:
                columns, rows = [], []

            if columns or rows:
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerows(rows)
                logger.info(f"CSV created: {csv_file_path}, rows: {len(rows)}")
            else:
                logger.warning(f"Task {run_task_id} invalid data: columns={len(columns)}, rows={len(rows)}")
        else:
            logger.warning(f"Task {run_task_id} result success is False")
        json_file_path = os.path.join(task_output_path, "trace.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
        logger.info(f"Result JSON created: {json_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Elian Data Agent Runner")
    parser.add_argument("--config", type=str, help="Path to baseline config YAML")
    parser.add_argument("--name", type=str, default="vper-simple", help="agent name")
    parser.add_argument("--context-dir", type=str, help="Context directory")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--no-dag", action="store_true", help="Disable DAG parallelism")
    args = parser.parse_args()
    # asyncio.run(run_from_baseline_config(args.config, args.task_id))
    asyncio.run(run_from_baseline_config(args.config, args.name))