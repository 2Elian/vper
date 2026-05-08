#!/usr/bin/env python3
"""DABench 单任务调试脚本。

用法:
    uv run python3 debug_task.py task_11                # 跑单个任务
    uv run python3 debug_task.py task_11 -s 30          # 指定 max_steps
    uv run python3 debug_task.py task_11 --compare      # 跑完自动对比 gold
    uv run python3 debug_task.py task_25 task_75        # 跑多个任务
    uv run python3 debug_task.py all --compare          # 跑全部50个 + 对比

配置: 直接修改下面 MODEL / API_BASE / DATA / GOLD 变量。
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

# 加 src 目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.plan_mode import PlanAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.tools.registry import create_default_tool_registry
from data_agent_baseline.agents.runtime import AgentRunResult

# ========== 配置 ==========
MODEL_PATH = "/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3_6-35b-a3b"
API_BASE = "http://172.16.107.15:8000/v1"
API_KEY = "EMPTY"
DATA_DIR = "/data1/nuist_llm/TrainLLM/kddCup/vper/data/input"
GOLD_DIR = "/data1/nuist_llm/TrainLLM/kddCup/vper/data/output"
DEFAULT_MAX_STEPS = 30
# =========================

# 关闭 HTTP 日志，只保留 agent 日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("debug_task")


def compute_score(pred_csv: str, gold_csv: str) -> float:
    """简化的列签名匹配评分 (同官方逻辑)。"""
    from decimal import Decimal, ROUND_HALF_UP
    from collections import Counter

    def norm(val):
        if val is None: return ""
        v = str(val).strip()
        if v.lower() in ("", "null", "none", "nan", "nat", "<na>"): return ""
        try:
            d = Decimal(v)
            return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        except:
            return v

    with open(pred_csv) as f: pred = list(csv.reader(f))
    with open(gold_csv) as f: gold = list(csv.reader(f))
    if not gold or not pred: return 0.0

    pdata = pred[1:]  # skip header
    gdata = gold[1:]

    if not gdata: return 1.0 if not pdata else 0.9
    if not pdata: return 0.0

    psigs = [tuple(sorted(norm(row[i] if i < len(row) else "") for row in pdata)) for i in range(len(pred[0]))]
    gsigs = [tuple(sorted(norm(row[i] if i < len(row) else "") for row in gdata)) for i in range(len(gold[0]))]

    gc = Counter(gsigs)
    pc = Counter(psigs)
    matched = sum(min(pc[s], gc[s]) for s in pc if s in gc)
    extra = len(psigs) - matched
    recall = matched / len(gsigs)
    penalty = 0.1 * (extra / len(psigs)) if psigs else 0
    return max(0, recall - penalty)


def run_single(task_id: str, max_steps: int, compare: bool) -> dict:
    """跑单个任务，返回结果字典。"""
    model = OpenAIModelAdapter(model=MODEL_PATH, api_base=API_BASE, api_key=API_KEY, temperature=0.0)
    tools = create_default_tool_registry()
    dataset = DABenchPublicDataset(Path(DATA_DIR))
    task = dataset.get_task(task_id)

    print(f"\n{'='*60}")
    print(f"📋 {task_id}: {task.question}")
    print(f"{'='*60}")

    agent = PlanAgent(model=model, tools=tools, config=ReActAgentConfig(max_steps=max_steps))
    result = agent.run(task)

    print(f"\n✅ 成功: {result.succeeded}")
    print(f"❌ 失败原因: {result.failure_reason}")
    print(f"📊 步数: {len(result.steps)}")

    if result.answer:
        print(f"📋 列: {result.answer.columns}")
        print(f"📋 行数: {len(result.answer.rows)}")
        for i, row in enumerate(result.answer.rows[:10]):
            print(f"  [{i}] {row}")
        if len(result.answer.rows) > 10:
            print(f"  ... ({len(result.answer.rows) - 10} more)")

    # 显示每步摘要
    print(f"\n--- 步骤摘要 ---")
    for step in result.steps:
        act = step.action
        thought = step.thought[:120].replace("\n", " ")
        obs_preview = ""
        if isinstance(step.observation, dict):
            if "content" in step.observation:
                c = str(step.observation["content"])
                obs_preview = c[:80].replace("\n", " ")
            elif "error" in step.observation:
                obs_preview = f"ERROR: {str(step.observation['error'])[:80]}"
        print(f"  Step{step.step_index:2d} | {act:<25s} | {thought[:100]}")
        if obs_preview:
            print(f"         {'' :<27s} | → {obs_preview}")

    # 对比 gold
    if compare and result.answer:
        gold_path = Path(GOLD_DIR) / task_id / "gold.csv"
        if gold_path.exists():
            with open(gold_path) as f:
                gold_rows = list(csv.reader(f))
            print(f"\n--- Gold ({len(gold_rows)-1} rows) ---")
            for row in gold_rows[:5]:
                print(f"  {row}")

            # 计算得分
            from pathlib import Path as P
            tmp_pred = P(f"/tmp/dabench_debug_{task_id}.csv")
            with open(tmp_pred, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(result.answer.columns)
                for row in result.answer.rows:
                    w.writerow(row)
            score = compute_score(str(tmp_pred), str(gold_path))
            tmp_pred.unlink()
            print(f"\n📊 得分: {score:.4f}")
        else:
            print(f"\n⚠️  Gold 文件不存在: {gold_path}")

    return {"task_id": task_id, "succeeded": result.succeeded, "steps": len(result.steps)}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    compare = "--compare" in args
    args = [a for a in args if a != "--compare"]

    max_steps = DEFAULT_MAX_STEPS
    if "-s" in args:
        idx = args.index("-s")
        max_steps = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    task_ids = args
    if not task_ids:
        print("请指定 task_id，如: task_11")
        sys.exit(1)

    if "all" in task_ids:
        task_ids = sorted(os.listdir(GOLD_DIR))

    total = len(task_ids)
    ok = 0
    for i, tid in enumerate(task_ids):
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{total}]")
        try:
            r = run_single(tid, max_steps, compare)
            if r["succeeded"]:
                ok += 1
        except Exception as e:
            print(f"\n💥 {tid} 崩溃: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"总计: {ok}/{total} 成功")


if __name__ == "__main__":
    main()
