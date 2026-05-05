"""
Evaluator for KBB Cup Data Agent baseline predictions.

Metrics:
  1. Column-Matching Accuracy (binary): score 1 iff every gold column is fully
     contained in the prediction (unordered values, column names ignored).
  2. Failure Rate: proportion of tasks that did NOT produce a result.
  3. Average Steps: mean number of agent steps (from trace.json).

All metrics are reported both globally and per-difficulty-level (easy/medium/hard/extreme).

Usage:
    python evaler.py \
        --pred_dir  G:/项目成果打包/kbbcup_dataAgent/elian/output/vper \
        --gold_dir  G:/项目成果打包/kbbcup_dataAgent/data_process/gt \
        --task_dir  G:/项目成果打包/kbbcup_dataAgent/data_process/data_split_by_difficulty \
        --difficulty easy
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_task_difficulty(task_dir: Path) -> Dict[str, str]:
    """Return {task_id: difficulty} by reading task.json from every sub-folder."""
    mapping: Dict[str, str] = {}

    # 支持两种目录结构：
    # 1. 扁平结构: task_dir/task_1/task.json
    # 2. 按难度分类: task_dir/easy/task_1/task.json
    for sub in sorted(task_dir.iterdir()):
        if sub.is_dir():
            # 检查是否是难度分类目录
            if sub.name in ['easy', 'medium', 'hard', 'extreme']:
                for task_sub in sub.iterdir():
                    if task_sub.is_dir() and task_sub.name.startswith('task_'):
                        tj = task_sub / "task.json"
                        if tj.is_file():
                            with open(tj, "r", encoding="utf-8") as f:
                                d = json.load(f)
                            mapping[d["task_id"]] = d.get("difficulty", "unknown")
            else:
                # 扁平结构
                tj = sub / "task.json"
                if tj.is_file():
                    with open(tj, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    mapping[d["task_id"]] = d.get("difficulty", "unknown")
    return mapping


def _read_text(path: Path) -> Optional[str]:
    """Read a text file, trying multiple encodings."""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def read_csv_as_columns(path: Path) -> Optional[List[List[str]]]:
    """Read a CSV file and return a list of column-value-lists.

    Each column is represented as a sorted list of stringified values
    (excluding the header).  Returns None if the file cannot be read.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        # empty file => treat as 0 columns
        return []

    # first row is header (we ignore column names during matching)
    header = rows[0]
    n_cols = len(header)
    columns: List[List[str]] = []
    for ci in range(n_cols):
        col_vals = [str(rows[ri][ci]).strip() for ri in range(1, len(rows))
                    if ci < len(rows[ri])]
        columns.append(sorted(col_vals))
    return columns


def column_match_score(
        pred_cols: Optional[List[List[str]]],
        gold_cols: List[List[str]],
) -> int:
    """Binary column-matching score.

    Returns 1 iff every gold column (as an unordered sorted value-vector)
    has an exact match among the predicted columns.  Extra predicted columns
    are allowed.
    """
    if pred_cols is None:
        return 0

    # Edge case: gold has zero columns => always correct
    if len(gold_cols) == 0:
        return 1

    # For each gold column, check if it appears in pred columns
    remaining_pred = list(pred_cols)
    for gc in gold_cols:
        found = False
        for pi, pc in enumerate(remaining_pred):
            if gc == pc:
                remaining_pred.pop(pi)
                found = True
                break
        if not found:
            return 0
    return 1


def load_trace(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    text = _read_text(path)
    if text is None or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def get_gold_path(gold_dir: Path, task_id: str, difficulty: str) -> Path:
    """Get gold CSV path, supporting both flat and difficulty-sorted structures."""
    # 尝试多种可能的路径
    possible_paths = [
        gold_dir / task_id / "gold.csv",  # 扁平: gt/task_1/gold.csv
        gold_dir / difficulty / task_id / "gold.csv",  # 分类: gt/easy/task_1/gold.csv
        gold_dir / difficulty / f"{task_id}.csv",  # 单文件: gt/easy/task_1.csv
        gold_dir / f"{task_id}.csv",  # 扁平单文件: gt/task_1.csv
    ]
    for path in possible_paths:
        if path.is_file():
            return path
    return gold_dir / task_id / "gold.csv"  # 返回默认路径


def get_pred_paths(pred_dir: Path, task_id: str, difficulty: str) -> Tuple[Path, Path]:
    """Get prediction CSV and trace paths."""
    possible_paths = [
        # 扁平结构
        (pred_dir / task_id / "prediction.csv", pred_dir / task_id / "trace.json"),
        (pred_dir / task_id / "data.csv", pred_dir / task_id / "trace.json"),
        (pred_dir / task_id / "result.csv", pred_dir / task_id / "trace.json"),

        # 分类结构
        (pred_dir / difficulty / task_id / "prediction.csv", pred_dir / difficulty / task_id / "trace.json"),
        (pred_dir / difficulty / task_id / "data.csv", pred_dir / difficulty / task_id / "trace.json"),
        (pred_dir / difficulty / task_id / "result.csv", pred_dir / difficulty / task_id / "trace.json"),

        # output 子目录
        (pred_dir / "output" / task_id / "prediction.csv", pred_dir / "output" / task_id / "trace.json"),
        (pred_dir / "output" / task_id / "data.csv", pred_dir / "output" / task_id / "trace.json"),
        (pred_dir / task_id / "output" / "prediction.csv", pred_dir / task_id / "output" / "trace.json"),
        (pred_dir / task_id / "output" / "data.csv", pred_dir / task_id / "output" / "trace.json"),
    ]

    for pred_path, trace_path in possible_paths:
        if pred_path.exists():
            return pred_path, trace_path

    return pred_dir / task_id / "data.csv", pred_dir / task_id / "trace.json"


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------

def evaluate(
        pred_dir: Path,
        gold_dir: Path,
        task_dir: Path,
        difficulty_filter: Optional[str] = None,
) -> None:
    """
    Evaluate predictions.

    Args:
        pred_dir: Directory containing prediction outputs
        gold_dir: Directory containing gold answers
        task_dir: Directory containing task metadata
        difficulty_filter: Only evaluate tasks of this difficulty (e.g., 'easy')
    """
    difficulty_map = load_task_difficulty(task_dir)

    # Collect all task_ids
    all_tasks = set()

    # 从 gold_dir 收集任务
    for sub in gold_dir.iterdir():
        if sub.is_dir():
            if sub.name in ['easy', 'medium', 'hard', 'extreme']:
                for task_sub in sub.iterdir():
                    if task_sub.is_dir() and task_sub.name.startswith('task_'):
                        all_tasks.add(task_sub.name)
            elif sub.name.startswith('task_'):
                all_tasks.add(sub.name)

    # 也从 pred_dir 收集
    for sub in pred_dir.iterdir():
        if sub.is_dir():
            if sub.name in ['easy', 'medium', 'hard', 'extreme']:
                for task_sub in sub.iterdir():
                    if task_sub.is_dir() and task_sub.name.startswith('task_'):
                        all_tasks.add(task_sub.name)
            elif sub.name.startswith('task_'):
                all_tasks.add(sub.name)

    gold_tasks = sorted(all_tasks)

    # 按难度筛选
    if difficulty_filter:
        gold_tasks = [t for t in gold_tasks if difficulty_map.get(t, 'unknown') == difficulty_filter]
        print(f"Filtering to difficulty: {difficulty_filter}")
        print(f"Found {len(gold_tasks)} tasks with this difficulty")

    # Per-task results
    results: Dict[str, dict] = {}

    for tid in gold_tasks:
        difficulty = difficulty_map.get(tid, "unknown")

        # 获取 gold 路径
        gold_path = get_gold_path(gold_dir, tid, difficulty)
        gold_cols = read_csv_as_columns(gold_path)
        if gold_cols is None:
            print(f"Warning: No gold.csv found for {tid}, skipping...")
            continue

        # 获取 pred 路径
        pred_csv_path, trace_path = get_pred_paths(pred_dir, tid, difficulty)
        pred_cols = read_csv_as_columns(pred_csv_path)
        trace = load_trace(trace_path)

        score = column_match_score(pred_cols, gold_cols)

        # Determine if the task "got a result"
        has_prediction = pred_cols is not None

        # 处理 trace.json 缺失的情况
        if trace is not None:
            succeeded = trace.get("succeeded", False)
            n_steps = len(trace.get("steps", [])) if "steps" in trace else 0
        else:
            # 没有 trace.json 时：有 prediction 就算成功，steps 计为 1
            succeeded = has_prediction
            n_steps = 1 if has_prediction else 0
            if has_prediction:
                print(f"  Note: {tid} has no trace.json, assuming success based on data.csv")

        results[tid] = {
            "task_id": tid,
            "difficulty": difficulty,
            "score": score,
            "has_prediction": has_prediction,
            "succeeded": succeeded,
            "n_steps": n_steps,
        }

    if not results:
        print("No tasks to evaluate!")
        return

    # ------ aggregate ------
    levels = ["easy", "medium", "hard", "extreme"]
    if difficulty_filter:
        levels = [difficulty_filter]

    def aggregate(task_ids: List[str], label: str) -> dict:
        scores = [results[t]["score"] for t in task_ids if t in results]
        has_pred = [results[t]["has_prediction"] for t in task_ids if t in results]
        succ = [results[t]["succeeded"] for t in task_ids if t in results]
        steps = [results[t]["n_steps"] for t in task_ids if t in results]
        n = len(task_ids)
        if n == 0:
            return {
                "label": label,
                "n_tasks": 0,
                "accuracy": float("nan"),
                "failure_rate": float("nan"),
                "avg_steps": float("nan"),
            }
        return {
            "label": label,
            "n_tasks": n,
            "accuracy": sum(scores) / n,
            "failure_rate": 1.0 - sum(succ) / n,
            "avg_steps": sum(steps) / n,
        }

    # global
    all_task_ids = list(results.keys())
    global_stats = aggregate(all_task_ids, "global")

    # per-difficulty
    per_level = {}
    for lvl in levels:
        lvl_ids = [t for t in all_task_ids if results[t]["difficulty"] == lvl]
        per_level[lvl] = aggregate(lvl_ids, lvl)

    # ------ pretty print ------
    print("=" * 72)
    print("EVALUATION RESULTS")
    if difficulty_filter:
        print(f"Difficulty: {difficulty_filter.upper()}")
    print("=" * 72)

    # Global
    print("\n--- Global ---")
    n_correct = sum(1 for t in all_task_ids if results[t]["score"] == 1)
    n_failed = sum(1 for t in all_task_ids if not results[t]["succeeded"])
    print(f"  {'Tasks':<12}: {len(all_task_ids)}")
    print(f"  {'Accuracy':<12}: {global_stats['accuracy']:.4f}  ({n_correct}/{len(all_task_ids)})")
    print(f"  {'Failure Rate':<12}: {global_stats['failure_rate']:.4f}  ({n_failed}/{len(all_task_ids)})")
    print(f"  {'Avg Steps':<12}: {global_stats['avg_steps']:.2f}")

    # Per difficulty
    for lvl in levels:
        s = per_level[lvl]
        if s["n_tasks"] == 0:
            continue
        lvl_tasks = sorted(
            [t for t in all_task_ids if results[t]["difficulty"] == lvl],
            key=lambda x: int(x.split("_")[1]),
        )
        matched = [t for t in lvl_tasks if results[t]["score"] == 1]
        mismatched = [t for t in lvl_tasks if results[t]["score"] == 0 and results[t]["succeeded"]]
        failed = [t for t in lvl_tasks if not results[t]["succeeded"]]
        n_l = s["n_tasks"]
        print(f"\n--- {lvl.capitalize()} ---")
        print(f"  {'Tasks':<12}: {n_l}")
        print(f"  {'Accuracy':<12}: {s['accuracy']:.4f}  ({len(matched)}/{n_l})")
        print(f"  {'Failure Rate':<12}: {s['failure_rate']:.4f}  ({len(failed)}/{n_l})")
        print(f"  {'Avg Steps':<12}: {s['avg_steps']:.2f}")
        if matched:
            print(f"  Matched    ({len(matched)}): {', '.join(matched)}")
        if mismatched:
            print(f"  Mismatched ({len(mismatched)}): {', '.join(mismatched)}")
        if failed:
            print(f"  Failed     ({len(failed)}): {', '.join(failed)}")

    # ------ summary table ------
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    hdr = f"{'Level':<10} {'Tasks':>6} {'Accuracy':>10} {'FailRate':>10} {'AvgSteps':>10}"
    print(hdr)
    print("-" * len(hdr))
    if not difficulty_filter:
        n_c = sum(1 for t in all_task_ids if results[t]["score"] == 1)
        n_f = sum(1 for t in all_task_ids if not results[t]["succeeded"])
        n_all = len(all_task_ids)
        print(f"{'global':<10} {n_all:>6} {n_c / n_all:>10.4f} {n_f / n_all:>10.4f} {global_stats['avg_steps']:>10.2f}")
    for lvl in levels:
        s = per_level[lvl]
        if s["n_tasks"] == 0:
            print(f"{lvl:<10} {0:>6} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        n_c_l = sum(1 for t in all_task_ids
                    if results[t]["difficulty"] == lvl and results[t]["score"] == 1)
        n_f_l = sum(1 for t in all_task_ids
                    if results[t]["difficulty"] == lvl and not results[t]["succeeded"])
        n_l = s["n_tasks"]
        print(f"{lvl:<10} {n_l:>6} {n_c_l / n_l:>10.4f} {n_f_l / n_l:>10.4f} {s['avg_steps']:>10.2f}")

    # ------ per-task detail ------
    print("\n" + "=" * 72)
    print("PER-TASK DETAIL")
    print("=" * 72)
    print(f"{'TaskID':<12} {'Difficulty':<10} {'Score':>6} {'Succeeded':>10} {'Steps':>7}")
    print("-" * 50)
    for tid in sorted(results.keys(), key=lambda x: int(x.split("_")[1])):
        r = results[tid]
        print(f"{r['task_id']:<12} {r['difficulty']:<10} {r['score']:>6} "
              f"{'YES' if r['succeeded'] else 'NO':>10} {r['n_steps']:>7}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate KBB Cup Data Agent predictions")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=r"G:\项目成果打包\kbbcup_dataAgent\elian\output\vper",
        help="Directory containing prediction outputs (task_*/prediction.csv + trace.json)",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        default=r"G:\项目成果打包\kbbcup_dataAgent\data_process\gt",
        help="Directory containing gold answers (task_*/gold.csv)",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=r"G:\项目成果打包\kbbcup_dataAgent\data_process\data_split_by_difficulty",
        help="Directory containing task metadata (task_*/task.json with difficulty field)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=['easy', 'medium', 'hard', 'extreme'],
        help="Only evaluate tasks of this difficulty (default: all)",
    )
    args = parser.parse_args()

    evaluate(
        pred_dir=Path(args.pred_dir),
        gold_dir=Path(args.gold_dir),
        task_dir=Path(args.task_dir),
        difficulty_filter=args.difficulty,
    )


if __name__ == "__main__":
    main()