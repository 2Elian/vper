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
        --pred_dir  G:/项目成果打包/kbbcup_dataAgent/artifacts/runs/baseline_task \
        --gold_dir  G:/项目成果打包/kbbcup_dataAgent/demo_samples/output \
        --task_dir  G:/项目成果打包/kbbcup_dataAgent/demo_samples/input
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
    for sub in sorted(task_dir.iterdir()):
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


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pred_dir: Path,
    gold_dir: Path,
    task_dir: Path,
) -> None:
    difficulty_map = load_task_difficulty(task_dir)

    # Collect all task_ids present in gold_dir
    gold_tasks = sorted(
        [sub.name for sub in gold_dir.iterdir() if sub.is_dir()]
    )

    # Per-task results
    results: Dict[str, dict] = {}

    for tid in gold_tasks:
        gold_path = gold_dir / tid / "gold.csv"
        pred_csv_path = pred_dir / tid / "prediction.csv"
        trace_path = pred_dir / tid / "trace.json"

        gold_cols = read_csv_as_columns(gold_path)
        if gold_cols is None:
            # skip if gold is missing
            continue

        pred_cols = read_csv_as_columns(pred_csv_path)
        trace = load_trace(trace_path)

        score = column_match_score(pred_cols, gold_cols)

        # Determine if the task "got a result"
        has_prediction = pred_cols is not None
        succeeded = False
        if trace is not None:
            succeeded = trace.get("succeeded", False)

        # Count steps
        n_steps = 0
        if trace is not None and "steps" in trace:
            n_steps = len(trace["steps"])

        difficulty = difficulty_map.get(tid, "unknown")

        results[tid] = {
            "task_id": tid,
            "difficulty": difficulty,
            "score": score,
            "has_prediction": has_prediction,
            "succeeded": succeeded,
            "n_steps": n_steps,
        }

    # ------ aggregate ------
    levels = ["easy", "medium", "hard", "extreme"]

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
        matched   = [t for t in lvl_tasks if results[t]["score"] == 1]
        mismatched = [t for t in lvl_tasks if results[t]["score"] == 0 and results[t]["succeeded"]]
        failed    = [t for t in lvl_tasks if not results[t]["succeeded"]]
        n_l = s["n_tasks"]
        print(f"\n--- {lvl.capitalize()} ---")
        print(f"  {'Tasks':<12}: {n_l}")
        print(f"  {'Accuracy':<12}: {s['accuracy']:.4f}  ({len(matched)}/{n_l})")
        print(f"  {'Failure Rate':<12}: {s['failure_rate']:.4f}  ({len(failed)}/{n_l})")
        print(f"  {'Avg Steps':<12}: {s['avg_steps']:.2f}")
        print(f"  Matched    ({len(matched)}): {', '.join(matched) if matched else '(none)'}")
        print(f"  Mismatched ({len(mismatched)}): {', '.join(mismatched) if mismatched else '(none)'}")
        print(f"  Failed     ({len(failed)}): {', '.join(failed) if failed else '(none)'}")

    # ------ summary table ------
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    hdr = f"{'Level':<10} {'Tasks':>6} {'Accuracy':>10} {'FailRate':>10} {'AvgSteps':>10}"
    print(hdr)
    print("-" * len(hdr))
    n_c = sum(1 for t in all_task_ids if results[t]["score"] == 1)
    n_f = sum(1 for t in all_task_ids if not results[t]["succeeded"])
    n_all = len(all_task_ids)
    print(f"{'global':<10} {n_all:>6} {n_c/n_all:>10.4f} {n_f/n_all:>10.4f} {global_stats['avg_steps']:>10.2f}")
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
        print(f"{lvl:<10} {n_l:>6} {n_c_l/n_l:>10.4f} {n_f_l/n_l:>10.4f} {s['avg_steps']:>10.2f}")

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
        default="G:/项目成果打包/kbbcup_dataAgent/artifacts/runs/baseline_task",
        help="Directory containing prediction outputs (task_*/prediction.csv + trace.json)",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        default="G:/项目成果打包/kbbcup_dataAgent/demo_samples/output",
        help="Directory containing gold answers (task_*/gold.csv)",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default="G:/项目成果打包/kbbcup_dataAgent/demo_samples/input",
        help="Directory containing task metadata (task_*/task.json with difficulty field)",
    )
    args = parser.parse_args()

    evaluate(
        pred_dir=Path(args.pred_dir),
        gold_dir=Path(args.gold_dir),
        task_dir=Path(args.task_dir),
    )


if __name__ == "__main__":
    main()
