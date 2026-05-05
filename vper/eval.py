"""
Evaluator for KBB Cup Data Agent baseline predictions.

Metrics:
  - Column-Matching Score (0-1): recall - λ * (extra_cols/pred_cols), min 0
  - Failure Rate: proportion of tasks that did NOT produce a result.
  - Average Steps: mean number of agent steps (from trace.json).

Only evaluates a single difficulty level at a time.

Usage:
    python evaler.py \
        --pred_dir  G:/项目成果打包/kbbcup_dataAgent/artifacts/runs/baseline_task \
        --gold_dir  G:/项目成果打包/kbbcup_dataAgent/demo_samples/output \
        --task_dir  G:/项目成果打包/kbbcup_dataAgent/demo_samples/input \
        --difficulty easy \
        --lambda 0.5
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter


# ---------------------------------------------------------------------------
# Value normalization (Section 6.5)
# ---------------------------------------------------------------------------

def normalize_null(value: str) -> str:
    """Normalize null values to empty string."""
    if value is None:
        return ""
    v = str(value).strip().lower()
    null_patterns = ["", "null", "none", "nan", "nat", "<na>", "na"]
    if v in null_patterns:
        return ""
    return str(value).strip()


def normalize_number(value: str) -> str:
    """
    Parse number and round to 2 decimal places (ROUND_HALF_UP).
    Returns formatted string with 2 decimal places.
    """
    v = normalize_null(value)
    if v == "":
        return ""
    try:
        # Remove commas and whitespace
        cleaned = re.sub(r'[,\s]', '', v)
        d = Decimal(cleaned)
        # Round to 2 decimal places with ROUND_HALF_UP
        rounded = d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return f"{rounded:.2f}"
    except Exception:
        return v  # Not a number, return as-is


def normalize_datetime(value: str) -> str:
    """
    Normalize datetime to ISO 8601 format.
    With timezone: convert to UTC and append Z.
    Without timezone: keep original ISO format.
    Date only: YYYY-MM-DD.
    """
    v = normalize_null(value)
    if v == "":
        return ""

    # Try to parse as ISO format
    # Supported formats: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, with/without timezone
    v_clean = v.strip()

    # Date only pattern: YYYY-MM-DD
    date_pattern = r'^(\d{4})-(\d{1,2})-(\d{1,2})$'
    match = re.match(date_pattern, v_clean)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # Datetime patterns
    # Try to parse with common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(v_clean, fmt)
            # If timezone-aware, convert to UTC and use Z
            if dt.tzinfo is not None:
                dt_utc = dt.astimezone(timezone.utc)
                return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue

    # Try to handle "2024-3-1" -> "2024-03-01"
    try:
        parts = v_clean.split('-')
        if len(parts) == 3:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2].split()[0])  # Handle date only
            return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        pass

    return v  # Return as-is if cannot parse


# Simple timezone UTC for normalization
from datetime import timezone


def normalize_string(value: str) -> str:
    """Remove leading/trailing spaces and \\r\\n."""
    if value is None:
        return ""
    v = str(value).strip()
    v = v.replace('\r', '').replace('\n', '')
    return v


def normalize_cell(value: str, col_hint: str = "") -> str:
    """
    Apply all normalization rules to a cell value.

    Args:
        value: The cell value to normalize
        col_hint: Optional hint about column type (e.g., "date", "number", "name")
    """
    if value is None:
        return ""

    # Step 1: Strip whitespace
    v = normalize_string(value)

    # Step 2: Normalize null
    v = normalize_null(v)
    if v == "":
        return ""

    # Step 3: Determine type and normalize accordingly
    # Try number first
    num_result = normalize_number(v)
    if num_result != v and num_result != "":
        # Successfully parsed as number
        return num_result

    # Try datetime
    dt_result = normalize_datetime(v)
    if dt_result != v and dt_result != "":
        # Successfully parsed as datetime
        return dt_result

    # Default: return normalized string
    return v


def build_column_signature(col_values: List[str]) -> Tuple[str, ...]:
    """
    Build a column signature: sorted list of normalized values.
    Used for comparing columns regardless of order.
    """
    normalized = [normalize_cell(v) for v in col_values]
    # Sort for order-independent comparison
    return tuple(sorted(normalized))


def read_csv_as_column_signatures(path: Path) -> Optional[List[Tuple[str, ...]]]:
    """
    Read a CSV file and return list of column signatures.

    Each column is represented as a tuple of sorted normalized values
    (excluding the header row).
    Returns None if the file cannot be read.
    """
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return None

    if not rows or len(rows) < 2:
        # Empty file or only header -> no data columns
        return []

    # First row is header (ignored for matching)
    n_cols = len(rows[0])
    column_signatures: List[Tuple[str, ...]] = []

    for ci in range(n_cols):
        col_vals = []
        for ri in range(1, len(rows)):
            if ci < len(rows[ri]):
                val = rows[ri][ci].strip() if rows[ri][ci] else ""
                col_vals.append(val)
            else:
                col_vals.append("")
        signature = build_column_signature(col_vals)
        column_signatures.append(signature)

    return column_signatures


# ---------------------------------------------------------------------------
# Scoring (Section 6.3)
# ---------------------------------------------------------------------------

def compute_column_match_score(
        pred_signatures: Optional[List[Tuple[str, ...]]],
        gold_signatures: List[Tuple[str, ...]],
        lambda_penalty: float = 0.5,
) -> float:
    """
    Compute the column matching score.

    Score = recall - λ * (extra_cols / pred_cols)
    Score is capped at 0 (minimum).

    Args:
        pred_signatures: List of column signatures from prediction (None if file missing)
        gold_signatures: List of column signatures from gold
        lambda_penalty: Weight for extra column penalty (default 0.5)

    Returns:
        Score between 0 and 1
    """
    # 6.3: File missing -> score 0
    if pred_signatures is None:
        return 0.0

    n_gold = len(gold_signatures)
    n_pred = len(pred_signatures)

    # Edge case: gold has zero columns
    if n_gold == 0:
        # If gold empty, prediction should also be empty for perfect score
        if n_pred == 0:
            return 1.0
        else:
            # All prediction columns are "extra"
            return max(0.0, 0.0 - lambda_penalty * (n_pred / n_pred))  # = -λ, capped to 0

    # Count matching columns (considering duplicates via multiplicity)
    # Use Counter to handle duplicate column signatures
    pred_counter = Counter(pred_signatures)
    gold_counter = Counter(gold_signatures)

    matched_cols = 0
    for sig, gold_count in gold_counter.items():
        pred_count = pred_counter.get(sig, 0)
        matched_cols += min(gold_count, pred_count)

    # Calculate metrics
    recall = matched_cols / n_gold

    extra_cols = n_pred - matched_cols
    if n_pred > 0:
        penalty = lambda_penalty * (extra_cols / n_pred)
    else:
        penalty = 0.0

    score = recall - penalty

    # Score下限为0 (Section 6.3)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_task_folders(task_dir: Path, difficulty: str) -> List[str]:
    """
    Find all task folders for a specific difficulty level.
    Task folders are named like task_1, task_2, etc.
    """
    task_folders = []
    for sub in sorted(task_dir.iterdir()):
        if not sub.is_dir():
            continue
        tj = sub / "task.json"
        if tj.is_file():
            with open(tj, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("difficulty", "").lower() == difficulty.lower():
                task_folders.append(sub.name)
    return sorted(task_folders)


def load_trace(path: Path) -> Optional[dict]:
    """Load trace.json file."""
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
        pred_dir: Path,
        gold_dir: Path,
        task_dir: Path,
        difficulty: str,
        lambda_penalty: float = 0.5,
) -> None:
    """
    Evaluate predictions for a single difficulty level.

    Args:
        pred_dir: Directory containing prediction outputs
        gold_dir: Directory containing gold answers
        task_dir: Directory containing task metadata
        difficulty: One of "easy", "medium", "hard", "extreme"
        lambda_penalty: Penalty weight for extra columns
    """
    # Find all tasks for this difficulty
    task_ids = find_task_folders(task_dir, difficulty)

    if not task_ids:
        print(f"Error: No tasks found with difficulty '{difficulty}'")
        print("Available difficulty levels: easy, medium, hard, extreme")
        return

    print(f"\nEvaluating {len(task_ids)} tasks with difficulty: {difficulty.upper()}")
    print(f"Lambda penalty: {lambda_penalty}")
    print("=" * 72)

    # Per-task results
    results: Dict[str, dict] = {}

    for tid in task_ids:
        gold_path = gold_dir / tid / "gold.csv"
        pred_csv_path = pred_dir / tid / "prediction.csv"
        trace_path = pred_dir / tid / "trace.json"

        # Read column signatures
        gold_signatures = read_csv_as_column_signatures(gold_path)
        if gold_signatures is None:
            print(f"Warning: Cannot read gold file for {tid}, skipping")
            continue

        pred_signatures = read_csv_as_column_signatures(pred_csv_path)
        trace = load_trace(trace_path)

        # Compute score
        score = compute_column_match_score(pred_signatures, gold_signatures, lambda_penalty)

        # Determine if task produced a result
        has_prediction = pred_signatures is not None
        succeeded = True

        # Count steps
        n_steps = 0
        if trace is not None and "steps" in trace:
            n_steps = len(trace["steps"])

        # For detailed reporting, also track matching info
        n_gold = len(gold_signatures)
        n_pred = len(pred_signatures) if pred_signatures is not None else 0

        # Calculate matching count for reporting
        if pred_signatures is not None:
            pred_counter = Counter(pred_signatures)
            gold_counter = Counter(gold_signatures)
            matched = 0
            for sig, gold_count in gold_counter.items():
                matched += min(gold_count, pred_counter.get(sig, 0))
        else:
            matched = 0

        results[tid] = {
            "task_id": tid,
            "difficulty": difficulty,
            "score": score,
            "has_prediction": has_prediction,
            "succeeded": succeeded,
            "n_steps": n_steps,
            "n_gold": n_gold,
            "n_pred": n_pred,
            "matched_cols": matched,
        }

    if not results:
        print(f"Error: No valid tasks found for difficulty '{difficulty}'")
        return

    # ----- Aggregate statistics -----
    n_tasks = len(results)
    scores = [r["score"] for r in results.values()]
    succeeded = [r["succeeded"] for r in results.values()]
    steps = [r["n_steps"] for r in results.values()]

    n_failed = sum(1 for s in succeeded if not s)
    failure_rate = n_failed / n_tasks
    avg_score = sum(scores) / n_tasks
    avg_steps = sum(steps) / n_tasks

    # ----- Print results -----
    print("\n" + "=" * 72)
    print(f"EVALUATION RESULTS - {difficulty.upper()}")
    print("=" * 72)

    print(f"\n--- Summary ---")
    print(f"  {'Tasks':<15}: {n_tasks}")
    print(f"  {'Avg Score':<15}: {avg_score:.4f}")
    print(f"  {'Failure Rate':<15}: {failure_rate:.4f} ({n_failed}/{n_tasks})")
    print(f"  {'Avg Steps':<15}: {avg_steps:.2f}")

    # Score distribution
    print(f"\n--- Score Distribution ---")
    score_bins = [(0.9, 1.0, "0.9-1.0"), (0.7, 0.9, "0.7-0.9"),
                  (0.5, 0.7, "0.5-0.7"), (0.3, 0.5, "0.3-0.5"),
                  (0.0, 0.3, "0.0-0.3")]
    for low, high, label in score_bins:
        count = sum(1 for s in scores if low <= s < high or (high == 1.0 and s == 1.0))
        pct = count / n_tasks * 100
        print(f"  {label:>8}: {count:>3} tasks ({pct:5.1f}%)")

    # Failed tasks
    failed_tasks = [tid for tid, r in results.items() if not r["succeeded"]]
    if failed_tasks:
        print(f"\n--- Failed Tasks ({len(failed_tasks)}) ---")
        print(f"  {', '.join(failed_tasks)}")

    # Low score tasks (but succeeded)
    low_score_tasks = [(tid, r["score"]) for tid, r in results.items()
                       if r["succeeded"] and r["score"] < 0.5]
    if low_score_tasks:
        print(f"\n--- Low Score Tasks (<0.5) ---")
        for tid, score in sorted(low_score_tasks, key=lambda x: x[1]):
            r = results[tid]
            print(f"  {tid}: score={score:.4f} (gold={r['n_gold']}, pred={r['n_pred']}, matched={r['matched_cols']})")

    # ----- Per-task detail table -----
    print("\n" + "=" * 72)
    print("PER-TASK DETAIL")
    print("=" * 72)
    print(f"{'TaskID':<12} {'Score':>8} {'Gold':>5} {'Pred':>5} {'Match':>6} {'Succeeded':>10} {'Steps':>7}")
    print("-" * 65)

    # Sort by task ID number
    def task_num(tid: str) -> int:
        try:
            return int(tid.split("_")[1])
        except:
            return 0

    for tid in sorted(results.keys(), key=task_num):
        r = results[tid]
        print(f"{r['task_id']:<12} {r['score']:>8.4f} {r['n_gold']:>5} {r['n_pred']:>5} "
              f"{r['matched_cols']:>6} {'YES' if r['succeeded'] else 'NO':>10} {r['n_steps']:>7}")

    # Final summary line
    print("\n" + "=" * 72)
    print(f"SUMMARY: {difficulty.upper()} | "
          f"Tasks: {n_tasks} | "
          f"Avg Score: {avg_score:.4f} | "
          f"Failure Rate: {failure_rate:.4f} | "
          f"Avg Steps: {avg_steps:.2f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KBB Cup Data Agent predictions for a single difficulty level"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=r"G:\项目成果打包\kbbcup_dataAgent\elian\output\vper-simple",
        help="Directory containing prediction outputs (task_*/prediction.csv + trace.json)",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        default=r"G:\项目成果打包\kbbcup_dataAgent\demo_samples\output\easy",
        help="Directory containing gold answers (task_*/gold.csv)",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=r"G:\项目成果打包\kbbcup_dataAgent\data_process\data_split_by_difficulty\easy",
        help="Directory containing task metadata (task_*/task.json with difficulty field)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "extreme"],
        help="Difficulty level to evaluate (only one at a time)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.0,
        dest="lambda_penalty",
        help="Penalty weight for extra columns (default: 0.0)",
    )

    args = parser.parse_args()

    evaluate(
        pred_dir=Path(args.pred_dir),
        gold_dir=Path(args.gold_dir),
        task_dir=Path(args.task_dir),
        difficulty=args.difficulty,
        lambda_penalty=args.lambda_penalty,
    )


if __name__ == "__main__":
    main()