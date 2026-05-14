from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

NULL_PATTERNS = frozenset({"", "null", "none", "nan", "nat", "<na>"})

TASK_DIR_PREFIX = "task_"
SCORE_FLOOR = 0.0


def _is_null(raw: str | None) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() in NULL_PATTERNS


def normalize_value(raw: str | None) -> str:
    if raw is None:
        return ""
    s = raw.strip()
    if _is_null(s):
        return ""

    # datetime with timezone
    if "T" in s and any(indicator in s for indicator in ("+", "Z", "-")):
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            pass

    # datetime without timezone
    if "T" in s:
        try:
            dt = datetime.fromisoformat(s)
            return dt.isoformat()
        except (ValueError, TypeError):
            pass

    # date only
    try:
        d = date.fromisoformat(s)
        return d.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    # numeric
    try:
        cleaned = s.replace(",", "").replace(" ", "").replace("\xa0", "")
        d = Decimal(cleaned)
        return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, ArithmeticError):
        pass

    # string
    s = s.rstrip("\r\n").lstrip("\r\n").rstrip("\r").lstrip("\r").rstrip("\n").lstrip("\n")
    return s


def _read_csv_columns(path: Path, skip_header: bool = True) -> list[tuple[str, ...]]:
    """Read CSV and return columns as tuples.
    
    Args:
        path: Path to CSV file
        skip_header: If True, skip the first row (header) to match official 
                     evaluation which ignores column names.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        return []

    # Skip header row if requested (matches official evaluation)
    if skip_header and len(rows) > 1:
        rows = rows[1:]

    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    columns: list[list[str]] = [[] for _ in range(max_cols)]
    for row in rows:
        for col_idx in range(max_cols):
            value = row[col_idx] if col_idx < len(row) else ""
            columns[col_idx].append(value)
    return [tuple(normalize_value(v) for v in col) for col in columns]


def _build_signatures(columns: list[tuple[str, ...]]) -> Counter[tuple[str, ...]]:
    return Counter(columns)


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    matched: int
    gold_columns: int
    pred_columns: int
    recall: float
    penalty: float
    score: float

    @property
    def extra_columns(self) -> int:
        return max(self.pred_columns - self.matched, 0)


def evaluate_single_task(
    prediction_csv: Path,
    gold_csv: Path,
    *,
    penalty_lambda: float = 0.1,
) -> TaskScore:
    if not prediction_csv.exists():
        return TaskScore(
            task_id=gold_csv.parent.name,
            matched=0,
            gold_columns=0,
            pred_columns=0,
            recall=0.0,
            penalty=0.0,
            score=0.0,
        )

    # Skip header to match official evaluation (ignores column names)
    gold_cols = _read_csv_columns(gold_csv, skip_header=True)
    pred_cols = _read_csv_columns(prediction_csv, skip_header=True)

    gold_sigs = _build_signatures(gold_cols)
    pred_sigs = _build_signatures(pred_cols)

    matched = 0
    for sig, pred_count in pred_sigs.items():
        gold_count = gold_sigs.get(sig, 0)
        matched += min(pred_count, gold_count)

    gold_total = len(gold_cols)
    pred_total = len(pred_cols)

    recall = matched / gold_total if gold_total > 0 else 0.0

    extra = max(pred_total - matched, 0)
    penalty = (penalty_lambda * extra / pred_total) if pred_total > 0 else 0.0

    score = max(recall - penalty, SCORE_FLOOR)

    return TaskScore(
        task_id=gold_csv.parent.name,
        matched=matched,
        gold_columns=gold_total,
        pred_columns=pred_total,
        recall=recall,
        penalty=penalty,
        score=score,
    )


def _task_number(task_id: str) -> int:
    if not task_id.startswith(TASK_DIR_PREFIX):
        return -1
    try:
        return int(task_id.removeprefix(TASK_DIR_PREFIX))
    except ValueError:
        return -1


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    task_scores: list[TaskScore]
    penalty_lambda: float

    @property
    def overall_score(self) -> float:
        if not self.task_scores:
            return 0.0
        return sum(t.score for t in self.task_scores) / len(self.task_scores)

    @property
    def succeeded_tasks(self) -> int:
        return sum(1 for t in self.task_scores if t.score > 0.0)

    @property
    def missing_tasks(self) -> int:
        return sum(1 for t in self.task_scores if t.gold_columns == 0 and t.pred_columns == 0)

    @property
    def total_tasks(self) -> int:
        return len(self.task_scores)


def benchmark_scores(
    prediction_root: Path,
    gold_root: Path,
    *,
    penalty_lambda: float = 0.1,
) -> BenchmarkResult:
    if not gold_root.exists():
        raise FileNotFoundError(f"Gold root not found: {gold_root}")

    gold_tasks: dict[str, Path] = {}
    for task_dir in sorted(gold_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith(TASK_DIR_PREFIX):
            continue
        gold_file = task_dir / "gold.csv"
        if gold_file.exists():
            gold_tasks[task_dir.name] = gold_file

    scores: list[TaskScore] = []
    for task_id in sorted(gold_tasks, key=_task_number):
        gold_path = gold_tasks[task_id]
        pred_path = prediction_root / task_id / "prediction.csv"
        scores.append(
            evaluate_single_task(pred_path, gold_path, penalty_lambda=penalty_lambda)
        )

    return BenchmarkResult(task_scores=scores, penalty_lambda=penalty_lambda)
