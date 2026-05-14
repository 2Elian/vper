"""
CodexAgent — Heuristic Learning Data Analysis (结合 hldaa + submit-2)

核心理念（来自 Learning Beyond Gradients）:
  策略 = Python 代码    学习 = 代码编辑
  记忆 = 文件系统       反馈 = 执行结果 + GOLD_SCORE

关键改进:
  - 答案不是 "跑通了" 就算成功 — 必须对 gold 评分 > threshold
  - 只压缩高评分 trial 的模板
  - 评分直接写入 trial record，驱动策略迭代
"""

from __future__ import annotations

import csv, json, logging, re, subprocess, tempfile, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.llm_extractor import LLMExtractor
from data_agent_baseline.agents.memory import HLMemory, TrialRecord
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("CodexAgent")

# ============================================================
# Prompts — template-based (reduces coding burden on model)
# ============================================================

SYSTEM_PROMPT = """You are an Expert Data Analyst. You fill in Python code templates to answer data questions.
You do NOT write free-form code — you fill in a structured template using exact file paths and column names from the provided data context.
Be precise, use exact names, and keep the code simple."""

STRATEGY_WRITER_PROMPT = """## Task: Fill in the Analysis Template

Fill in ONLY Sections 1 and 2 of the template to answer the question.

## Question: {goal}

## Data Context (EXACT file paths, schemas, column names, sample data):
{data_context}

## Memory
{memory_context}

## TEMPLATE — OUTPUT THE COMPLETE CODE INCLUDING ALL 3 SECTIONS:

```python
import json, os, sys, sqlite3, csv

# === SECTION 1: DATA LOADING (FILL IN) ===
# Use EXACT file paths and column names from the Data Context above
# Print loaded data for verification:
#   rows = list(csv.DictReader(open('csv/filename.csv')))
#   print("DEBUG loaded", len(rows), "rows; first:", rows[0] if rows else "EMPTY")

# === SECTION 2: QUERY / ANALYSIS (FILL IN) ===
# Answer the question: {goal}
# Print intermediate results

# === SECTION 3: OUTPUT (DO NOT MODIFY THIS BLOCK) ===
columns = ['col1', 'col2']  # FILL IN actual column names
rows = [['val1', 'val2']]   # FILL IN actual data rows
print(f"RESULT_JSON: {{json.dumps({{'columns': columns, 'rows': rows}})}}")
```

CRITICAL RULES:
1. Use EXACT file paths from Data Context (e.g., 'csv/data.csv', 'file:db.sqlite?mode=ro')
2. Use EXACT column names from Data Context — do NOT guess
3. Print debug info in Sections 1-2 to verify data is loaded correctly
4. DO NOT modify Section 3 structure — just fill in columns and rows
5. Keep it SIMPLE — prefer sqlite3 for DB files, csv.DictReader for CSV files
"""

FEEDBACK_ABSORBER_PROMPT = """## Task: Fix the Failing Template

Fix ONLY Sections 1 and 2 based on the execution feedback below.

## Goal: {goal}

## Execution Feedback:
{feedback}

## Current Code:
```python
{current_code}
```

## Data Context (use EXACT names from here):
{data_context}

## Memory
{memory_context}

## INSTRUCTIONS:
1. Find the specific error in the feedback (file not found? wrong column name? syntax error?)
2. Fix ONLY Section 1 (file paths, column names) or Section 2 (query logic)
3. Use EXACT paths and column names from the Data Context above
4. Print debug info to help diagnose remaining issues
5. Output the COMPLETE code (all 3 sections) in ```python ... ```
"""

HISTORY_COMPRESSOR_PROMPT = """## Task: Compress Analysis Strategies into Memory Templates

## Successful Trials
{successful_trials}

## Output Format — JSON in ```json:
[
  {{
    "name": "pattern_name",
    "description": "When to use this pattern",
    "template": "Key steps and code skeleton",
    "tags": ["tag1", "tag2"]
  }}
]
"""

# ============================================================
# Config
# ============================================================

@dataclass(frozen=True, slots=True)
class CodexAgentConfig:
    max_steps: int = 8
    code_timeout_seconds: int = 60
    trace_save_path: Path | None = None
    enable_observation: bool = True
    enable_memory: bool = True
    enable_compression: bool = True
    memory_dir: str = "artifacts/hl_memory"
    gold_dir: str = ""                  # gold 答案目录
    gold_score_threshold: float = 0.5   # 低于此分的 trial 不算 success，不压缩
    llm_extractor_max_retries: int = 3


# ============================================================
# Agent
# ============================================================

class CodexAgent:

    def __init__(self, *, model: ModelAdapter, tools: ToolRegistry | None = None,
                 config: CodexAgentConfig | None = None, **kwargs) -> None:
        self.model = model
        self.tools = tools
        self.config = config or CodexAgentConfig()
        # Support env vars for gold scoring (avoids modifying config.py)
        import os
        gold_dir = os.environ.get("GOLD_DIR", self.config.gold_dir)
        gold_threshold = float(os.environ.get("GOLD_SCORE_THRESHOLD", self.config.gold_score_threshold))
        self.config = CodexAgentConfig(
            max_steps=self.config.max_steps,
            code_timeout_seconds=self.config.code_timeout_seconds,
            trace_save_path=self.config.trace_save_path,
            enable_observation=self.config.enable_observation,
            enable_memory=self.config.enable_memory,
            enable_compression=self.config.enable_compression,
            memory_dir=self.config.memory_dir,
            gold_dir=gold_dir,
            gold_score_threshold=gold_threshold,
            llm_extractor_max_retries=self.config.llm_extractor_max_retries,
        )
        self.extractor = LLMExtractor(max_retries=self.config.llm_extractor_max_retries)
        self._memory = HLMemory(Path(self.config.memory_dir)) if self.config.enable_memory else None

    # ---- observation ----

    def _observe(self, task: PublicTask) -> str:
        lines = []
        if not self.tools:
            return self._basic_observe(task)

        # 1. File tree
        try:
            r = self.tools.execute(task, "list_context", {"max_depth": 4})
            if r.ok:
                lines.append(f"## Files:\n{json.dumps(r.content, ensure_ascii=False)[:2000]}")
        except Exception as e:
            lines.append(f"list_context: {e}")

        # 2. knowledge.md
        try:
            r = self.tools.execute(task, "read_doc", {"path": "knowledge.md", "max_chars": 6000})
            if r.ok:
                lines.append(f"## knowledge.md:\n{json.dumps(r.content, ensure_ascii=False)[:5000]}")
        except Exception:
            lines.append("## knowledge.md: not found")

        # 3. SQLite: schema + sample data
        for f in sorted(task.context_dir.iterdir()):
            if f.suffix in (".sqlite", ".db"):
                try:
                    r = self.tools.execute(task, "inspect_sqlite_schema", {"path": f.name})
                    if r.ok:
                        lines.append(f"## DB Schema ({f.name}):\n{json.dumps(r.content, ensure_ascii=False)[:4000]}")
                    # Sample first 2 tables
                    for tbl in (r.content.get("tables", []) if r.ok else [])[:2]:
                        tn = tbl.get("name", "")
                        if tn:
                            try:
                                r2 = self.tools.execute(task, "execute_context_sql",
                                    {"path": f.name, "sql": f'SELECT * FROM "{tn}" LIMIT 3', "limit": 3})
                                if r2.ok:
                                    lines.append(f"## Sample {tn}:\n{json.dumps(r2.content, ensure_ascii=False)[:1500]}")
                            except Exception:
                                pass
                except Exception as e:
                    lines.append(f"DB {f.name}: {e}")

        # 4. CSV previews
        for f in sorted(task.context_dir.iterdir()):
            if f.suffix == ".csv":
                try:
                    r = self.tools.execute(task, "read_csv", {"path": f.name, "max_rows": 5})
                    if r.ok:
                        lines.append(f"## CSV ({f.name}):\n{json.dumps(r.content, ensure_ascii=False)[:2000]}")
                except Exception as e:
                    lines.append(f"CSV {f.name}: {e}")

        # 5. JSON previews
        for f in sorted(task.context_dir.iterdir()):
            if f.suffix == ".json":
                try:
                    r = self.tools.execute(task, "read_json", {"path": f.name, "max_chars": 3000})
                    if r.ok:
                        lines.append(f"## JSON ({f.name}):\n{json.dumps(r.content, ensure_ascii=False)[:3000]}")
                except Exception as e:
                    lines.append(f"JSON {f.name}: {e}")

        obs = "\n\n".join(lines) if lines else self._basic_observe(task)
        logger.info("Observation: %d chars", len(obs))
        return obs

    def _basic_observe(self, task: PublicTask) -> str:
        code = """import os, json as _j
files = sorted(os.listdir('.'))
print("FILES:", _j.dumps(files))
for f in files:
    if f.endswith('.md'):
        try:
            with open(f) as fh: print(f"--- {f} ---"); print(fh.read()[:3000])
        except: pass
    if f.endswith(('.sqlite', '.db')):
        import sqlite3
        try:
            conn = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            cur = conn.cursor()
            tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            print(f"DB {f} tables:", [t[0] for t in tables])
            if tables:
                cur.execute(f"PRAGMA table_info('{tables[0][0]}')")
                print(f"Schema {tables[0][0]}:", cur.fetchall()[:10])
            conn.close()
        except Exception as e: print(f"DB Error {f}: {e}")
"""
        try:
            r = subprocess.run(["python", "-c", code], capture_output=True, text=True,
                              cwd=str(task.context_dir), timeout=20)
            return r.stdout[:4000]
        except Exception as e:
            return f"Observe error: {e}"

    # ---- code generation ----

    def _write_strategy(self, task: PublicTask, observation: str) -> str | None:
        mem_ctx = self._memory.format_memory_context() if self._memory else ""
        prompt = STRATEGY_WRITER_PROMPT.format(
            goal=task.question, data_context=observation[:6000],
            memory_context=mem_ctx,
        )
        msgs = [ModelMessage(role="system", content=SYSTEM_PROMPT), ModelMessage(role="user", content=prompt)]
        results, _ = self.extractor.extract_with_retry(model=self.model, messages=msgs, rule_parser=_extract_code, n=1)
        return results[0] if results else None

    def _absorb_feedback(self, task: PublicTask, code: str, feedback: str, observation: str) -> str | None:
        mem_ctx = self._memory.format_memory_context() if self._memory else ""
        prompt = FEEDBACK_ABSORBER_PROMPT.format(
            goal=task.question, current_code=code[:4000], feedback=feedback,
            data_context=observation[:3000], memory_context=mem_ctx,
        )
        msgs = [ModelMessage(role="system", content=SYSTEM_PROMPT), ModelMessage(role="user", content=prompt)]
        results, _ = self.extractor.extract_with_retry(model=self.model, messages=msgs, rule_parser=_extract_code, n=1)
        return results[0] if results else None

    def _compress_history(self) -> None:
        if not self._memory or not self.config.enable_compression:
            return
        successes = self._memory.recent_successes(limit=10)
        if len(successes) < 2:
            return
        logger.info("Compressing %d trials (scores: %s)...", len(successes),
                     [f"{t.get('score',0):.2f}" for t in successes])
        summary = "\n\n".join(
            f"## Trial: {t.get('trial_id','?')} | Q: {t.get('question','')[:100]}\n"
            f"Status: {t.get('status','?')} | GoldScore: {t.get('score',0):.3f}\n"
            f"Code snippet: {t.get('error_summary','')[:200]}"
            for t in successes
        )
        prompt = HISTORY_COMPRESSOR_PROMPT.format(successful_trials=summary)
        msgs = [ModelMessage(role="user", content=prompt)]

        def parse(r: str) -> list | None:
            m = re.search(r"```json\s*\n(.*?)\n```", r, re.DOTALL)
            if not m: m = re.search(r"```\s*\n(\[.*?\])\s*\n```", r, re.DOTALL)
            if m:
                try: return json.loads(m.group(1))
                except: pass
            return None

        results, _ = self.extractor.extract_with_retry(model=self.model, messages=msgs, rule_parser=parse, n=1)
        if results:
            for tmpl in results[0]:
                # Add gold score info to template for later filtering
                scores = [t.get("score", 0) for t in successes]
                avg_score = sum(scores) / len(scores) if scores else 0
                tmpl.setdefault("tags", [])
                tmpl["tags"].append(f"gold_avg={avg_score:.2f}")
                tmpl["tags"].append(f"gold_max={max(scores):.2f}")
                tmpl["description"] = f"[GoldScore: {avg_score:.2f}] {tmpl.get('description', '')}"
                self._memory.save_template(tmpl)
            logger.info("Compressed %d templates (avg gold score: %.2f).", len(results[0]),
                         sum(t.get("score",0) for t in successes)/max(1,len(successes)))

    # ---- execution ----

    @staticmethod
    def _execute(code: str, context_dir: Path, timeout: int) -> tuple[str, str, int]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="hl_") as f:
            f.write(code); sp = f.name
        try:
            r = subprocess.run(["python", sp], capture_output=True, text=True, cwd=str(context_dir), timeout=timeout)
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {timeout}s", -1
        except Exception as e:
            return "", str(e), -1
        finally:
            try: Path(sp).unlink()
            except: pass

    @staticmethod
    def _extract_answer(stdout: str) -> dict | None:
        m = re.search(r'RESULT_JSON:\s*(\{.*?\})\s*$', stdout, re.DOTALL | re.MULTILINE)
        if m:
            try: return json.loads(m.group(1))
            except: pass
        for m in re.finditer(r'\{[^}]*"columns"\s*:\s*\[.*?\][^}]*"rows"\s*:\s*\[.*?\][^}]*\}', stdout, re.DOTALL):
            try: return json.loads(m.group(0))
            except: continue
        return None

    @staticmethod
    def _score_against_gold(task_id: str, answer: AnswerTable, gold_dir: str) -> float:
        """用 gold.csv 或 prediction.csv 评分，返回 0.0~1.0。忽略列名，只比 value 集合。"""
        if not gold_dir:
            return 1.0
        gold_csv = Path(gold_dir) / task_id / "gold.csv"
        if not gold_csv.exists():
            gold_csv = Path(gold_dir) / task_id / "prediction.csv"  # pseudo-gold from another agent
        if not gold_csv.exists():
            logger.warning("Gold not found: %s (tried gold.csv and prediction.csv)", task_id)
            return 1.0
        from collections import Counter
        gold_cols = _read_csv_columns(gold_csv)
        pred_cols = [tuple(str(v) for v in row) for row in answer.rows]
        if not gold_cols:
            return 0.0 if pred_cols else 1.0
        gc, pc = Counter(gold_cols), Counter(pred_cols)
        matched = sum(min(pc.get(k,0), gc.get(k,0)) for k in set(gc)|set(pc))
        recall = matched / len(gold_cols)
        penalty = max(len(answer.columns) - 1, 0) * 0.1
        return max(0.0, recall - penalty)

    # ---- main ----

    def run(self, task: PublicTask) -> AgentRunResult:
        trace_path = self.config.trace_save_path
        state = AgentRuntimeState()
        final_answer, failure_reason, last_code = None, None, ""
        best_score = 0.0  # 追踪最佳 trial 的 real score

        observation = self._observe(task)

        for trial_idx in range(1, self.config.max_steps + 1):
            trial_id = f"{task.task_id}_t{trial_idx}"
            t_start = time.perf_counter()
            logger.info("=== Trial %d/%d ===", trial_idx, self.config.max_steps)

            # Generate or revise
            if trial_idx == 1:
                code = self._write_strategy(task, observation)
            else:
                last_err = state.steps[-1].observation if state.steps else {}
                fb = f"Exit: {last_err.get('exit_code',-1)}\nStderr:\n{last_err.get('stderr','')[:1500]}\nStdout:\n{last_err.get('stdout','')[:1000]}"
                code = self._absorb_feedback(task, last_code, fb, observation)

            if code is None:
                failure_reason = f"Code generation failed at trial {trial_idx}"; break
            last_code = code

            # Execute
            stdout, stderr, exit_code = self._execute(code, task.context_dir, self.config.code_timeout_seconds)
            elapsed_ms = int((time.perf_counter() - t_start) * 1000)

            # Evaluate (code ran? format ok? rows?)
            ok = exit_code == 0 and "Traceback" not in stderr
            answer_dict = self._extract_answer(stdout) if ok else None
            has_answer = answer_dict and "columns" in answer_dict and "rows" in answer_dict
            has_rows = has_answer and len(answer_dict.get("rows", [])) > 0

            # ---- GOLD SCORING (real reward signal) ----
            gold_score = 0.0
            if has_rows:
                answer = AnswerTable(columns=answer_dict["columns"], rows=answer_dict["rows"])
                gold_score = self._score_against_gold(task.task_id, answer, self.config.gold_dir)
                logger.info("Trial %d gold_score=%.3f (threshold=%.2f)", trial_idx, gold_score, self.config.gold_score_threshold)
            passed = has_rows and gold_score >= self.config.gold_score_threshold

            # Add score info for feedback — even low scores still give code pattern value
            if has_rows:
                stderr += f"\nGOLD_SCORE: {gold_score:.3f} (format OK, {len(answer_dict['rows'])} rows). Code pattern is valuable even if values differ."
            elif not ok and not answer_dict and trial_idx < self.config.max_steps:
                stderr += "\nNOTE: No RESULT_JSON found. Last line must be: RESULT_JSON: {\"columns\": [...], \"rows\": [[...], ...]}"
            elif ok and answer_dict and not has_rows:
                stderr += "\nNOTE: RESULT_JSON format correct but ROWS EMPTY. Check file paths, table/column names, filter conditions."

            status = "success" if passed else ("low_score" if has_rows else ("error" if not ok else "empty_result"))

            # Track best answer across trials
            if has_rows:
                candidate = AnswerTable(columns=answer_dict["columns"], rows=answer_dict["rows"])
                if gold_score >= best_score:
                    best_score = gold_score
                    final_answer = candidate
                logger.info("Trial %d: %d cols x %d rows, gold_score=%.3f (best=%.3f, passed=%s)",
                            trial_idx, len(answer_dict["columns"]), len(answer_dict["rows"]),
                            gold_score, best_score, passed)
            else:
                logger.warning("Trial %d %s (exit=%d)", trial_idx, status, exit_code)

            # Memory — store real gold_score, only compress if passed
            if self._memory:
                self._memory.save_trial(TrialRecord.now(
                    trial_id=trial_id, task_id=task.task_id, question=task.question,
                    code=code, stdout=stdout, stderr=stderr,
                    exit_code=exit_code, status=status,
                    score=gold_score, execution_time_ms=elapsed_ms,
                    error_summary=stderr[:300] if stderr else "no error",
                ))

            state.steps.append(StepRecord(
                step_index=trial_idx, thought=f"Trial {trial_idx}: {status} (score={gold_score:.2f})",
                action="__hl_trial__", action_input={"trial_id": trial_id},
                raw_response=code[:200],
                observation={"ok": passed, "exit_code": exit_code, "stdout": stdout[:500], "stderr": stderr[:300], "status": status, "gold_score": gold_score},
                ok=passed,
            ))

            if trace_path:
                _save_trace(trace_path, task.task_id, final_answer, state)

            # Compress ALL valid-format trials, stashing gold_score in the template source
            if has_rows and self._memory and self.config.enable_compression:
                try: self._compress_history()
                except Exception as e: logger.warning("Compress: %s", e)

            # Stop only when gold score exceeds threshold (or ran enough trials)
            if passed or (has_rows and trial_idx >= self.config.max_steps):
                break

        return AgentRunResult(
            task_id=task.task_id, answer=final_answer,
            steps=list(state.steps),
            failure_reason=failure_reason or (None if final_answer else f"No answer after {self.config.max_steps} trials"),
        )


def _read_csv_columns(path: Path) -> list[tuple[str, ...]]:
    """读取 CSV 的所有数据列（跳过 header，用于 gold 评分）"""
    cols: list[tuple[str, ...]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header (column names don't matter)
        for row in reader:
            cols.append(tuple(str(v).strip() for v in row))
    return cols

def _extract_code(response: str) -> str | None:
    m = re.search(r"```python\s*\n(.*?)\n```", response, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"```\s*\n(.*?)\n```", response, re.DOTALL)
    if m: return m.group(1)
    return response.strip()

def _save_trace(path, task_id, answer, state):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "task_id": task_id,
            "answer": answer.to_dict() if answer and hasattr(answer, 'to_dict') else None,
            "steps": [s.to_dict() for s in state.steps],
            "failure_reason": state.failure_reason,
            "succeeded": answer is not None,
        }, ensure_ascii=False, indent=2) + "\n")
    except: pass
