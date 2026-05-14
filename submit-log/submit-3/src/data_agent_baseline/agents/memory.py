"""
HL Memory System — 移植自 hldaa，参考 Learning Beyond Gradients 的显式记忆

两个核心操作:
  absorb_feedback  → 把失败/成功写回系统
  compress_history → 把成功模式折叠为可复用模板

存储:
  trials/  → trials.jsonl (每行一条试验记录)
  memory/  → *.md (压缩后的可复用模板)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("HLMemory")


@dataclass
class TrialRecord:
    """单次试验记录"""
    trial_id: str
    task_id: str
    question: str = ""
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    status: str = "error"  # success | error | empty_result
    score: float = 0.0
    execution_time_ms: int = 0
    error_summary: str = ""
    answer_preview: str = ""
    timestamp: str = ""

    @classmethod
    def now(cls, **kwargs) -> TrialRecord:
        return cls(timestamp=datetime.now(timezone.utc).isoformat(), **kwargs)

    def to_jsonl(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "question": self.question[:200],
            "exit_code": self.exit_code,
            "status": self.status,
            "score": self.score,
            "execution_time_ms": self.execution_time_ms,
            "error_summary": self.error_summary[:300],
            "answer_preview": self.answer_preview[:200],
            "timestamp": self.timestamp,
        }

    def to_feedback(self) -> str:
        """格式化为 feedback，供 LLM absorb 使用"""
        parts = [f"Exit Code: {self.exit_code}"]
        if self.stderr:
            parts.append(f"Stderr:\n{self.stderr[:2000]}")
        if self.stdout:
            parts.append(f"Stdout:\n{self.stdout[:2000]}")
        if self.error_summary:
            parts.append(f"Error: {self.error_summary}")
        return "\n\n".join(parts)


@dataclass
class MemoryTemplate:
    """压缩后的可复用模板"""
    name: str
    description: str
    template: str
    tags: list[str] = field(default_factory=list)
    source: str = ""


class HLMemory:
    """HL 记忆系统 — 跨任务积累经验"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.trials_dir = self.base_dir / "trials"
        self.memory_dir = self.base_dir / "memory"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._trial_count = 0

    # ---- Trial I/O ----

    def save_trial(self, record: TrialRecord) -> None:
        trial_file = self.trials_dir / "trials.jsonl"
        with open(trial_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_jsonl(), ensure_ascii=False) + "\n")
        self._trial_count += 1

    def recent_trials(self, limit: int = 10) -> list[dict]:
        trial_file = self.trials_dir / "trials.jsonl"
        if not trial_file.exists():
            return []
        lines = []
        with open(trial_file, encoding="utf-8") as f:
            for line in f:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return lines[-limit:]

    def recent_successes(self, limit: int = 10) -> list[dict]:
        return [t for t in self.recent_trials(limit * 3) if t.get("status") == "success"][-limit:]

    def trial_count(self) -> int:
        trial_file = self.trials_dir / "trials.jsonl"
        if not trial_file.exists():
            return 0
        return sum(1 for _ in open(trial_file, encoding="utf-8"))

    # ---- Memory Templates ----

    def save_template(self, tmpl: dict) -> None:
        name = tmpl.get("name", "untitled").lower().replace(" ", "_")
        path = self.memory_dir / f"{name}.md"
        content = (
            f"# {tmpl.get('name', 'Untitled')}\n\n"
            f"**Description**: {tmpl.get('description', '')}\n\n"
            f"**Tags**: {', '.join(tmpl.get('tags', []))}\n\n"
            f"**Template**:\n\n{tmpl.get('template', '')}\n"
        )
        path.write_text(content, encoding="utf-8")

    def load_templates(self) -> list[MemoryTemplate]:
        templates: list[MemoryTemplate] = []
        if not self.memory_dir.exists():
            return templates
        for f in sorted(self.memory_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            templates.append(MemoryTemplate(
                name=f.stem,
                description=_extract_field(text, "Description"),
                template=_extract_field(text, "Template"),
                tags=_extract_field(text, "Tags").split(", "),
                source="compressed",
            ))
        return templates

    def format_memory_context(self) -> str:
        templates = self.load_templates()
        if not templates:
            return "(no memory yet)"
        return "\n\n".join(
            f"## {t.name}\n"
            f"Description: {t.description}\n"
            f"Tags: {', '.join(t.tags)}\n"
            f"Template:\n{t.template}"
            for t in templates
        )

    def format_trial_context(self, limit: int = 5) -> str:
        trials = self.recent_trials(limit)
        if not trials:
            return "(no trials yet)"
        return "\n".join(
            f"[{t.get('status','?')}] {t.get('trial_id','?')}: {t.get('question','')[:80]} "
            f"score={t.get('score',0)} {t.get('error_summary','')[:100]}"
            for t in trials
        )

    def format_feedback(self, stdout: str, stderr: str, exit_code: int, error: str = "") -> str:
        """格式化执行反馈（参考 hldaa _format_feedback）"""
        parts = [f"Exit Code: {exit_code}"]
        if stderr:
            parts.append(f"Stderr:\n{stderr[:2000]}")
        if stdout:
            parts.append(f"Stdout:\n{stdout[:2000]}")
        if error:
            parts.append(f"Error: {error}")
        return "\n\n".join(parts)


def _extract_field(text: str, field: str) -> str:
    import re
    match = re.search(rf"\*\*{field}\*\*:\s*(.*)", text)
    return match.group(1).strip() if match else ""
