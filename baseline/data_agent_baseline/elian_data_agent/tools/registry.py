from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pathlib import Path


@dataclass()
class ToolSpec:
    """工具规格"""
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass()
class ToolResult:
    """工具执行结果"""
    ok: bool
    content: Dict[str, Any]
    is_terminal: bool = False

ToolHandler = Callable[[Dict[str, Any]], ToolResult]


class ToolRegistry:
    def __init__(self):
        self.specs: Dict[str, ToolSpec] = {}
        self.handlers: Dict[str, ToolHandler] = {}

    def register(self, name: str, description: str, input_schema: Dict[str, Any],
                 handler: ToolHandler) -> None:
        """注册工具"""
        self.specs[name] = ToolSpec(name=name, description=description, input_schema=input_schema)
        self.handlers[name] = handler

    def describe_for_prompt(self) -> str:
        """生成工具描述（用于 LLM 提示词）"""
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, action: str, action_input: Dict[str, Any]) -> ToolResult:
        """执行工具"""
        if action not in self.handlers:
            return ToolResult(ok=False, content={"error": f"Unknown tool: {action}"})
        return self.handlers[action](action_input)

    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self.handlers

    def list_tools(self) -> list:
        """列出所有工具名称"""
        return sorted(self.specs.keys())


def create_tool_registry(context_dir: Optional[Path] = None) -> ToolRegistry:
    """
    创建工具注册表
    - list_context
    - read_csv
    - read_json
    - read_doc
    - inspect_sqlite_schema
    - execute_context_sql
    - execute_python
    - answer
    """
    registry = ToolRegistry()
    _ctx_dir = context_dir

    def _resolve_path(relative_path: str) -> Path:
        """解析相对路径到绝对路径"""
        if _ctx_dir is None:
            raise ValueError("context_dir not set")
        candidate = (_ctx_dir / relative_path).resolve()
        context_root = _ctx_dir.resolve()
        if context_root not in candidate.parents and candidate != context_root:
            raise ValueError(f"Path escapes context dir: {relative_path}")
        if not candidate.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return candidate

    # 1. list_context
    def _list_context(action_input: Dict[str, Any]) -> ToolResult:
        import os
        max_depth = int(action_input.get("max_depth", 4))
        entries = []

        def walk(path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
                rel_path = child.relative_to(_ctx_dir).as_posix()
                entries.append({
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                })
                if child.is_dir():
                    walk(child, depth + 1)

        if _ctx_dir:
            walk(_ctx_dir, 1)
        return ToolResult(ok=True, content={"root": str(_ctx_dir), "entries": entries})

    registry.register(
        name="list_context",
        description="List files and directories available under context.",
        input_schema={"max_depth": 4},
        handler=_list_context,
    )

    # 2. read_csv
    def _read_csv(action_input: Dict[str, Any]) -> ToolResult:
        import csv as csv_mod
        path = str(action_input.get("path", ""))
        max_rows = int(action_input.get("max_rows", 20))
        full_path = _resolve_path(path)

        with full_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv_mod.reader(handle)
            rows = list(reader)

        if not rows:
            return ToolResult(ok=True, content={"path": path, "columns": [], "rows": [], "row_count": 0})

        header = rows[0]
        data_rows = rows[1:]
        return ToolResult(ok=True, content={
            "path": path, "columns": header,
            "rows": data_rows[:max_rows], "row_count": len(data_rows),
        })

    registry.register(
        name="read_csv",
        description="Read a preview of a CSV file inside context.",
        input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        handler=_read_csv,
    )

    # 3. read_json
    def _read_json(action_input: Dict[str, Any]) -> ToolResult:
        import json
        path = str(action_input.get("path", ""))
        max_chars = int(action_input.get("max_chars", 4000))
        full_path = _resolve_path(path)
        payload = json.loads(full_path.read_text(encoding="utf-8", errors="replace"))
        preview = json.dumps(payload, ensure_ascii=False, indent=2)
        return ToolResult(ok=True, content={
            "path": path, "preview": preview[:max_chars],
            "truncated": len(preview) > max_chars,
        })

    registry.register(
        name="read_json",
        description="Read a preview of a JSON file inside context.",
        input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        handler=_read_json,
    )

    # 4. read_doc
    def _read_doc(action_input: Dict[str, Any]) -> ToolResult:
        path = str(action_input.get("path", ""))
        max_chars = int(action_input.get("max_chars", 8000))
        full_path = _resolve_path(path)
        text = full_path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(ok=True, content={
            "path": path, "content": text[:max_chars],
            "truncated": len(text) > max_chars,
        })

    registry.register(
        name="read_doc",
        description="Read a text-like document inside context.",
        input_schema={"path": "relative/path/to/file.md", "max_chars": 8000},
        handler=_read_doc,
    )

    # 5. inspect_sqlite_schema
    def _inspect_sqlite_schema(action_input: Dict[str, Any]) -> ToolResult:
        import sqlite3
        path = str(action_input.get("path", ""))
        full_path = _resolve_path(path)

        with sqlite3.connect(str(full_path)) as conn:
            rows = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            tables = [{"name": name, "create_sql": create_sql} for name, create_sql in rows]

        return ToolResult(ok=True, content={"path": str(full_path), "tables": tables})

    registry.register(
        name="inspect_sqlite_schema",
        description="Inspect tables and columns in a sqlite/db file inside context.",
        input_schema={"path": "relative/path/to/file.sqlite"},
        handler=_inspect_sqlite_schema,
    )

    # 6. execute_context_sql
    def _execute_context_sql(action_input: Dict[str, Any]) -> ToolResult:
        import sqlite3
        path = str(action_input.get("path", ""))
        sql = str(action_input.get("sql", ""))
        limit = int(action_input.get("limit", 200))
        full_path = _resolve_path(path)

        normalized = sql.lstrip().lower()
        if not normalized.startswith(("select", "with", "pragma")):
            return ToolResult(ok=False, content={"error": "Only read-only SQL allowed"})

        with sqlite3.connect(str(full_path)) as conn:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit)

        return ToolResult(ok=True, content={
            "path": str(full_path), "columns": columns,
            "rows": [list(r) for r in rows], "row_count": len(rows),
        })

    registry.register(
        name="execute_context_sql",
        description="Run a read-only SQL query against a sqlite/db file inside context.",
        input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        handler=_execute_context_sql,
    )

    # 7. execute_python
    def _execute_python(action_input: Dict[str, Any]) -> ToolResult:
        import subprocess
        import sys
        import tempfile
        import os

        code = str(action_input.get("code", ""))
        if not code:
            return ToolResult(ok=False, content={"error": "No code provided"})

        context_dir_str = str(_ctx_dir).replace("\\", "/") if _ctx_dir else "."
        wrapped_code = (
            "import os, sys, json\n"
            f"os.chdir(r'{context_dir_str}')\n"
            "import pandas as pd\n"
            f"{code}\n"
        )

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(wrapped_code)
                tmp_path = tmp.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
            os.unlink(tmp_path)

            return ToolResult(
                ok=proc.returncode == 0,
                content={
                    "output": proc.stdout[-4000:] if proc.stdout else "",
                    "error": proc.stderr[-2000:] if proc.stderr else "",
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, content={"error": "Execution timed out (30s)"})
        except Exception as exc:
            return ToolResult(ok=False, content={"error": str(exc)})

    registry.register(
        name="execute_python",
        description=(
            "Execute arbitrary Python code with the task context directory as the "
            "working directory. Returns captured stdout."
        ),
        input_schema={"code": "import os\nprint(sorted(os.listdir('.')))"},
        handler=_execute_python,
    )

    # 8. answer (终端动作)
    def _answer(action_input: Dict[str, Any]) -> ToolResult:
        columns = action_input.get("columns", [])
        rows = action_input.get("rows", [])
        if not isinstance(columns, list) or not columns:
            return ToolResult(ok=False, content={"error": "columns must be a non-empty list"})
        if not isinstance(rows, list):
            return ToolResult(ok=False, content={"error": "rows must be a list"})
        return ToolResult(
            ok=True,
            content={"status": "submitted", "column_count": len(columns), "row_count": len(rows)},
            is_terminal=True,
        )

    registry.register(
        name="answer",
        description="Submit the final answer table. This is the only valid terminating action.",
        input_schema={"columns": ["column_name"], "rows": [["value_1"]]},
        handler=_answer,
    )

    return registry