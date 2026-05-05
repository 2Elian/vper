"""
Workspace - 共享工作空间
参考 data_agent 的 Workspace 实现，增加 Session 支持。
冷启动一次性扫描所有文件，建立 Schema 地图。
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from elian_data_agent.core.types import TaskContext


@dataclass
class FileInfo:
    """文件基本信息"""
    path: str
    file_type: str      # csv, json, db, md, txt
    size: int
    description: str = ""


@dataclass
class SchemaInfo:
    """数据源的 Schema 信息"""
    path: str
    file_type: str
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    tables: List[str] = field(default_factory=list)
    table_columns: Dict[str, List[str]] = field(default_factory=dict)


class Workspace:
    """共享工作空间
    解决的核心问题:
        1. 冷启动 -> 一次性扫描所有文件，建立 Schema 图
        2. 缓存 -> 文件内容缓存，避免重复读取
        3. 跨源集成 -> Schema Map 数据地图
        4. 精确查询 -> 通过列名查找文件
    """

    def __init__(self, context_dir: Path):
        self.context_dir = context_dir
        self.files: Dict[str, FileInfo] = {}
        self.schemas: Dict[str, SchemaInfo] = {}
        self.cache: Dict[str, Any] = {}
        self.knowledge: Optional[str] = None
        self._cold_started = False

    def cold_start(self) -> None:
        """冷启动 --> 扫描文件 --> 建立Schema图"""
        if self._cold_started:
            return
        # 扫描上下文内的全部文件
        self._scan_files()
        # 为每个文件建立schema
        for path, info in self.files.items():
            if info.file_type in ("csv", "json", "db"):
                self._build_schema(path, info)
        # 解析 knowledge.md
        knowledge_path = self.context_dir / "knowledge.md"
        if knowledge_path.exists():
            self.knowledge = knowledge_path.read_text(encoding="utf-8", errors="replace")
        self._cold_started = True

    def to_task_context(self, task_id: str, question: str, difficulty: str) -> TaskContext:
        """将 Workspace 转换为 TaskContext"""
        return TaskContext(
            task_id=task_id,
            question=question,
            difficulty=difficulty,
            context_dir=self.context_dir,
            files={p: {"type": f.file_type, "size": f.size} for p, f in self.files.items()},
            schemas={p: {"columns": s.columns, "row_count": s.row_count,
                         "tables": s.tables, "table_columns": s.table_columns}
                     for p, s in self.schemas.items()},
            knowledge=self.knowledge,
        )

    def get_schema_summary(self) -> str:
        """获取Schema摘要 --> 用于注入提示词"""
        lines = ["=== 数据源概览 ==="]
        for rel_path, schema in sorted(self.schemas.items()):
            if schema.file_type == "csv":
                lines.append(f"[CSV] {rel_path} ({schema.row_count}行): {', '.join(schema.columns[:10])}")
            elif schema.file_type == "json":
                lines.append(f"[JSON] {rel_path} ({schema.row_count}条): {', '.join(schema.columns[:10])}")
            elif schema.file_type == "db":
                for table_name in schema.tables:
                    cols = schema.table_columns.get(table_name, [])
                    lines.append(f"[DB] {rel_path} -> {table_name} (列: {', '.join(cols[:10])})")

        for rel_path, info in sorted(self.files.items()):
            if info.file_type == "doc":
                lines.append(f"[DOC] {rel_path} ({info.size / 1024:.1f}KB)")

        if self.knowledge:
            lines.append("")
            lines.append("=== 知识指南可用 ===")
            lines.append(f"长度: {len(self.knowledge)} 字符")

        return "\n".join(lines)

    def find_files_with_column(self, column_name: str) -> List[Tuple[str, str]]:
        """查找包含某列的所有文件"""
        results = []
        col_lower = column_name.lower()
        for rel_path, schema in self.schemas.items():
            if schema.file_type == "csv":
                if any(c.lower() == col_lower for c in schema.columns):
                    results.append((rel_path, ""))
            elif schema.file_type == "json":
                if any(c.lower() == col_lower for c in schema.columns):
                    results.append((rel_path, ""))
            elif schema.file_type == "db":
                for table_name, cols in schema.table_columns.items():
                    if any(c.lower() == col_lower for c in cols):
                        results.append((rel_path, table_name))
        return results

    def read_csv(self, rel_path: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """读取 CSV 文件（带缓存）"""
        cache_key = f"csv:{rel_path}"
        if cache_key in self.cache:
            data = self.cache[cache_key]
            if max_rows is None:
                return data
            return {"path": rel_path, "columns": data["columns"],
                    "rows": data["rows"][:max_rows], "row_count": data["row_count"]}

        full_path = self.context_dir / rel_path
        with full_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        if not rows:
            return {"path": rel_path, "columns": [], "rows": [], "row_count": 0}

        header = rows[0]
        data_rows = rows[1:]
        result = {"path": rel_path, "columns": header, "rows": data_rows, "row_count": len(data_rows)}
        self.cache[cache_key] = result

        if max_rows is not None:
            return {"path": rel_path, "columns": header, "rows": data_rows[:max_rows], "row_count": len(data_rows)}
        return result

    def read_json(self, rel_path: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
        """读取 JSON 文件（带缓存）"""
        cache_key = f"json:{rel_path}"
        if cache_key not in self.cache:
            full_path = self.context_dir / rel_path
            self.cache[cache_key] = json.loads(full_path.read_text(encoding="utf-8", errors="replace"))

        data = self.cache[cache_key]
        preview = json.dumps(data, ensure_ascii=False, indent=2)
        if max_chars and len(preview) > max_chars:
            return {"path": rel_path, "preview": preview[:max_chars], "truncated": True, "total_chars": len(preview)}
        return {"path": rel_path, "preview": preview, "truncated": False, "total_chars": len(preview)}

    def read_doc(self, rel_path: str, max_chars: int = 8000) -> Dict[str, Any]:
        """读取文档文件（带缓存）"""
        cache_key = f"doc:{rel_path}"
        if cache_key not in self.cache:
            full_path = self.context_dir / rel_path
            self.cache[cache_key] = full_path.read_text(encoding="utf-8", errors="replace")

        text = self.cache[cache_key]
        return {"path": rel_path, "content": text[:max_chars],
                "truncated": len(text) > max_chars, "total_chars": len(text)}

    def _scan_files(self) -> None:
        """扫描context目录"""
        for root, dirs, files in os.walk(str(self.context_dir)):
            for fname in sorted(files):
                full_path = Path(root) / fname
                rel_path = str(full_path.relative_to(self.context_dir)).replace("\\", "/")
                ext = full_path.suffix.lower().lstrip(".")

                file_type = ext
                if ext in ("md", "txt"):
                    file_type = "doc"
                elif ext == "db":
                    file_type = "db"

                self.files[rel_path] = FileInfo(
                    path=rel_path,
                    file_type=file_type,
                    size=full_path.stat().st_size,
                )

    def _build_schema(self, rel_path: str, info: FileInfo) -> None:
        """为文件建立 Schema"""
        full_path = self.context_dir / rel_path
        schema = SchemaInfo(path=rel_path, file_type=info.file_type)
        try:
            if info.file_type == "csv":
                with full_path.open(newline="", encoding="utf-8", errors="replace") as handle:
                    reader = csv.reader(handle)
                    rows = list(reader)
                if rows:
                    schema.columns = rows[0]
                    schema.row_count = len(rows) - 1

            elif info.file_type == "json":
                data = json.loads(full_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, list) and data:
                    if isinstance(data[0], dict):
                        schema.columns = list(data[0].keys())
                    schema.row_count = len(data)
                elif isinstance(data, dict):
                    schema.columns = list(data.keys())

            elif info.file_type == "db":
                schema.tables, schema.table_columns = self._inspect_db(full_path)
        except Exception:
            pass

        self.schemas[rel_path] = schema

    def _inspect_db(self, db_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
        """检查 SQLite 数据库结构"""
        import sqlite3
        tables = []
        table_columns = {}
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for row in cursor.fetchall():
                table_name = row[0]
                tables.append(table_name)
                cursor.execute(f"PRAGMA table_info({table_name})")
                table_columns[table_name] = [col[1] for col in cursor.fetchall()]
            conn.close()
        except Exception:
            pass
        return tables, table_columns