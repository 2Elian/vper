"""
Workspace - 共享工作空间

解决的核心问题:
1. 第一步总是 list_context 浪费步数 -> 冷启动一次性扫描
2. 逐个读文件浪费时间 -> 缓存 + 批量读取
3. 只读文档前4000字符 -> 分块读取策略
4. 跨源集成失败 -> Schema Map 数据地图
5. 假设表存在 (task_350) -> 精确的 schema 信息
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_agent.skills.knowledge_parser import parse_knowledge, KnowledgeGuide


@dataclass
class FileInfo:
    """文件基本信息"""
    path: str           # 相对路径
    file_type: str      # csv, json, db, md, txt
    size: int           # 字节数
    description: str = ""  # 简要描述


@dataclass
class SchemaInfo:
    """数据源的 Schema 信息"""
    path: str
    file_type: str
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    tables: List[str] = field(default_factory=list)  # 仅 DB
    table_columns: Dict[str, List[str]] = field(default_factory=dict)  # 仅 DB


class Workspace(object):
    """
    共享工作空间：一次性冷启动 + 缓存 + 跨源数据地图
    """

    def __init__(self, context_dir):
        # type: (Path) -> None
        self.context_dir = context_dir
        self.files = {}  # type: Dict[str, FileInfo]
        self.schemas = {}  # type: Dict[str, SchemaInfo]
        self.cache = {}  # type: Dict[str, Any]
        self.knowledge_guide = None  # type: Optional[KnowledgeGuide]
        self._cold_started = False

    def cold_start(self):
        # type: () -> None
        """
        冷启动：一次性扫描所有可用文件，建立 Schema 地图，解析 knowledge.md
        替代 baseline 的逐步 list_context + read 操作
        """
        if self._cold_started:
            return
        # 扫描上下文目录的所有文件
        self._scan_files()

        # 为每个数据文件建立 Schema
        for path, info in self.files.items():
            if info.file_type in ("csv", "json", "db"):
                self._build_schema(path, info)

        # 解析 knowledge.md
        knowledge_path = self.context_dir / "knowledge.md"
        if knowledge_path.exists():
            content = knowledge_path.read_text(encoding="utf-8", errors="replace")
            self.knowledge_guide = parse_knowledge(content)

        self._cold_started = True

    def _scan_files(self):
        # type: () -> None
        """扫描 context 目录下的所有文件"""
        for root, dirs, files in os.walk(str(self.context_dir)):
            for fname in sorted(files):
                full_path = Path(root) / fname
                rel_path = str(full_path.relative_to(self.context_dir)).replace("\\", "/")
                ext = full_path.suffix.lower().lstrip(".")
                # 映射文件类型
                file_type = ext
                if ext in ("md", "txt"):
                    file_type = "doc"
                elif ext == "db":
                    file_type = "db"
                elif ext == "csv":
                    file_type = "csv"
                elif ext == "json":
                    file_type = "json"

                self.files[rel_path] = FileInfo(
                    path=rel_path,
                    file_type=file_type,
                    size=full_path.stat().st_size,
                )

    def _build_schema(self, rel_path: str, info: FileInfo) -> None:
        """为文件建立 Schema 信息"""
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
                # 使用现有的 sqlite 工具
                schema.tables, schema.table_columns = self._inspect_db(full_path)
        except Exception:
            pass  # 静默忽略解析失败的文件

        self.schemas[rel_path] = schema

    def _inspect_db(self, db_path):
        # type: (Path) -> Tuple[List[str], Dict[str, List[str]]]
        """检查 SQLite 数据库的表结构"""
        import sqlite3
        tables = []  # type: List[str]
        table_columns = {}  # type: Dict[str, List[str]]
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for row in cursor.fetchall():
                table_name = row[0]
                tables.append(table_name)
                cursor.execute("PRAGMA table_info({})".format(table_name))
                table_columns[table_name] = [col[1] for col in cursor.fetchall()]
            conn.close()
        except Exception:
            pass
        return tables, table_columns

    def read_csv(self, rel_path, max_rows=None):
        # type: (str, Optional[int]) -> Dict[str, Any]
        """读取 CSV 文件（带缓存）"""
        cache_key = "csv:{}".format(rel_path)
        if cache_key in self.cache:
            data = self.cache[cache_key]
            if max_rows is None:
                return data
            return {
                "path": rel_path,
                "columns": data["columns"],
                "rows": data["rows"][:max_rows],
                "row_count": data["row_count"],
            }

        full_path = self.context_dir / rel_path
        with full_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        if not rows:
            return {"path": rel_path, "columns": [], "rows": [], "row_count": 0}

        header = rows[0]
        data_rows = rows[1:]
        result = {
            "path": rel_path,
            "columns": header,
            "rows": data_rows,
            "row_count": len(data_rows),
        }
        # 缓存完整数据
        self.cache[cache_key] = result

        if max_rows is not None:
            result = dict(result)
            result["rows"] = data_rows[:max_rows]

        return result

    def read_json(self, rel_path, max_chars=None):
        # type: (str, Optional[int]) -> Dict[str, Any]
        """读取 JSON 文件（带缓存）"""
        cache_key = "json:{}".format(rel_path)
        if cache_key not in self.cache:
            full_path = self.context_dir / rel_path
            data = json.loads(full_path.read_text(encoding="utf-8", errors="replace"))
            self.cache[cache_key] = data

        data = self.cache[cache_key]
        preview = json.dumps(data, ensure_ascii=False, indent=2)
        if max_chars and len(preview) > max_chars:
            return {
                "path": rel_path,
                "preview": preview[:max_chars],
                "truncated": True,
                "total_chars": len(preview),
            }
        return {
            "path": rel_path,
            "preview": preview,
            "truncated": False,
            "total_chars": len(preview),
        }

    def read_doc_chunks(self, rel_path, chunk_size=4000):
        # type: (str, int) -> List[Dict[str, Any]]
        """
        分块读取文档（解决 4000 字符截断问题）
        返回所有 chunk，每个 chunk 包含 offset 和 content
        """
        cache_key = "doc:{}".format(rel_path)
        if cache_key not in self.cache:
            full_path = self.context_dir / rel_path
            self.cache[cache_key] = full_path.read_text(encoding="utf-8", errors="replace")

        text = self.cache[cache_key]
        chunks = []
        offset = 0
        while offset < len(text):
            chunk_text = text[offset:offset + chunk_size]
            chunks.append({
                "path": rel_path,
                "chunk_index": len(chunks),
                "offset": offset,
                "content": chunk_text,
                "is_last": (offset + chunk_size >= len(text)),
            })
            offset += chunk_size

        return chunks

    def get_schema_summary(self):
        # type: () -> str
        """获取跨源数据地图摘要，用于注入 Agent 提示词"""
        lines = []
        lines.append("=== 数据源概览 ===")

        for rel_path, schema in sorted(self.schemas.items()):
            if schema.file_type == "csv":
                lines.append("[CSV] {} ({}行): {}".format(
                    rel_path, schema.row_count, ", ".join(schema.columns[:10])
                ))
            elif schema.file_type == "json":
                lines.append("[JSON] {} ({}条): {}".format(
                    rel_path, schema.row_count, ", ".join(schema.columns[:10])
                ))
            elif schema.file_type == "db":
                for table_name in schema.tables:
                    cols = schema.table_columns.get(table_name, [])
                    lines.append("[DB] {} -> {} (列: {})".format(
                        rel_path, table_name, ", ".join(cols[:10])
                    ))

        # 添加 doc 文件信息
        for rel_path, info in sorted(self.files.items()):
            if info.file_type == "doc":
                size_kb = info.size / 1024
                lines.append("[DOC] {} ({}KB)".format(rel_path, round(size_kb, 1)))

        # 添加 knowledge 信息
        if self.knowledge_guide:
            guide = self.knowledge_guide
            lines.append("")
            lines.append("=== 知识指南 ===")
            lines.append("数据库: {}".format(guide.database_name))
            for entity_name, fields in guide.entities.items():
                lines.append("实体 {}: {}".format(
                    entity_name,
                    ", ".join(f.name for f in fields)
                ))
            if guide.constraints:
                lines.append("约束:")
                for c in guide.constraints:
                    for rule in c.rules:
                        lines.append("  - {}".format(rule))

        return "\n".join(lines)

    def find_files_with_column(self, column_name):
        # type: (str) -> List[Tuple[str, str]]
        """
        查找包含某列的所有文件
        返回: [(file_path, table_name), ...]  table_name 仅 DB 有
        """
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
