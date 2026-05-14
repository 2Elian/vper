"""
Schema Utilities — 移植自 DeepEye nl2sql/utils/schema_utils.py
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


def get_database_schema_profile(metadata: dict) -> str:
    """从 metadata dict 生成 schema 文本描述。

    metadata 格式::
        {
            "name": "db_name",
            "tables": [
                {
                    "name": "table1",
                    "columns": [
                        {"name": "col1", "type": "TEXT", "examples": [...], "enums": [...]},
                    ]
                }
            ]
        }
    """
    lines = [f"Database: {metadata.get('name', 'unknown')}", ""]

    tables = metadata.get("tables", [])
    for table in tables:
        lines.append(f"Table: {table['name']}")
        lines.append("  Columns:")
        for col in table.get("columns", []):
            col_info = f"    - {col['name']} ({col.get('type', 'TEXT')})"
            if col.get("is_primary"):
                col_info += " [PRIMARY KEY]"
            if col.get("is_foreign"):
                col_info += " [FOREIGN KEY]"
            lines.append(col_info)

            examples = col.get("examples", [])
            if examples:
                examples_str = ", ".join(str(v) for v in examples[:5])
                lines.append(f"      Value Examples: [{examples_str}]")

            enums = col.get("enums", [])
            if enums:
                enum_values = [e.get("value", e) if isinstance(e, dict) else e for e in enums[:10]]
                enum_str = ", ".join(str(v) for v in enum_values)
                lines.append(f"      Enum Values: [{enum_str}]")
        lines.append("")

    profile = "\n".join(lines)
    max_chars = 28000
    if len(profile) > max_chars:
        profile = profile[:max_chars] + "\n... [schema truncated]\n"
    return profile


def merge_schema_linking_results(
    results: list[dict[str, list[str]] | None]
) -> dict[str, list[str]]:
    """合并多个 schema linking 结果。"""
    merged: dict[str, set[str]] = {}
    for result in results:
        if result is None:
            continue
        for table_name, columns in result.items():
            if table_name not in merged:
                merged[table_name] = set()
            merged[table_name].update(columns)
    return {table_name: list(columns) for table_name, columns in merged.items()}


def filter_used_database_schema(
    metadata: dict,
    linked_tables_and_columns: dict[str, list[str]]
) -> dict:
    """根据 linked schema 过滤 metadata，只保留相关表/列。"""
    filtered = deepcopy(metadata)
    filtered_tables = []

    tables = metadata.get("tables", [])
    table_map = {t["name"]: t for t in tables}

    for table_name, column_names in linked_tables_and_columns.items():
        table = table_map.get(table_name)
        if table is None:
            continue

        filtered_table = deepcopy(table)
        column_names_set = set(col.lower() for col in column_names)

        filtered_columns = []
        for col in table.get("columns", []):
            if (col["name"].lower() in column_names_set or
                    col.get("is_primary") or col.get("is_foreign")):
                filtered_columns.append(deepcopy(col))

        filtered_table["columns"] = filtered_columns
        filtered_tables.append(filtered_table)

    filtered["tables"] = filtered_tables
    return filtered


def map_lower_table_name_to_original(table_name: str, metadata: dict) -> str | None:
    table_name_lower = table_name.lower()
    for table in metadata.get("tables", []):
        if table["name"].lower() == table_name_lower:
            return table["name"]
    return None


def map_lower_column_name_to_original(
    table_name: str, column_name: str, metadata: dict
) -> str | None:
    column_name_lower = column_name.lower()
    table_map = {t["name"]: t for t in metadata.get("tables", [])}
    table = table_map.get(table_name)
    if table is None:
        return None
    for col in table.get("columns", []):
        if col["name"].lower() == column_name_lower:
            return col["name"]
    return None
