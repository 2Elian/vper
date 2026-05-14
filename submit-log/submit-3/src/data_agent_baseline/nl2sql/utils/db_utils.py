"""
Database utilities — execute SQL and parse results.
移植自 DeepEye nl2sql/utils/db_utils.py
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SQLResult:
    result_type: str = ""  # "success", "empty_result", "all_null_result", "error"
    result_rows: list[Any] | None = None
    result_table_str: str = ""
    error_message: str = ""
    row_count: int = 0


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def execute_sql(database_path: str, sql: str, timeout: int = 30) -> SQLResult:
    """Execute a SQL query and return structured results.

    Args:
        database_path: Path to the SQLite database file
        sql: SQL query to execute
        timeout: Timeout in seconds
    """
    if not sql or not sql.strip():
        return SQLResult(
            result_type="error",
            error_message="Empty SQL statement",
        )

    sql = sql.strip()

    if not sql.upper().startswith(("SELECT", "WITH", "PRAGMA")):
        return SQLResult(
            result_type="error",
            error_message=f"Only read-only queries allowed: {sql[:50]}",
        )

    try:
        db_path = Path(database_path)
        conn = _connect_read_only(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return SQLResult(
                result_type="empty_result",
                result_rows=[],
                result_table_str=_format_table(columns, []),
                row_count=0,
            )

        # Check if all values are None
        all_null = all(
            all(val is None for val in row) for row in rows
        )

        if all_null:
            return SQLResult(
                result_type="all_null_result",
                result_rows=[list(row) for row in rows],
                result_table_str=_format_table(columns, [list(row) for row in rows]),
                row_count=len(rows),
            )

        return SQLResult(
            result_type="success",
            result_rows=[list(row) for row in rows],
            result_table_str=_format_table(columns, [list(row) for row in rows]),
            row_count=len(rows),
        )

    except sqlite3.OperationalError as e:
        return SQLResult(
            result_type="error",
            error_message=f"SQL execution error: {str(e)}",
        )
    except Exception as e:
        return SQLResult(
            result_type="error",
            error_message=f"Unexpected error: {str(e)}",
        )


def _format_table(columns: list[str], rows: list[list[Any]], max_rows: int = 50) -> str:
    if not columns:
        return "Empty result set"

    display_rows = rows[:max_rows]
    lines = [" | ".join(columns)]
    lines.append("-" * len(lines[0]))

    for row in display_rows:
        lines.append(" | ".join(str(v) for v in row))

    if len(rows) > max_rows:
        lines.append(f"... and {len(rows) - max_rows} more rows")

    return "\n".join(lines)
