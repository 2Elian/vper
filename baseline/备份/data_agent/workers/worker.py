"""
Worker Agent - 独立 ReAct 循环的执行者

Worker Agent 接收子任务，使用共享 Workspace 和工具执行，
完成后进入 idle 状态等待 Lead Agent 的下一步指令。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_agent.core.workspace import Workspace


@dataclass
class WorkerResult:
    """Worker 执行结果"""
    task_id: str
    worker_id: str
    success: bool
    data: Any = None  # 执行结果数据
    error: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)


class WorkerAgent(object):
    """
    Worker Agent: 独立执行子任务

    与 baseline 的 ReActAgent 不同：
    1. 使用共享 Workspace（缓存 + Schema Map）
    2. 带有 knowledge.md 注入的上下文
    3. 完成后返回结构化结果
    """

    def __init__(self, worker_id, model, workspace, tools=None):
        # type: (str, Any, Workspace, Any) -> None
        self.worker_id = worker_id
        self.model = model
        self.workspace = workspace
        self.tools = tools
        self.status = "idle"  # idle | running | done

    def execute(self, subtask):
        # type: (Dict[str, Any]) -> WorkerResult
        """
        执行子任务

        subtask 结构:
        {
            "task_id": "task_344",
            "subtask_id": "sub_1",
            "description": "从 Patient.md 中提取所有男性患者的 ID",
            "hint": "根据 knowledge.md，SEX='M' 表示男性",
            "max_steps": 8,
            "expected_output": "list of patient IDs"
        }
        """
        self.status = "running"
        max_steps = subtask.get("max_steps", 8)
        steps = []

        # 构建消息
        messages = self._build_messages(subtask)

        result_data = None
        success = False

        for step_idx in range(max_steps):
            try:
                raw_response = self.model.complete(messages)
                messages.append({"role": "assistant", "content": raw_response})

                # 解析 response
                parsed = self._parse_response(raw_response)
                if parsed is None:
                    steps.append({"step": step_idx + 1, "error": "Failed to parse response"})
                    continue

                thought = parsed.get("thought", "")
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", {})

                # 执行工具
                observation = self._execute_action(action, action_input)
                steps.append({
                    "step": step_idx + 1,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })

                # 添加 observation 到消息
                messages.append({
                    "role": "user",
                    "content": "Observation:\n{}".format(json.dumps(observation, ensure_ascii=False, indent=2))
                })

                # 检查是否完成
                if action == "return_result":
                    result_data = action_input
                    success = True
                    break

                if action == "answer":
                    result_data = action_input
                    success = True
                    break

            except Exception as exc:
                steps.append({"step": step_idx + 1, "error": str(exc)})

        self.status = "done"

        return WorkerResult(
            task_id=subtask.get("task_id", ""),
            worker_id=self.worker_id,
            success=success,
            data=result_data,
            error="" if success else "Max steps reached or parse error",
            steps=steps,
        )

    def _build_messages(self, subtask):
        # type: (Dict[str, Any]) -> List[Dict[str, str]]
        """构建 LLM 消息列表"""
        # 注入 Workspace 的 Schema 信息
        schema_summary = self.workspace.get_schema_summary()

        # 注入 knowledge.md 相关信息
        knowledge_context = ""
        if self.workspace.knowledge_guide:
            guide = self.workspace.knowledge_guide
            knowledge_context = "\n=== 知识指南 ===\n"
            knowledge_context += "数据库: {}\n".format(guide.database_name)

            for entity_name, fields in guide.entities.items():
                knowledge_context += "实体 {}:\n".format(entity_name)
                for f in fields:
                    knowledge_context += "  - {}: {}".format(f.name, f.description)
                    if f.values:
                        knowledge_context += " (值域: {})".format(", ".join(f.values))
                    knowledge_context += "\n"

            if guide.constraints:
                knowledge_context += "约束:\n"
                for c in guide.constraints:
                    for rule in c.rules:
                        knowledge_context += "  - {}\n".format(rule)

            if guide.use_cases:
                knowledge_context += "示例:\n"
                for uc in guide.use_cases[:3]:  # 最多3个示例
                    knowledge_context += "  - {}: {}\n".format(uc.name, uc.sql_formula[:100])

        system_prompt = (
            "You are a data analysis worker agent. You analyze data to answer specific subtasks.\n\n"
            "{schema}\n\n"
            "{knowledge}\n\n"
            "Rules:\n"
            "1. Use tools to inspect data before answering.\n"
            "2. Base your answer only on observed data.\n"
            "3. When you have the result, call return_result with the data.\n"
            "4. Always return exactly one JSON object with keys: thought, action, action_input.\n"
            "5. Wrap the JSON in a ```json code block.\n"
            "6. For CSV files, prefer using execute_python with pandas for complex queries.\n"
        ).format(schema=schema_summary, knowledge=knowledge_context)

        task_prompt = (
            "Subtask: {description}\n\n"
            "Hint: {hint}\n\n"
            "Expected output: {expected}\n\n"
            "When done, call return_result with your findings."
        ).format(
            description=subtask.get("description", ""),
            hint=subtask.get("hint", "No hint provided"),
            expected=subtask.get("expected_output", "Any useful data"),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        return messages

    def _parse_response(self, raw_response):
        # type: (str) -> Optional[Dict[str, Any]]
        """解析 LLM 返回的 JSON"""
        text = raw_response.strip()

        # 去除 code fence
        fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        else:
            fence_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()

        try:
            payload, end = json.JSONDecoder().raw_decode(text)
            if not isinstance(payload, dict):
                return None
            return payload
        except (json.JSONDecodeError, ValueError):
            return None

    def _execute_action(self, action, action_input):
        # type: (str, Dict[str, Any]) -> Dict[str, Any]
        """执行工具调用"""
        if action == "return_result" or action == "answer":
            return {"ok": True, "status": "returned", "data": action_input}

        if action == "list_context":
            return {
                "ok": True,
                "files": list(self.workspace.files.keys()),
                "schema_summary": self.workspace.get_schema_summary(),
            }

        if action == "read_csv":
            path = str(action_input.get("path", ""))
            max_rows = int(action_input.get("max_rows", 100))
            return self.workspace.read_csv(path, max_rows=max_rows)

        if action == "read_json":
            path = str(action_input.get("path", ""))
            max_chars = int(action_input.get("max_chars", 8000))
            return self.workspace.read_json(path, max_chars=max_chars)

        if action == "read_doc":
            path = str(action_input.get("path", ""))
            chunk_index = int(action_input.get("chunk_index", 0))
            chunks = self.workspace.read_doc_chunks(path, chunk_size=4000)
            if chunk_index < len(chunks):
                return chunks[chunk_index]
            return {"ok": False, "error": "chunk_index out of range", "total_chunks": len(chunks)}

        if action == "read_doc_full":
            """读取完整文档（解决截断问题）"""
            path = str(action_input.get("path", ""))
            chunks = self.workspace.read_doc_chunks(path, chunk_size=4000)
            return {
                "ok": True,
                "path": path,
                "total_chunks": len(chunks),
                "content": "\n--- CHUNK BOUNDARY ---\n".join(
                    c["content"] for c in chunks
                ),
            }

        if action == "execute_python":
            return self._execute_python(action_input)

        if action == "execute_context_sql":
            return self._execute_sql(action_input)

        if action == "inspect_sqlite_schema":
            return self._inspect_schema(action_input)

        if action == "find_column":
            """查找某列在哪些文件中"""
            col_name = str(action_input.get("column", ""))
            results = self.workspace.find_files_with_column(col_name)
            return {"ok": True, "column": col_name, "found_in": results}

        return {"ok": False, "error": "Unknown action: {}".format(action)}

    def _execute_python(self, action_input):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """执行 Python 代码"""
        code = str(action_input.get("code", ""))
        if not code:
            return {"ok": False, "error": "No code provided"}

        import subprocess
        import sys
        import tempfile

        # 添加 workspace 上下文路径到代码
        context_dir = str(self.workspace.context_dir).replace("\\", "/")
        wrapped_code = (
            "import os, sys, json\n"
            "os.chdir(r'{context_dir}')\n"
            "import pandas as pd\n"
            "{code}\n"
        ).format(context_dir=context_dir, code=code)

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

            return {
                "ok": proc.returncode == 0,
                "output": proc.stdout[-4000:] if proc.stdout else "",
                "error": proc.stderr[-2000:] if proc.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Execution timed out (30s)"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _execute_sql(self, action_input):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """执行 SQL 查询"""
        import sqlite3

        path = str(action_input.get("path", ""))
        sql = str(action_input.get("sql", ""))
        limit = int(action_input.get("limit", 200))

        full_path = self.workspace.context_dir / path
        if not full_path.exists():
            return {"ok": False, "error": "File not found: {}".format(path)}

        try:
            conn = sqlite3.connect(str(full_path))
            cursor = conn.cursor()
            cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit)
            conn.close()

            return {
                "ok": True,
                "columns": columns,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _inspect_schema(self, action_input):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """检查数据库 Schema"""
        path = str(action_input.get("path", ""))
        schema = self.workspace.schemas.get(path)
        if schema is None:
            return {"ok": False, "error": "Schema not found for {}".format(path)}

        return {
            "ok": True,
            "path": path,
            "tables": schema.tables,
            "table_columns": schema.table_columns,
        }
