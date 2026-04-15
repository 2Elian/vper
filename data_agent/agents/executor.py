"""
Executor Agent - 步骤执行器

参考 Eino 的 Executor 和 baseline 的 ReActAgent，职责：
1. 执行单个计划步骤
2. 使用 ReAct 循环 + 工具
3. 返回步骤结果（StepResult）

核心流程（Think-Act-Observe）：
- 接收步骤定义 + 依赖结果
- 迭代执行 LLM 调用
- 调用工具获取观察结果
- 检查是否完成（return_result）
"""


import json
import re
from typing import Any, Dict, List, Optional
from data_agent.llms import BaseLLMClient
from data_agent.agents.base import ChatModelAgent
from data_agent.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Session,
    StepResult,
    StepStatus,
)


EXECUTOR_SYSTEM_PROMPT = """You are a data analysis executor agent.

You execute a specific step in a larger execution plan. You have access to tools for:
- Reading data files (CSV, JSON, SQLite, documents)
- Executing Python code and SQL queries
- Listing available files
- Submitting the final answer

Your workflow (ReAct pattern):
1. THINK: Analyze what to do next
2. ACT: Use a tool or return result
3. OBSERVE: See the tool output
4. REPEAT until the step is complete

Rules:
1. Use tools to inspect data before answering.
2. Base your answer ONLY on observed data.
3. When the step is complete, call `return_result` with the data.
4. Always return exactly one JSON object with keys: thought, action, action_input.
5. Wrap the JSON in a ```json code block.
6. The action_input must match the tool's expected parameters.
7. Use execute_python for complex data transformations using pandas.
"""


class ExecutorAgent(ChatModelAgent):
    """
    Executor Agent: 执行单个计划步骤

    参考 Eino 的 Executor 和 baseline ReActAgent：
    - 使用 ReAct 循环执行步骤
    - 调用工具完成任务
    - 返回结构化结果
    """

    def __init__(self, model: BaseLLMClient, tools: Any, schema_summary: str = ""):
        super().__init__(model=model, tools=tools)
        self._schema_summary = schema_summary

    @property
    def name(self) -> str:
        return "Executor"

    @property
    def description(self) -> str:
        return "Executes a plan step using ReAct loop with tools."

    def _build_system_prompt(self) -> str:
        tools_desc = ""
        if self._tools and hasattr(self._tools, 'describe_for_prompt'):
            tools_desc = self._tools.describe_for_prompt()
        return f"{EXECUTOR_SYSTEM_PROMPT}\n\nAvailable tools:\n{tools_desc}"

    def run(
        self,
        agent_input: AgentInput,
        session: Session,
        history: History,
    ) -> AgentEvent:
        """执行单个计划步骤"""

        step = agent_input.step
        if not step:
            return AgentEvent(
                agent_name=self.name,
                output=None,
                action=AgentAction.ERROR,
                error="No step provided to Executor",
            )

        # 从 session 获取 schema 信息
        schema_summary = session.get("schema_summary", self._schema_summary)

        # 从依赖步骤获取结果
        dep_results = agent_input.dependency_results or {}

        # 构建执行消息
        messages = self._build_messages(
            step=step,
            schema_summary=schema_summary,
            dep_results=dep_results,
            history=history,
        )

        steps_taken = 0
        tool_calls = []
        result_data = None

        for step_idx in range(step.max_steps):
            steps_taken = step_idx + 1

            try:
                # 调用 LLM
                raw_response = self._call_model(messages)
                history.add_message("assistant", raw_response)

                # 解析响应
                parsed = self._parse_json_response(raw_response)
                if parsed is None:
                    history.add_message("user", "Failed to parse JSON response")
                    continue

                thought = parsed.get("thought", "")
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", {})

                # 检查是否完成
                if action in ("return_result", "answer"):
                    result_data = action_input
                    break

                # 执行工具
                observation = self._execute_tool(action, action_input, session)
                tool_calls.append({
                    "step": step_idx + 1,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })

                history.add_message("user", f"Observation: {json.dumps(observation, ensure_ascii=False)}")

            except Exception as exc:
                tool_calls.append({
                    "step": step_idx + 1,
                    "error": str(exc),
                })
                history.add_message("user", f"Error: {str(exc)}")

        # 构建结果
        success = result_data is not None

        step_result = StepResult(
            step_id=step.step_id,
            success=success,
            data=result_data,
            error="" if success else "Max steps reached or no result",
            steps_taken=steps_taken,
            tool_calls=tool_calls,
        )

        # 存储产出的主题数据到 Session
        if success and step.produces:
            for topic in step.produces:
                session.set_topic(topic, result_data)

        return AgentEvent(
            agent_name=self.name,
            output=step_result.to_dict(),
            action=AgentAction.EXIT if success else AgentAction.CONTINUE,
            step_result=step_result,
        )

    def _build_messages(
        self,
        step: Any,
        schema_summary: str,
        dep_results: Dict[str, StepResult],
        history: History,
    ) -> List[Dict[str, str]]:
        """构建 LLM 消息列表"""

        # 构建上下文信息
        context_parts = []

        if schema_summary:
            context_parts.append(f"=== Data Sources ===\n{schema_summary}")

        # 添加依赖结果
        if dep_results:
            context_parts.append("\n=== Previous Step Results ===")
            for dep_id, result in dep_results.items():
                context_parts.append(f"\n[{dep_id}]: {json.dumps(result.data, ensure_ascii=False)[:500]}")

        context_str = "\n".join(context_parts)

        # 步骤提示
        step_prompt = f"""=== Current Step ===
Step: {step.step_id}
Description: {step.description}
Hint: {step.hint}
Expected Output: {step.expected_output if hasattr(step, 'expected_output') else 'Any relevant data'}

When complete, call return_result with your findings."""

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"{context_str}\n\n{step_prompt}"},
        ]

        # 添加历史消息（限制长度）
        historical_messages = history.get_messages()[-20:]
        messages.extend(historical_messages)

        return messages

    def _execute_tool(self, action: str, action_input: Dict[str, Any], session: Session) -> Dict[str, Any]:
        """执行工具调用"""

        if not self._tools:
            return {"ok": False, "error": "No tools available"}

        if action == "return_result" or action == "answer":
            return {"ok": True, "status": "returned", "data": action_input}

        if action == "list_context":
            # 从 session 获取 context_dir
            context_dir = session.get("context_dir")
            if context_dir:
                import os
                entries = []
                for root, dirs, files in os.walk(str(context_dir)):
                    for fname in sorted(files):
                        rel = os.path.relpath(os.path.join(root, fname), str(context_dir))
                        entries.append({"path": rel.replace("\\", "/"), "kind": "file"})
                return {"root": str(context_dir), "entries": entries}
            return {"ok": False, "error": "No context_dir in session"}

        try:
            tool_result = self._tools.execute(None, action, action_input)
            if hasattr(tool_result, 'content'):
                return tool_result.content
            return {"ok": tool_result.ok, "content": tool_result.content}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}