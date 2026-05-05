import json
import re
from typing import Any, Dict, List, Optional
from vper.llms import BaseLLMClient
from vper.agents.base import ChatModelAgent
from vper.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Session,
    StepResult,
    StepStatus,
    StepRecord,
)


EXECUTOR_SYSTEM_PROMPT = """You are a data analysis executor agent.

You execute a specific step in a larger execution plan. You have access to tools for:
- Reading data files (CSV, JSON, SQLite, documents)
- Executing Python code and SQL queries
- Submitting the current step result
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

# class ReActAgent(ChatModelAgent):
class ExecutorAgent(ChatModelAgent):
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

    def build_current_messages(self, messages: List[Dict[str, str]], react_local_state: List[Any]) -> List[Dict[str, str]]:
        messages = messages.copy()
        for step in react_local_state:
            messages.append(
                {
                    "role": "assistant", "content": step.get("raw_response") if step.get("error") is None else step.get("error")
                }
            )
            obv = step.get("observation") if step.get("error") is None else step.get("error")
            messages.append(
                {
                    "role": "user", "content": f"Observation: {json.dumps(obv, ensure_ascii=False)}"
                }
            )
        return messages

    async def run(
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
        self.logger.info("🧺"*150)
        self.logger.info(f"The plan: {step.step_id} start processing")
        # 从 session 获取 schema 信息
        schema_summary = session.get("schema_summary", self._schema_summary)

        # 从依赖步骤获取结果
        dep_results = agent_input.dependency_results or {}

        # 构建执行消息
        base_messages = self._build_messages(
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
                call_messages = self.build_current_messages(base_messages, tool_calls)
                # raw_response = self._call_model(call_messages)
                raw_response = await self._call_async_model(call_messages)
                history.add_message("assistant", raw_response)

                # 解析响应
                parsed = self._parse_json_response(raw_response)
                if parsed is None:
                    history.add_message("user", "Failed to parse JSON response")
                    continue

                thought = parsed.get("thought", "")
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", {})
                # 执行工具
                tool_result = self._tools.execute(action, action_input)
                # 检查是否完成
                if action == "return_result":
                    self.logger.info(f"{step.step_id}: complete")
                    result_data = action_input
                    break
                if tool_result.is_terminal:
                    self.logger.info(f"{step.step_id}: complete and is terminal answer")
                    result_data = tool_result.content
                    break
                observation = {
                    "ok": tool_result.ok,
                    "tool": action,
                    "content": tool_result.content,
                }
                self.logger.info(f"{step.step_id}-->{step_idx+1}/{step.max_steps}: "
                                 f"the thought, action , action_input data and observation is: "
                                 f"\n thought: {thought}\n action: {action}\n action_input: "
                                 f"{action_input} \n observation: {observation}")
                tool_calls.append({
                    "step": step_idx + 1,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                    "raw_response": raw_response,
                    "error": None
                })
                history.add_message(role="user", content=f"Observation: {json.dumps(observation, ensure_ascii=False)}")

            except Exception as exc:
                self.logger.warning(f"{step.step_id}-{step_idx+1}/{step.max_steps}: error {str(exc)} in step {step_idx}")
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
        elif not success:
            self.logger.warning(f"{step.step_id}: failed, may be `Exceeding the maximum iteration steps`")

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
                context_parts.append(f"\n[{dep_id}]: {json.dumps(result.data, ensure_ascii=False)}")

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

        # 添加历史消息（一期限制长度） --> 后期需要做一个智能摘要，保留前面的初始问题以及最后的结果，把中间结果压缩起来（到外部/摘要)，因为数据的话摘要效果不好，所以我们load到外部，
        # 然后给一个message，告诉模型，之前的数据你可以在外部获取
        historical_messages = history.get_messages()# history.get_messages()[-20:]
        messages.extend(historical_messages)

        return messages