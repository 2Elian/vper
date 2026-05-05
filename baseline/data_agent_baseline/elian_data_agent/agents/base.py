import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple, List

from elian_data_agent.core.types import (
    AgentEvent,
    AgentInput,
    History,
    Session,
)
from elian_data_agent.llms import BaseLLMClient


class Agent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 名称"""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 职责描述"""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        agent_input: AgentInput,
        session: Session,
        history: History,
    ) -> AgentEvent:
        """
        Args:
            agent_input: Agent 输入数据
            session: 共享会话状态（跨Agent共享）
            history: 历史记录

        Returns:
            AgentEvent
        """
        raise NotImplementedError


class ChatModelAgent(Agent):
    def __init__(self, model: BaseLLMClient, tools: Any = None):
        self._model = model
        self._tools = tools

    def _call_model(self, messages: List[Dict[str, str]], histroy: Optional[List[Dict[str, str]]] = None) -> Union[str, Tuple[str, Optional[str]]]:
        """call LLM"""
        return self._model.sync_generate_answer(messages, histroy)

    def _build_system_prompt(self) -> str:
        """构建系统提示词 --> 子类可覆写"""
        return "You are a helpful assistant."

    def _parse_json_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """解析 JSON 响应"""
        text = raw_response.strip()
        fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        else:
            fence_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(text)
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass
        return None