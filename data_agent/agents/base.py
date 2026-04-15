import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple, List

from data_agent.utils import get_logger

from data_agent.core.types import (
    AgentEvent,
    AgentInput,
    History,
    Session,
)
from data_agent.llms import BaseLLMClient


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
    def run(self, agent_input: AgentInput, session: Session, history: History) -> AgentEvent:
        raise NotImplementedError

    @property
    def logger(self) -> logging.Logger:
        """获取与该 Agent 关联的 logger，自动使用 agent name"""
        return get_logger(self.name)


class ChatModelAgent(Agent):
    def __init__(self, model: BaseLLMClient, tools: Any = None):
        self._model = model
        self._tools = tools

    def _call_model(self, messages: List[Dict[str, str]], histroy: Optional[List[Dict[str, str]]] = None) -> Union[str, Tuple[str, Optional[str]]]:
        """call LLM"""
        return self._model.sync_generate_answer(current_messages=messages, history=histroy)

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