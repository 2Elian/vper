from __future__ import annotations

import abc
import re
from typing import Any, List, Optional, Union, Tuple, Dict

from ..tokenizer import BaseTokenizer, Token


class BaseLLMClient(abc.ABC):
    """
    LLM client base class, agnostic to specific backends (OpenAI / Ollama / ...).
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 32768,
        repetition_penalty: float = 1.05,
        top_p: float = 0.95,
        top_k: int = 50,
        tokenizer: Optional[BaseTokenizer] = None,
        **kwargs: Any,
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.tokenizer = tokenizer

        for k, v in kwargs.items():
            setattr(self, k, v)

    # TODO 接受message改造
    @abc.abstractmethod
    async def generate_answer(
        self, current_messages: List[Dict[str, str]], history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """async Generate answer from the model."""
        raise NotImplementedError

    # TODO 接受message改造
    @abc.abstractmethod
    def sync_generate_answer(
        self, current_messages: List[Dict[str, str]], history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """sync Generate answer from the model."""
        raise NotImplementedError

    @abc.abstractmethod
    async def generate_topk_per_token(
        self, current_messages: List[Dict[str, str]], history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> List[Token]:
        """Generate top-k tokens for the next token prediction."""
        raise NotImplementedError

    @abc.abstractmethod
    def sync_generate_topk_per_token(
        self, current_messages: List[Dict[str, str]], history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> List[Token]:
        """sync Generate top-k tokens for the next token prediction."""
        raise NotImplementedError

    @abc.abstractmethod
    async def generate_inputs_prob(
        self, current_messages: List[Dict[str, str]], history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> List[Token]:
        """Generate probabilities for each token in the input."""
        raise NotImplementedError

    @abc.abstractmethod
    def sync_generate_inputs_prob(
        self, current_messages: List[Dict[str, str]], history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> List[Token]:
        """sync Generate probabilities for each token in the input."""
        raise NotImplementedError

    @staticmethod
    def filter_think_tags(text: str, think_tag: str = "think") -> str:
        """
        Remove <think> tags from the text.
        If the text contains <think> and </think>, it removes everything between them and the tags themselves.
        """
        think_pattern = re.compile(rf"<{think_tag}>.*?</{think_tag}>", re.DOTALL)
        filtered_text = think_pattern.sub("", text).strip()
        return filtered_text if filtered_text else text.strip()
