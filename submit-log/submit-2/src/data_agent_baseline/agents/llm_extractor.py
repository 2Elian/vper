"""
LLM Extractor — 鲁棒的 LLM 结构化提取器（移植自 DeepEye）

提供统一的 LLM 调用 + 规则解析 + 失败重试机制。
适配 submit-1 的 ModelAdapter 接口（非 langchain）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

logger = logging.getLogger("LLMExtractor")


class LLMExtractor:
    """从 LLM 响应中提取结构化数据，支持失败重试。

    用法::

        extractor = LLMExtractor(max_retries=3)
        results, token_usage = await extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content="...")],
            rule_parser=my_parse_fn,
            n=1,
        )
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def extract_with_retry(
        self,
        model: ModelAdapter,
        messages: list[ModelMessage],
        rule_parser: Callable[..., Any | None],
        parser_kwargs: dict[str, Any] | None = None,
        fix_end_token: bool = False,
        end_token: str = "",
        n: int = 1,
    ) -> tuple[list[Any], dict[str, int]]:
        """调用 LLM 并解析响应，失败自动重试。

        Args:
            model: 模型适配器
            messages: 对话消息列表
            rule_parser: 解析函数，接收 (response_text, **parser_kwargs)，成功返回解析结果，失败返回 None
            parser_kwargs: 传给 rule_parser 的额外参数
            fix_end_token: 是否在响应末尾补 end_token
            end_token: 要补的结束标签
            n: 目标成功解析数

        Returns:
            (解析结果列表, token_usage 字典)
        """
        parser_kwargs = parser_kwargs or {}
        all_results: list[Any] = []
        # submit-1 的 ModelAdapter 没有返回 token usage，使用估算值
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for _sample_idx in range(n):
            retry_count = 0
            estimated_tokens = 0

            while retry_count < self.max_retries:
                try:
                    raw_response = model.complete(messages)
                    response_text = raw_response

                    if fix_end_token and end_token and end_token not in response_text:
                        response_text = response_text.rstrip() + end_token

                    parsed = rule_parser(response_text, **parser_kwargs)

                    if parsed is not None:
                        all_results.append(parsed)
                        # Rough token estimate
                        estimated_tokens += len(raw_response.split())
                        total_usage["completion_tokens"] += estimated_tokens
                        break

                    logger.warning(
                        "LLMExtractor: parse failed (attempt %d/%d), retrying...",
                        retry_count + 1, self.max_retries,
                    )
                    retry_count += 1

                except Exception:
                    logger.exception(
                        "LLMExtractor: model call failed (attempt %d/%d)",
                        retry_count + 1, self.max_retries,
                    )
                    retry_count += 1

            if retry_count >= self.max_retries:
                logger.warning("LLMExtractor: max retries exhausted, giving up on this sample")

        return all_results, total_usage
