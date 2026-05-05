"""
OpenJudge 适配层

将 OpenJudge 的异步接口适配到项目的同步调用模式。
提供统一的 OJModel 适配器，让 OpenJudge 的评估器可以直接使用项目的 LLM 配置。
"""

import asyncio
from typing import Any, Dict, Optional

from openjudge.models.openai_chat_model import OpenAIChatModel


def create_oj_model(model_config):
    """
    从项目的 model_config 创建 OpenJudge 的 OpenAIChatModel

    Args:
        model_config: 项目的模型配置 dict
            - model: 模型名称
            - api_key: API key
            - api_base: API base URL
            - temperature: 温度

    Returns:
        OpenAIChatModel 实例
    """
    return OpenAIChatModel(
        model=model_config.get("model", "gpt-4.1-mini"),
        api_key=model_config.get("api_key", ""),
        base_url=model_config.get("api_base", "https://api.openai.com/v1"),
    )


def run_async(coro):
    """
    同步运行异步协程

    OpenJudge 的评估器全部是异步的 (aevaluate)，
    我们的项目是同步的，需要用 asyncio.run() 桥接。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
