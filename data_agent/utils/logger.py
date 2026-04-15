#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Elian
@github: https://github.com/2Elian
"""

import sys
import logging
from typing import Any

# Try loguru first, fallback to standard logging
try:
    from loguru import logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False
    logger = logging.getLogger("dwa")


def get_logger(name):
    if _HAS_LOGURU:
        return logger.bind(module=name)
    else:
        lg = logging.getLogger("dwa.{}".format(name))
        if not lg.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            ))
            lg.addHandler(handler)
            lg.setLevel(logging.INFO)
        return lg


def setup_logging(level="INFO", serialize=False):
    if _HAS_LOGURU:
        logger.remove()
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[module]}</cyan> | "
                "<level>{message}</level>"
            ),
            filter=lambda record: "module" in record["extra"],
            serialize=serialize,
        )
        # 没有 module 标记的日志走默认格式
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<level>{message}</level>"
            ),
            filter=lambda record: "module" not in record["extra"],
            serialize=serialize,
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            stream=sys.stderr,
        )


class NodeLoggerMixin:
    def _get_logger(self):
        name = getattr(self, "name", self.__class__.__name__)
        return get_logger("node.{}".format(name))

    def log_input(self, state, keys=None):
        """记录节点输入"""
        lg = self._get_logger()
        if keys:
            summary = {k: state.get(k, "<missing>") for k in keys}
            lg.info(f"  >> 输入: {summary}")
        else:
            lg.info(f"  >> 输入字段: {list(state.keys())}")

    def log_output(self, result):
        """记录节点输出"""
        lg = self._get_logger()
        # 简要摘要输出内容
        if isinstance(result, dict):
            parts = []
            for k, v in result.items():
                if isinstance(v, dict):
                    top_keys = list(v.keys())[:5]
                    parts.append(f"{k}({len(v)} keys): {top_keys}")
                elif isinstance(v, list):
                    parts.append(f"{k}: [{len(v)} items]")
                else:
                    snippet = str(v)
                    if len(snippet) > 80:
                        snippet = snippet[:80] + "..."
                    parts.append(f"{k}: {snippet}")
            lg.info(f"  << 输出: {parts}")
        else:
            lg.info(f"  << 输出: {type(result).__name__}")

    def log_error(self, error):
        """记录节点异常"""
        lg = self._get_logger()
        lg.error(f"  !! 异常: {type(error).__name__}: {error}")
