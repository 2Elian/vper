"""SQL Reviser — 整合所有 checker，统一修正接口（移植自 DeepEye）"""

from __future__ import annotations

import logging
import re
from typing import Any

from data_agent_baseline.agents.llm_extractor import LLMExtractor
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

logger = logging.getLogger(__name__)


def _extract_sql_from_response(response: str) -> str | None:
    """从 checker 响应中提取修正后的 SQL"""
    match = re.search(r"<r>(.*?)</r>", response, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Remove ```sql fences if present inside <r>
        fence_match = re.search(r"```sql\s*(.*?)\s*```", sql, re.IGNORECASE | re.DOTALL)
        if fence_match:
            sql = fence_match.group(1).strip()
        return sql
    return None


class BaseChecker:
    """Checker 基类"""
    def __init__(self, fix_end_token: bool = True):
        self.fix_end_token = fix_end_token
        self.extractor = LLMExtractor()

    async def check_and_revise(self, *args, **kwargs):
        # Sync wrapper
        return self.check_and_revise_sync(*args, **kwargs)


class SyntaxChecker(BaseChecker):
    """语法检查 — 最重要，放第一位"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        from data_agent_baseline.nl2sql.utils.schema_utils import get_database_schema_profile
        from data_agent_baseline.nl2sql.sql_revision.prompt import SYNTAX_CHECK_PROMPT

        schema = get_database_schema_profile(metadata)
        prompt = SYNTAX_CHECK_PROMPT.format(SCHEMA=schema, QUESTION=question, SQL=sql)

        results, tokens = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=_extract_sql_from_response,
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=1,
        )
        return (results[0] if results else sql), tokens


class JoinChecker(BaseChecker):
    """JOIN 语法检查"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        from data_agent_baseline.nl2sql.utils.schema_utils import get_database_schema_profile
        from data_agent_baseline.nl2sql.sql_revision.prompt import JOIN_CHECK_PROMPT

        schema = get_database_schema_profile(metadata)
        prompt = JOIN_CHECK_PROMPT.format(SCHEMA=schema, SQL=sql)

        results, tokens = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=_extract_sql_from_response,
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=1,
        )
        return (results[0] if results else sql), tokens


class MaxMinChecker(BaseChecker):
    """MAX/MIN 函数检查"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        from data_agent_baseline.nl2sql.sql_revision.prompt import MAX_MIN_CHECK_PROMPT

        prompt = MAX_MIN_CHECK_PROMPT.format(SQL=sql, QUESTION=question)

        results, tokens = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=_extract_sql_from_response,
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=1,
        )
        return (results[0] if results else sql), tokens


class OrderByLimitChecker(BaseChecker):
    """ORDER BY LIMIT 检查"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        from data_agent_baseline.nl2sql.sql_revision.prompt import ORDER_BY_CHECK_PROMPT

        prompt = ORDER_BY_CHECK_PROMPT.format(SQL=sql, QUESTION=question)

        results, tokens = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=_extract_sql_from_response,
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=1,
        )
        return (results[0] if results else sql), tokens


class TimeChecker(BaseChecker):
    """时间格式检查（无需 LLM，规则匹配）"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        # 纯规则检查：修复常见日期格式问题
        import re as _re
        revised = sql
        # Fix YYYY-M-D to YYYY-MM-DD where obvious
        revised = _re.sub(r"(\d{4})-(\d{1})-(?=\d{1}\D)", r"\1-0\2-", revised)
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return revised, tokens


class SelectChecker(BaseChecker):
    """SELECT 语句检查"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        from data_agent_baseline.nl2sql.utils.schema_utils import get_database_schema_profile
        from data_agent_baseline.nl2sql.sql_revision.prompt import SELECT_CHECK_PROMPT

        schema = get_database_schema_profile(metadata)
        prompt = SELECT_CHECK_PROMPT.format(SCHEMA=schema, SQL=sql)

        results, tokens = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=_extract_sql_from_response,
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=1,
        )
        return (results[0] if results else sql), tokens


class OrderByNullChecker(BaseChecker):
    """ORDER BY NULL 检查"""

    def check_and_revise_sync(
        self, sql: str, metadata: dict, model: ModelAdapter,
        database_path: str = "", question: str = "", evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        from data_agent_baseline.nl2sql.sql_revision.prompt import ORDER_BY_NULL_CHECK_PROMPT

        prompt = ORDER_BY_NULL_CHECK_PROMPT.format(SQL=sql)

        results, tokens = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=_extract_sql_from_response,
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=1,
        )
        return (results[0] if results else sql), tokens


# ============================================================
# SQLReviser — 组合所有 checker
# ============================================================

class SQLReviser:
    """SQL修正器：组合多个checker来修正SQL

    默认checker顺序：
    1. SyntaxChecker - 语法检查（最重要）
    2. JoinChecker - JOIN语法检查
    3. MaxMinChecker - MAX/MIN函数检查
    4. OrderByLimitChecker - ORDER BY LIMIT检查
    5. TimeChecker - 时间格式检查（无需LLM）
    6. SelectChecker - SELECT语句检查
    7. OrderByNullChecker - ORDER BY NULL检查
    """

    def __init__(self, fix_end_token: bool = True, custom_checkers: list[BaseChecker] | None = None):
        self.fix_end_token = fix_end_token

        if custom_checkers is not None:
            self.checkers = custom_checkers
        else:
            self.checkers: list[BaseChecker] = [
                SyntaxChecker(fix_end_token),
                JoinChecker(fix_end_token),
                MaxMinChecker(fix_end_token),
                OrderByLimitChecker(fix_end_token),
                TimeChecker(fix_end_token),
                SelectChecker(fix_end_token),
                OrderByNullChecker(fix_end_token),
            ]

    def revise(
        self,
        sql: str,
        metadata: dict,
        model: ModelAdapter,
        database_path: str = "",
        question: str = "",
        evidence: str = ""
    ) -> tuple[str, dict[str, int]]:
        """依次应用所有checker修正SQL"""
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        current_sql = sql

        for checker in self.checkers:
            checker_name = checker.__class__.__name__

            try:
                revised_sql, tokens = checker.check_and_revise_sync(
                    current_sql, metadata, model, database_path, question, evidence
                )

                for key in total_tokens:
                    total_tokens[key] += tokens.get(key, 0)

                if revised_sql != current_sql:
                    logger.debug("[%s] SQL revised", checker_name)
                    current_sql = revised_sql

            except Exception as e:
                logger.error("[%s] Error during check: %s", checker_name, e)

        return current_sql, total_tokens

    def add_checker(self, checker: BaseChecker, position: int | None = None):
        if position is None:
            self.checkers.append(checker)
        else:
            self.checkers.insert(position, checker)

    def remove_checker(self, checker_class: type) -> bool:
        for i, checker in enumerate(self.checkers):
            if isinstance(checker, checker_class):
                self.checkers.pop(i)
                return True
        return False

    def clear_checkers(self):
        self.checkers = []
