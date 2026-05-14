"""DC (Divide-and-Conquer) SQL Generator — 移植自 DeepEye"""

from __future__ import annotations

import logging
import re
from typing import Any

from data_agent_baseline.agents.llm_extractor import LLMExtractor
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.nl2sql.sql_generation.prompt import DC_SQL_GENERATION_PROMPT
from data_agent_baseline.nl2sql.utils.schema_utils import get_database_schema_profile

logger = logging.getLogger(__name__)


class DCGenerator:
    """Divide-Conquer SQL生成器 — 递归分而治之将复杂问题拆分为子问题"""

    def __init__(self, fix_end_token: bool = True):
        self.fix_end_token = fix_end_token
        self.extractor = LLMExtractor()

    def generate(
        self,
        question: str,
        metadata: dict,
        model: ModelAdapter,
        sampling_budget: int = 1,
        **kwargs
    ) -> tuple[list[str], dict[str, int]]:
        """使用DC方法生成SQL"""
        if sampling_budget == 0:
            return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        database_schema_profile = get_database_schema_profile(metadata)

        prompt = DC_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema_profile,
            QUESTION=question
        ).strip()

        def parse_sql(response: str) -> str | None:
            return self._parse_llm_response(response)

        all_sql_candidates, total_token_usage = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            n=sampling_budget,
            fix_end_token=self.fix_end_token,
            rule_parser=parse_sql,
            end_token="</result>",
        )

        return all_sql_candidates, total_token_usage

    @staticmethod
    def _parse_llm_response(response: str) -> str | None:
        """Parse the LLM response to extract SQL"""
        try:
            match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if match:
                sql = match.group(1).strip()
            else:
                # Try ```sql fence
                match = re.search(r"```sql\s*(.*?)\s*```", response, re.IGNORECASE | re.DOTALL)
                if match:
                    sql = match.group(1).strip()
                else:
                    # Try standalone SELECT
                    lines = response.strip().split("\n")
                    select_lines = []
                    started = False
                    for line in lines:
                        if line.strip().upper().startswith(("SELECT", "WITH")):
                            started = True
                        if started:
                            select_lines.append(line)
                        if started and line.rstrip().endswith(";"):
                            sql = "\n".join(select_lines).rstrip(";").strip()
                            break
                    else:
                        if started:
                            sql = "\n".join(select_lines).rstrip(";").strip()
                        else:
                            return None

            # Validate
            sql_upper = sql.strip().upper()
            if sql_upper.startswith("SELECT") or sql_upper.startswith("WITH"):
                return sql.strip()
            return None

        except Exception as e:
            logger.warning("Error parsing DC generation response: %s", e)
            return None


class SkeletonGenerator:
    """Skeleton SQL生成器 — 先确定骨架再填入具体表名/列名"""

    def __init__(self, fix_end_token: bool = True):
        self.fix_end_token = fix_end_token
        self.extractor = LLMExtractor()

    def generate(
        self,
        question: str,
        metadata: dict,
        model: ModelAdapter,
        sampling_budget: int = 1,
        **kwargs
    ) -> tuple[list[str], dict[str, int]]:
        """使用Skeleton方法生成SQL"""
        if sampling_budget == 0:
            return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        from data_agent_baseline.nl2sql.sql_generation.prompt import SKELETON_SQL_GENERATION_PROMPT

        database_schema_profile = get_database_schema_profile(metadata)

        prompt = SKELETON_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema_profile,
            QUESTION=question
        ).strip()

        # Reuse DC's SQL parser
        dc_parser = DCGenerator(fix_end_token=False)

        def parse_sql(response: str) -> str | None:
            return dc_parser._parse_llm_response(response)

        all_sql_candidates, total_token_usage = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            n=sampling_budget,
            fix_end_token=self.fix_end_token,
            rule_parser=parse_sql,
            end_token="</result>",
        )

        return all_sql_candidates, total_token_usage
