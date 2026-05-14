"""Direct Schema Linker — 移植自 DeepEye"""

from __future__ import annotations

import logging
import re
from typing import Any

from data_agent_baseline.agents.llm_extractor import LLMExtractor
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.nl2sql.schema_linker.prompt import DIRECT_LINKING_PROMPT
from data_agent_baseline.nl2sql.utils.schema_utils import (
    get_database_schema_profile,
    map_lower_column_name_to_original,
    map_lower_table_name_to_original,
    merge_schema_linking_results,
)

logger = logging.getLogger(__name__)


class DirectSchemaLinker:
    """Direct schema linker — uses LLM to directly identify relevant tables and columns."""

    def __init__(self, fix_end_token: bool = True):
        self.fix_end_token = fix_end_token
        self.extractor = LLMExtractor()

    def link(
        self,
        question: str,
        metadata: dict,
        model: ModelAdapter,
        evidence: str = "",
        sampling_budget: int = 1,
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Link question to schema using direct LLM prompting."""
        if sampling_budget == 0:
            return {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        database_schema_profile = get_database_schema_profile(metadata)

        prompt = DIRECT_LINKING_PROMPT.format(
            DATABASE_SCHEMA=database_schema_profile,
            QUESTION=question,
            HINT=evidence or "No additional hints provided."
        ).strip()

        def rule_parser(response: str) -> dict[str, list[str]] | None:
            return self._parse_llm_response(response, metadata)

        all_selections, total_token_usage = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=rule_parser,
            parser_kwargs={"metadata": metadata},
            fix_end_token=self.fix_end_token,
            end_token="</r>",
            n=sampling_budget,
        )

        merged_result = merge_schema_linking_results(all_selections)
        return merged_result, total_token_usage

    def _parse_llm_response(
        self,
        response: str,
        metadata: dict,
    ) -> dict[str, list[str]] | None:
        try:
            answer_match = re.search(r"<r>(.*?)</r>", response, re.DOTALL)
            if not answer_match:
                answer_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)

            if not answer_match:
                logger.warning("No result tag found in LLM response")
                return None

            answer_content = answer_match.group(1).strip()
            result: dict[str, list[str]] = {}

            table_matches = re.findall(
                r'<table\s+table_name="([^"]+)"[^>]*>(.*?)</table>',
                answer_content, re.DOTALL,
            )

            for table_name, table_content in table_matches:
                original_table_name = map_lower_table_name_to_original(
                    table_name.lower(), metadata
                )
                if original_table_name is None:
                    logger.debug("Table not found in schema: %s", table_name)
                    continue

                result[original_table_name] = []

                column_matches = re.findall(
                    r'<column\s+column_name="([^"]+)"[^>]*/?>',
                    table_content,
                )

                for column_name in column_matches:
                    original_column_name = map_lower_column_name_to_original(
                        original_table_name, column_name.lower(), metadata
                    )
                    if original_column_name is None:
                        logger.debug("Column not found: %s in table %s", column_name, original_table_name)
                        continue
                    result[original_table_name].append(original_column_name)

            if result:
                logger.debug("Successfully parsed selection: %d tables selected", len(result))
                return result
            else:
                logger.warning("No valid table-column selections found")
                return None

        except Exception as e:
            logger.warning("Error parsing LLM response: %s", e)
            return None


class ValueLinker:
    """Value-based schema linker — uses retrieved values to guide schema linking."""

    def __init__(self, value_distance_threshold: float = 0.3):
        self.value_distance_threshold = value_distance_threshold
        self.extractor = LLMExtractor()

    def link(
        self,
        question: str,
        metadata: dict,
        model: ModelAdapter,
        retrieved_values: dict | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Link schema using value retrieval results."""
        if not retrieved_values:
            return {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        database_schema_profile = get_database_schema_profile(metadata)

        # Format value matching results
        value_matching_lines = []
        for table_name, columns in retrieved_values.items():
            for col_name, matches in columns.items():
                filtered = [m for m in matches if m.get("distance", 1) <= self.value_distance_threshold]
                if filtered:
                    values = [m["value"] for m in filtered[:5]]
                    value_matching_lines.append(f"{table_name}.{col_name}: {values}")

        value_matching_str = "\n".join(value_matching_lines) if value_matching_lines else "No values matched."

        from data_agent_baseline.nl2sql.schema_linker.prompt import VALUE_LINKING_PROMPT

        prompt = VALUE_LINKING_PROMPT.format(
            VALUE_MATCHING=value_matching_str,
            DATABASE_SCHEMA=database_schema_profile,
            QUESTION=question,
            HINT=""
        ).strip()

        # Reuse DirectSchemaLinker's parser since output format is the same
        direct_parser = DirectSchemaLinker(fix_end_token=True)

        def rule_parser(response: str) -> dict[str, list[str]] | None:
            return direct_parser._parse_llm_response(response, metadata)

        all_selections, total_token_usage = self.extractor.extract_with_retry(
            model=model,
            messages=[ModelMessage(role="user", content=prompt)],
            rule_parser=rule_parser,
            parser_kwargs={"metadata": metadata},
            fix_end_token=True,
            end_token="</r>",
            n=1,
        )

        merged = merge_schema_linking_results(all_selections)
        return merged, total_token_usage
