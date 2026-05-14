"""Schema Linker prompts — 移植自 DeepEye"""

DIRECT_LINKING_PROMPT = """
# Task: Database Entity Selection

You are performing schema linking for NL2SQL. Your task is to select the relevant tables and columns from the database schema based on the provided question.

# Database Schema:
{DATABASE_SCHEMA}

# Guidelines:
- Tables: Select tables that contain data needed to answer the question.
- Columns: Select only columns from the selected tables that are directly needed for filtering, sorting, grouping, and calculating the required result.
- If a column name implies it contains the needed data, include it.

# Question:
{QUESTION}

# Hint:
{HINT}

# Fields:
<r>
[provide your selected tables and columns as XML elements]
</r>
"""

VALUE_LINKING_PROMPT = """
# Task: Database Entity Selection via Value Matching

You are performing schema linking for NL2SQL. Based on the VALUE MATCHING results and the database schema, select the most relevant tables and columns.

# Value Matching Results (Keywords -> DB Values):
{VALUE_MATCHING}

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Hint:
{HINT}

# Fields:
<r>
[output selected tables and columns as XML elements]
</r>
"""
