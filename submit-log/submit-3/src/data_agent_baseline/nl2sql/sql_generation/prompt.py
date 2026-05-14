"""SQL Generation prompts — 移植自 DeepEye"""

DC_SQL_GENERATION_PROMPT = """
# Task: SQL Generation via Divide-and-Conquer

You are an expert SQL developer. Use the Divide-and-Conquer approach to generate a SQL query.

## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Instructions:
1. Break down the question into sub-problems
2. For each sub-problem, determine the SQL fragment needed
3. Combine the fragments into a complete SQL query
4. Only output the final SQL query inside a <result> tag

Output format:
<result>
SELECT ...
</result>
"""

SKELETON_SQL_GENERATION_PROMPT = """
# Task: SQL Generation via Skeleton Approach

You are an expert SQL developer. Generate a SQL query using the skeleton approach.

## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Instructions:
1. First, identify the SQL skeleton (SELECT ... FROM ... WHERE ... GROUP BY ... ORDER BY ...)
2. Then fill in the actual table/column names from the schema
3. Output the complete SQL query

Output format:
<result>
SELECT ...
</result>
"""

ICL_SQL_GENERATION_PROMPT = """
# Task: SQL Generation via In-Context Learning

You are an expert SQL developer. Generate a SQL query for the given question.

## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

{few_shot_examples}

Output format:
<result>
SELECT ...
</result>
"""
