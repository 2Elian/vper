"""SQL Revision prompts — 移植自 DeepEye"""

SYNTAX_CHECK_PROMPT = """
Check and fix syntax errors in this SQL query.

Database Schema:
{SCHEMA}

Question: {QUESTION}

SQL to check:
```sql
{SQL}
```

If the SQL has syntax errors, fix them and output the corrected SQL inside <r> tags.
If the SQL is correct, output it unchanged inside <r> tags.

<r>
SELECT ...
</r>
"""

JOIN_CHECK_PROMPT = """
Check and fix JOIN clauses in this SQL query.

Database Schema (with foreign keys):
{SCHEMA}

SQL to check:
```sql
{SQL}
```

If JOINs are missing or incorrect, add/fix them and output the corrected SQL in <r>.
If JOINs are correct, output unchanged in <r>.

<r>
SELECT ...
</r>
"""

MAX_MIN_CHECK_PROMPT = """
Check MAX/MIN function usage in this SQL query.
If MAX/MIN should be used with GROUP BY or subquery, fix it.

SQL: {SQL}
Question: {QUESTION}

<r>
SELECT ...
</r>
"""

ORDER_BY_CHECK_PROMPT = """
Check ORDER BY and LIMIT in this SQL query.
If ORDER BY is missing when LIMIT is used, or vice versa, fix it.

SQL: {SQL}
Question: {QUESTION}

<r>
SELECT ...
</r>
"""

TIME_CHECK_PROMPT = """
Check date/time formatting in this SQL query.
Ensure date strings use ISO format: YYYY-MM-DD.

SQL: {SQL}

<r>
SELECT ...
</r>
"""

SELECT_CHECK_PROMPT = """
Check SELECT clause in this SQL query.
Ensure all selected columns exist in the schema and are used correctly.

Schema: {SCHEMA}
SQL: {SQL}

<r>
SELECT ...
</r>
"""

ORDER_BY_NULL_CHECK_PROMPT = """
Check if ORDER BY should use NULLS LAST or defaults in this SQL.

SQL: {SQL}

<r>
SELECT ...
</r>
"""
