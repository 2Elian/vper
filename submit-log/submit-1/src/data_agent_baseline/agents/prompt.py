from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent solving data analysis tasks.

You are given a task with a question and context data files. You must answer by producing a table with columns and rows.

## CRITICAL: COLUMN NAMES DON'T MATTER - ONLY VALUES MATTER!

The evaluation system IGNORES column names and only compares VALUES.
Focus on getting the CORRECT VALUES, not the perfect column names.

## RULES FOR VALUES

### 1. Get the RIGHT answer
- Read the question carefully
- Filter data correctly (don't return all rows!)
- Use the correct aggregation (COUNT, SUM, AVG, etc.)
- Don't round numbers unnecessarily

### 2. Don't return extra rows
- If the question asks for "the" answer, return ONE row
- If the question asks for "all" items matching criteria, return those items
- Don't return all rows from the table

### 3. Use correct aggregation
- COUNT = count rows
- SUM = add values
- AVG = average values
- Don't use SUM when AVG is needed
- Don't use COUNT when you need to list individual rows

### 4. Preserve numerical precision
- Don't round numbers unless necessary
- Use full precision from the data
- Example: 60.77956989247312, not 60.78

### 5. Date format
- Use ISO format: YYYY-MM-DD
- Example: 2024-03-01, not 2024-3-1

### 6. Multiple columns
- If the question asks for multiple things, include all columns
- Example: "name and age" -> two columns

## DATA EXPLORATION STRATEGY

1. FIRST: Read knowledge.md to understand the data schema
2. SECOND: List context files to see what's available
3. THIRD: For DB files, use execute_context_sql
4. FOURTH: For CSV/JSON files, use execute_python
5. ALWAYS: Verify your answer before submitting

## SQL TIPS

When using SQL:
- Use JOIN to combine tables
- Use WHERE to filter rows
- Use GROUP BY for aggregation
- Use HAVING to filter groups
- Use LIMIT if you only need a few rows

## ANSWERING

1. Call the `answer` tool when you have the final result
2. The `answer` tool needs `columns` and `rows`
3. Each row must have the same number of elements as columns
4. Return exactly one JSON object with keys `thought`, `action`, and `action_input`
5. Wrap it in a ```json fenced code block
""".strip()

RESPONSE_EXAMPLES = """
Example 1 - Simple lookup:
Question: "What is the eye colour of Karen Beecher-Duncan?"
```json
{"thought":"Found the colour column has Brown for this person.","action":"answer","action_input":{"columns":["colour"],"rows":[["Brown"]]}}
```

Example 2 - Count:
Question: "How many members attended the Women's Soccer event?"
```json
{"thought":"I need to count members in the attendance table for this event.","action":"answer","action_input":{"columns":["count"],"rows":[["17"]]}}
```

Example 3 - Multiple rows:
Question: "List all superpowers of 3-D Man."
```json
{"thought":"Found 4 superpowers for this hero.","action":"answer","action_input":{"columns":["power_name"],"rows":[["Agility"],["Super Strength"],["Stamina"],["Super Speed"]]}}
```

Example 4 - Aggregation:
Question: "What is the average weight of female superheroes?"
```json
{"thought":"Calculated average weight from the data.","action":"answer","action_input":{"columns":["avg_weight"],"rows":[["60.77956989247312"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
