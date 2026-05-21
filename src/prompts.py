"""
SWE-SQL+ Prompt Templates
=========================
All prompt templates for the 4-stage pipeline:
- Stage 1: CoT single-path reasoning
- Stage 2: N-path (diagnostic / rewrite / minimal_fix)
- Stage 3: SQL-ACT agent (EXPLORE / FIX)
"""

# ─── Stage 1: CoT ────────────────────────────────────────────────────────────

COT_SYSTEM = (
    "You are an expert PostgreSQL database engineer tasked with debugging a faulty SQL query."
)

COT_USER_TEMPLATE = """\
Fix the faulty SQL query below so it passes all test cases.

Think briefly about what is wrong, then output the corrected SQL.

## Database Schema:
{schema}

## User Issue:
{query}

## Faulty SQL:
```sql
{issue_sql}
```

Output your corrected SQL in ```sql ... ``` tags.
For Management tasks (trigger/function/procedure/index), output ALL required SQL statements
in the correct execution order, each in its own ```sql ... ``` block."""


def cot_messages(schema: str, query: str, issue_sql: str) -> list[dict]:
    return [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": COT_USER_TEMPLATE.format(
            schema=schema, query=query, issue_sql=issue_sql,
        )},
    ]


# ─── Stage 2: N-path ─────────────────────────────────────────────────────────

DIAGNOSTIC_SYSTEM = (
    "You are a senior DBA performing systematic SQL debugging. "
    "Your approach: First classify the error type precisely, then trace the root cause "
    "step by step, then apply a targeted fix. Do NOT jump to solutions before diagnosing. "
    "Always consider: join conditions, aggregation logic, subquery structure, "
    "data type compatibility, and business semantic correctness."
)

REWRITE_SYSTEM = (
    "You are an expert SQL engineer who rewrites SQL from scratch. "
    "IGNORE the structure of the original faulty SQL entirely. "
    "Read only the user's intent and the database schema, then write a correct SQL "
    "from first principles. The faulty SQL may have led you in the wrong direction — "
    "treat it only as a hint about what the user wants, not how to achieve it."
)

MINIMAL_FIX_SYSTEM = (
    "You are a careful SQL debugger who makes minimal changes. "
    "Preserve as much of the original SQL structure as possible. "
    "Only change what is strictly necessary to fix the specific error. "
    "Do NOT refactor, restructure, or optimize beyond what is needed."
)

NPATH_USER_TEMPLATE = """\
Fix the faulty SQL below. Output the corrected SQL wrapped in ```sql ... ``` tags.

## Database Schema:
{schema}

## User Issue:
{query}

## Faulty SQL:
```sql
{issue_sql}
```"""


def diagnostic_messages(schema: str, query: str, issue_sql: str) -> list[dict]:
    return [
        {"role": "system", "content": DIAGNOSTIC_SYSTEM},
        {"role": "user", "content": NPATH_USER_TEMPLATE.format(
            schema=schema, query=query, issue_sql=issue_sql,
        )},
    ]


def rewrite_messages(schema: str, query: str, issue_sql: str) -> list[dict]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": NPATH_USER_TEMPLATE.format(
            schema=schema, query=query, issue_sql=issue_sql,
        )},
    ]


def minimal_fix_messages(schema: str, query: str, issue_sql: str) -> list[dict]:
    return [
        {"role": "system", "content": MINIMAL_FIX_SYSTEM},
        {"role": "user", "content": NPATH_USER_TEMPLATE.format(
            schema=schema, query=query, issue_sql=issue_sql,
        )},
    ]


NPATH_BUILDERS = [diagnostic_messages, rewrite_messages, minimal_fix_messages]
NPATH_NAMES = ["diagnostic", "rewrite", "minimal_fix"]


# ─── Stage 3: SQL-ACT Agent ──────────────────────────────────────────────────

SQLACT_SYSTEM = """\
You are an expert PostgreSQL debugger operating in an interactive environment.

## Two actions available:

**[EXPLORE]** — Execute any SQL to observe results (auto-rolled back, safe for DML/DDL):
[EXPLORE]
```sql
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'your_table' ORDER BY ordinal_position;
```

**[FIX]** — Submit your final answer (executed for real, evaluated by test cases):
[FIX]
```sql
SELECT account_id, SUM(amount) FROM loan
WHERE date_issued >= NOW() - INTERVAL '48 hours'
GROUP BY account_id;
```

## Rules (strictly follow):
1. [EXPLORE] and [FIX] must appear BEFORE the ```sql block, NOT inside it
2. After 2 or more EXPLORE turns, you MUST submit [FIX] next
3. For Management tasks (CREATE TRIGGER/FUNCTION/PROCEDURE/INDEX):
   - Use at most 1 EXPLORE to check schema
   - Then iterate quickly via [FIX] → observe test feedback → [FIX] again
4. If test feedback shows "returned 0 but expected 1", your result set is wrong
   — use [EXPLORE] to query actual data and compare with the expected logic"""

SQLACT_INITIAL_TEMPLATE = """\
## Current state:
The best SQL candidate so far (from N-path generation):
```sql
{best_sql}
```

## Failure information:
{failure_detail}

## Database Schema:
{schema}

## User Issue:
{query}

## Original Faulty SQL:
```sql
{issue_sql}
```

Now diagnose the remaining issue and fix it. You may explore the database first if needed."""


def sqlact_initial_messages(
    schema: str,
    query: str,
    issue_sql: str,
    best_sql: str,
    failure_detail: str,
) -> list[dict]:
    return [
        {"role": "system", "content": SQLACT_SYSTEM},
        {"role": "user", "content": SQLACT_INITIAL_TEMPLATE.format(
            schema=schema,
            query=query,
            issue_sql=issue_sql,
            best_sql=best_sql,
            failure_detail=failure_detail,
        )},
    ]


# ─── Stage 3: Feedback Templates ─────────────────────────────────────────────

EXPLORE_OBSERVATION_TEMPLATE = """\
## Query Result:
{observation}

Continue debugging."""

EXEC_ERROR_FEEDBACK_TEMPLATE = """\
Your SQL raised an error during execution:
Error: {error}

Please re-analyze the SQL and submit a fix.
You may use [EXPLORE] to investigate first if needed."""

LOGIC_FAIL_FEEDBACK_TEMPLATE = """\
Your SQL executed but failed the business logic test:

{test_feedback}

This is a logic/semantic error (not syntax). Your SQL ran without errors but
produced wrong results.

Debugging approach:
- Use [EXPLORE] to query actual data from the database
- Compare your query's result against what the User Issue requires
- Check: aggregation logic, join conditions, WHERE filters, column names
- For Management tasks: check if the trigger/function actually changed the DB state

Submit a corrected [FIX] when ready."""

URGENCY_WARNING = (
    "\n\n>>> WARNING: You have only {remaining} step(s) left. "
    "Please submit your [FIX] now."
)


# ─── Stage 3: Final SQL Summary (paper SQL-ACT alignment) ────────────────────

FINAL_SQL_SUMMARY_TEMPLATE = """\
You have completed a debugging session with multiple EXPLORE and FIX attempts.
None of your FIX submissions passed all test cases.

Review the COMPLETE debugging trajectory below, then synthesize a final corrected SQL
that incorporates ALL insights gathered during exploration.

## Database Schema:
{schema}

## User Issue:
{query}

## Original Faulty SQL:
```sql
{issue_sql}
```

## Debugging Trajectory:
{trajectory_summary}

Now, based on everything you learned above, output your FINAL corrected SQL.
Think about what each failed attempt got wrong and what the explorations revealed.
Output the SQL in ```sql ... ``` tags."""
