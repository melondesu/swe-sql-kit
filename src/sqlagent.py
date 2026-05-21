"""
SQL-ACT Agent — ReAct-style iterative SQL debugger
===================================================
Implements the SQL-ACT framework from the SWE-SQL paper (Section 5.1):
  - Two action types: [EXPLORE] (observe via SAVEPOINT rollback) and [FIX] (submit for real)
  - Multi-turn conversation history
  - MAX_ITER=7 steps, MAX_FIX_ATTEMPTS=4
  - Final SQL summary step (paper alignment: synthesize from full trajectory)
  - Urgency warnings and fallback logic
"""

import re

from prompts import (
    sqlact_initial_messages,
    EXPLORE_OBSERVATION_TEMPLATE,
    EXEC_ERROR_FEEDBACK_TEMPLATE,
    LOGIC_FAIL_FEEDBACK_TEMPLATE,
    URGENCY_WARNING,
    FINAL_SQL_SUMMARY_TEMPLATE,
)
from explore_executor import execute_explore_sql_with_rollback

MAX_ITER = 7
MAX_FIX_ATTEMPTS = 4


def _format_trajectory_for_summary(trajectory: list[dict]) -> str:
    """Format trajectory into a readable text summary for the Final SQL prompt."""
    parts = []
    for t in trajectory:
        step = t.get("step", "?")
        action = t.get("action_type", "?")
        sql = t.get("sql", "")
        obs = t.get("observation", "")
        thought = t.get("thought", "")

        part = f"--- Step {step} [{action}] ---"
        if thought:
            part += f"\nThought: {thought[:300]}"
        if sql:
            part += f"\nSQL: {sql[:500]}"
        if obs:
            part += f"\nResult: {obs[:500]}"
        parts.append(part)

    return "\n\n".join(parts)


def parse_action(response: str) -> tuple[str, str]:
    """
    Parse model output to extract action type and SQL.
    Strips [EXPLORE]/[FIX] markers from inside SQL blocks.
    """
    response_upper = response.upper()

    if re.search(r'\[?\s*EXPLORE\s*\]?', response_upper):
        action_type = "EXPLORE"
    elif re.search(r'\[?\s*FIX\s*\]?', response_upper):
        action_type = "FIX"
    else:
        action_type = "FIX"

    sql_match = re.search(r'```\s*sql\s*([\s\S]*?)```', response, re.IGNORECASE)
    if not sql_match:
        sql_match = re.search(r'```\s*([\s\S]*?)```', response)
    sql = sql_match.group(1).strip() if sql_match else ""

    sql = re.sub(
        r'^\s*\[?\s*(?:EXPLORE|FIX)\s*\]?:?\s*\n?',
        '', sql, flags=re.IGNORECASE
    ).strip()

    return action_type, sql


def extract_thought(response: str) -> str:
    """Extract reasoning/thought content from <think>...</think> tags or before action marker.

    Priority:
    1. If <think>...</think> tags exist, extract content between them (full reasoning)
    2. Otherwise, extract text before [EXPLORE] or [FIX] marker (partial reasoning)
    """
    # Priority 1: Extract from <think>...</think> tags (most accurate)
    think_match = re.search(r'<think>([\s\S]*?)</think>', response, re.IGNORECASE)
    if think_match:
        thought = think_match.group(1).strip()
        # Keep full thinking content, no truncation here (max 5000 chars)
        return thought[:5000] if thought else ""

    # Priority 2: Extract text before [EXPLORE] or [FIX] marker (fallback)
    m = re.search(r'^([\s\S]*?)(?:\[?\s*(?:EXPLORE|FIX)\s*\]?)', response, re.IGNORECASE)
    if m:
        thought = m.group(1).strip()
        # Clean up markdown formatting
        thought = re.sub(r'^#+\s*', '', thought, flags=re.MULTILINE)
        return thought[:2000] if thought else ""

    # Fallback: return first 500 chars if nothing matches
    return response[:500].strip()


def run_sqlact(
    *,
    schema: str,
    query: str,
    issue_sql: str,
    best_sql: str,
    failure_detail: str,
    db_conn_fn,
    call_llm_fn,
    run_test_cases_fn,
    run_test_cases_with_feedback_fn,
    get_execution_error_fn,
    is_management: bool = False,
    reset_db_fn=None,
    max_observation_chars: int = 1500,
) -> dict:
    """
    Run the SQL-ACT agent loop.

    Args:
        schema: Database schema text
        query: User's natural language question
        issue_sql: Original faulty SQL
        best_sql: Best SQL from previous stages
        failure_detail: Description of why previous stages failed
        db_conn_fn: Callable() -> psycopg2 connection for EXPLORE actions
        call_llm_fn: Function(messages) -> str
        run_test_cases_fn: Function(sql, instance, db_name) -> bool
        run_test_cases_with_feedback_fn: Function(sql, instance, db_name) -> (bool, str)
        get_execution_error_fn: Function(sql, db_name) -> str|None
        reset_db_fn: Optional callable to reset DB state after FIX failure

    Returns:
        dict with keys: pass_, sql, iterations, fix_attempts, trajectory, note
    """
    # Build initial conversation
    conversation = sqlact_initial_messages(
        schema=schema,
        query=query,
        issue_sql=issue_sql,
        best_sql=best_sql,
        failure_detail=failure_detail,
    )

    trajectory = []
    fix_attempts = 0
    explore_count = 0
    max_explore = 1 if is_management else 3
    last_sql = best_sql
    result_recorded = False
    final_pass = False
    final_sql = best_sql
    note = ""

    for step in range(MAX_ITER):
        remaining = MAX_ITER - step

        # Last step and never FIXed → force-submit best candidate
        if step == MAX_ITER - 1 and fix_attempts == 0 and last_sql:
            passed, feedback = run_test_cases_with_feedback_fn(last_sql)
            trajectory.append({
                "step": step,
                "action_type": "FIX",
                "sql": last_sql,
                "observation": "test_case: " + ("PASS" if passed else "FAIL (force_final_step)"),
                "thought": "Final step: force-submitting best candidate SQL",
            })
            final_pass = passed
            final_sql = last_sql
            result_recorded = True
            break

        # Urgency warning — trigger earlier and based on explore count
        if remaining <= 2 or (explore_count >= max_explore and fix_attempts == 0):
            conversation[-1]["content"] += (
                f"\n\n>>> WARNING: You have explored {explore_count} time(s) already. "
                f"Only {remaining} step(s) left. "
                f"You MUST submit [FIX] in the NEXT turn. Do NOT use [EXPLORE] again."
            )

        # Call LLM
        response = call_llm_fn(conversation)
        if not response:
            break

        action_type, sql = parse_action(response)
        thought = extract_thought(response)

        # Track the latest SQL seen
        if sql:
            last_sql = sql

        if action_type == "EXPLORE":
            explore_count += 1
            # Management tasks: exceed max_explore → force switch to FIX
            if explore_count > max_explore and sql:
                action_type = "FIX"
                # fall through to FIX handling below

        if action_type == "EXPLORE":
            # Execute exploration SQL with SAVEPOINT rollback
            observation = execute_explore_sql_with_rollback(
                sql=sql,
                conn=db_conn_fn(),
                timeout_sec=5,
            )

            trajectory.append({
                "step": step,
                "action_type": "EXPLORE",
                "sql": sql,
                "observation": observation[:max_observation_chars],
                "thought": thought,
            })

            # Add to conversation history
            conversation.append({"role": "assistant", "content": response})
            conversation.append({
                "role": "user",
                "content": EXPLORE_OBSERVATION_TEMPLATE.format(observation=observation[:max_observation_chars]),
            })

        elif action_type == "FIX":
            fix_attempts += 1

            # Run test cases for real
            passed, feedback = run_test_cases_with_feedback_fn(sql)

            if passed:
                trajectory.append({
                    "step": step,
                    "action_type": "FIX",
                    "sql": sql,
                    "observation": "test_case: PASS",
                    "thought": thought,
                })
                final_pass = True
                final_sql = sql
                result_recorded = True
                break
            else:
                # Check if it's an execution error or a logic error
                exec_err = get_execution_error_fn(sql)

                if exec_err:
                    feedback_content = EXEC_ERROR_FEEDBACK_TEMPLATE.format(
                        error=exec_err[:max_observation_chars],
                    )
                    obs_text = f"Execution error: {exec_err[:300]}"
                else:
                    feedback_content = LOGIC_FAIL_FEEDBACK_TEMPLATE.format(
                        test_feedback=feedback[:max_observation_chars],
                    )
                    obs_text = f"test_case: FAIL\n{feedback[:300]}"

                trajectory.append({
                    "step": step,
                    "action_type": "FIX",
                    "sql": sql,
                    "observation": obs_text,
                    "thought": thought,
                })

                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": feedback_content})

                # Reset DB state after FIX failure to prevent side-effect
                # leakage (e.g. Management test_cases INSERT data that gets
                # committed, causing "duplicate key" on next FIX attempt)
                if reset_db_fn:
                    try:
                        reset_db_fn()
                    except Exception as e:
                        pass  # Best effort — continue even if reset fails

                if fix_attempts >= MAX_FIX_ATTEMPTS:
                    final_sql = sql
                    result_recorded = True
                    break

    # ─── Final SQL Summary (paper SQL-ACT alignment) ──────────────────────
    # When all FIX attempts failed, give the model one last chance to
    # synthesize a final SQL from the complete debugging trajectory.
    # This mirrors the paper's Final SQL Prompt after [DONE].
    if result_recorded and not final_pass and fix_attempts >= 2 and trajectory:
        trajectory_text = _format_trajectory_for_summary(trajectory)
        summary_prompt = FINAL_SQL_SUMMARY_TEMPLATE.format(
            schema=schema,
            query=query,
            issue_sql=issue_sql,
            trajectory_summary=trajectory_text,
        )
        summary_response = call_llm_fn([
            {"role": "system", "content": "You are an expert PostgreSQL debugger. Output only the corrected SQL."},
            {"role": "user", "content": summary_prompt},
        ])
        if summary_response:
            summary_sql = parse_action(summary_response)[1]
            if summary_sql:
                # Reset DB before testing the summary SQL
                if reset_db_fn:
                    try:
                        reset_db_fn()
                    except Exception:
                        pass
                summary_passed, _ = run_test_cases_with_feedback_fn(summary_sql)
                trajectory.append({
                    "step": len(trajectory),
                    "action_type": "FINAL_SUMMARY",
                    "sql": summary_sql,
                    "observation": "test_case: " + ("PASS" if summary_passed else "FAIL"),
                    "thought": "Final SQL summary: synthesized from complete trajectory",
                })
                if summary_passed:
                    final_pass = True
                    final_sql = summary_sql
                    note = "passed_via_final_summary"

    # ─── Fallback logic ───────────────────────────────────────────────────
    # If the agent never submitted a FIX, or exhausted MAX_ITER without recording
    if not result_recorded:
        # Force-submit last_sql
        passed, _ = run_test_cases_with_feedback_fn(last_sql)
        final_pass = passed
        final_sql = last_sql
        if not trajectory or trajectory[-1].get("sql") != last_sql:
            trajectory.append({
                "step": len(trajectory),
                "action_type": "FIX",
                "sql": last_sql,
                "observation": "force_submitted_on_exhaust",
                "thought": "Agent exhausted steps without explicit FIX submission",
            })

    note = note or ""
    if fix_attempts == 0 and not note:
        note = "force_submitted_on_exhaust"

    return {
        "pass_": final_pass,
        "sql": final_sql,
        "iterations": len(trajectory),
        "fix_attempts": fix_attempts,
        "trajectory": trajectory,
        "note": note,
    }
