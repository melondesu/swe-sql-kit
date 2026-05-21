"""
Explore Executor — SAVEPOINT-based safe SQL exploration
=======================================================
Executes arbitrary SQL inside SAVEPOINT + ROLLBACK so the agent can observe
results (including DML/DDL effects) without polluting database state.
"""

import psycopg2


def format_table(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as a readable text table."""
    if not columns:
        return "(no columns)"
    if not rows:
        return " | ".join(columns) + "\n(0 rows)"

    # Convert all values to strings
    str_rows = [[str(v) if v is not None else "NULL" for v in row] for row in rows]
    # Calculate column widths
    widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(columns)]
    # Build header
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    sep = "-+-".join("-" * w for w in widths)
    # Build rows
    body = "\n".join(
        " | ".join(val.ljust(w) for val, w in zip(row, widths))
        for row in str_rows
    )
    return f"{header}\n{sep}\n{body}"


def execute_explore_sql_with_rollback(
    sql: str,
    conn,
    timeout_sec: int = 5,
) -> str:
    """
    Execute arbitrary SQL inside a SAVEPOINT, capture the result, then rollback.

    The agent sees the execution effect (result set, affected rows, or error)
    but the database state is restored to pre-execution — DML/DDL side effects
    are undone. This matches the SQL-ACT paper's "arbitrary SQL as action" design.

    Args:
        sql: The SQL to execute (any type — SELECT, INSERT, CREATE, etc.)
        conn: An open psycopg2 connection (autocommit must be False)
        timeout_sec: Statement timeout in seconds

    Returns:
        A string observation describing the result.
    """
    cur = conn.cursor()
    try:
        cur.execute("SAVEPOINT explore_sp")
        cur.execute(f"SET LOCAL statement_timeout = '{timeout_sec}s'")
        cur.execute(sql)

        if cur.description:
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(20)
            observation = format_table(columns, rows)
            total = cur.rowcount
            if total > 20:
                observation += f"\n... ({total} total rows, showing first 20)"
        else:
            observation = f"Execution successful. Rows affected: {cur.rowcount}"

    except psycopg2.errors.QueryCanceled:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT explore_sp")
        except Exception:
            conn.rollback()
        return f"Query timeout (exceeded {timeout_sec}s limit)"
    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT explore_sp")
        except Exception:
            conn.rollback()
        return f"Error: {type(e).__name__}: {str(e)}"
    else:
        cur.execute("ROLLBACK TO SAVEPOINT explore_sp")
        return observation
    finally:
        try:
            cur.execute("RELEASE SAVEPOINT explore_sp")
        except Exception:
            pass
        cur.close()
