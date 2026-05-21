#!/usr/bin/env python3
"""
eval_sft_agent.py — 对合并后的 SFT 模型在 val_530.jsonl 上做 SQL-ACT Agent 评测

用法：
    python eval_sft_agent.py --model /root/autodl-tmp/merged_models/B_multi \
                              --output /root/eval_results/B_multi_agent.jsonl

评估流程：
    1. 加载模型（transformers，bfloat16）
    2. 对每条 val_530 先做单轮推理得到 best_sql（与 eval_sft.py 完全一致）
    3. 若 best_sql 已通过 test_case → 直接记录 PASS，跳过 Agent
    4. 否则进入 SQL-ACT Agent 循环（MAX_ITER=7, MAX_FIX_ATTEMPTS=4）
       - [EXPLORE]：SAVEPOINT 安全执行，观察结果后回滚
       - [FIX]：真实执行 + test_case 评测
       - FIX 失败后 reset_database（从 _template 重建）
       - fix_attempts >= 2 且全部失败 → Final SQL Summary
    5. 统计 Success Rate（单轮 + Agent 合并）

所有 DB 工具函数、eval_one_instance、reset_database 均内联（不依赖外部模块）。
SQL-ACT Agent 逻辑完全内联自 sqlagent.py（不 import src/）。
"""

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import psycopg2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── PostgreSQL 连接配置 ────────────────────────────────────────────────────────
PG_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "root",
    "password": "root",
}

# ── Agent 超参数 ───────────────────────────────────────────────────────────────
MAX_ITER = 7
MAX_FIX_ATTEMPTS = 4
MAX_STAGE3_MESSAGES = 8   # 消息压缩阈值（保留 system + first_user + 最近消息）
MAX_OBSERVATION_CHARS = 1500

# ═══════════════════════════════════════════════════════════════════════════════
# 一、PostgreSQL 工具函数（完全对齐官方 postgresql_test_utils.py）
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_results(results):
    """将 date/datetime 转为 yyyy-mm-dd 字符串（与官方完全一致）"""
    processed = []
    for row in results:
        new_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                new_row.append(item.strftime("%Y-%m-%d"))
            else:
                new_row.append(item)
        processed.append(tuple(new_row))
    return processed


def remove_distinct(sql_list):
    """用 \\bDISTINCT\\b 正则去除（与官方完全一致）"""
    cleaned_queries = []
    for query in sql_list:
        cleaned_query = re.sub(r'\bDISTINCT\b', '', query, flags=re.IGNORECASE)
        cleaned_queries.append(cleaned_query)
    return cleaned_queries


def perform_query_on_postgresql_databases(query, db_name, conn=None):
    """
    对齐官方 postgresql_utils.py 的同名函数。
    执行单条 SQL，返回 (result, conn)。
    内置 60s 超时（WITH RECURSIVE 用 15s）。
    """
    MAX_ROWS = 10000
    if conn is None:
        raise ValueError("conn must be provided")

    cursor = conn.cursor()
    upper_query = query.upper()
    try:
        if "WITH RECURSIVE" in upper_query:
            try:
                cursor.execute("SET max_recursive_iterations = 100;")
                cursor.execute("SET statement_timeout = '15s';")
            except Exception:
                conn.rollback()
                cursor.execute("SET statement_timeout = '15s';")
        else:
            cursor.execute("SET statement_timeout = '60s';")

        cursor.execute(query)
        conn.commit()

        try:
            rows = cursor.fetchmany(MAX_ROWS + 1)
            if len(rows) > MAX_ROWS:
                rows = rows[:MAX_ROWS]
            result = rows
        except psycopg2.ProgrammingError:
            result = None

        return (result, conn)

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        try:
            cursor.execute("SET statement_timeout = '60s';")
        except Exception:
            pass
        cursor.close()


def execute_queries(queries, db_name, conn, logger=None, section_title=""):
    """
    完全对齐官方 postgresql_utils.py execute_queries：
    签名：(queries, db_name, conn, logger=None, section_title="")
    返回 (query_result, execution_error_flag, timeout_flag)
    """
    query_result = None
    execution_error = False
    timeout_error = False

    for query in queries:
        query = query.strip()
        if not query:
            continue
        try:
            query_result, conn = perform_query_on_postgresql_databases(
                query, db_name, conn=conn
            )
        except psycopg2.errors.QueryCanceled:
            timeout_error = True
            break
        except psycopg2.OperationalError:
            execution_error = True
            break
        except psycopg2.Error:
            execution_error = True
            break
        except Exception:
            execution_error = True
            break

        if execution_error or timeout_error:
            break

    return query_result, execution_error, timeout_error


def ex_base(pred_sqls, sol_sqls, db_name, conn):
    """
    完全对齐官方 postgresql_test_utils.py ex_base：
    - 用 execute_queries(sqls, db_name, conn, None, "") 5参数调用
    - 结果为空（None 或 []）时返回 0
    - preprocess_results 后做 set 比较
    """
    if not pred_sqls or not sol_sqls:
        return 0

    predicted_res, pred_execution_error, pred_timeout_error = execute_queries(
        pred_sqls, db_name, conn, None, ""
    )
    ground_truth_res, gt_execution_error, gt_timeout_error = execute_queries(
        sol_sqls, db_name, conn, None, ""
    )

    if (gt_execution_error or gt_timeout_error
            or pred_execution_error or pred_timeout_error):
        return 0

    if not predicted_res or not ground_truth_res:
        return 0

    predicted_res = preprocess_results(predicted_res)
    ground_truth_res = preprocess_results(ground_truth_res)
    return 1 if set(predicted_res) == set(ground_truth_res) else 0


def check_sql_function_usage(sqls, required_keywords):
    """与官方完全一致"""
    if not sqls:
        return 0
    combined_sql = " ".join(sql.lower() for sql in sqls)
    for kw in required_keywords:
        if kw.lower() not in combined_sql:
            return 0
    return 1


# ═══════════════════════════════════════════════════════════════════════════════
# 二、数据库 reset（对齐 pipeline reset_and_restore_database）
# ═══════════════════════════════════════════════════════════════════════════════

def reset_database(db_name):
    """
    每条评测后从 _template 重建数据库，确保状态干净。
    对齐 pipeline 的 reset_and_restore_database 逻辑。
    """
    template_name = f"{db_name}_template"
    try:
        admin_conn = psycopg2.connect(
            host=PG_CONFIG['host'], port=PG_CONFIG['port'],
            user=PG_CONFIG['user'], password=PG_CONFIG['password'],
            database='postgres',
        )
        admin_conn.autocommit = True
        cur = admin_conn.cursor()
        try:
            cur.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """)
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            cur.execute(f'CREATE DATABASE "{db_name}" TEMPLATE "{template_name}"')
        finally:
            cur.close()
            admin_conn.close()
    except Exception as e:
        print(f"[WARN] reset_database({db_name}) failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 三、eval_one_instance（对齐 pipeline._run_test_cases_internal）
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_sqls(val):
    """解析 preprocess_sql / clean_up_sql（支持 list 或 [split] 分隔的字符串）"""
    if not val:
        return []
    if isinstance(val, list):
        return [s.strip() for s in val if s and s.strip()]
    return [s.strip() for s in re.split(r'\[split\]\s*', val) if s.strip()]


def eval_one_instance(record, pred_sqls, sol_sqls):
    """
    完全对齐 pipeline._run_test_cases_internal：
    1. 每条独立新建连接（不复用，避免状态污染）
    2. 执行 preprocess_sql
    3. 执行 pred_sqls 得到 pred_query_result
    4. pred_sqls 执行失败 → 直接返回 False
    5. exec test_case，用单一 local_ns（pipeline 风格）
    6. result is not None and result != 1 → FAIL
    7. 执行 clean_up_sql（finally 里）

    注意：不在内部调用 reset_database，由调用方在评测后显式 reset。

    Returns:
        (passed: bool, msg: str)
    """
    db_name = record.get('db_id', '')
    test_cases = record.get('test_cases', [])
    pre_sqls = _parse_sqls(record.get('preprocess_sql'))
    clean_sqls = _parse_sqls(record.get('clean_up_sql'))

    # 每条独立新建连接
    try:
        conn = psycopg2.connect(database=db_name, **PG_CONFIG)
        conn.autocommit = False
    except Exception as e:
        return False, f"DB connection failed: {e}"

    passed = False
    msg = "no_test"
    try:
        # 1. 执行 preprocess_sql
        if pre_sqls:
            execute_queries(pre_sqls, db_name, conn, None, "Preprocess SQL")

        # 2. 执行 pred_sqls，得到 pred_query_result
        pred_query_result, pred_exec_error, pred_timeout = execute_queries(
            pred_sqls, db_name, conn, None, ""
        )

        # 3. pred_sqls 执行失败 → 直接 FAIL
        if pred_exec_error or pred_timeout:
            msg = f"pred_sqls exec error (exec={pred_exec_error}, timeout={pred_timeout})"
            return False, msg

        # 4. 构造 local_ns（完全对齐 pipeline 的 local_ns）
        local_ns = {
            "perform_query_on_postgresql_databases": perform_query_on_postgresql_databases,
            "execute_queries": execute_queries,
            "ex_base": ex_base,
            "check_sql_function_usage": check_sql_function_usage,
            "remove_distinct": remove_distinct,
            "preprocess_results": preprocess_results,
            "pred_query_result": pred_query_result,
            "date": date,
        }

        # 5. 逐个执行 test_case
        passed = True
        msg = "PASS"
        for tc_code in test_cases:
            try:
                exec("from datetime import date\n" + tc_code, local_ns)
                result = local_ns["test_case"](
                    pred_sqls=pred_sqls,
                    sol_sqls=sol_sqls,
                    db_name=db_name,
                    conn=conn,
                )
                if result is not None and result != 1:
                    passed = False
                    msg = f"FAIL: test_case returned {result} (expected 1)"
                    break
            except AssertionError as e:
                passed = False
                msg = f"FAIL: {e}"
                break
            except Exception as e:
                passed = False
                msg = f"ERROR: {e}"
                break

        return passed, msg

    finally:
        # 6. 执行 clean_up_sql
        if clean_sqls:
            cur = conn.cursor()
            for sql in clean_sqls:
                try:
                    cur.execute(sql)
                    conn.commit()
                except Exception:
                    conn.rollback()
            cur.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 四、SQL 提取工具
# ═══════════════════════════════════════════════════════════════════════════════

def extract_last_sql(text):
    """从模型输出中提取最后一个 ```sql...``` 代码块"""
    blocks = re.findall(r'```sql\s*([\s\S]*?)```', text, re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    # 没有代码块，尝试直接提取 SQL
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    sql_lines = []
    in_sql = False
    for line in lines:
        upper = line.upper()
        if any(upper.startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE',
                                                 'CREATE', 'DROP', 'ALTER', 'WITH',
                                                 'EXPLAIN', '--']):
            in_sql = True
        if in_sql:
            sql_lines.append(line)
    return '\n'.join(sql_lines) if sql_lines else text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 五、Prompt 构造（与训练时 _make_instruction 一字不差）
# ═══════════════════════════════════════════════════════════════════════════════

def build_single_turn_prompt(record):
    """
    必须与 extract_sft_B.py 的 _make_instruction 完全一致：
      ## Database Schema:\n{schema}\n\n
      ## User Issue:\n{query}\n\n
      ## Faulty SQL:\n```sql\n{issue_sql}\n```\n\n
      Fix the buggy SQL. Think step by step, then output the corrected SQL.
    """
    schema = record.get('preprocess_schema', '')
    query = record.get('query', '')
    issue_sql = record.get('issue_sql', '')
    if isinstance(issue_sql, list):
        issue_sql = '\n'.join(issue_sql)
    return (
        f"## Database Schema:\n{schema}\n\n"
        f"## User Issue:\n{query}\n\n"
        f"## Faulty SQL:\n```sql\n{issue_sql}\n```\n\n"
        "Fix the buggy SQL. Think step by step, then output the corrected SQL."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 六、SQL-ACT Agent Prompt 模板（内联自 prompts.py）
# ═══════════════════════════════════════════════════════════════════════════════

# 注意：multi 训练样本（build_multiturn_sample）没有 system 消息，
# 对话从 user 消息开始。评测时同样不加 system 消息，保持与训练格式一致。
SQLACT_SYSTEM = None  # multi 模型训练时无 system 消息

# 初始 user 消息格式（对齐 build_multiturn_sample 的第一条 user 消息）：
# f"## Database Schema:\n{schema}\n\n## User Issue:\n{query}\n\n## Faulty SQL:\n```sql\n{issue_sql}\n```"
# 注意：训练时没有 "best_sql" / "failure_detail" 字段，评测时也不加，保持格式一致。
SQLACT_INITIAL_TEMPLATE = """\
## Database Schema:
{schema}

## User Issue:
{query}

## Faulty SQL:
```sql
{issue_sql}
```"""

# EXPLORE 反馈格式（对齐 build_multiturn_sample 的 user observation 消息）：
# f"## Observation:\n{observation}\n\nContinue debugging."
EXPLORE_OBSERVATION_TEMPLATE = """\
## Observation:
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


# ═══════════════════════════════════════════════════════════════════════════════
# 七、EXPLORE 执行器（内联自 explore_executor.py，SAVEPOINT 机制）
# ═══════════════════════════════════════════════════════════════════════════════

def _format_table(columns, rows):
    """格式化查询结果为可读文本表格"""
    if not columns:
        return "(no columns)"
    if not rows:
        return " | ".join(columns) + "\n(0 rows)"
    str_rows = [[str(v) if v is not None else "NULL" for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(columns)]
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(
        " | ".join(val.ljust(w) for val, w in zip(row, widths))
        for row in str_rows
    )
    return f"{header}\n{sep}\n{body}"


def execute_explore_sql_with_rollback(sql, conn, timeout_sec=5):
    """
    在 SAVEPOINT 内执行任意 SQL，捕获结果后回滚。
    Agent 可观察执行效果（结果集、受影响行数、错误）但数据库状态不变。
    对齐 explore_executor.py 的同名函数。
    """
    cur = conn.cursor()
    try:
        cur.execute("SAVEPOINT explore_sp")
        cur.execute(f"SET LOCAL statement_timeout = '{timeout_sec}s'")
        cur.execute(sql)

        if cur.description:
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(20)
            observation = _format_table(columns, rows)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 八、SQL-ACT Agent 核心逻辑（内联自 sqlagent.py）
# ═══════════════════════════════════════════════════════════════════════════════

def _format_trajectory_for_summary(trajectory):
    """将 trajectory 格式化为可读文本，用于 Final SQL Summary prompt"""
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


def parse_action(response):
    """
    解析模型输出，提取 action_type 和 SQL。
    去除 SQL 块内的 [EXPLORE]/[FIX] 标记。
    返回 (action_type: str, sql: str)
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


def extract_thought(response):
    """
    从 <think>...</think> 标签或 action 标记前的文本提取推理内容。
    优先级：<think> 标签 > [EXPLORE]/[FIX] 前的文本 > 前 500 字符
    """
    think_match = re.search(r'<think>([\s\S]*?)</think>', response, re.IGNORECASE)
    if think_match:
        thought = think_match.group(1).strip()
        return thought[:5000] if thought else ""

    m = re.search(r'^([\s\S]*?)(?:\[?\s*(?:EXPLORE|FIX)\s*\]?)', response, re.IGNORECASE)
    if m:
        thought = m.group(1).strip()
        thought = re.sub(r'^#+\s*', '', thought, flags=re.MULTILINE)
        return thought[:2000] if thought else ""

    return response[:500].strip()


def _compress_conversation(conversation):
    """
    消息压缩：当消息数超过 MAX_STAGE3_MESSAGES 时，
    保留 system + first_user + 最近消息，丢弃中间消息。
    对齐 pipeline._stage3 的消息压缩逻辑。
    """
    if len(conversation) <= MAX_STAGE3_MESSAGES:
        return conversation

    system_msgs = [m for m in conversation if m["role"] == "system"]
    non_system = [m for m in conversation if m["role"] != "system"]

    if not non_system:
        return conversation

    first_user = non_system[0]
    recent = non_system[-(MAX_STAGE3_MESSAGES - len(system_msgs) - 1):]

    return system_msgs + [first_user] + recent


def _get_execution_error(sql, db_name):
    """
    尝试执行 SQL，返回执行错误信息（字符串）或 None（无错误）。
    用于区分 exec error 和 logic error。
    """
    if not sql:
        return "Empty SQL"
    try:
        conn = psycopg2.connect(database=db_name, **PG_CONFIG)
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute("SET statement_timeout = '30s';")
            cur.execute(sql)
            conn.commit()
            return None  # 执行成功，无错误
        except Exception as e:
            conn.rollback()
            return str(e)
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        return str(e)


def run_sqlact_agent(
    record,
    sol_sqls,
    best_sql,
    call_llm_fn,
):
    """
    运行 SQL-ACT Agent 循环（对齐 multi 训练数据格式）。

    训练时 multi 样本（build_multiturn_sample）的格式：
    - 无 system 消息
    - 第一条 user 消息：## Database Schema / ## User Issue / ## Faulty SQL
    - assistant 消息：[ACTION]\n<think>\n{thought}\n\n\n```sql\n{sql}\n```
    - 后续 user 消息：## Observation:\n{obs}\n\nContinue debugging.

    Args:
        record: val_530 中的一条记录（含 db_id, test_cases, preprocess_sql 等）
        sol_sqls: 参考 SQL 列表
        best_sql: 单轮推理得到的最佳 SQL（Agent 的起点，仅用于 fallback）
        call_llm_fn: Function(messages) -> str，调用本地模型

    Returns:
        dict with keys: pass_, sql, iterations, fix_attempts, trajectory, note
    """
    db_name = record.get('db_id', '')
    schema = record.get('preprocess_schema', '')
    query = record.get('query', '')
    issue_sql = record.get('issue_sql', '')
    if isinstance(issue_sql, list):
        issue_sql = '\n'.join(issue_sql)

    # 判断是否为 Management 任务
    is_management = record.get('category', '').lower() == 'management'
    max_explore = 1 if is_management else 3

    # 构造初始对话（对齐训练格式：无 system 消息）
    conversation = [
        {"role": "user", "content": SQLACT_INITIAL_TEMPLATE.format(
            schema=schema,
            query=query,
            issue_sql=issue_sql,
        )},
    ]

    # 为 EXPLORE 建立持久连接（SAVEPOINT 需要同一连接）
    explore_conn = None
    try:
        explore_conn = psycopg2.connect(database=db_name, **PG_CONFIG)
        explore_conn.autocommit = False
    except Exception as e:
        print(f"  [WARN] EXPLORE 连接失败: {e}，EXPLORE 将不可用")

    trajectory = []
    fix_attempts = 0
    explore_count = 0
    last_sql = best_sql
    result_recorded = False
    final_pass = False
    final_sql = best_sql
    note = ""

    def _run_test_cases_with_feedback(sql):
        """运行 test_case，返回 (passed, feedback_str)"""
        sqls = [sql] if sql else []
        passed, msg = eval_one_instance(record, sqls, sol_sqls)
        return passed, msg

    def _reset_db():
        """FIX 失败后 reset 数据库"""
        reset_database(db_name)
        # 重建 explore_conn（reset 后旧连接失效）
        nonlocal explore_conn
        if explore_conn is not None:
            try:
                explore_conn.close()
            except Exception:
                pass
        try:
            explore_conn = psycopg2.connect(database=db_name, **PG_CONFIG)
            explore_conn.autocommit = False
        except Exception as e:
            print(f"  [WARN] reset 后重建 EXPLORE 连接失败: {e}")
            explore_conn = None

    try:
        for step in range(MAX_ITER):
            remaining = MAX_ITER - step

            # 最后一步且从未 FIX → 强制提交 best candidate
            if step == MAX_ITER - 1 and fix_attempts == 0 and last_sql:
                passed, feedback = _run_test_cases_with_feedback(last_sql)
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

            # 紧迫警告
            if remaining <= 2 or (explore_count >= max_explore and fix_attempts == 0):
                conversation[-1]["content"] += (
                    f"\n\n>>> WARNING: You have explored {explore_count} time(s) already. "
                    f"Only {remaining} step(s) left. "
                    f"You MUST submit [FIX] in the NEXT turn. Do NOT use [EXPLORE] again."
                )

            # 消息压缩
            conversation = _compress_conversation(conversation)

            # 调用 LLM
            response = call_llm_fn(conversation)
            if not response:
                break

            action_type, sql = parse_action(response)
            thought = extract_thought(response)

            if sql:
                last_sql = sql

            if action_type == "EXPLORE":
                explore_count += 1
                # 超过 max_explore → 强制转为 FIX
                if explore_count > max_explore and sql:
                    action_type = "FIX"

            if action_type == "EXPLORE":
                # SAVEPOINT 安全执行
                if explore_conn is not None:
                    observation = execute_explore_sql_with_rollback(
                        sql=sql,
                        conn=explore_conn,
                        timeout_sec=5,
                    )
                else:
                    observation = "Error: EXPLORE connection unavailable"

                trajectory.append({
                    "step": step,
                    "action_type": "EXPLORE",
                    "sql": sql,
                    "observation": observation[:MAX_OBSERVATION_CHARS],
                    "thought": thought,
                })

                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": EXPLORE_OBSERVATION_TEMPLATE.format(
                        observation=observation[:MAX_OBSERVATION_CHARS]
                    ),
                })

            elif action_type == "FIX":
                fix_attempts += 1

                passed, feedback = _run_test_cases_with_feedback(sql)

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
                    # FIX 失败：反馈格式对齐训练数据
                    # 训练时 observation 格式：直接是 test_case 返回的 msg 字符串
                    # user 消息格式：## Observation:\n{obs}\n\nContinue debugging.
                    obs_text = feedback[:MAX_OBSERVATION_CHARS] if feedback else "test_case: FAIL"
                    feedback_content = EXPLORE_OBSERVATION_TEMPLATE.format(
                        observation=obs_text,
                    )

                    trajectory.append({
                        "step": step,
                        "action_type": "FIX",
                        "sql": sql,
                        "observation": obs_text[:300],
                        "thought": thought,
                    })

                    conversation.append({"role": "assistant", "content": response})
                    conversation.append({"role": "user", "content": feedback_content})

                    # FIX 失败后 reset DB（防止 Management test_case 的 INSERT/UPDATE 污染）
                    _reset_db()

                    if fix_attempts >= MAX_FIX_ATTEMPTS:
                        final_sql = sql
                        result_recorded = True
                        break

        # ── Final SQL Summary（fix_attempts >= 2 且全部失败时）────────────────
        if result_recorded and not final_pass and fix_attempts >= 2 and trajectory:
            trajectory_text = _format_trajectory_for_summary(trajectory)
            summary_prompt = FINAL_SQL_SUMMARY_TEMPLATE.format(
                schema=schema,
                query=query,
                issue_sql=issue_sql,
                trajectory_summary=trajectory_text,
            )
            # 无 system 消息，对齐训练格式
            summary_response = call_llm_fn([
                {"role": "user", "content": summary_prompt},
            ])
            if summary_response:
                summary_sql = parse_action(summary_response)[1]
                if summary_sql:
                    # reset DB 后测试 summary SQL
                    _reset_db()
                    summary_passed, _ = _run_test_cases_with_feedback(summary_sql)
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

        # ── Fallback：Agent 从未提交 FIX 或耗尽 MAX_ITER ─────────────────────
        if not result_recorded:
            passed, _ = _run_test_cases_with_feedback(last_sql)
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

    finally:
        # 关闭 EXPLORE 连接
        if explore_conn is not None:
            try:
                explore_conn.close()
            except Exception:
                pass

    return {
        "pass_": final_pass,
        "sql": final_sql,
        "iterations": len(trajectory),
        "fix_attempts": fix_attempts,
        "trajectory": trajectory,
        "note": note,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 九、主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SFT 模型 SQL-ACT Agent 评测")
    parser.add_argument('--model', required=True, help='合并后模型路径')
    parser.add_argument('--val', default='/root/swe-sql-kit/data/val_530.jsonl',
                        help='评测集路径')
    parser.add_argument('--output', required=True, help='结果输出路径（.jsonl）')
    parser.add_argument('--limit', type=int, default=None, help='只评估前N条（调试用）')
    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='单轮推理最大 token 数')
    parser.add_argument('--agent_max_new_tokens', type=int, default=4096,
                        help='Agent 每步最大 token 数')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--skip_if_passed', action='store_true',
                        help='单轮已通过则跳过 Agent（默认开启）')
    args = parser.parse_args()

    # 默认开启 skip_if_passed
    skip_if_passed = True  # 单轮通过则跳过 Agent，节省时间

    # 加载测试集
    with open(args.val) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"评估集: {len(records)} 条")

    # 加载模型
    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("模型加载完成")

    def _generate(messages, max_new_tokens):
        """调用本地模型生成文本"""
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    do_sample=(args.temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
        except Exception as e:
            print(f"  [ERROR] 推理失败: {e}")
            return ""

    def call_llm_single(messages):
        return _generate(messages, args.max_new_tokens)

    def call_llm_agent(messages):
        return _generate(messages, args.agent_max_new_tokens)

    # 准备输出
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    passed_single = 0   # 单轮通过数
    passed_agent = 0    # Agent 通过数（含单轮通过）
    total = 0

    for i, record in enumerate(records):
        iid = record.get('instance_id', f'#{i}')
        db_name = record.get('db_id', '')
        category = record.get('category', '')
        sol_sqls = record.get('sol_sql', [])
        if isinstance(sol_sqls, str):
            sol_sqls = [sol_sqls]

        total += 1
        print(f"\n[{i+1:3d}/{len(records)}] {iid} ({db_name}, {category})")

        # ── 阶段1：单轮推理 ────────────────────────────────────────────────────
        prompt = build_single_turn_prompt(record)
        messages = [{"role": "user", "content": prompt}]

        t0 = time.time()
        response_single = call_llm_single(messages)
        latency_single = time.time() - t0

        best_sql = extract_last_sql(response_single)
        best_sqls = [best_sql] if best_sql else []

        # 评测单轮结果
        single_passed, single_msg = eval_one_instance(record, best_sqls, sol_sqls)
        reset_database(db_name)  # 单轮评测后 reset

        if single_passed:
            passed_single += 1
            passed_agent += 1
            print(f"  单轮: ✓ ({latency_single:.1f}s) → 跳过 Agent")

            result = {
                "instance_id": iid,
                "db_id": db_name,
                "category": category,
                "passed_single": True,
                "passed_agent": True,
                "agent_used": False,
                "pred_sql_single": best_sql,
                "pred_sql_agent": best_sql,
                "sol_sql": sol_sqls[0] if sol_sqls else "",
                "single_msg": single_msg,
                "agent_note": "skipped_single_passed",
                "agent_iterations": 0,
                "agent_fix_attempts": 0,
                "latency_single_s": round(latency_single, 2),
                "latency_agent_s": 0.0,
            }
            results.append(result)
            with open(args.output, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            continue

        print(f"  单轮: ✗ {single_msg} ({latency_single:.1f}s) → 进入 Agent")

        # ── 阶段2：SQL-ACT Agent ───────────────────────────────────────────────
        t1 = time.time()
        try:
            agent_result = run_sqlact_agent(
                record=record,
                sol_sqls=sol_sqls,
                best_sql=best_sql,
                call_llm_fn=call_llm_agent,
            )
        except Exception as e:
            print(f"  Agent 异常: {e}")
            traceback.print_exc()
            agent_result = {
                "pass_": False,
                "sql": best_sql,
                "iterations": 0,
                "fix_attempts": 0,
                "trajectory": [],
                "note": f"agent_exception: {e}",
            }
        latency_agent = time.time() - t1

        # Agent 结束后 reset 数据库（确保干净）
        reset_database(db_name)

        agent_passed = agent_result["pass_"]
        if agent_passed:
            passed_agent += 1
            print(f"  Agent: ✓ ({latency_agent:.1f}s, "
                  f"iter={agent_result['iterations']}, fix={agent_result['fix_attempts']}, "
                  f"note={agent_result['note']})")
        else:
            print(f"  Agent: ✗ ({latency_agent:.1f}s, "
                  f"iter={agent_result['iterations']}, fix={agent_result['fix_attempts']}, "
                  f"note={agent_result['note']})")

        result = {
            "instance_id": iid,
            "db_id": db_name,
            "category": category,
            "passed_single": False,
            "passed_agent": agent_passed,
            "agent_used": True,
            "pred_sql_single": best_sql,
            "pred_sql_agent": agent_result["sql"],
            "sol_sql": sol_sqls[0] if sol_sqls else "",
            "single_msg": single_msg,
            "agent_note": agent_result["note"],
            "agent_iterations": agent_result["iterations"],
            "agent_fix_attempts": agent_result["fix_attempts"],
            "latency_single_s": round(latency_single, 2),
            "latency_agent_s": round(latency_agent, 2),
            # trajectory 可选保存（较大，按需开启）
            # "trajectory": agent_result["trajectory"],
        }
        results.append(result)
        with open(args.output, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # ── 统计汇总 ───────────────────────────────────────────────────────────────
    sr_single = passed_single / total if total else 0
    sr_agent = passed_agent / total if total else 0

    cat_single_pass = Counter()
    cat_agent_pass = Counter()
    cat_total = Counter()
    for r in results:
        cat = r['category']
        cat_total[cat] += 1
        if r['passed_single']:
            cat_single_pass[cat] += 1
        if r['passed_agent']:
            cat_agent_pass[cat] += 1

    print(f"\n{'='*60}")
    print(f"模型: {args.model}")
    print(f"单轮 Success Rate:  {passed_single}/{total} = {sr_single:.1%}")
    print(f"Agent Success Rate: {passed_agent}/{total} = {sr_agent:.1%}")
    print(f"Agent 提升: +{passed_agent - passed_single} 条 (+{sr_agent - sr_single:.1%})")
    print(f"\n按类别（单轮 → Agent）:")
    for cat in sorted(cat_total):
        t = cat_total[cat]
        ps = cat_single_pass[cat]
        pa = cat_agent_pass[cat]
        print(f"  {cat:20s}: {ps}/{t}={ps/t:.1%} → {pa}/{t}={pa/t:.1%} (+{pa-ps})")
    print(f"\n结果已保存: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
