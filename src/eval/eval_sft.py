#!/usr/bin/env python3
"""
eval_sft.py — 对合并后的 SFT 模型在 val_530.jsonl 上做推理评估

用法：
    python eval_sft.py --model /root/autodl-tmp/merged_models/B_single \
                       --output /root/eval_results/B_single.jsonl

评估流程：
    1. 加载模型（transformers pipeline）
    2. 对每条 val_530 构造 prompt，让模型修复 issue_sql
    3. 提取模型输出中最后一个 ```sql...``` 代码块
    4. 执行 test_cases 验证，统计 Success Rate
"""

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import psycopg2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── PostgreSQL 连接配置 ────────────────────────────────────────────────────────
PG_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "root",
    "password": "root",
}

# ── 测试工具函数（完全对齐官方 postgresql_test_utils.py）────────────────────
from datetime import date, datetime


def preprocess_results(results):
    """与官方完全一致：将 date/datetime 转为 yyyy-mm-dd 字符串"""
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
    """与官方完全一致：用 \bDISTINCT\b 正则去除"""
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
    - 签名：(queries, db_name, conn, logger=None, section_title="")
    - 使用 perform_query_on_postgresql_databases 执行每条 SQL
    - 区分 QueryCanceled(timeout) 和其他错误
    - 返回 (query_result, execution_error_flag, timeout_flag)
    """
    query_result = None
    execution_error = False
    timeout_error = False

    for i, query in enumerate(queries):
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


# ── SQL 提取 ──────────────────────────────────────────────────────────────────
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


# ── 数据库 reset（对齐 pipeline reset_and_restore_database）─────────────────
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
            # 终止所有到目标库的连接
            cur.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """)
            # 删除并从 template 重建
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            cur.execute(f'CREATE DATABASE "{db_name}" TEMPLATE "{template_name}"')
        finally:
            cur.close()
            admin_conn.close()
    except Exception as e:
        print(f"[WARN] reset_database({db_name}) failed: {e}")


# ── 运行单条实例的完整评测（对齐 pipeline._run_test_cases_internal）────────────
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
    8. 评测结束后 reset 数据库（从 template 重建）
    """
    db_name = record.get('db_id', '')
    test_cases = record.get('test_cases', [])

    # 解析 preprocess_sql / clean_up_sql（支持 list 或 [split] 分隔的字符串）
    def parse_sqls(val):
        if not val:
            return []
        if isinstance(val, list):
            return [s.strip() for s in val if s and s.strip()]
        return [s.strip() for s in re.split(r'\[split\]\s*', val) if s.strip()]

    pre_sqls = parse_sqls(record.get('preprocess_sql'))
    clean_sqls = parse_sqls(record.get('clean_up_sql'))

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
        for i, tc_code in enumerate(test_cases):
            try:
                exec("from datetime import date\n" + tc_code, local_ns)
                result = local_ns["test_case"](
                    pred_sqls=pred_sqls,
                    sol_sqls=sol_sqls,
                    db_name=db_name,
                    conn=conn,
                )
                # pipeline 逻辑：result is not None and result != 1 → FAIL
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
        # 7. 从 template 重建数据库，确保状态干净
        reset_database(db_name)


# ── prompt 构造（与训练时 _make_instruction 一字不差）────────────────────────
def build_prompt(record):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='合并后模型路径')
    parser.add_argument('--val', default='/root/swe-sql-kit/data/val_530.jsonl')
    parser.add_argument('--output', required=True, help='结果输出路径')
    parser.add_argument('--limit', type=int, default=None, help='只评估前N条（调试用）')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args()

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

    # 评估（不预建连接池，每条独立连接）
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    passed = 0

    for i, record in enumerate(records):
        iid = record.get('instance_id', f'#{i}')
        db_name = record.get('db_id', '')
        sol_sqls = record.get('sol_sql', [])
        if isinstance(sol_sqls, str):
            sol_sqls = [sol_sqls]
        elif isinstance(sol_sqls, list):
            # sol_sql 可能是 [split] 分隔的字符串列表，直接用
            pass

        print(f"[{i+1:3d}/{len(records)}] {iid} ({db_name})", end=' ', flush=True)

        # 构造 prompt（与训练时 _make_instruction 一字不差）
        prompt = build_prompt(record)
        messages = [{"role": "user", "content": prompt}]

        # 推理
        t0 = time.time()
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
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=(args.temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        except Exception as e:
            print(f"推理失败: {e}")
            response = ""

        latency = time.time() - t0

        # 提取 SQL
        pred_sql = extract_last_sql(response)
        pred_sqls = [pred_sql] if pred_sql else []

        # 执行评测（每条独立连接 + preprocess_sql + clean_up_sql）
        test_passed, test_msg = eval_one_instance(record, pred_sqls, sol_sqls)

        if test_passed:
            passed += 1
            print(f"✓ ({latency:.1f}s)")
        else:
            print(f"✗ {test_msg} ({latency:.1f}s)")

        result = {
            "instance_id": iid,
            "db_id": db_name,
            "category": record.get('category', ''),
            "passed": bool(test_passed),
            "pred_sql": pred_sql,
            "sol_sql": sol_sqls[0] if sol_sqls else "",
            "test_msg": test_msg,
            "latency_s": round(latency, 2),
        }
        results.append(result)

        with open(args.output, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 统计
    from collections import Counter
    sr = passed / len(results) if results else 0
    cat_pass = Counter()
    cat_total = Counter()
    for r in results:
        cat = r['category']
        cat_total[cat] += 1
        if r['passed']:
            cat_pass[cat] += 1

    print(f"\n{'='*50}")
    print(f"模型: {args.model}")
    print(f"总体 Success Rate: {passed}/{len(results)} = {sr:.1%}")
    for cat in sorted(cat_total):
        p = cat_pass[cat]
        t = cat_total[cat]
        print(f"  {cat}: {p}/{t} = {p/t:.1%}")
    print(f"结果已保存: {args.output}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
