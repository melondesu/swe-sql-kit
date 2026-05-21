# postgresql_utils.py
# 改造版：host 从硬编码 bird_critic_postgresql 改为可通过环境变量配置
import subprocess
import psycopg2
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool
from logger import log_section_header, log_section_footer, PrintLogger, NullLogger
import time
import sys
import json
import re
import os
import csv

_postgresql_pools = {}

# 从环境变量读取，默认 localhost（本地开发）
_PG_HOST = os.environ.get("PG_HOST", "localhost")
_PG_PORT = int(os.environ.get("PG_PORT", "5432"))
_PG_USER = os.environ.get("PG_USER", "root")
_PG_PASS = os.environ.get("PG_PASSWORD", "123123")

DEFAULT_DB_CONFIG = {
    "minconn": 1,
    "maxconn": 5,
    "user": _PG_USER,
    "password": _PG_PASS,
    "host": _PG_HOST,
    "port": _PG_PORT,
}


def _get_or_init_pool(db_name):
    if db_name not in _postgresql_pools:
        config = DEFAULT_DB_CONFIG.copy()
        config.update({"dbname": db_name})
        _postgresql_pools[db_name] = SimpleConnectionPool(
            config["minconn"],
            config["maxconn"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            host=config["host"],
            port=config["port"],
        )
    return _postgresql_pools[db_name]


def perform_query_on_postgresql_databases(query, db_name, conn=None):
    MAX_ROWS = 10000
    need_to_put_back = False

    if conn is None:
        # Only use pool when no external connection is provided
        pool = _get_or_init_pool(db_name)
        conn = pool.getconn()
        need_to_put_back = True

    cursor = conn.cursor()
    upper_query = query.upper()
    if "WITH RECURSIVE" in upper_query:
        try:
            cursor.execute("SET max_recursive_iterations = 100;")
            cursor.execute("SET statement_timeout = '15s';")
        except Exception:
            conn.rollback()
            cursor.execute("SET statement_timeout = '15s';")
    else:
        cursor.execute("SET statement_timeout = '60s';")

    try:
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
        cursor.close()
        # Return connection to pool if we borrowed it
        if need_to_put_back and db_name in _postgresql_pools:
            try:
                _postgresql_pools[db_name].putconn(conn)
            except Exception:
                pass


def close_postgresql_connection(db_name, conn):
    if db_name in _postgresql_pools:
        pool = _postgresql_pools[db_name]
        pool.putconn(conn)


def close_all_postgresql_pools():
    for pool in _postgresql_pools.values():
        pool.closeall()
    _postgresql_pools.clear()


def close_postgresql_pool(db_name):
    if db_name in _postgresql_pools:
        pool = _postgresql_pools.pop(db_name)
        pool.closeall()


def execute_queries(queries, db_name, conn, logger=None, section_title=""):
    if logger is None:
        logger = NullLogger()

    log_section_header(section_title, logger)
    query_result = None
    execution_error = False
    timeout_error = False

    for i, query in enumerate(queries):
        try:
            query_result, conn = perform_query_on_postgresql_databases(
                query, db_name, conn=conn
            )
        except psycopg2.errors.QueryCanceled as e:
            timeout_error = True
            break
        except OperationalError as e:
            execution_error = True
            break
        except psycopg2.Error as e:
            execution_error = True
            break
        except Exception as e:
            execution_error = True
            break

        if execution_error or timeout_error:
            break

    log_section_footer(logger)
    return query_result, execution_error, timeout_error


def reset_and_restore_database(db_name, pg_password=None):
    """
    Reset a database by dropping it and recreating from its _template.

    Uses SQL via psycopg2 (connecting to 'postgres' db) instead of shell
    commands, so it works from macOS host without psql CLI tools installed.

    Args:
        db_name: The database to reset (e.g. "financial_template")
        pg_password: PG password (defaults to _PG_PASS from env/config)
    """
    if pg_password is None:
        pg_password = _PG_PASS

    # Determine template name
    # If db_name IS a template (e.g. "financial_template"), we reset it from itself.
    # The convention: base = db_name without _template suffix, template = base + _template
    if db_name.endswith("_template"):
        # Pipeline uses templates directly — we need to drop and recreate.
        # But we can't drop a template from itself. Instead, close pool and
        # recreate by: terminate connections → drop → create from template.
        # Since the db IS the template, we need a different approach:
        # just close all connections and let the pool reconnect fresh.
        # The clean_up_sql in each instance should handle state reset.
        close_postgresql_pool(db_name)
        return

    # For ephemeral copies (e.g. "financial_process_0"):
    base_db_name = db_name.split("_process_")[0]
    template_db_name = f"{base_db_name}_template"

    # 1) Close pool
    close_postgresql_pool(db_name)

    # 2) Connect to 'postgres' admin DB
    admin_conn = psycopg2.connect(
        host=_PG_HOST, port=_PG_PORT,
        user=_PG_USER, password=pg_password,
        dbname="postgres",
    )
    admin_conn.autocommit = True
    cur = admin_conn.cursor()

    try:
        # Terminate all connections to target DB
        cur.execute(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
        """)
        # Drop
        cur.execute(f"DROP DATABASE IF EXISTS \"{db_name}\"")
        # Recreate from template
        cur.execute(f"CREATE DATABASE \"{db_name}\" TEMPLATE \"{template_db_name}\"")
    finally:
        cur.close()
        admin_conn.close()
