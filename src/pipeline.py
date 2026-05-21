"""
SWE-SQL+ Pipeline Orchestrator
==============================
Runs Stages 0-3 with short-circuit logic:
  Stage 0: Baseline (submit issue_sql as-is)
  Stage 1: CoT single-path reasoning
  Stage 2: N-path 3-way parallel LLM + serial test-case validation
  Stage 3: SQL-ACT iterative agent with EXPLORE/FIX
"""

import gc
import re
import time
import concurrent.futures
from datetime import date

import psycopg2

from prompts import (
    cot_messages,
    NPATH_BUILDERS,
    NPATH_NAMES,
)
from sqlagent import run_sqlact
from postgresql_utils import (
    perform_query_on_postgresql_databases,
    close_postgresql_pool,
    reset_and_restore_database,
)
from postgresql_test_utils import (
    ex_base,
    remove_distinct,
    execute_queries,
    check_sql_function_usage,
    preprocess_results,
    performance_compare_by_qep,
)


def extract_sqls(text: str) -> list[str]:
    """Extract all SQL from ```sql ... ``` code blocks in LLM output.
    Returns a list of SQL strings (one per code block).
    For Management tasks, LLM often outputs multiple blocks (CREATE FUNCTION + CREATE TRIGGER).
    """
    blocks = re.findall(r'```\s*sql\s*([\s\S]*?)```', text, re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    blocks = re.findall(r'```\s*([\s\S]*?)```', text)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    return [text.strip()]


def extract_sql(text: str) -> str:
    """Extract first SQL block (backward compat for Stage 3 single-SQL actions)."""
    sqls = extract_sqls(text)
    return sqls[0] if sqls else text.strip()


class PipelineRunner:
    """
    Runs the 4-stage pipeline for a single task instance.

    Requires:
        call_llm: function(messages, **kwargs) -> str
        get_conn: function(db_name=None) -> psycopg2 connection
        cfg: dict with PG connection params
    """

    def __init__(self, call_llm_fn, get_conn_fn, cfg: dict, skip_stage3: bool = False):
        self.call_llm = call_llm_fn
        self.get_conn = get_conn_fn
        self.cfg = cfg
        self.skip_stage3 = skip_stage3

    def _call_llm_no_thinking(self, messages, **kwargs):
        """Call LLM with thinking forcibly disabled (used in Stage 2).

        Stage 2 is N-path parallel generation — its outputs are NOT used for
        SFT data extraction (B_single / B_multi). Disabling thinking here saves
        ~2x Stage 2 latency without affecting SFT data quality.

        Note: kwargs may already contain thinking=True (injected by the outer
        lambda in run_pipeline.py). We must pop it before passing to avoid
        "multiple values for keyword argument 'thinking'" errors.
        """
        kwargs.pop("thinking", None)  # remove any thinking=True from outer lambda
        return self.call_llm(messages, thinking=False, **kwargs)

    # ─── Ephemeral DB management ─────────────────────────────────────────

    def _create_work_db(self, template_db_name: str, work_db_name: str):
        """Create a working copy of a template database."""
        conn = self.get_conn()  # connect to default DB
        conn.autocommit = True
        cur = conn.cursor()
        try:
            # Terminate stale connections to the work DB (if it exists)
            cur.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{work_db_name}' AND pid <> pg_backend_pid();
            """)
            cur.execute(f'DROP DATABASE IF EXISTS "{work_db_name}"')
            cur.execute(f'CREATE DATABASE "{work_db_name}" TEMPLATE "{template_db_name}"')
        finally:
            cur.close()
            conn.close()

    def _drop_work_db(self, work_db_name: str):
        """Drop a working database copy after evaluation."""
        close_postgresql_pool(work_db_name)
        conn = self.get_conn()  # connect to default DB
        conn.autocommit = True
        cur = conn.cursor()
        try:
            cur.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{work_db_name}' AND pid <> pg_backend_pid();
            """)
            cur.execute(f'DROP DATABASE IF EXISTS "{work_db_name}"')
        finally:
            cur.close()
            conn.close()

    # ─── Test case infrastructure ─────────────────────────────────────────

    def _run_test_cases_internal(self, pred_sqls, instance, db_name, collect_feedback=False):
        """Run official test cases against predicted SQL. Returns (passed, feedback_str).

        IMPORTANT: Matches baseline evaluation logic exactly:
        1. Execute pred_sqls first to get pred_query_result (used by some test_cases)
        2. Test passes if no exception is raised (return value is NOT checked,
           because many test_cases use assert-only and return None)

        When collect_feedback=True (Stage 3 FIX evaluation):
        - Captures pred_query_result and sol_query_result for diagnostic feedback
        - Provides actual vs expected comparison for ex_base-style tests
        - DB state cleanup is handled by reset_db_fn in run_sqlact after FIX failure
        """
        if isinstance(pred_sqls, str):
            pred_sqls = [pred_sqls]

        sol_sqls = instance["sol_sql"]
        test_cases = instance["test_cases"]
        pre_sqls = [s for s in (instance.get("preprocess_sql") or []) if s and s.strip()]
        clean_sqls = [s for s in (instance.get("clean_up_sql") or []) if s and s.strip()]

        try:
            conn = self.get_conn(db_name)
            conn.autocommit = False
        except Exception as e:
            return False, f"DB connection failed: {e}"

        feedback_parts = []
        all_pass = True

        try:
            # Run preprocessing SQL (baseline style: via execute_queries, stops on first error)
            if pre_sqls:
                execute_queries(pre_sqls, db_name, conn, None, "Preprocess SQL")

            # Execute pred_sqls first to get pred_query_result
            # (baseline does this in run_evaluation_phase via execute_queries)
            pred_query_result, pred_exec_error, pred_timeout = execute_queries(
                pred_sqls, db_name, conn, None, ""
            )

            # If pred_sqls have execution/timeout errors, fail immediately
            if pred_exec_error or pred_timeout:
                err_msg = f"pred_sqls execution error: {pred_exec_error or 'timeout'}"
                return False, err_msg

            # Match official evaluation's global_env exactly
            # (see BIRD-CRITIC-1/evaluation/src/single_instance_eval_postgresql.py)
            local_ns = {
                "perform_query_on_postgresql_databases": perform_query_on_postgresql_databases,
                "execute_queries": execute_queries,
                "ex_base": ex_base,
                "performance_compare_by_qep": performance_compare_by_qep,
                "check_sql_function_usage": check_sql_function_usage,
                "remove_distinct": remove_distinct,
                "preprocess_results": preprocess_results,
                "pred_query_result": pred_query_result,
                "date": date,
            }

            for i, tc_code in enumerate(test_cases):
                try:
                    exec(
                        "from datetime import date\n" + tc_code,
                        local_ns,
                    )
                    result = local_ns["test_case"](
                        pred_sqls=pred_sqls,
                        sol_sqls=sol_sqls,
                        db_name=db_name,
                        conn=conn,
                    )
                    # Match baseline: if no exception raised, test passes.
                    # Many test_cases use assert-only and return None (not 1).
                    # Only fail if result is explicitly 0 (from ex_base etc.)
                    if result is not None and result != 1:
                        all_pass = False
                        if collect_feedback:
                            excerpt = "\n".join(tc_code.strip().splitlines()[:8])
                            # Build rich feedback with actual vs expected output
                            actual_output_str = self._build_actual_output_feedback(
                                pred_sqls, sol_sqls, pred_query_result, db_name, conn, tc_code
                            )
                            feedback_parts.append(
                                f"[test_case {i+1} FAILED] returned {result} (expected 1)\n"
                                f"Test logic (excerpt):\n{excerpt}"
                                + actual_output_str
                            )
                        break
                except AssertionError as e:
                    all_pass = False
                    if collect_feedback:
                        excerpt = "\n".join(tc_code.strip().splitlines()[:8])
                        # Build rich feedback with actual vs expected output
                        actual_output_str = self._build_actual_output_feedback(
                            pred_sqls, sol_sqls, pred_query_result, db_name, conn, tc_code
                        )
                        feedback_parts.append(
                            f"[test_case {i+1} AssertionError] {e}\n"
                            f"Test logic (excerpt):\n{excerpt}"
                            + actual_output_str
                        )
                    break
                except Exception as e:
                    all_pass = False
                    if collect_feedback:
                        excerpt = "\n".join(tc_code.strip().splitlines()[:8])
                        feedback_parts.append(
                            f"[test_case {i+1} Exception] {type(e).__name__}: {e}\n"
                            f"Test logic (excerpt):\n{excerpt}"
                        )
                    break

            return all_pass, "\n\n".join(feedback_parts)

        finally:
            if clean_sqls:
                cur = conn.cursor()
                for sql in clean_sqls:
                    try:
                        cur.execute(sql)
                        conn.commit()
                    except Exception:
                        conn.rollback()
            conn.close()

    def run_test_cases(self, pred_sqls, instance, db_name) -> bool:
        passed, _ = self._run_test_cases_internal(pred_sqls, instance, db_name)
        return passed

    def run_test_cases_with_feedback(self, pred_sqls, instance, db_name):
        return self._run_test_cases_internal(pred_sqls, instance, db_name, collect_feedback=True)

    def _build_actual_output_feedback(
        self, pred_sqls, sol_sqls, pred_query_result, db_name, conn, tc_code
    ) -> str:
        """Build rich diagnostic feedback showing actual vs expected SQL output.

        Strategy:
        1. For ex_base-style tests: execute sol_sqls to get expected result,
           show both pred and sol results for comparison.
        2. For assert-style tests with pred_query_result: show the actual
           pred_query_result value so the model can see what it produced.
        3. For other tests: try to re-execute pred_sqls and show output.

        Returns a string to append to the feedback message.
        """
        parts = []

        # --- Show pred_query_result if available (useful for assert-style tests) ---
        if pred_query_result is not None:
            try:
                pred_str = repr(pred_query_result)
                if len(pred_str) > 500:
                    pred_str = pred_str[:500] + "..."
                parts.append(f"\n[Your SQL actual result (pred_query_result)]:\n{pred_str}")
            except Exception:
                pass

        # --- For ex_base-style tests, also show the expected (sol_sqls) result ---
        tc_lower = tc_code.lower()
        if "ex_base" in tc_lower:
            try:
                # Execute sol_sqls on a fresh cursor to get expected result
                sol_result, sol_err, sol_timeout = execute_queries(
                    sol_sqls, db_name, conn, None, ""
                )
                if sol_result is not None and not sol_err and not sol_timeout:
                    sol_str = repr(sol_result)
                    if len(sol_str) > 500:
                        sol_str = sol_str[:500] + "..."
                    parts.append(f"\n[Expected result (from ground truth SQL)]:\n{sol_str}")
            except Exception:
                pass

        if parts:
            return "\n" + "\n".join(parts)
        return ""

    def get_execution_error(self, sqls, db_name) -> str | None:
        """Check if SQL executes without error. Returns error string or None."""
        if isinstance(sqls, str):
            sqls = [sqls]
        try:
            conn = self.get_conn(db_name)
            conn.autocommit = False
            cur = conn.cursor()
            error = None
            try:
                for sql in sqls:
                    if sql.strip():
                        cur.execute(sql)
                conn.rollback()
            except Exception as e:
                error = str(e)
                conn.rollback()
            conn.close()
            return error
        except Exception as e:
            return str(e)

    # ─── Stage 0: Baseline ────────────────────────────────────────────────

    def _stage0(self, instance, db_name) -> dict:
        issue_sql = instance["issue_sql"]
        passed = self.run_test_cases(issue_sql, instance, db_name)
        return {
            "stage0_pass": passed,
            "stage0_sql": issue_sql,
        }

    # ─── Stage 1: CoT ────────────────────────────────────────────────────

    def _stage1(self, instance, db_name) -> dict:
        schema = instance["schema"]
        query = instance["query"]
        issue_sql_text = instance.get("issue_sql_text") or "\n\n".join(instance["issue_sql"])

        t0 = time.time()
        response = self.call_llm(cot_messages(schema, query, issue_sql_text))
        sqls = extract_sqls(response) if response else instance["issue_sql"]
        passed = self.run_test_cases(sqls, instance, db_name)

        return {
            "stage1_pass": passed,
            "stage1_sql": sqls,
            "stage1_response": response,
            "stage1_latency_s": round(time.time() - t0, 1),
        }

    # ─── Stage 2: N-path ─────────────────────────────────────────────────

    def _stage2(self, instance, db_name) -> dict:
        schema = instance["schema"]
        query = instance["query"]
        issue_sql_text = instance.get("issue_sql_text") or "\n\n".join(instance["issue_sql"])

        t0 = time.time()

        # Step 1: Parallel LLM calls
        # Stage 2 uses _call_llm_no_thinking: its outputs are NOT used for SFT
        # data extraction (B_single / B_multi come from Stage 1 & Stage 3).
        # Disabling thinking here saves ~2x Stage 2 latency with no SFT impact.
        def call_one(args):
            name, builder = args
            raw = self._call_llm_no_thinking(builder(schema, query, issue_sql_text))
            sqls = extract_sqls(raw) if raw else instance["issue_sql"]
            return name, sqls, raw

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as exe:
            candidates = list(exe.map(call_one, zip(NPATH_NAMES, NPATH_BUILDERS)))

        # Step 2: Serial test-case validation (critical for Management tasks)
        paths_result = {}
        selected_sql = None
        selected_path = None
        selected_pass = False

        no_error_candidates = []

        for name, sqls, raw in candidates:
            passed = self.run_test_cases(sqls, instance, db_name)
            exec_err = None if passed else self.get_execution_error(sqls, db_name)

            paths_result[name] = {
                "sql": sqls,
                "pass": passed,
                "error": exec_err or "",
            }

            if passed and selected_sql is None:
                selected_sql = sqls
                selected_path = name
                selected_pass = True

            if not passed and exec_err is None:
                no_error_candidates.append((name, sqls))

        # Selection priority
        if selected_sql is None:
            if no_error_candidates:
                selected_path, selected_sql = no_error_candidates[0]
            else:
                # Fallback to diagnostic
                selected_sql = paths_result.get("diagnostic", {}).get("sql", instance["issue_sql"])
                selected_path = "diagnostic(fallback)"

        return {
            "stage2_pass": selected_pass,
            "stage2_selected_path": selected_path,
            "stage2_sql": selected_sql,
            "stage2_paths": paths_result,
            "stage2_latency_s": round(time.time() - t0, 1),
        }

    # ─── Stage 2 → Stage 3 start SQL ─────────────────────────────────────

    @staticmethod
    def _join_sqls(sqls) -> str:
        """Join a list of SQL strings into one text block for prompts."""
        if isinstance(sqls, str):
            return sqls
        return "\n\n".join(sqls)

    def _get_stage3_start(self, stage1_result, stage2_result, issue_sql_text) -> tuple[str, str]:
        """Determine starting SQL and failure context for Stage 3.
        Returns (sql_text_for_prompt, failure_detail).
        """
        # Priority 1: Stage 2 pass (shouldn't reach Stage 3, but safety)
        if stage2_result.get("stage2_pass"):
            return self._join_sqls(stage2_result["stage2_sql"]), "Stage 2 already passed."

        # Priority 2: Stage 2 executable (no error)
        s2_sql = stage2_result.get("stage2_sql")
        s2_text = self._join_sqls(s2_sql) if s2_sql else ""
        if s2_text and s2_text != issue_sql_text:
            path_details = []
            for pname, pdata in stage2_result.get("stage2_paths", {}).items():
                status = "PASS" if pdata.get("pass") else f"FAIL: {pdata.get('error', 'logic error')[:100]}"
                path_details.append(f"  {pname}: {status}")
            failure_detail = (
                f"N-path generation produced 3 candidates but none passed test cases.\n"
                f"Best candidate (from {stage2_result.get('stage2_selected_path', 'unknown')}) "
                f"selected for iteration.\n\nPath results:\n" + "\n".join(path_details)
            )
            return s2_text, failure_detail

        # Priority 3: Stage 1 CoT SQL
        s1_sql = stage1_result.get("stage1_sql")
        s1_text = self._join_sqls(s1_sql) if s1_sql else ""
        if s1_text and s1_text != issue_sql_text:
            failure_detail = (
                "All previous attempts failed. "
                "N-path generation failed. Best available SQL is from CoT single-path."
            )
            return s1_text, failure_detail

        # Priority 4: Original issue_sql
        failure_detail = (
            "All previous attempts failed. "
            "Starting from scratch with the original faulty SQL."
        )
        return issue_sql_text, failure_detail

    # ─── Stage 3: SQL-ACT ────────────────────────────────────────────────

    def _stage3(self, instance, db_name, template_db, start_sql, failure_detail) -> dict:
        schema = instance["schema"]
        query = instance["query"]
        issue_sql = instance.get("issue_sql_text") or "\n\n".join(instance["issue_sql"])

        t0 = time.time()

        # Mutable container for explore_conn so reset_db can refresh it
        explore_state = {"conn": None}

        def get_explore_conn():
            if explore_state["conn"] is None:
                explore_state["conn"] = self.get_conn(db_name)
                explore_state["conn"].autocommit = False
            return explore_state["conn"]

        try:
            get_explore_conn()
        except Exception as e:
            return {
                "stage3_pass": False,
                "stage3_sql": start_sql,
                "stage3_iterations": 0,
                "stage3_fix_attempts": 0,
                "stage3_trajectory": [],
                "stage3_note": f"db_error: {e}",
                "stage3_latency_s": 0,
            }

        def reset_db_after_fix():
            """Reset work DB to clean state after a FIX attempt.

            Management test_cases often INSERT/UPDATE data that gets committed
            by perform_query_on_postgresql_databases. Without reset, subsequent
            FIX attempts see stale data (e.g. duplicate key errors).
            """
            # Close the explore connection first
            try:
                if explore_state["conn"]:
                    explore_state["conn"].close()
                    explore_state["conn"] = None
            except Exception:
                explore_state["conn"] = None

            # Close any pooled connections to this DB
            close_postgresql_pool(db_name)

            # Recreate work DB from template
            try:
                self._create_work_db(template_db, db_name)
            except Exception as e:
                print(f"    [WARN] Failed to reset work DB after FIX: {e}")

            # Re-establish explore connection
            try:
                explore_state["conn"] = self.get_conn(db_name)
                explore_state["conn"].autocommit = False
            except Exception as e:
                print(f"    [WARN] Failed to reconnect after DB reset: {e}")

        # Wrapper to provide higher max_tokens and conversation compression for Stage 3
        # Multi-turn interactions. Without compression, the conversation history grows
        # unboundedly and causes thinking/content truncation in later turns.
        MAX_STAGE3_MESSAGES = 8  # Keep within SiliconFlow's 10-message limit
        MAX_OBSERVATION_CHARS = 1500  # Truncate observation text to save tokens

        def compress_messages(messages: list[dict]) -> list[dict]:
            """Trim conversation to stay within max messages while preserving context.

            Strategy:
            - Always keep: system prompt (idx 0) + initial user prompt (idx 1)
            - Keep the most recent turns (assistant + user pairs)
            - Insert a summary marker between old and new content
            """
            if len(messages) <= MAX_STAGE3_MESSAGES:
                return messages

            # Preserve system + first user instruction (schema + query + SQLs)
            header = messages[:2]

            # Keep the most recent messages
            recent_count = MAX_STAGE3_MESSAGES - len(header)
            recent = messages[-recent_count:]

            # Count dropped messages for debug logging
            dropped = len(messages) - len(header) - len(recent)

            return header + [{
                "role": "user",
                "content": f"[... {dropped} intermediate messages omitted for brevity ...]",
            }] + recent

        STAGE3_DEADLINE_S = 600  # 10 min hard cap per task for Stage 3

        def call_llm_stage3(messages):
            # Hard deadline: if Stage 3 has been running > 10 min, abort immediately.
            # Without this, a single stuck LLM call (180s) × 7 turns = 21 min hang.
            elapsed = time.time() - t0
            if elapsed > STAGE3_DEADLINE_S:
                print(f"    [Stage3 TIMEOUT] {elapsed:.0f}s > {STAGE3_DEADLINE_S}s deadline, aborting.")
                return ""  # triggers `if not response: break` in run_sqlact

            # Compress conversation history to prevent context window exhaustion
            # Each turn adds 2 messages (assistant + user observation), so history
            # grows quickly. SiliconFlow also enforces maxItems=10 on messages.
            messages = compress_messages(messages)

            # Stage 3 is multi-turn with longer context. Each turn includes:
            # thinking_content + observation + plan + SQL output.
            # Increased from 8192 to 12000 to capture full reasoning chains.
            return self.call_llm(messages, max_tokens=12000)

        try:
            agent_result = run_sqlact(
                schema=schema,
                query=query,
                issue_sql=issue_sql,
                best_sql=start_sql,
                failure_detail=failure_detail,
                db_conn_fn=get_explore_conn,
                call_llm_fn=call_llm_stage3,
                run_test_cases_fn=lambda sql: self.run_test_cases(sql, instance, db_name),
                run_test_cases_with_feedback_fn=lambda sql: self.run_test_cases_with_feedback(sql, instance, db_name),
                get_execution_error_fn=lambda sql: self.get_execution_error([sql], db_name),
                is_management=instance.get("category") == "Management",
                reset_db_fn=reset_db_after_fix,
            )
        finally:
            try:
                if explore_state["conn"]:
                    explore_state["conn"].close()
            except Exception:
                pass

        return {
            "stage3_pass": agent_result["pass_"],
            "stage3_sql": agent_result["sql"],
            "stage3_iterations": agent_result["iterations"],
            "stage3_fix_attempts": agent_result["fix_attempts"],
            "stage3_trajectory": agent_result["trajectory"],
            "stage3_note": agent_result.get("note", ""),
            "stage3_latency_s": round(time.time() - t0, 1),
        }

    # ─── Full pipeline for one instance ───────────────────────────────────

    def run_one(self, instance: dict) -> dict:
        """Run the full 4-stage pipeline for a single instance with short-circuit.

        Creates an ephemeral working DB copy from the template for this instance,
        runs all stages on it, then drops it — ensuring no cross-instance pollution.
        """
        iid = instance["instance_id"]
        template_db = instance["db_name"]  # e.g. "financial_template"
        work_db = template_db.replace("_template", f"_work_{iid}")

        # Create a fresh working copy of the DB
        self._create_work_db(template_db, work_db)

        result = {
            "instance_id": iid,
            "db_id": instance["db_id"],
            "category": instance["category"],
        }

        try:
            # ── Stage 0 ──
            s0 = self._stage0(instance, work_db)
            result.update(s0)
            if s0["stage0_pass"]:
                result["final_sql"] = s0["stage0_sql"]
                result["first_pass_stage"] = "stage0"
                self._fill_empty_stages(result, 1)
                print(f"    Stage0 Baseline=PASS")
                return result
            print(f"    Stage0 Baseline=FAIL")

            # ── Stage 1 ──
            s1 = self._stage1(instance, work_db)
            result.update(s1)
            if s1["stage1_pass"]:
                result["final_sql"] = s1["stage1_sql"]
                result["first_pass_stage"] = "stage1"
                self._fill_empty_stages(result, 2)
                print(f"    Stage1 CoT=PASS ({s1['stage1_latency_s']:.1f}s)")
                return result
            print(f"    Stage1 CoT=FAIL ({s1['stage1_latency_s']:.1f}s)")

            # ── Stage 2 ──
            s2 = self._stage2(instance, work_db)
            result.update(s2)
            if s2["stage2_pass"]:
                result["final_sql"] = s2["stage2_sql"]
                result["first_pass_stage"] = "stage2"
                self._fill_empty_stages(result, 3)
                print(f"    Stage2 N-path({s2['stage2_selected_path']})=PASS ({s2['stage2_latency_s']:.1f}s)")
                return result
            print(f"    Stage2 N-path({s2['stage2_selected_path']})=FAIL ({s2['stage2_latency_s']:.1f}s)")

            # ── Stage 3 ──
            if self.skip_stage3:
                self._fill_empty_stages(result, 3)
                result["final_sql"] = self._join_sqls(
                    s2.get("stage2_sql") or s1.get("stage1_sql") or instance["issue_sql"]
                )
                result["first_pass_stage"] = None
                print(f"    Stage3 SKIPPED (--skip-stage3)")
            else:
                issue_sql_text = instance.get("issue_sql_text") or "\n\n".join(instance["issue_sql"])
                start_sql, failure_detail = self._get_stage3_start(s1, s2, issue_sql_text)
                s3 = self._stage3(instance, work_db, template_db, start_sql, failure_detail)
                result.update(s3)
                if s3["stage3_pass"]:
                    result["final_sql"] = s3["stage3_sql"]
                    result["first_pass_stage"] = "stage3"
                else:
                    result["final_sql"] = s3["stage3_sql"]
                    result["first_pass_stage"] = None
                s3_note = s3.get("stage3_note", "")
                note_str = f" [{s3_note}]" if s3_note else ""
                print(
                    f"    Stage3 SQL-ACT={'PASS' if s3['stage3_pass'] else 'FAIL'} "
                    f"({s3['stage3_latency_s']:.1f}s) "
                    f"fix={s3['stage3_fix_attempts']} iter={s3['stage3_iterations']}"
                    f"{note_str}"
                )

            return result

        finally:
            # Always drop the ephemeral working DB to prevent pollution
            try:
                self._drop_work_db(work_db)
            except Exception as e:
                print(f"    [WARN] Failed to drop work DB {work_db}: {e}")

            # Explicit GC to release memory after each task
            # This is critical when running 400+ tasks sequentially —
            # accumulated LLM response strings and result dicts can exhaust swap.
            gc.collect()

    @staticmethod
    def _fill_empty_stages(result: dict, from_stage: int):
        """Fill skipped stages with empty defaults."""
        if from_stage <= 1:
            result.setdefault("stage1_pass", False)
            result.setdefault("stage1_sql", "")
            result.setdefault("stage1_response", "")
            result.setdefault("stage1_latency_s", 0)
        if from_stage <= 2:
            result.setdefault("stage2_pass", False)
            result.setdefault("stage2_selected_path", "")
            result.setdefault("stage2_sql", "")
            result.setdefault("stage2_paths", {})
            result.setdefault("stage2_latency_s", 0)
        if from_stage <= 3:
            result.setdefault("stage3_pass", False)
            result.setdefault("stage3_sql", "")
            result.setdefault("stage3_iterations", 0)
            result.setdefault("stage3_fix_attempts", 0)
            result.setdefault("stage3_trajectory", [])
            result.setdefault("stage3_note", "")
            result.setdefault("stage3_latency_s", 0)
