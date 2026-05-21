"""
SWE-SQL+ Pipeline — CLI Entry Point
=====================================
Stages:
  0: Baseline   — submit issue_sql as-is
  1: CoT        — 4-step chain-of-thought single path
  2: N-path     — 3-way parallel (diagnostic/rewrite/minimal_fix) + serial test
  3: SQL-ACT    — ReAct agent with [EXPLORE]/[FIX], SAVEPOINT exploration

Usage:
    python run_pipeline.py --limit 10                           # quick test
    python run_pipeline.py --output results/v2_results.jsonl    # full run
    python run_pipeline.py --skip 100 --limit 50                # resume
    python run_pipeline.py --num_threads 2                      # parallel tasks

Config priority: CLI > env vars > config.json > defaults
"""

import sys
import json
import time
import os
import argparse
import warnings
from pathlib import Path

import pandas as pd
import psycopg2
import httpx
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import PipelineRunner

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    cfg = {
        "LLM_API_KEY": "YOUR_API_KEY_HERE",
        "LLM_BASE_URL": "https://maas.devops.xiaohongshu.com/v1",
        "LLM_MODEL": "deepseek-v3",
        "PG_HOST": "localhost",
        "PG_PORT": 5432,
        "PG_USER": "root",
        "PG_PASSWORD": "123123",
    }
    cfg_file = BASE_DIR / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg.update(json.load(f))
    for k in cfg:
        env_val = os.environ.get(k)
        if env_val:
            cfg[k] = int(env_val) if k == "PG_PORT" else env_val
    return cfg


CFG = load_config()


# ── LLM Client ───────────────────────────────────────────────────────────────
LLM_TIMEOUT = 180

llm_client = OpenAI(
    api_key=CFG["LLM_API_KEY"],
    base_url=CFG["LLM_BASE_URL"],
    timeout=httpx.Timeout(90.0, connect=15.0, read=90.0, write=30.0),
    http_client=httpx.Client(verify=False),
)

def _call_llm_raw(messages, max_tokens, temperature, thinking=False):
    """Raw LLM call — relies on httpx timeout for hard cutoff."""
    kwargs = {
        "model": CFG["LLM_MODEL"],
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Handle thinking mode for Qwen3 series (SiliconFlow)
    # Qwen3 models output reasoning_content by DEFAULT — must explicitly disable when not wanted.
    if "qwen" in CFG["LLM_MODEL"].lower():
        if thinking:
            # Enable thinking with budget control
            # Valid range: 128 <= thinking_budget <= 32768, default: 4096
            # For Stage 3 multi-turn (max_tokens>=8000): use 1/3 for thinking, 2/3 for output
            if max_tokens >= 8000:
                thinking_budget = min(max_tokens // 3, 2500)  # Multi-turn: more conservative
            else:
                thinking_budget = min(max_tokens // 2, 2000)  # Single-turn: balance it
            kwargs["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": thinking_budget,
            }
            print(f"[DEBUG] Thinking ON: thinking_budget={thinking_budget}, max_tokens={max_tokens}")
        else:
            # MUST explicitly disable thinking for Qwen3 — it's ON by default on SiliconFlow
            kwargs["extra_body"] = {
                "enable_thinking": False,
            }

    resp = llm_client.chat.completions.create(**kwargs)
    message = resp.choices[0].message

    # Extract thinking content ONLY when thinking mode is explicitly enabled
    result = ""
    if thinking and hasattr(message, 'reasoning_content') and message.reasoning_content:
        result += f"<think>\n{message.reasoning_content}\n</think>\n\n"

    # Append main content
    if message.content:
        result += message.content.strip()

    return result if result else ""


def call_llm(messages, max_tokens=2048, temperature=0.0, thinking=False):
    """Call LLM with retry. Supports thinking mode for Qwen3.5."""
    # Increase max_tokens when thinking is enabled to accommodate full thinking chain + output
    effective_max_tokens = max_tokens
    if thinking:
        effective_max_tokens = max(max_tokens, 4096)  # Ensure at least 4096 for thinking mode

    for attempt in range(3):
        try:
            return _call_llm_raw(messages, effective_max_tokens, temperature, thinking=thinking)
        except httpx.TimeoutException as e:
            print(f"    [LLM TIMEOUT attempt {attempt+1}/3] {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    [LLM error attempt {attempt+1}/3] {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return ""


# ── DB Connection ─────────────────────────────────────────────────────────────
def get_conn(db_name=None, max_retries=10, retry_interval=30):
    """Connect to PostgreSQL with auto-retry.

    If Docker restarts or the connection is temporarily lost, this will
    wait up to max_retries * retry_interval seconds before giving up.
    This prevents a transient Docker restart from killing the entire pipeline run.
    """
    kwargs = dict(
        host=CFG["PG_HOST"],
        port=int(CFG["PG_PORT"]),
        user=CFG["PG_USER"],
        password=CFG["PG_PASSWORD"],
        connect_timeout=15,
    )
    if db_name:
        kwargs["dbname"] = db_name

    for attempt in range(max_retries):
        try:
            return psycopg2.connect(**kwargs)
        except Exception as e:
            if attempt == 0:
                print(f"    [PG] Connection failed: {e}")
            if attempt < max_retries - 1:
                print(f"    [PG] Retrying in {retry_interval}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_interval)
            else:
                raise


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data_full():
    """Load the full 530-task dataset from postgresql_530.jsonl."""
    data_file = DATA_DIR / "postgresql_530.jsonl"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found.")
        print("Copy it from BIRD-CRITIC-1/baseline/data/postgresql_530.jsonl")
        sys.exit(1)

    records = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Data loaded: {len(records)} tasks (full BIRD-CRITIC-PG)")
    return records


def load_data_train():
    """Load the 451-task training split from train_530.jsonl (DB-level isolated)."""
    data_file = DATA_DIR / "train_530.jsonl"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found.")
        print("Expected at: data/train_530.jsonl (12 databases, 451 tasks)")
        sys.exit(1)

    records = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Data loaded: {len(records)} tasks (train split, 12 DBs, val excluded)")
    return records


def load_data_flash():
    """Load the 200-task flash dataset (legacy format)."""
    df = pd.read_parquet(DATA_DIR / "flash_dataset.parquet")

    sol_map = {}
    tc_map = {}
    with open(DATA_DIR / "bird-critic-1.0-flash_w_sol.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iid = d["instance_id"]
            sqls = d["sol_sql"]
            sol_map[iid] = sqls if isinstance(sqls, list) else [sqls]
            tc_map[iid] = d.get("test_cases", [])

    schema_map = {}
    schema_file = DATA_DIR / "flash_schema_full.jsonl"
    if schema_file.exists():
        with open(schema_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    schema_map[d["instance_id"]] = (
                        d.get("preprocess_schema") or d.get("original_schema", "")
                    )
                except Exception:
                    pass

    print(f"Data loaded: {len(df)} tasks (flash) | {len(sol_map)} solutions | {len(schema_map)} schemas")
    return df, sol_map, tc_map, schema_map


def build_instance_from_jsonl(record):
    """Build instance dict from a full JSONL record (postgresql_530.jsonl format)."""
    iid = record["instance_id"]
    issue_sql = record["issue_sql"]
    if isinstance(issue_sql, str):
        issue_sql = [issue_sql]

    sol_sql = record.get("sol_sql", [])
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql]

    pre_sqls = record.get("preprocess_sql") or []
    if isinstance(pre_sqls, str):
        pre_sqls = [pre_sqls]

    clean_sqls = record.get("clean_up_sql") or []
    if isinstance(clean_sqls, str):
        clean_sqls = [clean_sqls]

    issue_sql_text = "\n\n".join(issue_sql)

    return {
        "instance_id": iid,
        "db_id": record["db_id"],
        "db_name": record["db_id"] + "_template",
        "category": record.get("category", ""),
        "query": record["query"],
        "issue_sql": issue_sql,
        "issue_sql_text": issue_sql_text,
        "sol_sql": sol_sql,
        "test_cases": record.get("test_cases", []),
        "preprocess_sql": [s for s in pre_sqls if s and s.strip()],
        "clean_up_sql": [s for s in clean_sqls if s and s.strip()],
        "schema": record.get("preprocess_schema") or "",
    }


def build_instance_from_flash(row, sol_map, tc_map, schema_map):
    """Build instance dict from a flash parquet row + separate maps."""
    iid = row["instance_id"]
    issue_sqls = row["issue_sql"]
    issue_sql = issue_sqls[0] if hasattr(issue_sqls, "__len__") and len(issue_sqls) > 0 else str(issue_sqls)
    pre_sqls = list(row["preprocess_sql"]) if row.get("preprocess_sql") is not None else []
    clean_sqls = list(row["clean_up_sql"]) if row.get("clean_up_sql") is not None else []

    return {
        "instance_id": iid,
        "db_id": row["db_id"],
        "db_name": row["db_id"] + "_template",
        "category": row.get("category", ""),
        "query": row["query"],
        "issue_sql": issue_sql,
        "sol_sql": sol_map.get(iid, []),
        "test_cases": tc_map.get(iid, []),
        "preprocess_sql": [s for s in pre_sqls if s and s.strip()],
        "clean_up_sql": [s for s in clean_sqls if s and s.strip()],
        "schema": schema_map.get(iid, ""),
    }


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results_or_path, model):
    """Compute and print summary statistics.

    Accepts either:
    - a list of result dicts (legacy, used in parallel mode)
    - a Path/str pointing to the output JSONL file (memory-efficient, used in serial mode)

    Reading from file avoids keeping all result dicts in memory simultaneously,
    which is critical when running 400+ tasks with --thinking mode (each result
    can be hundreds of KB due to stage3_trajectory thought fields).
    """
    import pathlib

    if isinstance(results_or_path, (str, pathlib.Path)):
        # Stream from file — O(1) memory
        results_iter = []
        with open(results_or_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results_iter.append(json.loads(line))
                except Exception:
                    pass
        results = results_iter
    else:
        results = results_or_path

    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return {}

    s0 = sum(1 for r in results if r.get("stage0_pass"))
    s1 = sum(1 for r in results if r.get("stage1_pass"))
    s2 = sum(1 for r in results if r.get("stage2_pass"))
    s3 = sum(1 for r in results if r.get("stage3_pass"))
    final = sum(1 for r in results if r.get("first_pass_stage") is not None)

    cats = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in cats:
            cats[cat] = {"n": 0, "s0": 0, "s1": 0, "s2": 0, "s3": 0, "final": 0}
        cats[cat]["n"] += 1
        cats[cat]["s0"] += r.get("stage0_pass", False)
        cats[cat]["s1"] += r.get("stage1_pass", False)
        cats[cat]["s2"] += r.get("stage2_pass", False)
        cats[cat]["s3"] += r.get("stage3_pass", False)
        cats[cat]["final"] += (r.get("first_pass_stage") is not None)

    # First-pass stage distribution
    stage_dist = {}
    for r in results:
        fps = r.get("first_pass_stage") or "none"
        stage_dist[fps] = stage_dist.get(fps, 0) + 1

    # N-path selection distribution
    npath_dist = {}
    for r in results:
        sp = r.get("stage2_selected_path", "")
        if sp:
            npath_dist[sp] = npath_dist.get(sp, 0) + 1

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Pipeline Complete: {n} tasks | model={model}")
    print(sep)
    print(f"Stage 0  Baseline:   {s0/n*100:5.1f}%  ({s0}/{n})")
    print(f"Stage 1  CoT:        {s1/n*100:5.1f}%  ({s1}/{n})")
    print(f"Stage 2  N-path:     {s2/n*100:5.1f}%  ({s2}/{n})")
    print(f"Stage 3  SQL-ACT:    {s3/n*100:5.1f}%  ({s3}/{n})")
    print(f"{'─' * 70}")
    print(f"Final SR (any stage): {final/n*100:5.1f}%  ({final}/{n})")
    print(f"Paper DeepSeek-V3:    27.74%")
    print(f"Delta vs paper:       {final/n*100-27.74:+.1f}pp")
    print(f"{'─' * 70}")
    print("First-pass stage distribution:")
    for stage, cnt in sorted(stage_dist.items()):
        print(f"  {stage:<10s}: {cnt:3d}  ({cnt/n*100:.1f}%)")
    print(f"{'─' * 70}")
    print("By category:")
    for cat, v in sorted(cats.items()):
        nn = v["n"]
        print(
            f"  {cat:<20s}  S0={v['s0']/nn*100:4.0f}%  S1={v['s1']/nn*100:4.0f}%  "
            f"S2={v['s2']/nn*100:4.0f}%  S3={v['s3']/nn*100:4.0f}%  "
            f"Final={v['final']/nn*100:4.0f}%  (n={nn})"
        )
    if npath_dist:
        print(f"{'─' * 70}")
        print("N-path selection distribution:")
        for path, cnt in sorted(npath_dist.items(), key=lambda x: -x[1]):
            print(f"  {path:<25s}: {cnt}")
    print(sep)

    return {
        "model": model,
        "version": "v3-sqlact",
        "sample_size": n,
        "stage0_sr": round(s0 / n * 100, 2),
        "stage1_sr": round(s1 / n * 100, 2),
        "stage2_sr": round(s2 / n * 100, 2),
        "stage3_sr": round(s3 / n * 100, 2),
        "final_sr": round(final / n * 100, 2),
        "delta_vs_paper_pp": round(final / n * 100 - 27.74, 2),
        "first_pass_stage_distribution": stage_dist,
        "npath_selection_distribution": npath_dist,
        "by_category": {
            cat: {
                "n": v["n"],
                "stage0_sr": round(v["s0"] / v["n"] * 100, 1),
                "stage1_sr": round(v["s1"] / v["n"] * 100, 1),
                "stage2_sr": round(v["s2"] / v["n"] * 100, 1),
                "stage3_sr": round(v["s3"] / v["n"] * 100, 1),
                "final_sr": round(v["final"] / v["n"] * 100, 1),
            }
            for cat, v in cats.items()
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SWE-SQL+ Pipeline")
    parser.add_argument("--dataset", type=str, default="train",
                        choices=["train", "full", "flash"],
                        help="Dataset: 'train' = 451 tasks (train_530.jsonl, DB-isolated train split), "
                             "'full' = 530 tasks (postgresql_530.jsonl), "
                             "'flash' = 200 tasks (flash_dataset.parquet)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of tasks to process")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N tasks (for resuming)")
    parser.add_argument("--output", type=str, default="results/v2_results.jsonl",
                        help="Output JSONL path (relative to project dir)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Parallel task threads (default 1, use 2 max)")
    parser.add_argument("--skip-stage3", action="store_true",
                        help="Skip Stage 3 SQL-ACT agent (faster initial run)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for Qwen3.5 (v5.0b experimental)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"SWE-SQL+ Pipeline  model={CFG['LLM_MODEL']}  PG={CFG['PG_HOST']}:{CFG['PG_PORT']}")
    print(f"Stages: Baseline -> CoT -> N-path(x3) -> SQL-ACT(EXPLORE/FIX)")
    thinking_mode = "ON (v5.0b)" if args.thinking else "OFF (v5.0a)"
    print(f"Thinking mode: {thinking_mode}")
    print("=" * 70)

    # Check PG connection
    try:
        conn = get_conn()
        conn.close()
        print("PostgreSQL connection OK")
    except Exception as e:
        print(f"PostgreSQL connection FAILED: {e}")
        print("Start the Docker container first — see scripts/setup_docker.sh")
        sys.exit(1)

    # Auto-adjust output path based on dataset + thinking mode
    if "v2_results" in args.output:
        tag = "v5_0b_thinking" if args.thinking else "v5_0a_nothinking"
        out_path_str = args.output.replace("v2_results", tag)
    else:
        out_path_str = args.output

    out_path = BASE_DIR / out_path_str
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Auto-resume: load already-completed instance_ids from output file ──
    completed_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    iid = rec.get("instance_id")
                    if iid:
                        completed_ids.add(iid)
                except Exception:
                    pass
        if completed_ids:
            print(f"[RESUME] Found {len(completed_ids)} already-completed tasks in {out_path.name}")
            print(f"[RESUME] These will be skipped automatically.\n")

    # Load data — train (451) / full (530) / flash (200)
    if args.dataset == "train":
        records = load_data_train()
        records = records[args.skip:]
        if args.limit:
            records = records[:args.limit]
        instances = [build_instance_from_jsonl(r) for r in records]
    elif args.dataset == "full":
        records = load_data_full()
        records = records[args.skip:]
        if args.limit:
            records = records[:args.limit]
        instances = [build_instance_from_jsonl(r) for r in records]
    else:
        df, sol_map, tc_map, schema_map = load_data_flash()
        df_run = df.iloc[args.skip:]
        if args.limit:
            df_run = df_run.iloc[:args.limit]
        instances = [build_instance_from_flash(row, sol_map, tc_map, schema_map)
                     for _, row in df_run.iterrows()]

    # Filter out already-completed instances
    if completed_ids:
        instances_before = len(instances)
        instances = [inst for inst in instances if inst["instance_id"] not in completed_ids]
        print(f"[RESUME] Filtered {instances_before - len(instances)} completed tasks, "
              f"{len(instances)} remaining.\n")

    print(f"Processing: {len(instances)} tasks (skip={args.skip}, limit={args.limit})\n")

    # Initialize pipeline runner
    # Pass thinking flag to call_llm via lambda.
    # kw.get("thinking", args.thinking) lets callers (e.g. _call_llm_no_thinking)
    # override the default thinking flag by passing thinking=False explicitly.
    runner = PipelineRunner(
        call_llm_fn=lambda messages, **kw: call_llm(
            messages,
            thinking=kw.pop("thinking", args.thinking),
            **kw,
        ),
        get_conn_fn=get_conn,
        cfg=CFG,
        skip_stage3=args.skip_stage3,
    )

    # Serial mode: do NOT accumulate result dicts in memory.
    # Each result (with --thinking) can be hundreds of KB due to stage3_trajectory
    # thought fields. Keeping 451 results in RAM = potential OOM on 32 GB machines.
    # Instead: write to file immediately, release the dict, read back for summary.
    results = []  # only used in parallel mode (small num_threads, short runs)

    if args.num_threads <= 1:
        # Serial mode — memory-efficient: write & release each result immediately
        total_global = args.skip + len(instances) + len(completed_ids)
        for idx, inst in enumerate(instances):
            iid = inst["instance_id"]
            done_so_far = len(completed_ids) + idx + 1
            print(f"[{done_so_far:3d}/{total_global}] id={iid} db={inst['db_id']} cat={inst['category']}")

            try:
                r = runner.run_one(inst)

                # Summary line (detail already printed by pipeline.py)
                fps = r.get("first_pass_stage") or "NONE"
                print(
                    f"  -> first_pass={fps}  "
                    f"S0={'P' if r.get('stage0_pass') else 'F'} "
                    f"S1={'P' if r.get('stage1_pass') else 'F'} "
                    f"S2={'P' if r.get('stage2_pass') else 'F'} "
                    f"S3={'P' if r.get('stage3_pass') else 'F'}\n"
                )
                with open(out_path, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                # Release the large result dict immediately — do NOT append to results[]
                del r

            except Exception as e:
                print(f"  -> ERROR: {e}\n")
                error_result = {
                    "instance_id": iid,
                    "db_id": inst["db_id"],
                    "category": inst["category"],
                    "error": str(e),
                    "first_pass_stage": None,
                }
                with open(out_path, "a") as f:
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        # Summary: read from file (not from in-memory results list)
        summary = print_summary(out_path, CFG["LLM_MODEL"])

    else:
        # Parallel mode
        import concurrent.futures
        print(f"Parallel mode: {args.num_threads} threads")

        def process_one(inst):
            return runner.run_one(inst)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as exe:
            future_map = {exe.submit(process_one, inst): i for i, inst in enumerate(instances)}
            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                try:
                    r = future.result()
                    results.append(r)
                    fps = r.get("first_pass_stage") or "NONE"
                    print(f"[{idx+1:3d}/{len(instances)}] id={r['instance_id']} first_pass={fps}")
                    with open(out_path, "a") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[{idx+1:3d}] ERROR: {e}")

        # Parallel mode: results list is already populated, use it directly
        summary = print_summary(results, CFG["LLM_MODEL"])

    # Auto-adjust summary path based on thinking mode
    if args.thinking:
        summary_path = RESULTS_DIR / "v5_0b_thinking_summary.json"
    else:
        summary_path = RESULTS_DIR / "v5_0a_nothinking_summary.json"

    # Warn if running full/flash dataset (not the intended train split)
    if args.dataset != "train":
        print(f"[WARN] Running on '{args.dataset}' dataset, not the train split. "
              f"Use --dataset train for SFT data collection.")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
