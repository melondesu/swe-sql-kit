"""
SFT Data Builder — Extract training samples from pipeline results
==================================================================
Reads v2_results.jsonl (new format with stage3_trajectory) and produces
SFT training data in two formats:
  1. Single-turn: instruction + output (with <think> reasoning chain)
  2. Multi-turn: messages array with EXPLORE/FIX conversation turns

Usage:
    # 原始模式（单文件输出）
    python src/sft/build_sft_data.py --input results/pipeline/v5_0a_nothinking.jsonl --output results/sft/sft_data.jsonl
    python src/sft/build_sft_data.py --include_multiturn   # include multi-turn EXPLORE/FIX samples

    # 生成 4 个 SFT 文件（对应微调指南规划的 A/B x single/multi）
    python src/sft/build_sft_data.py --build_four_files
    # 路径 A (nothinking): results/sft/sft_A_single.jsonl + results/sft/sft_A_multi.jsonl
    # 路径 B (thinking):   results/sft/sft_B_single.jsonl + results/sft/sft_B_multi.jsonl
"""

import sys
import json
import time
import os
import argparse
from pathlib import Path

import httpx
from openai import OpenAI

# 项目根目录（src/sft/ 的上两级）
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "sft"
PIPELINE_DIR = BASE_DIR / "results" / "pipeline"
DATA_DIR = BASE_DIR / "data"


# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    cfg = {
        "LLM_API_KEY": "YOUR_API_KEY_HERE",
        "LLM_BASE_URL": "https://maas.devops.xiaohongshu.com/v1",
        "LLM_MODEL": "deepseek-v3",
    }
    cfg_file = BASE_DIR / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg.update(json.load(f))
    for k in cfg:
        env_val = os.environ.get(k)
        if env_val:
            cfg[k] = env_val
    return cfg


CFG = load_config()

llm = OpenAI(
    api_key=CFG["LLM_API_KEY"],
    base_url=CFG["LLM_BASE_URL"],
    http_client=httpx.Client(verify=False),
)


def call_llm(messages, max_tokens=2000, temperature=0.3):
    for attempt in range(3):
        try:
            resp = llm.chat.completions.create(
                model=CFG["LLM_MODEL"],
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            print(f"  [LLM retry {attempt+1}/3] {e}")
            time.sleep(2 ** attempt)
    return ""


# ── Schema loader ─────────────────────────────────────────────────────────────
def load_schema_map():
    schema_map = {}
    for fname in ["postgresql_530.jsonl", "flash_schema_full.jsonl"]:
        schema_file = DATA_DIR / fname
        if not schema_file.exists():
            continue
        with open(schema_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    iid = d.get("instance_id")
                    if iid and iid not in schema_map:
                        schema_map[iid] = (
                            d.get("preprocess_schema") or d.get("original_schema", "")
                        )
                except Exception:
                    pass
    return schema_map


# ── f-Plan Prompt ─────────────────────────────────────────────────────────────
FPLAN_SYS = (
    "You are an expert SQL debugging teacher. "
    "Your ONLY job is to output a structured debugging reasoning chain (f-Plan) in the EXACT format specified. "
    "Do NOT deviate from the format. Do NOT add extra commentary before or after the format."
)


def fplan_prompt(schema, query, issue_sql, final_sql):
    return [
        {"role": "system", "content": FPLAN_SYS},
        {"role": "user", "content": f"""## Database Schema
{schema}

## User Question
{query}

## Buggy SQL (the input with errors)
```sql
{issue_sql}
```

## Correct Solution SQL (the target output)
```sql
{final_sql}
```

You MUST output your response in this EXACT format — no deviations, no extra text before or after:

Step 1 ERROR ANALYSIS: [Classify the error type: Runtime/Logic/Schema/Incomplete. Describe what is wrong with the buggy SQL, referencing actual table/column names.]

Step 2 ROOT CAUSE: [One concise sentence identifying the root cause.]

Step 3 FIX PLAN: [Describe the fix strategy: what needs to change and why, referencing actual table/column names.]

```sql
{final_sql}
```

CRITICAL RULES:
- Start your response DIRECTLY with "Step 1 ERROR ANALYSIS:" — no preamble.
- End your response with the sql code block above — nothing after it.
- Do NOT wrap in <think> tags.
- Do NOT omit the sql code block."""},
    ]


# ── Multi-turn SFT from trajectory ───────────────────────────────────────────
def build_multiturn_sample(record, schema):
    """Build a multi-turn SFT sample from stage3_trajectory."""
    trajectory = record.get("stage3_trajectory", [])
    if not trajectory:
        return None

    query = record.get("query", "")
    issue_sql = record.get("stage0_sql", record.get("issue_sql", ""))
    if isinstance(issue_sql, list):
        issue_sql = "\n".join(issue_sql)

    messages = [
        {
            "role": "user",
            "content": (
                f"## Database Schema:\n{schema}\n\n"
                f"## User Issue:\n{query}\n\n"
                f"## Faulty SQL:\n```sql\n{issue_sql}\n```"
            ),
        }
    ]

    for step in trajectory:
        action_type = step.get("action_type", "FIX")
        sql = step.get("sql", "")
        observation = step.get("observation", "")

        thought = step.get("thought", "")
        if not thought or not thought.strip():
            return None  # Skip truncated samples with no reasoning chain

        # Assistant turn: include reasoning chain for multi-turn SFT
        assistant_content = f"[{action_type}]\n<think>\n{thought}\n\n\n```sql\n{sql}\n```"
        messages.append({"role": "assistant", "content": assistant_content})

        # User turn (observation) — skip for the last step if it's a PASS
        if observation and observation != "test_case: PASS":
            messages.append({
                "role": "user",
                "content": f"## Observation:\n{observation}\n\nContinue debugging.",
            })

    return {
        "messages": messages,
        "source_stage": "stage3_multiturn",
        "db_id": record.get("db_id", ""),
        "category": record.get("category", ""),
        "instance_id": record.get("instance_id"),
    }


# ── Single-turn extractors ────────────────────────────────────────────────────
def _make_instruction(schema, query, issue_sql):
    if isinstance(issue_sql, list):
        issue_sql = "\n".join(issue_sql)
    return (
        f"## Database Schema:\n{schema}\n\n"
        f"## User Issue:\n{query}\n\n"
        f"## Faulty SQL:\n```sql\n{issue_sql}\n```\n\n"
        "Fix the buggy SQL. Think step by step, then output the corrected SQL."
    )


def extract_single_turn_B(record, schema):
    """路径 B (thinking)：直接取 stage1_response，含真实 <think> 推理链。"""
    final_sql = record.get("final_sql", "")
    if not final_sql:
        return None
    stage1_response = record.get("stage1_response", "")
    if not stage1_response:
        return None

    return {
        "instruction": _make_instruction(schema, record.get("query", ""), record.get("stage0_sql", "")),
        "output": stage1_response,
        "source_stage": record.get("first_pass_stage", ""),
        "source_tag": "thinking",
        "db_id": record.get("db_id", ""),
        "category": record.get("category", ""),
        "instance_id": record.get("instance_id"),
    }


def extract_single_turn_A(record, schema):
    """路径 A (nothinking)：调 LLM 从正确答案逆向生成 f-Plan 推理链。"""
    final_sql = record.get("final_sql", "")
    if not final_sql:
        return None
    # final_sql 有时是 list（pipeline 存储格式），需要展开为字符串
    if isinstance(final_sql, list):
        final_sql = "\n".join(str(s) for s in final_sql)

    query = record.get("query", "")
    issue_sql = record.get("stage0_sql", "")
    if isinstance(issue_sql, list):
        issue_sql = "\n".join(issue_sql)

    messages = fplan_prompt(schema, query, issue_sql, final_sql)
    fplan_response = call_llm(messages, temperature=0.3)
    if not fplan_response:
        return None

    return {
        "instruction": _make_instruction(schema, query, issue_sql),
        "output": fplan_response,
        "source_stage": record.get("first_pass_stage", ""),
        "source_tag": "nothinking",
        "db_id": record.get("db_id", ""),
        "category": record.get("category", ""),
        "instance_id": record.get("instance_id"),
    }


def extract_multi_turn(record, schema, source_tag):
    """从 stage3_trajectory 提取多轮 SFT 样本。"""
    trajectory = record.get("stage3_trajectory", [])
    if not trajectory:
        return None
    mt = build_multiturn_sample(record, schema)
    if mt is None:
        return None
    mt["source_tag"] = source_tag
    return mt


def build_four_files(only_A=False):
    """生成 4 个 SFT 文件：sft_A_single / sft_A_multi / sft_B_single / sft_B_multi。

    A = v5_0a_nothinking.jsonl（non-thinking pipeline）
      - sft_A_single: 调 LLM 逆向生成 f-Plan 推理链（需要 API）
      - sft_A_multi:  直接从 stage3_trajectory 提取（无需 API）

    B = v5_0b_thinking.jsonl（thinking pipeline）
      - sft_B_single: 直接取 stage1_response（含真实 <think> 推理链，无需 API）
      - sft_B_multi:  直接从 stage3_trajectory 提取（无需 API）

    only_A: 若为 True，只处理 A 路径（跳过 B）
    """
    schema_map = load_schema_map()

    configs = [
        {
            "input":      PIPELINE_DIR / "v5_0a_nothinking.jsonl",
            "out_single": RESULTS_DIR / "sft_A_single.jsonl",
            "out_multi":  RESULTS_DIR / "sft_A_multi.jsonl",
            "tag":        "nothinking",
            "use_llm":    True,   # A 路径需要调 LLM 生成 f-Plan
        },
        {
            "input":      PIPELINE_DIR / "v5_0b_thinking.jsonl",
            "out_single": RESULTS_DIR / "sft_B_single.jsonl",
            "out_multi":  RESULTS_DIR / "sft_B_multi.jsonl",
            "tag":        "thinking",
            "use_llm":    False,  # B 路径直接取 stage1_response
        },
    ]

    if only_A:
        configs = configs[:1]  # 只保留 A 路径

    for cfg in configs:
        if not cfg["input"].exists():
            print(f"[SKIP] {cfg['input']} not found, skipping.")
            continue

        records = []
        with open(cfg["input"]) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        success = [r for r in records if r.get("first_pass_stage") is not None]
        needs_llm = cfg["use_llm"]
        print(f"\n[{cfg['tag']}] total={len(records)}, success={len(success)}"
              f"{'  (will call LLM for f-Plan)' if needs_llm else ''}")

        n_single = n_multi = 0
        cfg["out_single"].parent.mkdir(parents=True, exist_ok=True)
        cfg["out_multi"].parent.mkdir(parents=True, exist_ok=True)
        cfg["out_single"].unlink(missing_ok=True)
        cfg["out_multi"].unlink(missing_ok=True)

        for idx, record in enumerate(success):
            iid = record.get("instance_id")
            schema = schema_map.get(iid, "")

            # 单轮：A 路径调 LLM，B 路径直接提取
            if needs_llm:
                s = extract_single_turn_A(record, schema)
                time.sleep(0.2)  # 避免 API 限速
            else:
                s = extract_single_turn_B(record, schema)

            if s:
                with open(cfg["out_single"], "a") as f:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_single += 1

            # 多轮：A/B 路径相同，均从 stage3_trajectory 提取
            m = extract_multi_turn(record, schema, cfg["tag"])
            if m:
                with open(cfg["out_multi"], "a") as f:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                n_multi += 1

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(success)}] single={n_single}, multi={n_multi}")

        print(f"  → {cfg['out_single'].name}: {n_single} samples")
        print(f"  → {cfg['out_multi'].name}: {n_multi} samples")

    print("\n[Done] 4 SFT files generated in results/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SFT Data Builder")
    parser.add_argument("--input", default="results/v2_results.jsonl")
    parser.add_argument("--output", default="results/sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include_multiturn", action="store_true",
                        help="Include multi-turn EXPLORE/FIX conversation samples")
    parser.add_argument("--build_four_files", action="store_true",
                        help="生成 4 个 SFT 文件 (sft_A/B_single/multi.jsonl)")
    parser.add_argument("--only_A", action="store_true",
                        help="配合 --build_four_files 使用：只处理 A 路径 (v5_0a_nothinking.jsonl)，跳过 B 路径")
    args = parser.parse_args()

    # ── 新模式：直接生成 4 个文件 ──────────────────────────────────────────────
    if args.build_four_files:
        build_four_files(only_A=args.only_A)
        return

    input_path = BASE_DIR / args.input
    output_path = BASE_DIR / args.output

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run the pipeline first: python run_pipeline.py --limit 200")
        sys.exit(1)

    # Load results
    all_records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    # Filter: only records that passed at some stage
    success_records = [r for r in all_records if r.get("first_pass_stage") is not None]

    schema_map = load_schema_map()

    print(f"Total records: {len(all_records)}")
    print(f"Successful records: {len(success_records)}")

    if args.limit:
        success_records = success_records[:args.limit]

    # Category distribution
    cat_dist = {}
    stage_dist = {}
    for r in success_records:
        c = r.get("category", "Unknown")
        s = r.get("first_pass_stage", "unknown")
        cat_dist[c] = cat_dist.get(c, 0) + 1
        stage_dist[s] = stage_dist.get(s, 0) + 1
    print(f"By category: {cat_dist}")
    print(f"By first-pass stage: {stage_dist}")
    print()

    sft_samples = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    for idx, record in enumerate(success_records):
        iid = record.get("instance_id")
        schema = schema_map.get(iid, "")
        issue_sql = record.get("stage0_sql", "")
        final_sql = record.get("final_sql", "")
        first_pass = record.get("first_pass_stage", "")
        query = record.get("query", "")

        if not final_sql:
            continue
        if isinstance(final_sql, list):
            final_sql = "\n".join(str(s) for s in final_sql)
        if isinstance(issue_sql, list):
            issue_sql = "\n".join(issue_sql)

        print(f"[{idx+1:3d}/{len(success_records)}] id={iid} cat={record.get('category','')} stage={first_pass}")

        # 1. Single-turn f-Plan sample
        messages = fplan_prompt(schema, query, issue_sql, final_sql)
        fplan_response = call_llm(messages, temperature=0.3)

        if fplan_response:
            sample = {
                "instruction": (
                    f"## Database Schema:\n{schema}\n\n"
                    f"## User Issue:\n{query}\n\n"
                    f"## Faulty SQL:\n```sql\n{issue_sql}\n```\n\n"
                    "Fix the buggy SQL. Think step by step, then output the corrected SQL."
                ),
                "output": fplan_response,
                "source_stage": first_pass,
                "db_id": record.get("db_id", ""),
                "category": record.get("category", ""),
                "instance_id": iid,
            }
            sft_samples.append(sample)
            with open(output_path, "a") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  + single-turn sample ({len(fplan_response)} chars)")

        # 2. Multi-turn sample (if requested and has trajectory)
        if args.include_multiturn and first_pass == "stage3":
            mt_sample = build_multiturn_sample(record, schema)
            if mt_sample and len(mt_sample["messages"]) > 2:
                sft_samples.append(mt_sample)
                with open(output_path, "a") as f:
                    f.write(json.dumps(mt_sample, ensure_ascii=False) + "\n")
                print(f"  + multi-turn sample ({len(mt_sample['messages'])} turns)")

        time.sleep(0.2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SFT data build complete!")
    print(f"  Input success records: {len(success_records)}")
    print(f"  Generated samples: {len(sft_samples)}")
    print(f"  Output: {output_path}")

    type_dist = {}
    for s in sft_samples:
        t = s.get("source_stage", "unknown")
        if "messages" in s:
            t = "multiturn"
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"  Sample types: {type_dist}")

    if len(sft_samples) >= 150:
        print(f"  Target met: {len(sft_samples)} >= 150 samples")
    else:
        print(f"  Below target: {len(sft_samples)} < 150. Run pipeline on more tasks.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
