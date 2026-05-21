#!/usr/bin/env python3
"""
extract_sft_B.py — 从 v5_0b_thinking.jsonl 提取 sft_B_single 和 sft_B_multi

路径 B (thinking)：
  - sft_B_single: 取 stage1_response 的 <think> 推理链，但把最后的 SQL 代码块
                  替换为 final_sql（保证 SQL 与 Pipeline 对齐，匹配率 ~100%）
  - sft_B_multi:  直接从 stage3_trajectory 提取多轮对话

对齐要点：
  1. 遍历对象：success = [r for r in records if r.get("first_pass_stage") is not None]
  2. single 过滤：final_sql 非空 + stage1_response 非空
  3. multi 过滤：有 stage3_trajectory + 每步有 thought（无 thought 则整条跳过，不截断）
  4. schema 注入：通过 schema_map 按 instance_id 查找
  5. SQL 对齐：用 final_sql 替换 stage1_response 最后一个 ```sql...``` 代码块

用法：
  python src/sft/extract_sft_B.py
  python src/sft/extract_sft_B.py --input results/pipeline/v5_0b_thinking.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

# 项目根目录（src/sft/ 的上两级）
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "sft"
PIPELINE_DIR = BASE_DIR / "results" / "pipeline"
DATA_DIR = BASE_DIR / "data"


# ── Schema loader（与 build_sft_data.py 完全一致）────────────────────────────
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


# ── Instruction builder（与 build_sft_data.py _make_instruction 完全一致）────
def _make_instruction(schema, query, issue_sql):
    if isinstance(issue_sql, list):
        issue_sql = "\n".join(issue_sql)
    return (
        f"## Database Schema:\n{schema}\n\n"
        f"## User Issue:\n{query}\n\n"
        f"## Faulty SQL:\n```sql\n{issue_sql}\n```\n\n"
        "Fix the buggy SQL. Think step by step, then output the corrected SQL."
    )


# ── sft_B_single ─────────────────────────────────────────────────────────────
def replace_last_sql_block(response: str, final_sql: str) -> str:
    """把 stage1_response 最后一个 ```sql...``` 代码块替换为 final_sql。

    保留 <think> 推理链，只对齐最终答案 SQL，确保与 Pipeline 的 final_sql 一致。
    若 response 中没有 sql 代码块，则直接在末尾追加。
    """
    if isinstance(final_sql, list):
        final_sql = "\n".join(str(s) for s in final_sql)
    final_sql = final_sql.strip()

    # 找到所有 ```sql...``` 代码块的位置
    pattern = re.compile(r'```sql\s*[\s\S]*?```', re.IGNORECASE)
    matches = list(pattern.finditer(response))

    if matches:
        # 替换最后一个 sql 代码块
        last = matches[-1]
        new_block = f"```sql\n{final_sql}\n```"
        return response[:last.start()] + new_block + response[last.end():]
    else:
        # 没有 sql 代码块，直接追加
        return response.rstrip() + f"\n\n```sql\n{final_sql}\n```"


def extract_single_turn_B(record, schema):
    """路径 B single：取 stage1_response 的 <think> 推理链，
    但把最后的 SQL 代码块替换为 final_sql，保证 SQL 与 Pipeline 对齐。
    """
    final_sql = record.get("final_sql", "")
    if not final_sql:
        return None
    stage1_response = record.get("stage1_response", "")
    if not stage1_response:
        return None

    # 用 final_sql 替换 stage1_response 最后一个 sql 代码块
    aligned_output = replace_last_sql_block(stage1_response, final_sql)

    return {
        "instruction": _make_instruction(
            schema,
            record.get("query", ""),
            record.get("stage0_sql", "")
        ),
        "output": aligned_output,
        "source_stage": record.get("first_pass_stage", ""),
        "source_tag": "thinking",
        "db_id": record.get("db_id", ""),
        "category": record.get("category", ""),
        "instance_id": record.get("instance_id"),
    }


# ── sft_B_multi（与 build_sft_data.py build_multiturn_sample 完全一致）─────────
def build_multiturn_sample(record, schema):
    """从 stage3_trajectory 提取多轮 SFT 样本。

    与 build_sft_data.py 的 build_multiturn_sample 完全一致：
    - 遇到 thought 为空的 step → 整条返回 None（不截断保留）
    - 最后一步的 observation 若为 "test_case: PASS" 则不加入 user 消息
    """
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
            return None  # 整条跳过（与 build_sft_data.py 一致）

        # Assistant turn：含推理链（build_sft_data_与论文对齐分析.md 改动③）
        assistant_content = f"[{action_type}]\n<think>\n{thought}\n\n\n```sql\n{sql}\n```"
        messages.append({"role": "assistant", "content": assistant_content})

        # User turn（observation）：最后一步的 PASS 不加
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
        "stage3_pass": record.get("stage3_pass"),
    }


def extract_multi_turn(record, schema, source_tag):
    """与 build_sft_data.py extract_multi_turn 完全一致。"""
    trajectory = record.get("stage3_trajectory", [])
    if not trajectory:
        return None
    mt = build_multiturn_sample(record, schema)
    if mt is None:
        return None
    mt["source_tag"] = source_tag
    return mt


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract sft_B_single and sft_B_multi from v5_0b_thinking.jsonl\n"
                    "完全对齐 build_sft_data.py 的 build_four_files() B 路径逻辑"
    )
    parser.add_argument(
        "--input",
        default=str(PIPELINE_DIR / "v5_0b_thinking.jsonl"),
        help="Input JSONL file (default: results/pipeline/v5_0b_thinking.jsonl)"
    )
    parser.add_argument(
        "--out-single",
        default=str(RESULTS_DIR / "sft_B_single.jsonl"),
        help="Output single-turn file (default: results/sft/sft_B_single.jsonl)"
    )
    parser.add_argument(
        "--out-multi",
        default=str(RESULTS_DIR / "sft_B_multi.jsonl"),
        help="Output multi-turn file (default: results/sft/sft_B_multi.jsonl)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_single = Path(args.out_single)
    out_multi = Path(args.out_multi)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    # 加载 schema 映射
    print("Loading schema map...")
    schema_map = load_schema_map()
    print(f"  Loaded {len(schema_map)} schemas")

    # 读取输入
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {input_path.name}")

    # 与 build_four_files 一致：只处理 first_pass_stage is not None 的记录
    success = [r for r in records if r.get("first_pass_stage") is not None]
    print(f"\n[thinking] total={len(records)}, success={len(success)}")

    # 清空输出文件
    out_single.parent.mkdir(parents=True, exist_ok=True)
    out_multi.parent.mkdir(parents=True, exist_ok=True)
    out_single.unlink(missing_ok=True)
    out_multi.unlink(missing_ok=True)

    n_single = n_multi = 0

    for idx, record in enumerate(success):
        iid = record.get("instance_id")
        schema = schema_map.get(iid, "")

        # ── sft_B_single ──────────────────────────────────────────────────────
        s = extract_single_turn_B(record, schema)
        if s:
            with open(out_single, "a", encoding="utf-8") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_single += 1

        # ── sft_B_multi ───────────────────────────────────────────────────────
        m = extract_multi_turn(record, schema, "thinking")
        if m:
            with open(out_multi, "a", encoding="utf-8") as f:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
            n_multi += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(success)}] single={n_single}, multi={n_multi}")

    # ── 统计 ──────────────────────────────────────────────────────────────────
    single_samples, multi_samples = [], []
    if out_single.exists():
        with open(out_single, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    single_samples.append(json.loads(line))
    if out_multi.exists():
        with open(out_multi, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    multi_samples.append(json.loads(line))

    print(f"\n{'=' * 50}")
    print(f"  → sft_B_single.jsonl: {n_single} samples")
    if single_samples:
        cat = Counter(s.get("category", "?") for s in single_samples)
        stage = Counter(s.get("source_stage", "?") for s in single_samples)
        print(f"     By category: {dict(cat)}")
        print(f"     By stage:    {dict(stage)}")

    print(f"  → sft_B_multi.jsonl:  {n_multi} samples")
    if multi_samples:
        cat2 = Counter(s.get("category", "?") for s in multi_samples)
        msg_counts = [len(s["messages"]) for s in multi_samples]
        print(f"     By category: {dict(cat2)}")
        print(f"     Messages per sample: min={min(msg_counts)} max={max(msg_counts)} avg={sum(msg_counts)/len(msg_counts):.1f}")

    print(f"\nOutput:")
    print(f"  {out_single}")
    print(f"  {out_multi}")
    print("[Done]")


if __name__ == "__main__":
    main()
