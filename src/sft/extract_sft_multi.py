#!/usr/bin/env python3
"""
extract_sft_multi.py — 专用 sft_A_multi 数据提取脚本

从 v5_0a_nothinking.jsonl（Pipeline 产出）中提取多轮对话 SFT 数据。

核心改进：
  - 截断保留：遇到缺 thought 的 step 就截断，不丢弃整条
  - 可纳入失败的 Stage 3 样本（--include-failed）
  - 输出 ShareGPT 格式，直接用于 LLaMA-Factory 训练

用法：
  python extract_sft_multi.py --input v5_0a_nothinking.jsonl
  python extract_sft_multi.py --input v5_0a_nothinking.jsonl --include-failed
  python extract_sft_multi.py --input v5_0a_nothinking.jsonl --include-failed --output sft_A_multi.jsonl
"""

import argparse
import json
import sys
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Extract sft_A_multi from Pipeline output")
    parser.add_argument("--input", required=True, help="Pipeline JSONL file path")
    parser.add_argument("--output", default="sft_A_multi.jsonl", help="Output JSONL file path")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include Stage 3 failed samples (trajectory is still useful for SFT)")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum assistant turns required (default: 2)")
    return parser.parse_args()


def extract_multiturn(record, include_failed=False, min_turns=2):
    """Extract multi-turn SFT sample from a single Pipeline record.

    Returns dict (ShareGPT format) or None if not extractable.

    Strategy:
    - Only process Stage 3 instances (has stage3_trajectory)
    - By default: only include passed samples (first_pass_stage == 'stage3')
    - With --include-failed: include all Stage 3 instances
    - Truncate at first step missing thought (don't discard entire sample)
    - Skip last observation if it's "test_case: PASS"
    """
    trajectory = record.get("stage3_trajectory", [])
    if not trajectory:
        return None

    # Filter: only Stage 3 instances
    first_pass = record.get("first_pass_stage")
    if not include_failed and first_pass != "stage3":
        return None

    schema = record.get("schema", "")
    query = record.get("query", "")
    issue_sql = record.get("stage0_sql", record.get("issue_sql", ""))
    if isinstance(issue_sql, list):
        issue_sql = "\n".join(issue_sql)

    # Build initial user message
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

    assistant_turns = 0
    truncated_at = None

    for i, step in enumerate(trajectory):
        action_type = step.get("action_type", "FIX")
        sql = step.get("sql", "")
        observation = step.get("observation", "")
        thought = step.get("thought", "")

        # Truncate at first step missing thought (don't discard entire sample)
        if not thought or not thought.strip():
            truncated_at = i
            break

        # Assistant turn: [action_type] + thought + SQL
        assistant_content = f"[{action_type}]\n<think>\n{thought}\n\n\n```sql\n{sql}\n```"
        messages.append({"role": "assistant", "content": assistant_content})
        assistant_turns += 1

        # User turn (observation) — skip for the last step
        # Don't add observation if it's the final step (no more debugging needed)
        is_last = (i == len(trajectory) - 1)
        if not is_last and observation and observation != "test_case: PASS":
            messages.append({
                "role": "user",
                "content": f"## Observation:\n{observation}\n\nContinue debugging.",
            })

    # Minimum turns check: need at least 1 assistant turn to be useful
    if assistant_turns < 1:
        return None
    if assistant_turns < min_turns:
        # Still include if it has at least 1 turn, but log a warning
        pass

    # Build result
    result = {
        "messages": messages,
        "source_stage": "stage3_multiturn",
        "db_id": record.get("db_id", ""),
        "category": record.get("category", ""),
        "instance_id": record.get("instance_id", ""),
        "stage3_pass": record.get("stage3_pass", False),
        "truncated": truncated_at is not None,
        "truncated_at_step": truncated_at,
        "assistant_turns": assistant_turns,
    }

    return result


def main():
    args = parse_args()

    # Load Pipeline output
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.input}")

    # Filter Stage 3 instances
    s3_records = [r for r in records if r.get("stage3_trajectory")]
    s3_passed = [r for r in s3_records if r.get("first_pass_stage") == "stage3"]
    s3_failed = [r for r in s3_records if r.get("first_pass_stage") is None]

    print(f"\nStage 3 instances: {len(s3_records)}")
    print(f"  Passed: {len(s3_passed)}")
    print(f"  Failed: {len(s3_failed)}")

    # Extract multi-turn samples
    samples = []
    skipped_no_trajectory = 0
    skipped_not_stage3 = 0
    skipped_no_thought = 0
    extracted_truncated = 0
    extracted_full = 0

    for record in s3_records:
        result = extract_multiturn(record, include_failed=args.include_failed)
        if result is None:
            # Check why it was skipped
            if not record.get("stage3_trajectory"):
                skipped_no_trajectory += 1
            elif not args.include_failed and record.get("first_pass_stage") != "stage3":
                skipped_not_stage3 += 1
            else:
                skipped_no_thought += 1
            continue

        samples.append(result)
        if result.get("truncated"):
            extracted_truncated += 1
        else:
            extracted_full += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Extraction Results")
    print(f"{'=' * 50}")
    print(f"Total extracted: {len(samples)}")
    print(f"  Full trajectory: {extracted_full}")
    print(f"  Truncated: {extracted_truncated}")
    print(f"\nSkipped:")
    print(f"  No trajectory: {skipped_no_trajectory}")
    if not args.include_failed:
        print(f"  Not stage3-passed (use --include-failed to include): {skipped_not_stage3}")
    print(f"  No thought at all: {skipped_no_thought}")

    # Category distribution
    cat_dist = Counter(s.get("category", "unknown") for s in samples)
    print(f"\nBy category: {dict(cat_dist)}")

    # Message count distribution
    msg_counts = [len(s["messages"]) for s in samples]
    turn_counts = [s["assistant_turns"] for s in samples]
    if msg_counts:
        print(f"\nMessages per sample: min={min(msg_counts)} max={max(msg_counts)} avg={sum(msg_counts)/len(msg_counts):.1f}")
        print(f"Assistant turns per sample: min={min(turn_counts)} max={max(turn_counts)} avg={sum(turn_counts)/len(turn_counts):.1f}")

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nOutput written to {args.output}")
    print(f"Done.")


if __name__ == "__main__":
    main()
