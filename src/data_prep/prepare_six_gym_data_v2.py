#!/usr/bin/env python3
"""
SIX-GYM 数据准备脚本 v2
======================

改进：
1. 下载 SIX-GYM 数据集 (birdsql/six-gym-pg-1.5)
2. 从本地 postgre_table_dumps/ 提取 schema
3. 适配字段格式到 Pipeline 输入格式
4. 输出为 JSONL 文件

关键改进：
  - 自动从 postgre_table_dumps 提取 CREATE TABLE DDL
  - 支持映射 db_id 到本地 schema 文件
  - 保留完整的数据库架构信息
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from datasets import load_dataset
except ImportError:
    print("❌ 需要安装 datasets 库：")
    print("   pip install datasets")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 1: 提取 Schema
# ─────────────────────────────────────────────────────────────────────────────

def extract_schema_from_sql_files(db_dir: Path) -> Optional[str]:
    """
    从数据库目录的 .sql 文件提取 CREATE TABLE 定义。

    Args:
        db_dir: 数据库目录路径（如 postgre_table_dumps/financial_template）

    Returns:
        合并后的 schema 字符串，包含所有的 CREATE TABLE 定义
    """
    if not db_dir.exists():
        return None

    schema_parts = []
    sql_files = sorted(db_dir.glob("*.sql"))

    for sql_file in sql_files:
        with open(sql_file) as f:
            content = f.read()

        # 提取 CREATE TABLE 部分
        # 模式：从 CREATE TABLE 开始到下一个 CREATE 或文件结尾
        create_table_pattern = r'(CREATE TABLE.*?);'
        matches = re.findall(create_table_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            # 清理：移除注释、多余空白
            lines = match.split('\n')
            cleaned = []
            in_comment = False
            for line in lines:
                # 移除行注释
                if '--' in line:
                    line = line[:line.index('--')]
                # 移除多余空白
                line = line.strip()
                if line:
                    cleaned.append(line)

            if cleaned:
                schema_parts.append(' '.join(cleaned))

    if schema_parts:
        return '\n\n'.join(schema_parts) + ';'
    return None


# 项目根目录（src/data_prep/ 的上两级）
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_schema_cache(
    db_dump_dir: Path = _PROJECT_ROOT / "postgre_table_dumps"
) -> Dict[str, str]:
    """
    加载所有本地数据库的 schema 缓存。

    Args:
        db_dump_dir: 数据库导入目录

    Returns:
        {db_id: schema_text} 的字典
    """
    cache = {}

    if not db_dump_dir.exists():
        print(f"⚠️  数据库目录不存在: {db_dump_dir}")
        print("   将使用空 schema（数据库必须在线运行）")
        return cache

    # 遍历所有 _template 目录
    template_dirs = list(db_dump_dir.glob("*_template"))
    print(f"📦 正在加载 {len(template_dirs)} 个数据库的 schema...")

    for template_dir in template_dirs:
        db_id = template_dir.name.replace("_template", "")
        schema = extract_schema_from_sql_files(template_dir)
        if schema:
            cache[db_id] = schema
            print(f"   ✓ {db_id}: {len(schema)} chars")

    print(f"✅ 加载完成: {len(cache)} 个 schema")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 2: 下载数据
# ─────────────────────────────────────────────────────────────────────────────

def download_six_gym():
    """从 HuggingFace 下载 SIX-GYM 数据集。"""
    print("📥 正在从 HuggingFace 下载 SIX-GYM 数据集...")
    print("   数据集: birdsql/six-gym-pg-1.5")

    try:
        ds = load_dataset('birdsql/six-gym-pg-1.5', split='train')
        print(f"✅ 下载完成: {len(ds)} 条数据")
        return ds
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 3: 适配单条记录
# ─────────────────────────────────────────────────────────────────────────────

def adapt_record(
    record: Dict[str, Any],
    idx: int,
    schema_cache: Dict[str, str]
) -> Dict[str, Any]:
    """
    将 SIX-GYM 记录适配为 Pipeline 输入格式。

    SIX-GYM 格式 → Pipeline 格式
    """
    db_id = record.get("db_id", f"six_gym_{idx}")

    # 确保字段是列表
    issue_sql = record.get("issue_sql", [])
    if isinstance(issue_sql, str):
        issue_sql = [issue_sql] if issue_sql.strip() else []

    sol_sql = record.get("sol_sql", [])
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql] if sol_sql.strip() else []

    test_cases = record.get("test_cases", [])
    if isinstance(test_cases, str):
        test_cases = [test_cases] if test_cases.strip() else []

    preprocess_sql = record.get("preprocess_sql", [])
    if isinstance(preprocess_sql, str):
        preprocess_sql = [preprocess_sql] if preprocess_sql.strip() else []

    clean_up_sql = record.get("clean_up_sql", [])
    if isinstance(clean_up_sql, str):
        clean_up_sql = [clean_up_sql] if clean_up_sql.strip() else []

    # 获取 schema（优先使用本地缓存，否则为空）
    schema = schema_cache.get(db_id, "")

    return {
        "instance_id": record.get("instance_id", f"six_gym_{idx}"),
        "db_id": db_id,
        "db_name": f"{db_id}_template",
        "query": record.get("query", ""),
        "issue_sql": issue_sql,
        "sol_sql": sol_sql,
        "test_cases": test_cases,
        "preprocess_schema": schema,
        "preprocess_sql": preprocess_sql,
        "clean_up_sql": clean_up_sql,
        "category": record.get("category", "Query"),
        "dialect": record.get("dialect", "PostgreSQL"),
        "version": record.get("version", "14.12"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 4: 批量适配并保存
# ─────────────────────────────────────────────────────────────────────────────

def adapt_and_save(
    dataset,
    output_path: Path,
    schema_cache: Dict[str, str]
):
    """适配整个数据集并保存为 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 正在适配 {len(dataset)} 条记录...")

    with open(output_path, "w") as f:
        for idx, record in enumerate(dataset):
            adapted = adapt_record(record, idx, schema_cache)
            f.write(json.dumps(adapted, ensure_ascii=False) + "\n")

            if (idx + 1) % 500 == 0:
                print(f"   已处理: {idx + 1}/{len(dataset)}")

    print(f"✅ 适配完成，保存到: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 5: 验证输出
# ─────────────────────────────────────────────────────────────────────────────

def validate_output(output_path: Path):
    """验证输出文件的格式和完整性。"""
    print(f"\n🔍 验证输出文件: {output_path}")

    required_fields = [
        "instance_id", "db_id", "db_name", "query", "issue_sql",
        "sol_sql", "test_cases", "preprocess_schema"
    ]

    with open(output_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"   总记录数: {len(records)}")

    # 统计字段完整性
    missing_count = 0
    schema_missing = 0
    for record in records:
        for field in required_fields:
            if field not in record:
                missing_count += 1
        if not record.get("preprocess_schema", "").strip():
            schema_missing += 1

    if missing_count == 0:
        print(f"   ✅ 所有字段都完整")
    else:
        print(f"   ⚠️  {missing_count} 个字段缺失")

    print(f"   ⚠️  {schema_missing}/{len(records)} 条记录缺少 schema")

    # 统计字段大小
    sample = records[0] if records else {}
    print(f"\n📊 数据统计:")
    print(f"   平均 query 长度: {sum(len(r.get('query', '')) for r in records) // len(records) if records else 0}")
    print(f"   包含 issue_sql 的: {sum(1 for r in records if r.get('issue_sql'))}/{len(records)}")
    print(f"   包含 sol_sql 的: {sum(1 for r in records if r.get('sol_sql'))}/{len(records)}")
    print(f"   包含 test_cases 的: {sum(1 for r in records if r.get('test_cases'))}/{len(records)}")

    # 显示样本
    print(f"\n📋 样本记录 (第1条):")
    if records:
        r = records[0]
        print(f"   instance_id: {r['instance_id']}")
        print(f"   db_id: {r['db_id']}")
        print(f"   query: {r['query'][:100]}...")
        print(f"   issue_sql: {len(r['issue_sql'])} 条")
        print(f"   sol_sql: {len(r['sol_sql'])} 条")
        print(f"   schema: {len(r['preprocess_schema'])} chars")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """主流程"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SIX-GYM 数据准备脚本 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python prepare_six_gym_data_v2.py
  python prepare_six_gym_data_v2.py --limit 100
  python prepare_six_gym_data_v2.py --output data/six-gym-custom.jsonl --validate
        """
    )
    parser.add_argument(
        "--output", "-o",
        default="data/six-gym-adapted.jsonl",
        help="输出文件路径 (默认: data/six-gym-adapted.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条记录（用于测试）"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="验证输出文件"
    )

    args = parser.parse_args()
    output_path = _PROJECT_ROOT / args.output

    print("=" * 70)
    print("SIX-GYM 数据准备工具 v2")
    print("=" * 70)

    # 步骤 1: 加载 schema 缓存
    db_dump_dir = _PROJECT_ROOT / "postgre_table_dumps"
    schema_cache = load_schema_cache(db_dump_dir)

    # 步骤 2: 下载数据
    dataset = download_six_gym()

    # 如果指定了 --limit，只取前 N 条
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"⚠️  限制为 {len(dataset)} 条")

    # 步骤 3: 适配并保存
    adapt_and_save(dataset, output_path, schema_cache)

    # 步骤 4: 验证（可选）
    if args.validate:
        validate_output(output_path)

    print("\n" + "=" * 70)
    print(f"✅ 数据准备完成！")
    print(f"   输出文件: {output_path}")
    print(f"   缓存的 schema: {len(schema_cache)} 个数据库")
    print(f"\n   下一步运行 Pipeline:")
    print(f"   python run_pipeline.py --input {args.output} --limit 50")
    print("=" * 70)


if __name__ == "__main__":
    main()
