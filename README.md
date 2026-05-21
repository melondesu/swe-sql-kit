# SWE-SQL+ 实验框架

基于 BIRD-CRITIC-PG 基准，实现 4-Stage Pipeline 超越论文 Baseline。

---

## 目录结构

```
swe-sql-kit/
├── config.json              # ← 改这里（API Key、PG 配置）
├── requirements.txt         # Python 依赖
├── run_pipeline.py          # 主入口：Pipeline 全四阶段
├── run_finetune.sh          # 主入口：SFT 数据生成 + LoRA 微调一键脚本
├── src/
│   ├── pipeline.py              # PipelineRunner：Stage 0-3 编排 + 短路逻辑
│   ├── sqlagent.py              # SQL-ACT Agent：[EXPLORE]/[FIX] ReAct 循环
│   ├── explore_executor.py      # SAVEPOINT 回滚探测（任意 SQL 安全执行）
│   ├── prompts.py               # 全部 Prompt 模板（CoT/N-path/SQL-ACT）
│   ├── postgresql_utils.py      # PG 连接池 + 数据库重置
│   ├── postgresql_test_utils.py # 官方评测函数
│   ├── logger.py                # 日志工具
│   ├── sft/                     # SFT 数据构造脚本
│   │   ├── build_sft_data.py    # A 系列：f-Plan 倒推（需调 LLM）
│   │   ├── extract_sft_B.py     # B 系列：thinking 直接蒸馏
│   │   └── extract_sft_multi.py # 多轮 SFT 数据提取
│   ├── eval/                    # 评测脚本
│   │   ├── eval_sft.py          # 单轮评测（4 个模型通用）
│   │   └── eval_sft_agent.py    # Agent 评测（仅 Multi 模型）
│   └── data_prep/               # 数据预处理脚本
│       └── prepare_six_gym_data_v2.py  # SIX-GYM 数据集适配
├── data/
│   ├── postgresql_530.jsonl             # 完整 530 条（BIRD-CRITIC-PG 全集）
│   ├── train_530.jsonl                  # 训练集（451 条，12 个数据库）
│   ├── val_530.jsonl                    # 验证集（79 条，3 个数据库）
│   ├── flash_dataset.parquet            # 200 条子集（快速验证用）
│   ├── bird-critic-1.0-flash_w_sol.jsonl
│   └── flash_schema_full.jsonl
├── results/                     # 实验输出（自动创建）
│   ├── pipeline/                # Pipeline 运行产出
│   │   ├── v5_0a_nothinking.jsonl       # nothinking 模式（训练集 451 条）
│   │   ├── v5_0a_nothinking_summary.json
│   │   ├── v5_0b_thinking.jsonl         # thinking 模式（训练集 451 条）
│   │   ├── v5_0b_thinking_summary.json
│   │   └── v4_full.jsonl                # 历史版本存档
│   ├── sft/                     # SFT 训练数据
│   │   ├── sft_A_single.jsonl   # f-Plan 倒推，单轮
│   │   ├── sft_A_multi.jsonl    # non-thinking 多轮轨迹
│   │   ├── sft_B_single.jsonl   # thinking 单轮
│   │   ├── sft_B_multi.jsonl    # thinking 多轮轨迹
│   │   └── v4_sft_data.jsonl    # 合并数据（v4 历史存档）
│   └── eval/                    # 模型评测结果
│       ├── A_single.jsonl
│       ├── A_multi_single.jsonl
│       ├── A_multi_agent.jsonl
│       ├── B_single.jsonl
│       ├── B_multi_single.jsonl
│       ├── B_multi_agent.jsonl
│       └── baseline_qwen.jsonl
├── finetuning/
│   ├── LLaMA-Factory/           # 微调框架（子模块）
│   ├── train_lora_sft_A_single.yaml
│   ├── train_lora_sft_B_single.yaml
│   ├── train_lora_sft_A_multi.yaml
│   ├── train_lora_sft_B_multi.yaml
│   └── verify_setup.py          # 环境验证脚本
└── scripts/
    ├── setup_docker.sh              # 一键启动 PG Docker
    ├── setup_pg.sh                  # PG 环境初始化
    └── init-databases_postgresql.sh # PG 数据库导入脚本
```

---

## 快速开始

### Step 1：配置

编辑 `config.json`：

```json
{
  "LLM_API_KEY":  "你的 API Key",
  "LLM_BASE_URL": "https://api.siliconflow.cn/v1",
  "LLM_MODEL":    "Qwen/Qwen3-235B-A22B",
  "PG_HOST":      "localhost",
  "PG_PORT":      5432,
  "PG_USER":      "root",
  "PG_PASSWORD":  "123123"
}
```

也可以用环境变量覆盖（环境变量优先级更高）：
```bash
export LLM_API_KEY="your-key"
export PG_HOST="localhost"
```

### Step 2：安装依赖

```bash
pip install -r requirements.txt
```

### Step 3：下载数据库 Dump

`postgre_table_dumps/` 目录因文件体积过大（单文件最大 267MB）未纳入 Git 仓库，需从官方渠道单独下载：

**官方地址：** https://bird-critic.github.io/

在页面找到 **PostgreSQL Database** 下载入口，下载后解压，将 `postgre_table_dumps/` 目录放到本仓库根目录下：

```
swe-sql-kit/
└── postgre_table_dumps/
    ├── card_games_template/
    ├── financial_template/
    └── ...（共 15 个数据库）
```

### Step 4：启动 PostgreSQL 数据库

```bash
# 修改 DUMPS_DIR 后运行
export DUMPS_DIR=/path/to/postgre_table_dumps
bash scripts/setup_docker.sh
```

首次启动需要 5~10 分钟导入数据，等脚本打印"环境搭建完成"再继续。

手动验证：
```bash
docker exec bird_critic_pg psql -U root -c "\l" | grep template
# 应能看到 financial_template, card_games_template 等 15 个数据库
```

### Step 5：验证运行（10条）

```bash
python run_pipeline.py --limit 10
```

### Step 6：全量运行（训练集 451 条）

```bash
# nothinking 模式（路径 A）
python run_pipeline.py --input data/train_530.jsonl \
  --output results/pipeline/v5_0a_nothinking.jsonl --thinking off

# thinking 模式（路径 B）
python run_pipeline.py --input data/train_530.jsonl \
  --output results/pipeline/v5_0b_thinking.jsonl --thinking on

# 限制条数
python run_pipeline.py --limit 100

# 从断点恢复
python run_pipeline.py --skip 200
```

---

## Pipeline 架构

```
issue_sql（有 bug 的 SQL）
    │
    ▼
Stage 0: Baseline ────────── 原样提交 issue_sql
    │ fail
    ▼
Stage 1: CoT ─────────────── 4 步推理链（分析→定位→计划→修复）
    │ fail
    ▼
Stage 2: N-path ──────────── 三路并行（diagnostic / rewrite / minimal_fix）+ 选优
    │ fail
    ▼
Stage 3: SQL-ACT Agent ───── [EXPLORE] 探测数据库（SAVEPOINT 回滚）
                              [FIX] 提交修复 → test_case 反馈 → 迭代（最多 5 轮）
    │
    ▼
输出 results/*.jsonl（完整轨迹）+ SR%
```

各 Stage 之间是**短路逻辑**：一旦某个 Stage 通过 test_case，立刻记录结果并跳过后续 Stage。

每个 instance 在独立的临时数据库副本上运行，跑完自动清理，防止 Management 类任务的跨实例污染。

---

## SFT 数据构造

先跑完 Pipeline，再构造 SFT 数据：

```bash
# A 系列：f-Plan 倒推（需要调 LLM）
python src/sft/build_sft_data.py \
  --input results/pipeline/v5_0a_nothinking.jsonl \
  --output results/sft/sft_A_single.jsonl

# B 系列：直接从 thinking Pipeline 提取（不需要调 LLM）
python src/sft/extract_sft_B.py \
  --input results/pipeline/v5_0b_thinking.jsonl \
  --output_single results/sft/sft_B_single.jsonl \
  --output_multi results/sft/sft_B_multi.jsonl
```

4 个 SFT 数据文件说明：

| 文件 | 来源 | 推理链 | 需要调 LLM |
|------|------|--------|-----------|
| `sft_A_single.jsonl` | nothinking Pipeline + f-Plan 倒推 | 事后编造 | ✅ 需要 |
| `sft_A_multi.jsonl` | nothinking Pipeline trajectory | Agent thought | ❌ 不需要 |
| `sft_B_single.jsonl` | thinking Pipeline stage1_response | 真实推理链 | ❌ 不需要 |
| `sft_B_multi.jsonl` | thinking Pipeline trajectory | 真实推理链 | ❌ 不需要 |

---

## 数据集划分

530 条数据按**数据库级别**划分，确保验证集数据库的 schema 在训练时从未出现：

| 集合 | 条数 | 数据库 |
|------|------|--------|
| 训练集 (`train_530.jsonl`) | 451 | 12 个数据库 |
| 验证集 (`val_530.jsonl`) | 79 | erolp / thrombosis_prediction / codebase_community |

详细划分理由见 [`DATASET_SPLIT_RATIONALE.md`](DATASET_SPLIT_RATIONALE.md)。

---

## 实验结果

### Pipeline 全量结果（DeepSeek-V3，530 条）

```
总通过率：326/530 = 61.5%
  stage0（直接提交原始 SQL）：6 题  (1.1%)
  stage1（CoT 单轮推理）：   162 题 (30.6%)
  stage2（N-path 三路并行）：  72 题 (13.6%)
  stage3（SQL-ACT ReAct）：   86 题 (16.2%)
  未通过（unsolved）：        204 题 (38.5%)
```

### SFT 微调结果（Qwen2.5-Coder-7B，验证集 79 条）

| 模型 | 数据类型 | 单轮 SR | Agent SR |
|------|---------|---------|---------|
| Qwen2.5-Coder-7B（基座，无微调） | — | 12.7% | — |
| A_single | non-thinking 单轮 | 7.6% | — |
| B_single | thinking 单轮 | 12.7% | — |
| A_multi | non-thinking 多轮 | 10.1% | ~20% |
| B_multi | thinking 多轮 | 13.9% | 20.3% |

**关键结论**：
1. thinking 数据（B 系列）≥ non-thinking 数据（A 系列）
2. Agent 模式显著提升 SR：B_multi Agent(20.3%) vs B_multi 单轮(13.9%)
3. 多轮 trajectory 数据对 Agent 模式有帮助

---

## 常见问题

**Q: PG 连接失败？**
```bash
docker ps  # 确认容器在跑
docker logs bird_critic_pg | tail -20  # 查看初始化状态
```

**Q: LLM API 调用失败？**
检查 `config.json` 的 API Key 和 Base URL，或设置环境变量 `LLM_API_KEY`

**Q: 如何只跑特定区间？**
```bash
python run_pipeline.py --skip 100 --limit 50
```

**Q: 结果文件中断了怎么继续？**
检查已跑到哪条，用 `--skip` 从断点继续（结果会 append 到同一文件）

**Q: 服务器重启后 PostgreSQL 连不上？**
```bash
service postgresql start
```
