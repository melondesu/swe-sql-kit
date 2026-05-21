#!/usr/bin/env bash
# =============================================================================
# run_finetune.sh — 一键完成 SFT 数据生成 + LoRA 微调
# =============================================================================
# 用法：
#   bash run_finetune.sh [选项]
#
# 选项：
#   --skip-data-build   跳过 SFT 数据生成（4 个 jsonl 已存在时使用）
#   --only-data-build   只生成 SFT 数据，不启动训练
#   --train A_single    只训练指定的一个配置（A_single / B_single / A_multi / B_multi）
#   --help              显示帮助
#
# 示例：
#   bash run_finetune.sh                        # 完整流程：生成数据 + 训练全部 4 个
#   bash run_finetune.sh --skip-data-build      # 数据已有，直接训练
#   bash run_finetune.sh --only-data-build      # 只生成数据，不训练
#   bash run_finetune.sh --train B_single       # 只训练 B_single
# =============================================================================

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
FINETUNE_DIR="$SCRIPT_DIR/finetuning"

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SKIP_DATA_BUILD=false
ONLY_DATA_BUILD=false
TRAIN_TARGET="all"   # all | A_single | B_single | A_multi | B_multi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data-build)  SKIP_DATA_BUILD=true; shift ;;
        --only-data-build)  ONLY_DATA_BUILD=true; shift ;;
        --train)            TRAIN_TARGET="$2"; shift 2 ;;
        --help)
            sed -n '3,20p' "$0"
            exit 0
            ;;
        *) log_error "未知参数: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  SWE-SQL LoRA 微调一键脚本"
echo "============================================================"
echo ""

# ── Step 0: 环境检查 ──────────────────────────────────────────────────────────
log_info "Step 0: 检查环境..."

# 检查 Python
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    log_error "未找到 python / python3，请先安装 Python 3.10+"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)
log_ok "Python: $($PYTHON --version)"

# 检查 llamafactory-cli
if ! command -v llamafactory-cli &>/dev/null; then
    log_warn "llamafactory-cli 未在 PATH 中，尝试从 LLaMA-Factory 目录安装..."
    if [[ -d "$FINETUNE_DIR/LLaMA-Factory" ]]; then
        pip install -e "$FINETUNE_DIR/LLaMA-Factory" -q
        if ! command -v llamafactory-cli &>/dev/null; then
            log_error "安装失败，请手动执行: pip install -e finetuning/LLaMA-Factory"
            exit 1
        fi
        log_ok "llamafactory-cli 安装成功"
    else
        log_error "finetuning/LLaMA-Factory 目录不存在"
        exit 1
    fi
else
    log_ok "llamafactory-cli: $(llamafactory-cli version 2>/dev/null || echo '已安装')"
fi

# 检查基座模型
MODEL_PATH="$SCRIPT_DIR/models/qwen/Qwen2___5-Coder-7B-Instruct"
if [[ ! -d "$MODEL_PATH" ]]; then
    log_error "基座模型不存在: $MODEL_PATH"
    log_error "请先下载模型，参考 FINETUNE_GUIDE.md"
    exit 1
fi
log_ok "基座模型: $MODEL_PATH"

# 检查 pipeline 结果文件
PIPELINE_A="$RESULTS_DIR/pipeline/v5_0a_nothinking.jsonl"
PIPELINE_B="$RESULTS_DIR/pipeline/v5_0b_thinking.jsonl"
if [[ ! -f "$PIPELINE_A" ]]; then
    log_error "Pipeline A 结果不存在: $PIPELINE_A"
    log_error "请先运行 pipeline: python run_pipeline.py --output results/pipeline/v5_0a_nothinking.jsonl"
    exit 1
fi
if [[ ! -f "$PIPELINE_B" ]]; then
    log_error "Pipeline B 结果不存在: $PIPELINE_B"
    log_error "请先运行 pipeline: python run_pipeline.py --output results/pipeline/v5_0b_thinking.jsonl"
    exit 1
fi
log_ok "Pipeline 结果文件: results/pipeline/v5_0a_nothinking.jsonl + v5_0b_thinking.jsonl"

echo ""

# ── Step 1: 生成 4 个 SFT 数据文件 ───────────────────────────────────────────
if [[ "$SKIP_DATA_BUILD" == "true" ]]; then
    log_warn "Step 1: 跳过 SFT 数据生成（--skip-data-build）"
    # 验证文件存在
    for f in sft_A_single sft_A_multi sft_B_single sft_B_multi; do
        if [[ ! -f "$RESULTS_DIR/sft/${f}.jsonl" ]]; then
            log_error "数据文件不存在: results/sft/${f}.jsonl，请先生成或去掉 --skip-data-build"
            exit 1
        fi
        cnt=$(wc -l < "$RESULTS_DIR/sft/${f}.jsonl")
        log_ok "  ${f}.jsonl: ${cnt} 条样本"
    done
else
    log_info "Step 1: 生成 4 个 SFT 数据文件..."
    log_warn "  注意：sft_A_single 需要调用 LLM API（deepseek-v3）生成 f-Plan，会消耗 token"
    echo ""

    cd "$SCRIPT_DIR"
    $PYTHON src/sft/build_sft_data.py --build_four_files

    echo ""
    # 验证输出
    all_ok=true
    for f in sft_A_single sft_A_multi sft_B_single sft_B_multi; do
        if [[ -f "$RESULTS_DIR/sft/${f}.jsonl" ]]; then
            cnt=$(wc -l < "$RESULTS_DIR/sft/${f}.jsonl")
            log_ok "  ${f}.jsonl: ${cnt} 条样本"
        else
            log_error "  ${f}.jsonl 未生成！"
            all_ok=false
        fi
    done
    if [[ "$all_ok" == "false" ]]; then
        log_error "部分 SFT 数据文件生成失败，请检查日志"
        exit 1
    fi
fi

echo ""

# ── 如果只生成数据，到此结束 ──────────────────────────────────────────────────
if [[ "$ONLY_DATA_BUILD" == "true" ]]; then
    log_ok "Step 1 完成，--only-data-build 模式，退出"
    echo ""
    echo "下一步：运行训练"
    echo "  bash run_finetune.sh --skip-data-build"
    exit 0
fi

# ── Step 2: 创建输出目录 ──────────────────────────────────────────────────────
log_info "Step 2: 创建训练输出目录..."
mkdir -p "$FINETUNE_DIR/output/A_single"
mkdir -p "$FINETUNE_DIR/output/A_multi"
mkdir -p "$FINETUNE_DIR/output/B_single"
mkdir -p "$FINETUNE_DIR/output/B_multi"
log_ok "输出目录已就绪: finetuning/output/{A_single,A_multi,B_single,B_multi}"

echo ""

# ── Step 3: 训练 ──────────────────────────────────────────────────────────────
log_info "Step 3: 开始 LoRA 微调训练..."
cd "$FINETUNE_DIR"

run_train() {
    local name="$1"
    local yaml="train_lora_sft_${name}.yaml"
    echo ""
    echo "------------------------------------------------------------"
    log_info "训练 ${name}  (配置: ${yaml})"
    echo "------------------------------------------------------------"
    if llamafactory-cli train "$yaml"; then
        log_ok "${name} 训练完成！权重保存在 output/${name}/"
    else
        log_error "${name} 训练失败，请检查日志"
        exit 1
    fi
}

case "$TRAIN_TARGET" in
    all)
        run_train "A_single"
        run_train "B_single"
        run_train "A_multi"
        run_train "B_multi"
        ;;
    A_single|B_single|A_multi|B_multi)
        run_train "$TRAIN_TARGET"
        ;;
    *)
        log_error "未知训练目标: $TRAIN_TARGET（可选: all / A_single / B_single / A_multi / B_multi）"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
log_ok "全部完成！"
echo ""
echo "  LoRA 权重位置:"
if [[ "$TRAIN_TARGET" == "all" ]]; then
    echo "    finetuning/output/A_single/"
    echo "    finetuning/output/B_single/"
    echo "    finetuning/output/A_multi/"
    echo "    finetuning/output/B_multi/"
else
    echo "    finetuning/output/${TRAIN_TARGET}/"
fi
echo ""
echo "  合并权重（示例）:"
echo "    llamafactory-cli export \\"
echo "      --model_name_or_path models/qwen/Qwen2___5-Coder-7B-Instruct \\"
echo "      --adapter_name_or_path finetuning/output/A_single/checkpoint-xxx \\"
echo "      --template qwen \\"
echo "      --finetuning_type lora \\"
echo "      --export_dir finetuning/output/A_single_merged"
echo "============================================================"
