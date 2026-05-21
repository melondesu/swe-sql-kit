#!/bin/bash
# ============================================================
# 快速启动 BIRD-CRITIC PostgreSQL Docker 环境
# 使用前请确保：
#   1. Docker 已安装并运行
#   2. 已下载 postgre_table_dumps.zip 并解压
#   3. 修改下方 DUMPS_DIR 为实际路径
# ============================================================
set -e

# ── 配置（请修改这里）────────────────────────────────────────
DUMPS_DIR="${DUMPS_DIR:-/Users/yidele/Desktop/swe-sql/swe-sql-kit/swe-sql-kit/postgre_table_dumps}"
CONTAINER_NAME="bird_critic_pg"
PG_PASSWORD="123123"
PG_PORT=5432
# ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_SCRIPT="$SCRIPT_DIR/init-databases_postgresql.sh"

# 检查 dumps 目录
if [ ! -d "$DUMPS_DIR" ]; then
    echo "❌ DUMPS_DIR 不存在: $DUMPS_DIR"
    echo ""
    echo "请先下载并解压 BIRD-CRITIC 官方数据库 dump："
    echo "  下载地址: https://huggingface.co/datasets/birdsql/bird-critic-1.0"
    echo "  然后设置: export DUMPS_DIR=/path/to/postgre_table_dumps"
    echo "  或者直接修改本脚本中的 DUMPS_DIR 变量"
    exit 1
fi

# 停止并删除旧容器（如果存在）
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "⚠️  发现已有容器 ${CONTAINER_NAME}，停止并删除..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm   "$CONTAINER_NAME" 2>/dev/null || true
fi

echo "🚀 启动 PostgreSQL Docker 容器..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -e POSTGRES_USER=root \
    -e POSTGRES_PASSWORD="$PG_PASSWORD" \
    -e TZ=Asia/Hong_Kong \
    -v "$DUMPS_DIR:/docker-entrypoint-initdb.d/postgre_table_dumps" \
    -v "$INIT_SCRIPT:/docker-entrypoint-initdb.d/init-databases_postgresql.sh" \
    -p "$PG_PORT:5432" \
    --shm-size=256m \
    postgres:14

echo ""
echo "⏳ 等待数据库初始化（首次启动需要几分钟导入数据，请耐心等待）..."
echo "   实时日志: docker logs -f $CONTAINER_NAME"
echo ""

# 等待 PG 就绪
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker exec "$CONTAINER_NAME" psql -U root -c '\l' >/dev/null 2>&1; then
        # 检查是否已经有业务数据库（不只是 postgres）
        DB_COUNT=$(docker exec "$CONTAINER_NAME" psql -U root -t -c "SELECT count(*) FROM pg_database WHERE datname LIKE '%_template';" 2>/dev/null | tr -d ' ')
        if [ "${DB_COUNT:-0}" -gt 5 ]; then
            echo "✅ PostgreSQL 就绪！已导入 ${DB_COUNT} 个 template 数据库"
            break
        fi
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "   已等待 ${WAITED}s..."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠️  等待超时，请手动检查: docker logs $CONTAINER_NAME"
    exit 1
fi

echo ""
echo "🔍 验证数据库列表："
docker exec "$CONTAINER_NAME" psql -U root -c "\l" | grep template | head -20

echo ""
echo "✅ 环境搭建完成！现在可以运行实验："
echo "   python run_pipeline.py --limit 10                           # 先跑 10 条验证"
echo "   python run_pipeline.py --output results/v2_results.jsonl    # 全量 530 条"
