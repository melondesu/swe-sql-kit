#!/bin/bash
set -e

echo "=== [1/5] 安装 PostgreSQL ==="
apt-get update -qq
apt-get install -y postgresql postgresql-client
echo "安装完成"

echo "=== [2/5] 启动 PostgreSQL 服务 ==="
service postgresql start
sleep 3
pg_isready && echo "PostgreSQL 已就绪"

echo "=== [3/5] 创建用户和数据库 ==="
sudo -u postgres psql -c "CREATE USER root WITH SUPERUSER PASSWORD 'root';" 2>/dev/null || echo "user root already exists"
for db in codebase_community erolp thrombosis_prediction; do
    sudo -u postgres psql -c "CREATE DATABASE $db OWNER root;" 2>/dev/null || echo "db $db already exists"
done

echo "=== [4/5] 导入数据 ==="
DUMP_DIR=/root/swe-sql-kit/postgre_table_dumps
export PGPASSWORD=root

for db in codebase_community erolp thrombosis_prediction; do
    dir="${DUMP_DIR}/${db}_template"
    echo "--- 导入 $db ---"
    for sql_file in "$dir"/*.sql; do
        echo "  $(basename $sql_file)"
        psql -U root -d "$db" -f "$sql_file" > /dev/null 2>&1 && echo "    OK" || echo "    WARN (可能已存在)"
    done
done

echo "=== [5/5] 验证 ==="
for db in codebase_community erolp thrombosis_prediction; do
    cnt=$(psql -U root -d "$db" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | tr -d ' ')
    echo "$db: ${cnt} 张表"
done

echo ""
echo "=== PostgreSQL 安装并导入完成！==="
