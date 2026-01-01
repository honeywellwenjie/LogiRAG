#!/bin/bash
# RAG 服务重启脚本（完全重建）
# 用法: ./tools/restart-rag.sh
#
# 本服务运行在 knowledge-base-network 网络中
# 其他项目应该加入此网络来使用 RAG 服务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT/server"

echo "🔄 正在完全重建 RAG 服务..."
echo ""

# 停止并删除容器
echo "🛑 停止并删除容器..."
docker compose down --rmi local 2>/dev/null || true

# 删除旧镜像（如果存在）
echo "🗑️  删除旧镜像..."
docker rmi server-rag-server 2>/dev/null || true

# 重新构建镜像
echo "🔨 重新构建镜像..."
docker compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 8

# 检查健康状态
echo "🏥 检查服务状态..."
if curl -s http://localhost:3003/health > /dev/null 2>&1; then
    # 健康检查
    STATUS=$(curl -s http://localhost:3003/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','error'))")
    echo "✅ 服务状态: $STATUS"
    
    # 获取统计信息
    curl -s http://localhost:3003/fstats | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"📚 已加载文档: {d.get('documents_loaded', 0)} 个\")
print(f\"📊 知识库大小: {d.get('total_chars', 0):,} 字符 ({d.get('estimated_tokens', 0):,} tokens)\")
print(f\"📝 总节点数: {d.get('total_nodes', 0)} 个\")
"
else
    echo "❌ 服务启动失败，请检查日志："
    docker logs server-rag-server-1 --tail 20
    exit 1
fi

echo ""
echo "🎉 RAG 服务已完全重建并启动！"
echo ""
echo "📡 API 接口："
echo "   - 演示页面: http://localhost:3003/demo"
echo "   - 上传页面: http://localhost:3003/upload"
echo "   - 健康检查: GET  http://localhost:3003/health"
echo "   - 文件统计: GET  http://localhost:3003/fstats"
echo "   - 查询接口: POST http://localhost:3003/query"
echo ""
echo "🌐 Docker 网络: knowledge-base-network"
echo ""
echo "📌 其他项目连接方式："
echo "   在 docker-compose.yml 中添加："
echo "   networks:"
echo "     knowledge-base-network:"
echo "       external: true"


