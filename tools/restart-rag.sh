#!/bin/bash
# RAG Service Restart Script (Full Rebuild)
# Usage: ./tools/restart-rag.sh
#
# This service runs on the knowledge-base-network network
# Other projects should join this network to use the RAG service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT/server"

echo "🔄 Rebuilding RAG service..."
echo ""

# Stop and remove container
echo "🛑 Stopping and removing container..."
docker compose down --rmi local 2>/dev/null || true

# Remove old image (if exists)
echo "🗑️  Removing old image..."
docker rmi server-rag-server 2>/dev/null || true

# Rebuild image
echo "🔨 Rebuilding image..."
docker compose build --no-cache

# Start service
echo "🚀 Starting service..."
docker compose up -d

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 8

# Check health status
echo "🏥 Checking service status..."
if curl -s http://localhost:3003/health > /dev/null 2>&1; then
    # Health check
    STATUS=$(curl -s http://localhost:3003/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','error'))")
    echo "✅ Service status: $STATUS"
    
    # Get statistics
    curl -s http://localhost:3003/fstats | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"📚 Documents loaded: {d.get('documents_loaded', 0)}\")
print(f\"📊 Knowledge base size: {d.get('total_chars', 0):,} chars ({d.get('estimated_tokens', 0):,} tokens)\")
print(f\"📝 Total nodes: {d.get('total_nodes', 0)}\")
"
else
    echo "❌ Service failed to start, check logs:"
    docker logs server-rag-server-1 --tail 20
    exit 1
fi

echo ""
echo "🎉 RAG service has been fully rebuilt and started!"
echo ""
echo "📡 API Endpoints:"
echo "   - Demo page: http://localhost:3003/demo"
echo "   - Upload page: http://localhost:3003/upload"
echo "   - Health check: GET  http://localhost:3003/health"
echo "   - File stats: GET  http://localhost:3003/fstats"
echo "   - Query API: POST http://localhost:3003/query"
echo ""
echo "🌐 Docker network: knowledge-base-network"
echo ""
echo "📌 How to connect from other projects:"
echo "   Add to docker-compose.yml:"
echo "   networks:"
echo "     knowledge-base-network:"
echo "       external: true"


