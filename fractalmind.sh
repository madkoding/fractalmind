#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Fractal-Mind..."

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Start Docker services
echo "📦 Starting Docker services (Ollama, SurrealDB, SearXNG)..."
docker-compose up -d ollama surrealdb searxng

# Wait for Ollama
echo "⏳ Waiting for Ollama..."
sleep 5

for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama ready"
        break
    fi
    echo "   Waiting for Ollama... ($i/30)"
    sleep 2
done

# Pull required local models if not present
echo "📥 Ensuring required local models are available..."
if ! docker exec fractalmind-ollama ollama list | grep -q qwen3-embedding:0.6b; then
    echo "   Pulling qwen3-embedding:0.6b..."
    docker exec fractalmind-ollama ollama pull qwen3-embedding:0.6b
else
    echo "   ✅ qwen3-embedding:0.6b already present"
fi

if ! docker exec fractalmind-ollama ollama list | grep -q llama3.2:1b; then
    echo "   Pulling llama3.2:1b..."
    docker exec fractalmind-ollama ollama pull llama3.2:1b
else
    echo "   ✅ llama3.2:1b already present"
fi

# Check if services are healthy
for i in {1..10}; do
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        echo "✅ SurrealDB ready"
        break
    fi
    echo "   Waiting for SurrealDB... ($i/10)"
    sleep 1
done

# Start backend in background
echo "🔧 Starting backend..."
WEB_SEARCH_PROVIDER=searxng WEB_SEARCH_BASE_URL=http://localhost:18080 cargo run &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend..."
cd ui && npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Fractal-Mind is running!"
echo ""
echo "   Backend API:  http://localhost:9000"
echo "   Frontend:     http://localhost:9001"
echo ""
echo "   Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; docker-compose stop ollama surrealdb searxng; exit" SIGINT SIGTERM

wait
