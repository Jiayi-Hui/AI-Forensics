#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== AI Forensics Demo ==="

# Kill any existing ngrok and backend processes
pkill -f ngrok 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

# Start backend in background
echo "[1/2] Starting backend..."
cd "$ROOT/backend"
conda run -n py311 python main.py &
BACKEND_PID=$!

# Wait for backend to be ready
for i in $(seq 1 15); do
  if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "      Backend ready at http://localhost:8000"
    break
  fi
  sleep 1
done

# Start ngrok tunnel
echo "[2/2] Opening ngrok tunnel..."
echo "      (copy the https://xxxx.ngrok-free.app URL below)"
echo ""
ngrok http 8000 --pooling-enabled

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null
