#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load .env so OPENROUTER_API_KEY is available
if [ -f "$ROOT/.env" ]; then
  export $(grep -v '^#' "$ROOT/.env" | xargs)
fi

cleanup() {
  echo ""
  echo "Terminating..."
  kill "$RUST_PID" "$PYTHON_PID" "$VITE_PID" 2>/dev/null
  wait "$RUST_PID" "$PYTHON_PID" "$VITE_PID" 2>/dev/null
  echo "Execution Halted."
}
trap cleanup INT TERM

# 1. Rust MLP server
echo "▶ starting Rust MLP server..."
cd "$ROOT/src-tauri"
cargo build --release -q
./target/release/mlp-server &
RUST_PID=$! # PID of the last background process

# Wait for Rust to be ready
echo "  waiting for :9000..."
until curl -sf http://127.0.0.1:9000/healthz > /dev/null 2>&1; do sleep 0.3; done
echo "  ✓ Rust MLP server ready"

# 2. Python agent
echo "▶ starting Python agent..."
cd "$ROOT/backend"
source .venv/bin/activate
uvicorn app.main:app --port 8000 --log-level warning &
PYTHON_PID=$!

until curl -sf http://127.0.0.1:8000/healthz > /dev/null 2>&1; do sleep 0.3; done
echo "  ✓ Python agent ready"

# 3. Vite frontend
echo "▶ starting Vite frontend..."
cd "$ROOT/frontend"
npm run dev &
VITE_PID=$!

echo ""
echo "  all services running:"
echo "    frontend  → http://localhost:5173"
echo "    agent     → http://localhost:8000"
echo "    mlp       → http://127.0.0.1:9000"
echo ""
echo "  press Ctrl+C to stop all"

wait "$VITE_PID"
