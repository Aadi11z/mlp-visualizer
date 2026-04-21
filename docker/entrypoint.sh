#!/bin/sh
set -eu

# Start the Rust MLP server in the background on an internal port.
MLP_PORT="${MLP_PORT:-9000}" mlp-server &
RUST_PID=$!

# Make sure Rust exits when Python exits (fly runs PID 1; signals cascade).
trap 'kill -TERM "$RUST_PID" 2>/dev/null || true' INT TERM EXIT

# Small readiness wait so the first request doesn't race the Rust server.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:'+\"${MLP_PORT:-9000}\"+'/healthz', timeout=0.5); sys.exit(0)" 2>/dev/null; then
    break
  fi
  sleep 0.3
done

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
