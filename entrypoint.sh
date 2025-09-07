#!/usr/bin/env bash
set -euo pipefail

# Optional but nice: make Streamlit reload reliably in containers
export STREAMLIT_BROWSER_GATHERUSAGESTATS=false

uvicorn "image2text.api:app" --host 0.0.0.0 --port 8001 --reload &
API_PID=$!

# 🔥 Hot reload inside Docker
streamlit run image2text/web_app.py \
  --server.address 0.0.0.0 \
  --server.port 8502 \
  --server.fileWatcherType poll \
  --server.runOnSave true &
UI_PID=$!

term_handler() {
  kill -TERM "$API_PID" "$UI_PID" 2>/dev/null || true
  wait "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap term_handler SIGTERM SIGINT

wait -n "$API_PID" "$UI_PID" || true
kill -TERM "$API_PID" "$UI_PID" 2>/dev/null || true
wait || true
