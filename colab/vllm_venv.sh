#!/usr/bin/env bash
# vLLM in its own virtualenv so its torch never touches the main runtime (rfdetr, transformers, the
# slice). The agent talks to it over HTTP. Idempotent: reuses the venv if it exists.
#   bash colab/vllm_venv.sh start [model] [port]     # install if needed, serve in the background, wait
#   bash colab/vllm_venv.sh stop
set -euo pipefail
VENV="${VLLM_VENV:-/content/vllm-venv}"
MODEL="${2:-${AGENT_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${3:-8000}"
LOG="/tmp/vllm.log"
cmd="${1:-start}"
if [ "$cmd" = stop ]; then pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; echo "stopped"; exit 0; fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "creating $VENV and installing vllm (isolated torch; 3-6 min)"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip >/dev/null
  if "$VENV/bin/pip" install -q vllm > /tmp/vllm_install.log 2>&1; then echo "vllm installed"; else
    echo "vllm install failed: $(tail -3 /tmp/vllm_install.log | tr '\n' ' ')"; exit 1; fi
fi
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
  --max-model-len 16384 --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.6}" --dtype bfloat16 --max-num-seqs 4 > "$LOG" 2>&1 &
echo "waiting for http://127.0.0.1:$PORT (model download on first run)..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM up: $(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
    exit 0
  fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "server exited; log tail:"; tail -12 "$LOG"; exit 1; fi
  sleep 5
done
echo "server did not come up in 10 min; log tail:"; tail -12 "$LOG"; exit 1
