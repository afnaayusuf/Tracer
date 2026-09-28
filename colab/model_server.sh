#!/usr/bin/env bash
# A model-only runtime (an H100 in Colab): vLLM serving one model, reachable from another runtime through a tunnel.
#   bash colab/model_server.sh [model]      # prints  MODEL_URL: https://….trycloudflare.com/v1
# On the engine runtime:  os.environ["MODEL_URL"] = "<that url>"  before build_session_NN.sh  -> no local model there.
set -uo pipefail
MODEL="${1:-${AGENT_MODEL:-Qwen/Qwen3.5-35B-A3B}}"
PORT="${PORT:-8001}"
cd "${REPO_DIR:-/content/Tracer}" 2>/dev/null || { echo "clone the repo first (or set REPO_DIR)"; exit 1; }
bash colab/vllm_venv.sh start "$MODEL" "$PORT" || { echo "vLLM did not start; see /tmp/vllm*.log"; exit 1; }
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -sL -m 90 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
fi
pkill -f "[c]loudflared tunnel" 2>/dev/null; nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > /tmp/cloudflared_model.log 2>&1 &
for i in $(seq 1 30); do URL="$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/cloudflared_model.log | tail -1)"; [ -n "$URL" ] && break; sleep 1; done
[ -n "${URL:-}" ] || { echo "tunnel not ready; tail:"; tail -5 /tmp/cloudflared_model.log; exit 1; }
echo "MODEL_URL: $URL/v1     (model: $MODEL; this runtime must stay open)"
echo "test:  curl -s $URL/v1/models"
