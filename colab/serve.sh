#!/usr/bin/env bash
# Start the backend (API + frontend) in the background and open a public HTTPS URL to it.
#   bash colab/serve.sh start   # uvicorn on :8000, cloudflared quick tunnel, prints the URL
#   bash colab/serve.sh url     # print the last tunnel URL
#   bash colab/serve.sh stop
# Env: VI_DB VI_BACKEND (transformers|fake|openai) VI_MODEL VI_TZ PORT (8000)
set -uo pipefail
PORT="${PORT:-8000}"
LOG=/tmp/vi_api.log; TLOG=/tmp/cloudflared.log
cmd="${1:-start}"
if [ "$cmd" = stop ]; then pkill -f "[r]un_ingest.py" 2>/dev/null; pkill -f "[u]vicorn vi.api.server" 2>/dev/null; pkill -f "[c]loudflared tunnel" 2>/dev/null; sleep 1; echo "stopped (ingest, api, tunnel)"; exit 0; fi
if [ "$cmd" = url ]; then grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$TLOG" | tail -1; exit 0; fi
export VI_DB="${VI_DB:-${DB_URL:-sqlite+pysqlite:///data/vi.db}}" VI_BACKEND="${VI_BACKEND:-transformers}" VI_MODEL="${VI_MODEL:-Qwen/Qwen3.5-4B}" VI_TZ="${VI_TZ:-UTC}" VI_OPENAI_BASE="${VI_OPENAI_BASE:-http://127.0.0.1:8001/v1}" PORT="$PORT"
pkill -f "[u]vicorn vi.api.server" 2>/dev/null; sleep 1
nohup python -m uvicorn vi.api.server:app --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && echo "api up on :$PORT ($VI_BACKEND $VI_MODEL, db $VI_DB)" || { echo "api failed to start; log:"; tail -20 "$LOG"; exit 1; }
# wait for the model (first run downloads the weights: a 27B is ~54 GB); questions and deltas start only when it is loaded
echo "waiting for the model to load (weights download on first use)…"
for i in $(seq 1 360); do
  st="$(curl -s "http://127.0.0.1:$PORT/health" | python3 -c 'import sys,json; h=json.load(sys.stdin); print("ok" if h.get("model_loaded") else ("err:"+str(h.get("load_error"))) if h.get("load_error") else "wait")' 2>/dev/null)"
  case "$st" in ok) echo "model loaded: $VI_MODEL ($(curl -s "http://127.0.0.1:$PORT/health" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("load_s"))')s)"; break;;
    err:*) echo "MODEL LOAD FAILED: ${st#err:}"; tail -5 "$LOG"; break;; esac
  [ $((i % 12)) = 0 ] && echo "  … still loading ($((i * 5)) s); $(tail -1 "$LOG" | cut -c1-100)"
  sleep 5
done
# public HTTPS: cloudflared quick tunnel (no account); falls back to the Colab proxy hint
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -sL -m 90 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared 2>/dev/null && chmod +x /usr/local/bin/cloudflared || true
fi
if command -v cloudflared >/dev/null 2>&1; then
  pkill -f "[c]loudflared tunnel" 2>/dev/null
  nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$TLOG" 2>&1 &
  for i in $(seq 1 30); do
    URL="$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$TLOG" | tail -1)"
    [ -n "$URL" ] && break; sleep 1
  done
  if [ -n "${URL:-}" ]; then echo "PUBLIC URL: $URL   (open it on your laptop; the page and the API share this origin)"; else echo "tunnel not ready; tail:"; tail -5 "$TLOG"; fi
else
  echo "cloudflared unavailable; in a Python cell run:  from google.colab.output import eval_js; print(eval_js('google.colab.kernel.proxyPort($PORT)'))"
fi
