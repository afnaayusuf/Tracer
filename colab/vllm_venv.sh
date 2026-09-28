#!/usr/bin/env bash
# vLLM in its own environment so its torch never touches the runtime's (Colab ships a CUDA-13 torch;
# vLLM wheels bring a CUDA-12.8 torch and torchaudio refuses to import next to it).
# Colab's python has no ensurepip, so the venv is built by, in order: uv -> venv --without-pip +
# pip bootstrap -> virtualenv. Each failure is printed and the next method is tried.
#   bash colab/vllm_venv.sh start [model] [port]   # install if needed, serve in background, wait
#   bash colab/vllm_venv.sh stop
#   bash colab/vllm_venv.sh status                  # up | down
#   bash colab/vllm_venv.sh venv                    # only build the venv (smoke test)
set -uo pipefail
VENV="${VLLM_VENV:-/content/vllm-venv}"
MODEL="${2:-${AGENT_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${3:-8000}"
LOG="/tmp/vllm.log"
cmd="${1:-start}"
if [ "$cmd" = stop ]; then pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; echo "stopped"; exit 0; fi
if [ "$cmd" = status ]; then curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && echo "up" || { echo "down"; exit 1; }; exit 0; fi
if [ "$cmd" = start ] && curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  served="$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
  if [ "$served" = "$MODEL" ]; then echo "   vLLM already up: $served (reusing; run 'stop' to restart)"; exit 0; fi
  echo "   vLLM up with $served, restarting for $MODEL"; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; sleep 3
fi
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }

make_venv() {
  [ -x "$VENV/bin/python" ] && return 0
  echo "-- venv via uv"
  if (command -v uv >/dev/null 2>&1 || pipi uv) && uv venv "$VENV" --python "$(command -v python3)" >/tmp/venv.log 2>&1 \
     && uv pip install --python "$VENV/bin/python" -q pip >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  echo "-- venv via python -m venv --without-pip + get-pip"
  if python3 -m venv --without-pip "$VENV" >/tmp/venv.log 2>&1 \
     && curl -sSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
     && "$VENV/bin/python" /tmp/get-pip.py -q >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  echo "-- venv via virtualenv"
  if pipi virtualenv && python3 -m virtualenv -q "$VENV" >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  return 1
}

driver_cuda() {  # e.g. 12.8 from nvidia-smi; the torch build must not be newer than this
  nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1
}

install_vllm() {
  drv="$(driver_cuda)"
  if [ -x "$VENV/bin/python" ]; then
    if "$VENV/bin/python" -c "import vllm" >/tmp/vllm_import.log 2>&1; then
      have="$("$VENV/bin/python" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)"
      if [ -n "$have" ] && [ -n "$drv" ] && [ "$have" != "$drv" ] && [ "${VLLM_KEEP_VENV:-0}" != "1" ]; then
        echo "-- venv torch is cu$have but the driver is CUDA $drv: rebuilding with the exact driver build"
        rm -rf "$VENV"; make_venv || return 1
      else
        return 0
      fi
    else
      echo "-- venv exists but 'import vllm' fails ($(grep -oE "ImportError: [^\n]*|ModuleNotFoundError: [^\n]*" /tmp/vllm_import.log | head -1 | cut -c1-100)): rebuilding"
      rm -rf "$VENV"; make_venv || return 1
    fi
  fi
  # vLLM's PyPI wheel links libcudart of a fixed CUDA major (13 for 0.30); the torch build must match
  # the driver's CUDA, not the runtime's torch (session 18 forced cu128 and broke vllm._C).
  drv_tag="$(echo "$drv" | tr -d .)"                       # 13.0 -> cu130: the exact build for this driver
  backend="${VLLM_TORCH_BACKEND:-${drv_tag:+cu$drv_tag}}"; backend="${backend:-auto}"
  echo "-- installing vllm into the venv (driver CUDA ${drv:-?}, torch backend $backend; 3-6 min)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q --torch-backend="$backend" vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv, torch-backend=$backend)"; return 0; fi
  echo "   uv with --torch-backend=$backend failed: $(tail -1 /tmp/vllm_install.log | cut -c1-120)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q --torch-backend=auto vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv, torch-backend=auto)"; return 0; fi
  if "$VENV/bin/python" -m pip install -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (pip)"; return 0; fi
  echo "   failed: $(tail -2 /tmp/vllm_install.log | tr '\n' ' ')"; return 1
}

root_cause() {  # the ENGINE's error (EngineCore lines) comes first in the log; the API server's stack only repeats it
  local n; n="$(grep -nE "EngineCore.*(ERROR|Error|Traceback)|ERROR|Traceback|Error:" "$LOG" | head -1 | cut -d: -f1)"
  if [ -n "$n" ]; then tail -n "+$n" "$LOG" | grep -v "INFO\|TracerWarning\|APIServer" | head -45 | cut -c1-220
  else tail -25 "$LOG" | cut -c1-220; fi
  echo "   (full log: $LOG)"
}

make_venv || { echo "could not create a virtualenv by any method"; exit 1; }
if [ "$cmd" = venv ]; then "$VENV/bin/python" -c "import sys; print('   venv python', sys.version.split()[0])"; exit 0; fi
install_vllm || exit 1
"$VENV/bin/python" -c "import vllm, torch; print('   vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" || exit 1
echo "   driver CUDA $(driver_cuda)"
root_cause_last() {  # the exception itself is the LAST error line, not the first traceback frame
  grep -E "Error|error|Exception|not supported|No module|CUDA" "$LOG" | grep -v "INFO\|TracerWarning\|Traceback\|File \"" | tail -6 | cut -c1-240
}
serve_once() {  # $1 = label, rest = extra args
  local label="$1"; shift
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; sleep 2
  # clean process environment: no PYTHONPATH, no user site, and only the driver's library dir on LD_LIBRARY_PATH
  # (Colab's LD_LIBRARY_PATH points at the main environment's CUDA libraries, which can shadow the venv's)
  nohup env -u PYTHONPATH PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/usr/lib64-nvidia}" ${VLLM_ENV:-} \
    "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
    --max-model-len "${VLLM_MAX_LEN:-8192}" --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" --dtype bfloat16 --max-num-seqs 4 \
    --enable-prefix-caching --limit-mm-per-prompt '{"image": 8, "video": 1}' "$@" > "$LOG" 2>&1 &
  echo "-- waiting for http://127.0.0.1:$PORT ($label; weights download on first run)"
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "   vLLM up ($label): $(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"; return 0
    fi
    if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "   server exited ($label); cause:"; root_cause_last; return 1; fi
    sleep 5
  done
  echo "   server did not come up in 10 min ($label); cause:"; root_cause_last; return 1
}
# attempt 1: defaults. attempt 2: conservative kernels (eager, Triton attention: what a new GPU generation such as
# Blackwell sm_120 needs when the FlashAttention wheels have no kernels for it). attempt 3: smaller footprint.
serve_once "defaults" && exit 0
cp "$LOG" "${LOG%.log}.attempt1.log"
if grep -q "No supported CUDA architectures" "$LOG"; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
  echo "   this vLLM wheel has no kernels for this GPU generation (compute capability ${cap:-?}); flags cannot fix that."
  if [ "${VLLM_TRY_NIGHTLY:-1}" = "1" ]; then
    echo "-- trying the vLLM nightly wheel (newer builds add new GPU generations)"
    if "$VENV/bin/python" -m pip install -q --pre -U vllm --extra-index-url https://wheels.vllm.ai/nightly >/tmp/vllm_nightly.log 2>&1; then
      serve_once "nightly wheel" && exit 0
    else echo "   nightly install failed: $(tail -1 /tmp/vllm_nightly.log | cut -c1-120)"; fi
  fi
  echo "   -> use an H100 runtime for vLLM (supported), or serve the model on one (colab/model_server.sh) and set MODEL_URL here."
  exit 1
fi
VLLM_ENV="VLLM_ATTENTION_BACKEND=TRITON_ATTN" serve_once "eager + Triton attention" --enforce-eager && exit 0
cp "$LOG" "${LOG%.log}.attempt2.log"
VLLM_ENV="VLLM_ATTENTION_BACKEND=TRITON_ATTN" VLLM_MAX_LEN=6144 VLLM_GPU_UTIL=0.6 serve_once "eager + Triton + small" --enforce-eager --max-num-batched-tokens 4096 && exit 0
echo "   all attempts failed; logs: ${LOG%.log}.attempt1.log ${LOG%.log}.attempt2.log $LOG"
exit 1
