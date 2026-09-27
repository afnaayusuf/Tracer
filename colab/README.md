# Colab workflow (Colab + GitHub only, no laptop)

Every session is two or three cells. Never `!cmd` per cell; `%%bash` runs a whole script.

**Cell 1 — env from Colab Secrets (Python).** Create a fine-grained GitHub token once
(Settings → Developer settings → Fine-grained tokens → this repo → Contents: read and write),
store it in the Colab Secrets panel (key icon) as `GH_TOKEN` with notebook access on.

```python
from google.colab import userdata
import os
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
os.environ["GH_REPO"]  = "owner/Tracer"          # your GitHub path
```

**Cell 2 — bootstrap (bash).** Clone or pull, auth, install, `make check`.

```
%%bash
bash <(curl -sL "https://raw.githubusercontent.com/$GH_REPO/main/colab/bootstrap.sh")
```

**Cell 3 — the session (bash).** Each session ships as `session_NN.zip` (drag it into the
Files panel → lands in `/content`). Its `apply.sh` copies the payload into the repo, runs
`make check` and the session's bench, commits and pushes.

```
%%bash
unzip -oq /content/session_02.zip -d /content && bash /content/session_02/apply.sh
```

Then paste the printed output back into the chat. `git pull` next time picks up the commit
wherever you open Colab.

Rules: `%cd` not `!cd`; data lives in R2 (later) not in the runtime; a killed runtime loses
nothing that was committed.

## Reasoning model in Colab

vLLM must not share the runtime's torch (Colab ships a CUDA 13 torch; vLLM wheels bring a
CUDA 12.8 torch and torchaudio then refuses to import). It lives in its own venv:

```
%%bash
bash colab/vllm_venv.sh start Qwen/Qwen3.5-4B      # installs once per runtime, serves on :8000
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --backend openai --ask "..."
bash colab/vllm_venv.sh stop
```

## After a runtime reset (checklist)

1. Runtime → Change runtime type → **L4 GPU** (a fresh runtime defaults to CPU; the preflight line
   then shows `torch ...+cpu` and `cuda_available: false`).
2. Run the env cell (GH_TOKEN, GH_REPO). Without it the session commits locally but cannot push.
3. Re-upload the clip to `/content/HI_DEF_VIDEO.mp4` (or set SOURCE) — `/content` is wiped on reset.
4. Run the session script. Postgres and the vLLM venv are rebuilt automatically (a few minutes).

## Live demo (one hour of footage + a question interface)

```
%%bash
# 1) ingest (background; paced to the file's clock with --realtime, or as fast as possible without it)
nohup python bench/run_ingest.py --source /content/hour.mp4 --start-time "2026-09-27T10:00:00+05:30" \
  --db "$DB_URL" --profile tier4_industrial --reid siglip --writer qwen --episode-min 10 > /tmp/ingest.log 2>&1 &
# 2) interface (share link printed)
python ui/app.py --db "$DB_URL" --backend transformers --model Qwen/Qwen3.5-4B --tz Asia/Kolkata --share
```
Questions are grounded before any model call: future times, times before the footage, off-topic
requests, device actions and identity-by-face requests are answered by the guards. Everything else
gets a window script (absolute clock times) and window-aware numeric tools.

## Backend + frontend (laptop browser -> HTTPS -> Colab)

```
%%bash
cd /content/Tracer
VI_DB="$DB_URL" VI_TZ=Asia/Kolkata bash colab/serve.sh start       # API on :8000 + the page at /, public URL printed
# start the one-hour file as a live stream (paced to its own clock), from the API:
curl -s -X POST http://127.0.0.1:8000/ingest/start -H 'Content-Type: application/json' \
  -d '{"source":"/content/hour.mp4","start_time":"2026-09-27T10:00:00+05:30","realtime":true,"profile":"tier4_industrial","writer":"none"}'
```
Open the printed `https://….trycloudflare.com` on the laptop. Endpoints: `/health`, `POST /ask`, `/episodes`,
`/events`, `/keyframes/<path>`, `/live/latest.jpg`, `POST /ingest/start|stop`, `/ingest/status`.
Memory on an L4: ingest with `"writer":"qwen"` plus the 4B agent is ~19 GB; keep the writer off during a live hour
or run the agent on the 2B.

## A multiplexed NVR export (one video, a grid of cameras)

Every cell becomes a virtual camera (`cam01`…`camNN`, row-major); all cells of a frame are detected in one
batched call. Burned-in labels are media zones. Questions can name a camera ("on cam 4").

```
curl -s -X POST http://127.0.0.1:8000/ingest/start -H 'Content-Type: application/json' \
  -d '{"source":"/content/mall_hour.mp4","grid":"4x4","model":"medium","profile":"tier2_public","start_time":"2026-09-27T10:00:00+05:30","realtime":true}'
```
Use `"model":"medium"` for grids: a 4x4 cell of a 1080p export is 480x270 and people are 30-60 px tall; medium's
576-px input costs the same as nano on the L4 and sees them better. Watch `realtime_factor` in the ingest log:
above 1.0 the sixteen cameras keep up with the clock.

Grid layout is detected from the seams between cells (`"grid": "auto"`, the default); force it only if the
detection line in `data/live/ingest.log` is wrong. One person seen by several cameras is one W-id in answers
and counts (cross-camera fusion by appearance + time); per-camera tracks stay auditable underneath.

## Tiles: cameras that see the same area

Identity is per tile, not per camera. The map is learned from the footage: two cameras that keep producing the
same-looking person at the same time share a tile; cameras that see the same person only in sequence are adjacent.

```
# after 10-20 minutes of footage:
requests.post("http://127.0.0.1:8000/tiles/recompute").json()      # -> {"tiles": {"T1": ["cam01","cam02",...]}, ...}
requests.post("http://127.0.0.1:8000/ingest/stop"); requests.post("http://127.0.0.1:8000/ingest/start", json={...})
```
The restarted ingest runs one linker per tile: a person seen by four cameras of a tile has one id (`T1:E3`) from the
moment they are seen twice. Until a map exists, identities are per camera and answers join them as W-ids.
