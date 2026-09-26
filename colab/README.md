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
