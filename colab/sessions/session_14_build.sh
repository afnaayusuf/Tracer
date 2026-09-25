#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 14 build: first real answers.
#   * store: every *_ms column is BigInteger (UTC ms overflowed PostgreSQL INTEGER; that is why
#     get_script returned 0 in session 13); DDL test for the Postgres dialect
#   * agent_replay shows tool errors in the trace line
#   * colab/vllm_venv.sh: vLLM in its own virtualenv (isolated torch), served in the background
#  Runs: (Postgres) -> slice (frame + SigLIP) -> store -> vLLM venv serve Qwen/Qwen3.5-4B ->
#        three questions through the loop with the real model (fake backend only if it cannot start).
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_14.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       DB_URL AGENT_MODEL (Qwen/Qwen3.5-4B) VLLM_GPU_UTIL (0.6) SKIP_VLLM=1 NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 14"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
git pull -q --ff-only || die "local branch diverged from origin; resolve manually"
[ -n "$(git log --grep='^session 13' --format=%h)" ] || die "session 13 commit not found; run build_session_13.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/vllm_venv.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_reid.py
tests/test_schemas.py
tests/test_slice_gpu_smoke.py
tests/test_store_agent.py
tests/test_tubes.py
vi/__init__.py
vi/agent/__init__.py
vi/agent/loop.py
vi/agent/tools.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/fake.py
vi/detect/rfdetr.py
vi/detect/roi.py
vi/episode/__init__.py
vi/episode/debug.py
vi/episode/keyframes.py
vi/episode/writer.py
vi/eval/__init__.py
vi/eval/mot.py
vi/events/__init__.py
vi/events/compiler.py
vi/events/zones.py
vi/gate/__init__.py
vi/gate/base.py
vi/gate/framediff.py
vi/gate/heartbeat.py
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
vi/reid/__init__.py
vi/reid/base.py
vi/schemas/__init__.py
vi/schemas/common.py
vi/schemas/contact_sheet.py
vi/schemas/episode.py
vi/schemas/event.py
vi/schemas/export.py
vi/schemas/fact.py
vi/schemas/scene_card.py
vi/schemas/tick.py
vi/schemas/tube.py
vi/store/__init__.py
vi/store/db.py
vi/store/loader.py
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/bytetrack.py
vi/tubes/geometry.py
vi/tubes/kalman.py
vi/tubes/linker.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p colab/sessions data/bench
cat > vi/store/db.py << 'EOF_VI'
"""Relational store for episodes (R24). All *_ms columns are BigInteger: UTC milliseconds
(~1.7e12) overflow PostgreSQL INTEGER (E-ING-06; found in session 13). SQLAlchemy Core so the same code runs on SQLite (tests,
laptop) and PostgreSQL 18 (Colab, production). Embeddings are JSON here; pgvector columns come
with the retrieval work. Every table is keyed by the deterministic ids the writers already
produce, so loading is idempotent (E-STO-02)."""
from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

metadata = MetaData()

episodes = Table(
    "episodes", metadata,
    Column("episode_id", String, primary_key=True), Column("tile_id", String, index=True),
    Column("camera_ids", JSON), Column("t0_ms", BigInteger, index=True), Column("t1_ms", BigInteger),
    Column("status", String), Column("kb_version", Integer), Column("schema_version", String),
)
ticks = Table(
    "ticks", metadata,
    Column("episode_id", String, primary_key=True), Column("camera_id", String, primary_key=True),
    Column("tick_index", Integer, primary_key=True), Column("t_start_ms", BigInteger, index=True),
    Column("t_end_ms", BigInteger), Column("modality", String), Column("tubes", JSON),
    Column("event_ids", JSON), Column("scene_state", JSON), Column("gate_energy", Float),
)
tubes = Table(
    "tubes", metadata,
    Column("tube_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("camera_id", String, index=True), Column("tile_id", String), Column("entity_id", String, index=True),
    Column("named", String), Column("class_label", String, index=True), Column("state", String),
    Column("born_ms", BigInteger, index=True), Column("last_seen_ms", BigInteger), Column("box", JSON),
    Column("zone_ids", JSON), Column("modality", String), Column("keyframe_refs", JSON),
    Column("attributes", JSON), Column("merge_candidates", JSON),
)
entities = Table(
    "entities", metadata,
    Column("entity_id", String, primary_key=True), Column("camera_id", String), Column("named", String),
    Column("class_label", String), Column("tube_ids", JSON), Column("first_seen_ms", BigInteger, index=True),
    Column("last_seen_ms", BigInteger), Column("best_keyframe_ref", String), Column("embedding", JSON),
)
events = Table(
    "events", metadata,
    Column("event_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("type", String, index=True), Column("t_ms", BigInteger, index=True), Column("camera_id", String),
    Column("tile_id", String), Column("zone_id", String, index=True), Column("subject_tube_ids", JSON),
    Column("subject_entity_ids", JSON), Column("object_ids", JSON), Column("payload", JSON),
    Column("confidence", Float), Column("source", String), Column("dedupe_key", String),
)
patches = Table(
    "patches", metadata,
    Column("patch_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("tube_id", String, index=True), Column("produced_at_ms", BigInteger), Column("source", String),
    Column("payload", JSON), Column("confidence", Float), Column("modality", String),
    Column("enhanced", Boolean), Column("failed", Boolean),
)
custody = Table(
    "custody", metadata,
    Column("event_id", String, primary_key=True), Column("object_id", String, index=True),
    Column("person_tube_ids", JSON), Column("t_ms", BigInteger), Column("kind", String), Column("zone_id", String),
)
facts = Table(
    "facts", metadata,
    Column("fact_id", String, primary_key=True), Column("subject", String, index=True), Column("predicate", String),
    Column("object", JSON), Column("status", String), Column("confidence", Float), Column("source", String),
    Column("support", Integer), Column("contradictions", Integer), Column("evidence", JSON),
    Column("first_seen_ms", BigInteger), Column("last_confirmed_ms", BigInteger), Column("version", Integer),
)
scripts = Table(
    "scripts", metadata,
    Column("episode_id", String, primary_key=True), Column("text", Text), Column("rendered_at_ms", BigInteger),
)


def connect(url: str = "sqlite+pysqlite:///:memory:") -> Engine:
    """sqlite+pysqlite:///data/vi.db for a file, postgresql+psycopg://vi:vi@localhost/vi for Postgres."""
    engine = create_engine(url, future=True)
    metadata.create_all(engine)
    return engine


def insert_ignore(conn, table: Table, rows: list[dict]) -> int:
    """Idempotent insert for both dialects (E-STO-02)."""
    if not rows:
        return 0
    if conn.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    else:
        from sqlalchemy.dialects.sqlite import insert as sq_insert
        stmt = sq_insert(table).values(rows).on_conflict_do_nothing()
    return conn.execute(stmt).rowcount
EOF_VI
cat > bench/agent_replay.py << 'EOF_VI'
"""Block 2 dry run: load episode files into the store, print the scene script, answer the three
canned tool calls a question would need. No LLM yet; this is the evidence layer the agent reads.

  python bench/agent_replay.py --db sqlite+pysqlite:///data/vi.db data/episodes/*.jsonl
  python bench/agent_replay.py --db postgresql+psycopg://vi:vi@localhost/vi data/episodes/*.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vi.agent import FakeBackend, OpenAIBackend, ask, clip, get_script, search_entities, search_events, search_tubes
from vi.store import connect, load_episode_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--script-lines", type=int, default=40)
    ap.add_argument("--ask", action="append", default=[], help="question(s) to run through the agent loop")
    ap.add_argument("--backend", choices=["fake", "openai"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    engine = connect(a.db)
    loaded = [load_episode_file(engine, p) for p in a.paths]
    ep = loaded[-1]["episode_id"]
    script = get_script(engine, ep, cache=False)
    print("\n".join(script.splitlines()[: a.script_lines]))
    if len(script.splitlines()) > a.script_lines:
        print(f"  ... ({len(script.splitlines()) - a.script_lines} more lines)")
    ents = search_entities(engine, class_label="person")
    enters = search_events(engine, event_type="enter_zone")
    relinks = search_events(engine, event_type="relink")
    first = ents[0]["entity_id"] if ents else None
    evidence = clip(engine, entity_id=first) if first else {}
    row = {"ring": "agent", "db": a.db.split("://")[0], "episodes_loaded": len(loaded), "episode_id": ep,
           "rows": {k: sum(l.get(k, 0) for l in loaded) for k in ("ticks", "events", "tubes", "entities", "patches", "custody")},
           "person_entities": len(ents), "enter_events": len(enters), "relink_events": len(relinks),
           "script_chars": len(script), "script_lines": len(script.splitlines()),
           "clip_first_entity": {"tubes": len(evidence.get("tube_ids", [])), "keyframes": len(evidence.get("keyframe_refs", []))},
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    answers = []
    if a.ask:
        backend = FakeBackend() if a.backend == "fake" else OpenAIBackend(base_url=a.base_url, model=a.model)
        for q in a.ask:
            res = ask(engine, q, ep, backend)
            answers.append(res)
            calls = [f"{t['tool']}({', '.join(f'{k}={v}' for k, v in t['args'].items())}) -> "
                     + (f"ERROR {t['error'][:80]}" if t.get('error') else str(t['results'])) for t in res["trace"] if "tool" in t]
            print(f"\nQ: {q}\n   tools: " + " | ".join(calls))
            f = res["final"]
            if f["action"] == "answer":
                print(f"   A ({f.get('confidence', 0):.2f}{'' if f['cited'] else ', UNCITED'}): {f['text']}\n   cites: {', '.join(f['citations']) or '—'}")
            else:
                print(f"   clarify: {f['question']}")
        row["answers"] = [{"q": r["question"], "steps": r["steps"], "cited": r["final"].get("cited"),
                           "action": r["final"]["action"], "backend": r["backend"]} for r in answers]
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "agent.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "answers"}, indent=2))


if __name__ == "__main__":
    main()
EOF_VI
cat > tests/test_store_agent.py << 'EOF_VI'
import subprocess
import sys

import pytest

from vi.agent import clip, get_script, search_entities, search_events, search_tubes
from vi.store import connect, load_episode_file


@pytest.fixture
def episode_path(tmp_path):
    out = subprocess.run([sys.executable, "bench/slice_cpu.py", str(tmp_path / "episodes")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    return next((tmp_path / "episodes").glob("*.jsonl"))


@pytest.mark.edge("E-STO-02")
def test_loader_is_idempotent_and_derives_entities(episode_path):
    engine = connect()
    first = load_episode_file(engine, episode_path)
    second = load_episode_file(engine, episode_path)
    assert first["ticks"] > 0 and first["events"] > 0 and first["tubes"] >= 1 and first["entities"] >= 1
    assert all(second[k] == 0 for k in ("ticks", "events", "tubes", "entities"))       # nothing duplicated
    ents = search_entities(engine)
    assert ents and ents[0]["entity_id"].startswith("anon:")                             # no ReID in the CPU slice


def test_search_and_clip_return_citable_ids(episode_path):
    engine = connect()
    load_episode_file(engine, episode_path)
    pick = search_events(engine, event_type="pickup")
    assert len(pick) == 1 and pick[0]["object_ids"] == ["bike_keys"] and pick[0]["event_id"].startswith("ev_")
    suspect = pick[0]["subject_tube_ids"][0]
    tubes_ = search_tubes(engine, class_label="person")
    assert any(t["tube_id"] == suspect for t in tubes_)
    ev = clip(engine, tube_id=suspect)
    assert ev["tube_ids"] == [suspect] and ev["segments"][0]["camera_id"] == "cam1"
    window = search_events(engine, t_start_ms=9000, t_end_ms=11000)
    assert {e["type"] for e in window} <= {"enter_zone", "dwell", "exit_zone"} and window


def test_scene_script_is_deterministic_and_cites(episode_path):
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    s1 = get_script(engine, ep, cache=False)
    s2 = get_script(engine, ep, cache=True)
    s3 = get_script(engine, ep, cache=True)
    assert s1 == s2 == s3 and s1.startswith("EPISODE ") and "CAST:" in s1 and "TIMELINE:" in s1
    assert "pickup" in s1 and "object bike_keys" in s1 and "[ev_" in s1
    custody_line = next(l for l in s1.splitlines() if "pickup" in l)
    assert "anon:cam1:" in custody_line          # subject cited by entity + tube


@pytest.mark.edge("E-AGT-06")
def test_agent_loop_cites_only_ids_it_was_shown(episode_path):
    from vi.agent import FakeBackend, ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    out = ask(engine, "Who took the bike keys?", ep, FakeBackend())
    assert out["final"]["action"] == "answer" and out["final"]["cited"]
    assert any(c.startswith("ev_") for c in out["final"]["citations"])
    assert [t["tool"] for t in out["trace"] if "tool" in t][:2] == ["get_script", "search_events"]
    bad = ask(engine, "Who took the bike keys?", ep, FakeBackend(bogus_citation=True))
    revise = next(t for t in bad["trace"] if "revise" in t)     # asked to revise once, naming the bad id
    assert revise["invalid"] == ["ev_deadbeefdeadbeef"] and bad["final"]["cited"] is False


@pytest.mark.edge("E-AGT-02")
def test_agent_loop_reports_no_results_honestly(episode_path):
    from vi.agent import FakeBackend, ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    out = ask(engine, "Did anybody fall? (nobody did)", ep, FakeBackend())
    assert out["final"]["action"] == "answer" and "No matching" in out["final"]["text"] and out["final"]["citations"] == []


@pytest.mark.edge("E-ING-06")
def test_every_millisecond_column_is_64_bit_on_postgres():
    """UTC ms (~1.7e12) overflow PostgreSQL INTEGER; SQLite would never notice."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable
    from vi.store.db import metadata
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for col in table.columns:
            if col.name.endswith("_ms"):
                assert f"{col.name} BIGINT" in ddl, f"{table.name}.{col.name} must be BIGINT"


def test_store_round_trips_a_utc_millisecond_timestamp():
    import time
    from vi.store import connect
    from vi.store.db import insert_ignore, scripts
    from sqlalchemy import select
    engine = connect()
    now = int(time.time() * 1000)
    with engine.begin() as conn:
        insert_ignore(conn, scripts, [dict(episode_id="ep_x", text="t", rendered_at_ms=now)])
    with engine.connect() as conn:
        assert conn.execute(select(scripts.c.rendered_at_ms)).scalar() == now
EOF_VI
cat > colab/vllm_venv.sh << 'EOF_VI'
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
EOF_VI
cat > colab/README.md << 'EOF_VI'
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
EOF_VI
chmod +x colab/vllm_venv.sh
cp "$0" colab/sessions/session_14_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then
    echo "installing PostgreSQL (~1-2 min)"
    (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed: $(tail -2 /tmp/apt.log | tr '\n' ' ')"
  fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    # schema changed (BigInteger): start from a clean database
    sudo -u postgres psql -qc "DROP DATABASE vi;" && sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"
    DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
    echo "PostgreSQL: $(sudo -u postgres psql -tAc 'select version()' | cut -d, -f1)"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "5. slice (frame + SigLIP) -> episode -> store"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "\"(person_tubes|entities|relinks)\"" /tmp/slice_final.log | sed 's/^/  /'
else
  warn "no GPU or no clip; using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "6. reasoning model: vLLM in its own venv"
BACKEND=fake
if [ "${SKIP_VLLM:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if bash colab/vllm_venv.sh start "$AGENT_MODEL" 8000; then BACKEND=openai; else warn "falling back to the fake backend"; fi
else
  warn "no GPU or SKIP_VLLM=1: using the fake backend"
fi

step "7. three questions ($BACKEND backend, $AGENT_MODEL)"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 0 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was in the exit_right zone, when did they arrive, and is there a keyframe for them?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -v "^\s*\"" | grep -v "^[{}]"
[ "$BACKEND" = openai ] && bash colab/vllm_venv.sh stop >/dev/null 2>&1 || true

step "8. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: BigInteger ms columns (E-ING-06), tool errors surfaced, vLLM in an isolated venv"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "9. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste the three Q/A blocks back (tools, answer, cites)."
