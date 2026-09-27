#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 32 build: restart-safe, follow-ups, deadlines (supersedes 31).
#   * the script now STOPS the previous ingest/API/tunnel before touching the database (31 died there)
#   * 'a minute ago', 'just now', 'last few minutes' are grounded; 'yes exactly' resolves the last clarification
#   * /ask answers within 80 s or says why; churn episode cuts need 2 minutes (789 episodes/hour was noise)
#   * grid layout auto-detected from seams (30); cross-camera W-ids at query time (30)
#   * vi/fusion/tiles.py: cameras that keep showing the same-looking person at the same time share a tile;
#     POST /tiles/recompute learns the map from the store; the next ingest runs ONE linker per tile
#   * tile-aware linker: a tube born on camera B while the entity is live on camera A links (T1:E3)
#  (superset of sessions 27 + 28: live ingest, window agent with guards, backend API, your frontend)
#   * --grid RxC: every cell of a multiplexed NVR export is a virtual camera; all cells of a frame are
#     detected in ONE batched call; burned-in labels are media zones; live view recomposed as the grid
#   * questions can name a camera ('on cam 4'); unknown camera numbers are answered with the list
#   * border quality rule only for brief tubes (people who exit through the frame edge are people)
#   * vi/api/server.py (FastAPI): serves ui/web at /, POST /ask (guards -> window agent -> text, mood,
#     citations, evidence keyframes, window, latency), /health, /episodes, /events, /keyframes/<path>,
#     /live/latest.jpg, POST /ingest/start|stop (the one-hour file as a paced live stream), /ingest/status
#   * ui/web/index.html: your page, answer() wired to /ask, evidence strip under bot replies, live frame
#     + footage status when "Cameras" is selected; assets and fonts vendored (OFL)
#   * run_ingest writes data/live/latest.jpg (annotated) + status.json every 4 ticks
#   * colab/serve.sh: uvicorn in the background + cloudflared quick tunnel -> public HTTPS URL for the laptop
#  RUN (one Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    os.environ["SOURCE"] = "/content/PEDIDO 27057.mp4"                      # GRID defaults to auto
#    os.environ["START_TIME"] = "2026-09-27T10:00:00+05:30"; os.environ["TZ_NAME"] = "Asia/Kolkata"
#    !bash /content/build_session_32.sh
#  It leaves the API running and prints the public URL; the ingest of SOURCE starts as a live stream.
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/PEDIDO 27057.mp4}"
GRID="${GRID:-auto}"          # auto: layout detected from the seams; or 2x2 / 4x4 to force
DET_MODEL="${DET_MODEL:-medium}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-2B}"   # 2B while a multi-camera ingest shares the GPU; 4B once ingest is idle
PROFILE="${PROFILE:-tier2_public}"
TZ_NAME="${TZ_NAME:-UTC}"
START_TIME="${START_TIME:-}"
WRITER="${WRITER:-none}"          # qwen to describe people during the live run (needs ~19 GB with the 4B agent)
PORT="${PORT:-8000}"
SESSION="session 32"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] || warn "GH_TOKEN is not set: will commit locally but cannot push"

step "0. runtime checks"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ "$HAS_GPU" = 1 ]; then echo "GPU: $(nvidia-smi -L | head -1)"; else warn "NO GPU -> Runtime > Change runtime type > L4"; fi
[ -f "$SOURCE" ] && echo "clip: $SOURCE ($(du -h "$SOURCE" | cut -f1))" || warn "NO CLIP at $SOURCE"

step "1. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin || true
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"
if [ -n "$(git status --porcelain)" ]; then warn "discarding uncommitted changes from an interrupted run"; git checkout -- . && git clean -fdq; fi
git pull -q --ff-only 2>/dev/null || warn "pull skipped"
[ -n "$(git log --grep='^session 26' --format=%h)" ] || die "session 26 commit not found; run build_session_26.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/cast_sheet.py
bench/metamorphic.py
bench/reid_eval.py
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/run_ingest.py
bench/scenario_eval.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/preflight.sh
colab/serve.sh
colab/vllm_venv.sh
edge_cases.yaml
profiles/common.yaml
profiles/tier1_hospital.yaml
profiles/tier2_public.yaml
profiles/tier3_office.yaml
profiles/tier4_industrial.yaml
profiles/tier5_residential.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/scenarios/warehouse.yaml
tests/scenarios/warehouse_identities.yaml
tests/test_api.py
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_profiles_generator.py
tests/test_reid.py
tests/test_schemas.py
tests/test_slice_gpu_smoke.py
tests/test_store_agent.py
tests/test_tubes.py
ui/app.py
ui/web/assets/avatar.jpg
ui/web/assets/cam.png
ui/web/assets/dp-bot-unsure.png
ui/web/assets/dp-bot.png
ui/web/assets/dp-user.png
ui/web/assets/help.png
ui/web/assets/home.png
ui/web/assets/mic.png
ui/web/fonts/LICENSE-LINE-Seed-OFL.txt
ui/web/fonts/LINESeedJP-400.woff2
ui/web/fonts/LINESeedJP-700.woff2
ui/web/fonts/LINESeedJP-800.woff2
ui/web/index.html
vi/__init__.py
vi/agent/__init__.py
vi/agent/loop.py
vi/agent/scope.py
vi/agent/timeground.py
vi/agent/tools.py
vi/api/__init__.py
vi/api/server.py
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
vi/fusion/__init__.py
vi/fusion/tiles.py
vi/gate/__init__.py
vi/gate/base.py
vi/gate/framediff.py
vi/gate/heartbeat.py
vi/generator/__init__.py
vi/generator/registry.py
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/grid.py
vi/ingest/reader.py
vi/ingest/synthetic.py
vi/profiles.py
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
vi/tubes/quality.py
vi/tubes/simple_iou.py
vi/writer/__init__.py
vi/writer/contact_sheet.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); rerun with FORCE=1 to continue"; fi
echo "audit OK"

step "2. write $SESSION files (superset of sessions 27-32)"
mkdir -p vi/api vi/fusion ui/web/assets ui/web/fonts colab/sessions data/bench data/live
cat > vi/agent/timeground.py << 'EOF_VI'
"""Deterministic time grounding for questions (E-AGT-04). The model never resolves clock words;
this does, against the footage bounds, and the loop refuses or clarifies before any model call.
All times are absolute UTC milliseconds; display strings use the site timezone."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass
class Grounding:
    kind: str                      # ok | future | before_start | ambiguous | none
    t_start_ms: int | None = None
    t_end_ms: int | None = None
    message: str = ""
    matched: str = ""


NUMW = r"(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|couple of|few|half an?)"
REL = re.compile(r"\b(last|past|previous)\s+" + NUMW + r"?\s*(min|mins|minute|minutes|hour|hours|hr|hrs|sec|seconds)\b", re.I)
AGO = re.compile(r"\b" + NUMW + r"\s*(min|mins|minute|minutes|hour|hours|hr|hrs)\s+ago\b", re.I)
JUST_NOW = re.compile(r"\b(just now|right now|a moment ago|moments ago|just happened|currently|at the moment)\b", re.I)
WORDS_N = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "fifteen": 15, "twenty": 20, "thirty": 30, "couple of": 2, "few": 3, "half an": 0.5, "half a": 0.5}


def _n(tok: str | None) -> float:
    if not tok:
        return 1
    t = tok.strip().lower()
    return float(t) if t.isdigit() else WORDS_N.get(t, 1)
CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", re.I)
BETWEEN = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[,.?]|$)", re.I)
AFTER = re.compile(r"\b(after|since|from)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I)
BEFORE = re.compile(r"\b(before|until|till)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I)
WORDS = {"now": 0, "right now": 0, "currently": 0, "at the moment": 0}
TOMORROW = re.compile(r"\b(tomorrow|next week|next hour|later today|tonight|in \d+ (minutes|hours))\b", re.I)


def _clock_to_ms(text: str, ref_day: datetime, tz: ZoneInfo) -> int | None:
    m = CLOCK.search(text)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "").lower()
    if ap == "pm" and h < 12: h += 12
    if ap == "am" and h == 12: h = 0
    if not ap and h > 23:
        return None
    local = ref_day.astimezone(tz).replace(hour=h, minute=mi, second=0, microsecond=0)
    return int(local.timestamp() * 1000)


def fmt(ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ms / 1000, tz).strftime("%H:%M:%S")


def ground(question: str, now_ms: int, start_ms: int, end_ms: int, tz_name: str = "UTC") -> Grounding:
    """now_ms = latest processed timestamp; [start_ms, end_ms] = footage bounds in the store."""
    tz = ZoneInfo(tz_name)
    q = question.strip()
    ref = datetime.fromtimestamp(now_ms / 1000, tz)
    if TOMORROW.search(q):
        return Grounding("future", message=f"That is after the latest footage I have, which ends at {fmt(end_ms, tz)}.", matched=TOMORROW.search(q).group(0))
    m = JUST_NOW.search(q)
    if m:
        return Grounding("ok", max(start_ms, now_ms - 120_000), now_ms, matched=m.group(0))
    m = REL.search(q)
    if m:
        n, unit = _n(m.group(2)), m.group(3).lower()
        span = int(n * (3600 if unit.startswith("h") else 1 if unit.startswith("s") else 60) * 1000)
        return Grounding("ok", max(start_ms, now_ms - span), now_ms, matched=m.group(0))
    m = AGO.search(q)
    if m:
        n, unit = _n(m.group(1)), m.group(2).lower()
        span = int(n * (3600 if unit.startswith("h") else 60) * 1000)
        t = now_ms - span
        if t < start_ms:
            return Grounding("before_start", message=f"That is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
        return Grounding("ok", max(start_ms, t - 120_000), min(end_ms, t + 120_000), matched=m.group(0))
    m = BETWEEN.search(q)
    if m:
        a, b = _clock_to_ms(m.group(1), ref, tz), _clock_to_ms(m.group(2), ref, tz)
        if a is not None and b is not None:
            a, b = min(a, b), max(a, b)
            if a > end_ms:
                return Grounding("future", message=f"{fmt(a, tz)} is after the latest footage I have ({fmt(end_ms, tz)}).", matched=m.group(0))
            if b < start_ms:
                return Grounding("before_start", message=f"That window is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
            return Grounding("ok", max(a, start_ms), min(b, end_ms), matched=m.group(0))
    m = AFTER.search(q)
    if m:
        a = _clock_to_ms(m.group(2), ref, tz)
        if a is not None:
            if a > end_ms:
                return Grounding("future", message=f"{fmt(a, tz)} is after the latest footage I have ({fmt(end_ms, tz)}).", matched=m.group(0))
            return Grounding("ok", max(a, start_ms), end_ms, matched=m.group(0))
    m = BEFORE.search(q)
    if m:
        b = _clock_to_ms(m.group(2), ref, tz)
        if b is not None:
            if b < start_ms:
                return Grounding("before_start", message=f"That is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
            return Grounding("ok", start_ms, min(b, end_ms), matched=m.group(0))
    m = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)\b", q, re.I)
    if m:
        t = _clock_to_ms(m.group(1), ref, tz)
        if t is not None:
            if t > end_ms + 60_000:
                return Grounding("future", message=f"{fmt(t, tz)} is after the latest footage I have ({fmt(end_ms, tz)}).", matched=m.group(0))
            if t < start_ms - 60_000:
                return Grounding("before_start", message=f"{fmt(t, tz)} is before the footage starts at {fmt(start_ms, tz)}.", matched=m.group(0))
            return Grounding("ok", max(start_ms, t - 120_000), min(end_ms, t + 120_000), matched=m.group(0))
    return Grounding("none", start_ms, end_ms)
EOF_VI
cat > vi/agent/scope.py << 'EOF_VI'
"""Out-of-scope detection before any model call: questions that are not about the footage, or
that ask for things the system does not do (identify unnamed people by face, predict, act)."""
from __future__ import annotations

import re

OFF_TOPIC = re.compile(r"\b(weather|stock|recipe|write (me )?(code|a poem|an essay)|translate|capital of|who won|"
                       r"tell me a joke|your opinion|what do you think about (the )?(news|politics))\b", re.I)
PREDICT = re.compile(r"\b(will|going to|predict|forecast|expect(ed)? to)\b", re.I)
ACT = re.compile(r"\b(call|alert|notify|lock|unlock|open the|close the|turn (on|off)|send (an? )?(email|message))\b", re.I)
IDENTITY = re.compile(r"\b(who is (he|she|that|this)|what('s| is) (his|her|their) name|identify (him|her|them|the person))\b", re.I)
FOOTAGE_WORDS = re.compile(r"\b(camera|footage|video|clip|zone|conveyor|table|door|person|people|worker|someone|anyone|"
                           r"who|where|when|how many|what (happened|was|were)|did|entered|left|arrive|carry|wearing)\b", re.I)


def classify(question: str) -> tuple[str, str]:
    """returns (kind, message). kind: ok | off_topic | predict | act | identity"""
    q = question.strip()
    if ACT.search(q):
        return "act", "I only report what the cameras recorded; I can't take actions on devices or send messages."
    if PREDICT.search(q) and not re.search(r"\b(was|were|did|happened)\b", q, re.I):
        return "predict", "I can only describe what has already been recorded, not what will happen."
    if OFF_TOPIC.search(q) and not FOOTAGE_WORDS.search(q):
        return "off_topic", "That isn't something the footage can answer. Ask about people, objects, zones, times or events in the recording."
    if IDENTITY.search(q):
        return "identity", ("I don't know names unless someone has been named in the gallery; I can describe the person, "
                            "show their keyframe, and you can name them from there.")
    return "ok", ""
EOF_VI
cat > vi/agent/tools.py << 'EOF_VI'
"""Deterministic Block 2 tools (R26): the reasoning model calls these; it never sees video.
search_* are SQL filters; get_script renders the scene script the model reads; clip returns
evidence references. Every result carries the ids the model must cite (R27)."""
from __future__ import annotations

import time

from sqlalchemy import and_, func, or_, select
from sqlalchemy.engine import Engine

from vi.store.db import entities, episodes, events, insert_ignore, scripts, tubes


def _ts(ms: int, t0: int = 0) -> str:
    s = max(0, ms - t0) / 1000.0
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def search_events(engine: Engine, *, tile_id: str | None = None, camera_id: str | None = None,
                  t_start_ms: int | None = None, t_end_ms: int | None = None, event_type: str | list[str] | None = None,
                  zone_id: str | None = None, entity_id: str | None = None, tube_id: str | None = None,
                  limit: int = 200) -> list[dict]:
    q = select(events)
    conds = []
    if tile_id: conds.append(events.c.tile_id == tile_id)
    if camera_id: conds.append(events.c.camera_id == camera_id)
    if t_start_ms is not None: conds.append(events.c.t_ms >= t_start_ms)
    if t_end_ms is not None: conds.append(events.c.t_ms <= t_end_ms)
    if event_type:
        types = [event_type] if isinstance(event_type, str) else list(event_type)
        conds.append(events.c.type.in_(types))
    if zone_id: conds.append(events.c.zone_id == zone_id)
    if conds:
        q = q.where(and_(*conds))
    q = q.order_by(events.c.t_ms).limit(limit * 4 if (entity_id or tube_id) else limit)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(q)]
    if tube_id:
        rows = [r for r in rows if tube_id in (r["subject_tube_ids"] or [])]
    if entity_id:
        tube_ids = set(_tubes_of_entity(engine, entity_id))
        rows = [r for r in rows if entity_id in (r["subject_entity_ids"] or []) or tube_ids & set(r["subject_tube_ids"] or [])]
    return rows[:limit]


def search_tubes(engine: Engine, *, tile_id: str | None = None, camera_id: str | None = None,
                 class_label: str | None = None, t_start_ms: int | None = None, t_end_ms: int | None = None,
                 entity_id: str | None = None, named: str | None = None, min_life_ms: int = 0, limit: int = 200) -> list[dict]:
    q = select(tubes)
    conds = []
    if tile_id: conds.append(tubes.c.tile_id == tile_id)
    if camera_id: conds.append(tubes.c.camera_id == camera_id)
    if class_label: conds.append(tubes.c.class_label == class_label)
    if entity_id: conds.append(tubes.c.entity_id == entity_id)
    if named: conds.append(tubes.c.named == named)
    if t_start_ms is not None: conds.append(tubes.c.last_seen_ms >= t_start_ms)   # overlaps the window
    if t_end_ms is not None: conds.append(tubes.c.born_ms <= t_end_ms)
    if min_life_ms: conds.append(tubes.c.last_seen_ms - tubes.c.born_ms >= min_life_ms)
    if conds:
        q = q.where(and_(*conds))
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(q.order_by(tubes.c.born_ms).limit(limit))]


def search_entities(engine: Engine, *, camera_id: str | None = None, class_label: str | None = None,
                    t_start_ms: int | None = None, t_end_ms: int | None = None, named: str | None = None,
                    limit: int = 200) -> list[dict]:
    q = select(entities)
    conds = []
    if camera_id: conds.append(entities.c.camera_id == camera_id)
    if class_label: conds.append(entities.c.class_label == class_label)
    if named: conds.append(entities.c.named == named)
    if t_start_ms is not None: conds.append(entities.c.last_seen_ms >= t_start_ms)
    if t_end_ms is not None: conds.append(entities.c.first_seen_ms <= t_end_ms)
    if conds:
        q = q.where(and_(*conds))
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(q.order_by(entities.c.first_seen_ms).limit(limit))]


def episode_window(engine: Engine, episode_id: str) -> tuple[int, int]:
    with engine.connect() as conn:
        row = conn.execute(select(episodes.c.t0_ms, episodes.c.t1_ms).where(episodes.c.episode_id == episode_id)).first()
    if row is None:
        raise KeyError(episode_id)
    return int(row[0]), int(row[1] if row[1] is not None else row[0])


def coverage(engine: Engine, episode_id: str, class_label: str = "person", confirmed_only: bool = True) -> list[dict]:
    """Per entity: seen interval, coverage of the episode window (0-1), tube count, quality.
    Deterministic; this is what the agent must use for 'how long', 'whole time', 'how many'."""
    t0, t1 = episode_window(engine, episode_id)
    span = max(1, t1 - t0)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(and_(tubes.c.episode_id == episode_id, tubes.c.class_label == class_label)))]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["entity_id"], []).append(r)
    out = []
    for eid, rs in by.items():
        ok = any(r.get("quality", "ok") == "ok" for r in rs)
        if confirmed_only and not ok:
            continue
        # union of tube intervals (clipped to the window)
        ivs = sorted((max(t0, r["born_ms"]), min(t1, r["last_seen_ms"])) for r in rs)
        covered, cur = 0, None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: covered += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur: covered += cur[1] - cur[0]
        out.append({"entity_id": eid, "first_seen_ms": min(r["born_ms"] for r in rs), "last_seen_ms": max(r["last_seen_ms"] for r in rs),
                    "coverage": round(covered / span, 3), "tubes": len(rs), "quality": "ok" if ok else "low"})
    return sorted(out, key=lambda x: (-x["coverage"], x["first_seen_ms"]))


def count_entities(engine: Engine, episode_id: str, t_start_ms: int | None = None, t_end_ms: int | None = None,
                   class_label: str = "person", confirmed_only: bool = True) -> dict:
    """Distinct confirmed entities present in [t_start, t_end] (default: the whole episode)."""
    t0, t1 = episode_window(engine, episode_id)
    a, b = (t0 if t_start_ms is None else t_start_ms), (t1 if t_end_ms is None else t_end_ms)
    rows = coverage(engine, episode_id, class_label, confirmed_only)
    present = [r["entity_id"] for r in rows if r["last_seen_ms"] >= a and r["first_seen_ms"] <= b]
    return {"count": len(present), "entity_ids": present, "window_ms": [a, b], "confirmed_only": confirmed_only}


def entities_present(engine: Engine, episode_id: str, min_coverage: float = 0.9, class_label: str = "person") -> dict:
    """Entities seen for at least min_coverage of the episode: the deterministic answer to
    'who stayed the whole time' (0.9 tolerates short occlusions)."""
    rows = coverage(engine, episode_id, class_label)
    ids = [r["entity_id"] for r in rows if r["coverage"] >= min_coverage]
    return {"min_coverage": min_coverage, "count": len(ids), "entity_ids": ids,
            "coverage": {r["entity_id"]: r["coverage"] for r in rows}}


def _tubes_of_entity(engine: Engine, entity_id: str) -> list[str]:
    with engine.connect() as conn:
        row = conn.execute(select(entities.c.tube_ids).where(entities.c.entity_id == entity_id)).first()
    return list(row[0]) if row else []


def clip(engine: Engine, *, entity_id: str | None = None, tube_id: str | None = None, mode: str = "crop") -> dict:
    """Evidence references for an entity or tube: keyframe crops now; video segments when the
    retention tier exists (E-STO-03). Returns refs, never pixels."""
    with engine.connect() as conn:
        if tube_id:
            rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.tube_id == tube_id))]
        elif entity_id:
            rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.entity_id == entity_id).order_by(tubes.c.born_ms))]
        else:
            raise ValueError("entity_id or tube_id required")
    refs = [k for r in rows for k in (r["keyframe_refs"] or [])]
    return {"mode": mode, "entity_id": entity_id or (rows[0]["entity_id"] if rows else None),
            "tube_ids": [r["tube_id"] for r in rows], "keyframe_refs": refs,
            "segments": [{"camera_id": r["camera_id"], "t_start_ms": r["born_ms"], "t_end_ms": r["last_seen_ms"]} for r in rows],
            "available": bool(refs), "note": None if refs else "no keyframes stored for this subject"}


def get_script(engine: Engine, episode_id: str, max_events: int = 400, cache: bool = True) -> str:
    """The scene script: what the reasoning model reads instead of video. Deterministic, compact,
    every line carries the ids it can cite. Cached in the scripts table."""
    with engine.connect() as conn:
        if cache:
            cached = conn.execute(select(scripts.c.text).where(scripts.c.episode_id == episode_id)).first()
            if cached:
                return cached[0]
        ep = conn.execute(select(episodes).where(episodes.c.episode_id == episode_id)).first()
        if ep is None:
            raise KeyError(episode_id)
        ep = dict(ep._mapping)
        tube_rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.episode_id == episode_id).order_by(tubes.c.born_ms))]
        ev_rows = [dict(r._mapping) for r in conn.execute(select(events).where(events.c.episode_id == episode_id).order_by(events.c.t_ms).limit(max_events))]
    t0 = ep["t0_ms"]
    tube_ent = {r["tube_id"]: r["entity_id"] for r in tube_rows}
    by_entity: dict[str, list[dict]] = {}
    for r in tube_rows:
        by_entity.setdefault(r["entity_id"], []).append(r)
    def ent_line(eid: str, rows: list[dict]) -> str:
        first, last = min(r["born_ms"] for r in rows), max(r["last_seen_ms"] for r in rows)
        name = rows[0]["named"] or ("anonymous" if eid.startswith("anon:") else "unnamed")
        kf = next((k for r in rows for k in (r["keyframe_refs"] or [])), None)
        attrs = next((r["attributes"] for r in rows if r.get("attributes")), None)
        desc = ""
        if attrs:
            bits = [attrs.get("description") or "", f"top {attrs['top_color']}" if attrs.get("top_color") else "",
                    f"carrying {attrs['carried_item']}" if attrs.get("carried_item") else ""]
            desc = "  looks: " + "; ".join(b for b in bits if b)
        return (f"  {eid}  {rows[0]['class_label']}  {name}  tubes {','.join(r['tube_id'] for r in rows)}  "
                f"seen {_ts(first, t0)}–{_ts(last, t0)}  coverage {cov(rows):.0%}  states {','.join(sorted({r['state'] for r in rows}))}"
                + (f"  keyframe {kf}" if kf else "") + desc)

    span = max(1, (ep["t1_ms"] or t0) - t0)
    def cov(rows: list[dict]) -> float:
        ivs = sorted((max(t0, r["born_ms"]), min(ep["t1_ms"] or t0, r["last_seen_ms"])) for r in rows)
        total, cur = 0, None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: total += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur: total += cur[1] - cur[0]
        return total / span
    confirmed = {e: rows for e, rows in by_entity.items() if any(r.get("quality", "ok") == "ok" for r in rows)}
    brief = {e: rows for e, rows in by_entity.items() if e not in confirmed}
    people = sum(1 for rows in confirmed.values() if rows[0]["class_label"] == "person")
    lines = [f"EPISODE {ep['episode_id']} | tile {ep['tile_id']} | cameras {','.join(ep['camera_ids'] or [])} | "
             f"{_ts(t0, t0)}–{_ts(ep['t1_ms'] or t0, t0)} | status {ep['status']} | kb v{ep['kb_version']}",
             f"CAST: {len(confirmed)} confirmed entities ({people} people), {len(brief)} brief sightings, {len(tube_rows)} tubes"]
    for eid, rows in confirmed.items():
        lines.append(ent_line(eid, rows))
    if brief:
        lines.append("BRIEF SIGHTINGS (low quality: too short, too small or at the frame border; not counted as people):")
        for eid, rows in brief.items():
            reason = next((r.get("quality_reason") for r in rows if r.get("quality_reason")), "")
            lines.append(ent_line(eid, rows) + (f"  why: {reason}" if reason else ""))
    lines.append(f"TIMELINE: {len(ev_rows)} events")
    for e in ev_rows:
        subj = e["subject_tube_ids"] or []
        who = ", ".join(f"{tube_ent.get(t, 'anon:' + t)} (tube {t})" for t in subj) or "—"
        extra = ""
        if e["type"] == "relink" and e["payload"]:
            extra = f" sim {e['payload'].get('similarity')}"
        elif e["type"] == "dwell" and e["payload"]:
            extra = f" {e['payload'].get('dwell_ms', 0) / 1000:.1f}s"
        elif e["type"] in ("pickup", "asset_missing_from_home", "drop") and e["object_ids"]:
            extra = f" object {','.join(e['object_ids'])}"
        lines.append(f"  {_ts(e['t_ms'], t0)}  {e['type']:<22s} zone {e['zone_id'] or '-':<14s} {who}{extra}  [{e['event_id']}]")
    text = "\n".join(lines)
    if cache:
        with engine.begin() as conn:
            insert_ignore(conn, scripts, [dict(episode_id=episode_id, text=text, rendered_at_ms=int(time.time() * 1000))])
    return text


# ---------------------------------------------------------------- multi-episode (window) tools
def footage_bounds(engine: Engine) -> tuple[int, int] | None:
    with engine.connect() as conn:
        row = conn.execute(select(func.min(episodes.c.t0_ms), func.max(func.coalesce(episodes.c.t1_ms, episodes.c.t0_ms)))).first()
    return (int(row[0]), int(row[1])) if row and row[0] is not None else None


def episodes_in(engine: Engine, t_start_ms: int, t_end_ms: int) -> list[dict]:
    with engine.connect() as conn:
        q = select(episodes).where(and_(episodes.c.t0_ms <= t_end_ms, func.coalesce(episodes.c.t1_ms, episodes.c.t0_ms) >= t_start_ms)).order_by(episodes.c.t0_ms)
        return [dict(r._mapping) for r in conn.execute(q)]


def cameras_in_store(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        return sorted({r[0] for r in conn.execute(select(tubes.c.camera_id).distinct())})


def coverage_window(engine: Engine, t_start_ms: int, t_end_ms: int, class_label: str = "person", confirmed_only: bool = True,
                    camera_id: str | None = None) -> list[dict]:
    """Per entity across episodes: seen interval clipped to the window and coverage of the window."""
    span = max(1, t_end_ms - t_start_ms)
    conds = [tubes.c.class_label == class_label, tubes.c.last_seen_ms >= t_start_ms, tubes.c.born_ms <= t_end_ms]
    if camera_id:
        conds.append(tubes.c.camera_id == camera_id)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(and_(*conds)))]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["entity_id"], []).append(r)
    out = []
    for eid, rs in by.items():
        ok = any(r.get("quality", "ok") == "ok" for r in rs)
        if confirmed_only and not ok:
            continue
        ivs = sorted((max(t_start_ms, r["born_ms"]), min(t_end_ms, r["last_seen_ms"])) for r in rs)
        covered, cur = 0, None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: covered += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur: covered += cur[1] - cur[0]
        attrs = next((r["attributes"] for r in rs if r.get("attributes")), None)
        emb = next((r["embedding"] for r in rs if r.get("embedding")), None)
        out.append({"entity_id": eid, "embedding": emb, "first_seen_ms": max(t_start_ms, min(r["born_ms"] for r in rs)),
                    "last_seen_ms": min(t_end_ms, max(r["last_seen_ms"] for r in rs)), "coverage": round(covered / span, 3),
                    "tubes": len(rs), "quality": "ok" if ok else "low", "cameras": sorted({r["camera_id"] for r in rs}),
                    "looks": (attrs or {}).get("description") if attrs else None,
                    "keyframe": next((k for r in rs for k in (r["keyframe_refs"] or [])), None)})
    return sorted(out, key=lambda x: (-x["coverage"], x["first_seen_ms"]))


def world_groups(rows: list[dict], sim_thr: float = 0.85, max_gap_ms: int = 60_000) -> dict[str, str]:
    """Cross-camera fusion at query time (R14): entities on DIFFERENT cameras whose appearance
    embeddings agree and whose times overlap (or nearly) are one world person. Never joins two
    entities of the same camera: that camera's linker already decided they are different people.
    Returns entity_id -> world id (W1, W2, ...), ordered by first appearance."""
    import numpy as np
    ids = [r["entity_id"] for r in rows]
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    embs = {r["entity_id"]: (np.asarray(r["embedding"], np.float32) if r.get("embedding") else None) for r in rows}
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if set(a.get("cameras", [])) & set(b.get("cameras", [])):
                continue                                             # same camera: already distinct
            ea, eb = embs[a["entity_id"]], embs[b["entity_id"]]
            if ea is None or eb is None or ea.shape != eb.shape:
                continue
            gap = max(a["first_seen_ms"], b["first_seen_ms"]) - min(a["last_seen_ms"], b["last_seen_ms"])
            if gap > max_gap_ms:
                continue
            sim = float(ea @ eb) / (float(np.linalg.norm(ea)) * float(np.linalg.norm(eb)) + 1e-8)
            if sim >= sim_thr:
                parent[find(a["entity_id"])] = find(b["entity_id"])
    order: dict[str, str] = {}
    for r in sorted(rows, key=lambda x: x["first_seen_ms"]):
        root = find(r["entity_id"])
        if root not in order:
            order[root] = f"W{len(order) + 1}"
    return {r["entity_id"]: order[find(r["entity_id"])] for r in rows}


def count_entities_window(engine: Engine, t_start_ms: int, t_end_ms: int, class_label: str = "person", camera_id: str | None = None) -> dict:
    rows = coverage_window(engine, t_start_ms, t_end_ms, class_label, camera_id=camera_id)
    worlds = world_groups(rows)
    groups: dict[str, list[str]] = {}
    for eid, w in worlds.items():
        groups.setdefault(w, []).append(eid)
    return {"count": len(groups), "people": len(groups), "camera_entities": len(rows),
            "world_groups": groups, "entity_ids": [r["entity_id"] for r in rows], "window_ms": [t_start_ms, t_end_ms], "camera_id": camera_id,
            "note": "count = people after joining the same person across cameras; camera_entities = per-camera tracks"}


def entities_present_window(engine: Engine, t_start_ms: int, t_end_ms: int, min_coverage: float = 0.9, class_label: str = "person",
                            camera_id: str | None = None) -> dict:
    rows = coverage_window(engine, t_start_ms, t_end_ms, class_label, camera_id=camera_id)
    ids = [r["entity_id"] for r in rows if r["coverage"] >= min_coverage]
    return {"min_coverage": min_coverage, "count": len(ids), "entity_ids": ids, "coverage": {r["entity_id"]: r["coverage"] for r in rows}}


def window_script(engine: Engine, t_start_ms: int, t_end_ms: int, tz_name: str = "UTC", max_chars: int = 7000,
                  max_events: int = 120, camera_id: str | None = None) -> str:
    """Compact script for a time window across episodes: absolute clock times, confirmed cast with
    coverage and looks, then events inside the window (capped; counts by type when over the cap)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    def clock(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%H:%M:%S")
    eps = episodes_in(engine, t_start_ms, t_end_ms)
    if camera_id:
        eps = [e for e in eps if camera_id in (e["camera_ids"] or [])]
    all_cams = cameras_in_store(engine)
    cast = coverage_window(engine, t_start_ms, t_end_ms, camera_id=camera_id)
    worlds = world_groups(cast)
    n_people = len(set(worlds.values()))
    lines = [f"WINDOW {clock(t_start_ms)}–{clock(t_end_ms)} ({(t_end_ms - t_start_ms) / 60000:.1f} min) | episodes {len(eps)} | "
             + (f"camera {camera_id}" if camera_id else f"cameras {','.join(all_cams)}"),
             f"CAST: {n_people} people" + (f" ({len(cast)} camera tracks; W-ids join the same person seen on several cameras)" if len(all_cams) > 1 else "")]
    by_world: dict[str, list[dict]] = {}
    for c in cast:
        by_world.setdefault(worlds[c["entity_id"]], []).append(c)
    for w, members in by_world.items():
        if len(all_cams) > 1:
            lines.append(f"  {w}: " + " + ".join(f"{m['entity_id']} on {','.join(m['cameras'])}" for m in members))
        for c in members:
            lines.append(f"    {c['entity_id']}  seen {clock(c['first_seen_ms'])}–{clock(c['last_seen_ms'])}  coverage {c['coverage']:.0%}"
                         + (f"  looks: {c['looks']}" if c.get("looks") else "") + (f"  keyframe {c['keyframe']}" if c.get("keyframe") else ""))
    ev_conds = [events.c.t_ms >= t_start_ms, events.c.t_ms <= t_end_ms]
    if camera_id:
        ev_conds.append(events.c.camera_id == camera_id)
    with engine.connect() as conn:
        evs = [dict(r._mapping) for r in conn.execute(select(events).where(and_(*ev_conds)).order_by(events.c.t_ms))]
    tube_ent: dict[str, str] = {}
    with engine.connect() as conn:
        for r in conn.execute(select(tubes.c.tube_id, tubes.c.entity_id).where(and_(tubes.c.last_seen_ms >= t_start_ms, tubes.c.born_ms <= t_end_ms))):
            tube_ent[r[0]] = r[1]
    if len(evs) > max_events:
        from collections import Counter
        c = Counter(e["type"] for e in evs)
        lines.append(f"TIMELINE: {len(evs)} events (showing the {max_events} most recent; totals by type: " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + ")")
        evs = evs[-max_events:]
    else:
        lines.append(f"TIMELINE: {len(evs)} events")
    for e in evs:
        who = ", ".join(f"{tube_ent.get(t, 'anon:' + t)}" for t in (e["subject_tube_ids"] or [])) or "—"
        cam = f" {e['camera_id']:<6s}" if not camera_id and len(all_cams) > 1 else ""
        lines.append(f"  {clock(e['t_ms'])}{cam}  {e['type']:<20s} zone {e['zone_id'] or '-':<14s} {who}  [{e['event_id']}]")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n  ... (truncated)"
EOF_VI
cat > vi/agent/loop.py << 'EOF_VI'
"""Block 2 agent loop (R26/R27): a reasoning model that never sees video, only tool results, and
must cite ids for every claim. Every model turn is one JSON `AgentStep` under a schema, so it is
grammar-constrainable (R3) on any backend that supports structured outputs.

Backends implement `complete(messages, schema) -> str`. `FakeBackend` is a deterministic
scripted planner for tests and CPU runs; `OpenAIBackend` talks to any OpenAI-compatible server
(vLLM serve, SGLang) and asks for JSON-schema structured output.
"""
from __future__ import annotations

import json
import re
import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy.engine import Engine

from vi.schemas import EventType

from . import tools as T

EVENT_TYPES = [e.value for e in EventType]

TOOL_SPECS = {
    "search_events": "Events by time window / zone / type / entity. args: tile_id, camera_id, t_start_ms, t_end_ms, "
                     "event_type (one of EVENT_TYPES, or a list), zone_id, entity_id, tube_id, limit",
    "search_tubes": "Tubes (per-camera tracks). args: tile_id, camera_id, class_label, t_start_ms, t_end_ms, entity_id, named, min_life_ms, limit",
    "search_entities": "Entities (people/objects across tubes). args: camera_id, class_label, t_start_ms, t_end_ms, named, limit",
    "get_script": "The scene script of an episode (cast + timeline). args: episode_id",
    "clip": "Evidence references (keyframes, time segments) for an entity or tube. args: entity_id or tube_id",
    "count_entities": "Distinct confirmed people in a window. args: t_start_ms, t_end_ms (omit for whole episode)",
    "entities_present": "Entities seen for at least min_coverage of the episode ('stayed the whole time'). args: min_coverage (default 0.9)",
    "coverage": "Per-entity seen interval and coverage fraction. args: none",
}


class ToolStep(BaseModel):
    action: Literal["tool"] = "tool"
    tool: Literal["search_events", "search_tubes", "search_entities", "get_script", "clip", "count_entities", "entities_present", "coverage"]
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = Field("", max_length=400)


ANSWER_MAX_CHARS = 4000


class AnswerStep(BaseModel):
    action: Literal["answer"] = "answer"
    text: str = Field(max_length=ANSWER_MAX_CHARS)
    citations: list[str] = Field(default_factory=list, description="entity ids, tube ids or event ids from tool results")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ClarifyStep(BaseModel):
    action: Literal["clarify"] = "clarify"
    question: str = Field(max_length=300)


AgentStep = Annotated[Union[ToolStep, AnswerStep, ClarifyStep], Field(discriminator="action")]
step_adapter: TypeAdapter = TypeAdapter(AgentStep)
STEP_SCHEMA = step_adapter.json_schema()

SYSTEM = """You answer questions about video by calling tools over an evidence store; you never see video.
Rules: (1) the episode's scene script is given to you; call tools only when it does not answer the question;
(2) every claim in an answer must cite ids
(entity ids like cam1:E3, tube ids like cam1:4000:10, event ids like ev_...) that appeared in tool results;
(3) if nothing matches, say so and suggest how to widen the search; never invent people, times or events;
(4) if the question is ambiguous (which person, which time), ask one clarifying question;
(5) answer over entities, not tubes; a person may have several tubes; (6) times are episode-relative
mm:ss.s in scripts and absolute milliseconds in tool args; (7) any count, duration or "whole time" claim MUST come from
count_entities / entities_present / coverage, never from reading the script. Respond with exactly one JSON object per turn:
{"action":"tool","tool":...,"args":{...},"why":...} | {"action":"answer","text":...,"citations":[...],"confidence":...}
| {"action":"clarify","question":...}. Count people from CONFIRMED entities; BRIEF SIGHTINGS are not people.
Keep answers under 60 words and cite at most 8 ids; cite entity ids (cam1:E7), not tube ids, unless asked about tubes; mention only events
that appear in the script or tool results.
EVENT_TYPES: """ + ", ".join(EVENT_TYPES) + "\nTools: " + json.dumps(TOOL_SPECS)

ID_RE = re.compile(r"\b(ev_[0-9a-f]{16}|[A-Za-z0-9_]+:E\d+|anon:[A-Za-z0-9_:]+|[A-Za-z0-9_]+:\d+:\d+|W\d{1,3})\b")


class FakeBackend:
    """Scripted planner: script -> one search chosen from the question -> cited answer. Enough to
    exercise the loop, the citation check and the empty-result path without a model."""

    name = "fake"

    def __init__(self, bogus_citation: bool = False):
        self.bogus = bogus_citation

    def complete(self, messages: list[dict], schema: dict) -> str:
        user = [m for m in messages if m["role"] == "user"]
        q = user[0]["content"].split("Question:")[-1].lower()     # the question, not the inlined script
        n_tool_results = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("TOOL RESULT"))
        has_script = "SCENE SCRIPT:" in user[0]["content"]
        if n_tool_results == 0 and not has_script:
            ep = re.search(r"episode (ep_[0-9a-f]+)", user[0]["content"], re.IGNORECASE)
            return json.dumps({"action": "tool", "tool": "get_script", "args": {"episode_id": ep.group(1) if ep else ""}, "why": "read the script"})
        if n_tool_results == (0 if has_script else 1):
            if "key" in q or "pickup" in q or "took" in q:
                return json.dumps({"action": "tool", "tool": "search_events", "args": {"event_type": "pickup"}, "why": "custody"})
            if "nobody" in q or "unicorn" in q:
                return json.dumps({"action": "tool", "tool": "search_events", "args": {"event_type": "fall"}, "why": "probe"})
            return json.dumps({"action": "tool", "tool": "search_entities", "args": {"class_label": "person"}, "why": "who"})
        last = messages[-1]["content"]
        ids = ID_RE.findall(last)
        if not ids:
            return json.dumps({"action": "answer", "text": "No matching events in this episode; widen the time window or check another tile.",
                               "citations": [], "confidence": 0.9})
        cites = ["ev_deadbeefdeadbeef"] if self.bogus else sorted(set(ids))[:4]
        return json.dumps({"action": "answer", "text": f"Found {len(set(ids))} matching record(s); see citations.",
                           "citations": cites, "confidence": 0.8})


class OpenAIBackend:
    """Any OpenAI-compatible chat server (vLLM serve, SGLang). Requests JSON-schema structured
    output; falls back to vLLM's guided_json extra field on servers that predate response_format."""

    name = "openai"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen/Qwen3.5-4B",
                 api_key: str = "EMPTY", temperature: float = 0.0, max_tokens: int = 600, timeout: float = 120.0):
        import urllib.request
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self._req = urllib.request

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = self._req.Request(self.base_url + "/chat/completions", data=data, method="POST",
                                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with self._req.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def complete(self, messages: list[dict], schema: dict) -> str:
        base = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}      # Qwen3.x: no reasoning preamble before the JSON
        try:
            out = self._post({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema}}})
        except Exception:
            try:
                out = self._post({**base, "guided_json": schema})
            except Exception:
                base.pop("chat_template_kwargs", None)                   # servers that reject the field
                out = self._post({**base, "guided_json": schema})
        return extract_json(out["choices"][0]["message"]["content"])


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> str:
    """Take the first balanced {...} object out of a model reply (think blocks, code fences,
    prose, and trailing text are all tolerated). Returns the raw text if none is found, so
    validation fails loudly rather than silently."""
    t = THINK_RE.sub("", text).strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start = t.find("{")
    if start < 0:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
            continue
        if c == '"': in_str = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return text


class TransformersBackend:
    """In-process fallback when no server can be started: loads the model with transformers in the
    runtime's own torch. No structured-output grammar; the loop validates and retries instead."""

    name = "transformers"

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 300, device: str | None = None, warmup: bool = True):
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        last = None
        self.model = None
        try:
            import fla  # noqa: F401  (flash-linear-attention: fused kernels for Qwen3.5's GDN layers)
            self.fused = True
        except Exception:
            self.fused = False
        for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            try:
                cls = getattr(__import__("transformers", fromlist=[loader]), loader)
                try:
                    self.model = cls.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
                except Exception:
                    self.model = cls.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
                self.loader = loader
                break
            except Exception as e:  # pragma: no cover
                last = e
        self.last_gen_tokens = 0
        self.last_tok_s = 0.0
        if self.model is None:
            raise RuntimeError(f"could not load {model_id}: {last!r}")
        try:
            self.tok = AutoProcessor.from_pretrained(model_id)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        if warmup:   # CUDA context, kernel compilation (fused linear attention compiles per shape) and cache
            try:     # allocation happen here at a realistic prompt length, not inside the first question
                filler = "SCENE SCRIPT: " + ("cam1:E1 person unnamed seen 00:00.0–00:16.7 coverage 99%; " * 80)
                self.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": filler + "\nQuestion: how many people? Reply with {}"}], {})
            except Exception:
                pass

    def complete(self, messages: list[dict], schema: dict) -> str:
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\nReply with the JSON object only, no prose."}
        try:   # Qwen3.x templates: thinking is on by default and eats the whole token budget
            inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_tensors="pt", return_dict=True, enable_thinking=False)
        except TypeError:
            inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        dt = max(1e-6, time.perf_counter() - t0)
        self.last_gen_tokens = int(getattr(gen, "shape", [len(gen)])[0])
        self.last_tok_s = self.last_gen_tokens / dt
        tokenizer = getattr(self.tok, "tokenizer", self.tok)
        return extract_json(tokenizer.decode(gen, skip_special_tokens=True))


def _flatten_ids(x: Any) -> list[str]:
    """citations may arrive as strings, ints, dicts ({"entity_id": ...}) or nested lists"""
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return ID_RE.findall(str(x)) or ([str(x)] if isinstance(x, str) else [])
    if isinstance(x, dict):
        return [i for v in x.values() for i in _flatten_ids(v)]
    if isinstance(x, list):
        return [i for v in x for i in _flatten_ids(v)]
    return []


EVENT_WORDS = {"pickup": ["pickup", "pick-up", "picked up", "picking up", "picks up"], "drop": ["drop event", "dropped"],
               "fall": ["fall event", "fell", "falling"], "left_behind": ["left behind"], "loiter": ["loiter"],
               "crowd": ["crowd event"], "run": ["running event"], "handoff": ["handoff", "hand-off"],
               "impossible_transition": ["impossible transition"], "relink": ["relink"]}


def contradicted_event_claims(engine: Engine, episode_id: str, text: str) -> list[str]:
    """E-AGT-06 for prose: an answer that talks about an event type the episode does not contain
    is asserting evidence that does not exist. Returns the offending event types."""
    # sentence-level, ignoring negated mentions ("nobody fell", "no pickup events") which assert absence
    neg = re.compile(r"\b(no|not|nobody|none|never|didn't|did not|wasn't|weren't|isn't|aren't|without)\b")
    named: set[str] = set()
    for sent in re.split(r"(?<=[.!?;])\s+", text.lower()):
        if neg.search(sent):
            continue
        for et, words in EVENT_WORDS.items():
            if any(w in sent for w in words):
                named.add(et)
    if not named:
        return []
    have = {e["type"] for e in T.search_events(engine, limit=10000) if e["episode_id"] == episode_id}
    return sorted(et for et in named if et not in have)


WHOLE_TIME_RE = re.compile(r"whole (time|episode|clip)|entire (time|episode|clip)|throughout", re.I)
COUNT_Q_RE = re.compile(r"how many|number of (people|persons|workers)|count", re.I)
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}


def _numbers_in(text: str) -> list[int]:
    t = ID_RE.sub(" ", text)
    t = re.sub(r"\d{2}:\d{2}(\.\d)?", " ", t)                        # timestamps are not counts
    nums = [int(m) for m in re.findall(r"(?<![\d.:])(\d{1,3})(?![\d.:])", t)]
    nums += [v for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\b", t, re.I)]
    return nums


def numeric_claim_issue(engine: Engine, episode_id: str, question: str, step: AnswerStep) -> dict | None:
    """E-AGT-06 for numbers: a people count or a 'stayed the whole time' list in the answer is
    compared with the deterministic tools; a disagreement is sent back once with the true values."""
    try:
        if COUNT_Q_RE.search(question):
            truth = T.count_entities(engine, episode_id)["count"]
            nums = _numbers_in(step.text)
            if nums and truth not in nums and all(abs(n - truth) > 1 for n in nums[:3]):
                return {"msg": f"count_entities says {truth} confirmed people in this episode (your answer implies {nums[:3]})."}
        if WHOLE_TIME_RE.search(question):
            ep = T.entities_present(engine, episode_id, 0.9)
            truth_ids = set(ep["entity_ids"])
            cited = {c for c in step.citations if re.search(r":E\d+$", c)}
            if truth_ids and len(cited & truth_ids) < max(1, len(truth_ids) - 1):
                return {"msg": f"entities_present(min_coverage=0.9) says {len(truth_ids)} entities stayed the whole time: "
                               f"{sorted(truth_ids)}; your answer names {sorted(cited & truth_ids) or 'none of them'}."}
    except Exception:
        return None
    return None


def _salvage(raw: str) -> AnswerStep | ClarifyStep | None:
    """Accept an answer that only failed on length or a missing optional field; never a tool
    call, which must be exact. Retrying seven times to trim a paragraph is not a behaviour."""
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("action") == "answer" and isinstance(obj.get("text"), str):
        cites = _flatten_ids(obj.get("citations")) + ID_RE.findall(obj["text"])   # objects, nested lists, ids in prose
        seen: list[str] = []
        for c in cites:
            if c not in seen:
                seen.append(c)
        return AnswerStep(text=obj["text"][:ANSWER_MAX_CHARS], citations=seen[:50],
                          confidence=float(obj.get("confidence", 0.5)) if isinstance(obj.get("confidence"), (int, float)) else 0.5)
    if obj.get("action") == "clarify" and isinstance(obj.get("question"), str):
        return ClarifyStep(question=obj["question"][:300])
    return None


EPISODE_TOOLS = {"count_entities", "entities_present", "coverage"}


def run_tool(engine: Engine, step: ToolStep, episode_id: str | None = None) -> Any:
    fn = {"search_events": T.search_events, "search_tubes": T.search_tubes, "search_entities": T.search_entities,
          "get_script": T.get_script, "clip": T.clip, "count_entities": T.count_entities,
          "entities_present": T.entities_present, "coverage": T.coverage}[step.tool]
    import inspect
    allowed = set(inspect.signature(fn).parameters) - {"engine", "episode_id"} if step.tool != "get_script" else {"episode_id"}
    dropped = [k for k in step.args if k not in allowed]
    args = {k: v for k, v in step.args.items() if v is not None and k in allowed}
    if step.tool == "search_events" and args.get("event_type"):
        types = args["event_type"] if isinstance(args["event_type"], list) else [args["event_type"]]
        bad = [t for t in types if t not in EVENT_TYPES]
        if bad:
            raise ValueError(f"unknown event_type {bad}; valid: {', '.join(EVENT_TYPES)}")
    if step.tool == "get_script":
        return fn(engine, args.get("episode_id", ""))
    if step.tool in EPISODE_TOOLS:
        result = fn(engine, episode_id or step.args.get("episode_id", ""), **args)
        if dropped:
            result = {**result, "note": f"ignored unknown args {dropped}"}
        return result
    result = fn(engine, **args)
    if dropped and isinstance(result, list):
        result = [{"note": f"ignored unknown args {dropped}"}] + result if result else [{"note": f"ignored unknown args {dropped}; no matches"}]
    return result


def _compact(result: Any, limit: int = 6000) -> str:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def ask(engine: Engine, question: str, episode_id: str, backend, max_steps: int = 6,
        inline_script: bool = True) -> dict:
    """Run the loop. With inline_script the scene script is placed in the first user message so
    the model's first turn is already a search or an answer (one round trip saved) and the
    server's prefix cache holds system+script across questions on the same episode. Returns the
    final step, the trace, the citation audit (E-AGT-06) and latency."""
    t_start = time.perf_counter()
    seen_ids: set[str] = set()
    first = f"Episode {episode_id}."
    if inline_script:
        script = T.get_script(engine, episode_id)
        seen_ids.update(ID_RE.findall(script))
        first += f"\n\nSCENE SCRIPT:\n{script}"
    first += f"\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    trace: list[dict] = []
    final: dict | None = None
    model_ms, tool_ms = 0.0, 0.0
    for i in range(max_steps):
        t0 = time.perf_counter()
        raw = backend.complete(messages, STEP_SCHEMA)
        model_ms += (time.perf_counter() - t0) * 1000
        try:
            step = step_adapter.validate_json(raw)
        except ValidationError as e:
            step = _salvage(raw)                    # an over-long or slightly malformed answer is still an answer
            if step is None:
                reason = e.errors()[0]["msg"]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"TOOL RESULT: invalid step ({reason}); reply with one valid JSON object."})
                trace.append({"step": i, "invalid": raw[:200], "reason": reason})
                continue
            trace.append({"step": i, "salvaged": e.errors()[0]["msg"]})
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            result: Any = None
            t0 = time.perf_counter()
            try:
                result = run_tool(engine, step, episode_id)
                text = _compact(result)
                err = None
            except Exception as e:
                text, err = f"error: {type(e).__name__}: {e}", str(e)
            tool_ms += (time.perf_counter() - t0) * 1000
            seen_ids.update(ID_RE.findall(text))
            n = len(result) if isinstance(result, list) else (1 if result else 0)
            trace.append({"step": i, "tool": step.tool, "args": step.args, "results": n, "error": err})
            messages.append({"role": "user", "content": f"TOOL RESULT ({step.tool}, {n} item(s)):\n{text}"})
            continue
        if isinstance(step, ClarifyStep):
            final = {"action": "clarify", "question": step.question}
            break
        valid = [c for c in step.citations if c in seen_ids]
        invalid = [c for c in step.citations if c not in seen_ids]
        already_revised = any("revise" in t for t in trace)
        bad_claims = contradicted_event_claims(engine, episode_id, step.text)
        num_issue = numeric_claim_issue(engine, episode_id, question, step)
        if num_issue and not already_revised:
            trace.append({"step": i, "revise": "numeric claim disagrees with tools", "detail": num_issue["msg"]})
            messages.append({"role": "user", "content": "TOOL RESULT: " + num_issue["msg"] + " Answer again using these numbers."})
            continue
        if bad_claims and not already_revised:
            trace.append({"step": i, "revise": "event claim without evidence", "claims": bad_claims})
            messages.append({"role": "user", "content": f"TOOL RESULT: your answer mentions {bad_claims} but this episode contains no "
                                                        f"events of that type. Remove or correct that claim; cite only events that exist."})
            continue
        if invalid and not valid and not already_revised:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in any tool result. "
                                                        "Answer again citing only ids you were shown, or say nothing matched."})
            continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid,
                 "confidence": step.confidence, "cited": bool(valid), "unsupported_event_claims": bad_claims,
                 "numeric_issue": num_issue["msg"] if num_issue else None}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False,
                 "confidence": 0.0}
    total_ms = (time.perf_counter() - t_start) * 1000
    return {"question": question, "episode_id": episode_id, "backend": getattr(backend, "name", "?"),
            "steps": len(trace), "trace": trace, "final": final,
            "latency": {"total_ms": round(total_ms), "model_ms": round(model_ms), "tool_ms": round(tool_ms),
                        "turns": sum(1 for t in trace if "tool" in t or "revise" in t or "invalid" in t) + 1,
                        "first_prompt_chars": len(first),
                        "gen_tokens": getattr(backend, "last_gen_tokens", None), "tok_s": round(getattr(backend, "last_tok_s", 0.0), 1) or None,
                        "fused_kernels": getattr(backend, "fused", None)}}


# ---------------------------------------------------------------- multi-episode, live-footage entry point
WINDOW_TOOLS = {"count_entities", "entities_present", "coverage"}

SYSTEM_LIVE_SUFFIX = """
You are answering about recorded footage. FOOTAGE: {start}–{end} ({tz}); the latest processed moment is {now}.
The scene script below covers only the window {ws}–{we}. If the question needs a different time, say which
window you are answering about. W-ids (W1, W2…) are people; a person seen on several cameras has one W-id and several
camera tracks (cam03:E1 …). Count people by W-ids, never by camera tracks. If nothing in the window matches, say so plainly. Never guess names: people are
unnamed unless the cast says otherwise; describe them from their `looks` instead. Counts carry an uncertainty of
about ±1 person; say "about" for counts above 3. Do not speculate about what happened outside the footage.
"""


CAM_RE = re.compile(r"\b(?:cam(?:era)?)\s*-?\s*(\d{1,2})\b", re.I)


def ground_camera(question: str, cameras: list[str]) -> str | None:
    """'on cam 4' / 'camera 04' / 'CAM 12' -> the store's camera id with that number, if any."""
    m = CAM_RE.search(question)
    if not m or not cameras:
        return None
    n = int(m.group(1))
    for c in cameras:
        digits = re.sub(r"\D", "", c)
        if digits and int(digits) == n:
            return c
    return None


def _window_tool(engine: Engine, step: ToolStep, ws: int, we: int, camera_id: str | None = None) -> Any:
    args = {k: v for k, v in step.args.items() if v is not None}
    a = int(args.get("t_start_ms", ws)); b = int(args.get("t_end_ms", we))
    a, b = max(ws, min(a, b)), min(we, max(a, b))
    if step.tool == "count_entities":
        return T.count_entities_window(engine, a, b, camera_id=camera_id)
    if step.tool == "entities_present":
        return T.entities_present_window(engine, a, b, float(args.get("min_coverage", 0.9)), camera_id=camera_id)
    if step.tool == "coverage":
        return T.coverage_window(engine, a, b, camera_id=camera_id)
    return run_tool(engine, step, None)


AFFIRM_RE = re.compile(r"^\s*(?:(?:yes|yeah|yep|yup|exactly|correct|right|that one|that's right|ok|okay|sure|go ahead|please|do it|the first|the second|the latter|the former)[\s,!.]*)+$", re.I)


def resolve_followup(question: str, history: list[dict] | None) -> tuple[str, str | None]:
    """history: [{"q": ..., "a": ..., "action": "answer"|"clarify"}, ...] (most recent last).
    A bare confirmation after a clarification re-asks the previous question with the assistant's
    proposed reading; any other question carries the previous exchange as context."""
    if not history:
        return question, None
    last = history[-1]
    if AFFIRM_RE.match(question) and last.get("action") == "clarify":
        merged = f"{last['q']} (the user confirmed: {last['a']})"
        return merged, f"Previous question: {last['q']}\nYou asked: {last['a']}\nUser: {question}"
    ctx = f"Previous question: {last['q']}\nPrevious answer: {last['a'][:300]}"
    return question, ctx


def ask_window(engine: Engine, question: str, backend, now_ms: int, tz_name: str = "UTC", max_steps: int = 6,
               history: list[dict] | None = None) -> dict:
    """Live-footage question answering: scope check -> time grounding -> window script -> model loop
    with window-aware numeric tools. Refusals and clarifications happen before any model call."""
    t_start_total = time.perf_counter()
    question, context = resolve_followup(question, history)
    bounds = T.footage_bounds(engine)
    if bounds is None:
        return {"question": question, "final": {"action": "answer", "text": "No footage has been processed yet.", "citations": [], "cited": False},
                "trace": [], "steps": 0, "latency": {"total_ms": 0}}
    start_ms, end_ms = bounds
    end_ms = max(end_ms, now_ms if now_ms else end_ms)
    from .scope import classify
    from .timeground import fmt, ground
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    kind, msg = classify(question)
    if kind != "ok":
        return {"question": question, "grounding": kind, "steps": 0, "trace": [],
                "final": {"action": "answer", "text": msg, "citations": [], "cited": True, "handled_by": "scope"},
                "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000)}}
    g = ground(question, now_ms or end_ms, start_ms, end_ms, tz_name)
    if g.kind in ("future", "before_start"):
        return {"question": question, "grounding": g.kind, "steps": 0, "trace": [],
                "final": {"action": "answer", "text": g.message + f" I can answer about {fmt(start_ms, tz)}–{fmt(end_ms, tz)}.",
                          "citations": [], "cited": True, "handled_by": "time"},
                "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000)}}
    ws, we = (g.t_start_ms or start_ms), (g.t_end_ms or end_ms)
    cameras = T.cameras_in_store(engine)
    cam = ground_camera(question, cameras)
    if CAM_RE.search(question) and cam is None and cameras:
        return {"question": question, "grounding": "unknown_camera", "steps": 0, "trace": [],
                "final": {"action": "answer", "text": f"There is no camera with that number. Cameras in the footage: {', '.join(cameras)}.",
                          "citations": [], "cited": True, "handled_by": "scope"},
                "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000)}}
    script = T.window_script(engine, ws, we, tz_name, camera_id=cam)
    seen_ids: set[str] = set(ID_RE.findall(script))
    system = SYSTEM + SYSTEM_LIVE_SUFFIX.format(start=fmt(start_ms, tz), end=fmt(end_ms, tz), tz=tz_name,
                                                 now=fmt(now_ms or end_ms, tz), ws=fmt(ws, tz), we=fmt(we, tz))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"SCENE SCRIPT:\n{script}\n\n" + (f"CONTEXT:\n{context}\n\n" if context else "") + f"Question: {question}"}]
    trace: list[dict] = []
    final: dict | None = None
    model_ms = tool_ms = 0.0
    for i in range(max_steps):
        t0 = time.perf_counter()
        raw = backend.complete(messages, STEP_SCHEMA)
        model_ms += (time.perf_counter() - t0) * 1000
        try:
            step = step_adapter.validate_json(raw)
        except ValidationError as e:
            step = _salvage(raw)
            if step is None:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"TOOL RESULT: invalid step ({e.errors()[0]['msg']}); reply with one valid JSON object."}]
                trace.append({"step": i, "invalid": raw[:200], "reason": e.errors()[0]["msg"]}); continue
            trace.append({"step": i, "salvaged": e.errors()[0]["msg"]})
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            t0 = time.perf_counter()
            try:
                result = _window_tool(engine, step, ws, we, cam); text = _compact(result); err = None
            except Exception as e:
                result, text, err = None, f"error: {type(e).__name__}: {e}", str(e)
            tool_ms += (time.perf_counter() - t0) * 1000
            seen_ids.update(ID_RE.findall(text))
            n = len(result) if isinstance(result, list) else (1 if result else 0)
            trace.append({"step": i, "tool": step.tool, "args": step.args, "results": n, "error": err})
            messages.append({"role": "user", "content": f"TOOL RESULT ({step.tool}, {n} item(s)):\n{text}"}); continue
        if isinstance(step, ClarifyStep):
            final = {"action": "clarify", "question": step.question}; break
        valid = [c for c in step.citations if c in seen_ids]
        invalid = [c for c in step.citations if c not in seen_ids]
        already = any("revise" in t for t in trace)
        # numeric claims against the window tools
        issue = None
        try:
            if COUNT_Q_RE.search(question):
                truth = T.count_entities_window(engine, ws, we, camera_id=cam)["count"]; nums = _numbers_in(step.text)
                if nums and truth not in nums and all(abs(n - truth) > 1 for n in nums[:3]):
                    issue = f"count_entities says {truth} confirmed people in this window (your answer implies {nums[:3]})."
            if WHOLE_TIME_RE.search(question):
                ep = T.entities_present_window(engine, ws, we, 0.9, camera_id=cam); truth_ids = set(ep["entity_ids"])
                cited = {c for c in step.citations if re.search(r":E\d+$", c)}
                if truth_ids and len(cited & truth_ids) < max(1, len(truth_ids) - 1):
                    issue = f"entities_present(0.9) says {sorted(truth_ids)} stayed the whole window; you named {sorted(cited & truth_ids) or 'none'}."
        except Exception:
            issue = None
        bad_claims = contradicted_event_claims_window(engine, ws, we, step.text)
        if (issue or bad_claims) and not already:
            m = issue or f"your answer mentions {bad_claims} but no such events exist in this window."
            trace.append({"step": i, "revise": m}); messages.append({"role": "user", "content": "TOOL RESULT: " + m + " Answer again."}); continue
        if invalid and not valid and not already:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in the script or tool results; cite only ids you were shown."}); continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid, "confidence": step.confidence,
                 "cited": bool(valid), "unsupported_event_claims": bad_claims, "numeric_issue": issue}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False, "confidence": 0.0}
    return {"question": question, "grounding": g.kind, "camera_id": cam, "window_ms": [ws, we], "window": f"{fmt(ws, tz)}–{fmt(we, tz)}" + (f" on {cam}" if cam else ""),
            "backend": getattr(backend, "name", "?"), "steps": len(trace), "trace": trace, "final": final,
            "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000), "model_ms": round(model_ms), "tool_ms": round(tool_ms),
                        "turns": sum(1 for t in trace if "tool" in t or "revise" in t or "invalid" in t) + 1, "first_prompt_chars": len(messages[1]["content"]),
                        "gen_tokens": getattr(backend, "last_gen_tokens", None), "tok_s": round(getattr(backend, "last_tok_s", 0.0), 1) or None}}


def contradicted_event_claims_window(engine: Engine, ws: int, we: int, text: str) -> list[str]:
    neg = re.compile(r"\b(no|not|nobody|none|never|didn't|did not|wasn't|weren't|isn't|aren't|without)\b")
    named: set[str] = set()
    for sent in re.split(r"(?<=[.!?;])\s+", text.lower()):
        if neg.search(sent):
            continue
        for et, words in EVENT_WORDS.items():
            if any(w in sent for w in words):
                named.add(et)
    if not named:
        return []
    have = {e["type"] for e in T.search_events(engine, t_start_ms=ws, t_end_ms=we, limit=10000)}
    return sorted(et for et in named if et not in have)
EOF_VI
cat > vi/agent/__init__.py << 'EOF_VI'
from .loop import FakeBackend, OpenAIBackend, TransformersBackend, ask, ask_window, extract_json
from .tools import (clip, count_entities, count_entities_window, coverage, coverage_window, entities_present,
                    entities_present_window, episodes_in, footage_bounds, get_script, search_entities, search_events,
                    search_tubes, window_script)
from .scope import classify as classify_scope
from .timeground import Grounding, ground as ground_time
EOF_VI
cat > vi/store/loader.py << 'EOF_VI'
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from vi.episode import EpisodeWriter
from vi.schemas.episode import EpisodeClose, EpisodeHeader, EventRecord, PatchRecord, TickRecord, TubeRecord

from .db import custody, entities, episodes, events, insert_ignore, patches, ticks, tubes


def load_episode_file(engine: Engine, path: str | Path) -> dict:
    """Load one episode JSONL into the store. Safe to run twice: every row is keyed by the
    writer's deterministic ids. Entities are derived from tube records grouped by entity_id
    (anonymous tubes get a one-tube entity of their own so every tube is answerable)."""
    counts: dict[str, int] = defaultdict(int)
    header: EpisodeHeader | None = None
    close: EpisodeClose | None = None
    tick_rows, event_rows, patch_rows, tube_rows, custody_rows = [], [], [], [], []
    for rec in EpisodeWriter.read(path):
        if isinstance(rec, EpisodeHeader):
            header = rec
        elif isinstance(rec, TickRecord):
            t = rec.tick
            tick_rows.append(dict(episode_id=rec.episode_id, camera_id=t.camera_id, tick_index=t.tick_index,
                                  t_start_ms=t.t_start.corrected_ms(), t_end_ms=t.t_end.corrected_ms(),
                                  modality=t.modality.value, tubes=[s.model_dump(mode="json") for s in t.tubes],
                                  event_ids=t.event_ids, scene_state=t.scene_state, gate_energy=t.gate_energy))
        elif isinstance(rec, EventRecord):
            e = rec.event
            event_rows.append(dict(event_id=e.event_id, episode_id=rec.episode_id, type=e.type.value,
                                   t_ms=e.t.corrected_ms(), camera_id=e.camera_id, tile_id=e.tile_id, zone_id=e.zone_id,
                                   subject_tube_ids=e.subject_tube_ids, subject_entity_ids=e.subject_entity_ids,
                                   object_ids=e.object_ids, payload=e.payload, confidence=e.confidence,
                                   source=e.source, dedupe_key=e.dedupe_key))
            if e.type.value in ("pickup", "drop", "asset_missing_from_home", "left_behind"):
                for obj in e.object_ids or ["?"]:
                    custody_rows.append(dict(event_id=e.event_id, object_id=obj, person_tube_ids=e.subject_tube_ids,
                                             t_ms=e.t.corrected_ms(), kind=e.type.value, zone_id=e.zone_id))
        elif isinstance(rec, PatchRecord):
            p = rec.patch
            patch_rows.append(dict(patch_id=p.patch_id, episode_id=rec.episode_id, tube_id=p.tube_id,
                                   produced_at_ms=p.produced_at_ms, source=p.source, payload=p.payload,
                                   confidence=p.confidence, modality=p.modality.value, enhanced=p.enhanced, failed=p.failed))
        elif isinstance(rec, TubeRecord):
            t = rec.tube
            tube_rows.append(dict(tube_id=t.tube_id, episode_id=rec.episode_id, camera_id=t.camera_id, tile_id=t.tile_id,
                                  entity_id=t.entity_id or f"anon:{t.tube_id}", named=t.named, class_label=t.class_label,
                                  state=t.state.value, born_ms=t.born.corrected_ms(), last_seen_ms=t.last_seen.corrected_ms(),
                                  box=t.box.model_dump(), zone_ids=t.zone_ids, modality=t.modality.value,
                                  keyframe_refs=t.keyframe_refs, attributes=t.attributes.model_dump(mode="json") if t.attributes else None,
                                  merge_candidates=t.merge_candidates, quality=t.quality, quality_reason=t.quality_reason,
                                  embedding=t.embedding))
        elif isinstance(rec, EpisodeClose):
            close = rec
    if header is None:
        raise ValueError(f"{path}: no header record")
    ent_rows: dict[str, dict] = {}
    for r in tube_rows:
        e = ent_rows.setdefault(r["entity_id"], dict(entity_id=r["entity_id"], camera_id=r["camera_id"], named=r["named"],
                                                     class_label=r["class_label"], tube_ids=[], first_seen_ms=r["born_ms"],
                                                     last_seen_ms=r["last_seen_ms"], best_keyframe_ref=None, embedding=None,
                                                     quality="low"))
        e["tube_ids"].append(r["tube_id"])
        if r.get("quality", "ok") == "ok":
            e["quality"] = "ok"                     # an entity is confirmed if any of its tubes is
        e["first_seen_ms"] = min(e["first_seen_ms"], r["born_ms"])
        e["last_seen_ms"] = max(e["last_seen_ms"], r["last_seen_ms"])
        if e["best_keyframe_ref"] is None and r["keyframe_refs"]:
            e["best_keyframe_ref"] = r["keyframe_refs"][0]
        if e.get("embedding") is None and r.get("embedding"):
            e["embedding"] = r["embedding"]
    with engine.begin() as conn:
        counts["episodes"] += insert_ignore(conn, episodes, [dict(
            episode_id=header.episode_id, tile_id=header.tile_id, camera_ids=header.camera_ids,
            t0_ms=header.t0.corrected_ms(), t1_ms=close.t1.corrected_ms() if close else None,
            status=close.status.value if close else "open", kb_version=header.provenance.kb_version,
            schema_version=header.provenance.schema_version)])
        if close:  # a re-load after a late close updates the status (E-STO-01)
            conn.execute(update(episodes).where(episodes.c.episode_id == header.episode_id)
                         .values(t1_ms=close.t1.corrected_ms(), status=close.status.value))
        counts["ticks"] += insert_ignore(conn, ticks, tick_rows)
        counts["events"] += insert_ignore(conn, events, event_rows)
        counts["patches"] += insert_ignore(conn, patches, patch_rows)
        counts["tubes"] += insert_ignore(conn, tubes, tube_rows)
        for r in tube_rows:                    # a tube that continued past a soft cut: extend its record from the later episode
            cur = conn.execute(select(tubes.c.last_seen_ms, tubes.c.entity_id).where(tubes.c.tube_id == r["tube_id"])).first()
            if cur is not None and r["last_seen_ms"] > (cur[0] or 0):
                conn.execute(update(tubes).where(tubes.c.tube_id == r["tube_id"]).values(
                    last_seen_ms=r["last_seen_ms"], state=r["state"], box=r["box"], keyframe_refs=r["keyframe_refs"],
                    attributes=r["attributes"], quality=r["quality"], quality_reason=r["quality_reason"],
                    embedding=r["embedding"] or None, entity_id=r["entity_id"] if not r["entity_id"].startswith("anon:") else cur[1]))
        counts["entities"] += insert_ignore(conn, entities, list(ent_rows.values()))
        for e in ent_rows.values():        # an entity that already exists (earlier episode) grows its span and tube list
            cur = conn.execute(select(entities).where(entities.c.entity_id == e["entity_id"])).first()
            if cur is not None:
                cur = dict(cur._mapping)
                merged_tubes = sorted(set((cur["tube_ids"] or []) + e["tube_ids"]))
                if merged_tubes != (cur["tube_ids"] or []) or e["last_seen_ms"] > (cur["last_seen_ms"] or 0):
                    conn.execute(update(entities).where(entities.c.entity_id == e["entity_id"]).values(
                        tube_ids=merged_tubes, first_seen_ms=min(cur["first_seen_ms"], e["first_seen_ms"]),
                        last_seen_ms=max(cur["last_seen_ms"], e["last_seen_ms"]),
                        best_keyframe_ref=cur["best_keyframe_ref"] or e["best_keyframe_ref"],
                        quality="ok" if "ok" in (cur.get("quality"), e.get("quality")) else "low"))
        counts["custody"] += insert_ignore(conn, custody, custody_rows)
    counts["episode_id"] = header.episode_id  # type: ignore[assignment]
    return dict(counts)
EOF_VI
cat > vi/writer/contact_sheet.py << 'EOF_VI'
"""Ring 3b, the writer branch: pack person crops into one numbered grid, ask the VLM once, get one
JSON record per cell under the Attributes schema (R22). Runs at tube events, never per frame.
The VLM never sees a whole frame here; it sees crops with cell numbers, so it cannot confuse
people across cells without saying which cell it means."""
from __future__ import annotations

import json
import re
import time

import numpy as np

from vi.schemas import Attributes, Modality
from vi.schemas.contact_sheet import CellResult, ContactSheetResult

COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "multicolor"]

WRITER_PROMPT = (
    "This image is a grid of cells; each cell shows one person cropped from a camera, with the cell number in the "
    "yellow strip under it (the strip is a label, not part of the scene). For EVERY cell, describe that person only. "
    "Reply with a JSON array, one object per cell, no prose:\n"
    '[{"cell_id": 0, "top_color": colour of the most visible upper-body garment (a vest counts), one of ' + str(COLORS) + ' or null, '
    '"bottom_color": same or null, '
    '"headwear": short text or null, "carried_item": short text or null, "role": short text or null, '
    '"description": at most 12 words, "confidence": 0-1}]\n'
    "Use null when a field is not visible. Cells are numbered top-left to bottom-right starting at 0."
)


def pack_sheet(crops: list[np.ndarray], cell: int = 224, cols: int = 4, caption: int = 22):
    """Grid with hard borders and a number in a caption strip BELOW each crop, never over it (a badge
    on top of a 30-px crop is what the model ends up describing). Small crops are upscaled to fill
    the cell so the model sees a person, not a speck (E-FOV-04)."""
    from PIL import Image, ImageDraw
    n = len(crops)
    cols = min(cols, max(1, n))
    rows = (n + cols - 1) // cols
    ch = cell + caption
    sheet = Image.new("RGB", (cols * cell, rows * ch), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(crops):
        x, y = (i % cols) * cell, (i // cols) * ch
        im = Image.fromarray(np.ascontiguousarray(c)).convert("RGB")
        scale = min((cell - 8) / max(1, im.width), (cell - 8) / max(1, im.height))   # up or down to fill the cell
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        sheet.paste(im, (x + 4 + (cell - 8 - im.width) // 2, y + 4 + (cell - 8 - im.height) // 2))
        dr.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2], outline=(255, 255, 255), width=2)
        dr.rectangle([x, y + cell, x + cell - 1, y + ch - 1], fill=(255, 215, 0))
        dr.text((x + 6, y + cell + 4), f"cell #{i}", fill=(0, 0, 0))
    return sheet


def _color(v):
    return v.lower() if isinstance(v, str) and v.lower() in COLORS else None


def parse_sheet_reply(text: str, tube_ids: list[str], modality: Modality = Modality.rgb) -> ContactSheetResult | None:
    """Tolerant parse of the VLM reply into the schema; None if it cannot be made valid (E-FOV-07)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end < 0:
        return None
    try:
        items = json.loads(t[start:end + 1])
    except Exception:
        return None
    cells: list[CellResult] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            cid = int(it.get("cell_id"))
        except Exception:
            continue
        if not (0 <= cid < len(tube_ids)):
            continue
        desc_bits = [str(it.get("description") or "")[:120]]
        if it.get("headwear"):
            desc_bits.append(f"headwear: {str(it['headwear'])[:40]}")
        if it.get("role"):
            desc_bits.append(f"role: {str(it['role'])[:30]}")
        conf = it.get("confidence", 0.6)
        conf = float(conf) if isinstance(conf, (int, float)) else 0.6
        kwargs = dict(modality=modality, carried_item=(str(it["carried_item"])[:60] if it.get("carried_item") else None),
                      carried_item_confidence=0.6 if it.get("carried_item") else 0.0,
                      description="; ".join(b for b in desc_bits if b)[:240], confidence=max(0.0, min(1.0, conf)))
        if modality.has_color:
            kwargs.update(top_color=_color(it.get("top_color")), bottom_color=_color(it.get("bottom_color")))
        else:
            kwargs.update(color_reason="ir_mode")
        try:
            cells.append(CellResult(cell_id=cid, tube_id=tube_ids[cid], attributes=Attributes(**kwargs)))
        except Exception:
            continue
    # fill missing cells with an empty attributes record so the sheet validates (per-cell failure, not sheet failure)
    have = {c.cell_id for c in cells}
    for i, tid in enumerate(tube_ids):
        if i not in have:
            cells.append(CellResult(cell_id=i, tube_id=tid, attributes=Attributes(
                modality=modality, confidence=0.0, **({} if modality.has_color else {"color_reason": "ir_mode"}))))
    cells.sort(key=lambda c: c.cell_id)
    try:
        return ContactSheetResult(sheet_id=f"sheet_{int(time.time() * 1000)}", modality=modality,
                                  expected_cells=len(tube_ids), cells=cells)
    except Exception:
        return None


class WriterVLM:
    """Qwen3.5 (or any transformers image-text model) on a packed sheet. `backend` may be an
    existing TransformersBackend to reuse its loaded model and processor."""

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", backend=None, max_new_tokens: int = 700, device: str | None = None):
        import torch
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        if backend is not None and getattr(backend, "model", None) is not None:
            self.model, self.proc, self.device = backend.model, backend.tok, backend.device
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
            self.proc = AutoProcessor.from_pretrained(model_id)
        self.calls = 0
        self.last_ms = 0.0

    def describe(self, crops: list[np.ndarray], tube_ids: list[str], modality: Modality = Modality.rgb) -> ContactSheetResult | None:
        sheet = pack_sheet(crops)
        messages = [{"role": "user", "content": [{"type": "image", "image": sheet}, {"type": "text", "text": WRITER_PROMPT}]}]
        t0 = time.perf_counter()
        try:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt", enable_thinking=False)
        except TypeError:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tok = getattr(self.proc, "tokenizer", self.proc)
        text = tok.decode(gen, skip_special_tokens=True)
        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return parse_sheet_reply(text, tube_ids, modality)
EOF_VI
cat > bench/run_ingest.py << 'EOF_VI'
"""Long-running ingest, one or many cameras from one source:
  * a single camera (a file, paced to wall clock with --realtime, or an RTSP URL), or
  * a multiplexed NVR export (--grid RxC): every cell is a virtual camera; all cells of a frame are
    detected in ONE batched call (R12, cross-camera batching), then tracked, linked, described and
    compiled per camera; episodes are per camera and load into the store as they close.

  python bench/run_ingest.py --source /content/mall_hour.mp4 --grid 4x4 --profile tier2_public --model medium \
      --start-time "2026-09-27T10:00:00+05:30" --db "$DB_URL" --reid siglip --writer none --realtime
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi.detect import dedupe_detections, full_frame_roi, remap_detections
from vi.episode import EpisodeWriter, KeyframeStore, annotate, should_soft_cut
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate
from vi.ingest import GridSpec, cell_ids, compose, label_zones, split
from vi.ingest import VideoReader
from vi.profiles import load_profile
from vi.reid import HistogramEmbedder, crop_for_embedding, make_embedder
from vi.schemas import Box, CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.store import connect, load_episode_file
from vi.tubes import TRACKERS, TubeLinker, grade_tube


def parse_start(s: str | None) -> int:
    if not s:
        return int(time.time() * 1000)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass
class Cam:
    camera_id: str
    tile_id: str
    w: int = 0
    h: int = 0
    gate: object = None
    tracker: object = None
    compiler: object = None
    linker: object = None
    zones: list = field(default_factory=list)
    media: list = field(default_factory=list)
    ep: str | None = None
    ep_t0: int = 0
    ep_cast_prev: set = field(default_factory=set)
    ep_tubes: list = field(default_factory=list)
    last_live_ms: int | None = None
    described: set = field(default_factory=set)
    pending: dict = field(default_factory=dict)
    pending_aux: dict = field(default_factory=dict)
    current: dict = field(default_factory=lambda: {"frame": None})
    last_annotated: np.ndarray | None = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="file path or rtsp:// url")
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--camera", default="cam1"); ap.add_argument("--tile", default="floor")
    ap.add_argument("--grid", default=None, help="RxC, or 'auto' to detect the NVR layout from the seams in the first frames")
    ap.add_argument("--grid-margin", type=int, default=0)
    ap.add_argument("--start-time", default=None); ap.add_argument("--profile", default="common")
    ap.add_argument("--model", default="nano"); ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--reid", default="siglip"); ap.add_argument("--writer", choices=["none", "qwen", "fake"], default="none")
    ap.add_argument("--writer-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--zones", default=None, help="zones JSON (single camera); grid cameras get border exits + label media zones")
    ap.add_argument("--episode-min", type=float, default=10.0); ap.add_argument("--quiet-close-s", type=float, default=30.0)
    ap.add_argument("--realtime", action="store_true"); ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--max-width", type=int, default=0, help="decode width cap (0 = native); grids need native resolution")
    ap.add_argument("--out", default="data/episodes")
    ap.add_argument("--live-dir", default="data/live"); ap.add_argument("--live-every", type=int, default=4)
    ap.add_argument("--tiles", default="auto", help="tile map JSON (cameras that see the same area share one linker); auto = data/tiles.json if present; none")
    a = ap.parse_args()

    profile = load_profile(a.profile)
    tube_classes = set(profile.tube_classes)
    start_ms = parse_start(a.start_time)
    engine = connect(a.db)
    prov = Provenance(kb_version=1, pipeline_git="run_ingest")
    if a.grid and a.grid.lower() == "auto":
        from vi.ingest import detect_grid
        probe = VideoReader(a.camera, a.source, target_fps=1.0, max_width=1920, want_rgb=True)
        sample = []
        for f in probe.frames():
            sample.append(f.rgb)
            if len(sample) >= 5:
                break
        spec, ev = detect_grid(sample)
        print(f"[ingest] grid auto -> {(str(spec.rows) + 'x' + str(spec.cols)) if spec else 'single camera'}  (seam evidence {ev})", flush=True)
    else:
        spec = GridSpec.parse(a.grid, a.grid_margin) if a.grid else None
    n_cams = spec.n if spec else 1
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=(a.max_width or (1920 if spec else 640)), want_rgb=True)
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold, batch_size=n_cams)
    else:
        from vi.detect.rfdetr import RFDETRDetector
        det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=n_cams)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    aux = HistogramEmbedder() if embedder and embedder.name != "hist" else None
    vlm = None
    if a.writer == "qwen":
        try:
            from vi.writer import WriterVLM
            vlm = WriterVLM(model_id=a.writer_model); print(f"[writer] {a.writer_model} loaded", flush=True)
        except Exception as e:
            print(f"[writer] unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
    elif a.writer == "fake":
        from vi.writer.contact_sheet import parse_sheet_reply
        class _F:
            calls = 0; last_ms = 0.0
            def describe(self, crops, ids, modality=None):
                self.calls += 1
                return parse_sheet_reply(json.dumps([{"cell_id": i, "top_color": "orange", "description": "person", "confidence": 0.7} for i in range(len(ids))]), ids)
        vlm = _F()
    writer = EpisodeWriter(a.out)
    ids = cell_ids(spec) if spec else [a.camera]
    from vi.fusion import TileMap
    tilemap = None
    if a.tiles and a.tiles != "none":
        tilemap = TileMap.load("data/tiles.json" if a.tiles == "auto" else a.tiles)
    def tile_for(cid: str) -> str:
        if tilemap is not None:
            return tilemap.tile_of(cid)
        return cid if spec else a.tile
    cams = {cid: Cam(camera_id=cid, tile_id=tile_for(cid)) for cid in ids}
    tile_linkers: dict[str, TubeLinker] = {}
    if tilemap is not None:
        print(f"[ingest] tiles: {tilemap.tiles}", flush=True)
    tick_ms = int(1000 / a.fps)
    frames = 0; t_wall0 = time.time(); pts0 = None
    stats: Counter = Counter()
    detect_ms: list[float] = []

    def init_cam(cam: Cam, w: int, h: int) -> None:
        cam.w, cam.h = w, h
        if spec:
            cam.zones = [z for z in default_zones(cam.camera_id, w, h, tile_id=cam.tile_id) if z.kind == "exit"]
            cam.zones += label_zones(spec, w, h, cam.camera_id, cam.tile_id)
        else:
            zp = a.zones or (str(Path("data/zones") / (Path(a.source).stem + ".json")) if (Path("data/zones") / (Path(a.source).stem + ".json")).exists() else None)
            cam.zones = load_zones(zp, cam.camera_id) if zp else default_zones(cam.camera_id, w, h, tile_id=cam.tile_id)
        cam.media = [z for z in cam.zones if z.kind == "media"]
        exits = [Box(x1=min(p[0] for p in z.polygon), y1=min(p[1] for p in z.polygon), x2=max(p[0] for p in z.polygon), y2=max(p[1] for p in z.polygon))
                 for z in cam.zones if z.kind == "exit"]
        cam.gate = FrameDiffGate(cam.camera_id)
        cam.tracker = TRACKERS["byte"](cam.camera_id, exit_boxes=exits, keyframe_sink=kf.make_sink(lambda c=cam: c.current["frame"]))
        cam.compiler = EventCompiler(cam.camera_id, cam.zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=cam.tile_id)
        if embedder:
            cam.linker = tile_linkers.setdefault(cam.tile_id, TubeLinker(cam.tile_id)) if tilemap is not None else TubeLinker(cam.camera_id)
        else:
            cam.linker = None

    def close_episode(cam: Cam, t_end_ms: int, status: EpisodeStatus) -> None:
        if cam.ep is None:
            return
        heights = sorted(t.max_height_px for t in cam.ep_tubes if t.class_label == "person" and t.max_height_px > 0)
        med = heights[len(heights) // 2] if heights else None
        for t in cam.ep_tubes:
            if cam.linker is not None and t.class_label == "person":
                t.entity_id = cam.linker.entity_of(t.tube_id) or t.entity_id
                t.embedding = cam.linker.embedding_of(t.tube_id)
            grade_tube(t, cam.w, cam.h, median_height_px=med, min_life_ms=int(profile.quality.min_life_s * 1000),
                       min_height_frac=profile.quality.min_height_frac, border_px=profile.quality.border_px)
            writer.write_tube(cam.ep, t)
        cast = [CastMember(entity_id=t.entity_id, tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0]) for t in cam.ep_tubes]
        writer.close(cam.ep, CamTime(cam_utc_ms=t_end_ms), status, cast)
        counts = load_episode_file(engine, writer.path(cam.ep))
        stats["episodes"] += 1
        print(f"[episode] {cam.camera_id} {cam.ep} {status.value} {len(cam.ep_tubes)} tubes -> store tubes={counts['tubes']} events={counts['events']}", flush=True)
        cam.ep, cam.ep_tubes, cam.ep_cast_prev = None, [], set()

    def step_cam(cam: Cam, rgb: np.ndarray, gray: np.ndarray, pts_ms: int, t_ms: int, dets: list) -> None:
        cam.current["frame"] = rgb
        g = cam.gate.update(gray, pts_ms)
        events = cam.compiler.on_gate(g)
        dets = dedupe_detections([d for d in dets if d.class_label in tube_classes])
        if cam.media:
            dets = [d for d in dets if not any(z.contains(d.box.foot_point()) for z in cam.media)]
        live, closed = cam.tracker.update(dets, t_ms, det_source="heartbeat")
        if cam.ep is None and live:
            cam.ep = writer.open(cam.tile_id, [cam.camera_id], CamTime(cam_utc_ms=t_ms), prov); cam.ep_t0 = t_ms
            cam.compiler = EventCompiler(cam.camera_id, cam.zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=cam.tile_id)
        if live:
            cam.last_live_ms = t_ms
        if cam.linker is not None:
            due = [t for t in live if t.class_label == "person" and t.tube_id not in cam.pending and t.tube_id not in cam.described and cam.linker.entity_of(t.tube_id) is None]
            if due:
                crops = [crop_for_embedding(rgb, t.box) for t in due]
                embs = embedder.embed(crops); auxs = aux.embed(crops) if aux else [None] * len(crops)
                for t, e, ax in zip(due, embs, auxs):
                    cam.pending[t.tube_id] = e
                    if ax is not None: cam.pending_aux[t.tube_id] = ax
            for t in live:
                if t.class_label == "person" and t.tube_id not in cam.pending: cam.linker.on_state(t, t_ms)
            for t in live:
                if t.tube_id in cam.pending and t.state.value == "active":
                    ev = cam.linker.on_birth(t, cam.pending.pop(t.tube_id), t_ms, cam.pending_aux.pop(t.tube_id, None))
                    if ev is not None: events.append(ev); stats["relinks"] += 1
            live_ids = {t.tube_id for t in live}
            for tid in [k for k in cam.pending if k not in live_ids]:
                cam.pending.pop(tid); cam.pending_aux.pop(tid, None)
            for ghost in cam.linker.absorbed:
                dead = cam.tracker.drop(ghost)
                if dead is not None: closed.append(dead)
            cam.linker.absorbed.clear()
            live = [t for t in live if t.tube_id in cam.tracker._tracks]
            for t in closed:
                mev = cam.linker.on_close(t, t_ms)
                if mev is not None: events.append(mev); stats["merges"] += 1
            for t in live:
                if t.class_label == "person": t.entity_id = cam.linker.entity_of(t.tube_id)
        if vlm is not None and frames % 4 == 0:
            todo = [t for t in live if t.class_label == "person" and t.state.value == "active" and t.tube_id not in cam.described][:8]
            if todo:
                res = vlm.describe([crop_for_embedding(rgb, t.box, pad=0.15) for t in todo], [t.tube_id for t in todo])
                if res is not None:
                    by = {c.tube_id: c.attributes for c in res.cells}
                    for t in todo:
                        if t.tube_id in by and by[t.tube_id].confidence > 0: t.attributes = by[t.tube_id]
                for t in todo: cam.described.add(t.tube_id)
                stats["sheets"] += 1
        cam.ep_tubes += closed
        if cam.ep is not None:
            snaps = [TubeSnapshot(tube_id=t.tube_id, class_label=t.class_label, state=t.state, box=t.box,
                                  det_source="detector" if t.state.value == "active" else "predicted") for t in live]
            events += cam.compiler.on_tick(snaps, t_ms)
            writer.write_tick(cam.ep, Tick(camera_id=cam.camera_id, tile_id=cam.tile_id, tick_index=frames, t_start=CamTime(cam_utc_ms=t_ms),
                                           t_end=CamTime(cam_utc_ms=t_ms + tick_ms), tubes=snaps, event_ids=[e.event_id for e in events], provenance=prov))
            for e in events:
                writer.write_event(cam.ep, e); stats["events"] += 1
            cast_now = {t.entity_id or t.tube_id for t in live if t.class_label == "person"}
            quiet = cam.last_live_ms is not None and not live and t_ms - cam.last_live_ms > a.quiet_close_s * 1000
            if quiet or should_soft_cut(cam.ep_cast_prev, cast_now, t_ms - cam.ep_t0, max_duration_ms=int(a.episode_min * 60_000)):
                cam.ep_tubes += list(live)
                close_episode(cam, t_ms, EpisodeStatus.closed if quiet else EpisodeStatus.soft_cut)
            elif frames % 40 == 0:
                cam.ep_cast_prev = cast_now
        if a.live_every and frames % a.live_every == 0:
            cam.last_annotated = annotate(rgb, [], dets, live, f"{cam.camera_id} {datetime.fromtimestamp(t_ms / 1000, timezone.utc).strftime('%H:%M:%S')}Z live {len(live)}", None)
        cam_live[cam.camera_id] = len(live)

    cam_live: dict[str, int] = {}
    fr = None
    for fr in reader.frames():
        if pts0 is None:
            pts0 = fr.pts_ms
        if a.max_minutes and fr.pts_ms - pts0 > a.max_minutes * 60_000:
            break
        if a.realtime:
            lag = (fr.pts_ms - pts0) / 1000 - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(min(lag, 1.0))
        t_ms = start_ms + fr.pts_ms
        cells = split(fr.rgb, spec) if spec else [fr.rgb]
        grays = [c.mean(axis=2).astype(np.uint8) for c in cells] if spec else [fr.gray]
        for cid, cell in zip(ids, cells):
            if cams[cid].tracker is None:
                init_cam(cams[cid], cell.shape[1], cell.shape[0])
        t0 = time.perf_counter()
        batched = det.detect_batch(cells)                    # one call for every camera in the frame (R12)
        detect_ms.append((time.perf_counter() - t0) * 1000)
        for cid, cell, gray, dets in zip(ids, cells, grays, batched):
            h, w = cell.shape[:2]
            dets = remap_detections(list(dets), full_frame_roi(w, h), w, h)
            step_cam(cams[cid], cell, gray, fr.pts_ms, t_ms, dets)
        if a.live_every and frames % a.live_every == 0:
            live_dir = Path(a.live_dir); live_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            panels = [cams[cid].last_annotated for cid in ids if cams[cid].last_annotated is not None]
            if panels:
                img = compose(panels, spec) if spec else panels[0]
                Image.fromarray(img).save(live_dir / "latest.tmp.jpg", quality=80)
                try:
                    (live_dir / "latest.tmp.jpg").replace(live_dir / "latest.jpg")
                except Exception:
                    pass
            (live_dir / "status.json").write_text(json.dumps({"frames": frames, "footage_s": round((fr.pts_ms - pts0) / 1000, 1),
                                                              "wall_s": round(time.time() - t_wall0, 1), "cameras": n_cams,
                                                              "live_tubes": sum(cam_live.values()), "per_camera": cam_live, "now_ms": t_ms,
                                                              "episodes": stats["episodes"], "sheets": stats["sheets"],
                                                              "detect_ms_p50": round(float(np.median(detect_ms[-50:])), 1) if detect_ms else None}))
        frames += 1
        if frames % 200 == 0:
            el = time.time() - t_wall0
            print(f"[ingest] {frames} frames x {n_cams} cams  footage {(fr.pts_ms - pts0)/1000:6.0f}s  wall {el:6.0f}s  "
                  f"realtime x{((fr.pts_ms - pts0)/1000) / max(el, 1e-6):.2f}  detect {np.median(detect_ms[-50:]):.0f}ms/batch  live {sum(cam_live.values())}  {dict(stats)}", flush=True)
    if fr is not None:
        for cam in cams.values():
            if cam.ep is not None:
                cam.ep_tubes += [tr.tube for tr in cam.tracker._tracks.values()]
                close_episode(cam, start_ms + fr.pts_ms + tick_ms, EpisodeStatus.closed)
    footage_s = round((fr.pts_ms - (pts0 or 0)) / 1000, 1) if fr is not None else 0
    wall = round(time.time() - t_wall0, 1)
    row = {"ring": "ingest", "source": Path(a.source).name, "cameras": n_cams, "grid": a.grid, "model": a.model, "frames": frames,
           "footage_s": footage_s, "wall_s": wall, "realtime_factor": round(footage_s / max(wall, 1e-6), 2),
           "detect_ms_p50_per_batch": round(float(np.median(detect_ms)), 1) if detect_ms else None,
           "camera_frames_per_s": round(frames * n_cams / max(wall, 1e-6), 1), **dict(stats),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path("data/bench").mkdir(parents=True, exist_ok=True)
    with open("data/bench/ingest.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/slice_gpu.py << 'EOF_VI'
"""Real-footage slice (Day 4): reader -> gate -> ROIs -> RF-DETR -> tubes -> events -> episode file,
with a native-res keyframe saved at every tube birth. One row to data/bench/slice_gpu.jsonl.

  python bench/slice_gpu.py --source clip.mp4 --detect hybrid      # ROIs + 1 Hz full-frame heartbeat (default)
  python bench/slice_gpu.py --source clip.mp4 --detect roi         # motion ROIs only (session 04/05 behaviour)
  python bench/slice_gpu.py --source clip.mp4 --detect frame       # full frame every tick (upper bound on recall)
  python bench/slice_gpu.py --source clip.mp4 --zones data/zones/cam1.json --tile lobby
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vi.detect import (blobs_to_rois, crop_roi, dedupe_detections, full_frame_roi, is_tube_class, pad_batch,
                       remap_detections)
try:
    from vi.detect.rfdetr import RFDETRDetector
except ImportError:  # CPU runtime: only --model fake works
    RFDETRDetector = None  # type: ignore
from vi.episode import EpisodeWriter, KeyframeStore, annotate
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate, HeartbeatScheduler
from vi.ingest import VideoReader
from vi.schemas import CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.reid import HistogramEmbedder, crop_for_embedding, make_embedder
from vi.profiles import load_profile
from vi.tubes import TRACKERS, TubeLinker, grade_tube


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--tile", default="tile1")
    ap.add_argument("--model", default="nano")
    ap.add_argument("--fps", type=float, default=4.0, help="R11 range 2-5; IoU-only tracking collapses below ~4")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="detector floor; ByteTracker re-attaches 0.1-0.5 detections and births only >=0.5")
    ap.add_argument("--tracker", choices=sorted(TRACKERS), default="byte")
    ap.add_argument("--warmup", type=int, default=3, help="frames excluded from timing (JIT profiling runs)")
    ap.add_argument("--detect", choices=["roi", "frame", "hybrid"], default="frame",
                    help="frame (default, measured best at one camera): full frame every tick; roi: motion ROIs only; "
                         "hybrid: ROIs + full-frame heartbeat (multi-camera cost lever, under investigation)")
    ap.add_argument("--heartbeat-ms", type=int, default=1000, help="hybrid: full-frame detection period on active tiles")
    ap.add_argument("--quiet-heartbeat-ms", type=int, default=10000)
    ap.add_argument("--debug-frames", type=int, default=0, help="save N annotated frames to data/debug/<episode>/")
    ap.add_argument("--reid", choices=["none", "auto", "hist", "siglip", "osnet"], default="none",
                    help="appearance embeddings + TubeLinker (E-TUBE-04); auto = osnet > siglip > hist")
    ap.add_argument("--reid-every-ticks", type=int, default=8, help="gallery refresh cadence for active tubes")
    ap.add_argument("--reid-sim", type=float, default=0.88, help="from bench/reid_eval.py (SigLIP on the warehouse clip)")
    ap.add_argument("--reid-near-sim", type=float, default=0.85)
    ap.add_argument("--profile", default="common", help="build-frame profile (profiles/<name>.yaml): classes, branches, quality")
    ap.add_argument("--writer", choices=["none", "qwen", "fake"], default="none", help="contact-sheet attributes at tube confirmation")
    ap.add_argument("--writer-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--writer-batch", type=int, default=8, help="crops per sheet")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default: data/zones/<clip stem>.json if present, else edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    ap.add_argument("--skip-s", type=float, default=0.0, help="ignore the first N seconds (metamorphic time shift)")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    profile = load_profile(a.profile)
    tube_classes = set(profile.tube_classes)
    print(f"profile: {profile.name} branches={profile.branches}")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    batch = 1 if a.detect == "frame" else a.batch          # frame mode: one image per tick, trace at 1
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold, batch_size=batch)   # CPU smoke path (tests, no GPU)
    else:
        det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=batch)
    gate = FrameDiffGate(a.camera)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    current = {"frame": None}
    zones = None
    tracker = compiler = writer = ep = None
    stage = Counter()
    detect_ms: list[float] = []
    ev_types: Counter = Counter()
    births = 0
    closed_all = []
    tick_ms = int(1000 / a.fps)
    frames = 0
    hb = HeartbeatScheduler(active_ms=a.heartbeat_ms, quiet_ms=a.quiet_heartbeat_ms)
    heartbeats = 0
    births_by_origin: Counter = Counter()
    confirmed_by_origin: Counter = Counter()
    rebirths = 0                                   # tube born next to one that just died = fragmentation, counted directly
    recent_dead: list[tuple[int, float, float, float]] = []   # (t_ms, cx, cy, h)
    seen_ids: set[str] = set()
    confirmed_ids: set[str] = set()
    origin_of: dict[str, str] = {}
    debug_every = max(1, int(a.fps * 1.5)) if a.debug_frames else 0     # one frame every ~1.5 s until N saved
    duplicate_pairs: list[dict] = []                # two live person tubes on one body: the hybrid bug, caught in the act
    dup_frames: list[str] = []
    vlm_writer = None
    if a.writer == "qwen":
        try:
            from vi.writer import WriterVLM
            vlm_writer = WriterVLM(model_id=a.writer_model)
            print(f"[writer] {a.writer_model} loaded")
        except Exception as e:
            print(f"[writer] unavailable ({type(e).__name__}: {str(e)[:100]}); continuing without attributes")
    elif a.writer == "fake":
        from vi.writer.contact_sheet import parse_sheet_reply
        class _Fake:
            calls = 0; last_ms = 0.0
            def describe(self, crops, tube_ids, modality=None):
                self.calls += 1
                import json
                return parse_sheet_reply(json.dumps([{"cell_id": i, "top_color": "orange", "description": "worker in a vest", "confidence": 0.7}
                                                     for i in range(len(tube_ids))]), tube_ids)
        vlm_writer = _Fake()
    described: set[str] = set()
    writer_ms: list[float] = []
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    aux_embedder = HistogramEmbedder() if embedder and embedder.name != "hist" else None
    linker = TubeLinker(a.camera, sim_thr=a.reid_sim, near_sim_thr=a.reid_near_sim) if embedder else None
    absorbed_total = 0
    embed_ms: list[float] = []
    last_embed_tick: dict[str, int] = {}
    pending_link: dict[str, np.ndarray] = {}
    pending_aux: dict[str, np.ndarray] = {}
    relink_events = 0
    debug_paths: list[str] = []
    person_dets: list[int] = []
    concurrent_persons: list[int] = []
    state_ticks: Counter = Counter()

    for fr in reader.frames():
        if frames >= a.max_frames:
            break
        if a.skip_s and fr.pts_ms < a.skip_s * 1000:
            continue
        h, w = fr.rgb.shape[:2]
        current["frame"] = fr.rgb
        if zones is None:   # first frame: zones need the native size
            zones_path = a.zones or (str(Path("data/zones") / (Path(a.source).stem + ".json")) if (Path("data/zones") / (Path(a.source).stem + ".json")).exists() else None)
            zones = load_zones(zones_path, a.camera) if zones_path else default_zones(a.camera, w, h, tile_id=a.tile)
            media_zones = [z for z in zones if z.kind == "media"]
            print(f"zones: {[z.zone_id for z in zones]} ({'file ' + zones_path if zones_path else 'defaults'})")
            exit_boxes = [z.polygon for z in zones if z.kind == "exit"]
            from vi.schemas import Box
            exits = [Box(x1=min(p[0] for p in poly), y1=min(p[1] for p in poly),
                         x2=max(p[0] for p in poly), y2=max(p[1] for p in poly)) for poly in exit_boxes]
            tkw = {"iou_thr": a.iou_thr} if a.tracker == "simple" else {}
            tracker = TRACKERS[a.tracker](a.camera, max_occluded_ms=a.max_occluded_ms, exit_boxes=exits,
                                          keyframe_sink=kf.make_sink(lambda: current["frame"]), **tkw)
            compiler = EventCompiler(a.camera, zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=a.tile)
            writer = EpisodeWriter(a.out)
            ep = writer.open(a.tile, [a.camera], CamTime(cam_utc_ms=fr.pts_ms), prov)
        t0 = time.perf_counter()
        g = gate.update(fr.gray, fr.pts_ms)
        stage["gate"] += time.perf_counter() - t0
        events = compiler.on_gate(g)
        t0 = time.perf_counter()
        det_source = "detector"
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride) if a.detect in ("roi", "hybrid") else []
        tile_active = len(tracker._tracks) > 0
        if a.detect == "frame" or (a.detect == "hybrid" and hb.due(fr.pts_ms, tile_active)):
            rois.append(full_frame_roi(w, h))                   # heartbeat = one more crop in the batch
            det_source = "heartbeat"
            heartbeats += 1
        dets = []
        for i in range(0, len(rois), batch):
            chunk = rois[i:i + batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = dedupe_detections([d for d in dets if d.class_label in tube_classes])  # E-DET-10, classes from the profile
        if media_zones:   # E-DET-05: jackets on a rack, posters, screens are not people
            dets = [d for d in dets if not any(z.contains(d.box.foot_point()) for z in media_zones)]
        person_dets.append(sum(1 for d in dets if d.class_label == "person" and d.confidence >= 0.5))
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms, det_source=det_source)
        # provenance + rebirth accounting (person tubes only)
        for t in live:
            if t.class_label != "person":
                continue
            if t.tube_id not in seen_ids:
                seen_ids.add(t.tube_id)
                origin = next((d.origin for d in dets if d.box.iou(t.box) > 0.7), "unknown")
                births_by_origin[origin] += 1
                origin_of[t.tube_id] = origin
                cx, cy = (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2
                if any(fr.pts_ms - tm <= 1500 and ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 <= hh for tm, x, y, hh in recent_dead):
                    rebirths += 1
            if t.state.value == "active" and t.tube_id not in confirmed_ids:
                confirmed_ids.add(t.tube_id)
                confirmed_by_origin[origin_of.get(t.tube_id, "unknown")] += 1
        if linker is not None:
            # embed at first sight, but only *link* once the tracker has confirmed the tube (state
            # active): an unconfirmed tube can still be deleted next tick, and a link to a tube
            # that never existed corrupts the cast (seen in session 12).
            due_birth = [t for t in live if t.class_label == "person" and t.tube_id not in last_embed_tick]
            due_refresh = [t for t in live if t.class_label == "person" and t.state.value == "active"
                           and t.tube_id in last_embed_tick and t.tube_id not in pending_link
                           and frames - last_embed_tick[t.tube_id] >= a.reid_every_ticks]
            todo = due_birth + due_refresh
            if todo:
                t0 = time.perf_counter()
                crops = [crop_for_embedding(fr.rgb, t.box) for t in todo]
                embs = embedder.embed(crops)
                auxs = aux_embedder.embed(crops) if aux_embedder else [None] * len(todo)
                embed_ms.append((time.perf_counter() - t0) * 1000)
                birth_ids = {x.tube_id for x in due_birth}
                for t, e, ax in zip(todo, embs, auxs):
                    last_embed_tick[t.tube_id] = frames
                    if t.tube_id in birth_ids:
                        pending_link[t.tube_id] = e
                        if ax is not None:
                            pending_aux[t.tube_id] = ax
                    else:
                        linker.on_refresh(t, e, ax)
            live_ids = {t.tube_id for t in live}
            for t in live:
                if t.class_label == "person" and t.tube_id not in pending_link:
                    linker.on_state(t, fr.pts_ms)          # occluded tubes become relink candidates
            for t in live:
                if t.tube_id in pending_link and t.state.value == "active":
                    ev = linker.on_birth(t, pending_link.pop(t.tube_id), fr.pts_ms, pending_aux.pop(t.tube_id, None))
                    if ev is not None:
                        events.append(ev)
                        relink_events += 1
                        print(f"t={fr.pts_ms:7d}  relink                 {ev.subject_tube_ids[0]} -> {ev.subject_tube_ids[1]} sim={ev.payload['similarity']} thr={ev.payload['threshold']}")
            for tid in [k for k in pending_link if k not in live_ids]:
                pending_link.pop(tid); pending_aux.pop(tid, None)    # deleted while unconfirmed: never linked
            for ghost in linker.absorbed:                             # the occluded tube a link replaced
                dead = tracker.drop(ghost)
                if dead is not None:
                    closed_all.append(dead)
                    absorbed_total += 1
            linker.absorbed.clear()
            live = [t for t in live if t.tube_id in tracker._tracks]
            for t in live:
                if t.class_label == "person":
                    t.entity_id = linker.entity_of(t.tube_id)
            for t in closed:
                mev = linker.on_close(t, fr.pts_ms)
                if mev is not None:
                    events.append(mev)
                    print(f"t={fr.pts_ms:7d}  merge                  {mev.payload['merged_entity']} -> {mev.subject_entity_ids[0]} sim={mev.payload['similarity']}")
        if vlm_writer is not None and frames % 4 == 0:  # tube-event cadence: newly confirmed people, at most every 4 ticks
            todo = [t for t in live if t.class_label == "person" and t.state.value == "active" and t.tube_id not in described][: a.writer_batch]
            if todo:
                res = vlm_writer.describe([crop_for_embedding(fr.rgb, t.box, pad=0.15) for t in todo], [t.tube_id for t in todo])
                writer_ms.append(getattr(vlm_writer, "last_ms", 0.0))
                if res is not None:
                    by_tube = {c.tube_id: c.attributes for c in res.cells}
                    for t in todo:
                        if t.tube_id in by_tube and by_tube[t.tube_id].confidence > 0:
                            t.attributes = by_tube[t.tube_id]
                            print(f"t={fr.pts_ms:7d}  describe               {t.tube_id}: {t.attributes.description}"
                                  + (f" | top {t.attributes.top_color.value}" if t.attributes.top_color else ""))
                for t in todo:
                    described.add(t.tube_id)
        persons = [t for t in live if t.class_label == "person" and t.state.value in ("active", "born")]
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                ti, tj = persons[i], persons[j]
                iou = ti.box.iou(tj.box)
                if iou >= 0.5 and len(duplicate_pairs) < 12:
                    dets_here = [{"origin": d.origin, "conf": round(d.confidence, 2), "box": [round(v) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)],
                                  "iou_i": round(d.box.iou(ti.box), 2), "iou_j": round(d.box.iou(tj.box), 2)}
                                 for d in dets if d.class_label == "person" and (d.box.iou(ti.box) > 0.1 or d.box.iou(tj.box) > 0.1)]
                    duplicate_pairs.append({"t_ms": fr.pts_ms, "hb": det_source == "heartbeat", "iou": round(iou, 2),
                                            "a": {"id": ti.tube_id, "origin": origin_of.get(ti.tube_id), "state": ti.state.value,
                                                  "box": [round(v) for v in (ti.box.x1, ti.box.y1, ti.box.x2, ti.box.y2)]},
                                            "b": {"id": tj.tube_id, "origin": origin_of.get(tj.tube_id), "state": tj.state.value,
                                                  "box": [round(v) for v in (tj.box.x1, tj.box.y1, tj.box.x2, tj.box.y2)]},
                                            "dets_on_tick": dets_here})
                    if len(dup_frames) < 4:
                        p = annotate(fr.rgb, rois, dets, live, f"DUPLICATE t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                                     Path(a.out).parent / "debug" / ep / f"dup_{frames:04d}.jpg")
                        if p:
                            dup_frames.append(str(p))
        for t in closed:
            if t.class_label == "person":
                recent_dead.append((fr.pts_ms, (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2, t.box.height))
        recent_dead = [r for r in recent_dead if fr.pts_ms - r[0] <= 1500]
        if debug_every and frames % debug_every == 0 and len(debug_paths) < a.debug_frames:
            p = annotate(fr.rgb, rois, dets, live, f"t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                         Path(a.out).parent / "debug" / ep / f"tick_{frames:04d}.jpg")
            if p:
                debug_paths.append(str(p))
        concurrent_persons.append(sum(1 for t in live if t.class_label == "person" and t.state.value == "active"))
        state_ticks.update(t.state.value for t in live if t.class_label == "person")
        births += max(0, len(live) + len(closed) - before) if closed else max(0, len(live) - before)
        closed_all += closed
        stage["track"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        snaps = [TubeSnapshot(tube_id=t.tube_id, class_label=t.class_label, state=t.state, box=t.box,
                              det_source="detector" if t.state.value == "active" else "predicted") for t in live]
        events += compiler.on_tick(snaps, fr.pts_ms)
        tick = Tick(camera_id=a.camera, tile_id=a.tile, tick_index=frames, t_start=CamTime(cam_utc_ms=fr.pts_ms),
                    t_end=CamTime(cam_utc_ms=fr.pts_ms + tick_ms), tubes=snaps,
                    event_ids=[e.event_id for e in events], gate_energy=sum(b.energy for b in g.blobs), provenance=prov)
        writer.write_tick(ep, tick)
        for e in events:
            writer.write_event(ep, e)
            ev_types[e.type.value] += 1
            print(f"t={fr.pts_ms:7d}  {e.type.value:22s} zone={e.zone_id} subjects={e.subject_tube_ids}")
        stage["compile"] += time.perf_counter() - t0
        frames += 1

    if frames == 0:
        raise SystemExit("no frames read; check --source")
    tubes = [tr.tube for tr in tracker._tracks.values()] + closed_all
    if linker is not None:
        for t in tubes:
            if t.class_label == "person":
                t.entity_id = linker.entity_of(t.tube_id) or t.entity_id
    cast = [CastMember(tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0])
            for t in tubes]
    heights = sorted(t.max_height_px for t in tubes if t.class_label == "person" and t.max_height_px > 0)
    median_h = heights[len(heights) // 2] if heights else None
    for t in tubes:
        grade_tube(t, w, h, median_height_px=median_h, min_life_ms=int(profile.quality.min_life_s * 1000),
                   min_height_frac=profile.quality.min_height_frac, border_px=profile.quality.border_px)
        writer.write_tube(ep, t)
    writer.close(ep, CamTime(cam_utc_ms=fr.pts_ms + tick_ms), EpisodeStatus.closed, cast)
    st = reader.stats
    row = {
        "ring": "slice", "source": Path(a.source).name, "model": f"rf-detr-{a.model}", "sampled_fps": a.fps,
        "tracker": a.tracker, "detect": a.detect, "det_threshold": a.threshold, "heartbeats": heartbeats,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items() if k != "detect"},
        "detect_ms_p50": round(float(np.median(detect_ms)), 2) if detect_ms else None,
        "detect_ms_p95": round(float(np.percentile(detect_ms, 95)), 2) if detect_ms else None,
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
        "person_tubes": sum(1 for t in tubes if t.class_label == "person"),
        "person_tubes_low_quality": sum(1 for t in tubes if t.class_label == "person" and t.quality == "low"),
        "person_dets_per_frame": round(float(np.mean(person_dets)), 2) if person_dets else 0,
        "max_concurrent_persons": max(concurrent_persons) if concurrent_persons else 0,
        "person_visibility_duty": round(state_ticks["active"] / max(1, state_ticks["active"] + state_ticks["occluded"]), 3),
        "fragmentation_est": round(sum(1 for t in tubes if t.class_label == "person") / max(1.0, float(np.mean(person_dets))), 2) if person_dets else None,
        "reid": embedder.name if embedder else "none",
        "profile": profile.name, "median_person_height_px": median_h,
        "writer": a.writer, "writer_calls": getattr(vlm_writer, "calls", 0) if vlm_writer else 0,
        "writer_ms_p50": round(float(np.median(writer_ms)), 1) if writer_ms else None,
        "tubes_described": sum(1 for t in tubes if t.attributes is not None),
        "confirmed_entities": len({t.entity_id or t.tube_id for t in tubes if t.class_label == "person" and t.quality == "ok"}),
        "entities": linker.entities if linker else None, "relinks": linker.relinks if linker else None,
        "merges_on_death": linker.merges if linker else None,
        "ghosts_absorbed": absorbed_total if linker else None,
        "entities_per_concurrent": round(linker.entities / max(1, max(concurrent_persons) if concurrent_persons else 1), 2) if linker else None,
        "embed_ms_p50": round(float(np.median(embed_ms)), 2) if embed_ms else None,
        "births_by_origin": dict(births_by_origin), "confirmed_by_origin": dict(confirmed_by_origin),
        "rebirths": rebirths, "duplicate_pairs": duplicate_pairs, "duplicate_frames": dup_frames,
        "debug_frames": debug_paths,
        "tubes_per_concurrent": round(sum(1 for t in tubes if t.class_label == "person") / max(1, max(concurrent_persons) if concurrent_persons else 1), 2),
        "mean_person_tube_life_s": round(float(np.mean([(t.last_seen.corrected_ms() - t.born.corrected_ms()) / 1000
                                                        for t in tubes if t.class_label == "person"] or [0])), 2),
        "merge_candidates_flagged": sum(1 for t in tubes if t.merge_candidates),
        "tubes_by_final_state": dict(Counter(t.state.value for t in tubes)),
        "classes": dict(Counter(t.class_label for t in tubes)),
        "events": dict(ev_types), "keyframes_saved": kf.saved, "episode_id": ep,
        "episode_records": sum(1 for _ in EpisodeWriter.read(writer.path(ep))),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "slice_gpu.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}" + (f"\ndebug frames -> {Path(a.out).parent / 'debug' / ep}" if debug_paths else ""))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/metamorphic.py << 'EOF_VI'
"""Metamorphic tests without labels: the same clip under perturbations that must not change what
the system says. Invariants: confirmed-people count within ±1, no new event types, the whole-time
set stable. One row per variant to data/bench/metamorphic.jsonl; exit code 1 if any invariant fails.

  python bench/metamorphic.py --source /content/HI_DEF_VIDEO.mp4 --variants brightness_up,brightness_down,res540,fps6,hflip
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VARIANTS = ["brightness_up", "brightness_down", "res540", "fps6", "hflip", "shift2s"]


def make_variant(src: str, name: str, out_dir: Path) -> tuple[Path, dict]:
    """Re-encode the clip with one perturbation. fps6 and shift2s are sampling changes handled by
    slice args; the pixel variants are written with PyAV."""
    import av
    out = out_dir / f"{Path(src).stem}_{name}.mp4"
    args: dict = {}
    if name == "fps6":
        return Path(src), {"fps": 6}
    if name == "shift2s":
        return Path(src), {"skip_s": 2}
    if out.exists():
        return out, args
    cin = av.open(src); sin = cin.streams.video[0]
    cout = av.open(str(out), "w")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    w, h = sin.width, sin.height
    if name == "res540":
        w, h = (w * 540 // h) // 2 * 2, 540
    sout = cout.add_stream(codec, rate=int(sin.average_rate) if sin.average_rate else 6)
    sout.width, sout.height, sout.pix_fmt = w, h, "yuv420p"
    for fr in cin.decode(sin):
        img = fr.to_ndarray(format="rgb24").astype(np.float32)
        if name == "brightness_up":
            img = np.clip(img * 1.2, 0, 255)
        elif name == "brightness_down":
            img = np.clip(img * 0.8, 0, 255)
        elif name == "hflip":
            img = img[:, ::-1, :]
        img = img.astype(np.uint8)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        if name == "res540":
            vf = vf.reformat(width=w, height=h)
        for pkt in sout.encode(vf):
            cout.mux(pkt)
    for pkt in sout.encode():
        cout.mux(pkt)
    cout.close(); cin.close()
    return out, args


def run_slice(source: Path, out_dir: Path, base_args: list[str], fps: float, skip_s: float, zones: str | None) -> dict:
    cmd = [sys.executable, "bench/slice_gpu.py", "--source", str(source), "--out", str(out_dir / "episodes"), "--fps", str(fps)] + base_args
    if zones:
        cmd += ["--zones", zones]
    if skip_s:
        cmd += ["--skip-s", str(skip_s)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])
    txt = p.stdout
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--zones", default=None)
    ap.add_argument("--work", default="data/metamorphic")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("slice_args", nargs="*", help="extra args passed to slice_gpu.py after --")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    base = run_slice(Path(a.source), work / "base", a.slice_args, a.fps, 0, a.zones)
    base_people = base.get("confirmed_entities", base.get("person_tubes", 0) - base.get("person_tubes_low_quality", 0))
    rows, failures = [], []
    print(f"base: confirmed people {base_people}, entities {base.get('entities')}, events {sorted(base['events'])}")
    for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            src, extra = make_variant(a.source, v, work)
            r = run_slice(src, work / v, a.slice_args, extra.get("fps", a.fps), extra.get("skip_s", 0), a.zones)
        except Exception as e:
            print(f"  {v:15s} ERROR {str(e)[:120]}"); failures.append(v); continue
        people = r.get("confirmed_entities", r.get("person_tubes", 0) - r.get("person_tubes_low_quality", 0))
        new_events = sorted(set(r["events"]) - set(base["events"]))
        ok = abs(people - base_people) <= a.tolerance and not new_events
        rows.append({"variant": v, "people": people, "entities": r.get("entities"), "new_event_types": new_events, "pass": ok})
        print(f"  {v:15s} people {people:2d} (base {base_people}) entities {r.get('entities')} new events {new_events or '-'}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(v)
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "metamorphic.jsonl").open("a") as f:
        f.write(json.dumps({"ring": "metamorphic", "source": Path(a.source).name, "base_people": base_people, "rows": rows,
                            "failures": failures, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    print(f"{len(rows) - len([r for r in rows if not r['pass']])}/{len(rows)} invariants hold" + (f"; failed: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/scenario_eval.py << 'EOF_VI'
"""Score the agent against a ground-truth scenario file. Every question gets pass/fail on the
mention rules, the citation rules and the latency budget; the row goes to data/bench/scenario.jsonl.

  python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend openai
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask, search_events
from vi.agent.loop import ID_RE
from vi.store import connect


import re


def _one(r, text: str) -> bool:
    r = str(r).lower()
    if r.isdigit():                       # a number must stand alone: not part of 00:14.0, E14 or 140
        return re.search(rf"(?<![\d:.\w]){re.escape(r)}(?![\d:.\w])", text) is not None
    return r in text


def _mentioned(rule, text: str) -> bool:
    """a string must appear; a list means any of its strings must appear"""
    if isinstance(rule, list):
        return any(_one(r, text) for r in rule)
    return _one(rule, text)


def score(res: dict, spec: dict, budget_ms: int, engine=None, episode_id: str | None = None) -> dict:
    f = res["final"]
    text = ID_RE.sub(" ", f.get("text") or f.get("question") or "").lower()    # ids are not numbers
    whole_ok = True
    if spec.get("whole_time_recall") and engine is not None and episode_id:
        from vi.agent import entities_present
        truth = set(entities_present(engine, episode_id, 0.9)["entity_ids"])
        cited = {c for c in f.get("citations", []) if ":E" in c}
        whole_ok = (len(cited & truth) / len(truth) >= spec["whole_time_recall"]) if truth else True
    gt_ok = True
    if spec.get("whole_time_count") and f.get("citations"):
        lo, hi = spec["whole_time_count"]
        n = len({c for c in f["citations"] if ":E" in c})
        gt_ok = lo <= n <= hi
    checks = {
        "whole_time": whole_ok,
        "whole_time_gt": gt_ok,
        "answered": f["action"] == "answer",
        "mentions": all(_mentioned(m, text) for m in spec.get("must_mention", []) or []),
        "avoids": not any(_one(m, text) for m in spec.get("must_not_mention", []) or []),
        "cites": (f.get("cited", False) or spec.get("expect_uncited_ok", False))
                 and (not spec.get("must_cite_prefix") or any(any(c.startswith(p) for p in spec["must_cite_prefix"]) for c in f.get("citations", []))),
        "in_budget": res["latency"]["total_ms"] <= budget_ms,
    }
    return {"q": spec["q"], "pass": all(checks.values()), "checks": checks, "latency_ms": res["latency"]["total_ms"],
            "answer": f.get("text", f.get("question", ""))[:300], "citations": f.get("citations", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--db", required=True)
    ap.add_argument("--episode", default=None, help="default: the latest episode in the store")
    ap.add_argument("--backend", choices=["fake", "openai", "transformers"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.scenario).read_text())
    engine = connect(a.db)
    ep = a.episode or sorted({e["episode_id"] for e in search_events(engine, limit=10000)})[-1]
    backend = {"fake": lambda: FakeBackend(), "openai": lambda: OpenAIBackend(base_url=a.base_url, model=a.model),
               "transformers": lambda: TransformersBackend(model_id=a.model)}[a.backend]()
    budget = int(spec.get("latency_budget_ms", 10000))
    results = [score(ask(engine, q["q"], ep, backend), q, budget, engine, ep) for q in spec["questions"]]
    unscored = spec.get("ground_truth", {}).get("people_total") is None
    row = {"ring": "scenario", "scenario": Path(a.scenario).name, "episode_id": ep, "backend": a.backend, "model": a.model,
           "passed": sum(r["pass"] for r in results), "total": len(results), "ground_truth_filled": not unscored,
           "latency_p50_ms": sorted(r["latency_ms"] for r in results)[len(results) // 2],
           "latency_max_ms": max(r["latency_ms"] for r in results), "results": results,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "scenario.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    for r in results:
        flags = " ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["checks"].items())
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['latency_ms'] / 1000:5.1f}s  {r['q'][:70]}\n         {flags}\n         {r['answer'][:160]}")
    print(f"  {row['passed']}/{row['total']} passed, p50 {row['latency_p50_ms'] / 1000:.1f}s, max {row['latency_max_ms'] / 1000:.1f}s"
          + ("  (ground truth not filled in yet: mention rules are placeholders)" if unscored else ""))


if __name__ == "__main__":
    main()
EOF_VI
cat > tests/scenarios/warehouse.yaml << 'EOF_VI'
# Acceptance scenario for the warehouse clip (HI_DEF_VIDEO.mp4, 17 s, 6 fps, 1280x720).
# Ground truth established by watching the clip (session 16b). bench/scenario_eval.py scores the
# agent against it; a mention rule given as a list means "any of these".
clip: HI_DEF_VIDEO.mp4
ground_truth:
  people_total: 8
  people_whole_time: 6
  people_at_table: 5
  people_right_side: 3
  people_at_conveyor: 2
  pickups: 0
  notes: |
    Table, whole clip: dark jacket + orange vest (near-left); orange vest, black hair (near-left);
    woman, orange vest over dark top (far-left end); white hard hat + orange vest + blue jeans
    (far side, moves right behind boxes ~6 s); pink/dark cap + orange vest (right of table at 0 s,
    far-middle by 8 s). Right side: yellow-green hi-vis + white hard hat at the conveyor 0-17 s;
    second hi-vis worker, dark cap, beside him at 0 s, walks along the conveyor ~8 s; dark jacket +
    orange vest walking boxes in the dock aisle top-right, first visible ~4 s, in and out after.
    Hanging jackets on the far-left racking are NOT people (false "person" detections there).
questions:
  - q: How many distinct people were in this episode, and which of them stayed the whole time?
    must_mention: [["8", "eight", "7", "seven", "9", "nine"]]     # 8, ±1 tolerance while ReID matures
    must_not_mention: ["14", "13", "12", "11"]
    must_cite_prefix: ["cam1:E", "ev_"]
    whole_time_recall: 0.8           # the cited entities must cover >= 80% of entities_present(min_coverage=0.9)
    whole_time_count: [5, 7]         # ground truth: 6 people were present the whole clip (±1)
  - q: Who was at the conveyor on the right, and when did they arrive?
    must_mention: [["00:0", "start", "beginning", "0.0", "from the first", "already"]]   # the hard-hat worker is there from 0 s
    must_not_mention: []
    must_cite_prefix: ["cam1:E", "ev_"]      # any of: an entity id or the enter_zone event id
    later: ["hard hat", "hi-vis"]       # becomes must_mention once the writer VLM supplies attributes
  - q: Did anyone pick something up from a shelf?
    must_mention: [["no", "none", "nothing", "did not", "didn't", "not"]]
    must_not_mention: ["yes, "]
    must_cite_prefix: []
    expect_uncited_ok: true
latency_budget_ms: 10000
EOF_VI
cat > ui/app.py << 'EOF_VI'
"""Temporary question interface over the live store. Runs in Colab (share link) or locally.

  python ui/app.py --db "$DB_URL" --backend transformers --model Qwen/Qwen3.5-4B --tz Asia/Kolkata --share

Left: footage status, chat. Right: evidence keyframes for the cited entities, the trace, and the
episode list. Every answer shows its time window, latency and citations; refusals (future time,
off-topic, actions) come from the deterministic guards, not the model.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gradio as gr

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask_window, clip, episodes_in, footage_bounds
from vi.store import connect


def build(db_url: str, backend_name: str, model: str, tz_name: str, keyframes_dir: str = "data/keyframes"):
    engine = connect(db_url)
    tz = ZoneInfo(tz_name)
    if backend_name == "fake":
        backend = FakeBackend()
    elif backend_name == "openai":
        backend = OpenAIBackend(model=model)
    else:
        backend = TransformersBackend(model_id=model)

    def clock(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%Y-%m-%d %H:%M:%S")

    def status() -> str:
        b = footage_bounds(engine)
        if b is None:
            return "**No footage processed yet.** Start `bench/run_ingest.py` and ask again."
        eps = episodes_in(engine, b[0], b[1])
        return (f"**Footage** {clock(b[0])} → {clock(b[1])} ({tz_name})  ·  **episodes** {len(eps)}  ·  "
                f"**now** {clock(b[1])}  ·  backend `{getattr(backend, 'name', backend_name)}` `{model}`")

    def episode_rows() -> list[list]:
        b = footage_bounds(engine)
        if b is None:
            return []
        return [[e["episode_id"][:12], clock(e["t0_ms"]), clock(e["t1_ms"] or e["t0_ms"]), e["status"], e["tile_id"]] for e in episodes_in(engine, b[0], b[1])][-25:]

    def keyframes_for(citations: list[str]) -> list:
        ims = []
        for c in citations:
            if ":E" not in c and "anon:" not in c:
                continue
            try:
                ev = clip(engine, entity_id=c)
            except Exception:
                continue
            for ref in ev.get("keyframe_refs", [])[:2]:
                p = Path(keyframes_dir) / ref.replace("kf://", "")
                if p.exists():
                    ims.append((str(p), c))
        return ims[:12]

    def ask(question: str, history: list):
        history = history or []
        if not question.strip():
            return history, "", [], {}, status(), episode_rows()
        b = footage_bounds(engine)
        now_ms = b[1] if b else int(time.time() * 1000)
        t0 = time.perf_counter()
        r = ask_window(engine, question, backend, now_ms, tz_name)
        f = r["final"]
        text = f.get("text") or f.get("question") or ""
        meta = []
        if r.get("window"):
            meta.append(f"window {r['window']}")
        meta.append(f"{r['latency']['total_ms'] / 1000:.1f}s")
        if f.get("citations"):
            meta.append("cites " + ", ".join(f["citations"][:8]))
        if f.get("handled_by"):
            meta.append(f"guard: {r.get('grounding')}")
        if f.get("numeric_issue"):
            meta.append("numbers checked")
        answer = text + "\n\n" + " · ".join(f"`{m}`" for m in meta)
        history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        trace = {"grounding": r.get("grounding"), "window": r.get("window"), "steps": r.get("steps"), "trace": r.get("trace", []),
                 "latency": r.get("latency"), "citations": f.get("citations", []), "rejected": f.get("rejected_citations", [])}
        return history, "", keyframes_for(f.get("citations", [])), trace, status(), episode_rows()

    with gr.Blocks(title="vi-engine") as demo:
        gr.Markdown("## vi-engine — ask the footage")
        st = gr.Markdown(status())
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Answers", height=460)
                q = gr.Textbox(label="Question", placeholder="Who was at the conveyor between 10:10 and 10:20? · How many people in the last 15 minutes? · What was the person in the white hat carrying?")
                with gr.Row():
                    btn = gr.Button("Ask", variant="primary")
                    refresh = gr.Button("Refresh status")
            with gr.Column(scale=2):
                gal = gr.Gallery(label="Evidence (keyframes of cited entities)", columns=3, height=300)
                tr = gr.JSON(label="Trace")
                eps = gr.Dataframe(headers=["episode", "start", "end", "status", "tile"], value=episode_rows(), label="Episodes", interactive=False)
        btn.click(ask, [q, chat], [chat, q, gal, tr, st, eps])
        q.submit(ask, [q, chat], [chat, q, gal, tr, st, eps])
        refresh.click(lambda: (status(), episode_rows()), None, [st, eps])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--backend", choices=["fake", "transformers", "openai"], default="transformers")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tz", default="UTC")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo = build(a.db, a.backend, a.model, a.tz)
    demo.launch(share=a.share, server_port=a.port, server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    main()
EOF_VI
cat > vi/api/__init__.py << 'EOF_VI'
from .server import create_app
EOF_VI
cat > vi/api/server.py << 'EOF_VI'
"""HTTP API for the engine: serves the static frontend, answers questions over the live store,
exposes footage state, keyframes and the live frame, and starts/stops the ingest as a paced
live stream. One process; the agent model loads once at startup.

  uvicorn vi.api.server:app --host 0.0.0.0 --port 8000
  env: VI_DB, VI_BACKEND (transformers|fake|openai), VI_MODEL, VI_TZ, VI_WEB (static dir), VI_KEYFRAMES, VI_LIVE
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vi.agent import ask_window, clip, episodes_in, footage_bounds, search_events
from vi.store import connect

GREETING = ("hi", "hey", "hello", "yo", "hii", "helo")


class AskIn(BaseModel):
    question: str
    mode: str | None = None
    history: list[dict] | None = None      # previous exchanges [{"q","a","action"}], most recent last


class IngestIn(BaseModel):
    source: str
    start_time: str | None = None
    realtime: bool = True
    profile: str = "common"
    writer: str = "none"
    reid: str = "siglip"
    model: str = "nano"
    fps: float = 4.0
    episode_min: float = 10.0
    grid: str | None = "auto"        # "auto" detects an NVR multiplex layout from the seams; "2x2"/"4x4" to force; null = single camera
    max_width: int = 0


def create_app(db_url: str | None = None, backend_name: str | None = None, model: str | None = None,
               tz_name: str | None = None, web_dir: str | None = None, keyframes_dir: str | None = None,
               live_dir: str | None = None, load_backend: bool = True) -> FastAPI:
    db_url = db_url or os.environ.get("VI_DB", "sqlite+pysqlite:///data/vi.db")
    backend_name = backend_name or os.environ.get("VI_BACKEND", "transformers")
    model = model or os.environ.get("VI_MODEL", "Qwen/Qwen3.5-4B")
    tz_name = tz_name or os.environ.get("VI_TZ", "UTC")
    web_dir = Path(web_dir or os.environ.get("VI_WEB", "ui/web"))
    keyframes_dir = Path(keyframes_dir or os.environ.get("VI_KEYFRAMES", "data/keyframes"))
    live_dir = Path(live_dir or os.environ.get("VI_LIVE", "data/live"))
    tz = ZoneInfo(tz_name)
    engine = connect(db_url)
    app = FastAPI(title="vi-engine")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    state = {"backend": None, "ingest": None, "started": time.time(), "model": model, "backend_name": backend_name}

    def backend():
        if state["backend"] is None:
            if not load_backend or backend_name == "fake":
                from vi.agent import FakeBackend
                state["backend"] = FakeBackend()
            elif backend_name == "openai":
                from vi.agent import OpenAIBackend
                state["backend"] = OpenAIBackend(model=model)
            else:
                from vi.agent import TransformersBackend
                state["backend"] = TransformersBackend(model_id=model)
        return state["backend"]

    def clock(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%H:%M:%S")

    def ingest_status() -> dict:
        p = state["ingest"]
        running = p is not None and p.poll() is None
        st = {}
        try:
            st = json.loads((live_dir / "status.json").read_text())
        except Exception:
            pass
        return {"running": running, "pid": p.pid if running else None, "exit_code": (p.returncode if p is not None and not running else None), **st}

    @app.get("/health")
    def health():
        b = footage_bounds(engine)
        eps = episodes_in(engine, b[0], b[1]) if b else []
        return {"ok": True, "backend": backend_name, "model": model, "tz": tz_name, "uptime_s": round(time.time() - state["started"]),
                "footage": {"start": clock(b[0]), "end": clock(b[1]), "start_ms": b[0], "end_ms": b[1]} if b else None,
                "episodes": len(eps), "ingest": ingest_status()}

    @app.post("/ask")
    def ask(inp: AskIn):
        q = inp.question.strip()
        if not q:
            raise HTTPException(400, "empty question")
        if q.lower().rstrip("!?. ") in GREETING or q.lower().startswith(("hi ", "hey ", "hello ")):
            return {"text": "Hey — ask me about the cameras: who was where, when, what they were carrying, how many people, or what happened in a time window.",
                    "mood": "bot", "evidence": [], "grounding": "greeting", "latency_ms": 0}
        b = footage_bounds(engine)
        now_ms = b[1] if b else int(time.time() * 1000)
        import concurrent.futures
        deadline = float(os.environ.get("VI_ASK_DEADLINE_S", "80"))     # under Cloudflare's 100 s origin limit
        pool = state.setdefault("pool", concurrent.futures.ThreadPoolExecutor(max_workers=1))
        fut = pool.submit(ask_window, engine, q, backend(), now_ms, tz_name, 6, (inp.history or [])[-3:])
        try:
            r = fut.result(timeout=deadline)
        except concurrent.futures.TimeoutError:
            return {"text": f"That took longer than {int(deadline)} s, probably because the engine is busy ingesting. "
                            "Ask a narrower question (a camera or a time window), or try again in a moment.",
                    "mood": "botUnsure", "citations": [], "evidence": [], "grounding": "timeout", "latency_ms": int(deadline * 1000)}
        f = r["final"]
        text = f.get("text") or f.get("question") or ""
        unsure = f.get("action") == "clarify" or f.get("handled_by") in ("scope", "time") or not f.get("cited", False)
        evidence = []
        for c in f.get("citations", [])[:6]:
            if ":E" not in c and "anon:" not in c:
                continue
            try:
                ev = clip(engine, entity_id=c)
            except Exception:
                continue
            for ref in ev.get("keyframe_refs", [])[:1]:
                rel = ref.replace("kf://", "")
                if (keyframes_dir / rel).exists():
                    evidence.append({"entity_id": c, "url": f"/keyframes/{rel}", "label": c.split(":")[-1]})
        return {"text": text, "mood": "botUnsure" if unsure else "bot", "citations": f.get("citations", []), "evidence": evidence,
                "window": r.get("window"), "grounding": r.get("grounding"), "latency_ms": r.get("latency", {}).get("total_ms"),
                "action": f.get("action", "answer"), "trace": r.get("trace", [])}

    @app.get("/episodes")
    def episodes():
        b = footage_bounds(engine)
        if not b:
            return []
        return [{"episode_id": e["episode_id"], "start": clock(e["t0_ms"]), "end": clock(e["t1_ms"] or e["t0_ms"]), "status": e["status"], "tile": e["tile_id"]}
                for e in episodes_in(engine, b[0], b[1])]

    @app.get("/events")
    def events(t_start_ms: int | None = None, t_end_ms: int | None = None, limit: int = 200):
        b = footage_bounds(engine)
        if not b:
            return []
        rows = search_events(engine, t_start_ms=t_start_ms or b[0], t_end_ms=t_end_ms or b[1], limit=limit)
        return [{"t": clock(e["t_ms"]), "type": e["type"], "zone": e["zone_id"], "subjects": e["subject_tube_ids"], "id": e["event_id"]} for e in rows]

    @app.get("/keyframes/{path:path}")
    def keyframe(path: str):
        p = (keyframes_dir / path).resolve()
        if not str(p).startswith(str(keyframes_dir.resolve())) or not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p))

    @app.get("/live/latest.jpg")
    def live_frame():
        p = live_dir / "latest.jpg"
        if not p.exists():
            raise HTTPException(404, "no live frame yet")
        return FileResponse(str(p), headers={"Cache-Control": "no-store"})

    @app.post("/ingest/start")
    def ingest_start(inp: IngestIn):
        if state["ingest"] is not None and state["ingest"].poll() is None:
            return {"started": False, "reason": "already running", **ingest_status()}
        cmd = [sys.executable, "bench/run_ingest.py", "--source", inp.source, "--db", db_url, "--profile", inp.profile, "--reid", inp.reid,
               "--writer", inp.writer, "--model", inp.model, "--fps", str(inp.fps), "--episode-min", str(inp.episode_min), "--live-dir", str(live_dir)]
        if inp.start_time:
            cmd += ["--start-time", inp.start_time]
        if inp.grid:
            cmd += ["--grid", inp.grid]
        if inp.max_width:
            cmd += ["--max-width", str(inp.max_width)]
        if inp.realtime:
            cmd.append("--realtime")
        live_dir.mkdir(parents=True, exist_ok=True)
        log = open(live_dir / "ingest.log", "a")
        state["ingest"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
        return {"started": True, "pid": state["ingest"].pid, "cmd": " ".join(cmd)}

    @app.post("/ingest/stop")
    def ingest_stop():
        p = state["ingest"]
        if p is not None and p.poll() is None:
            p.terminate()
            return {"stopped": True}
        return {"stopped": False, "reason": "not running"}

    @app.get("/ingest/status")
    def ingest_state():
        return ingest_status()

    tiles_path = Path(os.environ.get("VI_TILES", "data/tiles.json"))

    @app.get("/tiles")
    def tiles():
        from vi.fusion import TileMap
        tm = TileMap.load(tiles_path)
        return {"path": str(tiles_path), "tiles": tm.tiles if tm else None, "affinity": tm.affinity if tm else None}

    @app.post("/tiles/recompute")
    def tiles_recompute():
        """Learn which cameras see the same area from the entities recorded so far, and save the map.
        The next ingest start uses it (one identity per tile instead of per camera)."""
        from vi.fusion import discover_tiles
        from vi.fusion.tiles import entities_for_affinity
        from vi.agent.tools import cameras_in_store
        ents = entities_for_affinity(engine)
        tm = discover_tiles(ents, cameras=cameras_in_store(engine))
        tm.save(tiles_path)
        return {"tiles": tm.tiles, "affinity": tm.affinity, "entities_used": len(ents), "saved": str(tiles_path),
                "note": "restart the ingest (POST /ingest/stop then /ingest/start) to apply"}

    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


app = create_app(load_backend=os.environ.get("VI_LAZY_BACKEND", "1") != "0")
EOF_VI
cat > ui/web/index.html << 'EOF_VI'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Tracer</title>
<style>
@font-face{font-family:"LINE Seed JP";font-weight:400;font-style:normal;font-display:swap;src:url(fonts/LINESeedJP-400.woff2) format("woff2")}
@font-face{font-family:"LINE Seed JP";font-weight:700;font-style:normal;font-display:swap;src:url(fonts/LINESeedJP-700.woff2) format("woff2")}
@font-face{font-family:"LINE Seed JP";font-weight:800;font-style:normal;font-display:swap;src:url(fonts/LINESeedJP-800.woff2) format("woff2")}

:root{
  --bg:#FFFFFF; --ink:#000000; --field:#F2F2F2;
  --green:#5CB85C; --green-soft:rgba(92,184,92,.55); --green-mist:rgba(92,184,92,.18);
  --stroke:0.5px;
  --font:"LINE Seed JP","Helvetica Neue",Arial,sans-serif;
  --u:min(calc(100vw / 1728), 1px);
  --ease:cubic-bezier(.2,.8,.2,1);
  --t:560ms;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){ --bg:#FFFFFF; --ink:#000000; --field:#F2F2F2; } }
:root[data-theme="dark"]{ --bg:#FFFFFF; --ink:#000000; --field:#F2F2F2; }
html{scroll-padding-top:env(safe-area-inset-top,0px);height:100%}
*,*::before,*::after{box-sizing:inherit}
body{margin:0;height:100%;background:var(--bg);color:var(--ink);font-family:var(--font);-webkit-font-smoothing:antialiased;overflow:hidden}
.page{position:relative;width:100%;height:100%}

/* one shared soft-green halo, used for selection, listening, and keyboard focus */
.halo::before{
  content:"";position:absolute;inset:-95%;border-radius:50%;z-index:-1;pointer-events:none;
  background:radial-gradient(circle,rgba(92,184,92,.6) 0%,rgba(92,184,92,.42) 34%,rgba(92,184,92,.22) 52%,rgba(92,184,92,.08) 72%,rgba(92,184,92,0) 92%);
  opacity:0;transform:scale(.45);
  transition:opacity 480ms var(--ease),transform 620ms var(--ease);
}
.halo{position:relative;isolation:isolate}
.halo.on::before,.halo:has(:focus-visible)::before{opacity:1;transform:scale(1)}
@keyframes breathe{0%,100%{transform:scale(.92)}50%{transform:scale(1.12)}}

/* ---------- entrance (runs once .ready is set) ---------- */
@keyframes fadeUp{from{opacity:0;transform:translateY(calc(14 * var(--u)))}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:none}}
.logo,.avatar,.modes li,.ask textarea,.mic-wrap{opacity:0}
.page.ready .logo{animation:fadeUp 720ms var(--ease) 60ms forwards}
.page.ready .avatar{animation:pop 720ms var(--ease) 180ms forwards}
.page.ready .ask textarea,.page.ready .mic-wrap{animation:fadeUp 700ms var(--ease) 620ms forwards}
.page.ready .modes li{animation:pop 640ms var(--ease) forwards}
.page.ready .modes li:nth-child(1){animation-delay:820ms}
.page.ready .modes li:nth-child(2){animation-delay:920ms}
.page.ready .modes li:nth-child(3){animation-delay:1020ms}

/* ---------- header ---------- */
.logo{
  position:absolute;left:calc(89 * var(--u));top:calc(92 * var(--u));
  margin:0;font-weight:700;font-size:calc(64 * var(--u));line-height:1;letter-spacing:-0.09em;
  text-decoration:none;color:var(--ink);
}
.avatar{
  position:absolute;right:calc(89 * var(--u));top:calc(80 * var(--u));
  width:calc(84 * var(--u));height:calc(84 * var(--u));border-radius:50%;
  padding:0;border:0;background:var(--ink);overflow:hidden;cursor:pointer;
  transition:transform 320ms var(--ease);
}
.avatar:hover{transform:scale(1.06)}
.avatar img,.modes img,.row-avatar img{width:100%;height:100%;display:block}
.avatar img{object-fit:cover}

/* ---------- source icons ---------- */
.modes{
  position:absolute;left:50%;transform:translateX(-50%);
  top:calc(676 * var(--u));
  display:flex;gap:calc(34 * var(--u));margin:0;padding:0;list-style:none;
}
.modes li{width:calc(70 * var(--u));height:calc(70 * var(--u))}
.modes button{
  width:100%;height:100%;border-radius:50%;
  padding:0;border:0;background:var(--ink);overflow:hidden;cursor:pointer;outline:none;
  transition:transform 300ms var(--ease);
}
.modes button:hover{transform:translateY(calc(-4 * var(--u))) scale(1.04)}
.modes button:active{transform:scale(.96);transition-duration:120ms}
.modes img{transform:scale(1.15)}
.page[data-state="chat"] .modes{top:calc(176 * var(--u))}

/* ---------- headline ---------- */
.prompt{
  position:absolute;left:0;right:0;top:calc(372 * var(--u));
  margin:0;font-weight:400;font-size:calc(72 * var(--u));line-height:1.2;letter-spacing:-0.05em;
  text-align:center;white-space:nowrap;cursor:text;opacity:0;
  transition:opacity 720ms var(--ease),transform var(--t) var(--ease);
}
.page.ready .prompt{opacity:1}
.prompt strong{font-weight:700}
.caret{
  display:inline-block;width:calc(3 * var(--u));height:0.9em;margin:0 0.02em 0 0.01em;
  background:var(--ink);vertical-align:-0.12em;animation:blink 1s steps(1) infinite;
}
@keyframes blink{50%{opacity:0}}
.page[data-state="chat"] .prompt{opacity:0;transform:translateY(calc(-24 * var(--u)));pointer-events:none;transition-duration:360ms}

/* ---------- ask pill ---------- */
.ask{
  position:absolute;left:50%;top:calc(549 * var(--u));
  width:calc(533 * var(--u));
  transform:translate(-50%,-50%);
  transition:width 200ms var(--ease);
}
.page[data-state="chat"] .ask{top:auto;bottom:calc(64 * var(--u));transform:translateX(-50%)}
.ask textarea{
  display:block;width:100%;min-height:calc(52 * var(--u));max-height:calc(140 * var(--u));
  resize:none;overflow:auto;scrollbar-width:none;
  border:var(--stroke) solid var(--ink);border-radius:calc(26 * var(--u));
  background:var(--field);color:var(--ink);
  font:inherit;font-size:calc(24 * var(--u));line-height:1.3;letter-spacing:-0.05em;
  padding:calc(10 * var(--u)) calc(60 * var(--u)) calc(10 * var(--u)) calc(24 * var(--u));
  outline:none;box-shadow:0 0 0 0 rgba(92,184,92,0);
  transition:height 160ms var(--ease),box-shadow 360ms var(--ease);
}
.ask textarea::-webkit-scrollbar{display:none}
.ask textarea:focus{box-shadow:0 0 0 calc(7 * var(--u)) var(--green-mist),0 0 calc(30 * var(--u)) rgba(92,184,92,.22)}
.mirror{
  position:absolute;left:0;top:0;visibility:hidden;pointer-events:none;white-space:pre;
  font:inherit;font-size:calc(24 * var(--u));line-height:1.3;letter-spacing:-0.05em;
  padding:0 calc(60 * var(--u)) 0 calc(24 * var(--u));
}
.mic-wrap{
  position:absolute;right:calc(5 * var(--u));bottom:calc(5 * var(--u));
  width:calc(42 * var(--u));height:calc(42 * var(--u));
}
.mic{
  width:100%;height:100%;border-radius:50%;
  padding:0;border:var(--stroke) solid var(--ink);background:#fff;overflow:hidden;cursor:pointer;outline:none;
  transition:transform 260ms var(--ease);
}
.mic:hover{transform:scale(1.06)}
.mic img{width:100%;height:100%;display:block;transform:scale(1.35)}
.mic-wrap.on::before{animation:breathe 1.6s ease-in-out infinite}

/* ---------- thread ---------- */
.thread{
  position:absolute;left:50%;transform:translateX(-50%);
  top:calc(300 * var(--u));bottom:calc(170 * var(--u));
  width:calc(740 * var(--u));max-width:calc(100% - 40px);
  margin:0;padding:calc(20 * var(--u)) 0;list-style:none;
  overflow-y:auto;overscroll-behavior:contain;scrollbar-width:none;
  opacity:0;visibility:hidden;transition:opacity var(--t) var(--ease),visibility 0s var(--t);
  -webkit-mask-image:linear-gradient(to bottom,transparent,#000 calc(20 * var(--u)),#000 calc(100% - 20 * var(--u)),transparent);
          mask-image:linear-gradient(to bottom,transparent,#000 calc(20 * var(--u)),#000 calc(100% - 20 * var(--u)),transparent);
}
.thread::-webkit-scrollbar{display:none}
.page[data-state="chat"] .thread{opacity:1;visibility:visible;transition-delay:160ms,0s}

.row{
  display:flex;align-items:center;gap:calc(40 * var(--u));
  margin:0 0 calc(28 * var(--u));
}
.row.bot{opacity:0;transform:translateY(calc(14 * var(--u)));animation:rise 520ms var(--ease) forwards}
@keyframes rise{to{opacity:1;transform:none}}
.row-avatar{
  flex:0 0 auto;width:calc(50 * var(--u));height:calc(50 * var(--u));border-radius:50%;
  overflow:hidden;background:var(--ink);
}
.row.user .row-avatar{animation:pop 460ms var(--ease) 120ms both}
.row-avatar img{transform:scale(1.02)}
.bubble{
  font-size:calc(24 * var(--u));line-height:1.2;letter-spacing:-0.05em;
  padding:calc(12 * var(--u)) 0;max-width:calc(100% - 90 * var(--u));
  overflow-wrap:anywhere;
}
.row.user .bubble{
  background:var(--field);border:var(--stroke) solid var(--ink);border-radius:calc(26 * var(--u));
  padding:calc(12 * var(--u)) calc(24 * var(--u));white-space:pre-wrap;
}
.row.bot .bubble .w{opacity:0;filter:blur(5px);transition:opacity 340ms ease,filter 420ms ease}
.row.bot .bubble .w.on{opacity:1;filter:blur(0)}
.dots{display:inline-flex;gap:calc(6 * var(--u));vertical-align:middle;height:1.2em;align-items:center}
.dots i{width:calc(8 * var(--u));height:calc(8 * var(--u));border-radius:50%;background:var(--ink);opacity:.25;animation:dot 1s ease-in-out infinite}
.dots i:nth-child(2){animation-delay:.16s}.dots i:nth-child(3){animation-delay:.32s}
@keyframes dot{40%{opacity:1;transform:translateY(-2px)}}

.sr{position:absolute;left:-9999px}

/* ---------- narrow screens ---------- */
@media (max-width:760px){
  :root{--u:calc(100vw / 760)}
  .logo{left:20px;top:22px;font-size:36px}
  .avatar{right:16px;top:14px;width:48px;height:48px}
  .modes{top:340px;gap:22px}
  .modes li{width:60px;height:60px}
  .page[data-state="chat"] .modes{top:84px}
  .prompt{top:150px;font-size:40px;white-space:normal;padding:0 20px}
  .ask{top:270px;width:calc(100% - 40px)!important;max-width:none}
  .page[data-state="chat"] .ask{bottom:calc(20px + env(safe-area-inset-bottom,0px))}
  .ask textarea{min-height:48px;max-height:132px;border-radius:24px;padding:11px 54px 11px 18px}
  .ask textarea,.mirror{font-size:17px;letter-spacing:-0.03em}
  .mic-wrap{width:40px;height:40px;right:4px;bottom:4px}
  .thread{top:160px;bottom:90px;width:calc(100% - 32px);padding:12px 0}
  .row{gap:12px;margin-bottom:16px}
  .row-avatar{width:40px;height:40px}
  .bubble{font-size:17px;letter-spacing:-0.03em;padding:10px 0;max-width:calc(100% - 52px)}
  .row.user .bubble{padding:10px 16px;border-radius:24px}
  .dots i{width:6px;height:6px}
}

/* ---------- Tracer backend additions ---------- */
.evidence{display:flex;gap:calc(8 * var(--u));margin:calc(6 * var(--u)) 0 0 calc(56 * var(--u));flex-wrap:wrap}
.evidence figure{margin:0;width:calc(84 * var(--u));text-align:center}
.evidence img{width:100%;height:calc(84 * var(--u));object-fit:cover;border-radius:calc(10 * var(--u));border:var(--stroke) solid var(--ink);background:var(--field)}
.evidence figcaption{font-size:calc(11 * var(--u));opacity:.7;margin-top:calc(3 * var(--u));white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .meta{margin:calc(4 * var(--u)) 0 0 calc(56 * var(--u));font-size:calc(11 * var(--u));opacity:.55}
.live{position:fixed;right:calc(24 * var(--u));bottom:calc(24 * var(--u));width:calc(300 * var(--u));display:none;flex-direction:column;gap:calc(6 * var(--u));z-index:5}
.live.on{display:flex}
.live img{width:100%;border-radius:calc(14 * var(--u));border:var(--stroke) solid var(--ink);background:var(--field);box-shadow:0 0 0 calc(10 * var(--u)) var(--green-mist)}
.live .status{font-size:calc(11 * var(--u));opacity:.7;text-align:right}
@media (max-width:760px){.live{width:calc(180 * var(--u))}}
</style>
</head>
<body>
<main class="page" data-state="hero">
  <a class="logo" href="#" aria-label="Tracer home">Tracer</a>
  <button class="avatar" type="button" aria-label="Account"><img src="assets/avatar.jpg" alt=""></button>

  <ul class="modes" role="group" aria-label="Choose a source">
    <li class="halo"><button type="button" aria-label="Cameras" aria-pressed="false"><img src="assets/cam.png" alt=""></button></li>
    <li class="halo"><button type="button" aria-label="Home" aria-pressed="false"><img src="assets/home.png" alt=""></button></li>
    <li class="halo"><button type="button" aria-label="Help" aria-pressed="false"><img src="assets/help.png" alt=""></button></li>
  </ul>

  <h1 class="prompt" title="Use this question"><span>“</span><span class="pt"></span><span class="caret" aria-hidden="true"></span><span>”</span></h1>

  <ol class="thread" aria-live="polite" aria-label="Conversation"></ol>

  <form class="ask" onsubmit="return false">
    <label for="q" class="sr">Ask a question about your video</label>
    <div class="mirror" aria-hidden="true"></div>
    <textarea id="q" rows="1" autocomplete="off" spellcheck="false"></textarea>
    <span class="mic-wrap halo"><button class="mic" type="button" aria-label="Ask by voice" aria-pressed="false"><img src="assets/mic.png" alt=""></button></span>
  </form>
</main>

<script>
(function(){
  var EASE = 'cubic-bezier(.2,.8,.2,1)';
  var U = function(){ return parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--u')) || 1; };
  var page = document.querySelector('.page');
  var modes = document.querySelector('.modes');
  var ask = document.querySelector('.ask');
  var ta = document.getElementById('q');
  var mirror = document.querySelector('.mirror');
  var thread = document.querySelector('.thread');
  var prompt = document.querySelector('.prompt');
  var pt = prompt.querySelector('.pt');
  var AV = { user:'assets/dp-user.png', bot:'assets/dp-bot.png', botUnsure:'assets/dp-bot-unsure.png' };

  /* ---------- 0. entrance: wait for the font, then run the sequence ---------- */
  var started = false;
  function start(){
    if(started) return; started = true;
    page.classList.add('ready');
    setTimeout(tick, 520);          /* headline starts typing once it has faded in */
  }
  if(document.fonts && document.fonts.ready) document.fonts.ready.then(start);
  setTimeout(start, 1200);          /* never wait longer than this */

  /* ---------- 1. typewriter headline (always on) ---------- */
  var phrases = [
    [['Where’s my ',0],['keys',1],['..?',0]],
    [['Who left the ',0],['door',1],[' open?',0]],
    [['When did the ',0],['dog',1],[' get out?',0]],
    [['Tell me how my ',0],['garden',1],[' looks',0]],
    [['Did the ',0],['parcel',1],[' arrive today?',0]],
    [['Where did I put my ',0],['bag',1],['?',0]]
  ];
  var pi = 0, shown = 0, dir = 1, typing = true, timer = null;
  function total(segs){ return segs.reduce(function(a,s){ return a + s[0].length; }, 0); }
  function render(segs, n){
    pt.textContent = '';
    var left = n;
    segs.forEach(function(s){
      if(left <= 0) return;
      var t = s[0].slice(0, left); left -= s[0].length;
      if(s[1]){ var b = document.createElement('strong'); b.textContent = t; pt.appendChild(b); }
      else pt.appendChild(document.createTextNode(t));
    });
  }
  function tick(){
    if(!typing) return;
    var segs = phrases[pi], L = total(segs), wait;
    shown += dir;
    render(segs, shown);
    if(dir === 1 && shown >= L){ dir = -1; wait = 2200; }
    else if(dir === -1 && shown <= 0){ dir = 1; pi = (pi+1) % phrases.length; wait = 380; }
    else wait = dir === 1 ? 46 + Math.random()*40 : 22;
    timer = setTimeout(tick, wait);
  }
  prompt.addEventListener('click', function(){
    ta.value = phrases[pi].map(function(s){ return s[0]; }).join('');
    fit(); ta.focus();
  });

  /* ---------- 2. growing input ---------- */
  var MIN = 533, MAX = 780;
  function fit(){
    var u = U(), narrow = window.matchMedia('(max-width: 760px)').matches;
    if(!narrow){
      mirror.textContent = ta.value.split('\n').reduce(function(a,l){ return a.length >= l.length ? a : l; }, '') || ' ';
      ask.style.width = Math.min(MAX*u, Math.max(MIN*u, mirror.offsetWidth + 2)) + 'px';
    }
    ta.style.height = 'auto';
    ta.style.height = ta.scrollHeight + 'px';
  }
  ta.addEventListener('input', fit);
  window.addEventListener('resize', fit);
  fit();
  ta.addEventListener('keydown', function(e){
    if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); submit(); }
  });

  /* ---------- 3. FLIP: animate anything that changes place between states ---------- */
  function setState(next){
    if(page.dataset.state === next) return;
    var els = [modes, ask];
    var first = els.map(function(e){ return e.getBoundingClientRect(); });
    page.dataset.state = next;
    var last = els.map(function(e){ return e.getBoundingClientRect(); });
    els.forEach(function(e, i){
      var dx = first[i].left - last[i].left, dy = first[i].top - last[i].top;
      if(!dx && !dy) return;
      var base = getComputedStyle(e).transform; if(base === 'none') base = '';
      e.animate(
        [{ transform: 'translate(' + dx + 'px,' + dy + 'px) ' + base }, { transform: base }],
        { duration: 680, easing: EASE }
      );
    });
  }

  /* ---------- 4. thread ---------- */
  function row(kind, avatar){
    var li = document.createElement('li'); li.className = 'row ' + kind;
    var a = document.createElement('div'); a.className = 'row-avatar';
    var img = document.createElement('img'); img.src = avatar; img.alt = ''; a.appendChild(img);
    var b = document.createElement('div'); b.className = 'bubble';
    li.appendChild(a); li.appendChild(b); thread.appendChild(li);
    return b;
  }
  function scrollEnd(){ thread.scrollTo({ top: thread.scrollHeight, behavior: 'smooth' }); }

  /* the typed text leaves the input and lands in its thread position */
  function flyIn(bubble, from){
    var to = bubble.getBoundingClientRect();
    bubble.animate(
      [{ transform: 'translate(' + (from.left - to.left) + 'px,' + (from.top - to.top) + 'px)', opacity: .35 },
       { transform: 'none', opacity: 1 }],
      { duration: 620, easing: EASE, fill: 'backwards' }
    );
  }

  function stream(bubble, text, done){
    bubble.textContent = '';
    var words = text.split(' ');
    words.forEach(function(w, i){
      var s = document.createElement('span'); s.className = 'w'; s.textContent = w + (i < words.length-1 ? ' ' : '');
      bubble.appendChild(s);
    });
    var spans = bubble.querySelectorAll('.w'), i = 0;
    (function next(){
      if(i >= spans.length){ if(done) done(); return; }
      spans[i++].classList.add('on'); scrollEnd();
      setTimeout(next, 55 + Math.random()*70);
    })();
  }

  /* ---------- Tracer backend ---------- */
  var API = (window.TRACER_API || '').replace(/\/$/, '');   /* same origin by default; set window.TRACER_API to point elsewhere */
  var mode = 'cameras';
  var history = [];                                            /* last exchanges, so "yes exactly" means something */
  function answer(q){
    return fetch(API + '/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ question:q, mode:mode, history: history.slice(-3) }) })
      .then(function(r){ if(!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .catch(function(e){ return { text: 'I could not reach the engine (' + e.message + '). Is the backend running?', mood: 'botUnsure', evidence: [] }; });
  }
  function evidenceStrip(bubble, items){
    if(!items || !items.length) return;
    var strip = document.createElement('div'); strip.className = 'evidence';
    items.slice(0, 6).forEach(function(it){
      var fig = document.createElement('figure');
      var img = document.createElement('img'); img.src = API + it.url; img.alt = it.entity_id; img.loading = 'lazy';
      var cap = document.createElement('figcaption'); cap.textContent = it.label || it.entity_id;
      fig.appendChild(img); fig.appendChild(cap); strip.appendChild(fig);
    });
    bubble.parentNode.appendChild(strip);
  }
  function metaLine(bubble, r){
    var bits = [];
    if(r.window) bits.push(r.window);
    if(typeof r.latency_ms === 'number') bits.push((r.latency_ms/1000).toFixed(1) + 's');
    if(r.grounding && r.grounding !== 'ok' && r.grounding !== 'none') bits.push(r.grounding.replace('_',' '));
    if(!bits.length) return;
    var m = document.createElement('div'); m.className = 'meta'; m.textContent = bits.join(' · ');
    bubble.parentNode.appendChild(m);
  }
  /* live frame + footage status when Cameras is selected */
  var live = document.createElement('div'); live.className = 'live'; live.innerHTML = '<img alt="live frame"><div class="status"></div>';
  document.querySelector('.page').appendChild(live);
  var liveTimer = null;
  function pollLive(){
    var img = live.querySelector('img');
    img.src = API + '/live/latest.jpg?t=' + Date.now();
    fetch(API + '/health').then(function(r){ return r.json(); }).then(function(h){
      live.querySelector('.status').textContent = h.footage ? (h.footage.start + ' → ' + h.footage.end + ' · ' + h.episodes + ' episodes' + (h.ingest && h.ingest.running ? ' · live' : '')) : 'no footage yet';
    }).catch(function(){});
  }
  function setLive(on){
    live.classList.toggle('on', on);
    clearInterval(liveTimer); liveTimer = null;
    if(on){ pollLive(); liveTimer = setInterval(pollLive, 2000); }
  }

  var busy = false;
  function submit(){
    var q = ta.value.trim();
    if(!q || busy) return;
    busy = true;
    typing = false; clearTimeout(timer);
    var firstTurn = page.dataset.state !== 'chat';
    var from = ta.getBoundingClientRect();
    setState('chat');
    ta.value = ''; fit();

    setTimeout(function(){
      var ub = row('user', AV.user); ub.textContent = q;
      flyIn(ub, from);
      scrollEnd();

      var b = row('bot', AV.bot);
      b.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>';
      scrollEnd();

      answer(q).then(function(reply){
        history.push({ q: q, a: reply.text || '', action: reply.action || 'answer' });
        if(reply.mood === 'botUnsure') b.parentNode.querySelector('.row-avatar img').src = AV.botUnsure;
        stream(b, reply.text, function(){ metaLine(b, reply); evidenceStrip(b, reply.evidence); scrollEnd(); busy = false; });
      });
    }, firstTurn ? 260 : 0);
  }

  /* ---------- 5. controls ---------- */
  var micWrap = document.querySelector('.mic-wrap'), mic = micWrap.querySelector('.mic');
  mic.addEventListener('click', function(){
    var on = mic.getAttribute('aria-pressed') === 'true';
    mic.setAttribute('aria-pressed', on ? 'false' : 'true');
    mic.setAttribute('aria-label', on ? 'Ask by voice' : 'Stop listening');
    micWrap.classList.toggle('on', !on);
  });
  var items = modes.querySelectorAll('li');
  items.forEach(function(li){
    var btn = li.querySelector('button');
    btn.addEventListener('click', function(){
      var wasOn = li.classList.contains('on');
      items.forEach(function(o){ o.classList.toggle('on', o === li && !wasOn); o.querySelector('button').setAttribute('aria-pressed', o === li && !wasOn ? 'true' : 'false'); });
      var label = (btn.getAttribute('aria-label') || '').toLowerCase();
      mode = (!wasOn && label) ? label : 'cameras';
      setLive(!wasOn && label === 'cameras');
    });
  });
})();
</script>
</body>
</html>
EOF_VI
cat > tests/test_api.py << 'EOF_VI'
import os
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope="module")
def store_with_footage(tmp_path_factory):
    pytest.importorskip("fastapi")
    from vi.ingest.synthetic import write_walk_clip
    tmp = tmp_path_factory.mktemp("api")
    clip = write_walk_clip(tmp / "walk.mp4", seconds=8, fps=10)
    db = f"sqlite+pysqlite:///{tmp / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(clip), "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "fake", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--out", str(tmp / "ep"),
                          "--live-dir", str(tmp / "live")], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1200:]
    return db, tmp


def test_api_serves_page_health_ask_and_evidence(store_with_footage):
    from fastapi.testclient import TestClient
    from vi.api import create_app
    db, tmp = store_with_footage
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"),
                     live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    assert c.get("/").status_code == 200 and "Tracer" in c.get("/").text and "fetch(API + '/ask'" in c.get("/").text
    h = c.get("/health").json()
    assert h["ok"] and h["footage"]["start"] == "10:00:00" and h["episodes"] >= 1
    r = c.post("/ask", json={"question": "hi"}).json()
    assert r["mood"] == "bot" and r["grounding"] == "greeting"
    r = c.post("/ask", json={"question": "Who was at the door at 23:00?"}).json()
    assert r["grounding"] == "future" and r["mood"] == "botUnsure" and "23:00" in r["text"]
    r = c.post("/ask", json={"question": "Who was there in the last 5 seconds?"}).json()
    assert r["grounding"] == "ok" and r["window"] and isinstance(r["evidence"], list)
    if r["evidence"]:
        assert c.get(r["evidence"][0]["url"]).status_code == 200
    assert c.get("/episodes").json()[0]["status"] in ("closed", "soft_cut")
    assert c.get("/live/latest.jpg").status_code == 200
    assert c.get("/keyframes/../../etc/passwd").status_code in (404, 400)
    assert c.get("/ingest/status").json()["running"] is False


def test_followups_and_deadline(store_with_footage):
    from fastapi.testclient import TestClient
    from vi.agent.loop import resolve_followup
    from vi.api import create_app
    db, tmp = store_with_footage
    q, ctx = resolve_followup("yes exactly", [{"q": "what happened a minute ago?", "a": "Do you mean 10:00:05?", "action": "clarify"}])
    assert q.startswith("what happened a minute ago?") and "confirmed" in q and "User: yes exactly" in ctx
    q2, ctx2 = resolve_followup("and before that?", [{"q": "who came in?", "a": "cam1:E1 at 10:00:02.", "action": "answer"}])
    assert q2 == "and before that?" and "Previous answer: cam1:E1" in ctx2
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"),
                     live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    r = c.post("/ask", json={"question": "what just happened a minute ago?"}).json()
    assert r["grounding"] == "ok" and r["window"] and r["action"] in ("answer", "clarify")
    r = c.post("/ask", json={"question": "yes exactly", "history": [{"q": "who was there just now?", "a": "Which person?", "action": "clarify"}]}).json()
    assert r["action"] in ("answer", "clarify") and r["grounding"] != "off_topic"
EOF_VI
cat > colab/serve.sh << 'EOF_VI'
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
export VI_DB="${VI_DB:-${DB_URL:-sqlite+pysqlite:///data/vi.db}}" VI_BACKEND="${VI_BACKEND:-transformers}" VI_MODEL="${VI_MODEL:-Qwen/Qwen3.5-4B}" VI_TZ="${VI_TZ:-UTC}"
pkill -f "[u]vicorn vi.api.server" 2>/dev/null; sleep 1
nohup python -m uvicorn vi.api.server:app --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && echo "api up on :$PORT ($VI_BACKEND $VI_MODEL, db $VI_DB)" || { echo "api failed to start; log:"; tail -20 "$LOG"; exit 1; }
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
EOF_VI
cat > pyproject.toml << 'EOF_VI'
[project]
name = "vi-engine"
version = "0.1.0"
description = "Video intelligence engine: tube-centric perception (Block 1) + evidence-backed reasoning (Block 2)"
requires-python = ">=3.10"
dependencies = ["pydantic>=2.6", "numpy>=1.26", "pyyaml>=6", "sqlalchemy>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8"]
ingest = ["av>=13", "pillow>=10", "scipy>=1.11"]
perception = ["rfdetr==1.7.0", "av>=13", "torch", "torchreid"]
serving = ["vllm>=0.12", "boto3"]
store = ["psycopg[binary]>=3.2"]
ui = ["gradio>=5", "fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["vi*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["edge(id): test covers the edge case with this ID from edge_cases.yaml"]
EOF_VI
cat > vi/ingest/grid.py << 'EOF_VI'
"""Multiplexed feeds: one video whose frame is an R×C grid of cameras (the usual NVR export).
Each cell becomes a virtual camera with its own id. Burned-in labels ("CAM 04") sit in a corner
of every cell; that corner is a media zone so the text never becomes a detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridSpec:
    rows: int
    cols: int
    margin_px: int = 0                     # border between cells, if the NVR draws one
    label_corners: tuple[str, ...] = ("top-right", "bottom-left")   # channel name / timestamp overlays (NVRs use both)
    label_frac: tuple[float, float] = (0.36, 0.14)   # label box as a fraction of cell width/height

    @classmethod
    def parse(cls, s: str, margin_px: int = 0) -> "GridSpec":
        r, c = s.lower().split("x")
        return cls(rows=int(r), cols=int(c), margin_px=margin_px)

    @property
    def n(self) -> int:
        return self.rows * self.cols


def cell_ids(spec: GridSpec, prefix: str = "cam") -> list[str]:
    return [f"{prefix}{i + 1:02d}" for i in range(spec.n)]


def cell_boxes(spec: GridSpec, frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
    """(x1, y1, x2, y2) per cell in frame pixels, row-major."""
    cw, ch = frame_w / spec.cols, frame_h / spec.rows
    m = spec.margin_px
    out = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            x1, y1 = int(round(c * cw)) + m, int(round(r * ch)) + m
            x2, y2 = int(round((c + 1) * cw)) - m, int(round((r + 1) * ch)) - m
            out.append((x1, y1, max(x1 + 2, x2), max(y1 + 2, y2)))
    return out


def split(frame: np.ndarray, spec: GridSpec) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    return [np.ascontiguousarray(frame[y1:y2, x1:x2]) for (x1, y1, x2, y2) in cell_boxes(spec, w, h)]


def label_zones(spec: GridSpec, cell_w: int, cell_h: int, camera_id: str, tile_id: str) -> list:
    """Media zones over the burned-in overlay corners of one cell (E-DET-05)."""
    from vi.events import Zone
    lw, lh = int(cell_w * spec.label_frac[0]), int(cell_h * spec.label_frac[1])
    polys = {"top-left": [(0, 0), (lw, 0), (lw, lh), (0, lh)],
             "top-right": [(cell_w - lw, 0), (cell_w, 0), (cell_w, lh), (cell_w - lw, lh)],
             "bottom-left": [(0, cell_h - lh), (lw, cell_h - lh), (lw, cell_h), (0, cell_h)],
             "bottom-right": [(cell_w - lw, cell_h - lh), (cell_w, cell_h - lh), (cell_w, cell_h), (cell_w - lw, cell_h)]}
    return [Zone(zone_id=f"label_{c.replace('-', '_')}", camera_id=camera_id, tile_id=tile_id, kind="media", polygon=polys[c])
            for c in spec.label_corners if c in polys]


def label_zone(spec: GridSpec, cell_w: int, cell_h: int, camera_id: str, tile_id: str):
    return label_zones(spec, cell_w, cell_h, camera_id, tile_id)[0]


CANDIDATES = [(2, 2), (3, 3), (4, 4), (1, 2), (2, 1), (2, 3), (3, 2), (2, 4), (4, 2)]


def seam_scores(gray: np.ndarray, rows: int, cols: int, band: int = 3) -> list[float]:
    """For every interior seam of an R×C layout, how much stronger the image gradient is across the
    seam than elsewhere (1.0 = no seam). NVR multiplexes have hard borders between cells."""
    h, w = gray.shape[:2]
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)       # per column
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)       # per row
    base_x, base_y = float(np.median(dx)) + 1e-3, float(np.median(dy)) + 1e-3
    scores = []
    for c in range(1, cols):
        x = int(round(w * c / cols))
        scores.append(float(dx[max(0, x - band): min(len(dx), x + band)].max()) / base_x)
    for r in range(1, rows):
        y = int(round(h * r / rows))
        scores.append(float(dy[max(0, y - band): min(len(dy), y + band)].max()) / base_y)
    return scores


def detect_grid(frames: list[np.ndarray], min_ratio: float = 6.0) -> tuple[GridSpec | None, dict]:
    """Pick the layout with the MOST cells whose WEAKEST seam is still clearly a border (min seam
    ratio >= min_ratio). A 4×4 contains the 2×2's seams, so it qualifies only if its quarter seams
    are strong too; a 2×2 outranks 1×2 because its horizontal seam also qualifies.
    Returns (spec or None for a single camera, evidence)."""
    grays = [f.mean(axis=2).astype(np.float32) if f.ndim == 3 else f.astype(np.float32) for f in frames]
    evidence = {}
    best = None
    for (r, c) in sorted(CANDIDATES, key=lambda rc: -(rc[0] * rc[1])):
        per_frame = [seam_scores(g, r, c) for g in grays]
        mins = [min(s) for s in per_frame]
        score = float(np.median(mins))
        evidence[f"{r}x{c}"] = round(score, 1)
        if score >= min_ratio and best is None:
            best = (r, c)
    return (GridSpec(rows=best[0], cols=best[1]) if best else None), evidence



def compose(cells: list[np.ndarray], spec: GridSpec) -> np.ndarray:
    """Put annotated cells back into one grid frame (for the live view)."""
    if not cells:
        return np.zeros((2, 2, 3), np.uint8)
    ch = max(c.shape[0] for c in cells); cw = max(c.shape[1] for c in cells)
    out = np.zeros((spec.rows * ch, spec.cols * cw, 3), np.uint8)
    for i, c in enumerate(cells[: spec.n]):
        r, k = divmod(i, spec.cols)
        out[r * ch: r * ch + c.shape[0], k * cw: k * cw + c.shape[1]] = c
    return out
EOF_VI
cat > vi/ingest/__init__.py << 'EOF_VI'
from .reader import Frame, PTSFilter, ReaderStats, SizeGuard, VideoReader, gate_mode_for_codec
from .grid import GridSpec, cell_boxes, cell_ids, compose, detect_grid, label_zone, label_zones, split
EOF_VI
cat > vi/episode/debug.py << 'EOF_VI'
from __future__ import annotations

from pathlib import Path

import numpy as np

from vi.detect import ROI, Detection
from vi.schemas import Tube

COL = {"roi": (60, 200, 60), "full": (60, 160, 255), "det_roi": (255, 220, 40), "det_full": (80, 230, 255),
       "active": (255, 60, 60), "occluded": (255, 140, 40), "born": (200, 200, 200)}


def annotate(frame_rgb: np.ndarray, rois: list[ROI], dets: list[Detection], tubes: list[Tube],
             title: str, path: str | Path | None):
    """Debug frame: crops (green; full frame blue), detections (yellow from crops, cyan from the
    full frame, dashed-ish by confidence label), tubes (red active, orange predicted, grey born)
    with id suffix and state. Opens in Colab's file browser; upload one to the chat to review."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    im = Image.fromarray(np.ascontiguousarray(frame_rgb))
    dr = ImageDraw.Draw(im)
    for r in rois:
        c = COL["full"] if r.source_blobs == 0 else COL["roi"]
        dr.rectangle([r.x1, r.y1, r.x2 - 1, r.y2 - 1], outline=c, width=1)
    for d in dets:
        c = COL["det_full"] if d.origin == "full" else COL["det_roi"]
        b = d.box
        dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=1)
        dr.text((b.x1 + 2, b.y2 - 12), f"{d.class_label[:6]} {d.confidence:.2f}{'t' if d.roi_truncated else ''}", fill=c)
    for t in tubes:
        c = COL.get(t.state.value, COL["born"])
        b = t.box
        dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=3)
        dr.text((b.x1 + 2, max(0, b.y1 - 12)), f"#{t.tube_id.split(':')[-1]} {t.state.value[:3]}", fill=c)
    dr.text((6, 6), title, fill=(255, 255, 255))
    if path is None:
        return np.asarray(im)                 # in-memory panel (live view composition)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=85)
    return path
EOF_VI
cat > vi/tubes/quality.py << 'EOF_VI'
from __future__ import annotations

from vi.schemas import Tube

MIN_LIFE_MS = 1500
MIN_HEIGHT_PX = 48
EDGE_PX = 8


def grade_tube(tube: Tube, frame_w: int, frame_h: int, median_height_px: float | None = None,
               min_life_ms: int = MIN_LIFE_MS, min_height_frac: float = 0.4, border_px: int = EDGE_PX) -> Tube:
    """E-DET-01 / resolution ceiling: a tube that lived under 1.5 s, is under 48 px tall, or was
    born hugging the frame border is real evidence of *something*, not a confirmed person. It
    stays in the store, flagged, so the script can count it apart."""
    life = tube.last_seen.corrected_ms() - tube.born.corrected_ms()
    b = tube.box
    reasons = []
    if life < min_life_ms:
        reasons.append(f"life {life / 1000:.1f}s")
    h = max(tube.max_height_px, b.height)
    floor = min_height_frac * median_height_px if median_height_px else MIN_HEIGHT_PX   # scene units when known
    if h < floor:
        reasons.append(f"height {int(h)}px < {int(floor)}px")
    if life < min_life_ms:
        pass
    at_border = b.x1 <= border_px or b.y1 <= border_px or b.x2 >= frame_w - border_px or b.y2 >= frame_h - border_px
    if at_border and life < 2 * min_life_ms:      # a brief flicker at the edge; a long tube that EXITS at the edge is a person
        reasons.append("at frame border")
    tube.quality = "low" if reasons else "ok"
    tube.quality_reason = ", ".join(reasons) or None
    return tube
EOF_VI
cat > tests/test_profiles_generator.py << 'EOF_VI'
import json
import os
import subprocess
import sys
from datetime import datetime

import pytest

from vi.generator import BRANCHES, plan
from vi.profiles import available_profiles, load_profile


def test_every_profile_loads_extends_common_and_names_known_or_declared_branches():
    names = available_profiles()
    assert "common" in names and len(names) >= 6
    common = load_profile("common")
    for n in names:
        p = load_profile(n)
        assert set(common.branches) <= set(p.branches)            # tiers add, never remove
        assert p.quality.min_height_frac == 0.4 and p.sampling["active_fps"] == 4
    t1 = load_profile("tier1_hospital")
    assert "fall" in t1.events and "enter_zone" in t1.events and "wheelchair" in t1.tube_classes


def test_plan_reports_unavailable_branches_instead_of_failing():
    p = plan(["detect", "pose", "made_up"], available_modules={"rfdetr"})
    assert p["detect"]["runnable"] and p["pose"]["runnable"] is False and "mmpose" in p["pose"]["why"]
    assert p["made_up"]["branch"] is None
    assert all(b.license and b.status in ("measured", "planned", "stub") for b in BRANCHES.values())


def test_media_zone_fires_no_events():
    from vi.events import EventCompiler, Zone
    from vi.schemas import TubeSnapshot, TubeState, Box
    z = Zone(zone_id="rack", camera_id="c1", kind="media", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    ec = EventCompiler("c1", [z], enter_ticks=1, open_grace_ms=0)
    snap = TubeSnapshot(tube_id="a", class_label="person", state=TubeState.active, box=Box(x1=10, y1=10, x2=30, y2=90))
    assert ec.on_tick([], -500) == [] and ec.on_tick([snap], 0) == [] and ec.on_tick([snap], 500) == []


def test_writer_parser_tolerates_prose_missing_cells_and_bad_colors():
    from vi.writer import parse_sheet_reply
    reply = 'Here you go:\n```json\n[{"cell_id": 0, "top_color": "ORANGE", "headwear": "white hard hat", "description": "worker at a table", "confidence": 0.9},' \
            ' {"cell_id": 1, "top_color": "neon", "carried_item": "box"}]\n```'
    r = parse_sheet_reply(reply, ["t0", "t1", "t2"])
    assert r is not None and r.expected_cells == 3 and len(r.cells) == 3
    a0, a1, a2 = (c.attributes for c in r.cells)
    assert a0.top_color.value == "orange" and "hard hat" in a0.description and a0.confidence == 0.9
    assert a1.top_color is None and a1.carried_item == "box"
    assert a2.confidence == 0.0                                      # missing cell: empty, not invented
    assert parse_sheet_reply("no json here", ["t0"]) is None


def test_pack_sheet_numbers_cells():
    import numpy as np
    from vi.writer import pack_sheet
    sheet = pack_sheet([np.zeros((120, 40, 3), np.uint8)] * 5, cell=100, cols=3)
    assert sheet.size == (300, 2 * (100 + 22))      # two rows, each cell + caption strip


def test_numeric_tools_and_whole_time_check(tmp_path):
    from vi.agent import ask, count_entities, coverage, entities_present, search_events
    from vi.store import connect, load_episode_file
    out = subprocess.run([sys.executable, "bench/slice_cpu.py", str(tmp_path / "ep")], capture_output=True, text=True)
    assert out.returncode == 0
    path = next((tmp_path / "ep").glob("*.jsonl"))
    engine = connect(); load_episode_file(engine, path)
    ep = search_events(engine)[0]["episode_id"]
    cov = coverage(engine, ep)
    assert cov and 0 < cov[0]["coverage"] <= 1
    assert count_entities(engine, ep)["count"] == 1
    assert entities_present(engine, ep, 0.99)["count"] == 0          # the walker is not present for the whole clip

    class Overcounter:
        name = "over"
        def __init__(self): self.n = 0
        def complete(self, messages, schema):
            self.n += 1
            if self.n == 1:
                return json.dumps({"action": "answer", "text": "There were 4 people.", "citations": ["cam1:2000:1"], "confidence": 0.9})
            assert "count_entities says 1" in messages[-1]["content"]
            return json.dumps({"action": "answer", "text": "There was 1 person.", "citations": ["cam1:2000:1"], "confidence": 0.9})
    res = ask(engine, "How many people were there?", ep, Overcounter())
    assert any(t.get("revise") == "numeric claim disagrees with tools" for t in res["trace"]) and "1 person" in res["final"]["text"]


def test_metamorphic_bench_runs_on_synthetic_with_fake_detector(tmp_path):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/metamorphic.py", "--source", str(clip), "--variants", "brightness_up,hflip,fps6",
                          "--fps", "5", "--work", str(tmp_path / "mm"), "--", "--model", "fake", "--detect", "frame"],
                         capture_output=True, text=True, cwd=".", env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    assert "3/3 invariants hold" in out.stdout


def test_time_grounding_and_scope_guards():
    from datetime import datetime, timezone
    from vi.agent import classify_scope, ground_time
    start = int(datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc).timestamp() * 1000); end = start + 3600_000
    assert ground_time("Who was at the door at 12:30?", end, start, end).kind == "future"
    assert ground_time("What happened at 9:30?", end, start, end).kind == "before_start"
    g = ground_time("How many people in the last 15 minutes?", end, start, end)
    assert g.kind == "ok" and g.t_start_ms == end - 15 * 60_000
    g = ground_time("What happened between 10:10 and 10:20?", end, start, end)
    assert (g.t_start_ms - start, g.t_end_ms - start) == (10 * 60_000, 20 * 60_000)
    assert ground_time("How many people were there?", end, start, end).kind == "none"
    assert classify_scope("What's the weather?")[0] == "off_topic" and classify_scope("Lock the door")[0] == "act"
    assert classify_scope("Who was at the conveyor?")[0] == "ok" and classify_scope("Who is he?")[0] == "identity"


def test_ask_window_refuses_future_and_off_topic_without_a_model_call(tmp_path):
    from vi.agent import ask_window
    from vi.store import connect
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=8, fps=10)
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(clip), "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "fake", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--out", str(tmp_path / "ep")],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1200:]
    engine = connect(db)

    class Never:
        name = "never"
        def complete(self, messages, schema):
            raise AssertionError("the model must not be called for guarded questions")
    now = int(datetime.fromisoformat("2026-09-27T10:00:10+00:00").timestamp() * 1000)
    assert ask_window(engine, "Who was at the door at 15:00?", Never(), now)["grounding"] == "future"
    assert ask_window(engine, "What's the weather like?", Never(), now)["grounding"] == "off_topic"
    assert ask_window(engine, "Unlock the door", Never(), now)["grounding"] == "act"
    from vi.agent import FakeBackend
    r = ask_window(engine, "How many people in the last 5 seconds?", FakeBackend(), now)
    assert r["grounding"] == "ok" and r["final"]["action"] == "answer" and r["window"]


def test_ui_builds_without_launching(tmp_path):
    pytest.importorskip("gradio")
    import sys as _sys
    _sys.argv = ["x"]
    from ui.app import build
    demo = build("sqlite+pysqlite:///:memory:", "fake", "fake", "UTC")
    assert demo is not None


def test_border_only_flags_brief_tubes():
    from vi.tubes import grade_tube
    from vi.schemas import Box, CamTime, Tube
    long = Tube(tube_id="c1:0:1", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0), last_seen=CamTime(cam_utc_ms=15000),
                box=Box(x1=290, y1=80, x2=318, y2=200), max_height_px=120)
    grade_tube(long, 320, 240)
    assert long.quality == "ok"                                     # walked out through the edge after 15 s
    brief = Tube(tube_id="c1:0:2", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0), last_seen=CamTime(cam_utc_ms=2000),
                 box=Box(x1=0, y1=80, x2=20, y2=200), max_height_px=120)
    grade_tube(brief, 320, 240)
    assert brief.quality == "low" and "border" in brief.quality_reason


def test_grid_ingest_makes_a_camera_per_cell_and_camera_questions_ground(tmp_path):
    import av
    import numpy as np
    from vi.agent import ask_window, FakeBackend, footage_bounds
    from vi.store import connect
    rng = np.random.default_rng(1)
    path = tmp_path / "grid.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 30):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:
            f[cy + 4: cy + 20, cx + 250: cx + 316] = 240               # burned-in labels
        if 3 <= t < 18:
            x = 10 + int(((t - 3) / 15) * 260); f[80:200, x:x + 24] = 235            # cell 0 walker
        if 10 <= t < 28:
            x = 330 + int(((t - 10) / 18) * 260); f[320:440, x:x + 24] = 235        # cell 3 walker
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "2x2", "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "fake", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
                          "--out", str(tmp_path / "ep"), "--live-dir", str(tmp_path / "live")], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    row = json.loads(out.stdout[out.stdout.rindex("{\n"):])
    assert row["cameras"] == 4 and row["episodes"] == 2 and row["realtime_factor"] > 1
    engine = connect(db)
    from vi.agent.tools import cameras_in_store, count_entities_window
    assert cameras_in_store(engine) == ["cam01", "cam04"]              # the label boxes never became tubes
    b = footage_bounds(engine)
    assert count_entities_window(engine, b[0], b[1], camera_id="cam04")["count"] == 1
    r = ask_window(engine, "Who was on cam 4?", FakeBackend(), b[1])
    assert r["camera_id"] == "cam04"
    assert ask_window(engine, "What happened on camera 9?", FakeBackend(), b[1])["grounding"] == "unknown_camera"
    assert (tmp_path / "live" / "latest.jpg").exists()


def test_grid_autodetect_and_cross_camera_fusion(tmp_path):
    """A 2x2 NVR view of ONE walker seen by all four cells at once: auto grid, and the count is 1 person / 4 tracks."""
    import av
    import numpy as np
    from vi.agent import ask_window, FakeBackend, footage_bounds
    from vi.agent.tools import count_entities_window, window_script
    from vi.store import connect
    rng = np.random.default_rng(3)
    path = tmp_path / "shop.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 24):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        f[:, 318:322] = 0; f[238:242, :] = 0                          # NVR seams
        if 3 <= t < 21:
            x = 10 + int(((t - 3) / 18) * 250)
            for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:  # the same bright figure in every cell = same person, 4 angles
                f[cy + 60: cy + 180, cx + x: cx + x + 24] = 235
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "auto", "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "none", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
                          "--out", str(tmp_path / "ep"), "--live-dir", str(tmp_path / "live")], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "grid auto -> 2x2" in out.stdout
    engine = connect(db)
    b = footage_bounds(engine)
    r = count_entities_window(engine, b[0], b[1])
    assert r["camera_entities"] == 4 and r["people"] == 1 and len(r["world_groups"]) == 1        # one man, four cameras
    script = window_script(engine, b[0], b[1])
    assert "CAST: 1 people (4 camera tracks" in script and "W1: " in script
    res = ask_window(engine, "How many people were there?", FakeBackend(), b[1])
    assert res["final"]["action"] == "answer"


def test_tile_discovery_then_tile_identity_end_to_end(tmp_path):
    """Phase 1 (per camera) records one man on four cameras; discovery groups the cameras; phase 2
    with the tile map gives him ONE id at ingest time."""
    import av
    import numpy as np
    from vi.agent import footage_bounds
    from vi.agent.tools import cameras_in_store, coverage_window
    from vi.fusion import discover_tiles
    from vi.fusion.tiles import entities_for_affinity
    from vi.store import connect
    rng = np.random.default_rng(5)
    path = tmp_path / "shop.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 24):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        f[:, 318:322] = 0; f[238:242, :] = 0
        # two different-looking people, one after the other, each seen by all four cells at once
        for (t0, t1, val) in [(3, 10, 250), (13, 21, 205)]:      # both above the fake detector luma (200), different histogram bins
            if t0 <= t < t1:
                x = 10 + int(((t - t0) / (t1 - t0)) * 250)
                for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:
                    f[cy + 60: cy + 180, cx + x: cx + x + 24] = val
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    base = [sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "2x2", "--db", db, "--model", "fake", "--reid", "hist",
            "--writer", "none", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
            "--live-dir", str(tmp_path / "live")]
    out = subprocess.run(base + ["--out", str(tmp_path / "ep1"), "--tiles", "none"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    engine = connect(db)
    tm = discover_tiles(entities_for_affinity(engine), cameras=cameras_in_store(engine))
    assert list(tm.tiles.values()) == [["cam01", "cam02", "cam03", "cam04"]]            # all four see the same area
    tm.save(tmp_path / "tiles.json")
    db2 = f"sqlite+pysqlite:///{tmp_path / 'vi2.db'}"
    out = subprocess.run([*(base[:base.index('--db') + 1]), db2, *base[base.index('--db') + 2:], "--out", str(tmp_path / "ep2"),
                          "--tiles", str(tmp_path / "tiles.json")], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "tiles: {'T1': ['cam01', 'cam02', 'cam03', 'cam04']}" in out.stdout
    e2 = connect(db2); b = footage_bounds(e2)
    rows = coverage_window(e2, b[0], b[1])
    assert len(rows) == 2 and {r["entity_id"] for r in rows} == {"T1:E1", "T1:E2"}      # two people, each ONE id across four cameras
    assert all(sorted(r["cameras"]) == ["cam01", "cam02", "cam03", "cam04"] for r in rows)
EOF_VI
cat > vi/schemas/tube.py << 'EOF_VI'
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .common import Box, CamTime, FloorPoint, Modality


class TubeState(str, Enum):
    born = "born"
    active = "active"
    occluded = "occluded"   # E-TUBE-02: bounded; becomes lost after max_occluded_ms
    exited = "exited"       # left through a known exit zone
    lost = "lost"           # vanished without an exit; candidate for fusion re-link
    dead = "dead"           # closed, no further deltas


class Color(str, Enum):
    black = "black"
    white = "white"
    gray = "gray"
    red = "red"
    orange = "orange"
    yellow = "yellow"
    green = "green"
    blue = "blue"
    purple = "purple"
    pink = "pink"
    brown = "brown"
    multicolor = "multicolor"


class Tone(str, Enum):
    light = "light"
    mid = "mid"
    dark = "dark"


class Attributes(BaseModel):
    """Writer-VLM output for one crop. E-FOV-05: in non-color modalities every color
    field must be None with color_reason='ir_mode'; the grammar and this validator
    both enforce it so a monochrome crop can never yield 'red shirt'."""

    modality: Modality
    top_color: Color | None = None
    bottom_color: Color | None = None
    top_tone: Tone | None = None
    bottom_tone: Tone | None = None
    color_reason: Literal["ir_mode", "not_visible", "low_confidence"] | None = None
    carried_item: str | None = Field(None, max_length=60)
    carried_item_confidence: float = Field(0.0, ge=0.0, le=1.0)
    description: str = Field("", max_length=240)
    enhanced: bool = False            # E-FOV-01: SR or heavy enhancement applied
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ir_rule(self) -> "Attributes":
        if not self.modality.has_color:
            if self.top_color is not None or self.bottom_color is not None:
                raise ValueError("color attributes are not allowed in non-color modality")
            if self.color_reason != "ir_mode":
                raise ValueError("color_reason must be 'ir_mode' in non-color modality")
        return self


class TubeSnapshot(BaseModel):
    """Per-tick view of a tube (what a Tick carries)."""

    tube_id: str
    class_label: str
    state: TubeState
    box: Box
    foot: FloorPoint | None = None
    zone_ids: list[str] = Field(default_factory=list)
    det_confidence: float = Field(0.0, ge=0.0, le=1.0)
    det_source: Literal["detector", "heartbeat", "predicted"] = "detector"


class Tube(BaseModel):
    """Lifecycle record for one object in one camera. entity_id is assigned by fusion."""

    tube_id: str
    camera_id: str
    tile_id: str | None = None
    entity_id: str | None = None
    named: str | None = None          # gallery match, e.g. "Jay"; None => anonymous
    class_label: str
    state: TubeState = TubeState.born
    born: CamTime
    last_seen: CamTime
    box: Box
    foot: FloorPoint | None = None
    zone_ids: list[str] = Field(default_factory=list)
    modality: Modality = Modality.rgb
    keyframe_refs: list[str] = Field(default_factory=list)   # E-FOV-06: persisted at birth
    attributes: Attributes | None = None
    occluded_since_ms: int | None = None
    merge_candidates: list[str] = Field(default_factory=list)  # E-TUBE-01: prefer flag over wrong merge
    reflection_suspect: bool = False                            # E-DET-04
    max_height_px: float = 0.0             # largest observed box height; quality is judged on this, not the last box
    embedding: list[float] | None = None   # the entity's appearance embedding (EMA) at close, for cross-camera fusion (R14)
    quality: Literal["ok", "low"] = "ok"   # low: brief, tiny or border-hugging (E-DET-01); counted apart from confirmed people
    quality_reason: str | None = None

    @model_validator(mode="after")
    def _times(self) -> "Tube":
        if self.last_seen.corrected_ms() < self.born.corrected_ms():
            raise ValueError("last_seen precedes born")
        return self


class EnrichmentPatch(BaseModel):
    """Slow-path result applied to a tube after the tick was already committed (R4)."""

    patch_id: str
    tube_id: str
    produced_at_ms: int
    source: Literal["vlm", "pose", "ocr", "siglip", "reid", "sr"]
    payload: dict[str, Any]
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    modality: Modality = Modality.rgb
    enhanced: bool = False
    failed: bool = False               # E-FOV-07: null-fill rather than block


ENHANCED_DISCOUNT = 0.7


def effective_confidence(attr: Attributes) -> float:
    """E-FOV-01: attributes read through SR/enhancement are discounted so that a
    rule needing >=0.8 confidence never fires on an upscaled 60-px crop."""
    return attr.confidence * (ENHANCED_DISCOUNT if attr.enhanced else 1.0)
EOF_VI
cat > vi/tubes/linker.py << 'EOF_VI'
"""Appearance-based re-linking of tubes into entities within one camera (E-TUBE-04), with a
drifting gallery per entity (E-TUBE-08). Cross-camera fusion reuses the same gallery format.

Rules:
  * a newborn tube is compared against entities whose last tube closed within max_gap_ms and
    whose last position is within max_jump_px; tubes that closed through an exit zone need a
    stricter similarity (exit zones may be placeholders, and people step out and back);
  * cosine similarity >= sim_thr links it to that entity, else it starts a new one;
  * each entity keeps an EMA embedding plus up to `exemplars` raw embeddings; refreshes happen
    on confident active ticks so the gallery follows jacket-on/jacket-off;
  * a link is emitted as an Event(relink) with both tube ids and the similarity, so the agent
    can cite it and a wrong link can be audited.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from vi.schemas import CamTime, Event, EventType, Tube, TubeState


@dataclass
class _Entity:
    entity_id: str
    tube_ids: list[str]
    ema: np.ndarray
    exemplars: list[np.ndarray] = field(default_factory=list)
    aux: np.ndarray | None = None
    last_box_center: tuple[float, float] = (0.0, 0.0)
    last_box_h: float = 1.0
    born_ms: int = 0
    last_active_ms: int = 0
    lost_at_ms: int | None = None      # set when its tube closed (lost/exited) OR went occluded; None while seen
    closed_as_exit: bool = False
    live_tube_id: str | None = None    # an occluded-but-live tube: linking a newborn to this entity absorbs it
    live_by_cam: dict = field(default_factory=dict)   # camera -> live tube id (a tile entity is seen on several cameras at once)
    cameras: set = field(default_factory=set)


class TubeLinker:
    def __init__(self, camera_id: str, sim_thr: float = 0.88, max_gap_ms: int = 30_000,
                 max_jump_px: float = 400.0, ema_alpha: float = 0.3, exemplars: int = 5,
                 exited_sim_thr: float = 0.90, near_sim_thr: float = 0.85, near_gap_ms: int = 5000,
                 near_jump_px: float = 200.0, aux_thr: float = 0.80, cross_camera_sim_thr: float = 0.85):
        self.camera_id = camera_id          # for a tile linker this is the tile id; tubes carry their own camera_id
        self.cross_camera_sim_thr = cross_camera_sim_thr
        self.sim_thr = sim_thr
        # a tube reappearing within a few seconds and a couple of body-widths of where one vanished
        # is the same person unless appearance says otherwise: the bar drops to near_sim_thr there
        self.near_sim_thr, self.near_gap_ms, self.near_jump_px = near_sim_thr, near_gap_ms, near_jump_px
        # a second, cheap appearance signal (colour histogram) must agree; it stops a generic image
        # embedding from joining a green hi-vis vest to an orange one at the same spot
        self.aux_thr = aux_thr
        self.exited_sim_thr = exited_sim_thr   # placeholder exit zones misclassify lost as exited; allow with more evidence
        self.max_gap_ms = max_gap_ms
        self.max_jump_px = max_jump_px
        self.ema_alpha = ema_alpha
        self.n_exemplars = exemplars
        self._entities: dict[str, _Entity] = {}
        self._tube_entity: dict[str, str] = {}
        self._seq = 0
        self.relinks = 0
        self.merges = 0
        self.max_coexist_ms = 1000
        self.absorbed: list[str] = []      # ghost tubes the tracker should drop (read and clear each tick)

    # ---------------------------------------------------------------- helpers
    def _new_entity(self, tube: Tube, emb: np.ndarray, aux: np.ndarray | None = None) -> _Entity:
        self._seq += 1
        ent = _Entity(entity_id=f"{self.camera_id}:E{self._seq}", tube_ids=[tube.tube_id], ema=emb.copy(),
                      exemplars=[emb.copy()], last_box_center=_center(tube), last_box_h=tube.box.height,
                      aux=None if aux is None else aux.copy(), born_ms=tube.born.corrected_ms(),
                      last_active_ms=tube.last_seen.corrected_ms(), cameras={tube.camera_id})
        ent.live_by_cam[tube.camera_id] = tube.tube_id
        self._entities[ent.entity_id] = ent
        self._tube_entity[tube.tube_id] = ent.entity_id
        return ent

    @staticmethod
    def _sim(ent: _Entity, emb: np.ndarray) -> float:
        best = float(ent.ema @ emb)
        for x in ent.exemplars:
            best = max(best, float(x @ emb))
        return best

    def _candidates(self, tube: Tube, t_ms: int) -> list[_Entity]:
        cx, cy = _center(tube)
        out = []
        for e in self._entities.values():
            if tube.camera_id in e.live_by_cam:               # already seen on THIS camera right now: a different person here
                continue
            if e.cameras and tube.camera_id not in e.cameras:  # another camera of the tile: concurrent sighting is allowed
                if e.live_by_cam or e.lost_at_ms is None or t_ms - e.lost_at_ms <= self.max_gap_ms:
                    out.append(e)                            # no distance gate across cameras
                continue
            if e.lost_at_ms is None or t_ms - e.lost_at_ms > self.max_gap_ms:
                continue
            if ((cx - e.last_box_center[0]) ** 2 + (cy - e.last_box_center[1]) ** 2) ** 0.5 > self.max_jump_px:
                continue
            out.append(e)
        return out

    # ---------------------------------------------------------------- API
    def _thr(self, ent: _Entity, tube: Tube, t_ms: int) -> float:
        if ent.cameras and tube.camera_id not in ent.cameras:
            return self.cross_camera_sim_thr                  # a different view of the same person: appearance only
        if ent.closed_as_exit:
            return self.exited_sim_thr
        cx, cy = _center(tube)
        dist = ((cx - ent.last_box_center[0]) ** 2 + (cy - ent.last_box_center[1]) ** 2) ** 0.5
        gap = t_ms - (ent.lost_at_ms or t_ms)
        return self.near_sim_thr if (gap <= self.near_gap_ms and dist <= self.near_jump_px) else self.sim_thr

    def on_birth(self, tube: Tube, emb: np.ndarray, t_ms: int, aux: np.ndarray | None = None) -> Event | None:
        cands = self._candidates(tube, t_ms)
        if aux is not None:   # second-signal gate first: candidates whose colour disagrees are out
            cands = [e for e in cands if e.aux is None or float(e.aux @ aux) >= self.aux_thr]
        if cands:
            best = max(cands, key=lambda e: self._sim(e, emb))
            sim = self._sim(best, emb)
            thr = self._thr(best, tube, t_ms)
            if sim >= thr:
                prev = best.tube_ids[-1]
                cross = tube.camera_id not in best.cameras
                absorbed = None if cross else best.live_tube_id   # the occluded ghost this newborn replaces (same camera)
                best.tube_ids.append(tube.tube_id)
                if not cross:
                    best.lost_at_ms, best.live_tube_id = None, None
                best.cameras.add(tube.camera_id)
                best.live_by_cam[tube.camera_id] = tube.tube_id
                self._tube_entity[tube.tube_id] = best.entity_id
                self.on_refresh(tube, emb, aux)
                self.relinks += 1
                self.absorbed.append(absorbed) if absorbed else None
                return Event(event_id="ev_" + hashlib.sha1(f"relink|{prev}|{tube.tube_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[prev, tube.tube_id], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(sim, 3), "threshold": round(thr, 2), "cross_camera": cross,
                                      "gap_ms": t_ms - (tube.born.corrected_ms()), "absorbed_tube": absorbed},
                             confidence=min(1.0, sim))
        self._new_entity(tube, emb, aux)
        return None

    def on_refresh(self, tube: Tube, emb: np.ndarray, aux: np.ndarray | None = None) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.ema = e.ema * (1 - self.ema_alpha) + emb * self.ema_alpha
        e.ema /= max(1e-8, np.linalg.norm(e.ema))
        if aux is not None:
            e.aux = aux.copy() if e.aux is None else (e.aux * 0.7 + aux * 0.3)
            e.aux /= max(1e-8, np.linalg.norm(e.aux))
        e.exemplars.append(emb.copy())
        if len(e.exemplars) > self.n_exemplars:
            e.exemplars.pop(0)
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height

    def on_state(self, tube: Tube, t_ms: int) -> None:
        """Called every tick for live tubes. The moment a tube goes occluded its entity becomes a
        relink candidate: the same person re-detected nearby is a newborn tube the tracker could
        not associate with the drifted prediction (the hard-hat man, session 23)."""
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        if tube.state == TubeState.occluded:                 # not seen: it is a candidate, not a "seen here" exclusion
            e.live_by_cam.pop(tube.camera_id, None)
            if e.lost_at_ms is None:
                e.lost_at_ms = tube.occluded_since_ms if tube.occluded_since_ms is not None else t_ms
            e.live_tube_id = tube.tube_id
            e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        elif tube.state == TubeState.active:
            e.live_by_cam[tube.camera_id] = tube.tube_id
            e.last_active_ms = t_ms
            if e.live_tube_id == tube.tube_id:
                e.lost_at_ms, e.live_tube_id = None, None   # seen again: no longer a candidate

    def on_close(self, tube: Tube, t_ms: int) -> Event | None:
        """A tube that dies `lost` next to a live entity it overlapped with for under a second,
        and that looks like it, was a second box on the same body: merge it (mirror of on_birth).
        Long co-existence means two people, whatever the embeddings say."""
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return None
        e = self._entities[eid]
        if tube.state == TubeState.lost and len(e.tube_ids) >= 1:
            cx, cy = _center(tube)
            best, best_sim = None, 0.0
            for o in self._entities.values():
                if o is e or o.lost_at_ms is not None or tube.camera_id not in o.live_by_cam:
                    continue                                              # only entities live on THIS camera
                overlap = min(tube.last_seen.corrected_ms(), o.last_active_ms) - max(tube.born.corrected_ms(), o.born_ms)
                dist = ((cx - o.last_box_center[0]) ** 2 + (cy - o.last_box_center[1]) ** 2) ** 0.5
                if overlap > self.max_coexist_ms or dist > self.near_jump_px:
                    continue
                sim = self._sim(o, e.ema)
                if sim > best_sim:
                    best, best_sim = o, sim
            if best is not None and best_sim >= self.near_sim_thr:
                for tid in e.tube_ids:
                    self._tube_entity[tid] = best.entity_id
                best.tube_ids = sorted(set(best.tube_ids + e.tube_ids))
                best.exemplars = (best.exemplars + e.exemplars)[-self.n_exemplars:]
                del self._entities[eid]
                self.merges += 1
                return Event(event_id="ev_" + hashlib.sha1(f"merge|{eid}|{best.entity_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[tube.tube_id, best.tube_ids[-1]], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(best_sim, 3), "kind": "merge_on_death", "merged_entity": eid},
                             confidence=min(1.0, best_sim))
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        e.live_by_cam.pop(tube.camera_id, None)
        e.lost_at_ms = t_ms if tube.state in (TubeState.lost, TubeState.occluded, TubeState.exited) else None
        e.closed_as_exit = tube.state == TubeState.exited
        e.live_tube_id = None
        return None

    def entity_of(self, tube_id: str) -> str | None:
        return self._tube_entity.get(tube_id)

    def embedding_of(self, tube_id: str) -> list[float] | None:
        eid = self._tube_entity.get(tube_id)
        if eid is None or eid not in self._entities:
            return None
        return [round(float(x), 5) for x in self._entities[eid].ema]

    @property
    def entities(self) -> int:
        return len(self._entities)


def _center(t: Tube) -> tuple[float, float]:
    return ((t.box.x1 + t.box.x2) / 2.0, (t.box.y1 + t.box.y2) / 2.0)
EOF_VI
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
    Column("quality", String), Column("quality_reason", String), Column("embedding", JSON),
)
entities = Table(
    "entities", metadata,
    Column("entity_id", String, primary_key=True), Column("camera_id", String), Column("named", String),
    Column("class_label", String), Column("tube_ids", JSON), Column("first_seen_ms", BigInteger, index=True),
    Column("last_seen_ms", BigInteger), Column("best_keyframe_ref", String), Column("embedding", JSON),
    Column("quality", String),
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
cat > vi/fusion/__init__.py << 'EOF_VI'
from .tiles import TileMap, camera_affinity, discover_tiles
EOF_VI
cat > vi/fusion/tiles.py << 'EOF_VI'
"""Tile discovery (R14): which cameras look at the same physical area, learned from the cameras'
own output. Two overlapping cameras produce the same-looking person at the same time, again and
again; adjacent cameras see the same person only in sequence. The affinity of a camera pair is
the rate of simultaneous cross-camera appearance matches; pairs above the bar are one tile.
No configuration, no seams, no floor plan."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass
class TileMap:
    tiles: dict[str, list[str]] = field(default_factory=dict)     # tile id -> cameras
    affinity: dict[str, float] = field(default_factory=dict)       # "camA|camB" -> rate
    evidence: dict[str, dict] = field(default_factory=dict)
    version: int = 1

    def tile_of(self, camera_id: str) -> str:
        for t, cams in self.tiles.items():
            if camera_id in cams:
                return t
        return camera_id                                            # unassigned camera: its own tile

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"tiles": self.tiles, "affinity": self.affinity, "evidence": self.evidence, "version": self.version}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "TileMap | None":
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(tiles=d.get("tiles", {}), affinity=d.get("affinity", {}), evidence=d.get("evidence", {}), version=d.get("version", 1))


def camera_affinity(entities: list[dict], sim_thr: float = 0.85, min_overlap_ms: int = 1500) -> dict[str, dict]:
    """entities: [{entity_id, camera_id, first_seen_ms, last_seen_ms, embedding}]. For every camera
    pair: matched = entity pairs (one per camera) that overlap in time and agree in appearance;
    opportunities = entity pairs that overlap in time at all. rate = matched / opportunities,
    and coverage = matched / min(entities on A, entities on B) so a rare-but-perfect overlap
    counts less than a consistent one."""
    by_cam: dict[str, list[dict]] = {}
    for e in entities:
        if e.get("embedding"):
            by_cam.setdefault(e["camera_id"], []).append(e)
    out: dict[str, dict] = {}
    for a, b in combinations(sorted(by_cam), 2):
        matched = opp = 0
        for ea in by_cam[a]:
            va = np.asarray(ea["embedding"], np.float32)
            for eb in by_cam[b]:
                overlap = min(ea["last_seen_ms"], eb["last_seen_ms"]) - max(ea["first_seen_ms"], eb["first_seen_ms"])
                if overlap < min_overlap_ms:
                    continue
                opp += 1
                vb = np.asarray(eb["embedding"], np.float32)
                if va.shape == vb.shape and float(va @ vb) / (float(np.linalg.norm(va)) * float(np.linalg.norm(vb)) + 1e-8) >= sim_thr:
                    matched += 1
        n = min(len(by_cam[a]), len(by_cam[b]))
        out[f"{a}|{b}"] = {"matched": matched, "opportunities": opp, "rate": round(matched / opp, 3) if opp else 0.0,
                           "coverage": round(matched / n, 3) if n else 0.0, "entities": [len(by_cam[a]), len(by_cam[b])]}
    return out


def discover_tiles(entities: list[dict], cameras: list[str] | None = None, sim_thr: float = 0.85,
                   min_rate: float = 0.5, min_coverage: float = 0.3, min_matched: int = 2) -> TileMap:
    """Union cameras whose pair affinity is high enough; every other camera is its own tile.
    Thresholds are deliberately conservative: a wrong merge makes two areas one; a missed merge
    just leaves query-time fusion (W-ids) to join the person."""
    aff = camera_affinity(entities, sim_thr)
    cams = sorted(set(cameras or []) | {e["camera_id"] for e in entities})
    parent = {c: c for c in cams}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for key, v in aff.items():
        a, b = key.split("|")
        if v["matched"] >= min_matched and v["rate"] >= min_rate and v["coverage"] >= min_coverage:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for c in cams:
        groups.setdefault(find(c), []).append(c)
    tiles = {f"T{i + 1}": sorted(members) for i, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1])))}
    return TileMap(tiles=tiles, affinity={k: v["rate"] for k, v in aff.items()}, evidence=aff)


def entities_for_affinity(engine) -> list[dict]:
    """Rows from the store in the shape camera_affinity wants (one row per camera-entity)."""
    from sqlalchemy import select
    from vi.store.db import tubes
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.class_label == "person"))]
    by: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["camera_id"], r["entity_id"])
        e = by.setdefault(key, {"entity_id": r["entity_id"], "camera_id": r["camera_id"], "first_seen_ms": r["born_ms"],
                                "last_seen_ms": r["last_seen_ms"], "embedding": r.get("embedding")})
        e["first_seen_ms"] = min(e["first_seen_ms"], r["born_ms"]); e["last_seen_ms"] = max(e["last_seen_ms"], r["last_seen_ms"])
        if not e.get("embedding") and r.get("embedding"):
            e["embedding"] = r["embedding"]
    return [e for e in by.values() if e.get("embedding")]
EOF_VI
cat > tests/test_reid.py << 'EOF_VI'
import numpy as np
import pytest

from vi.reid import HistogramEmbedder, crop_for_embedding
from vi.schemas import Box, CamTime, EventType, Tube, TubeState
from vi.tubes import TubeLinker


def tube(tid, x, state=TubeState.active, t=0, h=120):
    return Tube(tube_id=tid, camera_id="c1", class_label="person", state=state, born=CamTime(cam_utc_ms=t),
                last_seen=CamTime(cam_utc_ms=t), box=Box(x1=x, y1=100, x2=x + 40, y2=100 + h))


def unit(seed, dim=48):
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_histogram_embedder_is_normalised_and_separates_colours():
    e = HistogramEmbedder()
    red = np.zeros((120, 40, 3), np.uint8); red[..., 0] = 220
    blue = np.zeros((120, 40, 3), np.uint8); blue[..., 2] = 220
    v = e.embed([red, blue, red.copy()])
    assert v.shape == (3, 48) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[2] > 0.99 and v[0] @ v[1] < 0.6
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert crop_for_embedding(frame, Box(x1=100, y1=100, x2=140, y2=220)).shape[0] > 120  # padded


@pytest.mark.edge("E-TUBE-04")
def test_newborn_relinks_to_recently_lost_entity_with_matching_appearance():
    lk = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.75, max_gap_ms=30_000, max_jump_px=400)
    a = unit(1)
    t1 = tube("c1:0:1", 300)
    assert lk.on_birth(t1, a, 0) is None and lk.entities == 1
    t1.state = TubeState.lost
    lk.on_close(t1, 5000)                              # hidden behind a colleague
    t2 = tube("c1:9000:2", 340, t=9000)
    ev = lk.on_birth(t2, a * 0.95 + unit(2) * 0.05, 9000)
    assert ev is not None and ev.type == EventType.relink and ev.subject_tube_ids == ["c1:0:1", "c1:9000:2"]
    assert lk.entity_of("c1:9000:2") == lk.entity_of("c1:0:1") and lk.entities == 1 and lk.relinks == 1
    # a different-looking person at the same spot starts a new entity
    t3 = tube("c1:9500:3", 300, t=9500)
    assert lk.on_birth(t3, unit(7), 9500) is None and lk.entities == 2


@pytest.mark.edge("E-TUBE-04")
def test_relink_refuses_long_gaps_far_jumps_and_exited_tubes():
    a = unit(1)
    lk = TubeLinker("c1", sim_thr=0.75, max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 1000)
    assert lk.on_birth(tube("c1:20000:2", 300, t=20000), a, 20000) is None            # too long ago
    lk2 = TubeLinker("c1", sim_thr=0.75, max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk2.on_birth(t1, a, 0); t1.state = TubeState.lost; lk2.on_close(t1, 1000)
    assert lk2.on_birth(tube("c1:2000:2", 900, t=2000), a, 2000) is None                # 600 px jump
    lk3 = TubeLinker("c1", sim_thr=0.75, exited_sim_thr=0.85)
    t1 = tube("c1:0:1", 300); lk3.on_birth(t1, a, 0); t1.state = TubeState.exited; lk3.on_close(t1, 1000)
    b = unit(9); b -= (b @ a) * a; b /= np.linalg.norm(b)                     # orthogonal direction
    weak = a * 0.8 + b * 0.6                                                  # cosine exactly 0.8: enough for lost, not for exited
    assert 0.75 < float(a @ weak) < 0.85
    assert lk3.on_birth(tube("c1:2000:2", 300, t=2000), weak, 2000) is None            # exited: needs stricter match
    assert lk3.on_birth(tube("c1:2500:3", 300, t=2500), a, 2500) is not None           # identical look: relinked


@pytest.mark.edge("E-TUBE-08")
def test_gallery_follows_appearance_drift():
    lk = TubeLinker("c1", sim_thr=0.8, near_sim_thr=0.8, ema_alpha=0.5, exemplars=3)
    base = unit(3); drift = unit(4)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, base, 0)
    steps = [(base * (1 - k) + drift * k) for k in (0.2, 0.4, 0.6, 0.8)]
    for s in steps:
        lk.on_refresh(t1, s / np.linalg.norm(s))
    t1.state = TubeState.lost; lk.on_close(t1, 5000)
    final = drift * 0.9 + base * 0.1
    ev = lk.on_birth(tube("c1:6000:2", 300, t=6000), final / np.linalg.norm(final), 6000)
    assert ev is not None                        # linked to the drifted gallery, not the birth look
    assert float(base @ (final / np.linalg.norm(final))) < 0.8      # which plain birth-embedding matching would miss


def test_pick_features_unwraps_every_transformers_return_shape():
    from types import SimpleNamespace
    from vi.reid import pick_features
    t = np.ones((2, 4))
    assert pick_features(t) is t                                                   # bare tensor
    assert pick_features((t, "aux")) is t                                          # tuple
    assert pick_features(SimpleNamespace(pooler_output=t, last_hidden_state=None)) is t   # output object (current)
    assert pick_features(SimpleNamespace(image_embeds=t)) is t                     # older naming
    class T:  # last_hidden_state only: mean-pool over tokens
        def __init__(self, a): self.a = a
        def mean(self, dim): return self.a.mean(axis=dim)
    out = pick_features(SimpleNamespace(pooler_output=None, image_embeds=None, last_hidden_state=T(np.ones((2, 5, 4)))))
    assert out.shape == (2, 4)


@pytest.mark.edge("E-TUBE-04")
def test_near_reappearance_relinks_at_the_relaxed_bar_and_colour_gate_blocks_wrong_pairs():
    a = unit(1)
    b = unit(11); b -= (b @ a) * a; b /= np.linalg.norm(b)
    weak = a * 0.7 + b * 0.714                               # cosine 0.70: under 0.75, over 0.65
    lk = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.65, near_gap_ms=5000, near_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 3000)
    assert lk.on_birth(tube("c1:4000:2", 340, t=4000), weak, 4000) is not None       # 1 s, 40 px: relaxed bar applies
    lk2 = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.65)
    t1 = tube("c1:0:1", 300); lk2.on_birth(t1, a, 0); t1.state = TubeState.lost; lk2.on_close(t1, 3000)
    assert lk2.on_birth(tube("c1:20000:2", 340, t=20000), weak, 20000) is None       # 17 s later: full bar, refused
    # colour gate: same SigLIP-ish look, different vest colour -> not linked
    green = np.zeros(48, np.float32); green[[3, 11, 19]] = 1; green /= np.linalg.norm(green)
    orange = np.zeros(48, np.float32); orange[[5, 9, 21]] = 1; orange /= np.linalg.norm(orange)
    lk3 = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.65, aux_thr=0.5)
    t1 = tube("c1:0:1", 1200); lk3.on_birth(t1, a, 0, aux=green); t1.state = TubeState.lost; lk3.on_close(t1, 4000)
    assert lk3.on_birth(tube("c1:5000:2", 1210, t=5000), a, 5000, aux=orange) is None    # identical embedding, wrong colour
    assert lk3.on_birth(tube("c1:5500:3", 1210, t=5500), a, 5500, aux=green) is not None  # same colour: linked


def test_quality_uses_the_largest_observed_height():
    from vi.tubes import grade_tube
    from vi.schemas import Box, CamTime, Tube
    t = Tube(tube_id="c1:0:9", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0),
             last_seen=CamTime(cam_utc_ms=9700), box=Box(x1=600, y1=200, x2=620, y2=241), max_height_px=160)
    grade_tube(t, 1280, 720)
    assert t.quality == "ok"        # a 41-px final box after a 160-px life is not "tiny"


@pytest.mark.edge("E-TUBE-04")
def test_newborn_absorbs_an_occluded_live_tube_of_the_same_person():
    """The hard-hat man: his tube goes occluded at 3.0 s, he is re-detected at 3.7 s as a new
    tube while the old one is still alive; the link must happen and the ghost must be dropped."""
    a = unit(1)
    lk = TubeLinker("c1", sim_thr=0.88, near_sim_thr=0.85)
    t1 = tube("c1:0:1", 700); lk.on_birth(t1, a, 0)
    t1.state = TubeState.occluded; t1.occluded_since_ms = 3000
    lk.on_state(t1, 3000)
    t2 = tube("c1:3667:2", 720, t=3667)
    ev = lk.on_birth(t2, a, 3667)
    assert ev is not None and ev.payload["absorbed_tube"] == "c1:0:1" and ev.payload["threshold"] == 0.85
    assert lk.absorbed == ["c1:0:1"] and lk.entity_of("c1:3667:2") == lk.entity_of("c1:0:1")
    # once the tracker drops the ghost, the entity is live again and not a candidate
    lk.absorbed.clear()
    t3 = tube("c1:4000:3", 720, t=4000)
    assert lk.on_birth(t3, a, 4000) is None            # nobody to link to: the entity is seen


def test_tracker_drop_returns_a_dead_tube():
    from vi.detect import Detection
    from vi.schemas import Box
    from vi.tubes import ByteTracker
    tr = ByteTracker("c1", confirm_ticks=1)
    live, _ = tr.update([Detection(box=Box(x1=0, y1=0, x2=40, y2=120), class_label="person", confidence=0.9)], 0)
    dead = tr.drop(live[0].tube_id)
    assert dead is not None and dead.state == TubeState.dead and tr.drop("nope") is None and tr._tracks == {}


@pytest.mark.edge("E-TUBE-04")
def test_dying_duplicate_merges_into_the_live_lookalike_but_long_coexistence_does_not():
    a = unit(1)
    lk = TubeLinker("c1", sim_thr=0.88, near_sim_thr=0.85)
    t1 = tube("c1:0:1", 700); lk.on_birth(t1, a, 0)                       # the person
    for t in (250, 500, 750, 1000, 1250):
        t1.state = TubeState.active; t1.last_seen = CamTime(cam_utc_ms=t); lk.on_state(t1, t)
    t2 = tube("c1:1000:2", 720, t=1000)                                     # a second box on the same body
    lk.on_birth(t2, a, 1000)                                                # t1 is active: no candidate -> new entity
    assert lk.entities == 2
    t2.state = TubeState.lost; t2.last_seen = CamTime(cam_utc_ms=1500)      # dies after 0.5 s of overlap
    ev = lk.on_close(t2, 1500)
    assert ev is not None and ev.payload["kind"] == "merge_on_death" and lk.entities == 1 and lk.merges == 1
    assert lk.entity_of("c1:1000:2") == lk.entity_of("c1:0:1")
    # two people who co-existed for 8 s are not merged however alike they look
    lk2 = TubeLinker("c1", near_sim_thr=0.85)
    p1 = tube("c1:0:1", 700); lk2.on_birth(p1, a, 0)
    p2 = tube("c1:0:2", 760); lk2.on_birth(p2, a, 0)
    for t in range(250, 8001, 250):
        p1.last_seen = CamTime(cam_utc_ms=t); p1.state = TubeState.active; lk2.on_state(p1, t)
    p2.state = TubeState.lost; p2.last_seen = CamTime(cam_utc_ms=8000)
    assert lk2.on_close(p2, 8000) is None and lk2.entities == 2


@pytest.mark.edge("E-TUBE-04")
def test_tile_linker_joins_the_same_person_across_cameras_but_not_two_people_on_one_camera():
    from vi.schemas import Box, CamTime, Tube
    def tb(tid, cam, x, t=0):
        return Tube(tube_id=tid, camera_id=cam, class_label="person", born=CamTime(cam_utc_ms=t), last_seen=CamTime(cam_utc_ms=t),
                    box=Box(x1=x, y1=100, x2=x + 40, y2=220))
    a = unit(1)
    lk = TubeLinker("T1", cross_camera_sim_thr=0.85)
    t_a = tb("cam01:0:1", "cam01", 300); assert lk.on_birth(t_a, a, 0) is None and lk.entities == 1
    t_a.state = TubeState.active; lk.on_state(t_a, 500)
    t_b = tb("cam02:500:1", "cam02", 900, t=500)                         # the same man from the second camera, at the same time
    ev = lk.on_birth(t_b, a, 500)
    assert ev is not None and ev.payload["cross_camera"] and lk.entities == 1
    assert lk.entity_of("cam02:500:1") == lk.entity_of("cam01:0:1") == "T1:E1" and lk.absorbed == []
    t_c = tb("cam01:600:2", "cam01", 700, t=600)                         # a look-alike on cam01 while E1 is live on cam01: a second person
    assert lk.on_birth(t_c, a, 600) is None and lk.entities == 2
EOF_VI
cat > vi/episode/writer.py << 'EOF_VI'
from __future__ import annotations

import hashlib
from pathlib import Path

from vi.schemas import CamTime, EnrichmentPatch, Event, Provenance, Tick, Tube
from vi.schemas.episode import (CastMember, EpisodeClose, EpisodeHeader, EpisodeStatus,
                                EventRecord, PatchRecord, TickRecord, TubeRecord, episode_record_adapter)


def episode_id_for(tile_id: str, t0_corrected_ms: int) -> str:
    """E-STO-02: deterministic id => a retried open() never creates a second episode."""
    return "ep_" + hashlib.sha1(f"{tile_id}|{t0_corrected_ms}".encode()).hexdigest()[:16]


class EpisodeWriter:
    """Append-only JSONL, one file per episode. Records are idempotent by id (E-STO-02);
    patches after close are accepted and appended (E-STO-01)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, EpisodeHeader] = {}
        self._closed: set[str] = set()
        self._seen: dict[str, set[str]] = {}

    def path(self, episode_id: str) -> Path:
        return self.root / f"{episode_id}.jsonl"

    def _append(self, episode_id: str, record_key: str, record) -> bool:
        seen = self._seen.setdefault(episode_id, set())
        if record_key in seen:
            return False
        seen.add(record_key)
        with self.path(episode_id).open("a") as f:
            f.write(record.model_dump_json() + "\n")
        return True

    def _resume(self, episode_id: str) -> None:
        """A writer restarted after a crash must not append a second header or duplicate ticks:
        rebuild the idempotency keys from what is already on disk."""
        path = self.path(episode_id)
        if not path.exists() or episode_id in self._seen:
            return
        seen = self._seen.setdefault(episode_id, set())
        for rec in self.read(path):
            k = rec.kind
            if k == "header": seen.add("header")
            elif k == "tick": seen.add(f"tick:{rec.tick.camera_id}:{rec.tick.tick_index}")
            elif k == "event": seen.add(f"event:{rec.event.event_id}")
            elif k == "patch": seen.add(f"patch:{rec.patch.patch_id}")
            elif k == "tube": seen.add(f"tube:{rec.tube.tube_id}")
            elif k == "close": seen.add("close"); self._closed.add(episode_id)

    def open(self, tile_id: str, camera_ids: list[str], t0: CamTime, provenance: Provenance) -> str:
        eid = episode_id_for(tile_id, t0.corrected_ms())
        self._resume(eid)
        if eid in self._open or eid in self._closed:
            return eid
        hdr = EpisodeHeader(episode_id=eid, tile_id=tile_id, camera_ids=camera_ids, t0=t0, provenance=provenance)
        self._open[eid] = hdr
        self._append(eid, "header", hdr)
        return eid

    def write_tick(self, episode_id: str, tick: Tick) -> bool:
        return self._append(episode_id, f"tick:{tick.camera_id}:{tick.tick_index}", TickRecord(episode_id=episode_id, tick=tick))

    def write_event(self, episode_id: str, event: Event) -> bool:
        event.episode_id = episode_id
        return self._append(episode_id, f"event:{event.event_id}", EventRecord(episode_id=episode_id, event=event))

    def write_patch(self, episode_id: str, patch: EnrichmentPatch) -> bool:
        return self._append(episode_id, f"patch:{patch.patch_id}", PatchRecord(episode_id=episode_id, patch=patch))

    def write_tube(self, episode_id: str, tube: Tube) -> bool:
        return self._append(episode_id, f"tube:{tube.tube_id}", TubeRecord(episode_id=episode_id, tube=tube))

    def close(self, episode_id: str, t1: CamTime, status: EpisodeStatus, cast: list[CastMember]) -> bool:
        ok = self._append(episode_id, "close", EpisodeClose(episode_id=episode_id, t1=t1, status=status, cast=cast))
        self._open.pop(episode_id, None)
        self._closed.add(episode_id)
        return ok

    @staticmethod
    def read(path: str | Path):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                yield episode_record_adapter.validate_json(line)


def should_soft_cut(cast_prev: set[str], cast_now: set[str], duration_ms: int,
                    max_duration_ms: int = 30 * 60_000, churn_thr: float = 0.6, min_duration_ms: int = 120_000) -> bool:
    """E-EVT-07: a busy tile never goes quiet, so an episode is cut when the cast has
    mostly turned over or the episode exceeds max duration. Churn cuts need a minimum length:
    on a busy camera the cast turns over every few seconds, and 800 two-second episodes an hour
    are not episodes (session 31's 16-camera run)."""
    if duration_ms >= max_duration_ms:
        return True
    if not cast_prev or duration_ms < min_duration_ms:
        return False
    churn = 1.0 - len(cast_prev & cast_now) / len(cast_prev | cast_now)
    return churn >= churn_thr
EOF_VI
base64 -d > ui/web/assets/avatar.jpg << 'EOF_B64'
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwMDAgQDAwMEBAQFBgoGBgUFBgwICQcKDgwPDg4M
DQ0PERYTDxAVEQ0NExoTFRcYGRkZDxIbHRsYHRYYGRj/2wBDAQQEBAYFBgsGBgsYEA0QGBgYGBgY
GBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBj/wAARCADwAPADASIA
AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA
AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3
ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm
p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA
AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSEx
BhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElK
U1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3
uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8/wCi
iigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKUd
aSlHWgBKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigApR1pKUdaAEooooAKKKKACiiigAooooAKKKKACiiigAop8cTyuqRozMxwqqMkn2r0Hw78Ef
iN4iSKaHw/JY20gyLjUWFuuPXB+Y/gKzqVYU1eckl5kTqwpq83Y87xSgEnivpnQv2WLNGSXxJ4pk
mGPmg06HYM/775/9Brv9O+Avwu04KT4de9cD715dSPn8AQP0ry6ueYWm7Jt+i/zsefUzfDw2d/Q+
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
/9k=
EOF_B64
base64 -d > ui/web/assets/cam.png << 'EOF_B64'
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
EOF_B64
base64 -d > ui/web/assets/dp-bot-unsure.png << 'EOF_B64'
iVBORw0KGgoAAAANSUhEUgAAAKAAAACgCAIAAAAErfB6AAAzK0lEQVR42u19d5wcV53n7/deVXWc
3JODZjTKOctyEg7ghG1hbKKNAcPCYsICyy372b0lLbcLt8Aee3uwcGcO7gNrwgKyzYKjJFuyZcuW
bVmSZWVpNEmTQ6eq997v/njV1dU9M/JYo9GMpCn7Y2taPdVd75e/vwRwSV4IM9dFdLIz5JxGh44z
3HxxfAG8SKmCF92BzFwXHZvO8NqMTzBzzVwXCz/jtDwGnNEYM4x6IX17fLOvOhN/X2BchZeYVOAl
S+mZa4Zm0+XL4AylZ65L+pRw5nxn6DpzzVwzvDVDiWlM6RmeuxRPDWcYYqyn9z8/5fxvCohEFyXv
4nk7UgRE1EQlBURjfiyyDOUJiGiqSH5ODhCnO1Um/vyIwAAISOV+ZQZGgBtBjpxpWiqhZFqKlMy/
A8dRfv3isz4XEFEBABGA5RDGKjCLmgpK5hQWzS4sqImESgJm2DBCBnJEQABUjpRpaQ85yb70UFu8
7+hA38HB/qODMi1dRsFzQObzfIwXow1GQIYk3WOM1oSrL6usWVdRNq84EgvxAAciJRUoIEUkifSv
EAADRGScaRVNQCKthtsSbbs7jz1+qvvVXq29p7/SzifwmypxugBJywO8/qqqxrfXVq6IhUpDIEmk
pHIUKQCkjIeFgAhEpEUYkEDLKAEBATAEZjEjaEhHtT3f+epPD3S/1ueS+QLR2BePk4XcJW2wNNB8
a+OcG+tLZheAApEUUigARIaj2WfKOwpNbldIiYgACBDBilpSqH2/PvTKv72ubHWh0PhiUNGIqGlh
FZjz75w9f1NTYW1UJoWTFASILP+JiQgRPe01Br8REBJ4xEaShIwCxVbnnt5t//WFeFtinDSeOAdP
5A4XPIG9U266uX7FhxeWNBU4cUfaChlChvD+Q0IAwlHDYP0Djm5gCQCIAEhAoMjobxt66os7B48P
T385xun5ncZ5ZoyhUlRQG1n7hWUNV1WrpHRSAjnTMu2jDnmPSlnSku/xM1YYiMD/d674+lEQactA
1BzqSj76yW2J06lpTmM+DdkB34rsNlxTc90/bojNL7EHHKWIMUQdJPnI58oyoNbMGcEdvZDLfaun
/dH9NfcFAuRMpmWkLFg8v+jY463T3AXlcGFemrpL7p13+V+t5AydhECGGcvqU72YGy9gvtrNvoOy
cu0ncEaSwa/WkYGTEGVzi8HA9udPayRkeirVC5LAyJEUrbx/0eo/W6QFFxnSCJuaJ465f+fp5DHP
NcMNmPMTZSSbo0zI6tXl7S93aYfrPIjyWdCeXXDUZRxJ0tL75q/8yIJUTxoQAZH8TnJWL2cUMmrl
mnWvAJAy0j7WORL5LThRrmOGwAARFKz53DIWYOfHgNFFL8GMo5I0+5aGyz6/LN1nA8+xj7nmMyvQ
5Glbl+qg/5y10yNkPyeKQo+8nhOuFT7KtCyaVZAcSHfv6fUr6unju7JJ0gyTZHeVpNIFxRv+crkY
FsAy1KVRDSvqn3JcaHLfQFmR9ILenCcldN9DkM07adgDs6EWIUc5LJe8b55VbIEixLcgauP1JSem
zC8cCUZARCNkXPvt9ZGykLAVMlfxYp4LjG8WaxHluVvjUIqY72BneE46KhwLpobSp1/p8YT4ApPg
aQJXkaLlH19QvrDUHnYyuCNl3V10fV4kv1M8qu+DQPk+FxBBfoaYMqoaXV0ASDlICGY8amf+7U1W
oUXKvS/NEPgsgqKyhcWL3t2c7rfRYH5DqZUxenoZAYEwo5yzxKIcmCP/rzzBo9wIinKMspZ/n9IG
QFBpWVRb0HRDPRCMgnjPEHic14qPL+QGU5LyghvUzpMvpM2Bq9CT0Hxtiz5yklu+QVoD5ARFGTea
/FKujbNONjKUadl8UwNyhGmAauFYBJ6ewLSOeqvXlddtqLSHHcb9SQLC3AA1q69zMCw3L5j7TiTy
+dwjPOgRzjRlXW5EQMqqD4ZOUpTNL6pYGSMC5FN8kDQWgacp6KYAABa/fy4Saokh8Cvjkd87I5lA
OjbCEeLuyzxQPrhBowMjrklH8COdfrfLMIzZ76ifPueIF4SK1hUUscUlVati9rCDjOVQF3KgSX8d
XTa1QD5ZxGyIRF4A5cNJdOrfb6/97laeKKPvmzDOREpWra4wIgYpwnF46Tj51L1gbPDc2xqNgOEm
bbwYFj1HOQ9dpAyFMMd7gtwI2A86IyFkE8uZf7PvxjzOIT+y4r4sbFlQE6lcERvnuU6qnHsMys6/
0ngroRGQokCxVXt5pUg6wDDPRGZRC8jGvuTBGVlC5XtlI02tW92RZ7szKcQM7kH+D3bZzLX4BESM
Y+0V1dPUiz4PjsFb5lmGAFB3VVWkIiRsBYg5sarSxbBAijxK+zAOooyXSz5ppTF5z0Ob/UaWPMwT
84Nj7aVlZR0ZyLSsWFKKBiNJ08RlZdPaw1IAAI3X1Ok/aBEioYgADWaETDNimRHTDJrMYEBAkkgq
IADEbDhDHqUpB6IkH8Th0+8wWo4pj0FGvOqKskyL4vqCosaCPP09hZcxqQp5IkyDiEQUrgyVLyoV
CQEIpABNNEOmsinVa6f7bHBQKWlGzECxaRVaRoiRUjKtpCOJKJNQylJaK19P0yJhLmbpc59RR1G6
KMv3KJivBrJ+PQApMgp5bElJ/+GB8ybBZz5n49xSy18WQxNXLhJq1lYES8xUnw0IzEJ7SB783fFT
T7f3Hup3EsJ7rxW1ihoKYotLqlaVVyyJBcssAhIJRwmFTNdTenEw5RtjyEW4fM6xK/ZZpHKEizAy
z6GgYlnZ4d8fH8/zn5OKUjrjbY2zvsuke4YEAFCzvpwkEZERNPtODD756R3pIRsAgqWBouYCM2go
SekBO9GV7Nrf07W/5/VfHw4WBWrWVzZeX1+9siJUwp2UI23JWAa+dJVD7td1EWwk12Gmkbqa8MxP
S4QAiMKWZfOKmcmUo6bKkaZxSvBUlrwjkCIzasYWl8q0dJvGEAMlVt3VVXNunVXUELGili6Zcmxp
DztD7cO9bwx27OrqeOn00cdOHn3sZPGswlnX1s56W03RrALpCGQsk2agEerGjYZHTU3k5SBwFGF2
X0ZE5VC0OhqtCQ+emMqay/zao+mIbyiqWhN7xz9f7sSlpgu3GBBxiyGgSEmSbnoWEBhn3GJoMhA0
3Jlo3d117E8tnS926bvd+K8bK1eUiLgE5iWbcFQMA7MvuaHPWFzuO0HM1HwQAJKiQFFg6399/vhj
LV4t/rRzss6b7I55fAgEEFtawkwOpJAhAClbIkOZIEXKxSBZJsKR5CSEpkyoJDDv1llz3lHfc7jv
5Jb23oMDZhFXkkhrWfIXemQJ6sOo3GT+CE9Cl8G7OQwf1Jk14G4KCym2uOT4Yy3T14umSabfm36Q
Pr/SOUUypUD/46aEgUAhYi42BYDAGAIgEUmhRJ9EhmVziisWlgGASktlK8R8nZyt9cAMqIkaqxzF
efJgaC3WGhehkc+DoBwVm18CMC06To1JFdmzu6cGsMyIWbu6OhixbC6YwZEhASlSoIAESEdIodxS
SpZBmXUghIgGAoBISZGQjLHMGygLNGIuskWEiCzAVVoqpXwYFuYkJHwxElKWvkRuBtqNhm0ZrQkH
SgLpvnS+p30xxcFnHf4yxhDRtIz9PzuS7k8LR5hRywgaRoSHY6FIVSQUCwRLrWBRAAwgoVRaSSEJ
ABn6OJUYY8D8ngZ6atYT2EyAjMBYvD0VrgwGw6ZyiCQpIlJKSzl5EKj7IymS7gfq9IRXZEtIQoVK
AoX1BV19aTinFD4LOTSmD4zFOQcAKaWUEgBEn3j1F/vHenOoPBCbW1a2sKRyeay4uSBUGgAAkRBS
KGTkpfIoD5gYBdYAJSlYbO399eEXv/tq7fqquiurC2ujwWKLBzkPcMYZ4ww5ImeAAIw4Z8xk3GAM
mRBCpATJjInWVl6hEeAlzYVde7rzQ+eJHTW9dYYwJo933pLUIqKma1FR0dKlS5ctW9bU2FQaKy0s
KjQNM5lMxuPxgYH+lpaWI0eOHjty9NiRYy3PtrU82wYAVqnZuL5u1sb68uVlgWLTSaaVoxjXGXmC
rD+MnshmPSWNIdsiUhkIlFqnnms/9Vx7lucCnJuM6X8Nxg3GLcYCPFBoFVYXFFRHIg3hyiXl4bJA
ajiFiOQGYUQERbOjZ6bQefBkaTqESYwxbfY2Xn31hz/ykeuvv76uru7MvyKlbGtr27d/34sv7tq2
7enntj8XT8YBIFodmXNT05zbZhXVRtNDKSUkMpbnJ2eyv+T3uJRURoinB53WFzu69/bH2xKJ3pQz
LERCKKGUIJJKCkViFIqEY+HF75uz4L2zVUrqoEoRmWGj9cWup/5ihwZcp9LkTQfq1tfXf/e7373z
zjvP7iYtLS1PPPnEgw8++NijjwFAqCy49P0Lm29rMMNMxB3k6IW6iPkoRaYOC0ERcmRBZIwpSdIh
JZRyFChQSikhSSolSDqKBDhJEe+ID7cm+w4OnHymrXRe0a0/udaOO16nMrPYUEfiD/dukWk5tX4W
Tjl1ly9f/tDmhxpmNXjVym1tbYcOHTp+/Hh7R3sinlBKMcYKCwvLY7Ga2tq62trq6prikuKRN9y9
e/cP/+2HD/z4/0hSsXml67+4snJFabI/wRijbNLPLbfwOvgz6WPSpoIhEiNdSaLbXZRSgEBInDHG
GCExzhhHZIwU9L7RZ4R4KBZUwpsOQYBIAh7+6FPxtvgZhPjcamkcA5c+x17cOL80QySA8lj5rhd3
NTQ0AEAikfj3f//3Bx988MUXX+zv7x/TLTSMWCw2d+7cVatWrV+/ft26dc3Nzf437Nq162/+9m8e
f+xxwzLWfX7F3E0N9mBa94NnqnzcYElJhRx50OAWV1JRCkRCKFtJUowxNMAKWcxCNEC7CNKR5JCU
EggMzjVrWGFLSSkc6aUzNLfwEH/0M9u7Xuk+a8DynJB/yiSYcy6l/J//8j/v//T9APD000//+Z//
+f79+z3yIxvRx01ERG6c6rnTodDq1atvueWWTZs2LViwwHv9W9/61pe//GUAWPup5Yvubk4Pphjn
mRuiksRMZoYNkZL9h4c6X+05vad74OTAcHdcJCVIAA48yMIloXBJqKi+sKihoGh2YVFTQShmGUED
iGRaKoeIlCf6mVECCESkwCoyt/7tCyeeODW1gCVOSWpBH0dpaemBAwdisdjul3ZfdfVVyWTSMAxN
wjPoNC1/OlYmIu17A0AwGNy0adMXPv/5tevWaa3+29/+9oN3fzCVTK3586XL71uQ6k2RIkBkJjej
VrI3dXJr+6GHj3Tt69V3MNEsKS0pKCwwTcO2nWQiMdDfn7BTPrUDpXOKqpZUVK+qLFlYWFgdBQPs
RFrZigCQebWcQFIFSgIv/sveff/voG6Yu7RssBbfK6+8ctu2pxnDu+++++c//7llWbZtnzUwIoTQ
Cvxzn/vcN77xDdM0DcP4wx/+88673p1Kptbfv3L++5u4wRjjid7kiS3te3/++uCpYQBYu3bNjTfd
tGHDhrlz55YUl4TDYe0cpO30QN9AZ1fnsWPH9u7du/e1vXteefXo8WMurQNYt6Zm1tV1lWtiBTVh
qYQGw13YXJJVZO775eGXvvealuDJkJ9xWswpiIA1gd/5znc+/PDDALBx49Xbt+/wQuGJQGBa/Ddu
3Lh58+ZQKGRZ1sMPP3znnXfatl21NFa1qkqm5NFtJ+IdCQB43/vf+6lP3X/5hss1xvKmV3w4fvDw
wZ07d2556qkdz+xo62gHABbE5msaG2+or15Vzjik47aGMc2odeyJlmf+btfEk4Z43gPoiQq+PtB1
a9cJRxDRZz/zGS18E/loL8y1LAsAPvD+9xNROpUmoq3btq5ZuybrpqHx7jvfvePZHeS7EonEsWPH
Xt798vbt25955pldu3bt37+/o709lUrRaFdPT8/mhzbfc+89ZaVl+ra1a6re/p2r793+7ru33f6+
P95yz/ZN133vinz07LzrZDz/JoEyNrigILpv3/66urrDhw+vXLkykUgwRKnUBLUTIjDGpZS7Xti1
Zu0a27Yty5JSbtm65eDBg4WFhWtWr/HcsY62joceeejRRx/du3dva2trIpHwzL9pmgUFBRUVFbW1
tYsXL166dOmqlavmL5gfiUT8H9fR2fGrX//6f//oR6+9thcAGjfWr/zEouLGqEiL3oODf/jkFrgg
h5hOPMthcAD4xte/oQXiF7/4hRcFTZyHOOeI+JOf/ISIHMcRQpBSfvlzbIeIvv/971dWVo6Km451
89mzZ99zzz2/+PnPW1tb/TdMpVIP/OSBufPmAkCoKHjt1zfc9+JdNz/wNuQIF8m4ubMymcVFxUeP
HtVn9OCDDxYUFmoFPk6jONadNZc8++yzRCQcoZRSUglHOI7j2I6dtonoa1/7msdSnHPtqeX0Kmjc
gzHOuWEYBs/hvNLS0rvvvvuJJ59QUmlnnogGBgY+/4XP6zdc9qVVd26+xTAYwKU6d50xBgAbNmwY
GhqSQhLRa3v33njTjVk6cc4Ye6v3NE3Ts8FCCCmkksq9pJJCENGRI0c8ur4lJmaM6V/0Xrziiise
eeQRIlJSajL/9Gc/NQyDMX7ll9aFSgKXLoE9b+ttb9vY29vrqbvf/va3V155pZ9m4xQyj1p33HHH
8PCw7di2bZMiTWnHcZRUjuMQ0S9/+Svv08/SeUHknLPMl3nve97b2dlJRMlkkoj+70/+LwAEIha3
+KVFURyDxosXL962bZvfqm3duvW+++7TKOaoenjU15ctW/bjH/9YSp2vd5EvTVStSPWf//jHP3oq
ZILP4hmU+fPnHzhwQNtjIrrrrjsBgDMOM5c+IMbYZz/72ZMnTvjJ3Nfb99ijj331q1+77bbblixZ
XFpaqqMgT7ij0WhTU+O111335S9/+dFHH00l3agmnbY3P7T53ns/tGLZ8kWLFm26Y9Pmzb/X9JZS
9vb2VldVMcYmSGOP0touzJ83v7+/XwihlHp2x7Ocv8n9LyHN7R1ESUnJ/fff/9yzz2pXKMdTTaZa
W1v37dv3/PPP79i+Y+dzz73yyivHjh0bHBj0v62tre0HP/zByhUrRn7Kffd9NJVKaRf6e9/7HmQI
c04ufat//MdvaVUxHI9r9XNOeOhi4A/MNYorViz/whe/8PAjDx07fiw9Btrgv062tPx+8+8/9vGP
xUpj+g5N1zU03TCLG9ww+OpPLS1rLgaA2zfdrquChCNuv+02TZizaBQb+QvaJXznO2/VJl9nQieV
wBckE2jnRYps8UQkEG6aM7t5TnPT7KaammqtpRFRCDk0PNTe3n7yxMnDBw/te+314eQQAGAYG6+s
m/+eOWaRte3zzw6eGlr+iYVrP720+1D/tv+yq+9w38c+dt+Pf/y/lVSpdOpDH/rQf/zHf2jeEuMG
SkeFWUzDEFJ++N4PP/CTB0hRIhFfuGhRS0uLV7JyiV6jb2RnyBirXVs155ZZwXJrPPcJVFiN76i7
4u/W3PHbm+7Z+a73b7m9cFYBACy4q/lDO9911yM3fvDpW9/1+xtLm0sA4L6P3aflXgjx9a9/PRAI
eDHx2bV9cs45YwCw5amtOiJ7/vnnXbd/SuUSJ8Kzk4d6uzMp39Hw9n++outQT6rPFoPKHnZSPemh
jmGRdJjBucmMIA/HwsFYIFhuRasjgWITFKkkGUHryb9+9tS2tqq15dd+5zK0USpJkkJFob5Tg09+
dsdwR+I9733PD//XD0pKSwFg98u7v/n33/zd736ncUrtfJ05celiNYjImA5/AeDb3/72l770pXQ6
HQgE7rrrrt/85jc6rTJB8HVaqOhzXH2CQASR8tANP9oYqQwBgLJpqGUYkErnFXGDKcfNwCEwIYQS
igRKKZWjwrHw/l8d3fWdl6O14Zt+eLUVNZVNyBEQSAIP40Br/LmvvNx9oHfhwgX/+q//es011+oP
3blz589+9rOHHn6o9VRrngOYE38TqRF1Bxs2bPjqV77yjhtucGzHtMwf/ejfPvGJT+oSsKktfMfz
yU1v0a9GpShSFV72kUWBosCL339luC0OAJVLY1f+7dpQuSnSDjK3bgYYA0RSygybfceHHv2zbUqo
6797efWqsvSQYNyblUZKUiAcSMZTz/3Dyy1b2wDgzz7xZ3/xuc8tXLhIf253d/eOHTu2bt26c+fO
I0eOdHd1jXUUsVisubn5iiuu2LRp0+WXZ3OO//Sdf/rSX37JCluzNtUdevAoqOlH4GlhmQnMiOHE
haexy1eWimHRd2iwfEnZ2//HBiQAYHoRkjsqhcCIWE9+cXv7ztNLPjxvzacWpXqycw8RkEAxxkgC
GMQsPP6n9pd+8FqyJxUMBD94zwc++tH71q9dx33ZjtOnT586daq9rW1gcHBwYCBt25zzwsLC0tLS
ysrKurq66ursyBUievLJJ7/53765dctWI8Sv+Ie1Mi63/82uqV3qgNNNcCHTOxpbXvq2b21o29H5
3N+/RETr/nrF/E2znGHx1F/uPP1yz7Xfvqzhqur0oI0MSfeBkTIiZufLvY9/+umC+sgtP7mGASip
EBgw5CYzQgZIdBI2IClJCBgotgY6hg/88ugbvzsiEwoAVq9addum26677roF8xaWxcrG822H48P7
9+9/8sknH9780HM7dwJA5eryVfcvrF5f8fx39ux74OAEa7ImuFMnJ0NC47vj5PIBAhCZEfOyL6+M
VAWH2+JENPvWhkV3zE6eTgfKrNk3N5x+pafv4GDjNbVElNmLRKCAm0bXaz2AMO+O2aHSgN3nBAos
4IiKxbuTx7e0hgtDVWvKhC30CoBEXypUYK359KL572o6ta3j8J+Ov7R790u7d3/l775aXVm9dPmS
+QsWzJ49e1bjrJKi4nAkYhiGUiqZTPb09nS0tx85fOT11w8c2Pf6keNH9XevWF624N3NdRsrSSpn
QIhhOfHzmGCri3EWd6TJUSPkia+k2bfUV8wv7Xq9f/8vDloF5oqPLnQGHWSo0ipaHQGCwY641IXU
+j/uLZSTsoFg8Pjw4KmkSMl012DX/t6O3V3de3vSQ7YZNt6z+WbMVMExhsohZctILLT4g3PmvKux
91B/xwvd7S92dhxsb3+s/bHHHh8HggUVK0pr11VXry8vnVtomDw1ZEuheAiSvakpV4fGpPvD4+AJ
f28YSeIBPuedswCw9ZlOkZTz7qovrImketOEBMjNoAkAylZAiIiEmaYjBiIl6q+qfv0XRw5tPnbk
P0/ojbH6zlah2fj2uqbr64ArUDlzkvRQb5lWyKBiUWnVsrJl985L9zsDJ4eGTsXjnanE6UR6KG0n
HT2j0IwYwcJgqCxYVBeN1kZCFWYoFjCDFjlKpGQ6aQMDZjAlINmTnhSBeCta3TjnAoqjjdgft+uM
IKl6fXlpc1G8O3H0T8eR4ZxbGkVKj7lDQtROKTMwMzg0UzLBQKZl5cKyt/3TZQcePDx0Ig4cCmqj
sYUl5UvLCueECyuiqDAdT+vNDdkyZp2b4gQAMikFETJmFRiVK0qr18SQMSIFChUpIqU/iHMDEZQi
kqRsUrZMp2xkGodjQASMRMJJdqUmm35vqmLPfX8wTfiXm66vN0Nm556+/qODscUlZc3FIimAASjg
nGu/2ooakI0/yO2/Z2jHndpVFTWrYiIhGOdm2OAGV0IKW9qDDgAic8uXcyEmd4SW3jALAEooJRDI
m7eV83SCpDf0AZEBQ8ZQbwrQYRs3eKo3lepJafd6Ck/YOFeMM0FnjzKN/aFYsHJFmbCdjldOA0Ld
1VXACRQwjgoIGaT60wBgRa1RtiQRAKKemsY4R0B7yCGyEQE5R85Ix1SZEbGUM/4K/c/gDgUgRFS5
Ywz1ukpvb4t/TKmrUkgBN3miKyWSYjI6z97SURvnSTTP+F397lXlqjKrxJA29B0YAIKq5TFU2e5e
AkoNpAHAKrJybuMLAHSdGxEQKHe5OyEplfEG3CkshAQssyRHjz6kTBOaj1qYnQtPkF25cqaxlwiI
HHuP9XtG58KQ4DMzEU34u2pOr1pXbhiGk1BDLcPcZOHyoHJIj4nWVtOJ2wBghc3cFYReV58Wouz+
HG4ZRtAAABIKABnjyIGQVFqIlCQCxt1Fh5S9DeWrb3dMmjuSfKyZWvorKQJA6D8yOB2gBeM8MNF4
76mImaxiYQwU2HE7fjoZrgxZhYYUKtP3k3GLEBhnpNDDp7Qm9GlSzQ9oBMx4V7JrT3v3673x1riy
FTe5VWwVNUYrV8ZiC0qZgfawjSyfdTM7hoGyk1vI3SkMONZeHkIAImQobRo4NjwJ9vd8Eficgy/a
rQ3FgoEyS6RFZk8oIjJ38IXLBBCpCANBx8vdczY1ISIpBYwy7ajozvdFIIU8wPtODvzxI1u9YCnv
ql1fteR986vWl4ukTUoh5nxxl2PyX9MDdXAks+tHIEBusnSfM3h8aDwe1lnASpNlgydXvhGAIFgS
MINcpaUVtkKxYLwtnhpwomUh4Ui9o12knKpVZYFC6+ijJytXlC/Y1OykbCfpkARgmekMmb0LpIgH
sXJNLFoeiS0piVaFeJAJW6Z6nd5D/aee7mh9vqP1+Y75tzev/vRiwzKkLfOXc+TtUsrMQfPm0IJn
GLx5XUTMYgOtw+n+NIzDwzoLWOktHe90qfnTc8UiFaHmm+ulrcwCs2tvX+/B/qplFWXzikRS6A5u
khCOBY0i3rKt/eTTrb1v9POAUVRbZBVaSiolJLqJPZ0cpEDUmnNzQ9M1dbEFxdHKUCQWKqiOls4t
rr2ssunG+lBVoGd/f+er3X2HBxuurEVOWsFmiZnvS2X3HqJvTjTkehJWxGzZ0d66o4NxnHIVzc9a
CU8GgbnF597ahAwN0zAs49hjJ+0Bp+mGOiWlPlDGUKRE+ZKSwjkFg8eHO17uOvZES+uz7em4Ha2K
RCsiSiiQBAzdlSuEZCuRFCIlpVAkQAqSaSmSDudYvbqiYWN1177ezt3dTlo0XlMtkwqReVoQR+wn
zcKi/tHDvvYUIuBB4/XfHOk72I/sgiXwJCFsIiHrN9aEYwGREAX14fbdXZ2vdEcqIpXLSkVKj8wh
ZEylVWxe8dybmwpmhe1Bp2t/b/uu08cfb1Fpis0rNSJcpAVj3upuBIbu+midF2bIGENAEZehskDN
ZZVOXMy6sqagOqIkaV5zF34g+ePmfPLmoCV6DDEhA1K056dvpLpTvl1NMwTWC7AkmUFz1tU1Ttwx
Arx4TvHhR0607+pasKmZmcybAIoMlU2IGFtUMufGxprLKhRRz96+thc7W7a3lTaXljQVyZQcrRyK
vO3veqGVtFUwEmi6rq6gOixSMmdnB2YHao2qtHx62l1QTADc4qke+7WfHlC2mg7J9ulVd48IfUcG
6zbURCqD9rAomVUUrgwVNIQrV5Zh7iYFTTyRliAoWh2edU1dw9U18b7k6Vd6Tjx1qqSpuKi5QKaE
t8PSG6TkGx/szoIgqWRSaNmFPJpl9+mM0naLnoB6GBaBGTR63ug7tPkY5m4dniojyM+noX3zp2Ko
HNV7cKD5xlncYnbcrlxWVru+SqalH7PykCZEBgjSUTKlouWh5hsbWIi3Pttx4pnW2nWV0ZqwdHTw
k51qOGKnlau0wbcfjcC36BJo9LrPvKHE6IbyZthsfaHz1PYO/77oS0iCc9bMjRYTIMP46cTA8eFZ
19UyhmJYirRExvKXa2NW3DSFHFvIlKy9vEKB6nihK34q2XRTgxISEXw7DM+81R382pgAMbOPAwFH
UDN/6yEhgCIjbB59vKVrTw+bfA9rPBJ4vovu85fajIZnIccT205t+audIJhZaOptCHl5H3dErO/Q
GUNkmO6xl94zv3xxWdvLne0vdZphTionfqQcKuPoEaY3Q9rdC56ZHUy5+6d9qy8zOoIRUaI7AePe
YTIRNUnTjcDj3botiXFsfa7jD5/Y0ntwMBKLklQedugLTUfJOytFpmnOvaMJENqfP22YpstTWVwK
ILvPnXK3n/lu5ltA6XIY+tbaZZwAv/elgzIlKd3vjPP4afKR6vNK4PE/jJKEHPuODvzxU9v2/fKw
GbUMkyshAfMnTUKuWDKGwhaFsyJAMHgyTi56nE0mYCakwRwckjLzStG31859b06wk5lGmTNUHDO7
AhBJUnrQhmlzsXHK2fl3uEgSMhRJ8dx/f+npv30hPSisAouEb1cojdhNCQgAUshA1GKMpYds27YR
0a9WPVbwr2TI5JAou6GUshuKc3aaklcs4K1tcNdxuLuZpFLOdBm7gqMSmM4VvDxhziBFOuo9tuXU
Ix/f0v5CT6g4rGRmD0Z2NbS3BokIiCHjzCAARYqAEFhW1jHPMyLMW+OVWWeJOKqhpmxQNPrB6Lzy
dGk1o4mraDwXahnP+Pva7UqcTjzxxe2HHz9pRLi7Rp1866uy1gy5acS7E6RUQXXUMgNE5Cb5fWP8
3aRThjMwuwY+u2yFvCxwbnSEuavexwoTplb/nTMbTOeI0d5UXXODKaUObj7CDKbD3/yz9Kwrp64D
PQBQVFdgRS0lZS6NyKObf6tGvqefLRnI2R1MqHT8pDc3+H6RvFCeGXwip4Rv5cdp5GRNkIv1WVYs
K2MMyd3oOtrCbwRS0PlyLwAc+O2R1/79YCRWgIyRUp4PTDCWAKK3XNa3OpzcV1G3NyFlFgrjSBYj
YIxxi51Ddp9g5Qw7b5Sb4LRGpYgHeNP19TIpc6TR08Da5Jgs0Z3u2tMDAKn+9HP/7aWd334ZGOMB
A3yb12ls1Ur+1eLkB0loxLT4HJWguRA5BIqtKdbL55bA58FlRIZA0LCxuqQp6iRlBmH2Y4auVTaC
Zve+XntATwAH5Lj/N4ee+MIOe0jyoDFWaYd/B3AOIuK5XVlb7wu0vEVp6O0eJ2AQKgucf8x5uiBZ
Z4luEjCLLXr/HGUr9IefOat9gQiAwamd7V5QpDGTzle6HvvM04lu2wgaMGqjH2WKcWgkvpRZR6q9
a/BPdfcHWN7rGK2JTlA88JIiMDAkRXNvbyxfWOIkBTDMWjyXGK4fxAxM9qc7dnUBZKvilSTGsf/4
4Ja/fk46xAw25tm6QRBl94fjiOMejSA6NtIKXUlVWB89C82Gk6MU2bQXXyRF4fLQ8nvni4QA5lWr
58XAQARG2Op5fSDensjbg6EkMQN73+h74Z9fNSJmRogRcv1gNwFIuQWaOTPxc3a2ekXT5BpiBETl
qILaCBpsOiwuvBAkmAEQrPzUwnBZUGT0c7YjCXNACI1gj/pYShByPPzI8VM7T5tRg5TKwli+qNaj
N7kOXMZIYB5bAfirsjK/jADSkZFYKFwedF2HXAHFsTUwXYIE1r0O9W+rbr6hIT1gI2dZOIpcAnhB
K3JID9qtem/Z2FDS3v93EJBB7urQM6S4KINt5VffoR8dywBfCCQgUGiVzC0aDQMDOl9u6QVAYF0v
ESwJrPvcUmV7uDPmaE7KOECSeNA8/Vrv0MnhkXuKMIOWAELny6e7DvSaIQNUrjzRSNkao2I189Fe
kW7OLYDQwMpVsemjAacRmpErvkBEqz6zuKAmKlLC3Q2JXhbITzoiIsax5Zm2UZ/JIxRjSJJan+3k
FncXnrkQBvqq6BDyVS/5AyfMpCSyaQw/TsJQpmT16nJmTgszfI4JfM42S3NUkmZdVzvv5sZ0fzrj
+o5eDkIEzGTxnmTLdq2faWw4DACg9/V+IQRmYyCfUSW/6s5IJObFYgRIOf1Jvr9EBJFySucU1V5e
pd2CS1qCxxpzB4pCZYF1f7FUpqSvbGKUHA4BkCIjZHTs7k6eTo6nFHmoPS5TgJy5a+2yTrO3rz0n
v+iv8yByXe9sTmIEloiIKHHRB+YAG6Me8zzCwFNgg9804NOxytrPL4uWh0Va+iojR1UYrqdzckvr
eFgMGSpH+poWyDPqfmGk/M9xE8Do1nL4004ZzvBV9abj6arlsUUfmEuSGGeTJB7jUZzs3H4kTfg9
eiRW822zZr+9Pj2QRj4KeuwvoSUCHuSDp+Jtz58GnUIem7R6QFPFihgLgJKUy2s0xrTGXHcuRx8r
H2FzQidkTAyJ1Z9YUntllRLKh66cb9B36r1ozBMvSUVNBes+s9QZdvzrHrPVbegPQEnXyp/a3qY3
yY5KI7clXJESqmZDxepPLpYp6U/0kt+xGtFz5FfClG1az32ZfIxIgIiKlLLFxq+vq7m8UgnFOE5J
+mF6ta4gIrf4xn9YW1gV0cqZsjGTD2rKNKXoQjxF8OL39yS7U5jbKqKbTzU10MCa9RWrPr1k2Yfn
cc5IZiuwvL4UytUN/u9FkFsPn30rgr9WBCFTgI2IoKRiDGff0JAcsHv292nn8TwXSxvTh76MoZK0
4v6FlctK030O45z8CIS7nJ0Q0QOZSJERNjpe7e090K9HfGSFD5EUAUGoPDj7hvqmt9eVNBdwxtPD
DgG569i9OmfKaUQZS3GiB7FkC/b8DEWZhnT3XYwzKYmTuuKvVpUtKH7pX15zhhytgc4igjq7YQHT
hcA6Lmq6uWHxe+bY/Rq08rRzJtWabRTKIovMNE480QIEwN1pGO7oQKLChuiCO5tnXVcTiQVlmkRC
OOQtfc/DLlz2GVHNngM6++mY6TPPWmTEkaAYMsYUqXR/av5tjRXLS/Y8cPD4oy26vj/ri09mCDot
stLa9ymZW3TjD67SlaeU2dMNlO0P8mMLWoaZwRJ96T985Cl7wM6QgYggFAsuuXvenJsbAkWWiAvh
SETQpM30bvunr/nGc6DPEhONdlCUHwdQZrBL/o4zDaVkymwlGSGDB/ip5zv3PHCg69Ue78EvYBs8
rnoiBEQ0w+Y1/319pDQo0woyHdyYO8wq82MGolAUKAwc/s/jLVvbGEPIHNa8O5qu+tra+suqlKOc
uCCATA2XewdEJH8HTE6PGI72rf1y7bPHmVEdmNPrlMU8fL4AImNKkEjJ0sai2Tc1BCsDXXt6ZUq6
XsIUEhgnXzmTosv+y4qGy6vTgzbwTONehrpeNXreJCtkqCS88J1XUz0p5IwkRWrCV3119dIPzOOM
2cOO7gzOA7ezuLHXpo/+cd84QiO6I1d8VdW+yQ2U026FGXwrt0UmU0ePCAxEWpJQ1SvLa66q6trT
m+xOTSqN2SSp/regnCU1XFsz750N6X6bGe7obp/fDBnE2G3A1a8qRUbEbN11uu9gPzOYEqpuY/VN
P766fkNVqj8tHcU48yQLvfL03NIczNa5a8cpV19kWo58BX3ZdGU2JMrWbSFBboM45UVb5OJ0CKme
dElt4fX/vKGwKUqKcoqQLpo42J2sUx5c94VlIikJM0UxeZ3bvip1/0wsAjz80DFEUELNva3x2n9Y
Hwyb6UFHpyXyMz0jEkW5qTzyjcMaaUKyZtmLrjxsy19+O/IB86HwzIBMZjJ72A4WBa74ymoe4kCT
tWZ4SoEOBCBYef+iaCwobcoMx8gnLxLmBTCklBEyuw/0tu3sIIJZ19Vd+TerRFxIWzHDqwTI7SPM
DqsbDRnzm96cPqTcigJXieRU9LhTHbK9FW71nb9IIFPji1l3HYAZ3B6SVUvKF753LtFEhRinG4G1
A1l/dfWcd9Sn+m3gI+BfrxbVX6HuEhjQZEf+cEI5KlwRWv+FpSIpiMBrI86BD31dnz5bl5e6oNFs
EpFP1Ak899uDr3U87avFc53/rEFHN0FFvuLPLKtxgznDYt6mJrPAVJImIsQ0Br3ZFClnACIzbKz8
5ELyZDcnLsQRBcgusRWREeCDLcPHH2sBgHm3N0YrwiKltG2DnBwt+jHHvE613Kp5HD0gyqW9P92f
M5wla439BoFGPonHJ14RvrBFtDJYtbocAODcWWLvyKZIghkSwdw7msrmFNtJQZgpW/Ups6yfQghZ
uUGSYITNY39qsQftQJE156ZGEZdn1G8+T9cHRmC2djI3DemVY+GoeSXK14i+r0p+DYHZkZmYLZ33
dym7qoYxVr2qYjKOmaYmXYhAkoywMffWBifuMJY3jDlXeSJ4s0W1deMWj3enDj98DADqrqyK1oSc
tBhR2up1debcBzxEAvPITV7LL7mxD/pzC+5+RMgWaI3wJXAMTZUTRmOmWh58rhpIiFaFJyliYVMi
vgBQvqS0sC4q0tI3FsFrQPGrQXIBIURAUFKZEevon1oSp5OAOPvt9VJIDzeikUTOkTa/JsgwjSJQ
BCqj/TUdR0sj+qrvcCSZEXPn4nkt5Oh3tnUchf6KeQBQSrEgmziBcZpg0VpCff0dmGkuolHmAmay
64QIhNzkw13xA786CAhFjdHYkhKRkMCAfHOAswEz+vwZ0p6tm5BHRJ2/y07NcrtCIeMFEykvdqVs
3wNSnmJ2P4hyVXe2HAC9ss+crElGYRMpBUra6pwo5GlBYP3sqb60VETkjoklcrdrUC6gD6T0ZBOl
JEgWrY6+8k8vJTqTAFBzWQWPMKdfItdMoBkl438jAiJjiCYygzHupg6VJGlLkZIiJWRaKkE66QRA
yBk3EE3GA9wIcG4ybjJk2gXUokZKKZJEEkhm+sszoZHr4pEvCamHklMW4NRm2jPJBEASmMk7Xuly
9ak8x6c9JQQmAOh6rS/V4UQqA+m4jRzR15vt2ilF4A1uR+SWYUTNNx46duCXh7nBpFAl80q4YQiU
zGDIkHFETUVAICRFyiGZlna/k+xJDnXEh1vj8Y7EcGci1Zuyhxwn4ci0JJmtgUeGzGDMRCNoGCHD
ChlmxLQKrECRGSwKWMWBYGkgWBIIFJpWoRWIWkbQYIZeAqF0OkFJRYqUBFAEvnRYpr/JL99ICpRS
VpE1cDJx9JETgDAZS/CmJpukg+CGq+uu+uZaACUSIlvxgsAYQ4bAgXHGTA6MlE3D7an9vzn0xq8O
Ibm5xcbr6tf+xTIpBBITCZkcSCd6ksnTyVRPOtVvJ/tS6f60PWinBmwn7pxbL9GKGlaBFSwKBMsC
obJgqDQQLg9FKkLB0mCg2LKiphHk3OJ6U4dSJB2lpAKlSxQYICADZnI0WN+RwWe+8kLfof5JyixN
WbpQP0/58rIVH19SNr/YCCJjht50pGwSScdJOMkBO96ZGDgx1LW3p2tPj0gJzHW1rYhphLh0SCSd
M5sxZOgDkXJxLco/DIS8FILf3zrTNlHkaEXNQKEVKguGy8Ph8mC0KhKuCAVLA4FCywhwXYBHkuyE
GDw13Pp8+7E/tTgJZ2Sx/ps6MdOdwODLhhY1Fkarwszi5CgnIVL96fSgLZJCpMSo78/4Y7mHgrlV
69nxSHQOww/fnOEROyPoTKtkjbBhWBw5AyAlyEkKZUs/Jj9JzuwUJ/zftLDBxafcPqQz8uc0mV3k
0d7HaqPqXv1oGRdvEmOWc8HREyZz3r1ym0Wm+zWuQ8DRYNCZa6oIdgF9PZzCU8AZZpmRgBlumFFx
U8kBeNEd48Vvw2auS9d5wZlDm7kuVHrjxcuYOMMHM0c5wQeZUaQzR3NRHwvOHNAMb16yRMLpz9w4
c+gzPPhW7zYN3Ry8cDgGZ9hu5pq5Lobr/wMnvez52aXLvwAAAABJRU5ErkJggg==
EOF_B64
base64 -d > ui/web/assets/dp-bot.png << 'EOF_B64'
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
EOF_B64
base64 -d > ui/web/assets/dp-user.png << 'EOF_B64'
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
LBazJacksrW1Te/g4elOklIIIaVcsWLFY4899u1/+LZ9/tatW7/3ve/deeed999//4G33lJUxV73
WbNmVZRXICKyIX5fScQYe+21121lUVtTCwR6xnS7viAiGv3GiahIZyexhyuQcPaYJQCsXrVq//79
nom7e/fue+65d/Xq1el0+qRXmDKl+YYbbnj88cct07K//sILL1x55RU8QL9kMnnvvfcSkV7Uieiu
u+4CAE3Thtq/VxDxM5/5jK2nf/zgjwBgyacXXP/Hqzc8sX7DE5dft+WjMzdMA7tn/OmsMJ5Cm8Kx
hUUPBLaqq6u+9rX/festtwaD/G1tbbt37961a9e+ffva2tp6e3t1XeecV1ZWNtQ3zD5v9oWLLpx3
/ryKigr7/B07d/7jP333oR89SADIkTMuTVm9MN25vVtK+uLffPHv7/t70zCFFOvWrdu8ebOqqpZl
DUnkjiHaluPmzZtXrVoFALfcesuDP35w1deWT13bUOzTJVCkIvr83X9qferIaAUbRm0veH7wihXL
f/GLX2T6M3Q6R6FQfHbTszfcdEM0EgUArUyd++ezozURAGh6b/11W66+9O9XaHENAH74wx8SkWmY
3d3da9eutV/E5ryzMd29K3zpb75ks29HZ0d1dbUS4Vf9cs21m668ZuMHN2y8/NoXrq5ZXH0GbtI7
sy+ODHXPttszokfmmTNnfuELX3j66aePtbWFKoADh14sthxq+c1vf/P5/3nH/PnznY0S57M3zFj/
qzUz/mw6AKSaExueWP/xJy+/9qUrVn59Medc1dRHH32UiEzTtCzrK1/9SiQSOV2H2PGGw65wLBa7
5557bOebiG648S8AYNrayZ944YprnvjgNf/9oWuevOKaZ64sm142in4wnvF3zqAi9EQgFwIIN6xU
VVU1c+bMqdOm1tXVVVSUc84LxWJvb+/brUf273urtaU1W8w6Z56fnvy+SZNW1tUtrtv28O5X73ud
qXjZdy6uW1hpZC0i0Cq1tx499MdvbdU09cGHHrr+uuvtL27btu073/3OI//5SC6XgwCUcSJKE1FJ
0md9ff1HPvKR22+77Xx3n91515333XtfJKWteeDi1KSE0CUwhgrXc8bGG3+vdxaHv4x0LOngQcls
53u8cwAxWq/VzK2qW1RTvaCybHqSKygEZQ4VnrnteSNjLLh5zqJb5uS7dMaRAKSgRHVs928ObPnW
VgD44pe++JW/vTuZStmX2rt376OPPvr444+/8cYbXV1dJ33Cmpqa6dOnL1u27LLLLrt45cV19XXe
dvnSnV/a+MRGrvH3fG3J5NX1xT6TKYwkYIRn2nL/fcMmqcvRqvcacwAp4wwI5lw/o/b86tzxIkni
GlPjSrRSi9fG1TKuJjjjTJpk6RaZoJZHfv+ll4+92F69IL3m+5fIoj0Ax6kKlZbgZfzYnzpf+/ab
mbdzc+fM+eY937zqyg8HkfCjR9v279vXcqiltbW1s7Mzk8lYlsUYi8Vi5eXltTU1zc3Nk5ubm5ub
6+rqgo/6xrY3fvLwT370wA8LxWKiLr78roWNF9UV+3S7kxdJ4nG1Y0fPM5/ZDKNXz4dDImZP8U6n
VPfNkCStuu+imR9qLvbqyJAxDpKEJUgQWSQsSWQPUSAtpR19uWPzF1/mUf7+719SOTtl5CxHRxIg
A0lEloxXxHI9xa3/vOvAY4cAYPHixddd/8n1H/zgzOkzTxcrzWYyu/fseenllx577NFnn3xWgGQc
Zlw1ff6nZiRr48V+w6EuAEmKlEUPPHP45bv/NIQFhqdLI2Wg3z1MZKbT2XKFzmKxXzf7TWQcwPLa
xSIgcs9cQSHkzn/fQ0TNH5hUsyBd7NSRoRCSq4oS5UK3mCRQebHfUKN8xV0LZ14xdefP9772/Guv
vfbaFz//N/POnzfn/Llz585pntJcX1efTqfj8biNV9vgc7FQ6M9kenp7jr599NChlt279uzetftg
y0H79pG0Nu09k2ZePbV2fqXIm0bGZAoLNmoCRrlj+bMXlAgD+kWcGYFHALA86Wa0P7TygnNuMcE4
A6JgoS66CYs8yjKt+e4dvUqET/9ws57RAYFHVS3K813FzKFc2eQ4cERC4CQFUb+omVex6ltL+1ry
x14+fuiFI9t3bN++Y/vpPnbZtGTN+dX1y2oaltTGqjRhWMWeIjIGdotUdwntLkvZo7mR441TJPCo
COqSz628cDoUSgrVV2OgGQPDQpcudJGojlU2VmpJXugzMq35Iy+0HXqyte9g/4Kb5yy6ZW6hu2hP
KgQORs5EhIopyapZ5bM/PrXYpeePFXNtxdzxfK49r/cXjZwpDAl2nb7KuMK0hKaltGg6Eq2KxOti
yYZ4tEqLpDQEsArS6DMBwRYq5Ow971lJCsq9XRhRphmwmCPNwXSyn+0fCj1F8haM/EFl3kmIQBYk
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
EOF_B64
base64 -d > ui/web/assets/help.png << 'EOF_B64'
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
EOF_B64
base64 -d > ui/web/assets/home.png << 'EOF_B64'
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
EOF_B64
base64 -d > ui/web/assets/mic.png << 'EOF_B64'
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
EOF_B64
base64 -d > ui/web/fonts/LICENSE-LINE-Seed-OFL.txt << 'EOF_B64'
R29vZ2xlIEluYy4KClRoaXMgRm9udCBTb2Z0d2FyZSBpcyBsaWNlbnNlZCB1bmRlciB0aGUgU0lM
IE9wZW4gRm9udCBMaWNlbnNlLCBWZXJzaW9uIDEuMS4KVGhpcyBsaWNlbnNlIGlzIGNvcGllZCBi
ZWxvdywgYW5kIGlzIGFsc28gYXZhaWxhYmxlIHdpdGggYSBGQVEgYXQ6Cmh0dHA6Ly9zY3JpcHRz
LnNpbC5vcmcvT0ZMCgoKLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t
LS0tLS0tLS0tLS0tLS0tLS0KU0lMIE9QRU4gRk9OVCBMSUNFTlNFIFZlcnNpb24gMS4xIC0gMjYg
RmVicnVhcnkgMjAwNwotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t
LS0tLS0tLS0tLS0tLS0tLQoKUFJFQU1CTEUKVGhlIGdvYWxzIG9mIHRoZSBPcGVuIEZvbnQgTGlj
ZW5zZSAoT0ZMKSBhcmUgdG8gc3RpbXVsYXRlIHdvcmxkd2lkZQpkZXZlbG9wbWVudCBvZiBjb2xs
YWJvcmF0aXZlIGZvbnQgcHJvamVjdHMsIHRvIHN1cHBvcnQgdGhlIGZvbnQgY3JlYXRpb24KZWZm
b3J0cyBvZiBhY2FkZW1pYyBhbmQgbGluZ3Vpc3RpYyBjb21tdW5pdGllcywgYW5kIHRvIHByb3Zp
ZGUgYSBmcmVlIGFuZApvcGVuIGZyYW1ld29yayBpbiB3aGljaCBmb250cyBtYXkgYmUgc2hhcmVk
IGFuZCBpbXByb3ZlZCBpbiBwYXJ0bmVyc2hpcAp3aXRoIG90aGVycy4KClRoZSBPRkwgYWxsb3dz
IHRoZSBsaWNlbnNlZCBmb250cyB0byBiZSB1c2VkLCBzdHVkaWVkLCBtb2RpZmllZCBhbmQKcmVk
aXN0cmlidXRlZCBmcmVlbHkgYXMgbG9uZyBhcyB0aGV5IGFyZSBub3Qgc29sZCBieSB0aGVtc2Vs
dmVzLiBUaGUKZm9udHMsIGluY2x1ZGluZyBhbnkgZGVyaXZhdGl2ZSB3b3JrcywgY2FuIGJlIGJ1
bmRsZWQsIGVtYmVkZGVkLApyZWRpc3RyaWJ1dGVkIGFuZC9vciBzb2xkIHdpdGggYW55IHNvZnR3
YXJlIHByb3ZpZGVkIHRoYXQgYW55IHJlc2VydmVkCm5hbWVzIGFyZSBub3QgdXNlZCBieSBkZXJp
dmF0aXZlIHdvcmtzLiBUaGUgZm9udHMgYW5kIGRlcml2YXRpdmVzLApob3dldmVyLCBjYW5ub3Qg
YmUgcmVsZWFzZWQgdW5kZXIgYW55IG90aGVyIHR5cGUgb2YgbGljZW5zZS4gVGhlCnJlcXVpcmVt
ZW50IGZvciBmb250cyB0byByZW1haW4gdW5kZXIgdGhpcyBsaWNlbnNlIGRvZXMgbm90IGFwcGx5
CnRvIGFueSBkb2N1bWVudCBjcmVhdGVkIHVzaW5nIHRoZSBmb250cyBvciB0aGVpciBkZXJpdmF0
aXZlcy4KCkRFRklOSVRJT05TCiJGb250IFNvZnR3YXJlIiByZWZlcnMgdG8gdGhlIHNldCBvZiBm
aWxlcyByZWxlYXNlZCBieSB0aGUgQ29weXJpZ2h0CkhvbGRlcihzKSB1bmRlciB0aGlzIGxpY2Vu
c2UgYW5kIGNsZWFybHkgbWFya2VkIGFzIHN1Y2guIFRoaXMgbWF5CmluY2x1ZGUgc291cmNlIGZp
bGVzLCBidWlsZCBzY3JpcHRzIGFuZCBkb2N1bWVudGF0aW9uLgoKIlJlc2VydmVkIEZvbnQgTmFt
ZSIgcmVmZXJzIHRvIGFueSBuYW1lcyBzcGVjaWZpZWQgYXMgc3VjaCBhZnRlciB0aGUKY29weXJp
Z2h0IHN0YXRlbWVudChzKS4KCiJPcmlnaW5hbCBWZXJzaW9uIiByZWZlcnMgdG8gdGhlIGNvbGxl
Y3Rpb24gb2YgRm9udCBTb2Z0d2FyZSBjb21wb25lbnRzIGFzCmRpc3RyaWJ1dGVkIGJ5IHRoZSBD
b3B5cmlnaHQgSG9sZGVyKHMpLgoKIk1vZGlmaWVkIFZlcnNpb24iIHJlZmVycyB0byBhbnkgZGVy
aXZhdGl2ZSBtYWRlIGJ5IGFkZGluZyB0bywgZGVsZXRpbmcsCm9yIHN1YnN0aXR1dGluZyAtLSBp
biBwYXJ0IG9yIGluIHdob2xlIC0tIGFueSBvZiB0aGUgY29tcG9uZW50cyBvZiB0aGUKT3JpZ2lu
YWwgVmVyc2lvbiwgYnkgY2hhbmdpbmcgZm9ybWF0cyBvciBieSBwb3J0aW5nIHRoZSBGb250IFNv
ZnR3YXJlIHRvIGEKbmV3IGVudmlyb25tZW50LgoKIkF1dGhvciIgcmVmZXJzIHRvIGFueSBkZXNp
Z25lciwgZW5naW5lZXIsIHByb2dyYW1tZXIsIHRlY2huaWNhbAp3cml0ZXIgb3Igb3RoZXIgcGVy
c29uIHdobyBjb250cmlidXRlZCB0byB0aGUgRm9udCBTb2Z0d2FyZS4KClBFUk1JU1NJT04gJiBD
T05ESVRJT05TClBlcm1pc3Npb24gaXMgaGVyZWJ5IGdyYW50ZWQsIGZyZWUgb2YgY2hhcmdlLCB0
byBhbnkgcGVyc29uIG9idGFpbmluZwphIGNvcHkgb2YgdGhlIEZvbnQgU29mdHdhcmUsIHRvIHVz
ZSwgc3R1ZHksIGNvcHksIG1lcmdlLCBlbWJlZCwgbW9kaWZ5LApyZWRpc3RyaWJ1dGUsIGFuZCBz
ZWxsIG1vZGlmaWVkIGFuZCB1bm1vZGlmaWVkIGNvcGllcyBvZiB0aGUgRm9udApTb2Z0d2FyZSwg
c3ViamVjdCB0byB0aGUgZm9sbG93aW5nIGNvbmRpdGlvbnM6CgoxKSBOZWl0aGVyIHRoZSBGb250
IFNvZnR3YXJlIG5vciBhbnkgb2YgaXRzIGluZGl2aWR1YWwgY29tcG9uZW50cywKaW4gT3JpZ2lu
YWwgb3IgTW9kaWZpZWQgVmVyc2lvbnMsIG1heSBiZSBzb2xkIGJ5IGl0c2VsZi4KCjIpIE9yaWdp
bmFsIG9yIE1vZGlmaWVkIFZlcnNpb25zIG9mIHRoZSBGb250IFNvZnR3YXJlIG1heSBiZSBidW5k
bGVkLApyZWRpc3RyaWJ1dGVkIGFuZC9vciBzb2xkIHdpdGggYW55IHNvZnR3YXJlLCBwcm92aWRl
ZCB0aGF0IGVhY2ggY29weQpjb250YWlucyB0aGUgYWJvdmUgY29weXJpZ2h0IG5vdGljZSBhbmQg
dGhpcyBsaWNlbnNlLiBUaGVzZSBjYW4gYmUKaW5jbHVkZWQgZWl0aGVyIGFzIHN0YW5kLWFsb25l
IHRleHQgZmlsZXMsIGh1bWFuLXJlYWRhYmxlIGhlYWRlcnMgb3IKaW4gdGhlIGFwcHJvcHJpYXRl
IG1hY2hpbmUtcmVhZGFibGUgbWV0YWRhdGEgZmllbGRzIHdpdGhpbiB0ZXh0IG9yCmJpbmFyeSBm
aWxlcyBhcyBsb25nIGFzIHRob3NlIGZpZWxkcyBjYW4gYmUgZWFzaWx5IHZpZXdlZCBieSB0aGUg
dXNlci4KCjMpIE5vIE1vZGlmaWVkIFZlcnNpb24gb2YgdGhlIEZvbnQgU29mdHdhcmUgbWF5IHVz
ZSB0aGUgUmVzZXJ2ZWQgRm9udApOYW1lKHMpIHVubGVzcyBleHBsaWNpdCB3cml0dGVuIHBlcm1p
c3Npb24gaXMgZ3JhbnRlZCBieSB0aGUgY29ycmVzcG9uZGluZwpDb3B5cmlnaHQgSG9sZGVyLiBU
aGlzIHJlc3RyaWN0aW9uIG9ubHkgYXBwbGllcyB0byB0aGUgcHJpbWFyeSBmb250IG5hbWUgYXMK
cHJlc2VudGVkIHRvIHRoZSB1c2Vycy4KCjQpIFRoZSBuYW1lKHMpIG9mIHRoZSBDb3B5cmlnaHQg
SG9sZGVyKHMpIG9yIHRoZSBBdXRob3Iocykgb2YgdGhlIEZvbnQKU29mdHdhcmUgc2hhbGwgbm90
IGJlIHVzZWQgdG8gcHJvbW90ZSwgZW5kb3JzZSBvciBhZHZlcnRpc2UgYW55Ck1vZGlmaWVkIFZl
cnNpb24sIGV4Y2VwdCB0byBhY2tub3dsZWRnZSB0aGUgY29udHJpYnV0aW9uKHMpIG9mIHRoZQpD
b3B5cmlnaHQgSG9sZGVyKHMpIGFuZCB0aGUgQXV0aG9yKHMpIG9yIHdpdGggdGhlaXIgZXhwbGlj
aXQgd3JpdHRlbgpwZXJtaXNzaW9uLgoKNSkgVGhlIEZvbnQgU29mdHdhcmUsIG1vZGlmaWVkIG9y
IHVubW9kaWZpZWQsIGluIHBhcnQgb3IgaW4gd2hvbGUsCm11c3QgYmUgZGlzdHJpYnV0ZWQgZW50
aXJlbHkgdW5kZXIgdGhpcyBsaWNlbnNlLCBhbmQgbXVzdCBub3QgYmUKZGlzdHJpYnV0ZWQgdW5k
ZXIgYW55IG90aGVyIGxpY2Vuc2UuIFRoZSByZXF1aXJlbWVudCBmb3IgZm9udHMgdG8KcmVtYWlu
IHVuZGVyIHRoaXMgbGljZW5zZSBkb2VzIG5vdCBhcHBseSB0byBhbnkgZG9jdW1lbnQgY3JlYXRl
ZAp1c2luZyB0aGUgRm9udCBTb2Z0d2FyZS4KClRFUk1JTkFUSU9OClRoaXMgbGljZW5zZSBiZWNv
bWVzIG51bGwgYW5kIHZvaWQgaWYgYW55IG9mIHRoZSBhYm92ZSBjb25kaXRpb25zIGFyZQpub3Qg
bWV0LgoKRElTQ0xBSU1FUgpUSEUgRk9OVCBTT0ZUV0FSRSBJUyBQUk9WSURFRCAiQVMgSVMiLCBX
SVRIT1VUIFdBUlJBTlRZIE9GIEFOWSBLSU5ELApFWFBSRVNTIE9SIElNUExJRUQsIElOQ0xVRElO
RyBCVVQgTk9UIExJTUlURUQgVE8gQU5ZIFdBUlJBTlRJRVMgT0YKTUVSQ0hBTlRBQklMSVRZLCBG
SVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBTkQgTk9OSU5GUklOR0VNRU5UCk9GIENP
UFlSSUdIVCwgUEFURU5ULCBUUkFERU1BUkssIE9SIE9USEVSIFJJR0hULiBJTiBOTyBFVkVOVCBT
SEFMTCBUSEUKQ09QWVJJR0hUIEhPTERFUiBCRSBMSUFCTEUgRk9SIEFOWSBDTEFJTSwgREFNQUdF
UyBPUiBPVEhFUiBMSUFCSUxJVFksCklOQ0xVRElORyBBTlkgR0VORVJBTCwgU1BFQ0lBTCwgSU5E
SVJFQ1QsIElOQ0lERU5UQUwsIE9SIENPTlNFUVVFTlRJQUwKREFNQUdFUywgV0hFVEhFUiBJTiBB
TiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HCkZST00sIE9V
VCBPRiBUSEUgVVNFIE9SIElOQUJJTElUWSBUTyBVU0UgVEhFIEZPTlQgU09GVFdBUkUgT1IgRlJP
TQpPVEhFUiBERUFMSU5HUyBJTiBUSEUgRk9OVCBTT0ZUV0FSRS4K
EOF_B64
base64 -d > ui/web/fonts/LINESeedJP-400.woff2 << 'EOF_B64'
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
4vlZAgAAAA==
EOF_B64
base64 -d > ui/web/fonts/LINESeedJP-700.woff2 << 'EOF_B64'
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
EOF_B64
base64 -d > ui/web/fonts/LINESeedJP-800.woff2 << 'EOF_B64'
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
yS7xS5EIMkjCJUXfd3lvB62ahOo7OwE=
EOF_B64
chmod +x colab/serve.sh
cp "$0" colab/sessions/session_32_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest,ui]"
if [ "$HAS_GPU" = 1 ]; then python -c "import fla" 2>/dev/null || pipi flash-linear-attention >/tmp/fla.log 2>&1 || true; pipi rfdetr==1.7.0; python -c "import transformers" 2>/dev/null || pipi transformers; fi
make check

step "4. stop whatever the previous run left (ingest, api, tunnel) and reset the database"
bash colab/serve.sh stop 2>/dev/null || true
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed"; fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi WITH (FORCE);" 2>/dev/null || sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" 2>/dev/null || warn "could not drop the old database; reusing it"
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"; DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"; echo "export DB_URL=\"$DB_URL\"" > /content/vi_env.sh 2>/dev/null || true

step "5. commit + push (before the long-running parts)"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: sessions 27-30: live ingest, guards, backend+frontend, grid auto-detect, cross-camera fusion"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed"; fi

step "6. backend up (API + page), public URL"
BACKEND=fake; [ "$HAS_GPU" = 1 ] && BACKEND=transformers
rm -rf data/episodes data/keyframes data/live; mkdir -p data/live
VI_DB="$DB_URL" VI_BACKEND=$BACKEND VI_MODEL="$AGENT_MODEL" VI_TZ="$TZ_NAME" bash colab/serve.sh start
curl -s http://127.0.0.1:$PORT/health | python -c "import sys,json; h=json.load(sys.stdin); print('  health:', {k: h[k] for k in ('ok','backend','model','tz','episodes')})"

step "7. start $SOURCE as ${GRID:+a $GRID grid of }live cameras (paced to its own clock) through the API"
if [ -f "$SOURCE" ]; then
  MODEL_ARG=$DET_MODEL; REID_ARG=siglip; [ "$HAS_GPU" = 1 ] || { MODEL_ARG=fake; REID_ARG=hist; }
  # build the request in Python: no shell quoting of JSON (a bash expansion bug here cost session 29's first run)
  SOURCE="$SOURCE" GRID="$GRID" START_TIME="$START_TIME" PROFILE="$PROFILE" WRITER="$WRITER" REID_ARG="$REID_ARG" MODEL_ARG="$MODEL_ARG" PORT="$PORT" python - << 'EOF_PY'
import json, os, urllib.request
body = {"source": os.environ["SOURCE"], "grid": os.environ.get("GRID") or None, "start_time": os.environ.get("START_TIME") or None,
        "realtime": True, "profile": os.environ["PROFILE"], "writer": os.environ["WRITER"], "reid": os.environ["REID_ARG"], "model": os.environ["MODEL_ARG"]}
req = urllib.request.Request(f"http://127.0.0.1:{os.environ['PORT']}/ingest/start", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=30))
print("  ingest:", "started pid", r.get("pid")) if r.get("started") else print("  ingest:", r)
EOF_PY
  sleep 40
  curl -s http://127.0.0.1:$PORT/ingest/status | python -c "import sys,json; s=json.load(sys.stdin); print('  status:', {k: s.get(k) for k in ('running','cameras','frames','footage_s','wall_s','live_tubes','detect_ms_p50','episodes')})"
  echo "  keep-up: footage/wall must stay >= 1.0 (see data/live/ingest.log: 'realtime x…')"
  curl -s -o /dev/null -w "  live frame: HTTP %{http_code}\n" http://127.0.0.1:$PORT/live/latest.jpg
else
  warn "no clip: start the stream later with POST /ingest/start (see colab/README.md)"
fi

step "8. tiles: learn which cameras see the same area (needs a few minutes of footage; rerun later via POST /tiles/recompute)"
curl -s -X POST http://127.0.0.1:$PORT/tiles/recompute | python -c "import sys,json; r=json.load(sys.stdin); print('  tiles:', r.get('tiles'), '| affinity:', r.get('affinity'), '| entities used:', r.get('entities_used'))"
echo "  (if cameras were grouped, restart the ingest to get one identity per area: POST /ingest/stop, then /ingest/start again)"

step "9. first question through the API"
curl -s -X POST http://127.0.0.1:$PORT/ask -H 'Content-Type: application/json' -d '{"question":"How many people are there right now, across all cameras?"}' \
  | python -c "import sys,json; r=json.load(sys.stdin); print('  ', r.get('grounding'), r.get('window'), f\"{(r.get('latency_ms') or 0)/1000:.1f}s\"); print('  A:', r['text'][:300]); print('  evidence:', len(r.get('evidence', [])))"

step "10. where to go"
echo "PUBLIC URL: $(bash colab/serve.sh url)"
echo "Open it on your laptop. 'Cameras' shows the live frame; ask in the box. Logs: /tmp/vi_api.log, data/live/ingest.log"
echo "Stop: bash colab/serve.sh stop ; curl -X POST http://127.0.0.1:$PORT/ingest/stop"
