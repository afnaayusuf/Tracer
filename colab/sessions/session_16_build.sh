#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 16 build: the fast query path + honest counting.
#   * scene script inlined into the first prompt (one round trip saved); vLLM prefix caching on;
#     the server is reused if already up and is NOT stopped at the end
#   * event vocabulary in the prompt, tool args validated (pick_up -> "valid: ... pickup ...")
#   * tube quality flag: brief / tiny / border tubes are "brief sightings", not people; the script
#     counts confirmed people separately
#   * latency measured per question (total, model, tools, turns, prompt size)
#   * tests/scenarios/warehouse.yaml + bench/scenario_eval.py: pass/fail per question against
#     ground truth (fill in people_total etc.), with a 10 s budget
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_16.sh
#  Env: GH_REPO GH_TOKEN REPO_DIR SOURCE DB_URL AGENT_MODEL (Qwen/Qwen3.5-4B) SKIP_VLLM=1 BACKEND NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 16"
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
[ -n "$(git log --grep='^session 15' --format=%h)" ] || die "session 15 commit not found; run build_session_15.sh first"
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
bench/scenario_eval.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/preflight.sh
colab/vllm_venv.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/scenarios/warehouse.yaml
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
vi/tubes/quality.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p tests/scenarios colab/sessions data/bench
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
cat > vi/tubes/quality.py << 'EOF_VI'
from __future__ import annotations

from vi.schemas import Tube

MIN_LIFE_MS = 1500
MIN_HEIGHT_PX = 48
EDGE_PX = 8


def grade_tube(tube: Tube, frame_w: int, frame_h: int) -> Tube:
    """E-DET-01 / resolution ceiling: a tube that lived under 1.5 s, is under 48 px tall, or was
    born hugging the frame border is real evidence of *something*, not a confirmed person. It
    stays in the store, flagged, so the script can count it apart."""
    life = tube.last_seen.corrected_ms() - tube.born.corrected_ms()
    b = tube.box
    reasons = []
    if life < MIN_LIFE_MS:
        reasons.append(f"life {life / 1000:.1f}s")
    if b.height < MIN_HEIGHT_PX:
        reasons.append(f"height {int(b.height)}px")
    if b.x1 <= EDGE_PX or b.y1 <= EDGE_PX or b.x2 >= frame_w - EDGE_PX or b.y2 >= frame_h - EDGE_PX:
        reasons.append("at frame border")
    tube.quality = "low" if reasons else "ok"
    tube.quality_reason = ", ".join(reasons) or None
    return tube
EOF_VI
cat > vi/tubes/__init__.py << 'EOF_VI'
from .base import Tracker
from .bytetrack import ByteTracker
from .kalman import KalmanBoxFilter
from .linker import TubeLinker
from .quality import grade_tube
from .simple_iou import SimpleIoUTracker

TRACKERS = {"simple": SimpleIoUTracker, "byte": ByteTracker}
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
    Column("quality", String), Column("quality_reason", String),
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
                                  merge_candidates=t.merge_candidates, quality=t.quality, quality_reason=t.quality_reason))
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
        counts["entities"] += insert_ignore(conn, entities, list(ent_rows.values()))
        counts["custody"] += insert_ignore(conn, custody, custody_rows)
    counts["episode_id"] = header.episode_id  # type: ignore[assignment]
    return dict(counts)
EOF_VI
cat > vi/agent/tools.py << 'EOF_VI'
"""Deterministic Block 2 tools (R26): the reasoning model calls these; it never sees video.
search_* are SQL filters; get_script renders the scene script the model reads; clip returns
evidence references. Every result carries the ids the model must cite (R27)."""
from __future__ import annotations

import time

from sqlalchemy import and_, or_, select
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
                f"seen {_ts(first, t0)}–{_ts(last, t0)}  states {','.join(sorted({r['state'] for r in rows}))}"
                + (f"  keyframe {kf}" if kf else "") + desc)

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
}


class ToolStep(BaseModel):
    action: Literal["tool"] = "tool"
    tool: Literal["search_events", "search_tubes", "search_entities", "get_script", "clip"]
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = Field("", max_length=200)


class AnswerStep(BaseModel):
    action: Literal["answer"] = "answer"
    text: str = Field(max_length=1500)
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
mm:ss.s in scripts and absolute milliseconds in tool args. Respond with exactly one JSON object per turn:
{"action":"tool","tool":...,"args":{...},"why":...} | {"action":"answer","text":...,"citations":[...],"confidence":...}
| {"action":"clarify","question":...}. Count people from CONFIRMED entities; BRIEF SIGHTINGS are not people.
EVENT_TYPES: """ + ", ".join(EVENT_TYPES) + "\nTools: " + json.dumps(TOOL_SPECS)

ID_RE = re.compile(r"\b(ev_[0-9a-f]{16}|[A-Za-z0-9_]+:E\d+|anon:[A-Za-z0-9_:]+|[A-Za-z0-9_]+:\d+:\d+)\b")


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
        base = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        try:
            out = self._post({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema}}})
        except Exception:
            out = self._post({**base, "guided_json": schema})
        return extract_json(out["choices"][0]["message"]["content"])


def extract_json(text: str) -> str:
    """Take the first balanced {...} object out of a model reply (code fences, prose, and
    trailing text are all tolerated). Returns the raw text if none is found, so validation
    fails loudly rather than silently."""
    t = text.strip()
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

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 600, device: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        last = None
        self.model = None
        for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            try:
                cls = getattr(__import__("transformers", fromlist=[loader]), loader)
                self.model = cls.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
                self.loader = loader
                break
            except Exception as e:  # pragma: no cover
                last = e
        if self.model is None:
            raise RuntimeError(f"could not load {model_id}: {last!r}")
        try:
            self.tok = AutoProcessor.from_pretrained(model_id)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_id)

    def complete(self, messages: list[dict], schema: dict) -> str:
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\nReply with the JSON object only, no prose."}
        inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                              return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tokenizer = getattr(self.tok, "tokenizer", self.tok)
        return extract_json(tokenizer.decode(gen, skip_special_tokens=True))


def run_tool(engine: Engine, step: ToolStep) -> Any:
    fn = {"search_events": T.search_events, "search_tubes": T.search_tubes, "search_entities": T.search_entities,
          "get_script": T.get_script, "clip": T.clip}[step.tool]
    args = {k: v for k, v in step.args.items() if v is not None}
    if step.tool == "search_events" and args.get("event_type"):
        types = args["event_type"] if isinstance(args["event_type"], list) else [args["event_type"]]
        bad = [t for t in types if t not in EVENT_TYPES]
        if bad:
            raise ValueError(f"unknown event_type {bad}; valid: {', '.join(EVENT_TYPES)}")
    if step.tool == "get_script":
        return fn(engine, args.get("episode_id", ""))
    return fn(engine, **args)


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
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"TOOL RESULT: invalid step ({e.errors()[0]['msg']}); reply with one valid JSON object."})
            trace.append({"step": i, "invalid": raw[:200]})
            continue
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            result: Any = None
            t0 = time.perf_counter()
            try:
                result = run_tool(engine, step)
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
        if invalid and not valid and not already_revised:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in any tool result. "
                                                        "Answer again citing only ids you were shown, or say nothing matched."})
            continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid,
                 "confidence": step.confidence, "cited": bool(valid)}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False,
                 "confidence": 0.0}
    total_ms = (time.perf_counter() - t_start) * 1000
    return {"question": question, "episode_id": episode_id, "backend": getattr(backend, "name", "?"),
            "steps": len(trace), "trace": trace, "final": final,
            "latency": {"total_ms": round(total_ms), "model_ms": round(model_ms), "tool_ms": round(tool_ms),
                        "turns": sum(1 for t in trace if "tool" in t or "revise" in t or "invalid" in t) + 1,
                        "first_prompt_chars": len(first)}}
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
from vi.reid import crop_for_embedding, make_embedder
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
    ap.add_argument("--reid-sim", type=float, default=0.75)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default = edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
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
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    linker = TubeLinker(a.camera, sim_thr=a.reid_sim) if embedder else None
    embed_ms: list[float] = []
    last_embed_tick: dict[str, int] = {}
    pending_link: dict[str, np.ndarray] = {}
    relink_events = 0
    debug_paths: list[str] = []
    person_dets: list[int] = []
    concurrent_persons: list[int] = []
    state_ticks: Counter = Counter()

    for fr in reader.frames():
        if frames >= a.max_frames:
            break
        h, w = fr.rgb.shape[:2]
        current["frame"] = fr.rgb
        if zones is None:   # first frame: zones need the native size
            zones = load_zones(a.zones, a.camera) if a.zones else default_zones(a.camera, w, h, tile_id=a.tile)
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
        dets = dedupe_detections([d for d in dets if is_tube_class(d.class_label)])   # E-DET-10
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
                embs = embedder.embed([crop_for_embedding(fr.rgb, t.box) for t in todo])
                embed_ms.append((time.perf_counter() - t0) * 1000)
                birth_ids = {x.tube_id for x in due_birth}
                for t, e in zip(todo, embs):
                    last_embed_tick[t.tube_id] = frames
                    if t.tube_id in birth_ids:
                        pending_link[t.tube_id] = e
                    else:
                        linker.on_refresh(t, e)
            live_ids = {t.tube_id for t in live}
            for t in live:
                if t.tube_id in pending_link and t.state.value == "active":
                    ev = linker.on_birth(t, pending_link.pop(t.tube_id), fr.pts_ms)
                    if ev is not None:
                        events.append(ev)
                        relink_events += 1
                        print(f"t={fr.pts_ms:7d}  relink                 {ev.subject_tube_ids[0]} -> {ev.subject_tube_ids[1]} sim={ev.payload['similarity']}")
            for tid in [k for k in pending_link if k not in live_ids]:
                pending_link.pop(tid)                       # deleted while unconfirmed: never linked
            for t in live:
                if t.class_label == "person":
                    t.entity_id = linker.entity_of(t.tube_id)
            for t in closed:
                linker.on_close(t, fr.pts_ms)
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
    cast = [CastMember(tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0])
            for t in tubes]
    for t in tubes:
        grade_tube(t, w, h)
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
        "entities": linker.entities if linker else None, "relinks": linker.relinks if linker else None,
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

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask, clip, get_script, search_entities, search_events, search_tubes
from vi.store import connect, load_episode_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--script-lines", type=int, default=40)
    ap.add_argument("--ask", action="append", default=[], help="question(s) to run through the agent loop")
    ap.add_argument("--backend", choices=["fake", "openai", "transformers"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--no-inline-script", action="store_true", help="A/B: fetch the script through a tool turn instead")
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
        if a.backend == "fake":
            backend = FakeBackend()
        elif a.backend == "openai":
            backend = OpenAIBackend(base_url=a.base_url, model=a.model)
        else:
            try:
                backend = TransformersBackend(model_id=a.model)
                print(f"[agent] transformers backend loaded {a.model} via {backend.loader}")
            except Exception as e:
                print(f"[agent] transformers backend failed ({type(e).__name__}: {str(e)[:120]}); using fake backend")
                backend = FakeBackend()
        for q in a.ask:
            res = ask(engine, q, ep, backend, inline_script=not a.no_inline_script)
            answers.append(res)
            calls = [f"{t['tool']}({', '.join(f'{k}={v}' for k, v in t['args'].items())}) -> "
                     + (f"ERROR {t['error'][:80]}" if t.get('error') else str(t['results'])) for t in res["trace"] if "tool" in t]
            lat = res["latency"]
            print(f"\nQ: {q}\n   latency: {lat['total_ms'] / 1000:.1f}s total (model {lat['model_ms'] / 1000:.1f}s, tools {lat['tool_ms']}ms, "
                  f"{lat['turns']} turn(s), first prompt {lat['first_prompt_chars']} chars)\n   tools: " + (" | ".join(calls) or "none"))
            f = res["final"]
            if f["action"] == "answer":
                print(f"   A ({f.get('confidence', 0):.2f}{'' if f['cited'] else ', UNCITED'}): {f['text']}\n   cites: {', '.join(f['citations']) or '—'}")
            else:
                print(f"   clarify: {f['question']}")
        row["answers"] = [{"q": r["question"], "steps": r["steps"], "cited": r["final"].get("cited"),
                           "action": r["final"]["action"], "backend": r["backend"], **r["latency"]} for r in answers]
        row["latency_p50_ms"] = int(sorted(r["latency"]["total_ms"] for r in answers)[len(answers) // 2])
        row["latency_max_ms"] = max(r["latency"]["total_ms"] for r in answers)
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "agent.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "answers"}, indent=2))


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
from vi.store import connect


def score(res: dict, spec: dict, budget_ms: int) -> dict:
    f = res["final"]
    text = (f.get("text") or f.get("question") or "").lower()
    checks = {
        "answered": f["action"] == "answer",
        "mentions": all(m.lower() in text for m in spec.get("must_mention", []) or []),
        "avoids": not any(m.lower() in text for m in spec.get("must_not_mention", []) or []),
        "cites": (f.get("cited", False) or spec.get("expect_uncited_ok", False))
                 and all(any(c.startswith(p) for c in f.get("citations", [])) for p in spec.get("must_cite_prefix", []) or []),
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
    results = [score(ask(engine, q["q"], ep, backend), q, budget) for q in spec["questions"]]
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
    assert [t["tool"] for t in out["trace"] if "tool" in t] == ["search_events"]     # script was inlined
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


def test_extract_json_tolerates_fences_prose_and_nesting():
    from vi.agent import extract_json
    assert extract_json('```json\n{"action":"answer","text":"hi {x}","citations":[]}\n```') == '{"action":"answer","text":"hi {x}","citations":[]}'
    assert extract_json('Sure! Here it is: {"action":"tool","tool":"clip","args":{"entity_id":"cam1:E1"}} thanks') \
        == '{"action":"tool","tool":"clip","args":{"entity_id":"cam1:E1"}}'
    assert extract_json('{"a": "brace in string }"}') == '{"a": "brace in string }"}'
    assert extract_json("no json here") == "no json here"


def test_transformers_backend_pipeline_with_a_stub_model():
    """The loader/generate/decode path, with a stub standing in for the 4B model."""
    import numpy as np
    from vi.agent.loop import TransformersBackend

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            assert msgs[0]["role"] == "system" and "JSON object only" in msgs[0]["content"]
            return {"input_ids": np.zeros((1, 3), dtype=int)}
        def decode(self, ids, skip_special_tokens=True):
            return 'ok: {"action":"clarify","question":"which person?"}'

    class Model:
        def generate(self, **kw):
            return np.zeros((1, 3 + 5), dtype=int)

    class T:  # minimal torch stand-in
        @staticmethod
        def no_grad():
            import contextlib; return contextlib.nullcontext()

    b = TransformersBackend.__new__(TransformersBackend)
    b.torch, b.model, b.tok, b.device, b.max_new_tokens = T(), Model(), Tok(), "cpu", 10
    out = b.complete([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}], {})
    assert out == '{"action":"clarify","question":"which person?"}'


def test_inline_script_saves_the_first_round_trip_and_reports_latency(episode_path):
    from vi.agent import FakeBackend, ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    fast = ask(engine, "Who took the bike keys?", ep, FakeBackend(), inline_script=True)
    slow = ask(engine, "Who took the bike keys?", ep, FakeBackend(), inline_script=False)
    assert [t["tool"] for t in fast["trace"] if "tool" in t] == ["search_events"]
    assert [t["tool"] for t in slow["trace"] if "tool" in t] == ["get_script", "search_events"]
    assert fast["latency"]["turns"] < slow["latency"]["turns"] and fast["latency"]["total_ms"] >= 0
    assert "SCENE SCRIPT" not in fast["final"]["text"] and fast["final"]["cited"]


@pytest.mark.edge("E-AGT-04")
def test_unknown_event_type_is_rejected_with_the_valid_list():
    from vi.agent.loop import ToolStep, run_tool
    engine = connect()
    with pytest.raises(ValueError, match="pickup"):
        run_tool(engine, ToolStep(tool="search_events", args={"event_type": "pick_up"}))


def test_script_separates_confirmed_people_from_brief_sightings(tmp_path, episode_path):
    from vi.tubes import grade_tube
    from vi.schemas import Box, CamTime, Tube
    t = Tube(tube_id="c1:0:9", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0),
             last_seen=CamTime(cam_utc_ms=300), box=Box(x1=1, y1=100, x2=30, y2=140))
    grade_tube(t, 1280, 720)
    assert t.quality == "low" and "life 0.3s" in t.quality_reason and "at frame border" in t.quality_reason
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    assert "confirmed entities" in get_script(engine, ep, cache=False)
EOF_VI
cat > tests/scenarios/warehouse.yaml << 'EOF_VI'
# Acceptance scenario for the warehouse clip (HI_DEF_VIDEO.mp4). Fill `expected` from ground truth:
# what a person watching the clip would answer. bench/scenario_eval.py scores the agent against it.
clip: HI_DEF_VIDEO.mp4
ground_truth:
  people_total: null          # e.g. 10
  people_at_conveyor: null    # e.g. 2
  notes: null
questions:
  - q: How many distinct people were in this episode, and which of them stayed the whole time?
    must_mention: []          # e.g. ["10"]
    must_not_mention: ["14"]  # counting brief sightings as people is wrong
    must_cite_prefix: ["cam1:E"]
  - q: Who was at the conveyor on the right, and when did they arrive?
    must_mention: []
    must_cite_prefix: ["cam1:E"]
  - q: Did anyone pick something up from a shelf?
    must_mention: ["no"]      # nothing was picked up in this clip
    must_cite_prefix: []
    expect_uncited_ok: true
latency_budget_ms: 10000
EOF_VI
cat > colab/vllm_venv.sh << 'EOF_VI'
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

install_vllm() {
  "$VENV/bin/python" -c "import vllm" 2>/dev/null && return 0
  echo "-- installing vllm into the venv (isolated torch; 3-6 min)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv)"; return 0; fi
  if "$VENV/bin/python" -m pip install -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (pip)"; return 0; fi
  echo "   failed: $(tail -2 /tmp/vllm_install.log | tr '\n' ' ')"; return 1
}

make_venv || { echo "could not create a virtualenv by any method"; exit 1; }
if [ "$cmd" = venv ]; then "$VENV/bin/python" -c "import sys; print('   venv python', sys.version.split()[0])"; exit 0; fi
install_vllm || exit 1
"$VENV/bin/python" -c "import vllm, torch; print('   vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" || exit 1
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
  --max-model-len 16384 --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.6}" --dtype bfloat16 --max-num-seqs 4 \
  --enable-prefix-caching > "$LOG" 2>&1 &
echo "-- waiting for http://127.0.0.1:$PORT (weights download on first run)"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "   vLLM up: $(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"; exit 0
  fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "   server exited; log tail:"; tail -12 "$LOG"; exit 1; fi
  sleep 5
done
echo "   server did not come up in 10 min; log tail:"; tail -12 "$LOG"; exit 1
EOF_VI
cat > edge_cases.yaml << 'EOF_VI'
# Edge-case registry. Source of truth for `make coverage`.
# status: implemented (code exists; MUST have a test marked @pytest.mark.edge("ID"))
#         planned     (in scope for month 1; coverage prints it as TODO)
#         deferred    (out of month-1 scope; listed so it is not forgotten)
# Adding a case: append here first, then code, then test. Never the other order.
version: 1
cases:
# ---------------- ingest / time ----------------
- id: E-ING-01
  ring: ingest
  title: Camera clocks disagree
  trigger: Cameras drift seconds apart; fusion hand-offs and the ±1 s fact-miner window fail.
  handling: Every timestamp is CamTime{cam_utc_ms, offset_ms, offset_confidence}; offsets estimated in the calibration walk and re-estimated from hand-offs; fusion uses corrected_ms() only.
  status: implemented
- id: E-ING-02
  ring: ingest
  title: Stream drops or reconnects
  trigger: RTSP disconnect mid-episode.
  handling: signal_lost/signal_restored events; open episode on that camera closes with status=truncated; tubes go to lost, not exited.
  status: implemented
- id: E-ING-03
  ring: ingest
  title: Variable or dropped frame rate
  trigger: Camera throttles under load; fps changes.
  handling: Ticks are wall-clock windows (t_start,t_end), never frame indices; tube age computed from corrected time.
  status: implemented
- id: E-ING-04
  ring: ingest
  title: Codec without motion vectors
  trigger: MJPEG or unknown codec stream.
  handling: Gate interface is codec-agnostic; FrameDiffGate fallback at 1 fps; camera profile records gate_mode.
  status: implemented
- id: E-ING-05
  ring: ingest
  title: Resolution or aspect change mid-stream
  trigger: Camera reconfigured; homography and zones now invalid.
  handling: Reader SizeGuard flags the change; EventCompiler.on_size_change emits camera_moved_suspect; fusion freezes that camera until recalibration (week 3).
  status: implemented
- id: E-ING-06
  ring: ingest
  title: Timezone and DST
  trigger: '"Yesterday afternoon" across a DST switch or a site in another timezone.'
  handling: All storage in UTC ms; site timezone stored in KB; agent resolves local windows explicitly and echoes them in the answer.
  status: implemented
- id: E-ING-07
  ring: ingest
  title: Duplicate or out-of-order frames
  trigger: RTSP jitter delivers repeated or reordered PTS.
  handling: Dedupe by PTS per camera; drop frames older than the last processed tick.
  status: implemented
# ---------------- ring 0 gate ----------------
- id: E-GATE-01
  ring: ring0
  title: Stationary object is invisible to the gate
  trigger: Person stops moving; bag placed and left; fallen person still.
  handling: HeartbeatScheduler runs a full-frame detection at episode open, every 1 s on active tiles and every 10 s on quiet ones (merged with ROI detections); measured on real footage as person_tubes and lost-vs-exited in bench/slice_gpu.py --detect hybrid.
  status: implemented
- id: E-GATE-02
  ring: ring0
  title: Lighting change looks like whole-frame motion
  trigger: Light switch, IR-cut toggle, cloud passes.
  handling: Global luminance step detected before differencing; emitted as illumination_change scene-state event; blobs suppressed for that update.
  status: implemented
- id: E-GATE-03
  ring: ring0
  title: Foliage, rain, insects, sensor noise
  trigger: Persistent low-level motion that is not an object.
  handling: Adaptive noise floor (median block energy over history) × k; MV-field coherence test once MVGate lands.
  status: implemented
- id: E-GATE-04
  ring: ring0
  title: Camera shake or PTZ move
  trigger: Most of the frame changes at once.
  handling: active_fraction > threshold => global_motion; blobs suppressed, background frozen; sustained => camera_moved_suspect.
  status: implemented
- id: E-GATE-05
  ring: ring0
  title: Very slow motion under threshold
  trigger: Someone creeping; object slid slowly.
  handling: Same HeartbeatScheduler full-frame pass catches motion below the gate threshold; gate FN rate stays a tracked metric.
  status: implemented
- id: E-GATE-06
  ring: ring0
  title: Motion in screens, mirrors, reflections
  trigger: TV playing; mirror shows a person in another zone; steel reflects motion.
  handling: Scene card media_zones and reflective_surfaces become gate masks; tubes born inside them are reflection_suspect.
  status: planned
- id: E-GATE-07
  ring: ring0
  title: Intra-only or all-I-frame streams
  trigger: Encoder configured with no P-frames; no motion vectors.
  handling: Same fallback as E-ING-04.
  status: planned
- id: E-GATE-08
  ring: ring0
  title: Night noise inflates motion energy
  trigger: High-gain low-light mode.
  handling: Per-modality threshold profile; coherence weighted higher at night.
  status: planned
# ---------------- ring 1 detect ----------------
- id: E-DET-01
  ring: ring1
  title: Tiny objects below detector floor
  trigger: Keys, phone, small tools at room-camera resolution.
  handling: Not detected as tubes; tracked via carried_item attribute and asset-home occupancy; inspect tool for on-demand native-res count/find; resolution ceiling stated in answers.
  status: planned
- id: E-DET-02
  ring: ring1
  title: Truncation at frame edge
  trigger: Half a person at the border.
  handling: remap_detections flags boxes within 2 px of the frame border as truncated; Ring 2 then uses fallback_bbox_bottom with wide uncertainty and keyframe scoring penalises them.
  status: implemented
- id: E-DET-03
  ring: ring1
  title: Class confusion
  trigger: Child vs small adult; dog vs bag; mannequin vs person.
  handling: Class is an attribute with confidence; enrichment can override; agent never asserts class below threshold.
  status: planned
- id: E-DET-04
  ring: ring1
  title: Ghost detections in reflections
  trigger: Mirror/glass shows a duplicate person.
  handling: Tubes inside reflective_surfaces are reflection_suspect; fusion never creates an entity from a suspect tube alone.
  status: planned
- id: E-DET-05
  ring: ring1
  title: People on screens, posters, photos
  trigger: TV shows a face; framed photo detected as person.
  handling: media_zones from scene card; detections fully inside a media zone with no tube motion are dropped.
  status: planned
- id: E-DET-06
  ring: ring1
  title: Dense crowds
  trigger: Heavy overlap; fragmentation.
  handling: crowd event on density; tube quality flag; enrichment skipped for low-quality tubes.
  status: deferred
- id: E-DET-07
  ring: ring1
  title: Fisheye or wide-lens distortion
  trigger: Homography from a distorted image.
  handling: fov_class from scene card; undistort before foot-point projection.
  status: deferred
- id: E-DET-08
  ring: ring1
  title: IR appearance shift lowers recall
  trigger: Night mode.
  handling: Measured on night eval set before any enhancer; enhancer only if gap is real and license clear.
  status: planned
- id: E-DET-09
  ring: ring1
  title: Static furniture and fixtures become tubes
  trigger: Detector emits dining table, tv, chair as objects; tracker births permanent tubes for them.
  handling: Only TUBE_CLASSES (people, animals, vehicles, carried bags) become tubes; furniture is a scene-card asset owned by zones and heartbeats.
  status: implemented
- id: E-DET-10
  ring: ring1
  title: One object, several detections
  trigger: A person straddles two adjacent ROI crops, or is seen by both a crop and the full-frame heartbeat with boxes of different extent; each becomes a tube.
  handling: Every tick's detections pass through dedupe_detections (class-wise, IoU>=0.5 or intersection-over-smaller>=0.6); complete boxes beat crop-truncated ones. Measured in session 06 as max_concurrent 13 on a 9-person frame.
  status: implemented
- id: E-DET-11
  ring: ring1
  title: Zoomed crops promote background specks and boxes into confident detections
  trigger: In an ROI crop a 20-px figure at the far dock reads as person 0.6 and cardboard as suitcase 0.5; each becomes a tube (13 suitcase tubes in session 05).
  handling: 'Birth gates in the tracker: minimum box height (32 px) and class-specific confidence (person 0.5, other classes 0.65); re-attachment thresholds unchanged.'
  status: implemented
# ---------------- ring 2 tubes / fusion ----------------
- id: E-TUBE-01
  ring: ring2
  title: ID switch when paths cross
  trigger: Two people cross; IoU ambiguous.
  handling: Keep best match; record other as merge_candidate; never silently merge; fusion may later resolve with ReID.
  status: implemented
- id: E-TUBE-02
  ring: ring2
  title: Occlusion
  trigger: Person behind furniture or another person.
  handling: state=occluded with occluded_since; after max_occluded_ms becomes lost (or exited if last box touched an exit zone).
  status: implemented
- id: E-TUBE-03
  ring: ring2
  title: Long stationary dwell
  trigger: Sleeping person; parked object.
  handling: Heartbeat detections with det_source=heartbeat keep the track active.
  status: implemented
- id: E-TUBE-04
  ring: ring2
  title: Re-entry after leaving
  trigger: Same person returns minutes later.
  handling: TubeLinker ties a newborn tube to an entity whose tube went lost within 30 s and 400 px when appearance cosine >= 0.75; emits Event(relink); the agent reports entities, not tubes.
  status: implemented
- id: E-TUBE-05
  ring: ring2
  title: Same person in two overlapping cameras
  trigger: Overlap region.
  handling: Per-camera tracker never merges; fusion merges by floor distance + ReID + time; enrichment once per entity, best view elected.
  status: planned
- id: E-TUBE-06
  ring: ring2
  title: Hand-off outside transit bounds
  trigger: Entity appears in a non-adjacent tile too fast.
  handling: impossible_transition event; no merge; anomaly surfaced.
  status: planned
- id: E-TUBE-07
  ring: ring2
  title: Blind-spot ambiguity
  trigger: Two enter a blind spot, one exits.
  handling: Delay-state keeps both candidates with probabilities; answer states ambiguity.
  status: deferred
- id: E-TUBE-08
  ring: ring2
  title: Appearance drift within a day
  trigger: Jacket on/off; bag picked up.
  handling: Per-entity EMA embedding plus up to 5 exemplars refreshed on active ticks; matching takes the best of EMA and exemplars so a jacket-off entity still links.
  status: implemented
- id: E-TUBE-09
  ring: ring2
  title: Carried object is not its own tube
  trigger: Bag on shoulder; keys in hand.
  handling: carried_item attribute on the person tube; object tube only when placed and stationary.
  status: planned
- id: E-TUBE-10
  ring: ring2
  title: Pets, strollers, wheelchairs
  trigger: Non-person moving classes; child inside stroller.
  handling: Class-specific lifecycle; stroller+child handled as one tube with carried_item=child hint.
  status: deferred
- id: E-TUBE-13
  ring: ring2
  title: Fragmentation at sampled frame rate
  trigger: Decode at 2 fps means a walking person moves half a box width between ticks; IoU association breaks the tube.
  handling: ByteTracker with dt-aware Kalman, buffered IoU and a first-tick centre gate keeps fragmentation at 1.0 down to ~4-6 fps (synthetic); below ~3 fps motion-only association is ambiguous by construction, so active tiles decode at >=4 fps and 2 fps is for quiet tiles only.
  status: implemented
- id: E-TUBE-14
  ring: ring2
  title: Predicted box balloons during long occlusion
  trigger: Kalman aspect/height velocities keep integrating while a track is unseen; the predicted box grows over the scene and re-attaches to junk (seen in the session-08 debug frames).
  handling: KalmanBoxFilter clamps predicted height and aspect to 0.6–1.6× the last measurement and zeroes the size velocities when the clamp engages.
  status: implemented
- id: E-TUBE-12
  ring: ring2
  title: Feet occluded, foot point wrong
  trigger: Person behind counter.
  handling: FloorPoint.source=fallback_bbox_bottom and uncertainty widened; fusion tolerances read it.
  status: implemented
# ---------------- ring 3a events ----------------
- id: E-EVT-01
  ring: ring3a
  title: Zone boundary jitter
  trigger: Foot point oscillates on a zone edge.
  handling: 'Hysteresis: enter after N consecutive inside ticks, exit after N outside.'
  status: implemented
- id: E-EVT-10
  ring: ring3a
  title: Everyone in frame at episode open "enters"
  trigger: First tick births every visible tube inside a zone; enter_zone fires for all of them at once.
  handling: EventCompiler seeds zone memberships silently for open_grace_ms (1.5 s) after its first tick, since the gate needs a frame to warm up; the open heartbeat makes that first tick non-empty.
  status: implemented
- id: E-EVT-02
  ring: ring3a
  title: Cross-camera event order under clock offset
  trigger: Kitchen camera 2 s ahead of hallway.
  handling: Sort by corrected_ms; hand-off window tolerates offset_confidence.
  status: planned
- id: E-EVT-03
  ring: ring3a
  title: False pickup from occlusion of the shelf
  trigger: Person stands in front of the asset home; heartbeat cannot see the asset.
  handling: Pickup requires asset absent on a heartbeat taken with nobody inside the zone, after a present reading; subject = visitors in between; no visitors => asset_missing_from_home with no blame.
  status: implemented
- id: E-EVT-04
  ring: ring3a
  title: Fall vs lying down on purpose
  trigger: Person lies on sofa/bed.
  handling: rest zones from scene card suppress fall; fall requires vertical velocity + pose primitive outside rest zones.
  status: deferred
- id: E-EVT-05
  ring: ring3a
  title: Left-behind vs placed at home
  trigger: Bag set on its usual hook.
  handling: left_behind only outside the object's home zone.
  status: planned
- id: E-EVT-06
  ring: ring3a
  title: Duplicate events from overlapping cameras
  trigger: Both cameras see the same enter.
  handling: dedupe_key = type|entity|time bucket at fusion; per-camera events keep camera_id for evidence.
  status: implemented
- id: E-EVT-07
  ring: ring3a
  title: Episode never closes
  trigger: Busy lobby active for hours.
  handling: soft_cut on cast churn > threshold or max duration; events may span episodes.
  status: implemented
- id: E-EVT-08
  ring: ring3a
  title: Event spans an episode boundary
  trigger: Dwell starts in one episode and ends in the next.
  handling: Events carry episode_id of emission; queries join across episodes by entity and time, never by episode alone.
  status: planned
# ---------------- ring 3b foveation / enrichment ----------------
- id: E-FOV-01
  ring: ring3b
  title: Crop too small to describe
  trigger: Short side < 96 px.
  handling: Enhancement ladder; enhanced=true propagates to attributes; confidence discounted.
  status: implemented
- id: E-FOV-02
  ring: ring3b
  title: Motion blur
  trigger: Fast movement.
  handling: Keyframe score uses Laplacian sharpness; blurred frames skipped; if none sharp, attributes low confidence.
  status: planned
- id: E-FOV-03
  ring: ring3b
  title: Backlit or overexposed subject
  trigger: Doorway against daylight.
  handling: Exposure score in keyframe selection; try other keyframes; color_reason=low_confidence.
  status: planned
- id: E-FOV-04
  ring: ring3b
  title: Cross-cell attribute bleed in contact sheets
  trigger: VLM describes cell 3 with cell 4's jacket.
  handling: Hard borders + cell ids; fixed expected_cells; per-cell schema; bleed rate measured on eval set; fall back to 12 cells if >5%.
  status: implemented
- id: E-FOV-05
  ring: ring3b
  title: Hallucinated color in IR
  trigger: Monochrome crop.
  handling: Attributes validator and grammar forbid colors when modality is ir/thermal; color_reason=ir_mode; tone fields instead.
  status: implemented
- id: E-FOV-06
  ring: ring3b
  title: Ring buffer expired before enrichment
  trigger: Slow path lags past buffer window.
  handling: Best keyframe crop persisted to R2 at tube birth; keyframe_refs on the tube.
  status: implemented
- id: E-FOV-07
  ring: ring3b
  title: VLM output violates schema
  trigger: Grammar bug or truncation.
  handling: Retry once; then EnrichmentPatch{failed=true}; tick never blocked.
  status: implemented
- id: E-FOV-08
  ring: ring3b
  title: Attributes contradict across keyframes
  trigger: Red jacket in one, dark in another.
  handling: Majority vote over keyframes; keep history; confidence reflects agreement.
  status: planned
# ---------------- storage / handoff ----------------
- id: E-STO-01
  ring: storage
  title: Patch arrives after episode closed
  trigger: Slow path finishes late.
  handling: Episode file is append-only; PatchRecord accepted after close.
  status: implemented
- id: E-STO-02
  ring: storage
  title: Retries create duplicates
  trigger: Writer crashes and replays.
  handling: Deterministic episode/event ids; per-record idempotency keys.
  status: implemented
- id: E-STO-03
  ring: storage
  title: Retention expired for a clip
  trigger: Clip requested for footage past retention.
  handling: clip tool checks availability; degrades to keyframes; answer states retention.
  status: planned
- id: E-STO-04
  ring: storage
  title: Schema version bump
  trigger: New field added mid-deployment.
  handling: Provenance.schema_version on every record; readers tolerate unknown fields; migration script per bump.
  status: implemented
- id: E-STO-05
  ring: storage
  title: KB version drift
  trigger: Transit bounds changed after an episode was compiled.
  handling: Provenance.kb_version on every record; replays pin the version.
  status: implemented
# ---------------- block 2 agent ----------------
- id: E-AGT-01
  ring: agent
  title: Ambiguous anchor
  trigger: '"After Jay went out" matches two exits.'
  handling: One clarifying question; never a silent pick.
  status: planned
- id: E-AGT-02
  ring: agent
  title: No results
  trigger: Nothing matches the window.
  handling: Empty tool results are shown to the model as such; the answer must say nothing matched and suggest widening; citations empty.
  status: implemented
- id: E-AGT-03
  ring: agent
  title: Unknown named entity
  trigger: '"Jay" not in gallery.'
  handling: Ask who Jay is; offer naming form.
  status: planned
- id: E-AGT-04
  ring: agent
  title: Fuzzy time expressions and guessed vocabulary
  trigger: '"yesterday afternoon", "after lunch"; the model invents event names like pick_up.'
  handling: Event and zone vocabularies are in the system prompt; tool args are validated and a miss returns the valid list so the model corrects itself; time windows are echoed in answers.
  status: implemented
- id: E-AGT-05
  ring: agent
  title: Color question on IR footage
  trigger: '"what color was his shirt" at night.'
  handling: State IR mode; give tone; never a color.
  status: planned
- id: E-AGT-06
  ring: agent
  title: Unverified claim
  trigger: Attribute from a low-confidence patch.
  handling: The agent loop keeps the set of ids that appeared in tool results; an answer whose citations are not in that set is sent back once for revision, then returned with cited=false. verify() lands with the writer VLM.
  status: implemented
- id: E-AGT-07
  ring: agent
  title: Broken custody chain
  trigger: carried_item missed; drop never fired.
  handling: 'Graceful degrade: last known custody + offer clip.'
  status: planned
- id: E-AGT-08
  ring: agent
  title: Duplicates from overlapping cameras in the answer
  trigger: Same person listed twice.
  handling: Answer over entities, not tubes.
  status: planned
# ---------------- KB / calibration ----------------
- id: E-KB-01
  ring: kb
  title: Camera moved after calibration
  trigger: Bumped or remounted.
  handling: Sustained global motion => camera_moved_suspect; landmark check on I-frames; freeze fusion; prompt recalibration.
  status: implemented
- id: E-KB-02
  ring: kb
  title: Scene card labels wrong
  trigger: '"Dining hall" is actually the kitchen.'
  handling: Everything on the card is a hypothesis; user correction bumps version and re-labels.
  status: planned
- id: E-KB-03
  ring: kb
  title: Causal fact contradicted
  trigger: Switch flipped, light did not change.
  handling: Fact.add_contradiction; retire at 2 unless user_confirmed.
  status: implemented
- id: E-KB-04
  ring: kb
  title: Furniture moved
  trigger: Shelf relocated; asset home zone stale.
  handling: Repeated asset_missing with asset detected elsewhere => propose new home.
  status: deferred
# ---------------- night ----------------
- id: E-NIGHT-01
  ring: night
  title: IR switch mid-tube
  trigger: Camera toggles IR while a person is in view.
  handling: modality_switch event; tube segments carry modality; attributes per segment.
  status: planned
- id: E-NIGHT-02
  ring: night
  title: ReID does not transfer day to IR
  trigger: Same person, different modality.
  handling: Separate exemplar sets per modality; tile/time continuity weighted higher at night.
  status: planned
- id: E-NIGHT-03
  ring: night
  title: IR blooming near camera
  trigger: Subject washed out by illuminator.
  handling: Exposure score rejects keyframe; low confidence attributes.
  status: deferred
# ---------------- privacy ----------------
- id: E-PRIV-01
  ring: privacy
  title: Gallery entry deletion
  trigger: User deletes a named person.
  handling: 'Cascade: exemplars, names on records, KB facts referencing the entity; anonymous ids remain.'
  status: planned
- id: E-PRIV-02
  ring: privacy
  title: Face embeddings without opt-in
  trigger: Face model enabled by default.
  handling: Site-level opt-in flag gates the face lane; default off.
  status: planned
EOF_VI
chmod +x colab/vllm_venv.sh
cp "$0" colab/sessions/session_16_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. preflight"
bash colab/preflight.sh | tee data/bench/preflight.json

step "4. install + harness"
pipi -e ".[dev,ingest]"
make check

step "5. database (schema changed: quality columns -> fresh database)"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed"
  fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" && sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"
    DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "6. slice (frame + SigLIP, quality flags) -> episode -> store"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "\"(person_tubes|person_tubes_low_quality|entities|relinks)\"" /tmp/slice_final.log | sed 's/^/  /'
else
  warn "no GPU or no clip at $SOURCE; using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "7. reasoning backend (server is reused if already up, and left running)"
BACKEND="${BACKEND:-}"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ -z "$BACKEND" ]; then
  if [ "$HAS_GPU" = 1 ] && [ "${SKIP_VLLM:-0}" != "1" ]; then
    if bash colab/vllm_venv.sh start "$AGENT_MODEL" 8000; then BACKEND=openai; else warn "vLLM path failed -> transformers in-process"; fi
  fi
  if [ -z "$BACKEND" ] && [ "$HAS_GPU" = 1 ]; then python -c "import transformers" 2>/dev/null || pipi transformers; BACKEND=transformers; fi
  [ -z "$BACKEND" ] && { BACKEND=fake; warn "no GPU: fake backend"; }
fi
echo "backend: $BACKEND ($AGENT_MODEL)"

step "8. scene script head, then the three questions with latency"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 6 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and when did they arrive?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -v "^\s*\"" | grep -v "^[{}]"

step "9. scenario acceptance (fill tests/scenarios/warehouse.yaml ground_truth to make this meaningful)"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | tail -12

step "10. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: inline script + prefix cache + persistent server, vocabulary validation, latency per question, tube quality flags, scenario eval"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "11. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste step 8 (latency lines + answers) and step 9 back. The vLLM server is still running for further questions."
