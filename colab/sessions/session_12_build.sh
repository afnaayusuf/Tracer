#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 12 build: the episode becomes queryable (Block 2 evidence layer).
#   * vi/store: SQLAlchemy schema (episodes, ticks, tubes, entities, events, patches, custody, facts,
#     scripts) on SQLite or PostgreSQL; idempotent loader keyed by the writers' deterministic ids
#   * vi/agent/tools.py: search_events / search_tubes / search_entities, clip (evidence refs),
#     get_script (the scene script the reasoning model reads; cached)
#   * episode files now carry final Tube records (entity ids, keyframes); writer resumes after a
#     crash without duplicating records (E-STO-02)
#   * linker: tubes that closed through an exit zone relink at a stricter 0.85
#  Runs: slice (frame + SigLIP) -> episode file -> Postgres (installed here) or SQLite -> scene script.
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_12.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       DB_URL (default: local Postgres if it can be started, else sqlite) NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 12"
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
[ -n "$(git log --grep='^session 11' --format=%h)" ] || die "session 11 commit not found; run build_session_11b.sh first"
if grep -rn "import cv2\|from cv2\|opencv" vi bench tests pyproject.toml 2>/dev/null; then
  [ "${FORCE:-0}" = "1" ] || die "cv2/opencv found in code (bypasses vi.ingest)"
fi
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
mkdir -p vi/store vi/agent colab/sessions data/bench
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
    last_box_center: tuple[float, float] = (0.0, 0.0)
    last_box_h: float = 1.0
    lost_at_ms: int | None = None      # set when its current tube closed (lost or exited); None while live
    closed_as_exit: bool = False


class TubeLinker:
    def __init__(self, camera_id: str, sim_thr: float = 0.75, max_gap_ms: int = 30_000,
                 max_jump_px: float = 400.0, ema_alpha: float = 0.3, exemplars: int = 5,
                 exited_sim_thr: float = 0.85):
        self.camera_id = camera_id
        self.sim_thr = sim_thr
        self.exited_sim_thr = exited_sim_thr   # placeholder exit zones misclassify lost as exited; allow with more evidence
        self.max_gap_ms = max_gap_ms
        self.max_jump_px = max_jump_px
        self.ema_alpha = ema_alpha
        self.n_exemplars = exemplars
        self._entities: dict[str, _Entity] = {}
        self._tube_entity: dict[str, str] = {}
        self._seq = 0
        self.relinks = 0

    # ---------------------------------------------------------------- helpers
    def _new_entity(self, tube: Tube, emb: np.ndarray) -> _Entity:
        self._seq += 1
        ent = _Entity(entity_id=f"{self.camera_id}:E{self._seq}", tube_ids=[tube.tube_id], ema=emb.copy(),
                      exemplars=[emb.copy()], last_box_center=_center(tube), last_box_h=tube.box.height)
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
            if e.lost_at_ms is None or t_ms - e.lost_at_ms > self.max_gap_ms:
                continue
            if ((cx - e.last_box_center[0]) ** 2 + (cy - e.last_box_center[1]) ** 2) ** 0.5 > self.max_jump_px:
                continue
            out.append(e)
        return out

    # ---------------------------------------------------------------- API
    def on_birth(self, tube: Tube, emb: np.ndarray, t_ms: int) -> Event | None:
        cands = self._candidates(tube, t_ms)
        if cands:
            best = max(cands, key=lambda e: self._sim(e, emb))
            sim = self._sim(best, emb)
            thr = self.exited_sim_thr if best.closed_as_exit else self.sim_thr
            if sim >= thr:
                prev = best.tube_ids[-1]
                best.tube_ids.append(tube.tube_id)
                best.lost_at_ms = None
                self._tube_entity[tube.tube_id] = best.entity_id
                self.on_refresh(tube, emb)
                self.relinks += 1
                return Event(event_id="ev_" + hashlib.sha1(f"relink|{prev}|{tube.tube_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[prev, tube.tube_id], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(sim, 3), "gap_ms": t_ms - (tube.born.corrected_ms())},
                             confidence=min(1.0, sim))
        self._new_entity(tube, emb)
        return None

    def on_refresh(self, tube: Tube, emb: np.ndarray) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.ema = e.ema * (1 - self.ema_alpha) + emb * self.ema_alpha
        e.ema /= max(1e-8, np.linalg.norm(e.ema))
        e.exemplars.append(emb.copy())
        if len(e.exemplars) > self.n_exemplars:
            e.exemplars.pop(0)
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height

    def on_close(self, tube: Tube, t_ms: int) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        e.lost_at_ms = t_ms if tube.state in (TubeState.lost, TubeState.occluded, TubeState.exited) else None
        e.closed_as_exit = tube.state == TubeState.exited

    def entity_of(self, tube_id: str) -> str | None:
        return self._tube_entity.get(tube_id)

    @property
    def entities(self) -> int:
        return len(self._entities)


def _center(t: Tube) -> tuple[float, float]:
    return ((t.box.x1 + t.box.x2) / 2.0, (t.box.y1 + t.box.y2) / 2.0)
EOF_VI
cat > vi/schemas/episode.py << 'EOF_VI'
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from .common import CamTime, Provenance
from .event import Event
from .tick import Tick
from .tube import EnrichmentPatch, Tube


class EpisodeStatus(str, Enum):
    open = "open"
    closed = "closed"        # tile went quiet
    soft_cut = "soft_cut"    # E-EVT-07: cast churn or max duration
    truncated = "truncated"  # E-ING-02: signal lost


class CastMember(BaseModel):
    entity_id: str | None = None
    tube_ids: list[str]
    class_label: str
    named: str | None = None
    best_keyframe_ref: str | None = None


class EpisodeHeader(BaseModel):
    kind: Literal["header"] = "header"
    episode_id: str
    tile_id: str
    camera_ids: list[str]
    t0: CamTime
    provenance: Provenance


class TickRecord(BaseModel):
    kind: Literal["tick"] = "tick"
    episode_id: str
    tick: Tick


class EventRecord(BaseModel):
    kind: Literal["event"] = "event"
    episode_id: str
    event: Event


class PatchRecord(BaseModel):
    """E-STO-01: patches may arrive after close; the file is append-only."""

    kind: Literal["patch"] = "patch"
    episode_id: str
    patch: EnrichmentPatch


class TubeRecord(BaseModel):
    """Final state of a tube (entity id, keyframes, merge candidates), written at close time."""

    kind: Literal["tube"] = "tube"
    episode_id: str
    tube: Tube


class EpisodeClose(BaseModel):
    kind: Literal["close"] = "close"
    episode_id: str
    t1: CamTime
    status: EpisodeStatus
    cast: list[CastMember] = Field(default_factory=list)


EpisodeRecord = Annotated[
    Union[EpisodeHeader, TickRecord, EventRecord, PatchRecord, TubeRecord, EpisodeClose],
    Field(discriminator="kind"),
]
episode_record_adapter: TypeAdapter = TypeAdapter(EpisodeRecord)
EOF_VI
cat > vi/schemas/export.py << 'EOF_VI'
"""Export JSON Schema for every contract model to ./schemas. Run: python -m vi.schemas.export"""
from __future__ import annotations

import json
from pathlib import Path

from . import (Attributes, ContactSheetResult, EnrichmentPatch, Event, Fact, SceneCard, Tick, Tube)
from .episode import EpisodeClose, EpisodeHeader, EventRecord, PatchRecord, TickRecord, TubeRecord

MODELS = [Tick, Tube, Attributes, EnrichmentPatch, Event, SceneCard, Fact, ContactSheetResult,
          EpisodeHeader, TickRecord, EventRecord, PatchRecord, TubeRecord, EpisodeClose]


def main(out_dir: str | Path = "schemas") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for m in MODELS:
        p = out / f"{m.__name__}.schema.json"
        p.write_text(json.dumps(m.model_json_schema(), indent=2))
        written.append(p)
    print(f"exported {len(written)} schemas to {out}/")
    return written


if __name__ == "__main__":
    main()
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
                    max_duration_ms: int = 30 * 60_000, churn_thr: float = 0.6) -> bool:
    """E-EVT-07: a busy tile never goes quiet, so an episode is cut when the cast has
    mostly turned over or the episode exceeds max duration."""
    if duration_ms >= max_duration_ms:
        return True
    if not cast_prev:
        return False
    churn = 1.0 - len(cast_prev & cast_now) / len(cast_prev | cast_now)
    return churn >= churn_thr
EOF_VI
cat > vi/store/__init__.py << 'EOF_VI'
from .db import connect, metadata
from .loader import load_episode_file
EOF_VI
cat > vi/store/db.py << 'EOF_VI'
"""Relational store for episodes (R24). SQLAlchemy Core so the same code runs on SQLite (tests,
laptop) and PostgreSQL 18 (Colab, production). Embeddings are JSON here; pgvector columns come
with the retrieval work. Every table is keyed by the deterministic ids the writers already
produce, so loading is idempotent (E-STO-02)."""
from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

metadata = MetaData()

episodes = Table(
    "episodes", metadata,
    Column("episode_id", String, primary_key=True), Column("tile_id", String, index=True),
    Column("camera_ids", JSON), Column("t0_ms", Integer, index=True), Column("t1_ms", Integer),
    Column("status", String), Column("kb_version", Integer), Column("schema_version", String),
)
ticks = Table(
    "ticks", metadata,
    Column("episode_id", String, primary_key=True), Column("camera_id", String, primary_key=True),
    Column("tick_index", Integer, primary_key=True), Column("t_start_ms", Integer, index=True),
    Column("t_end_ms", Integer), Column("modality", String), Column("tubes", JSON),
    Column("event_ids", JSON), Column("scene_state", JSON), Column("gate_energy", Float),
)
tubes = Table(
    "tubes", metadata,
    Column("tube_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("camera_id", String, index=True), Column("tile_id", String), Column("entity_id", String, index=True),
    Column("named", String), Column("class_label", String, index=True), Column("state", String),
    Column("born_ms", Integer, index=True), Column("last_seen_ms", Integer), Column("box", JSON),
    Column("zone_ids", JSON), Column("modality", String), Column("keyframe_refs", JSON),
    Column("attributes", JSON), Column("merge_candidates", JSON),
)
entities = Table(
    "entities", metadata,
    Column("entity_id", String, primary_key=True), Column("camera_id", String), Column("named", String),
    Column("class_label", String), Column("tube_ids", JSON), Column("first_seen_ms", Integer, index=True),
    Column("last_seen_ms", Integer), Column("best_keyframe_ref", String), Column("embedding", JSON),
)
events = Table(
    "events", metadata,
    Column("event_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("type", String, index=True), Column("t_ms", Integer, index=True), Column("camera_id", String),
    Column("tile_id", String), Column("zone_id", String, index=True), Column("subject_tube_ids", JSON),
    Column("subject_entity_ids", JSON), Column("object_ids", JSON), Column("payload", JSON),
    Column("confidence", Float), Column("source", String), Column("dedupe_key", String),
)
patches = Table(
    "patches", metadata,
    Column("patch_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("tube_id", String, index=True), Column("produced_at_ms", Integer), Column("source", String),
    Column("payload", JSON), Column("confidence", Float), Column("modality", String),
    Column("enhanced", Boolean), Column("failed", Boolean),
)
custody = Table(
    "custody", metadata,
    Column("event_id", String, primary_key=True), Column("object_id", String, index=True),
    Column("person_tube_ids", JSON), Column("t_ms", Integer), Column("kind", String), Column("zone_id", String),
)
facts = Table(
    "facts", metadata,
    Column("fact_id", String, primary_key=True), Column("subject", String, index=True), Column("predicate", String),
    Column("object", JSON), Column("status", String), Column("confidence", Float), Column("source", String),
    Column("support", Integer), Column("contradictions", Integer), Column("evidence", JSON),
    Column("first_seen_ms", Integer), Column("last_confirmed_ms", Integer), Column("version", Integer),
)
scripts = Table(
    "scripts", metadata,
    Column("episode_id", String, primary_key=True), Column("text", Text), Column("rendered_at_ms", Integer),
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
                                  merge_candidates=t.merge_candidates))
        elif isinstance(rec, EpisodeClose):
            close = rec
    if header is None:
        raise ValueError(f"{path}: no header record")
    ent_rows: dict[str, dict] = {}
    for r in tube_rows:
        e = ent_rows.setdefault(r["entity_id"], dict(entity_id=r["entity_id"], camera_id=r["camera_id"], named=r["named"],
                                                     class_label=r["class_label"], tube_ids=[], first_seen_ms=r["born_ms"],
                                                     last_seen_ms=r["last_seen_ms"], best_keyframe_ref=None, embedding=None))
        e["tube_ids"].append(r["tube_id"])
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
cat > vi/agent/__init__.py << 'EOF_VI'
from .tools import clip, get_script, search_entities, search_events, search_tubes
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
    lines = [f"EPISODE {ep['episode_id']} | tile {ep['tile_id']} | cameras {','.join(ep['camera_ids'] or [])} | "
             f"{_ts(t0, t0)}–{_ts(ep['t1_ms'] or t0, t0)} | status {ep['status']} | kb v{ep['kb_version']}",
             f"CAST: {len(by_entity)} entities, {len(tube_rows)} tubes"]
    for eid, rows in by_entity.items():
        first, last = min(r["born_ms"] for r in rows), max(r["last_seen_ms"] for r in rows)
        name = rows[0]["named"] or ("anonymous" if eid.startswith("anon:") else "unnamed")
        kf = next((k for r in rows for k in (r["keyframe_refs"] or [])), None)
        lines.append(f"  {eid}  {rows[0]['class_label']}  {name}  tubes {','.join(r['tube_id'] for r in rows)}  "
                     f"seen {_ts(first, t0)}–{_ts(last, t0)}  states {','.join(sorted({r['state'] for r in rows}))}"
                     + (f"  keyframe {kf}" if kf else ""))
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
from vi.tubes import TRACKERS, TubeLinker


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
            due_birth = [t for t in live if t.class_label == "person" and t.tube_id not in last_embed_tick]
            due_refresh = [t for t in live if t.class_label == "person" and t.state.value == "active"
                           and t.tube_id in last_embed_tick and frames - last_embed_tick[t.tube_id] >= a.reid_every_ticks]
            todo = due_birth + due_refresh
            if todo:
                t0 = time.perf_counter()
                embs = embedder.embed([crop_for_embedding(fr.rgb, t.box) for t in todo])
                embed_ms.append((time.perf_counter() - t0) * 1000)
                for t, e in zip(todo, embs):
                    last_embed_tick[t.tube_id] = frames
                    if t.tube_id in {x.tube_id for x in due_birth}:
                        ev = linker.on_birth(t, e, fr.pts_ms)
                        if ev is not None:
                            events.append(ev)
                            relink_events += 1
                            print(f"t={fr.pts_ms:7d}  relink                 {ev.subject_tube_ids[0]} -> {ev.subject_tube_ids[1]} sim={ev.payload['similarity']}")
                    else:
                        linker.on_refresh(t, e)
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
cat > bench/slice_cpu.py << 'EOF_VI'
"""End-to-end CPU slice on a synthetic scene: gate -> (blob-as-detection) -> tubes -> events -> episode file.
Proves the plumbing with zero models. Run: python bench/slice_cpu.py [out_dir]"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vi.detect import Detection
from vi.episode import EpisodeWriter
from vi.events import EventCompiler, Zone
from vi.gate import FrameDiffGate
from vi.schemas import Box, CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.tubes import SimpleIoUTracker

W, H, TICK_MS = 320, 240, 500
SHELF = Zone(zone_id="shelf", camera_id="cam1", tile_id="entry", kind="asset_home", asset_id="bike_keys",
             polygon=[(200, 100), (270, 100), (270, 235), (200, 235)])
WARMUP = 4   # empty frames so the background does not contain the person
DOOR = Zone(zone_id="door", camera_id="cam1", tile_id="entry", kind="exit", polygon=[(0, 0), (30, 0), (30, 240), (0, 240)])


def synth_frame(rng, t_idx: int, keys_present: bool) -> np.ndarray:
    f = rng.normal(110, 3, (H, W)).clip(0, 255).astype(np.uint8)
    f[150:156, 220:226] = 40 if keys_present else 110           # the "keys" on the shelf
    x = 10 + (t_idx - WARMUP) * 12                                # person walks left -> right
    if WARMUP <= t_idx and x < W - 30:
        f[100:200, x:x + 24] = 230
    return f


def blobs_to_detections(gate_result) -> list[Detection]:
    return [Detection(box=b.box, class_label="person", confidence=0.6)
            for b in gate_result.blobs if b.box.height > 40]      # tiny blobs (the keys) are not persons


def main(out_dir: str = "data/episodes") -> Path:
    rng = np.random.default_rng(0)
    prov = Provenance(kb_version=1, pipeline_git="slice")
    gate = FrameDiffGate("cam1")
    tracker = SimpleIoUTracker("cam1", exit_boxes=[Box(x1=0, y1=0, x2=30, y2=240)])
    compiler = EventCompiler("cam1", [SHELF, DOOR], enter_ticks=1, exit_ticks=1, dwell_ms=1500, tile_id="entry")
    writer = EpisodeWriter(out_dir)
    ep = writer.open("entry", ["cam1"], CamTime(cam_utc_ms=0), prov)

    keys_present = True
    all_closed = []
    for i in range(44):
        t_ms = i * TICK_MS
        # the person is inside the shelf zone around ticks 20-24; the keys vanish while they are there
        if i == 22:
            keys_present = False
        frame = synth_frame(rng, i, keys_present)
        g = gate.update(frame, t_ms)
        events = compiler.on_gate(g)
        live, closed = tracker.update(blobs_to_detections(g), t_ms)
        all_closed += closed
        snaps = [TubeSnapshot(tube_id=tb.tube_id, class_label=tb.class_label, state=tb.state, box=tb.box) for tb in live]
        events += compiler.on_tick(snaps, t_ms)
        if i % 4 == 0:   # heartbeat on the asset home every 2 s
            inside = [s.tube_id for s in snaps if SHELF.contains(s.box.foot_point())]
            events += compiler.on_heartbeat("shelf", keys_present, t_ms, persons_inside=inside)
        tick = Tick(camera_id="cam1", tile_id="entry", tick_index=i, t_start=CamTime(cam_utc_ms=t_ms),
                    t_end=CamTime(cam_utc_ms=t_ms + TICK_MS), tubes=snaps,
                    event_ids=[e.event_id for e in events], gate_energy=sum(b.energy for b in g.blobs), provenance=prov)
        writer.write_tick(ep, tick)
        for e in events:
            writer.write_event(ep, e)
            print(f"t={t_ms:6d}  {e.type.value:24s} zone={e.zone_id} subjects={e.subject_tube_ids} objects={e.object_ids}")
    tubes = [tr.tube for tr in tracker._tracks.values()] + all_closed
    cast = [CastMember(tube_ids=[tb.tube_id], class_label=tb.class_label) for tb in tubes]
    for tb in tubes:
        writer.write_tube(ep, tb)
    writer.close(ep, CamTime(cam_utc_ms=44 * TICK_MS), EpisodeStatus.closed, cast)
    print(f"tubes: {[(tb.tube_id, tb.state.value) for tb in tubes]}")
    path = writer.path(ep)
    n = sum(1 for _ in EpisodeWriter.read(path))
    print(f"\nepisode {ep}: {n} records -> {path}")
    return path


if __name__ == "__main__":
    main(*sys.argv[1:])
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

from vi.agent import clip, get_script, search_entities, search_events, search_tubes
from vi.store import connect, load_episode_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--script-lines", type=int, default=40)
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
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "agent.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/README.md << 'EOF_VI'
# bench/

One script per ring. Each prints exactly one row of the benchmark table (see PLAN.md) and
checkpoints it to `data/bench/<ring>.jsonl` as it goes, so a killed Colab session loses nothing.

| script | ring | GPU | dataset | row |
|---|---|---|---|---|
| slice_cpu.py | all (stubs) | none | synthetic | proves plumbing; run first, every day |
| slice_gpu.py | 0–3a on real footage | L4 | own clip | reader → gate → ROIs (+ heartbeat) → RF-DETR → tubes → events → episode file + birth keyframes; `--detect roi\|frame\|hybrid`, `--reid hist\|siglip\|osnet\|auto` (entities, relinks), person_tubes, tube life, rebirths, duplicate trap, debug frames |
| ring0_gate.py | 0 | CPU | VIRAT | gate FN rate, ms per GOP per stream, MV vs framediff |
| ring1_detect.py | 1 | L4 | own clip / VIRAT | `--mode frame`: ms p50/p95 per frame at sampled fps; `--mode roi`: gate → packed ROI batches (nano/medium) |
| ring2_tubes.py | 2 | CPU | synthetic, MOT17 | MOTA/IDF1/IDSW/fragmentation (vi/eval/mot.py); `--tracker simple\|byte`, `--sample-every` simulates 2–5 fps decode |
| ring3b_sheet.py | 3b | A100-40 | crops from ring2 | attribute accuracy, bleed rate (12 vs 16 cells), ms per sheet |
| ring3a_events.py | 3a | CPU | MEVA subset | event precision/recall per type |
| agent_replay.py | block 2 | CPU | episode files | loads episodes into SQLite/Postgres, renders the scene script, runs search/clip; LLM loop comes later |

Colab: GPU runtime, `pip install rfdetr==1.7.0`, then run from the repo root (see colab/README.md).
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

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["vi*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["edge(id): test covers the edge case with this ID from edge_cases.yaml"]
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
    lk = TubeLinker("c1", sim_thr=0.75, max_gap_ms=30_000, max_jump_px=400)
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
    lk = TubeLinker("c1", max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 1000)
    assert lk.on_birth(tube("c1:20000:2", 300, t=20000), a, 20000) is None            # too long ago
    lk2 = TubeLinker("c1", max_gap_ms=10_000, max_jump_px=200)
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
    lk = TubeLinker("c1", sim_thr=0.8, ema_alpha=0.5, exemplars=3)
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
EOF_VI
cat > tests/test_episode.py << 'EOF_VI'
import pytest

from vi.episode import EpisodeWriter, episode_id_for, should_soft_cut
from vi.schemas import EnrichmentPatch, Event, EventType, Tick
from vi.schemas.episode import EpisodeClose, EpisodeStatus, PatchRecord


def _event(eid, t):
    return Event(event_id=eid, type=EventType.enter_zone, t=t(1000), camera_id="c1", subject_tube_ids=["a"])


@pytest.mark.edge("E-STO-02")
def test_writes_are_idempotent(tmp_path, prov, t):
    w = EpisodeWriter(tmp_path)
    e1 = w.open("kitchen", ["c1"], t(1000), prov)
    e2 = w.open("kitchen", ["c1"], t(1000), prov)
    assert e1 == e2 == episode_id_for("kitchen", 1000)
    assert w.write_event(e1, _event("ev_x", t)) is True
    assert w.write_event(e1, _event("ev_x", t)) is False
    tick = Tick(camera_id="c1", tick_index=0, t_start=t(1000), t_end=t(1500), provenance=prov)
    assert w.write_tick(e1, tick) and not w.write_tick(e1, tick)
    records = list(EpisodeWriter.read(w.path(e1)))
    assert [r.kind for r in records] == ["header", "event", "tick"]


@pytest.mark.edge("E-STO-01")
def test_patch_after_close_is_appended(tmp_path, prov, t):
    w = EpisodeWriter(tmp_path)
    e = w.open("kitchen", ["c1"], t(1000), prov)
    w.close(e, t(9000), EpisodeStatus.closed, cast=[])
    late = EnrichmentPatch(patch_id="p9", tube_id="a", produced_at_ms=12_000, source="vlm",
                           payload={"top_color": "red"}, confidence=0.8)
    assert w.write_patch(e, late) is True
    records = list(EpisodeWriter.read(w.path(e)))
    assert isinstance(records[-2], EpisodeClose) and isinstance(records[-1], PatchRecord)


@pytest.mark.edge("E-ING-02")
def test_signal_loss_truncates_episode(tmp_path, prov, t):
    w = EpisodeWriter(tmp_path)
    e = w.open("porch", ["c3"], t(0), prov)
    lost = Event(event_id="ev_lost", type=EventType.signal_lost, t=t(5000), camera_id="c3")
    w.write_event(e, lost)
    w.close(e, t(5000), EpisodeStatus.truncated, cast=[])
    records = list(EpisodeWriter.read(w.path(e)))
    assert records[-1].status == EpisodeStatus.truncated and records[-2].event.type == EventType.signal_lost


@pytest.mark.edge("E-EVT-07")
def test_soft_cut_on_cast_churn_or_max_duration():
    assert should_soft_cut({"a", "b", "c"}, {"a", "b", "c"}, 5 * 60_000) is False
    assert should_soft_cut({"a", "b", "c"}, {"x", "y", "z"}, 5 * 60_000) is True
    assert should_soft_cut({"a", "b", "c"}, {"a", "b", "c"}, 31 * 60_000) is True
    assert should_soft_cut(set(), {"a"}, 1000) is False
    assert EpisodeStatus.soft_cut.value == "soft_cut"


@pytest.mark.edge("E-STO-02")
def test_writer_restart_resumes_without_duplicating_records(tmp_path, prov, t):
    w = EpisodeWriter(tmp_path)
    e = w.open("kitchen", ["c1"], t(1000), prov)
    w.write_tick(e, Tick(camera_id="c1", tick_index=0, t_start=t(1000), t_end=t(1500), provenance=prov))
    w.write_event(e, _event("ev_x", t))
    # process crashes; a new writer replays the same episode
    w2 = EpisodeWriter(tmp_path)
    e2 = w2.open("kitchen", ["c1"], t(1000), prov)
    assert e2 == e
    assert w2.write_tick(e2, Tick(camera_id="c1", tick_index=0, t_start=t(1000), t_end=t(1500), provenance=prov)) is False
    assert w2.write_event(e2, _event("ev_x", t)) is False
    assert w2.write_tick(e2, Tick(camera_id="c1", tick_index=1, t_start=t(1500), t_end=t(2000), provenance=prov)) is True
    kinds = [r.kind for r in EpisodeWriter.read(w2.path(e2))]
    assert kinds == ["header", "tick", "event", "tick"]
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
EOF_VI
cat > .gitignore << 'EOF_VI'
__pycache__/
*.egg-info/
.pytest_cache/
data/episodes/
data/synthetic/
data/keyframes/
data/debug/
data/*.db
EOF_VI
cp "$0" colab/sessions/session_12_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ]; then
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    if ! command -v psql >/dev/null 2>&1; then
      echo "installing PostgreSQL (one-time per runtime, ~1 min)"
      apt-get install -y -qq postgresql >/dev/null 2>&1 || warn "apt install postgresql failed"
    fi
    if command -v psql >/dev/null 2>&1; then
      service postgresql start >/dev/null 2>&1 || true
      sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || \
        sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
      sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='vi'" | grep -q 1 || \
        sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
      pipi "psycopg[binary]>=3.2"
      DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
      echo "PostgreSQL: $(sudo -u postgres psql -tAc 'select version()' | cut -d, -f1)"
    fi
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "5. slice (frame + SigLIP) -> episode file -> store -> scene script"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes            # the episode id is deterministic per clip; start the file fresh
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "relink |\"(person_tubes|entities|relinks)\"" /tmp/slice_final.log | sed 's/^/  /'
else
  warn "no GPU or no clip; loading the CPU slice episode instead"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 36 2>&1 | grep -v "^\s*\"" | grep -v "^[{}]"

step "6. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: store + idempotent loader (SQLite/Postgres), agent tools + scene script, tube records in episodes, writer resume, exited relink threshold"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "7. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste the scene script and the agent row back."
