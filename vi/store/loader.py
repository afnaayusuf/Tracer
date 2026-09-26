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
