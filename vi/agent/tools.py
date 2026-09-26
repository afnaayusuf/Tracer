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
        out.append({"entity_id": eid, "first_seen_ms": max(t_start_ms, min(r["born_ms"] for r in rs)),
                    "last_seen_ms": min(t_end_ms, max(r["last_seen_ms"] for r in rs)), "coverage": round(covered / span, 3),
                    "tubes": len(rs), "quality": "ok" if ok else "low", "cameras": sorted({r["camera_id"] for r in rs}),
                    "looks": (attrs or {}).get("description") if attrs else None,
                    "keyframe": next((k for r in rs for k in (r["keyframe_refs"] or [])), None)})
    return sorted(out, key=lambda x: (-x["coverage"], x["first_seen_ms"]))


def count_entities_window(engine: Engine, t_start_ms: int, t_end_ms: int, class_label: str = "person", camera_id: str | None = None) -> dict:
    rows = coverage_window(engine, t_start_ms, t_end_ms, class_label, camera_id=camera_id)
    return {"count": len(rows), "entity_ids": [r["entity_id"] for r in rows], "window_ms": [t_start_ms, t_end_ms], "camera_id": camera_id}


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
    lines = [f"WINDOW {clock(t_start_ms)}–{clock(t_end_ms)} ({(t_end_ms - t_start_ms) / 60000:.1f} min) | episodes {len(eps)} | "
             + (f"camera {camera_id}" if camera_id else f"cameras {','.join(all_cams)}"),
             f"CAST: {len(cast)} confirmed people in this window" + ("" if camera_id or len(all_cams) <= 1 else
             " (entities are per camera; the same person on two cameras appears twice)")]
    for c in cast:
        lines.append(f"  {c['entity_id']}" + (f"  on {','.join(c['cameras'])}" if len(all_cams) > 1 else "")
                     + f"  seen {clock(c['first_seen_ms'])}–{clock(c['last_seen_ms'])}  coverage {c['coverage']:.0%}"
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
