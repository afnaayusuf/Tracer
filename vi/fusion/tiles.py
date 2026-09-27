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
    """The BuF's structure. homo: one tile, every camera in it. hetero: several tiles plus the
    adjacency graph (which tiles hand people to which, and how long the walk takes)."""
    tiles: dict[str, list[str]] = field(default_factory=dict)     # tile id -> cameras
    affinity: dict[str, float] = field(default_factory=dict)       # "camA|camB" -> simultaneous-match rate
    adjacency: dict[str, dict] = field(default_factory=dict)       # "T1|T2" -> {"median_gap_ms", "max_gap_ms", "handoffs"}
    evidence: dict[str, dict] = field(default_factory=dict)
    kind: str = "auto"                                              # homo | hetero | auto
    version: int = 1

    @classmethod
    def homo(cls, cameras: list[str]) -> "TileMap":
        return cls(tiles={"T1": sorted(cameras)}, kind="homo")

    def tile_of(self, camera_id: str) -> str:
        for t, cams in self.tiles.items():
            if camera_id in cams:
                return t
        return camera_id                                            # unassigned camera: its own tile

    def relation(self, cam_a: str, cam_b: str) -> tuple[str, int]:
        """('same', 0) | ('adjacent', max_gap_ms) | ('none', 0) between two cameras."""
        ta, tb = self.tile_of(cam_a), self.tile_of(cam_b)
        if ta == tb:
            return "same", 0
        key = f"{min(ta, tb)}|{max(ta, tb)}"
        if key in self.adjacency:
            return "adjacent", int(self.adjacency[key].get("max_gap_ms", 60_000))
        return "none", 0

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"tiles": self.tiles, "affinity": self.affinity, "adjacency": self.adjacency,
                                          "evidence": self.evidence, "kind": self.kind, "version": self.version}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "TileMap | None":
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(tiles=d.get("tiles", {}), affinity=d.get("affinity", {}), adjacency=d.get("adjacency", {}),
                   evidence=d.get("evidence", {}), kind=d.get("kind", "auto"), version=d.get("version", 1))


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
    tm = TileMap(tiles=tiles, affinity={k: v["rate"] for k, v in aff.items()}, evidence=aff,
                 kind="homo" if len(tiles) == 1 and len(cams) > 1 else ("hetero" if len(tiles) > 1 else "auto"))
    tm.adjacency = tile_adjacency(entities, tm, sim_thr)
    return tm


def tile_adjacency(entities: list[dict], tm: TileMap, sim_thr: float = 0.85, max_travel_ms: int = 180_000,
                   min_handoffs: int = 2) -> dict[str, dict]:
    """Hand-offs between tiles: a person lost on tile A and matched on tile B shortly after, with
    no overlap. Enough of them make the tiles adjacent, with the observed travel time as the window."""
    by_tile: dict[str, list[dict]] = {}
    for e in entities:
        if e.get("embedding"):
            by_tile.setdefault(tm.tile_of(e["camera_id"]), []).append(e)
    out: dict[str, dict] = {}
    for a, b in combinations(sorted(by_tile), 2):
        gaps = []
        for ea in by_tile[a]:
            va = np.asarray(ea["embedding"], np.float32)
            for eb in by_tile[b]:
                gap = max(ea["first_seen_ms"], eb["first_seen_ms"]) - min(ea["last_seen_ms"], eb["last_seen_ms"])
                if gap <= 0 or gap > max_travel_ms:
                    continue
                vb = np.asarray(eb["embedding"], np.float32)
                if va.shape == vb.shape and float(va @ vb) / (float(np.linalg.norm(va)) * float(np.linalg.norm(vb)) + 1e-8) >= sim_thr:
                    gaps.append(gap)
        if len(gaps) >= min_handoffs:
            gaps.sort()
            out[f"{a}|{b}"] = {"handoffs": len(gaps), "median_gap_ms": int(gaps[len(gaps) // 2]), "max_gap_ms": int(min(max_travel_ms, gaps[-1] * 1.5 + 5000))}
    return out


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
