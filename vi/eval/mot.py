"""CLEAR-MOT (MOTA/MOTP/IDSW) and IDF1 from first principles, so Ring 2 has a baseline
number before any external evaluator is installed. Inputs are {frame: [(id, Box), ...]}.
Matching follows the CLEAR convention: keep last frame's pairing while IoU stays above the
threshold, Hungarian-match the rest."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from vi.schemas import Box

Tracks = dict[int, list[tuple[int, Box]]]


@dataclass
class MOTResult:
    num_gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    iou_sum: float = 0.0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    gt_ids: int = 0
    pred_ids: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def mota(self) -> float:
        return 1.0 - (self.fn + self.fp + self.idsw) / self.num_gt if self.num_gt else 0.0

    @property
    def motp(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def idf1(self) -> float:
        d = 2 * self.idtp + self.idfp + self.idfn
        return 2 * self.idtp / d if d else 0.0

    @property
    def fragmentation_ratio(self) -> float:
        """pred ids per gt id: 1.0 is perfect; 3.0 means every person became three tubes."""
        return self.pred_ids / self.gt_ids if self.gt_ids else 0.0

    def as_row(self) -> dict:
        return {"mota": round(self.mota, 4), "motp": round(self.motp, 4), "idf1": round(self.idf1, 4),
                "idsw": self.idsw, "fp": self.fp, "fn": self.fn, "tp": self.tp, "num_gt": self.num_gt,
                "gt_ids": self.gt_ids, "pred_ids": self.pred_ids,
                "fragmentation_ratio": round(self.fragmentation_ratio, 3), **self.extra}


def _iou_matrix(a: list[Box], b: list[Box]) -> np.ndarray:
    m = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            m[i, j] = x.iou(y)
    return m


def evaluate_mot(gt: Tracks, pred: Tracks, iou_thr: float = 0.5) -> MOTResult:
    res = MOTResult()
    frames = sorted(set(gt) | set(pred))
    last_match: dict[int, int] = {}                 # gt id -> pred id from previous frame
    overlap: dict[tuple[int, int], int] = {}        # (gt id, pred id) -> co-occurrence frames
    gt_frames: dict[int, int] = {}
    pred_frames: dict[int, int] = {}
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        res.num_gt += len(g)
        for gid, _ in g:
            gt_frames[gid] = gt_frames.get(gid, 0) + 1
        for pid, _ in p:
            pred_frames[pid] = pred_frames.get(pid, 0) + 1
        if not g or not p:
            res.fn += len(g)
            res.fp += len(p)
            continue
        iou = _iou_matrix([b for _, b in g], [b for _, b in p])
        matched_g: set[int] = set()
        matched_p: set[int] = set()
        pairs: list[tuple[int, int]] = []
        # 1. keep previous pairings that still overlap
        pid_index = {pid: j for j, (pid, _) in enumerate(p)}
        for i, (gid, _) in enumerate(g):
            pid = last_match.get(gid)
            if pid is not None and pid in pid_index and iou[i, pid_index[pid]] >= iou_thr:
                pairs.append((i, pid_index[pid]))
                matched_g.add(i)
                matched_p.add(pid_index[pid])
        # 2. Hungarian on the rest
        gi = [i for i in range(len(g)) if i not in matched_g]
        pj = [j for j in range(len(p)) if j not in matched_p]
        if gi and pj:
            cost = 1.0 - iou[np.ix_(gi, pj)]
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if iou[gi[a], pj[b]] >= iou_thr:
                    pairs.append((gi[a], pj[b]))
                    matched_g.add(gi[a])
                    matched_p.add(pj[b])
        for i, j in pairs:
            gid, pid = g[i][0], p[j][0]
            res.tp += 1
            res.iou_sum += iou[i, j]
            if gid in last_match and last_match[gid] != pid:
                res.idsw += 1
            last_match[gid] = pid
            overlap[(gid, pid)] = overlap.get((gid, pid), 0) + 1
        res.fn += len(g) - len(pairs)
        res.fp += len(p) - len(pairs)
    # IDF1: global one-to-one assignment of gt ids to pred ids maximising co-occurrence
    gids = sorted(gt_frames)
    pids = sorted(pred_frames)
    res.gt_ids, res.pred_ids = len(gids), len(pids)
    total_gt = sum(gt_frames.values())
    total_pred = sum(pred_frames.values())
    if gids and pids:
        m = np.zeros((len(gids), len(pids)))
        for (gid, pid), n in overlap.items():
            m[gids.index(gid), pids.index(pid)] = n
        r, c = linear_sum_assignment(-m)
        res.idtp = int(m[r, c].sum())
    res.idfp = total_pred - res.idtp
    res.idfn = total_gt - res.idtp
    return res


def load_mot_txt(path: str | Path, gt: bool = False, min_conf: float = 0.0) -> Tracks:
    """MOTChallenge text: frame,id,x,y,w,h,conf,class,visibility. For gt.txt keep
    pedestrians (class 1) flagged as considered (conf==1)."""
    tracks: Tracks = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split(",")[:9]]
        frame, tid, x, y, w, h = int(v[0]), int(v[1]), v[2], v[3], v[4], v[5]
        conf = v[6] if len(v) > 6 else 1.0
        cls = int(v[7]) if len(v) > 7 else 1
        if gt and (conf != 1 or cls != 1):
            continue
        if not gt and conf < min_conf:
            continue
        if w <= 0 or h <= 0:
            continue
        tracks.setdefault(frame, []).append((tid, Box(x1=x, y1=y, x2=x + w, y2=y + h)))
    return tracks
