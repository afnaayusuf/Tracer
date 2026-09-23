from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vi.detect import Detection
from vi.schemas import Box, CamTime, Modality, Tube, TubeState


@dataclass
class _Track:
    tube: Tube
    misses: int = 0
    history: list[Box] = field(default_factory=list)


class SimpleIoUTracker:
    """Slice implementation of Ring 2: greedy IoU association with an explicit
    lifecycle. Replaced by BoT-SORT-from-source in week 2; the lifecycle rules stay.

    E-TUBE-01: ambiguous association (two candidates above threshold within `ambig_margin`)
               keeps the best match and records the other as a merge_candidate instead of
               guessing.
    E-TUBE-02: a track with no detection becomes `occluded`; after max_occluded_ms it is
               `lost` (or `exited` if its last box touched an exit region).
    E-TUBE-03: detections flagged det_source='heartbeat' keep a stationary track alive.
    E-TUBE-05: this class is per-camera; cross-camera merging is fusion's job, never here.
    """

    def __init__(self, camera_id: str, iou_thr: float = 0.3, max_occluded_ms: int = 3000,
                 ambig_margin: float = 0.1, exit_boxes: list[Box] | None = None,
                 modality: Modality = Modality.rgb, offset_ms: int = 0,
                 keyframe_sink: Callable[[str, int, Box], str] | None = None):
        self.camera_id = camera_id
        self.iou_thr = iou_thr
        self.max_occluded_ms = max_occluded_ms
        self.ambig_margin = ambig_margin
        self.exit_boxes = exit_boxes or []
        self.modality = modality
        self.offset_ms = offset_ms
        self.keyframe_sink = keyframe_sink   # E-FOV-06: persist a native-res crop ref at birth
        self._tracks: dict[str, _Track] = {}
        self._seq = 0

    def _t(self, t_ms: int) -> CamTime:
        return CamTime(cam_utc_ms=t_ms, offset_ms=self.offset_ms)

    def _new_id(self, t_ms: int) -> str:
        self._seq += 1
        return f"{self.camera_id}:{t_ms}:{self._seq}"

    def _touches_exit(self, box: Box) -> bool:
        return any(box.iou(e) > 0.0 for e in self.exit_boxes)

    def update(self, detections: list[Detection], t_ms: int,
               det_source: str = "detector") -> tuple[list[Tube], list[Tube]]:
        live_ids = list(self._tracks.keys())
        # IoU matrix
        pairs: list[tuple[float, int, int]] = []
        for ti, tid in enumerate(live_ids):
            tb = self._tracks[tid].tube.box
            for di, d in enumerate(detections):
                iou = tb.iou(d.box)
                if iou >= self.iou_thr:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        matched: dict[int, int] = {}
        for iou, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matched[ti] = di
            used_t.add(ti)
            used_d.add(di)
        # E-TUBE-01: record ambiguity
        for iou, ti, di in pairs:
            if ti in matched and matched[ti] != di and di in used_d:
                best = self._tracks[live_ids[ti]].tube.box.iou(detections[matched[ti]].box)
                if best - iou < self.ambig_margin:
                    other_tid = next((live_ids[t2] for t2, d2 in matched.items() if d2 == di), None)
                    if other_tid:
                        tube = self._tracks[live_ids[ti]].tube
                        if other_tid not in tube.merge_candidates:
                            tube.merge_candidates.append(other_tid)

        closed: list[Tube] = []
        # matched -> active
        for ti, di in matched.items():
            tr = self._tracks[live_ids[ti]]
            d = detections[di]
            tr.tube.box = d.box
            tr.tube.last_seen = self._t(t_ms)
            tr.tube.state = TubeState.active
            tr.tube.occluded_since_ms = None
            tr.misses = 0
            tr.history.append(d.box)
        # unmatched tracks -> occluded / lost / exited
        for ti, tid in enumerate(live_ids):
            if ti in matched:
                continue
            tr = self._tracks[tid]
            tr.misses += 1
            if tr.tube.occluded_since_ms is None:
                tr.tube.occluded_since_ms = t_ms
                tr.tube.state = TubeState.occluded
            elif t_ms - tr.tube.occluded_since_ms > self.max_occluded_ms:
                tr.tube.state = TubeState.exited if self._touches_exit(tr.tube.box) else TubeState.lost
                closed.append(tr.tube)
                del self._tracks[tid]
        # unmatched detections -> born
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            tid = self._new_id(t_ms)
            tube = Tube(tube_id=tid, camera_id=self.camera_id, class_label=d.class_label,
                        state=TubeState.born, born=self._t(t_ms), last_seen=self._t(t_ms),
                        box=d.box, modality=self.modality)
            if self.keyframe_sink is not None:
                tube.keyframe_refs.append(self.keyframe_sink(self.camera_id, t_ms, d.box))
            self._tracks[tid] = _Track(tube=tube, history=[d.box])
        return [tr.tube for tr in self._tracks.values()], closed
