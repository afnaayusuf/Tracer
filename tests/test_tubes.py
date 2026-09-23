import numpy as np
import pytest

from vi.detect import Detection
from vi.schemas import TubeState
from vi.tubes import SimpleIoUTracker
from vi.tubes.geometry import FOOT_UNCERTAINTY_M, OCCLUDED_FEET_UNCERTAINTY_M, project_foot

from conftest import box


def det(x1, y1, x2, y2, label="person"):
    return Detection(box=box(x1, y1, x2, y2), class_label=label, confidence=0.9)


@pytest.mark.edge("E-TUBE-02")
def test_occlusion_then_lost_or_exited():
    tr = SimpleIoUTracker("c1", max_occluded_ms=1000, exit_boxes=[box(300, 0, 320, 240)])
    tr.update([det(10, 10, 50, 100)], 0)
    live, closed = tr.update([], 500)
    assert live[0].state == TubeState.occluded and live[0].occluded_since_ms == 500 and not closed
    live, closed = tr.update([], 1600)
    assert live == [] and closed[0].state == TubeState.lost
    # same story but the last box touched an exit region
    tr2 = SimpleIoUTracker("c1", max_occluded_ms=1000, exit_boxes=[box(300, 0, 320, 240)])
    tr2.update([det(290, 10, 320, 100)], 0)
    tr2.update([], 500)
    _, closed = tr2.update([], 1600)
    assert closed[0].state == TubeState.exited


@pytest.mark.edge("E-TUBE-01")
def test_crossing_paths_flag_merge_candidates_instead_of_guessing():
    tr = SimpleIoUTracker("c1", iou_thr=0.2, ambig_margin=0.2)
    tr.update([det(0, 0, 40, 100), det(30, 0, 70, 100)], 0)
    # two people shoulder to shoulder: each detection overlaps both tracks with similar IoU
    live, _ = tr.update([det(10, 0, 50, 100), det(15, 0, 55, 100)], 100)
    assert len(live) == 2                              # no new ids invented
    assert any(t.merge_candidates for t in live)       # ambiguity recorded, not resolved


@pytest.mark.edge("E-TUBE-03")
@pytest.mark.edge("E-GATE-01")
def test_heartbeat_detection_keeps_stationary_track_active():
    tr = SimpleIoUTracker("c1", max_occluded_ms=1000)
    tr.update([det(10, 10, 50, 100)], 0)
    for i in range(1, 30):
        live, closed = tr.update([det(10, 10, 50, 100)], i * 1000, det_source="heartbeat")
    assert live[0].state == TubeState.active and closed == []


@pytest.mark.edge("E-FOV-06")
def test_keyframe_ref_persisted_at_birth():
    refs = []
    def sink(cam, t_ms, b):
        ref = f"r2://site/{cam}/{t_ms}.jpg"
        refs.append(ref)
        return ref
    tr = SimpleIoUTracker("c1", keyframe_sink=sink)
    live, _ = tr.update([det(10, 10, 50, 100)], 42)
    assert live[0].keyframe_refs == ["r2://site/c1/42.jpg"] and refs == live[0].keyframe_refs


@pytest.mark.edge("E-TUBE-12")
def test_occluded_feet_widen_uncertainty_and_tag_source():
    H = np.eye(3) * np.array([0.01, 0.01, 1.0])[:, None]  # trivial scale homography
    b = box(100, 50, 140, 200)
    ok = project_foot(b, H, feet_visible=True)
    bad = project_foot(b, H, feet_visible=False)
    assert ok.source == "homography" and ok.uncertainty_m == FOOT_UNCERTAINTY_M
    assert bad.source == "fallback_bbox_bottom" and bad.uncertainty_m == OCCLUDED_FEET_UNCERTAINTY_M
    assert (ok.x_m, ok.y_m) == (bad.x_m, bad.y_m)
    assert project_foot(b, None) is None
