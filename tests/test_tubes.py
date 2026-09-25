import numpy as np
import pytest

from vi.detect import Detection
from vi.schemas import TubeState
from vi.tubes import ByteTracker, SimpleIoUTracker

TRACKERS = [
    pytest.param(lambda **kw: SimpleIoUTracker(iou_thr=0.2, **kw), id="simple"),
    pytest.param(lambda **kw: ByteTracker(confirm_ticks=1, **kw), id="byte"),
]
from vi.tubes.geometry import FOOT_UNCERTAINTY_M, OCCLUDED_FEET_UNCERTAINTY_M, project_foot

from conftest import box


def det(x1, y1, x2, y2, label="person"):
    return Detection(box=box(x1, y1, x2, y2), class_label=label, confidence=0.9)


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-TUBE-02")
def test_occlusion_then_lost_or_exited(make):
    tr = make(camera_id="c1", max_occluded_ms=1000, exit_boxes=[box(300, 0, 320, 240)])
    tr.update([det(10, 10, 50, 100)], 0)
    live, closed = tr.update([], 500)
    assert live[0].state == TubeState.occluded and live[0].occluded_since_ms == 500 and not closed
    live, closed = tr.update([], 1600)
    assert live == [] and closed[0].state == TubeState.lost
    # same story but the last box touched an exit region
    tr2 = make(camera_id="c1", max_occluded_ms=1000, exit_boxes=[box(300, 0, 320, 240)])
    tr2.update([det(290, 10, 320, 100)], 0)
    tr2.update([], 500)
    _, closed = tr2.update([], 1600)
    assert closed[0].state == TubeState.exited


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-TUBE-01")
def test_crossing_paths_flag_merge_candidates_instead_of_guessing(make):
    tr = make(camera_id="c1", ambig_margin=0.2)
    tr.update([det(0, 0, 40, 100), det(30, 0, 70, 100)], 0)
    # two people shoulder to shoulder: each detection overlaps both tracks with similar IoU
    live, _ = tr.update([det(10, 0, 50, 100), det(15, 0, 55, 100)], 100)
    assert len(live) == 2                              # no new ids invented
    assert any(t.merge_candidates for t in live)       # ambiguity recorded, not resolved


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-TUBE-03")
@pytest.mark.edge("E-GATE-01")
def test_heartbeat_detection_keeps_stationary_track_active(make):
    tr = make(camera_id="c1", max_occluded_ms=1000)
    tr.update([det(10, 10, 50, 100)], 0)
    for i in range(1, 30):
        live, closed = tr.update([det(10, 10, 50, 100)], i * 1000, det_source="heartbeat")
    assert live[0].state == TubeState.active and closed == []


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-FOV-06")
def test_keyframe_ref_persisted_at_birth(make):
    refs = []
    def sink(cam, t_ms, b):
        ref = f"r2://site/{cam}/{t_ms}.jpg"
        refs.append(ref)
        return ref
    tr = make(camera_id="c1", keyframe_sink=sink)
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


@pytest.mark.edge("E-TUBE-13")
def test_bytetracker_survives_sampled_frame_rate_where_iou_only_fragments():
    import sys
    sys.path.insert(0, "bench")
    from ring2_tubes import synthetic_gt
    from vi.eval import evaluate_mot

    gt = {f: v for f, v in synthetic_gt().items() if f % 5 == 0}          # 6 fps effective
    def run(tr):
        pred, ids = {}, {}
        for f in sorted(gt):
            live, _ = tr.update([det(b.x1, b.y1, b.x2, b.y2) for _, b in gt[f]], int(f * 1000 / 30))
            pred[f] = [(ids.setdefault(t.tube_id, len(ids) + 1), t.box) for t in live if t.state == TubeState.active]
        return evaluate_mot(gt, pred)
    simple = run(SimpleIoUTracker("c1", iou_thr=0.3, max_occluded_ms=2000))
    byte = run(ByteTracker("c1", max_occluded_ms=2000))
    assert simple.fragmentation_ratio > 3 and simple.idsw > 10        # the measured baseline
    assert byte.fragmentation_ratio == 1.0 and byte.idsw == 0 and byte.idf1 > 0.95


def test_kalman_predicts_constant_velocity_in_seconds():
    from vi.tubes import KalmanBoxFilter
    kf = KalmanBoxFilter(box(0, 0, 40, 120))
    for i in range(1, 6):                       # 100 px/s to the right, sampled irregularly
        kf.predict(0.1 if i % 2 else 0.3)
        t = 0.1 * ((i + 1) // 2) + 0.3 * (i // 2)
        kf.update(box(100 * t, 0, 100 * t + 40, 120))
    assert abs(kf.speed_px_s - 100) < 15
    predicted = kf.predict(1.0)
    assert abs(predicted.x1 - (kf.x[0] - 20)) < 1e-6 and 150 < predicted.x1 < 260
