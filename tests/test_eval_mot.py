import pytest

from vi.eval import evaluate_mot
from vi.schemas import Box


def b(x):
    return Box(x1=x, y1=0, x2=x + 40, y2=100)


def test_perfect_tracking_scores_one():
    gt = {f: [(1, b(f * 5)), (2, b(300 - f * 5))] for f in range(20)}
    r = evaluate_mot(gt, gt)
    assert r.mota == 1.0 and r.idf1 == 1.0 and r.idsw == 0 and r.fragmentation_ratio == 1.0


def test_id_switch_is_counted_and_lowers_idf1():
    gt = {f: [(1, b(f * 5))] for f in range(20)}
    pred = {f: [((7 if f < 10 else 8), b(f * 5))] for f in range(20)}
    r = evaluate_mot(gt, pred)
    assert r.idsw == 1 and r.mota == pytest.approx(1 - 1 / 20) and r.idf1 == 0.5
    assert r.fragmentation_ratio == 2.0


def test_misses_and_false_positives():
    gt = {f: [(1, b(f * 5))] for f in range(10)}
    pred = {f: ([(1, b(f * 5))] if f % 2 == 0 else []) for f in range(10)}
    pred[3] = [(9, b(900))]
    r = evaluate_mot(gt, pred)
    assert r.fn == 5 and r.fp == 1 and r.tp == 5


@pytest.mark.edge("E-DET-09")
def test_default_zones_and_tube_classes():
    from vi.detect import is_tube_class
    from vi.events import default_zones
    z = default_zones("c1", 1280, 720)
    assert [x.zone_id for x in z] == ["exit_left", "exit_right", "floor_centre"]
    assert z[0].contains((10, 300)) and z[1].contains((1270, 300)) and z[2].contains((640, 600))
    assert is_tube_class("person") and is_tube_class("suitcase")
    assert not is_tube_class("dining table") and not is_tube_class("tv")
