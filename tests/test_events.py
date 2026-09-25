import numpy as np
import pytest

from vi.events import EventCompiler, Zone
from vi.gate.base import GateResult
from vi.schemas import EventType, TubeSnapshot, TubeState

from conftest import box


def zone(zid, cam="c1", kind="generic", asset=None, poly=((100, 100), (200, 100), (200, 200), (100, 200))):
    return Zone(zone_id=zid, camera_id=cam, polygon=list(poly), kind=kind, asset_id=asset)


def snap(tid, x1, y1, x2, y2, label="person"):
    return TubeSnapshot(tube_id=tid, class_label=label, state=TubeState.active, box=box(x1, y1, x2, y2))


def inside(tid):   # foot point (150, 150)
    return snap(tid, 140, 100, 160, 150)


def outside(tid):  # foot point (150, 250)
    return snap(tid, 140, 200, 160, 250)


@pytest.mark.edge("E-EVT-01")
def test_zone_hysteresis_filters_boundary_jitter():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=2, exit_ticks=2, open_grace_ms=0)
    ec.on_tick([], -500)                        # episode opens on an empty tile
    evs = []
    for i in range(6):   # jitter: in, out, in, out, ...
        evs += ec.on_tick([inside("a") if i % 2 == 0 else outside("a")], i * 500)
    assert evs == []
    evs = ec.on_tick([inside("a")], 3000) + ec.on_tick([inside("a")], 3500)
    assert [e.type for e in evs] == [EventType.enter_zone]
    evs = ec.on_tick([outside("a")], 4000) + ec.on_tick([outside("a")], 4500)
    assert [e.type for e in evs] == [EventType.exit_zone]


@pytest.mark.edge("E-EVT-03")
def test_pickup_requires_absence_with_nobody_in_zone():
    ec = EventCompiler("c1", [zone("shelf", kind="asset_home", asset="bike_keys")], enter_ticks=1, exit_ticks=1,
                       open_grace_ms=0)
    ec.on_tick([], 0)
    assert ec.on_heartbeat("shelf", True, 0, persons_inside=[]) == []
    ec.on_tick([inside("emma")], 1000)
    # shelf hidden behind Emma: heartbeat says absent while she is inside -> no event
    assert ec.on_heartbeat("shelf", False, 1500, persons_inside=["emma"]) == []
    ec.on_tick([outside("emma")], 2000)
    evs = ec.on_heartbeat("shelf", False, 2500, persons_inside=[])
    assert len(evs) == 1 and evs[0].type == EventType.pickup
    assert evs[0].subject_tube_ids == ["emma"] and evs[0].object_ids == ["bike_keys"]
    # a second absent heartbeat must not fire again
    assert ec.on_heartbeat("shelf", False, 3000, persons_inside=[]) == []


@pytest.mark.edge("E-EVT-03")
def test_missing_asset_with_no_visitor_does_not_blame_anyone():
    ec = EventCompiler("c1", [zone("shelf", kind="asset_home", asset="bike_keys")])
    ec.on_heartbeat("shelf", True, 0, persons_inside=[])
    evs = ec.on_heartbeat("shelf", False, 10_000, persons_inside=[])
    assert evs[0].type == EventType.asset_missing_from_home and evs[0].subject_tube_ids == []


@pytest.mark.edge("E-EVT-06")
def test_overlapping_cameras_share_dedupe_key_but_not_event_id():
    a = EventCompiler("camA", [zone("door", cam="camA")], enter_ticks=1, open_grace_ms=0)
    b = EventCompiler("camB", [zone("door", cam="camB")], enter_ticks=1, open_grace_ms=0)
    a.on_tick([], 0)
    b.on_tick([], 0)
    ea = a.on_tick([inside("tA")], 1000)[0]
    eb = b.on_tick([inside("tB")], 1300)[0]
    assert ea.dedupe_key == eb.dedupe_key and ea.event_id != eb.event_id


@pytest.mark.edge("E-GATE-02")
@pytest.mark.edge("E-KB-01")
def test_gate_results_become_scene_state_events():
    ec = EventCompiler("c1", [], moved_after_updates=3)
    evs = ec.on_gate(GateResult(camera_id="c1", t_ms=0, blobs=[], luma_step=True, mean_luma=180))
    assert [e.type for e in evs] == [EventType.illumination_change]
    moved = []
    for i in range(5):
        moved += ec.on_gate(GateResult(camera_id="c1", t_ms=i * 100, blobs=[], global_motion=True))
    assert [e.type for e in moved] == [EventType.camera_moved_suspect]   # fires once


def test_dwell_fires_once():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=1, dwell_ms=2000, open_grace_ms=0)
    ec.on_tick([], -500)
    evs = []
    for i in range(6):
        evs += ec.on_tick([inside("a")], i * 1000)
    assert [e.type for e in evs] == [EventType.enter_zone, EventType.dwell]


@pytest.mark.edge("E-EVT-10")
def test_tubes_present_at_episode_open_do_not_enter():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=1, exit_ticks=1, open_grace_ms=800)
    assert ec.on_tick([], 0) == []                                         # gate warming up: empty tick
    assert ec.on_tick([inside("a"), inside("b")], 333) == []               # discovered inside grace: no enter
    assert ec.on_tick([inside("a"), inside("b")], 666) == []
    evs = ec.on_tick([outside("a"), inside("b"), inside("c")], 1000)      # a leaves, c arrives after grace
    assert sorted(e.type for e in evs) == sorted([EventType.exit_zone, EventType.enter_zone])
    assert [e.subject_tube_ids for e in evs if e.type == EventType.enter_zone] == [["c"]]
