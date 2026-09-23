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
