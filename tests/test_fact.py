import pytest

from vi.schemas import Fact, FactStatus


def _fact(source="observed"):
    return Fact(fact_id="f1", subject="actuator:switch_panel@room3", predicate="controls",
                object=["light:dining", "light:porch"], source=source, first_seen_ms=0, confidence=0.6)


@pytest.mark.edge("E-KB-03")
def test_facts_promote_on_support_and_retire_on_contradiction():
    f = _fact()
    f.add_support("episode:1", 1000)
    assert f.status == FactStatus.hypothesis
    f.add_support("episode:2", 2000)
    assert f.status == FactStatus.fact and f.version == 3
    f.add_contradiction("episode:3")
    assert f.status == FactStatus.fact
    f.add_contradiction("episode:4")
    assert f.status == FactStatus.retired


@pytest.mark.edge("E-KB-03")
def test_user_confirmed_facts_survive_contradictions():
    f = _fact()
    f.user_confirm(500)
    assert f.status == FactStatus.fact
    f.add_contradiction("e1")
    f.add_contradiction("e2")
    assert f.status == FactStatus.fact   # a human said so; a second switch is the likelier story
