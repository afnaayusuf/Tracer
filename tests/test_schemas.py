import pytest
from pydantic import ValidationError

from vi.schemas import (Attributes, CamTime, Color, ContactSheetResult, CellResult, EnrichmentPatch,
                        Modality, Provenance, Tick, Tone, Tube, TubeState, effective_confidence, parse_or_fail)
from vi.schemas.tube import ENHANCED_DISCOUNT

from conftest import box


@pytest.mark.edge("E-ING-01")
def test_camtime_offset_aligns_two_cameras():
    kitchen = CamTime(cam_utc_ms=10_000, offset_ms=-2_000, offset_confidence=0.9)
    hallway = CamTime(cam_utc_ms=8_000, offset_ms=0, offset_confidence=0.9)
    assert kitchen.corrected_ms() == hallway.corrected_ms()


@pytest.mark.edge("E-ING-03")
def test_tick_is_wall_clock_window_not_frame_index(prov, t):
    # a tick with an irregular duration is valid; nothing in the schema references frame numbers
    tick = Tick(camera_id="c1", tick_index=0, t_start=t(0), t_end=t(730), provenance=prov)
    assert tick.t_end.corrected_ms() - tick.t_start.corrected_ms() == 730
    assert "frame" not in " ".join(Tick.model_fields)


@pytest.mark.edge("E-ING-06")
def test_timestamps_are_utc_ms_only():
    ct = CamTime(cam_utc_ms=1_700_000_000_000)
    assert isinstance(ct.cam_utc_ms, int)
    assert "tz" not in " ".join(CamTime.model_fields) and "local" not in " ".join(CamTime.model_fields)


@pytest.mark.edge("E-FOV-05")
def test_ir_forbids_color_attributes():
    with pytest.raises(ValidationError):
        Attributes(modality=Modality.ir, top_color=Color.red, color_reason="ir_mode")
    with pytest.raises(ValidationError):
        Attributes(modality=Modality.ir)  # missing color_reason
    a = Attributes(modality=Modality.ir, color_reason="ir_mode", top_tone=Tone.light, confidence=0.7)
    assert a.top_color is None and a.top_tone == Tone.light
    # rgb keeps colors
    assert Attributes(modality=Modality.rgb, top_color=Color.red).top_color == Color.red


@pytest.mark.edge("E-FOV-01")
def test_enhanced_attributes_are_discounted():
    plain = Attributes(modality=Modality.rgb, confidence=0.9)
    enhanced = Attributes(modality=Modality.rgb, confidence=0.9, enhanced=True)
    assert effective_confidence(plain) == 0.9
    assert effective_confidence(enhanced) == pytest.approx(0.9 * ENHANCED_DISCOUNT)


def _cell(i, tube="tb", mod=Modality.rgb):
    return CellResult(cell_id=i, tube_id=f"{tube}{i}", attributes=Attributes(modality=mod, confidence=0.5))


@pytest.mark.edge("E-FOV-04")
def test_contact_sheet_rejects_duplicate_missing_or_mismatched_cells():
    with pytest.raises(ValidationError):
        ContactSheetResult(sheet_id="s", modality=Modality.rgb, expected_cells=2, cells=[_cell(0), _cell(0)])
    with pytest.raises(ValidationError):
        ContactSheetResult(sheet_id="s", modality=Modality.rgb, expected_cells=3, cells=[_cell(0), _cell(1)])
    with pytest.raises(ValidationError):
        ContactSheetResult(sheet_id="s", modality=Modality.rgb, expected_cells=1,
                           cells=[CellResult(cell_id=0, tube_id="x",
                                             attributes=Attributes(modality=Modality.ir, color_reason="ir_mode"))])
    ok = ContactSheetResult(sheet_id="s", modality=Modality.rgb, expected_cells=2, cells=[_cell(0), _cell(1)])
    assert len(ok.cells) == 2


@pytest.mark.edge("E-FOV-07")
def test_bad_vlm_output_returns_none_and_failed_patch_is_valid():
    assert parse_or_fail("I think the person is wearing red", "s", Modality.rgb, 2) is None
    assert parse_or_fail('[{"cell_id":0,"tube_id":"a","attributes":{"modality":"rgb"}}]', "s", Modality.rgb, 2) is None
    good = '```json\n[{"cell_id":0,"tube_id":"a","attributes":{"modality":"rgb","confidence":0.6}}]\n```'
    assert parse_or_fail(good, "s", Modality.rgb, 1) is not None
    p = EnrichmentPatch(patch_id="p1", tube_id="a", produced_at_ms=1, source="vlm", payload={}, failed=True)
    assert p.failed and p.confidence == 0.0


@pytest.mark.edge("E-STO-04")
@pytest.mark.edge("E-STO-05")
def test_provenance_required_and_unknown_fields_tolerated(prov, t):
    tick = Tick(camera_id="c1", tick_index=1, t_start=t(0), t_end=t(500), provenance=prov)
    d = tick.model_dump()
    assert d["provenance"]["schema_version"] and d["provenance"]["kb_version"] == 3
    d["future_field"] = {"x": 1}          # a newer writer added a field
    assert Tick.model_validate(d).tick_index == 1
    with pytest.raises(ValidationError):
        Tick(camera_id="c1", tick_index=1, t_start=t(0), t_end=t(500))  # no provenance


@pytest.mark.edge("E-FOV-06")
def test_tube_carries_keyframe_refs(t):
    tube = Tube(tube_id="c1:0:1", camera_id="c1", class_label="person", born=t(0), last_seen=t(0),
                box=box(0, 0, 10, 20), keyframe_refs=["r2://site/c1/0/1.jpg"], state=TubeState.born)
    assert tube.keyframe_refs[0].startswith("r2://")
