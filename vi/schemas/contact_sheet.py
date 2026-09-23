from __future__ import annotations

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from .common import Modality
from .tube import Attributes


class CellResult(BaseModel):
    cell_id: int = Field(ge=0)
    tube_id: str
    attributes: Attributes


class ContactSheetResult(BaseModel):
    """One writer-VLM call over a packed sheet. E-FOV-04: cell ids unique and count
    fixed; a missing or duplicated cell is a schema failure, not a silent skip."""

    sheet_id: str
    modality: Modality
    expected_cells: int = Field(ge=1, le=16)
    cells: list[CellResult]

    @model_validator(mode="after")
    def _cells_ok(self) -> "ContactSheetResult":
        ids = [c.cell_id for c in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate cell_id in contact sheet result")
        if len(ids) != self.expected_cells:
            raise ValueError(f"expected {self.expected_cells} cells, got {len(ids)}")
        for c in self.cells:
            if c.attributes.modality != self.modality:
                raise ValueError("cell modality must match sheet modality")
        return self



CellsAdapter: TypeAdapter = TypeAdapter(list[CellResult])


def parse_or_fail(text: str, sheet_id: str, modality: Modality, expected_cells: int) -> ContactSheetResult | None:
    """E-FOV-07: validate a writer-VLM response (either a full ContactSheetResult object
    or a bare JSON array of cells); return None on any violation so the caller emits
    EnrichmentPatch(failed=True) per cell and the tick is never blocked."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
            cleaned = cleaned.strip()
        if cleaned.startswith("["):
            cells = CellsAdapter.validate_json(cleaned)
            return ContactSheetResult(sheet_id=sheet_id, modality=modality,
                                      expected_cells=expected_cells, cells=cells)
        result = ContactSheetResult.model_validate_json(cleaned)
        if result.expected_cells != expected_cells or result.modality != modality:
            return None
        return result
    except Exception:
        return None
