from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    hypothesis = "hypothesis"
    fact = "fact"
    retired = "retired"


PROMOTE_AT_SUPPORT = 2
RETIRE_AT_CONTRADICTIONS = 2


class Fact(BaseModel):
    """KB edge with provenance. Promotion and retirement are the only state changes
    and both are explicit methods so they can be tested (E-KB-03)."""

    fact_id: str
    subject: str
    predicate: str
    object: list[str]
    status: FactStatus = FactStatus.hypothesis
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: Literal["calibration", "walk", "observed", "user_confirmed", "inferred"]
    support: int = Field(0, ge=0)
    contradictions: int = Field(0, ge=0)
    evidence: list[str] = Field(default_factory=list)
    first_seen_ms: int
    last_confirmed_ms: int | None = None
    version: int = Field(1, ge=1)

    def add_support(self, evidence_ref: str, t_ms: int) -> "Fact":
        self.support += 1
        self.evidence.append(evidence_ref)
        self.last_confirmed_ms = t_ms
        self.version += 1
        if self.status == FactStatus.hypothesis and (
            self.support >= PROMOTE_AT_SUPPORT or self.source == "user_confirmed"
        ):
            self.status = FactStatus.fact
        return self

    def add_contradiction(self, evidence_ref: str) -> "Fact":
        self.contradictions += 1
        self.evidence.append(evidence_ref)
        self.version += 1
        if self.contradictions >= RETIRE_AT_CONTRADICTIONS and self.source != "user_confirmed":
            self.status = FactStatus.retired
        return self

    def user_confirm(self, t_ms: int) -> "Fact":
        self.source = "user_confirmed"
        self.status = FactStatus.fact
        self.last_confirmed_ms = t_ms
        self.version += 1
        return self
