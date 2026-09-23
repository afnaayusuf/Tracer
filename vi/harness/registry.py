from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "edge_cases.yaml"
TESTS = ROOT / "tests"
EDGE_RE = re.compile(r'edge\(\s*["\'](E-[A-Z]+-\d+)["\']\s*\)')
STATUSES = ("implemented", "planned", "deferred")


@dataclass
class Case:
    id: str
    ring: str
    title: str
    trigger: str
    handling: str
    status: str


def load_cases(path: Path = REGISTRY) -> list[Case]:
    data = yaml.safe_load(path.read_text())
    cases = [Case(**c) for c in data["cases"]]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate edge-case ids: {dup}")
    bad = [c.id for c in cases if c.status not in STATUSES]
    if bad:
        raise ValueError(f"invalid status on: {bad}")
    return cases


def scan_tests(tests_dir: Path = TESTS) -> dict[str, list[str]]:
    """edge-case id -> list of 'file::test_name' references."""
    refs: dict[str, list[str]] = {}
    for f in sorted(tests_dir.glob("test_*.py")):
        pending: list[str] = []          # markers precede the def line they decorate
        for line in f.read_text().splitlines():
            m_def = re.match(r"\s*def (test_\w+)", line)
            if m_def:
                for cid in pending:
                    refs.setdefault(cid, []).append(f"{f.name}::{m_def.group(1)}")
                pending = []
                continue
            pending += [m.group(1) for m in EDGE_RE.finditer(line)]
    return refs
