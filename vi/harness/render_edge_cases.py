"""Render EDGE_CASES.md from edge_cases.yaml + test references. Run: python -m vi.harness.render_edge_cases"""
from __future__ import annotations

from collections import defaultdict

from .registry import ROOT, load_cases, scan_tests

RING_ORDER = ["ingest", "ring0", "ring1", "ring2", "ring3a", "ring3b", "storage", "agent", "kb", "night", "privacy"]
RING_TITLE = {"ingest": "Ingest and time", "ring0": "Ring 0 · bitstream / motion gate",
              "ring1": "Ring 1 · selective decode + detect", "ring2": "Ring 2 · tubes and fusion",
              "ring3a": "Ring 3a · event compiler", "ring3b": "Ring 3b · foveation and enrichment",
              "storage": "Storage and handoff", "agent": "Block 2 · agent", "kb": "Knowledge base and calibration",
              "night": "Night modality", "privacy": "Privacy"}


def main() -> None:
    cases = load_cases()
    refs = scan_tests()
    groups: dict[str, list] = defaultdict(list)
    for c in cases:
        groups[c.ring].append(c)
    n_impl = sum(c.status == "implemented" for c in cases)
    lines = ["# Edge-case registry", "",
             f"Generated from `edge_cases.yaml` by `make edge-doc`. {len(cases)} cases, {n_impl} implemented and tested.",
             "", "Status: **implemented** = code path exists and a test marked `@pytest.mark.edge(\"ID\")` exercises it; "
             "**planned** = month-1 scope; **deferred** = tracked, not month 1.", ""]
    for ring in RING_ORDER + [r for r in groups if r not in RING_ORDER]:
        if ring not in groups:
            continue
        lines += [f"## {RING_TITLE.get(ring, ring)}", "", "| ID | Case | Trigger | Handling | Status | Tests |", "|---|---|---|---|---|---|"]
        for c in groups[ring]:
            tests = "<br>".join(refs.get(c.id, [])) or "—"
            lines.append(f"| {c.id} | {c.title} | {c.trigger} | {c.handling} | {c.status} | {tests} |")
        lines.append("")
    (ROOT / "EDGE_CASES.md").write_text("\n".join(lines))
    print(f"wrote EDGE_CASES.md ({len(cases)} cases)")


if __name__ == "__main__":
    main()
