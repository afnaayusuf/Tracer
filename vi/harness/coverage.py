"""Fail the build when an edge case marked `implemented` has no test, or a test
references an id that is not in the registry. Run: python -m vi.harness.coverage"""
from __future__ import annotations

import sys

from .registry import load_cases, scan_tests


def main() -> int:
    cases = load_cases()
    refs = scan_tests()
    by_id = {c.id: c for c in cases}
    errors: list[str] = []
    for cid in refs:
        if cid not in by_id:
            errors.append(f"test references unknown edge case {cid} ({refs[cid]})")
    impl = [c for c in cases if c.status == "implemented"]
    for c in impl:
        if c.id not in refs:
            errors.append(f"{c.id} is 'implemented' but has no test")
    planned = [c for c in cases if c.status == "planned"]
    deferred = [c for c in cases if c.status == "deferred"]
    covered = sum(1 for c in impl if c.id in refs)
    print(f"edge cases: {len(cases)} total | implemented {len(impl)} (tested {covered}) | "
          f"planned {len(planned)} | deferred {len(deferred)}")
    tested_planned = [c.id for c in planned if c.id in refs]
    if tested_planned:
        print(f"note: planned cases already carrying tests (promote to implemented): {tested_planned}")
    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        return 1
    print("coverage OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
