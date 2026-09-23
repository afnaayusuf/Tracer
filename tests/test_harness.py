from vi.harness.coverage import main as coverage_main
from vi.harness.registry import load_cases, scan_tests


def test_registry_loads_and_ids_are_unique():
    cases = load_cases()
    assert len(cases) > 50
    assert len({c.id for c in cases}) == len(cases)


def test_every_implemented_case_has_a_test():
    assert coverage_main() == 0


def test_no_test_references_unknown_case():
    known = {c.id for c in load_cases()}
    assert set(scan_tests()) <= known
