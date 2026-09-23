.PHONY: check test coverage schemas edge-doc
check: test coverage schemas edge-doc
slice:
	python bench/slice_cpu.py
test:
	python -m pytest -q
coverage:
	python -m vi.harness.coverage
schemas:
	python -m vi.schemas.export
edge-doc:
	python -m vi.harness.render_edge_cases
