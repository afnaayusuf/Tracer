# vi-engine

Tube-centric video intelligence: Block 1 turns camera streams into episode files (tubes, events,
attributes); Block 2 answers questions over them with evidence. No training anywhere.

Start here, in this order: `SPEC.md` (what we are building), `PLAN.md` (30 days, solo),
`EDGE_CASES.md` (what must not be forgotten).

## Quickstart (CPU, no models)

```bash
pip install -e ".[dev]"
make check                 # tests + edge-case coverage + schema export + registry doc
python bench/slice_cpu.py  # gate -> tubes -> events -> episode JSONL on a synthetic scene
```

Working from Colab only? See `colab/README.md`: one env cell, one bootstrap cell, one session cell.

## Layout

```
vi/schemas/    the contract: Tick, Tube, Event, Episode, SceneCard, Fact, ContactSheetResult
vi/gate/       Ring 0  — Gate protocol; FrameDiffGate (slice + codec fallback); MVGate (week 3)
vi/detect/     Ring 1  — Detector protocol; RFDETRDetector (lazy import)
vi/tubes/      Ring 2  — Tracker protocol; SimpleIoUTracker (slice); geometry.project_foot
vi/events/     Ring 3a — Zone, EventCompiler (deterministic predicates, heartbeats, scene state)
vi/episode/    EpisodeWriter (append-only JSONL, idempotent), should_soft_cut
vi/ingest/     VideoReader (PyAV, file/RTSP), PTSFilter, SizeGuard, gate_mode_for_codec, synthetic clips
vi/harness/    edge-case registry loader, coverage checker, EDGE_CASES.md renderer
bench/         one script per ring; each prints one benchmark-table row
tests/         every implemented edge case has a test marked @pytest.mark.edge("E-…")
schemas/       exported JSON Schema (feeds xgrammar for constrained decoding)
```

## The harness rule

`edge_cases.yaml` is the source of truth. A case marked `implemented` without a test fails the
build. A test referencing an unknown case fails the build. Add the case first, then the code,
then the test.
