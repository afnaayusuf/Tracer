# 30-day solo build plan

One engineer, one month, Colab Pro+ for GPU, laptop for everything else. The goal is not the whole
system. The goal is: **the slice runs on real footage end to end, the two scripted scenarios replay
against WILDTRACK + your own cameras, and the benchmark table has a measured row for every ring.**
Everything else is a swap-in later.

## Rules that keep you honest

1. `make check` green before every commit. It runs tests, edge-case coverage, schema export and
   the registry doc. Never commit red.
2. New edge case → `edge_cases.yaml` first, code second, test third. Never the other order.
3. A ring is "done" when its bench script prints a row and the row is pasted into the table below.
4. Slice before depth: every ring gets its real implementation only after the stubbed slice runs
   on the dataset you are about to benchmark.
5. Colab is a batch runner (`colab run --gpu … bench/ring_N.py`), not an IDE. Code lives in git.
   Data lives in R2. Bench scripts checkpoint rows to `data/bench/*.jsonl` as they go.

## Explicitly cut from month 1 (tracked as `deferred` in the registry)

Edge gateway build · learned SR / burst SR (classical only) · SAM 3 inspect lane (stub returns
"not available") · OCR and pose specialists (schema slots only) · fact-miner automation (manual
seeding through the walks) · naming UI (CLI) · fall/crowd/loiter events · fisheye undistort ·
blind-spot probabilistic delay states.

## Week 1 — contract and slice on real footage

| Day | Do | Gate |
|---|---|---|
| 1 | Clone, `make check`, run `bench/slice_cpu.py`. Read SPEC.md once, end to end. Create R2 bucket; push WILDTRACK, VIRAT (2 h subset), MOT17-half. Order 2–3 RTSP cameras. | slice runs; data in R2 |
| 2 | RTSP/file ingest: PyAV reader → gray frames + PTS dedupe (E-ING-07) → `FrameDiffGate`. Camera clock offset field populated from PTS vs wall clock (E-ING-01). | gate on a WILDTRACK camera at 1 fps, blobs plotted |
| 3 | Colab L4: `bench/ring1_detect.py` — RF-DETR Nano/Medium on whole frames at 2 fps over WILDTRACK cam 1. Detections → `Detection` with class filter. | ms/frame, det count row |
| 4 | Wire detections → `SimpleIoUTracker` → `EventCompiler` (enter/exit/dwell) with hand-drawn zones on 2 WILDTRACK cams → `EpisodeWriter`. | episode JSONL from real footage |
| 5 | Postgres (Neon free tier or local) tables: ticks, events, tubes, entities, custody, facts. Loader from JSONL. One `search(tile,time,class)` SQL tool. | `search` returns real tubes |
| 6 | Own cameras online; 24 h recording to R2 begins (this becomes the eval set). Tag IR switch times manually. | footage accumulating |
| 7 | Buffer / catch-up. Write the week-1 benchmark rows. | table has rows for Ring 1 (stub gate, stub tracker) |

## Week 2 — real Ring 2 and the writer

| Day | Do | Gate |
|---|---|---|
| 8 | BoT-SORT/ByteTrack from source (Kalman + IoU + optional embedding). Keep the lifecycle rules from `SimpleIoUTracker`. | MOT17-half HOTA via TrackEval |
| 9 | OSNet ReID (torchreid) embeddings on detections. Appearance-assisted association. | IDF1 delta vs motion-only |
| 10 | Fusion v1 on WILDTRACK using its provided homographies: `project_foot` → floor distance + ReID + time → world entity ids. Overlap dedupe. | cross-camera merge accuracy row |
| 11 | Colab A100-40: vLLM + Qwen3.5-4B + xgrammar grammar from `ContactSheetResult.schema.json`. Contact-sheet packer (12/16 cells, borders, ids). | first valid sheet results |
| 12 | Label 200 crops (top/bottom color, carried item). Bleed-rate and accuracy bench (12 vs 16 cells). | Ring 3b row |
| 13 | Agent v0: Qwen3.5-9B on A100-40, tools `search`, `get_script`, `clip(crop)`. Question: "who crossed from camera 1 to camera 4, and when?" | first answered question with cited ids |
| 14 | Buffer. Promote any `planned` cases you touched to `implemented` with tests. | `make check` green |

## Week 3 — the real gate, events, night, own footage

| Day | Do | Gate |
|---|---|---|
| 15 | `MVGate`: PyAV `+export_mvs` → block-grid MV energy + coherence; same interface as `FrameDiffGate`. Compare to PyNvVideoCodec decode stats on Colab T4. | Ring 0 row: FN rate vs VIRAT activity, ms/GOP |
| 16 | Heartbeat scheduler (I-frame cadence per tile state) + `det_source=heartbeat` path. Asset-home zones on your own shelf. | pickup/asset_missing fires on your footage |
| 17 | Remaining events: approach/meet, left_behind, handoff, impossible_transition (needs tile graph), scene-state from gate. Event precision/recall on a MEVA subset. | Ring 3a row |
| 18 | Night profile: mode detection (chroma), threshold profile, `modality_switch` event, IR grammar variant. Run the writer on IR crops from your cameras. | zero color fields on IR sheets |
| 19 | Fusion v2 on your own cameras: manual 4-point homography per camera, tile graph with transit bounds from a movement walk you record. | hand-off works between two of your rooms |
| 20 | Episode soft-cut, patch-after-close path, retention check for `clip`. | E-STO-03 implemented |
| 21 | Buffer. Rows for Ring 0, 3a. | table ≥ 5 rows |

## Week 4 — KB, agent tools, scenarios, hardening

| Day | Do | Gate |
|---|---|---|
| 22 | Scene cards: one deep read per camera per regime with the biggest model you can reach; SAM 3 masks for assets (Colab). Facts table seeded. | cards for all your cameras |
| 23 | Actuation walk on your own site; scene-state events → manual fact seeding via `Fact.add_support`. `kb_lookup` tool. | switch → lights fact reaches `fact` |
| 24 | Agent tools `verify` (re-run writer on best crop), `clip(video)` stitched across cameras by entity, naming form (CLI) → gallery + relabel. | doorstep scenario replays |
| 25 | Custody chain + `carried_item` in sheets; keys scenario replay on your footage (place/remove an object on the shelf). | keys scenario replays |
| 26 | Colab G4: swap agent to Qwen3.8-27B for one session; compare tool-call count and latency. | agent row |
| 27 | Edge-case pass: walk `EDGE_CASES.md` top to bottom; every `planned` case is either implemented+tested or moved to `deferred` with a reason. | registry honest |
| 28 | Hardening: crash-replay test (kill mid-episode, restart, idempotency), 24 h soak on your cameras. | soak passes |
| 29 | Fill the benchmark table; write the "leave Colab" note (what goes to Hetzner vs RunPod). | table complete |
| 30 | Demo recording of both scenarios; retro on SPEC.md deltas. | v0.2 tag |

## Benchmark table (fill from `data/bench/*.jsonl`)

| Ring | Metric | Dataset | Hardware | Value | Date |
|---|---|---|---|---|---|
| 0 | gate FN rate (annotated activity) | VIRAT | Colab CPU | | |
| 0 | ms per GOP per stream (framediff / MV / NVDEC stats) | VIRAT | CPU / T4 | | |
| 1 | ms per packed ROI batch (Nano / Medium) | VIRAT ROIs | L4 | | |
| 1 | mAP on ROIs | VIRAT ROIs | L4 | | |
| 2 | HOTA / IDF1 (motion-only / +ReID) | MOT17-half | L4 | | |
| 2 | cross-camera merge accuracy | WILDTRACK | L4 | | |
| 3a | event precision / recall per type | MEVA subset | CPU | | |
| 3b | attribute accuracy | own crops | A100-40 | | |
| 3b | cross-cell bleed rate (12 / 16 cells) | own crops | A100-40 | | |
| 3b | ms per sheet | own crops | A100-40 | | |
| agent | tool calls / latency per question (9B / 27B) | WILDTRACK + own | A100-40 / G4 | | |
| all | cameras per GPU at 10% duty cycle (derived) | — | — | | |

## Definition of done for month 1

- `make check` green, ≥ 45 edge cases implemented and tested.
- Both scripted scenarios replay on your own footage with cited evidence.
- Every row above has a value or a written reason it could not be measured.
- SPEC.md updated with what the measurements changed.
