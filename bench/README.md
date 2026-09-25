# bench/

One script per ring. Each prints exactly one row of the benchmark table (see PLAN.md) and
checkpoints it to `data/bench/<ring>.jsonl` as it goes, so a killed Colab session loses nothing.

| script | ring | GPU | dataset | row |
|---|---|---|---|---|
| slice_cpu.py | all (stubs) | none | synthetic | proves plumbing; run first, every day |
| slice_gpu.py | 0–3a on real footage | L4 | own clip | reader → gate → ROIs (+ heartbeat) → RF-DETR → tubes → events → episode file + birth keyframes; `--detect roi\|frame\|hybrid`, person_tubes, visibility duty, fragmentation_est |
| ring0_gate.py | 0 | CPU | VIRAT | gate FN rate, ms per GOP per stream, MV vs framediff |
| ring1_detect.py | 1 | L4 | own clip / VIRAT | `--mode frame`: ms p50/p95 per frame at sampled fps; `--mode roi`: gate → packed ROI batches (nano/medium) |
| ring2_tubes.py | 2 | CPU | synthetic, MOT17 | MOTA/IDF1/IDSW/fragmentation (vi/eval/mot.py); `--tracker simple\|byte`, `--sample-every` simulates 2–5 fps decode |
| ring3b_sheet.py | 3b | A100-40 | crops from ring2 | attribute accuracy, bleed rate (12 vs 16 cells), ms per sheet |
| ring3a_events.py | 3a | CPU | MEVA subset | event precision/recall per type |
| agent_replay.py | block 2 | A100-40 -> G4 | WILDTRACK episodes | both scenarios replay; tool calls, latency |

Colab: GPU runtime, `pip install rfdetr==1.7.0`, then run from the repo root (see colab/README.md).
