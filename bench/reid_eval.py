"""Which appearance embedding actually tells these workers apart? For every embedder that loads,
embed each labelled tube's birth crop and compare same-person pairs against different-person
pairs. Prints one table and writes data/bench/reid.jsonl. Decides the linker's embedder/threshold.

  python bench/reid_eval.py data/episodes/*.jsonl --labels tests/scenarios/warehouse_identities.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from vi.episode import EpisodeWriter
from vi.reid import HistogramEmbedder
from vi.schemas.episode import TubeRecord


def load_embedders(names: list[str]) -> dict:
    out = {}
    for n in names:
        try:
            if n == "hist":
                out[n] = HistogramEmbedder()
            elif n == "siglip":
                from vi.reid.base import SigLIPEmbedder
                out[n] = SigLIPEmbedder()
            elif n == "dinov2":
                out[n] = DINOv2Embedder()
            elif n == "osnet":
                from vi.reid.base import OSNetEmbedder
                out[n] = OSNetEmbedder(model_name="osnet_x1_0")
            print(f"[reid_eval] {n}: loaded")
        except Exception as e:
            print(f"[reid_eval] {n}: unavailable ({type(e).__name__}: {str(e)[:100]})")
    return out


class DINOv2Embedder:
    """facebook/dinov2-small (Apache-2.0): self-supervised features, known to be strong for
    instance-level similarity; CLS token, L2-normalised."""

    name = "dinov2"
    dim = 384

    def __init__(self, model_id: str = "facebook/dinov2-small"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops):
        from PIL import Image
        ims = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        with self.torch.no_grad():
            inputs = self.proc(images=ims, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        f = feats.float().cpu().numpy()
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--labels", default="tests/scenarios/warehouse_identities.yaml")
    ap.add_argument("--keyframes", default="data/keyframes")
    ap.add_argument("--embedders", default="hist,siglip,dinov2,osnet")
    ap.add_argument("--tol", type=int, default=25, help="px tolerance matching labels to birth crops")
    a = ap.parse_args()
    from PIL import Image
    labels = yaml.safe_load(Path(a.labels).read_text())
    tubes = [r.tube for r in EpisodeWriter.read(a.episode) if isinstance(r, TubeRecord) and r.tube.class_label == "person"]
    rows = []
    for t in tubes:
        ref = (t.keyframe_refs or [""])[0]
        if not ref.startswith("kf://"):
            continue
        name = Path(ref).stem                       # <born>_birth_<x>_<y>
        parts = name.split("_")
        try:
            born, x, y = int(parts[0]), int(parts[2]), int(parts[3])
        except Exception:
            continue
        lab = next((l for l in labels["tubes"] if abs(l["born_ms"] - born) <= 400 and abs(l["x"] - x) <= a.tol and abs(l["y"] - y) <= a.tol), None)
        path = Path(a.keyframes) / ref.replace("kf://", "")
        if lab and path.exists():
            rows.append({"tube": t.tube_id, "person": lab["person"], "sure": bool(lab.get("sure", True)),
                         "crop": np.asarray(Image.open(path).convert("RGB"))})
    print(f"labelled tubes with crops: {len(rows)} / {len(tubes)}  (sure: {sum(r['sure'] for r in rows)})")
    if len(rows) < 3:
        raise SystemExit("not enough labelled crops; check keyframes dir and label coordinates")
    embedders = load_embedders([e.strip() for e in a.embedders.split(",") if e.strip()])
    sure = [r for r in rows if r["sure"]]
    results = {}
    for name, emb in embedders.items():
        E = emb.embed([r["crop"] for r in rows])
        idx = {r["tube"]: i for i, r in enumerate(rows)}
        pos, neg = [], []
        for r1, r2 in itertools.combinations(sure, 2):
            s = float(E[idx[r1["tube"]]] @ E[idx[r2["tube"]]])
            (pos if r1["person"] == r2["person"] else neg).append(s)
        if not pos or not neg:
            continue
        ths = np.linspace(0.3, 0.99, 70)
        acc = [(np.mean([p >= th for p in pos]) + np.mean([n < th for n in neg])) / 2 for th in ths]
        best = float(ths[int(np.argmax(acc))])
        results[name] = {"pairs_same": len(pos), "pairs_diff": len(neg), "same_mean": round(float(np.mean(pos)), 3),
                         "same_min": round(float(np.min(pos)), 3), "diff_mean": round(float(np.mean(neg)), 3),
                         "diff_max": round(float(np.max(neg)), 3), "gap": round(float(np.min(pos) - np.max(neg)), 3),
                         "best_thr": round(best, 2), "balanced_acc": round(float(max(acc)), 3)}
        # the pairs that matter on this clip
        p4 = [r for r in sure if r["person"] == "P4"]
        results[name]["hardhat_pairs"] = [round(float(E[idx[x["tube"]]] @ E[idx[y["tube"]]]), 3) for x, y in itertools.combinations(p4, 2)]
        p7 = next((r for r in rows if r["person"] == "P7" and r["sure"]), None)
        p8 = next((r for r in rows if r["person"] == "P8"), None)
        if p7 and p8:
            results[name]["greenvest_vs_orangevest"] = round(float(E[idx[p7["tube"]]] @ E[idx[p8["tube"]]]), 3)
    print(f"\n{'embedder':8s} {'same mean/min':>14s} {'diff mean/max':>14s} {'gap':>6s} {'thr':>5s} {'bal.acc':>7s}  hard-hat pairs      green vs orange")
    for name, r in results.items():
        print(f"{name:8s} {r['same_mean']:>6.3f}/{r['same_min']:<6.3f} {r['diff_mean']:>6.3f}/{r['diff_max']:<6.3f} "
              f"{r['gap']:>6.3f} {r['best_thr']:>5.2f} {r['balanced_acc']:>7.3f}  {r['hardhat_pairs']}  {r.get('greenvest_vs_orangevest', '-')}")
    print("\nread: a positive gap means every same-person pair scores above every different-person pair; "
          "hard-hat pairs must clear the threshold, green-vs-orange must fall below it.")
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "reid.jsonl").open("a") as f:
        f.write(json.dumps({"ring": "reid_eval", "labelled": len(rows), "results": results,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


if __name__ == "__main__":
    main()
