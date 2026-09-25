"""Appearance embeddings for re-identification (R14 / E-TUBE-04 / E-TUBE-08).

Backends, all open weights and no training:
  hist    CPU colour histogram (tests, fallback, weak baseline)
  siglip  google/siglip2-base-patch16-224 image tower (Apache-2.0, Hugging Face) — generic but
          reliable to download; the default GPU embedder until OSNet weight hosting is verified
  osnet   torchreid OSNet (MIT) — the dedicated ReID model in the BOM; weights come from the
          author's Google Drive, so this backend is optional and verified per runtime
Every backend returns L2-normalised float32 vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from vi.schemas import Box


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


def crop_for_embedding(frame_rgb: np.ndarray, box: Box, pad: float = 0.08) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    px, py = box.width * pad, box.height * pad
    x1, y1 = int(max(0, box.x1 - px)), int(max(0, box.y1 - py))
    x2, y2 = int(min(w, box.x2 + px)), int(min(h, box.y2 + py))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((8, 4, 3), np.uint8)
    return np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


class HistogramEmbedder:
    """Upper/lower body colour histograms (8 bins per RGB channel each) -> 48-D. Deliberately
    simple: clothing colour is what survives a 13-second occlusion in a warehouse."""

    name = "hist"
    dim = 48

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            h = c.shape[0]
            parts = (c[: h // 2], c[h // 2:])
            feats = []
            for part in parts:
                for ch in range(3):
                    hist, _ = np.histogram(part[..., ch], bins=8, range=(0, 256))
                    feats.append(hist.astype(np.float32) / max(1, part.size / 3))
            out[i] = np.concatenate(feats)
        return _l2(out)


class SigLIPEmbedder:
    name = "siglip"
    dim = 768

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        ims = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        with self.torch.no_grad():
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
        return _l2(feats.float().cpu().numpy())


class OSNetEmbedder:
    name = "osnet"
    dim = 512

    def __init__(self, model_name: str = "osnet_x0_25", model_path: str | None = None, device: str | None = None):
        import torch
        from torchreid.utils import FeatureExtractor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(model_name=model_name, model_path=model_path, device=self.device)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        feats = self.extractor(list(crops))
        return _l2(feats.cpu().numpy())


def make_embedder(kind: str = "auto") -> Embedder:
    """auto: osnet if importable and weights load, else siglip, else hist. Prints the choice."""
    order = ["osnet", "siglip", "hist"] if kind == "auto" else [kind]
    for k in order:
        try:
            if k == "hist":
                return HistogramEmbedder()
            if k == "siglip":
                e = SigLIPEmbedder()
            elif k == "osnet":
                e = OSNetEmbedder()
            else:
                raise ValueError(k)
            print(f"[reid] embedder: {e.name} ({e.dim}-d)")
            return e
        except Exception as ex:  # pragma: no cover
            print(f"[reid] {k} unavailable ({type(ex).__name__}: {str(ex)[:80]})")
    return HistogramEmbedder()
