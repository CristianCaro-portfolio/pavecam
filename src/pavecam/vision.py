"""Image side: frozen MobileCLIP2 encoder, zero-shot prompts and cheap physical cues.

The encoder is behind a small protocol so tests and CI can use a deterministic stand in
without downloading weights.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image

from .labels import CLASSES

log = logging.getLogger(__name__)

# Several phrasings per class; averaging text embeddings is the standard CLIP ensemble trick.
PROMPTS: dict[str, tuple[str, ...]] = {
    "dry": (
        "a traffic camera photo of a dry road",
        "a highway bridge with dry clear pavement",
        "a road surface that is dry and grey",
    ),
    "wet": (
        "a traffic camera photo of a wet road",
        "a highway bridge with wet shiny pavement reflecting light",
        "a road surface wet from rain",
    ),
    "frozen": (
        "a traffic camera photo of a snow covered road",
        "a highway bridge with icy pavement and snow",
        "a road surface with ice, frost or slush",
    ),
}


class Encoder(Protocol):
    name: str
    dim: int

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def zero_shot(self, emb: np.ndarray) -> np.ndarray: ...


def _prep(img: Image.Image, size: int) -> Image.Image:
    # Squash the full frame instead of a center crop: the instrumented lane is often near
    # the frame edge, and the RWIS imagery literature reports better results on full frames.
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


class MobileClipEncoder:
    """MobileCLIP2 image and text towers through open_clip, CPU only, no gradients."""

    def __init__(self, model: str = "MobileCLIP2-S0", pretrained: str = "dfndr2b", batch_size: int = 32) -> None:
        import open_clip
        import torch

        self._torch = torch
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.model, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.size = int(self.model.visual.image_size[0] if hasattr(self.model.visual, "image_size") else 256)
        # Keep the model's own normalisation, drop its resize and crop.
        self._norm = [t for t in preprocess.transforms if type(t).__name__ == "Normalize"][0]
        self.name = f"{model}/{pretrained}"
        self.batch_size = batch_size
        with torch.no_grad():
            probe = self.model.encode_text(self.tokenizer(["probe"]))
        self.dim = int(probe.shape[-1])
        self._text = self._class_text()

    def _class_text(self) -> np.ndarray:
        torch = self._torch
        rows = []
        with torch.no_grad():
            for c in CLASSES:
                t = self.model.encode_text(self.tokenizer(list(PROMPTS[c])))
                t = t / t.norm(dim=-1, keepdim=True)
                m = t.mean(0)
                rows.append((m / m.norm()).numpy())
        return np.stack(rows)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = []
            for img in images[i : i + self.batch_size]:
                arr = np.asarray(_prep(img, self.size), dtype=np.float32) / 255.0
                batch.append(self._norm(torch.from_numpy(arr).permute(2, 0, 1)))
            with torch.no_grad():
                e = self.model.encode_image(torch.stack(batch))
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def zero_shot(self, emb: np.ndarray) -> np.ndarray:
        """Class probabilities from cosine similarity to the prompt ensemble (CLIP logit scale 100)."""
        logits = 100.0 * emb @ self._text.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


class HashEncoder:
    """Deterministic stand in for tests and CI: colour histogram projected to a fixed dim."""

    name = "hash-encoder"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        seed = int(hashlib.sha256(b"pavecam").hexdigest()[:8], 16)
        self._proj = np.random.default_rng(seed).normal(size=(48, dim)).astype(np.float32)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for img in images:
            a = np.asarray(_prep(img, 64), dtype=np.float32) / 255.0
            h = np.concatenate([np.histogram(a[..., c], bins=16, range=(0, 1))[0] for c in range(3)]).astype(np.float32)
            v = h @ self._proj
            rows.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def zero_shot(self, emb: np.ndarray) -> np.ndarray:
        return np.full((emb.shape[0], len(CLASSES)), 1.0 / len(CLASSES))


# ---------------------------------------------------------------- physical cues
PHYSICS_FEATURES: tuple[str, ...] = (
    "lum_mean", "lum_std", "sat_mean", "white_frac", "dark_frac", "glare_frac",
    "low_lum_mean", "low_white_frac", "low_sat_mean", "low_edge_density",
)


def physics_features(img: Image.Image) -> dict[str, float]:
    """Cheap, explainable cues. 'low_' features use the lower half of the frame, where the
    deck usually is. Snow raises whiteness and lowers saturation, wet pavement darkens the
    surface and adds specular glare, and snow cover hides lane texture (edge density)."""
    hsv = np.asarray(img.convert("RGB").resize((192, 144)).convert("HSV"), dtype=np.float32) / 255.0
    s, v = hsv[..., 1], hsv[..., 2]
    low = slice(hsv.shape[0] // 2, None)
    white = (v > 0.72) & (s < 0.12)
    gy, gx = np.gradient(v[low])
    return {
        "lum_mean": float(v.mean()),
        "lum_std": float(v.std()),
        "sat_mean": float(s.mean()),
        "white_frac": float(white.mean()),
        "dark_frac": float((v < 0.2).mean()),
        "glare_frac": float((v > 0.95).mean()),
        "low_lum_mean": float(v[low].mean()),
        "low_white_frac": float(white[low].mean()),
        "low_sat_mean": float(s[low].mean()),
        "low_edge_density": float((np.hypot(gx, gy) > 0.08).mean()),
    }


def is_usable_frame(feats: dict[str, float]) -> tuple[bool, str]:
    """Reject frames that carry no road information: black (night or outage), blown out,
    or a flat single colour (camera error card, lens fully covered)."""
    if feats["lum_mean"] < 0.08:
        return False, "too dark"
    if feats["lum_mean"] > 0.97:
        return False, "blown out"
    if feats["lum_std"] < 0.02:
        return False, "flat frame"
    return True, "ok"
