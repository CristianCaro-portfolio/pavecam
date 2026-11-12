"""Virtual pavement sensor: one reading for one camera, now or at a past time.

Inputs are only public feeds: the camera frame and the nearest airport (ASOS) weather
with its 24 hour history. No RWIS station is needed at the target site.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, date, datetime, timedelta

import joblib
import numpy as np
import polars as pl
from PIL import Image

from . import dataset as ds
from .config import Settings
from .iem import IEMClient
from .labels import CLASSES
from .sites import SITES, nearest_station
from .vision import PHYSICS_FEATURES, MobileClipEncoder, is_usable_frame, physics_features

log = logging.getLogger(__name__)

# Camera evidence that strongly contradicts the model is worth a human look.
DISAGREE_HIGH, DISAGREE_LOW = 0.9, 0.15


def _row_blocks(model_name: str, weather: np.ndarray, zs: np.ndarray, phys: np.ndarray, emb: np.ndarray,
                ref: dict | None) -> np.ndarray:
    """Rebuild the exact feature vector the chosen model was trained on for a single frame."""
    zs_log = np.log(zs + 1e-6)
    if model_name == "weather_gbm":
        return weather[None, :]
    if model_name == "image_probe":
        return emb[None, :]
    if ref is None:
        raise ValueError(f"model {model_name} needs a camera reference; run with a known camera")
    zs_c = zs_log - np.asarray(ref["zs_log_mean"])
    phys_z = (phys - np.asarray(ref["phys_mean"])) / np.asarray(ref["phys_std"])
    if model_name == "fused_gbm":
        return np.hstack([weather, zs_c, phys_z])[None, :]
    raise ValueError(f"single frame scoring not supported for {model_name}")


def score(bundle: dict, weather_row: dict, img: Image.Image, encoder, camera_id: str) -> dict:
    phys_d = physics_features(img)
    usable, why = is_usable_frame(phys_d)
    emb = encoder.embed_images([img])[0]
    zs = encoder.zero_shot(emb[None, :])[0]
    weather = np.array([weather_row.get(c) if weather_row.get(c) is not None else np.nan
                        for c in (*bundle["weather_features"], *bundle["history_features"])], dtype=float)
    phys = np.array([phys_d[k] for k in PHYSICS_FEATURES])
    x = _row_blocks(bundle["model_name"], weather, zs, phys, emb, bundle["camera_reference"].get(camera_id))
    proba = np.zeros(len(CLASSES))
    proba[bundle["model"].classes_] = bundle["model"].predict_proba(x)[0]
    label = CLASSES[int(proba.argmax())]
    confident = bool(proba.max() >= bundle["confidence_threshold"])
    cam_frozen = float(zs[CLASSES.index("frozen")])
    disagree = (label != "frozen" and cam_frozen >= DISAGREE_HIGH) or (label == "frozen" and cam_frozen <= DISAGREE_LOW)
    return {
        "condition": label if confident else "uncertain",
        "best_guess": label,
        "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, proba, strict=True)},
        "camera_zero_shot": {c: round(float(p), 3) for c, p in zip(CLASSES, zs, strict=True)},
        "frame_usable": usable,
        "frame_check": why,
        "review_camera": bool(disagree or not usable),
        "model": bundle["model_name"],
    }


def _site_for_camera(camera_id: str):
    return next((s for s in SITES if s.camera_id == camera_id), None)


def predict_camera(camera_id: str, when: str | None = None, asos_id: str | None = None,
                   settings: Settings | None = None) -> dict:
    s = settings or Settings()
    bundle_path = s.artifacts_dir / "pavecam_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"{bundle_path} not found, run pavecam train first")
    bundle = joblib.load(bundle_path)
    with IEMClient(s.iem_base, s.raw_dir / "cache") as client:
        if when:
            target = datetime.strptime(when, "%Y-%m-%dT%H:%M").replace(tzinfo=UTC)
            times = client.frame_times(camera_id, target.date())
            if not times:
                raise LookupError(f"no archived frames for {camera_id} on {target.date()}")
            ts = min(times, key=lambda t: abs(t - target))
            if abs(ts - target) > timedelta(minutes=s.frame_tolerance_min):
                raise LookupError(f"closest frame is {ts:%H:%M}, more than {s.frame_tolerance_min} min away")
            url = client.frame_url(camera_id, ts)
        else:
            url, ts = client.current_frame_url(camera_id)
        img = Image.open(io.BytesIO(client.download(url))).convert("RGB")

        site = _site_for_camera(camera_id)
        if asos_id is None:
            if site is None:
                raise ValueError("unknown camera location, pass --asos with the nearest airport id")
            asos_id, _ = nearest_station(site.lat, site.lon, client.network_stations("IA_ASOS"))
        day: date = ts.date()
        wx = ds.asos_features(client.asos([asos_id], day - timedelta(days=1), day, use_cache=bool(when)))
        wx = ds.add_weather_history(wx).filter(pl.col("asos_time") <= ts).sort("asos_time")
        if wx.is_empty():
            raise LookupError(f"no ASOS observation at {asos_id} before {ts}")
        last = wx.row(-1, named=True)
        if ts - last["asos_time"] > timedelta(minutes=s.asos_tolerance_min):
            raise LookupError(f"latest ASOS at {asos_id} is older than {s.asos_tolerance_min} min")

    out = score(bundle, last, img, MobileClipEncoder(s.clip_model, s.clip_pretrained), camera_id)
    out.update({"camera": camera_id, "frame_time_utc": ts.isoformat(), "asos": asos_id,
                "asos_time_utc": last["asos_time"].isoformat(), "air_f": last["air_f"]})
    return out
