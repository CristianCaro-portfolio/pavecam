"""Prefect flows: ingest (network heavy, retried, parallel) and featurize (CPU heavy)."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict
from datetime import date

import numpy as np
import polars as pl
from PIL import Image
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.task_runners import ThreadPoolTaskRunner

from . import dataset as ds
from .config import Settings
from .iem import IEMClient
from .sites import SITES, nearest_station
from .vision import PHYSICS_FEATURES, Encoder, is_usable_frame, physics_features

log = logging.getLogger(__name__)


def _logger():
    try:
        return get_run_logger()
    except Exception:  # outside a flow run context
        return log


@task(retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def list_frames(client: IEMClient, station: str, camera_id: str, day: date) -> list[dict]:
    return [{"station": station, "frame_time": t} for t in client.frame_times(camera_id, day)]


@task(retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def fetch_frame(client: IEMClient, url: str, dest: str, width: int) -> bool:
    from pathlib import Path

    p = Path(dest)
    if p.exists() and p.stat().st_size > 0:
        return True
    raw = client.download(url)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p, "JPEG", quality=85)
    return True


@flow(name="pavecam-ingest", task_runner=ThreadPoolTaskRunner(max_workers=4), validate_parameters=False)
def ingest_flow(settings: Settings | None = None) -> str:
    s = settings or Settings()
    s.ensure_dirs()
    lg = _logger()
    client = IEMClient(s.iem_base, s.raw_dir / "cache")
    try:
        stations = [x.rwis_id for x in SITES]
        rwis = client.rwis(stations, s.start, s.end)
        labels = ds.rwis_labels(rwis)
        quality = ds.site_quality(labels)
        accepted = [q.station for q in quality if q.accepted]
        for q in quality:
            if not q.accepted:
                lg.warning("site %s rejected: %s", q.station, q.reason)
        labels = labels.filter(pl.col("station").is_in(accepted))
        lg.info("RWIS obs %d, accepted sites %d of %d", rwis.height, len(accepted), len(stations))

        days = ds.select_days(labels, s.n_frozen_days, s.n_wet_days, s.n_dry_days, s.seed)
        day_list = days["day"].to_list()
        cands = ds.candidate_slots(day_list, s.slots_utc, accepted)

        site = {x.rwis_id: x for x in SITES}
        futs = [list_frames.submit(client, st, site[st].camera_id, d) for st in accepted for d in day_list]
        frames = pl.DataFrame([r for f in futs for r in f.result()],
                              schema={"station": pl.Utf8, "frame_time": pl.Datetime("us", "UTC")})

        m = ds.attach_labels(cands, labels, s.label_tolerance_min)
        m = ds.attach_frames(m, frames, s.frame_tolerance_min)
        n0 = m.height
        m = m.drop_nulls(["label", "frame_time"]).unique(["station", "frame_time"], keep="first")
        lg.info("slots %d, with label and frame %d", n0, m.height)

        asos_sites = client.network_stations("IA_ASOS")
        site_to_asos, dist = {}, {}
        for st in accepted:
            sid, km = nearest_station(site[st].lat, site[st].lon, asos_sites)
            site_to_asos[st], dist[st] = sid, round(km, 1)
        asos = ds.add_weather_history(ds.asos_features(client.asos(sorted(set(site_to_asos.values())), s.start, s.end)))
        m = ds.attach_weather(m, asos, site_to_asos, s.asos_tolerance_min)

        m = m.with_columns(
            pl.col("station").replace_strict({k: v.camera_id for k, v in site.items()}).alias("camera_id"),
        )
        m = m.with_columns(
            pl.struct("camera_id", "frame_time").map_elements(
                lambda r: ds.frame_filename(r["camera_id"], r["frame_time"]), return_dtype=pl.Utf8
            ).alias("frame_file")
        )
        dl = [
            fetch_frame.submit(client, client.frame_url(r["camera_id"], r["frame_time"]),
                               str(s.frames_dir / r["frame_file"]), s.image_width)
            for r in m.iter_rows(named=True)
        ]
        ok = []
        for f in dl:
            try:
                ok.append(bool(f.result()))
            except Exception as exc:  # one bad frame must not kill the run
                lg.warning("frame download failed: %s", exc)
                ok.append(False)
        m = m.with_columns(pl.Series("downloaded", ok)).filter(pl.col("downloaded")).drop("downloaded")

        out = s.processed_dir / "manifest.parquet"
        m.write_parquet(out)
        report = {
            "window": [str(s.start), str(s.end)],
            "days": {k: int(v) for k, v in days.group_by("kind").len().iter_rows()},
            "sites": [asdict(q) for q in quality],
            "site_to_asos": site_to_asos,
            "asos_distance_km": dist,
            "slots": n0,
            "frames": m.height,
            "label_counts": dict(m.group_by("label").len().iter_rows()),
        }
        (s.processed_dir / "ingest_report.json").write_text(json.dumps(report, indent=2, default=str))
        lg.info("manifest written: %s (%d rows)", out, m.height)
        return str(out)
    finally:
        client.close()


@flow(name="pavecam-featurize", validate_parameters=False)
def featurize_flow(encoder: object, settings: Settings | None = None) -> str:
    """encoder: any object implementing vision.Encoder (typed loosely so Prefect skips validation)."""
    enc: Encoder = encoder  # type: ignore[assignment]
    s = settings or Settings()
    lg = _logger()
    m = pl.read_parquet(s.processed_dir / "manifest.parquet")
    imgs, phys, keep, reasons = [], [], [], {}
    for r in m.iter_rows(named=True):
        img = Image.open(s.frames_dir / r["frame_file"]).convert("RGB")
        f = physics_features(img)
        good, why = is_usable_frame(f)
        keep.append(good)
        if good:
            imgs.append(img)
            phys.append(f)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    lg.info("frame gate: kept %d of %d, rejected %s", sum(keep), len(keep), reasons)
    m = m.filter(pl.Series(keep))
    emb = enc.embed_images(imgs)
    zs = enc.zero_shot(emb)
    feats = m.with_columns(
        *[pl.Series(k, [p[k] for p in phys]) for k in PHYSICS_FEATURES],
        pl.Series("emb", emb.astype(np.float32).tolist(), dtype=pl.List(pl.Float32)),
        pl.Series("zs_dry", zs[:, 0]), pl.Series("zs_wet", zs[:, 1]), pl.Series("zs_frozen", zs[:, 2]),
        pl.lit(enc.name).alias("encoder"),
    )
    out = s.processed_dir / "features.parquet"
    feats.write_parquet(out)
    (s.processed_dir / "frame_gate.json").write_text(json.dumps({"kept": int(sum(keep)), "rejected": reasons}))
    return str(out)
