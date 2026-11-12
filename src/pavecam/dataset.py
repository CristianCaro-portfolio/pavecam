"""Turn raw RWIS, ASOS and camera listings into a labeled frame manifest.

All functions here are pure (DataFrame in, DataFrame out) so they are easy to test
without network access. The Prefect flow in flows.py wires them to the IEM client.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import polars as pl

from .labels import pick_label, to_class

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- labels
def rwis_labels(rwis: pl.DataFrame) -> pl.DataFrame:
    """One row per RWIS observation with the mapped class. Unusable codes become null."""
    return rwis.select(
        pl.col("station"),
        pl.col("obtime"),
        pl.struct("tfs0_text", "tfs1_text")
        .map_elements(lambda r: pick_label(r["tfs0_text"], r["tfs1_text"]), return_dtype=pl.Utf8)
        .alias("label"),
        pl.when(pl.col("tfs0_text").map_elements(to_class, return_dtype=pl.Utf8).is_not_null())
        .then(pl.col("tfs0_text"))
        .otherwise(pl.col("tfs1_text"))
        .alias("sensor_code"),
        pl.coalesce("tfs0", "tfs1").alias("pavement_f"),
        pl.col("tmpf").alias("rwis_air_f"),
    )


@dataclass(frozen=True)
class SiteQuality:
    station: str
    n_obs: int
    usable_frac: float
    frozen_frac: float
    accepted: bool
    reason: str


def site_quality(labels: pl.DataFrame, min_usable: float = 0.6, min_frozen: float = 0.02) -> list[SiteQuality]:
    """Quality gate per site: the pavement sensor must report usable codes most of the time,
    and must have seen some frozen conditions in a Midwest winter (else it is likely broken
    or stuck on a generic code)."""
    out = []
    stats = labels.group_by("station").agg(
        pl.len().alias("n"),
        pl.col("label").is_not_null().mean().alias("usable"),
        (pl.col("label") == "frozen").mean().alias("frozen"),
    )
    for row in stats.sort("station").iter_rows(named=True):
        usable, frozen = float(row["usable"] or 0), float(row["frozen"] or 0)
        if usable < min_usable:
            ok, why = False, f"only {usable:.0%} usable sensor codes"
        elif frozen < min_frozen:
            ok, why = False, f"frozen reported in {frozen:.1%} of obs, sensor likely stuck"
        else:
            ok, why = True, "ok"
        out.append(SiteQuality(row["station"], int(row["n"]), usable, frozen, ok, why))
    return out


# ---------------------------------------------------------------- sampling
def _slot_times(day: date, slots: tuple[str, ...]) -> list[datetime]:
    res = []
    for s in slots:
        hh, mm = (int(x) for x in s.split(":"))
        res.append(datetime.combine(day, time(hh, mm), tzinfo=UTC))
    return res


def select_days(
    labels: pl.DataFrame,
    n_frozen: int,
    n_wet: int,
    n_dry: int,
    seed: int,
    probe_hour_utc: int = 18,
    min_share: float = 0.15,
) -> pl.DataFrame:
    """Pick event days so the three classes are all well represented.

    A day is scored by the share of sites in each class around midday (probe hour).
    Frozen days are the most frozen ones, wet days the wettest of the rest, dry days a
    random draw from clearly dry days. Returns columns day, kind.
    """
    probe = (
        labels.drop_nulls("label")
        .filter(pl.col("obtime").dt.hour() == probe_hour_utc)
        .with_columns(pl.col("obtime").dt.date().alias("day"))
        .group_by("day")
        .agg(
            (pl.col("label") == "frozen").mean().alias("frozen"),
            (pl.col("label") == "wet").mean().alias("wet"),
            (pl.col("label") == "dry").mean().alias("dry"),
            pl.col("station").n_unique().alias("n_sites"),
        )
        .filter(pl.col("n_sites") >= 3)
    )
    frozen = probe.filter(pl.col("frozen") >= min_share).sort("frozen", descending=True).head(n_frozen)
    rest = probe.join(frozen.select("day"), on="day", how="anti")
    wet = rest.filter(pl.col("wet") >= min_share).sort("wet", descending=True).head(n_wet)
    rest = rest.join(wet.select("day"), on="day", how="anti")
    dry_pool = rest.filter(pl.col("dry") >= 0.8).sort("day")
    rng = np.random.default_rng(seed)
    k = min(n_dry, dry_pool.height)
    idx = sorted(rng.choice(dry_pool.height, size=k, replace=False).tolist()) if k else []
    dry = dry_pool[idx] if idx else dry_pool.head(0)
    return pl.concat([
        frozen.select("day").with_columns(pl.lit("frozen").alias("kind")),
        wet.select("day").with_columns(pl.lit("wet").alias("kind")),
        dry.select("day").with_columns(pl.lit("dry").alias("kind")),
    ]).sort("day")


def candidate_slots(days: list[date], slots: tuple[str, ...], stations: list[str]) -> pl.DataFrame:
    rows = [(s, t) for d in days for t in _slot_times(d, slots) for s in stations]
    return pl.DataFrame(rows, schema={"station": pl.Utf8, "slot": pl.Datetime("us", "UTC")}, orient="row")


# ---------------------------------------------------------------- pairing
def _asof(left: pl.DataFrame, right: pl.DataFrame, left_on: str, right_on: str, by: str,
          tolerance_min: int, strategy: str) -> pl.DataFrame:
    left = left.with_columns(pl.col(left_on).cast(pl.Datetime("us", "UTC"))).sort(left_on)
    right = right.with_columns(pl.col(right_on).cast(pl.Datetime("us", "UTC"))).sort(right_on)
    return left.join_asof(
        right, left_on=left_on, right_on=right_on, by=by,
        strategy=strategy, tolerance=timedelta(minutes=tolerance_min), check_sortedness=False,
    )


def attach_labels(cands: pl.DataFrame, labels: pl.DataFrame, tolerance_min: int) -> pl.DataFrame:
    lab = labels.drop_nulls("label").rename({"obtime": "label_time"})
    return _asof(cands, lab, "slot", "label_time", "station", tolerance_min, "nearest")


def attach_frames(cands: pl.DataFrame, frames: pl.DataFrame, tolerance_min: int) -> pl.DataFrame:
    """frames: columns station, frame_time (all archived frames for the site/day)."""
    return _asof(cands, frames, "slot", "frame_time", "station", tolerance_min, "nearest")


_WX_FLAGS = {
    "wx_snow": r"SN|SG|BLSN|DRSN",
    "wx_rain": r"(^|[\s+-])(SH|TS)?(RA|DZ)",  # not FZRA or FZDZ
    "wx_freezing": r"FZRA|FZDZ|FZFG|PL|IC",
    "wx_fog": r"FG|BR|HZ",
}


def asos_features(asos: pl.DataFrame) -> pl.DataFrame:
    """Numeric weather features from METAR rows. Present-weather codes become flags."""
    wx = pl.col("wxcodes").fill_null("")
    exprs = [pl.when(wx.str.contains(p)).then(1).otherwise(0).alias(k) for k, p in _WX_FLAGS.items()]
    return asos.select(
        pl.col("station").alias("asos_station"),
        pl.col("valid").alias("asos_time"),
        pl.col("tmpf").cast(pl.Float64).alias("air_f"),
        pl.col("dwpf").cast(pl.Float64).alias("dew_f"),
        pl.col("relh").cast(pl.Float64).alias("rh"),
        pl.col("sknt").cast(pl.Float64).alias("wind_kt"),
        pl.col("vsby").cast(pl.Float64).alias("vis_mi"),
        pl.col("p01i").cast(pl.Float64).fill_null(0.0).alias("precip_in"),
        *exprs,
    ).with_columns((pl.col("air_f") - pl.col("dew_f")).alias("dew_spread_f"))


def add_weather_history(feats: pl.DataFrame) -> pl.DataFrame:
    """Look back windows per airport. Pavement state has memory: snow that fell at 4 am is
    still on the deck at noon, and a night below freezing leaves ice after the sky clears."""
    f = feats.sort("asos_station", "asos_time")
    hist = f.rolling(index_column="asos_time", period="12h", group_by="asos_station").agg(
        pl.col("precip_in").sum().alias("precip_12h"),
        pl.col("wx_snow").max().alias("snow_12h"),
        pl.col("wx_freezing").max().alias("freezing_12h"),
        pl.col("air_f").min().alias("min_air_12h"),
        (pl.col("air_f") <= 32).sum().alias("hours_frozen_12h"),
    )
    hist24 = f.rolling(index_column="asos_time", period="24h", group_by="asos_station").agg(
        pl.col("wx_snow").max().alias("snow_24h"),
        pl.col("air_f").max().alias("max_air_24h"),
    )
    return (
        f.join(hist, on=["asos_station", "asos_time"], how="left")
        .join(hist24, on=["asos_station", "asos_time"], how="left")
        .unique(["asos_station", "asos_time"], keep="last")
        .sort("asos_station", "asos_time")
    )


def attach_weather(cands: pl.DataFrame, asos_feats: pl.DataFrame, site_to_asos: dict[str, str],
                   tolerance_min: int) -> pl.DataFrame:
    """Latest airport observation at or before the slot. Backward only: no peeking ahead."""
    c = cands.with_columns(pl.col("station").replace_strict(site_to_asos, default=None).alias("asos_station"))
    return _asof(c, asos_feats, "slot", "asos_time", "asos_station", tolerance_min, "backward")


def frame_filename(camera_id: str, ts: datetime) -> str:
    return f"{camera_id}_{ts:%Y%m%d%H%M}.jpg"


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_name(s: str) -> str:
    return _SAFE.sub("_", s)
