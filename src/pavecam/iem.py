"""Thin client for the Iowa Environmental Mesonet (IEM) public endpoints.

No API key is needed. The client is polite on purpose: bounded workers, retries with
backoff on transient errors, and an on-disk cache so reruns do not hit the server again.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

_UTC = UTC
_FRAME_RE = re.compile(r"_(\d{12})\.jpg")
USER_AGENT = "pavecam/0.1 (research; github.com/CristianCaro-portfolio/pavecam)"


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, httpx.TransportError)


class IEMClient:
    def __init__(self, base: str, cache_dir: Path, timeout: float = 90.0) -> None:
        self.base = base.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> IEMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception(_transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        reraise=True,
    )
    def _get(self, url: str, params: list[tuple[str, str]] | None = None) -> bytes:
        r = self._http.get(url, params=params)
        r.raise_for_status()
        return r.content

    def _cached(self, key: str, fetch) -> bytes:
        path = self.cache_dir / key
        if path.exists() and path.stat().st_size > 0:
            return path.read_bytes()
        data = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    # ---------- station metadata ----------
    def network_stations(self, network: str) -> dict[str, tuple[float, float]]:
        """{station_id: (lat, lon)} for active stations of an IEM network."""
        raw = self._cached(
            f"meta/{network}.geojson",
            lambda: self._get(f"{self.base}/geojson/network/{network}.geojson"),
        )
        out = {}
        for f in json.loads(raw)["features"]:
            if f["properties"].get("archive_end"):
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            out[f["id"]] = (lat, lon)
        return out

    # ---------- RWIS pavement sensors ----------
    def rwis(self, stations: list[str], start: date, end: date) -> pl.DataFrame:
        params = [("network", "IA_RWIS"), ("what", "txt"), ("delim", "comma"), ("tz", "UTC")]
        params += [("stations", s) for s in stations]
        for v in ("tmpf", "dwpf", "tfs0", "tfs0_text", "tfs1", "tfs1_text"):
            params.append(("vars", v))
        params += _date_params(start, end)
        key = f"rwis/{'-'.join(sorted(stations))[:80]}_{start}_{end}.csv"
        raw = self._cached(key, lambda: self._get(f"{self.base}/cgi-bin/request/rwis.py", params))
        df = pl.read_csv(io.BytesIO(raw), null_values=["", "M", "NA"], infer_schema_length=10000)
        return df.with_columns(pl.col("obtime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC"))

    # ---------- ASOS airport weather ----------
    def asos(self, stations: list[str], start: date, end: date, use_cache: bool = True) -> pl.DataFrame:
        params = [
            ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("missing", "empty"),
            ("trace", "0.0001"), ("direct", "no"), ("report_type", "3"),
        ]
        params += [("station", s) for s in stations]
        for v in ("tmpf", "dwpf", "relh", "sknt", "vsby", "p01i", "wxcodes"):
            params.append(("data", v))
        params += _date_params(start, end)
        key = f"asos/{'-'.join(sorted(stations))[:80]}_{start}_{end}.csv"
        url = f"{self.base}/cgi-bin/request/asos.py"
        # A window that includes today keeps growing, so live calls skip the cache.
        raw = self._cached(key, lambda: self._get(url, params)) if use_cache else self._get(url, params)
        df = pl.read_csv(io.BytesIO(raw), null_values=["", "M"], infer_schema_length=10000,
                         schema_overrides={"wxcodes": pl.Utf8})
        return df.with_columns(pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC"))

    # ---------- camera archive ----------
    def frame_times(self, camera_id: str, day: date) -> list[datetime]:
        url = f"{self.base}/archive/data/{day:%Y/%m/%d}/camera/{camera_id}/"
        try:
            raw = self._cached(f"listing/{camera_id}/{day}.html", lambda: self._get(url))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return []
            raise
        stamps = sorted(set(_FRAME_RE.findall(raw.decode("utf-8", "ignore"))))
        return [datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=_UTC) for s in stamps]

    def frame_url(self, camera_id: str, ts: datetime) -> str:
        return f"{self.base}/archive/data/{ts:%Y/%m/%d}/camera/{camera_id}/{camera_id}_{ts:%Y%m%d%H%M}.jpg"

    def current_frame_url(self, camera_id: str) -> tuple[str, datetime]:
        raw = self._get(f"{self.base}/json/webcams.py", [("network", "IDOT")])
        for img in json.loads(raw)["images"]:
            if img["cid"] == camera_id:
                m = _FRAME_RE.search(img["url"])
                if not m:
                    break
                return img["url"], datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=_UTC)
        raise LookupError(f"camera {camera_id} not in the current IDOT inventory")

    def download(self, url: str) -> bytes:
        return self._get(url)




def _date_params(start: date, end: date) -> list[tuple[str, str]]:
    end_incl = end + timedelta(days=1)
    return [
        ("year1", str(start.year)), ("month1", str(start.month)), ("day1", str(start.day)),
        ("hour1", "0"), ("minute1", "0"),
        ("year2", str(end_incl.year)), ("month2", str(end_incl.month)), ("day2", str(end_incl.day)),
        ("hour2", "0"), ("minute2", "0"),
    ]
