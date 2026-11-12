"""Runtime settings. Everything has a sane default and can be overridden by env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PAVECAM_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    iem_base: str = "https://mesonet.agron.iastate.edu"
    # Winter 2025-26 in Iowa. Long enough to see snow, freezing rain, wet and dry days.
    start: date = date(2025, 11, 15)
    end: date = date(2026, 3, 15)
    # Daylight slots in UTC (09:00 to 15:00 CST). Night frames are out of scope, see README.
    slots_utc: tuple[str, ...] = ("15:00", "16:30", "18:00", "19:30", "21:00")
    n_frozen_days: int = 18
    n_wet_days: int = 12
    n_dry_days: int = 10
    seed: int = 7
    max_workers: int = field(default_factory=lambda: int(_env("MAX_WORKERS", "4")))
    frame_tolerance_min: int = 10
    label_tolerance_min: int = 10
    asos_tolerance_min: int = 90
    image_width: int = 384
    clip_model: str = field(default_factory=lambda: _env("CLIP_MODEL", "MobileCLIP2-S0"))
    clip_pretrained: str = field(default_factory=lambda: _env("CLIP_PRETRAINED", "dfndr2b"))
    mlflow_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def frames_dir(self) -> Path:
        return self.raw_dir / "frames"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.frames_dir, self.processed_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)
