import numpy as np
from PIL import Image

from pavecam.predict import score
from pavecam.vision import HashEncoder


class _Model:
    classes_ = np.array([0, 1, 2])

    def __init__(self, p):
        self.p = np.array([p])

    def predict_proba(self, x):
        assert x.shape == (1, 18)  # 11 current weather + 7 history features
        return self.p


def _bundle(p):
    return {
        "model_name": "weather_gbm", "model": _Model(p), "confidence_threshold": 0.8,
        "weather_features": ["air_f", "dew_f", "dew_spread_f", "rh", "wind_kt", "vis_mi", "precip_in",
                             "wx_snow", "wx_rain", "wx_freezing", "wx_fog"],
        "history_features": ["precip_12h", "snow_12h", "freezing_12h", "min_air_12h", "hours_frozen_12h",
                             "snow_24h", "max_air_24h"],
        "camera_reference": {},
    }


def _frame(sample_dir):
    return Image.open(sorted((sample_dir / "frames").glob("*.jpg"))[0])


def test_confident_reading(sample_dir):
    out = score(_bundle([0.05, 0.05, 0.9]), {"air_f": 20.0}, _frame(sample_dir), HashEncoder(), "IDOT-X")
    assert out["condition"] == "frozen" and out["frame_usable"]


def test_low_confidence_becomes_uncertain(sample_dir):
    out = score(_bundle([0.4, 0.35, 0.25]), {"air_f": 40.0}, _frame(sample_dir), HashEncoder(), "IDOT-X")
    assert out["condition"] == "uncertain" and out["best_guess"] == "dry"
