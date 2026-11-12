"""Model comparison with leave-one-site-out validation, final model and MLflow tracking.

Why leave one site out: every camera has its own angle, lighting and background. A random
split puts frames of the same camera, minutes apart, on both sides and inflates scores.
The question that matters is whether the model works on a camera it never saw, because
that is exactly the camera without a pavement sensor we want to turn into one.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .labels import CLASSES
from .vision import PHYSICS_FEATURES

log = logging.getLogger(__name__)

WEATHER_FEATURES: tuple[str, ...] = (
    "air_f", "dew_f", "dew_spread_f", "rh", "wind_kt", "vis_mi", "precip_in",
    "wx_snow", "wx_rain", "wx_freezing", "wx_fog",
)
HISTORY_FEATURES: tuple[str, ...] = (
    "precip_12h", "snow_12h", "freezing_12h", "min_air_12h", "hours_frozen_12h", "snow_24h", "max_air_24h",
)
ZERO_SHOT: tuple[str, ...] = ("zs_dry", "zs_wet", "zs_frozen")
CONFIDENCE_THRESHOLD = 0.8


# ---------------------------------------------------------------- data
def load_xy(path: str | Path) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pl.read_parquet(path)
    need = (*WEATHER_FEATURES, *HISTORY_FEATURES, *PHYSICS_FEATURES, *ZERO_SHOT, "emb", "label", "station")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"features file is missing columns: {missing}")
    df = df.filter(pl.col("label").is_in(CLASSES))
    y = np.array([CLASSES.index(v) for v in df["label"].to_list()])
    groups = df["station"].to_numpy()
    emb = np.asarray(df["emb"].to_list(), dtype=np.float32)
    return df, y, groups, emb


def tabular(df: pl.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def center_by_camera(x: np.ndarray, groups: np.ndarray, scale: bool = False) -> np.ndarray:
    """Remove each camera's own average look (angle, background, white balance).

    Uses only unlabeled frames of that camera, so it is available for a brand new camera
    after a few days of footage. This is the main lever against camera identity leaking
    into the classifier.
    """
    out = np.array(x, dtype=np.float64, copy=True)
    for g in np.unique(groups):
        i = groups == g
        mu = np.nanmean(out[i], axis=0)
        out[i] -= mu
        if scale:
            out[i] /= np.nanstd(out[i], axis=0) + 1e-6
    return out


@dataclass(frozen=True)
class Blocks:
    emb: np.ndarray
    emb_c: np.ndarray
    phys_z: np.ndarray
    zs_log_c: np.ndarray
    zs_log: np.ndarray
    weather: np.ndarray


def blocks(df: pl.DataFrame, emb: np.ndarray, groups: np.ndarray) -> Blocks:
    zs_log = np.log(tabular(df, ZERO_SHOT) + 1e-6)
    return Blocks(
        emb=emb,
        emb_c=center_by_camera(emb, groups),
        phys_z=center_by_camera(tabular(df, PHYSICS_FEATURES), groups, scale=True),
        zs_log_c=center_by_camera(zs_log, groups),
        zs_log=zs_log,
        weather=np.hstack([tabular(df, WEATHER_FEATURES), tabular(df, HISTORY_FEATURES)]),
    )


# ---------------------------------------------------------------- models
def weather_rules(df: pl.DataFrame) -> np.ndarray:
    """What a dispatcher does without cameras: read the nearest airport observation."""
    out = []
    for r in df.select(WEATHER_FEATURES).iter_rows(named=True):
        air = r["air_f"] if r["air_f"] is not None else 40.0
        moist = (r["precip_in"] or 0) > 0 or (r["wx_snow"] or 0) or (r["wx_rain"] or 0) or (r["wx_freezing"] or 0)
        humid = (r["rh"] or 0) >= 90
        if air <= 32 and (moist or humid or (r["wx_fog"] or 0)):
            out.append(2)
        elif moist or humid:
            out.append(1)
        else:
            out.append(0)
    return np.array(out)


def _gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=250, class_weight="balanced", random_state=0)


def _probe() -> Pipeline:
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000))])


# name -> (feature builder, estimator factory). Every learned model is judged the same way.
MODELS = {
    "image_probe": (lambda b: b.emb, _probe),
    "image_probe_centered": (lambda b: b.emb_c, _probe),
    "weather_gbm": (lambda b: b.weather, _gbm),
    "fused_gbm": (lambda b: np.hstack([b.weather, b.zs_log_c, b.phys_z]), _gbm),
}


# ---------------------------------------------------------------- metrics
def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    fz = CLASSES.index("frozen")
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "frozen_recall": round(float(recall_score(y == fz, pred == fz, zero_division=0)), 4),
        "frozen_precision": round(float(precision_score(y == fz, pred == fz, zero_division=0)), 4),
    }


def cross_val_proba(factory, X: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    proba = np.zeros((len(y), len(CLASSES)))
    for tr, te in splits:
        m = factory().fit(X[tr], y[tr])
        p = np.zeros((len(te), len(CLASSES)))
        p[:, m.classes_] = m.predict_proba(X[te])
        proba[te] = p
    return proba


def evaluate(df: pl.DataFrame, y: np.ndarray, groups: np.ndarray, emb: np.ndarray) -> dict:
    b = blocks(df, emb, groups)
    logo = list(LeaveOneGroupOut().split(emb, y, groups))
    preds: dict[str, np.ndarray] = {"weather_rules": weather_rules(df)}
    if df["zs_dry"].std() > 0:
        preds["zero_shot"] = b.zs_log.argmax(1)
    probas = {}
    for name, (feat, factory) in MODELS.items():
        probas[name] = cross_val_proba(factory, feat(b), y, logo)
        preds[name] = probas[name].argmax(1)
    results = {k: metrics(y, p) for k, p in preds.items()}

    # Same models under a random split, to show how much camera leakage flatters the score.
    random_split = {}
    n_splits = min(5, int(np.bincount(y, minlength=3).min()))
    if n_splits >= 2:
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(emb, y))
        for name in ("image_probe", "fused_gbm"):
            feat, factory = MODELS[name]
            random_split[name] = metrics(y, cross_val_proba(factory, feat(b), y, folds).argmax(1))

    best = max(MODELS, key=lambda k: (results[k]["macro_f1"], results[k]["frozen_recall"]))
    conf = probas[best].max(1)
    keep = conf >= CONFIDENCE_THRESHOLD
    triage = {"model": best, "threshold": CONFIDENCE_THRESHOLD, "coverage": round(float(keep.mean()), 4)}
    if keep.any():
        triage.update({f"confident_{k}": v for k, v in metrics(y[keep], probas[best].argmax(1)[keep]).items()})
    per_site = {g: {"n": int((groups == g).sum()), **metrics(y[groups == g], preds[best][groups == g])}
                for g in sorted(set(groups))}
    return {
        "n_frames": int(len(y)),
        "n_sites": int(len(set(groups))),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "models": results,
        "random_split": random_split,
        "best_model": best,
        "triage": triage,
        "per_site": per_site,
        "confusion": {"model": best, "labels": list(CLASSES),
                      "matrix": confusion_matrix(y, preds[best], labels=[0, 1, 2]).tolist()},
        "_oof": {k: v for k, v in probas.items()},
    }


# ---------------------------------------------------------------- entry points
def camera_reference(df: pl.DataFrame, groups: np.ndarray, camera_ids: list[str]) -> dict[str, dict]:
    """Per camera means and spreads used by center_by_camera, stored with the model so a
    known camera can be scored one frame at a time."""
    zs_log = np.log(tabular(df, ZERO_SHOT) + 1e-6)
    phys = tabular(df, PHYSICS_FEATURES)
    ref = {}
    for g, cam in zip(groups, camera_ids, strict=True):
        if cam in ref:
            continue
        i = groups == g
        ref[cam] = {
            "zs_log_mean": np.nanmean(zs_log[i], 0).tolist(),
            "phys_mean": np.nanmean(phys[i], 0).tolist(),
            "phys_std": (np.nanstd(phys[i], 0) + 1e-6).tolist(),
            "n_frames": int(i.sum()),
        }
    return ref


def _log_mlflow(settings: Settings, report: dict, params: dict, artifacts: list[Path]) -> str | None:
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed, skipping tracking")
        return None
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment("pavecam")
    with mlflow.start_run(run_name="loso-eval") as run:
        mlflow.log_params(params)
        for model, ms in report["models"].items():
            mlflow.log_metrics({f"loso.{model}.{k}": v for k, v in ms.items()})
        for model, ms in report["random_split"].items():
            mlflow.log_metrics({f"random.{model}.{k}": v for k, v in ms.items()})
        mlflow.log_metrics({f"triage.{k}": v for k, v in report["triage"].items() if isinstance(v, int | float)})
        for a in artifacts:
            mlflow.log_artifact(str(a))
        mlflow.set_tag("best_model", report["best_model"])
        return run.info.run_id


def run_training(features: str | None = None, out_dir: str | None = None, settings: Settings | None = None,
                 track: bool = True) -> dict:
    s = settings or Settings()
    feats = Path(features) if features else s.processed_dir / "features.parquet"
    out = Path(out_dir) if out_dir else s.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    df, y, groups, emb = load_xy(feats)
    if len(set(groups)) < 3:
        raise ValueError("need at least 3 sites for leave-one-site-out evaluation")
    log.info("training on %d frames from %d sites", len(y), len(set(groups)))
    report = evaluate(df, y, groups, emb)
    oof = report.pop("_oof")
    encoder = df["encoder"][0] if "encoder" in df.columns else "unknown"
    report["encoder"] = encoder
    best = report["best_model"]

    feat, factory = MODELS[best]
    final = factory().fit(feat(blocks(df, emb, groups)), y)
    bundle = {
        "model_name": best, "model": final, "classes": list(CLASSES), "encoder": encoder,
        "weather_features": list(WEATHER_FEATURES), "history_features": list(HISTORY_FEATURES),
        "physics_features": list(PHYSICS_FEATURES), "confidence_threshold": CONFIDENCE_THRESHOLD,
        "camera_reference": (
            camera_reference(df, groups, df["camera_id"].to_list()) if "camera_id" in df.columns else {}
        ),
    }
    model_path = out / "pavecam_model.joblib"
    joblib.dump(bundle, model_path)
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    oof_path = out / "oof_predictions.parquet"
    cols = [c for c in ("station", "camera_id", "frame_time", "frame_file", "label") if c in df.columns]
    df.select(cols).with_columns(
        *[pl.Series(f"pred_{k}", [CLASSES[i] for i in v.argmax(1)]) for k, v in oof.items()],
        pl.Series("confidence", oof[best].max(1).round(4)),
    ).write_parquet(oof_path)
    if track:
        params = {"encoder": encoder, "n_frames": report["n_frames"], "n_sites": report["n_sites"],
                  "cv": "leave-one-site-out", "best_model": best}
        report["mlflow_run_id"] = _log_mlflow(s, report, params, [metrics_path, oof_path, model_path])
        metrics_path.write_text(json.dumps(report, indent=2))
    return {k: report[k] for k in ("n_frames", "n_sites", "class_counts", "models", "random_split", "best_model",
                                   "triage")}


def smoke() -> dict:
    """Offline end to end check on the bundled real sample (features from real frames)."""
    sample = Path(__file__).resolve().parents[2] / "data" / "sample" / "features_sample.parquet"
    if not sample.exists():
        raise FileNotFoundError(sample)
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(data_dir=Path(tmp), mlflow_uri=f"sqlite:///{tmp}/mlflow.db")
        res = run_training(str(sample), str(Path(tmp) / "artifacts"), settings=s)
        if not (Path(tmp) / "artifacts" / "pavecam_model.joblib").exists():
            raise RuntimeError("smoke run did not produce a model bundle")
    return res
