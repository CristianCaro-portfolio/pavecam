"""Figures for the README: architecture, model comparison, and where camera and sensor disagree."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from .config import Settings  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

LABELS = {
    "zero_shot": "Image, zero-shot prompts",
    "image_probe": "Image, linear probe",
    "image_probe_centered": "Image, probe + camera centering",
    "physics_gbm": "Image, physical cues",
    "weather_rules": "Airport weather, rules",
    "weather_gbm": "Airport weather + 24h history",
    "fused_gbm": "Weather + camera cues",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def results_chart(metrics: dict, out: Path) -> None:
    models = [m for m in LABELS if m in metrics["models"]]
    models.sort(key=lambda m: metrics["models"][m]["macro_f1"])
    vals = [metrics["models"][m]["macro_f1"] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160, facecolor=SURFACE)
    _style(ax)
    y = range(len(models))
    ax.barh(list(y), vals, height=0.55, color=BLUE, label="Unseen camera (leave one site out)")
    for i, m in enumerate(models):
        ax.text(vals[i] - 0.012, i, f"{vals[i]:.2f}", va="center", ha="right", color="white", fontsize=10, weight="bold")
        rnd = metrics.get("random_split", {}).get(m)
        if rnd:
            ax.plot([vals[i], rnd["macro_f1"]], [i, i], color=ORANGE, linewidth=2, linestyle=":")
            ax.scatter([rnd["macro_f1"]], [i], s=70, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=2,
                       label="Same cameras in train and test (random split)" if m == "image_probe" else None)
            ax.text(rnd["macro_f1"] + 0.012, i + 0.28, f"{rnd['macro_f1']:.2f}", color=INK2, fontsize=9)
    ax.set_yticks(list(y), [LABELS[m] for m in models], color=INK, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 against the pavement sensor (dry / wet / frozen)", color=INK2)
    ax.set_title(
        f"Can a road camera replace a pavement sensor?  {metrics['n_frames']:,} frames, {metrics['n_sites']} Iowa sites",
        loc="left", color=INK, fontsize=12.5, pad=14)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def disagreement_sheet(features: pl.DataFrame, frames_dir: Path, out: Path, n: int = 4) -> None:
    """Top row: sensor says frozen, camera sees bare pavement (frost, black ice risk).
    Bottom row: sensor says dry, camera sees a snowy scene (snow on the shoulders only)."""
    a = features.filter(pl.col("label") == "frozen").sort("zs_frozen").head(60).sample(n, seed=3)
    b = features.filter(pl.col("label") == "dry").sort("zs_frozen", descending=True).head(60).sample(n, seed=3)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.4), dpi=100, facecolor=SURFACE)
    for r, (sub, title) in enumerate(((a, "Sensor: frozen"), (b, "Sensor: dry"))):
        for k, row in enumerate(sub.iter_rows(named=True)):
            ax = axes[r][k]
            ax.imshow(Image.open(frames_dir / row["frame_file"]).convert("RGB"))
            ax.set_axis_off()
            ax.set_title(f"{title} | camera says frozen {row['zs_frozen']:.0%}\n{row['station']}  air {row['air_f']:.0f}F",
                         fontsize=8, color=INK)
    fig.suptitle("Where the camera and the pavement sensor disagree", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=160, facecolor=SURFACE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.set_axis_off()

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=color, linewidth=1.6))
        ax.text(x + 0.12, y + h - 0.22, title, fontsize=9.5, color=INK, weight="bold", va="top")
        ax.text(x + 0.12, y + h - 0.58, body, fontsize=8, color=INK2, va="top", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))

    box(0.1, 2.75, 2.3, 1.3, "RWIS pavement sensors", "Iowa DOT, 5 min\nDry / Wet / Ice Warning\n= free labels", BLUE)
    box(0.1, 1.40, 2.3, 1.2, "Roadside cameras", "Same sites, about 4 min\narchived by IEM", BLUE)
    box(0.1, 0.10, 2.3, 1.15, "Airport weather (ASOS)", "Nearest METAR\ntemp, dew point, wx codes", BLUE)
    box(2.9, 1.15, 2.4, 2.1, "Ingest flow (Prefect)",
        "site quality gate\nevent day sampling\nas-of joins (Polars)\n+-10 min labels and frames\nretries, disk cache", ORANGE)
    box(5.8, 2.35, 2.6, 1.7, "Image features",
        "MobileCLIP2-S0 on CPU\nembeddings + zero-shot\nphysical cues (glare,\nwhiteness, texture)", ORANGE)
    box(5.8, 0.25, 2.6, 1.85, "Weather features",
        "now: temp, dew spread, wx\nlast 12 to 24 h: snow,\nprecip, hours below 32F", ORANGE)
    box(8.9, 1.15, 3.0, 2.1, "Train and evaluate",
        "leave one site out\ncamera centering\nrandom split as a leak check\nMLflow run + model bundle\n-> pavecam predict", BLUE)
    for yy in (3.4, 2.0, 0.7):
        arrow(2.4, yy, 2.9, 2.2)
    arrow(5.3, 2.5, 5.8, 3.1)
    arrow(5.3, 1.8, 5.8, 1.2)
    arrow(8.4, 3.1, 8.9, 2.5)
    arrow(8.4, 1.2, 8.9, 1.8)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def build_report(settings: Settings | None = None, docs: Path = Path("docs")) -> None:
    s = settings or Settings()
    docs.mkdir(exist_ok=True)
    architecture(docs / "architecture.png")
    metrics_path = s.artifacts_dir / "metrics.json"
    if metrics_path.exists():
        results_chart(json.loads(metrics_path.read_text()), docs / "results.png")
    feats = s.processed_dir / "features.parquet"
    if feats.exists() and s.frames_dir.exists():
        disagreement_sheet(pl.read_parquet(feats), s.frames_dir, docs / "disagreements.png")
