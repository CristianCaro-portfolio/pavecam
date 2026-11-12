"""Command line entry point: pavecam ingest | featurize | train | predict | report | smoke."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Quiet defaults for local runs; any value already set in the environment wins.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _encoder(kind: str):
    from .config import Settings
    from .vision import HashEncoder, MobileClipEncoder

    if kind == "hash":
        return HashEncoder()
    s = Settings()
    return MobileClipEncoder(s.clip_model, s.clip_pretrained)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pavecam")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download RWIS, ASOS and camera frames, build the manifest")
    f = sub.add_parser("featurize", help="physical cues, MobileCLIP2 embeddings and zero-shot scores")
    f.add_argument("--encoder", choices=["mobileclip", "hash"], default="mobileclip")
    t = sub.add_parser("train", help="leave-one-site-out evaluation, final model, MLflow run")
    t.add_argument("--features", default=None, help="features parquet (default data/processed/features.parquet)")
    t.add_argument("--out", default=None, help="artifacts dir (default data/artifacts)")
    pr = sub.add_parser("predict", help="virtual sensor reading for a camera, now or at a past time")
    pr.add_argument("--camera", required=True, help="IEM camera id, for example IDOT-030-01")
    pr.add_argument("--time", default=None, help="UTC time YYYY-MM-DDTHH:MM, default latest frame")
    pr.add_argument("--asos", default=None, help="ASOS station for weather, default nearest to camera site")
    sub.add_parser("report", help="figures for README and docs")
    sub.add_parser("smoke", help="offline end to end run on the bundled real sample")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd == "ingest":
        from .flows import ingest_flow

        print(ingest_flow())
    elif args.cmd == "featurize":
        from .flows import featurize_flow

        print(featurize_flow(_encoder(args.encoder)))
    elif args.cmd == "train":
        from .train import run_training

        print(json.dumps(run_training(args.features, args.out), indent=2))
    elif args.cmd == "predict":
        from .predict import predict_camera

        print(json.dumps(predict_camera(args.camera, args.time, args.asos), indent=2))
    elif args.cmd == "report":
        from .report import build_report

        build_report()
    elif args.cmd == "smoke":
        from .train import smoke

        print(json.dumps(smoke(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
