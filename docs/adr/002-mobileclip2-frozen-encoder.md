# ADR 002: Frozen MobileCLIP2 encoder on CPU instead of fine tuning a CNN

Status: accepted

## Context

The project must run on a laptop CPU with no GPU and no paid API. Recent papers fine tune CNNs
or vision transformers on road images.

## Decision

Use MobileCLIP2-S0 through open_clip as a frozen encoder. One forward pass gives a 512 dimension
embedding for a linear probe and, through text prompts, a zero-shot score per class. Add ten
hand made physical cues (whiteness, glare, saturation, lane texture) that are cheap and
explainable. Feed the full frame resized to 256 by 256 instead of a center crop, because the
instrumented lane is often near the edge.

## Why

* About 80 ms per frame on 2 CPU cores, 2,545 frames in under 4 minutes.
* Zero-shot gives a baseline with no training at all, which shows what the foundation model
  knows before any site data.
* With 16 sites, fine tuning a large network would mostly memorize cameras. A frozen encoder plus
  a regularized probe, combined with per camera centering, is the safer bet and is measurable.

## Alternatives considered

* SigLIP2 or DINOv3 base models: better features but 4 to 10 times slower on CPU.
* Fine tuned ResNet or EfficientNet: needs a GPU to iterate and overfits camera identity.
