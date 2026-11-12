# ADR 003: Prefect for ingestion, MLflow for runs

Status: accepted

## Context

Ingestion makes several thousand HTTP calls to a public university server (listings and
frames). Training compares six models and must be reproducible and comparable across reruns.

## Decision

* Ingestion and featurization are Prefect 3 flows. Network calls are Prefect tasks with retries,
  submitted to a thread pool of 4 workers. The pure data logic lives in `dataset.py` and is tested
  without Prefect.
* Every training run logs parameters, leave one site out and random split metrics, out of fold
  predictions and the model bundle to MLflow, with a local SQLite backend by default.

## Why

* Prefect gives retries, concurrency limits and run history without a separate scheduler, and
  runs locally with a temporary server. The same flow can be deployed on a schedule later.
* The IEM client adds tenacity retries and a disk cache under the tasks, so a rerun costs no
  network at all and a single bad frame does not fail the flow.
* MLflow keeps the evidence behind every number in the README. SQLite needs no server and
  `mlflow ui` opens it directly.

## Alternatives considered

* Dagster: strong asset model, already used in another project of mine; Prefect is lighter for
  a mostly imperative ingest with fan out.
* Plain scripts with a CSV of metrics: no retries, no run history, easy to lose track of which
  run produced which number.
