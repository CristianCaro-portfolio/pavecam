.PHONY: install test lint smoke all

install:
	pip install -e ".[vision,dev]"

lint:
	ruff check src tests

test:
	pytest

smoke:
	pavecam smoke

all:
	pavecam ingest
	pavecam featurize
	pavecam train
	pavecam report
