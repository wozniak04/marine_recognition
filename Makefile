.PHONY: setup lint format typecheck test preprocess train infer evaluate export compare clean

CONFIG ?= configs/base.yaml
MODEL ?= models/lstm_gps_v1
INPUT ?=
OUTPUT ?=

setup:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run pyright src/

test:
	uv run pytest -v

preprocess:
	uv run python -m src.cli.preprocess --config $(CONFIG)

train:
	uv run python -m src.cli.train --config $(CONFIG)

infer:
	uv run python -m src.cli.infer --model $(MODEL) --input $(INPUT) $(if $(OUTPUT),--output $(OUTPUT),) $(if $(CONFIG:configs/base.yaml=),--config $(CONFIG),)

evaluate:
	uv run python -m src.cli.evaluate --config $(CONFIG)

export:
	uv run python -m src.cli.export_csv --config $(CONFIG)

compare:
	uv run python -m src.cli.compare

clean:
	rm -rf data/processed/* models/* reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
