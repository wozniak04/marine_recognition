setup:
	uv sync

preprocess:
	uv run -m src.cli.preprocess_data --config configs/test.yaml