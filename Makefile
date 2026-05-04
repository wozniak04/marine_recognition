setup:
	uv sync

preprocess:
	uv run -m src.cli.preprocess_data --dataset "data/raw/unclassified/Ship_Operation_example_dataset_unclassified_source_GPS_1.csv"
preprocess2:
	uv run -m src.cli.preprocess_data --dataset "data/raw/classified/Ship_Operation_example_dataset_classified_2.csv"

exp:
	uv run -m src.cli.exp
