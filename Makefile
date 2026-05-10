.PHONY: setup preprocess train classify convert-ports clean

CONFIG ?= configs/rule_based.yaml
INPUT ?=
OUTPUT ?=
LAT ?= LAT
LON ?= LON
LOCODE ?= LOCODE
SEP ?=

setup:
	uv sync

preprocess:
	uv run python preprocess.py --config $(CONFIG) $(if $(INPUT),--input $(INPUT),)

train:
	uv run python train.py --config $(CONFIG)

classify:
	uv run python classify.py --config $(CONFIG) --input $(INPUT) $(if $(OUTPUT),--output $(OUTPUT),)

convert-ports:
	uv run python convert_ports.py --input $(INPUT) --output $(OUTPUT) --lat $(LAT) --lon $(LON) --locode $(LOCODE) $(if $(SEP),--sep $(SEP),)

clean:
	rm -rf data/preprocessed/* models/* reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
