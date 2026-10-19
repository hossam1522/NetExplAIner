.PHONY: help install-uv install test lint format check run analyze download-data clean-data delete-data clean

ARGS ?=

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install-uv              Install the uv package manager (required)"
	@echo "  install                 Install the package and its dependencies"
	@echo "  test                    Run the tests"
	@echo "  lint                    Run ruff (lint + format check)"
	@echo "  format                  Format the code with ruff"
	@echo "  download-data           Download network files from Wireshark samples"
	@echo "  clean-data N=<number>   Keep network files with a maximum of <number> packets"
	@echo "  delete-data             Delete all network files"
	@echo "  run ARGS='...'          Run the evaluation (see: uv run python -m netexplainer run --help)"
	@echo "  analyze ARGS='...'      Analyze result files (JSONL or legacy logs)"
	@echo "  clean                   Remove build artifacts"

install-uv:
	wget -qO- https://astral.sh/uv/install.sh | sh

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint test

run:
	uv run python -m netexplainer run $(ARGS)

analyze:
	uv run python -m netexplainer analyze $(ARGS)

download-data:
	uv run python -m netexplainer download-data

clean-data:
ifndef N
	@echo "Error: The max packet number should be specified with N=<number>"
	@exit 1
else
	uv run python -m netexplainer clean-data $(N)
endif

delete-data:
	rm -rf netexplainer/data/raw/* netexplainer/data/cleaned/*

clean:
	rm -rf *.egg-info/ .pytest_cache/ .ruff_cache/ build/ dist/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
