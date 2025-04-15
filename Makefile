.PHONY: help check install test install-uv run dev clean download-data clean-data delete-data

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          	Show this help message"
	@echo "  install-uv    	Install the uv package manager (required)"
	@echo "  install       	Install the package and its dependencies"
	@echo "  test          	Run the tests"
	@echo "  download-data  	Download network files from Wireshark samples"
	@echo "  clean-data N=<number>	Keep network files with a maximum of <number> packets"
	@echo "  delete-data   	Delete all network files"
	@echo "  run           	Run the program"
	@echo "  dev           	Create a development environment"
	@echo "  clean         	Remove build artifacts"
	@echo "  check 	    	Check the code for syntax errors"

install-uv:
	wget -qO- https://astral.sh/uv/install.sh | sh

check:
	python3 -m py_compile netexplainer/*.py

install:
	uv run pip install .

test:
	uv run pytest

run:
	uv run python3 -m netexplainer

download-data:
	uv run python3 -m netexplainer --download-data

clean-data:
ifndef N
	@echo "Error: The max packet number should be specified with N=<number>"
	@exit 1
else
	uv run python3 -m netexplainer --clean-data $(N)
endif

delete-data:
	rm -rf netexplainer/data/raw/* netexplainer/data/cleaned/*

dev:
	uv venv dev
	. dev/bin/activate
	uv run pip install -e .[dev]

clean:
	rm -rf *.egg-info/ .pytest_cache/ __pycache__/ build/ dist/ netexplainer/__pycache__/ tests/__pycache__/
