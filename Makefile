.PHONY: help check install test install-uv run

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          	Show this help message"
	@echo "  install-uv    	Install the uv package manager (required)"
	@echo "  install       	Install the package and its dependencies"
	@echo "  test          	Run the tests"
	@echo "  run           	Run the program"
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
