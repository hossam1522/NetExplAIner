.PHONY: help check install

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          	Show this help message"
	@echo "  install       	Install the package"
	@echo "  check 	    	Check the code for syntax errors"

check:
	python3 -m py_compile netexplainer/*.py

install:
	uv build && uv lock
