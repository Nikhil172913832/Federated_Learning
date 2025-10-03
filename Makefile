PYTHON=python
PIP=pip

.PHONY: install test run-local lint build simulate

install:
	cd complete/fl && $(PIP) install -e .

test:
	cd complete/fl && pytest -q

simulate:
	cd complete/fl && flwr run .

run-local: simulate

lint:
	cd complete/fl && python -m flake8 . && python -m black --check .

build:
	cd complete/fl && python -m build


