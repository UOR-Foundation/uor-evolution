PYTHON ?= python3
VENV ?= .venv

.PHONY: install api test run clean lint

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

api:
	$(VENV)/bin/python demo_unified_api.py

test:
	$(VENV)/bin/pytest -vv

run:
	$(VENV)/bin/python backend/app.py

lint:
	$(VENV)/bin/flake8 backend core modules tests || true

clean:
	rm -rf $(VENV) **/__pycache__ .pytest_cache
