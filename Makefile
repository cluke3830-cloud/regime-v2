.PHONY: help install test validate clean lint

PYTHON ?= python

help:
	@echo "Regime_v2 — common targets"
	@echo ""
	@echo "  make install    Install dependencies into the current Python env"
	@echo "  make test       Run the full test suite (124 tests, ~7s)"
	@echo "  make validate   Regenerate validation_report.md on real SPY data"
	@echo "                  Requires FRED_API_KEY env var. First run: ~15 min."
	@echo "                  Cached runs: ~3 min."
	@echo "  make clean      Drop pytest + Python caches (keeps data/cache/)"
	@echo "  make wipe       Drop pytest caches AND data/cache/ (forces yfinance re-fetch)"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -q

validate:
	@if [ -z "$$FRED_API_KEY" ]; then \
	  echo "ERROR: FRED_API_KEY env var is required."; \
	  echo "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"; \
	  exit 1; \
	fi
	$(PYTHON) -W ignore scripts/make_validation_report.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

wipe: clean
	rm -rf data/cache/*.parquet