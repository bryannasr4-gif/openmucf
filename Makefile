.PHONY: install test lint format findings calibration validate forecast audit all

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

findings:
	python scripts/generate_findings.py

calibration:
	python scripts/generate_calibration.py

validate:
	python -c "from openmucf import validate, load_rates; open('VALIDATION.md','w').write(validate.report_markdown(validate.run(load_rates())))"
	@echo "wrote VALIDATION.md"

forecast:
	python scripts/generate_forecast.py

# Reproducibility gate: regenerate the deterministic docs and fail if they drift from what's committed.
# CALIBRATION.md and the FC-001 card payload (forecasts/FC-001-mufuse.json) are MCMC-derived and are NOT
# exact-diffed here; instead the card is checked for hash-consistency and FORECASTS.md (rendered
# deterministically from the on-disk card, no MCMC) IS exact-diffed. `--audit` runs both without the MCMC.
audit: findings validate
	python scripts/generate_forecast.py --audit
	python -m openmucf.provenance --check FINDINGS_MANIFEST.json
	git diff --exit-code -- FINDINGS.md VALIDATION.md FORECASTS.md FINDINGS_MANIFEST.json
	python scripts/generate_calibration.py --audit
	@echo "audit OK: docs match committed; manifest verified; FC-001 card hash-consistent"

all: lint test findings calibration forecast
