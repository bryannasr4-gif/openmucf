.PHONY: install test lint format findings calibration validate bench forecast twin-audit materiality audit all

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
	python -c "from openmucf import validate, load_rates; r=load_rates(); open('VALIDATION.md','w').write(validate.report_markdown(validate.run(r)))"
	python -c "from openmucf import validate, load_rates; r=load_rates(); open('VALIDATION_CHANNELS.md','w').write(validate.report_markdown(validate.run(r, channels='on'), channels='on'))"
	@echo "wrote VALIDATION.md + VALIDATION_CHANNELS.md"

bench:
	python -c "from openmucf import bench, load_rates; r=load_rates(); open('BENCHMARKS.md','w').write(bench.report_markdown(bench.run_all(r)))"
	@echo "wrote BENCHMARKS.md"

forecast:
	python scripts/generate_forecast.py

twin-audit:
	python scripts/generate_twin_audit.py

materiality:
	python scripts/generate_materiality.py

# Reproducibility gate: regenerate the deterministic docs and fail if they drift from what's committed.
# CALIBRATION.md and the FC-001 card payload (forecasts/FC-001-mufuse.json) are MCMC-derived and are NOT
# exact-diffed here; instead the card is checked for hash-consistency and FORECASTS.md (rendered
# deterministically from the on-disk card, no MCMC) IS exact-diffed. `--audit` runs both without the MCMC.
# TWIN_AUDIT.md is deterministic (its section-3 bands are the FC-001 card-interval envelope, no MCMC) and
# IS exact-diffed; the slow twin coverage MCMC (tests/test_twin_coverage.py) is a `slow` test, never here.
# MATERIALITY.md is deterministic (one-at-a-time channel toggles through the v1 ODE, no MCMC) and IS
# exact-diffed; its forward-UQ CI-width scale reference is read from the byte-stable FINDINGS_MANIFEST.json.
audit: findings validate bench twin-audit materiality
	python scripts/generate_forecast.py --audit
	python -m openmucf.provenance --check FINDINGS_MANIFEST.json TWIN_MANIFEST.json MATERIALITY_MANIFEST.json
	git diff --exit-code -- FINDINGS.md VALIDATION.md VALIDATION_CHANNELS.md FORECASTS.md FINDINGS_MANIFEST.json BENCHMARKS.md TWIN_AUDIT.md TWIN_MANIFEST.json MATERIALITY.md MATERIALITY_MANIFEST.json
	python scripts/generate_calibration.py --audit
	@echo "audit OK: docs match committed; manifest verified; FC-001 card hash-consistent"

all: lint test findings calibration forecast
