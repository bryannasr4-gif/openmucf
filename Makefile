.PHONY: install test lint format findings calibration validate bench forecast twin-audit materiality mucost systems frontier audit all

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

# materiality reads FINDINGS_MANIFEST.json (the forward-UQ CI width), so `findings` is a real
# prerequisite -- explicit, not just left-to-right list order, so `make -j` cannot race it.
materiality: findings
	python scripts/generate_materiality.py

# muon-cost ledger (WS-E). Regenerates MUON_COST.md + the 10^3-gap PNG + MUON_COST_MANIFEST.json.
# The PNG is NEVER byte-diffed (matplotlib/freetype bytes are not cross-platform stable); only the .md
# and the manifest join the audit git-diff list below. All committed numbers are pure arithmetic on the
# committed muon_cost.csv (no MCMC/solver), so the two byte-diffed artifacts are cross-arch stable.
mucost:
	python scripts/generate_mucost.py

# Q Rosetta stone + energy-balance graph (WS-S). Regenerates SYSTEMS.md + SYSTEMS_MANIFEST.json. All
# committed numbers are CLOSED-FORM algebra over openmucf.systems (a superset of the frozen
# energy.EnergyChain; no MCMC/solver), so both artifacts are byte-stable cross-arch and both join the
# audit git-diff list below.
systems:
	python scripts/generate_systems.py

# Inverse-design frontiers (WS-Q). Regenerates FRONTIER.md + FRONTIER_MANIFEST.json + the frontier PNG.
# FRONTIER.md + the manifest are CLOSED-FORM float64 (byte-stable cross-arch, like SYSTEMS.md) and join the
# audit git-diff list below; the solver inverses are cross-checked against those closed forms to <1e-9 in
# the tests, so no byte-diffed number depends on iterative-solver noise. The PNG (which draws the Kamimura
# MCMC posterior cloud) is NEVER byte-diffed.
frontier:
	python scripts/generate_frontier.py

# Reproducibility gate: regenerate the deterministic docs and fail if they drift from what's committed.
# CALIBRATION.md and the FC-001 card payload (forecasts/FC-001-mufuse.json) are MCMC-derived and are NOT
# exact-diffed here; instead the card is checked for hash-consistency and FORECASTS.md (rendered
# deterministically from the on-disk card, no MCMC) IS exact-diffed. `--audit` runs both without the MCMC.
# TWIN_AUDIT.md is deterministic (its section-3 bands are the FC-001 card-interval envelope, no MCMC) and
# IS exact-diffed; the slow twin coverage MCMC (tests/test_twin_coverage.py) is a `slow` test, never here.
# MATERIALITY.md is deterministic (one-at-a-time channel toggles through the v1 ODE, no MCMC) and IS
# exact-diffed; its forward-UQ CI-width scale reference is read from the byte-stable FINDINGS_MANIFEST.json.
audit: findings validate bench twin-audit materiality mucost systems frontier
	python scripts/generate_forecast.py --audit
	python -m openmucf.provenance --check FINDINGS_MANIFEST.json TWIN_MANIFEST.json MATERIALITY_MANIFEST.json MUON_COST_MANIFEST.json SYSTEMS_MANIFEST.json FRONTIER_MANIFEST.json
	git diff --exit-code -- FINDINGS.md VALIDATION.md VALIDATION_CHANNELS.md FORECASTS.md FINDINGS_MANIFEST.json BENCHMARKS.md TWIN_AUDIT.md TWIN_MANIFEST.json MATERIALITY.md MATERIALITY_MANIFEST.json MUON_COST.md MUON_COST_MANIFEST.json SYSTEMS.md SYSTEMS_MANIFEST.json FRONTIER.md FRONTIER_MANIFEST.json
	python scripts/generate_calibration.py --audit
	@echo "audit OK: docs match committed; manifest verified; FC-001 card hash-consistent"

all: lint test findings calibration forecast
