.PHONY: install test lint format findings calibration validate bench forecast twin-audit materiality mucost systems frontier neutronomics design audit all

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

# Neutrons-per-joule league table (neutronomics Layer 1). Regenerates NEUTRONOMICS.md +
# NEUTRONOMICS_MANIFEST.json. All committed numbers are pure deterministic arithmetic (X_mu from the
# ledger + the MUON_COST tier medians + published beam parameters; no MCMC/solver), so BOTH artifacts are
# byte-stable cross-arch and BOTH join the audit git-diff list below.
neutronomics:
	python scripts/generate_neutronomics.py

# Bayesian experimental-design ranking (WS-D). Regenerates DESIGN.md + DESIGN_MANIFEST.json. BOTH carry
# numpyro/NUTS-derived numbers (nested-MC EIG, sd-contraction refits) that are NOT byte-stable cross-arch
# (the CALIBRATION.md precedent, WAVE2 A1), so NEITHER joins the audit git-diff list below; instead
# `generate_design.py --audit` re-runs with pinned seeds and tolerance-checks every manifest number: each
# cell -- EIG and sd-contraction alike -- against 4 sigma of its OWN published Monte-Carlo SE, plus
# structural gates on the ranking claims (see the AMENDMENTs in scripts/generate_design.py). The
# manifest still joins `provenance --check` (doc<->manifest regenerate together). Runs ~10 min
# (n_synth=64 NUTS refits per class-candidate, plus 20 replicate base chains that measure the EIG SEs);
# it is the slowest step of `make audit`.
design:
	python scripts/generate_design.py

# The G4MuonicData datasets: the format example (data/g4/) and the D1 parity build (data/g4/d1/).
# All seven generated artifacts join the audit git-diff list below, but they are NOT all the same
# kind of artifact and the differences matter:
#   example.g4dat            -- pure deterministic text rendered from the hand-authored Layer-2 file
#                               (no MCMC/solver), byte-stable on every platform.
#   d1_*.g4dat, d1_*.prov.json -- BOTH layers generated from the vendored upstream Geant4 source
#                               (third_party/geant4/), parsed at build time. Pure deterministic text
#                               and byte-stable on every platform. Nothing here is hand-authored, so
#                               a drift in these bytes means the extraction or the source moved.
#   geant4_add_dataset.snippet -- deterministic text EXCEPT for one field: MD5SUM is the MD5 of the
#                               archive, i.e. of a gzip DEFLATE stream. zlib does not guarantee
#                               byte-identical compressed output across builds, so this line is
#                               stable for a given zlib, not by construction (FORMAT_SPEC.md 8).
#                               Measured identical on Windows/x86-64, ubuntu/x86-64 and macOS/arm64;
#                               a runner-image zlib change would red this target for a reason that
#                               has nothing to do with the data. Re-run `make g4data` and recommit.
# The .tar.gz archives are build products and are NOT committed; their determinism is proven by
# tests/test_g4spec.py and their MD5s are written into the snippets.
#
# NOT regenerated and NOT byte-diffed: data/g4/d1/d1_gp_sweep.oracle. It was HARVESTED from a
# Geant4-linked binary, so "regenerate and compare" is undefined for it -- no code here can produce
# it, which is exactly what makes it evidence rather than a restatement of our own output. It is
# guarded by RE-DERIVATION instead (tests/test_g4parity.py recomputes every value in Python and
# compares), which is a stronger check than a byte-diff because it re-derives rather than re-reads.
g4data:
	python scripts/generate_g4data.py

# Reproducibility gate: regenerate the deterministic docs and fail if they drift from what's committed.
# CALIBRATION.md and the FC-001 card payload (forecasts/FC-001-mufuse.json) are MCMC-derived and are NOT
# exact-diffed here; instead the card is checked for hash-consistency and FORECASTS.md (rendered
# deterministically from the on-disk card, no MCMC) IS exact-diffed. `--audit` runs both without the MCMC.
# TWIN_AUDIT.md is deterministic (its section-3 bands are the FC-001 card-interval envelope, no MCMC) and
# IS exact-diffed; the slow twin coverage MCMC (tests/test_twin_coverage.py) is a `slow` test, never here.
# MATERIALITY.md is deterministic (one-at-a-time channel toggles through the v1 ODE, no MCMC) and IS
# exact-diffed; its forward-UQ CI-width scale reference is read from the byte-stable FINDINGS_MANIFEST.json.
audit: findings validate bench twin-audit materiality mucost systems frontier neutronomics g4data
	python scripts/generate_forecast.py --audit
	python -m openmucf.provenance --check FINDINGS_MANIFEST.json TWIN_MANIFEST.json MATERIALITY_MANIFEST.json MUON_COST_MANIFEST.json SYSTEMS_MANIFEST.json FRONTIER_MANIFEST.json NEUTRONOMICS_MANIFEST.json DESIGN_MANIFEST.json
	git diff --exit-code -- FINDINGS.md VALIDATION.md VALIDATION_CHANNELS.md FORECASTS.md FINDINGS_MANIFEST.json BENCHMARKS.md TWIN_AUDIT.md TWIN_MANIFEST.json MATERIALITY.md MATERIALITY_MANIFEST.json MUON_COST.md MUON_COST_MANIFEST.json SYSTEMS.md SYSTEMS_MANIFEST.json FRONTIER.md FRONTIER_MANIFEST.json NEUTRONOMICS.md NEUTRONOMICS_MANIFEST.json data/g4/example.g4dat data/g4/geant4_add_dataset.snippet data/g4/d1/d1_capture.g4dat data/g4/d1/d1_capture.prov.json data/g4/d1/d1_zeff.g4dat data/g4/d1/d1_zeff.prov.json data/g4/d1/geant4_add_dataset.snippet
	python scripts/generate_g4data.py --audit
	python scripts/generate_calibration.py --audit
	python scripts/generate_design.py --audit
	@echo "audit OK: docs match committed; manifests verified; FC-001 card hash-consistent; NUTS docs tolerance-audited"

all: lint test findings calibration forecast
