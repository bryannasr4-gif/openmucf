"""openmucf.constants -- single home for cross-module physical constants.

The rate ledger (openmucf/data/rates.csv) is the source of truth; this module re-exports the three
constants that multiple engine modules need at import time, read once from the ledger. A broken ledger
therefore fails fast, at import. Values (and why they are safe to load eagerly): lambda_mu_decay is
PDG-settled; E_fusion is exact; E_mu_cost is the documented 2-10 GeV design-study default (5.0 GeV).
"""

from .rates import load_rates

_R = load_rates(check_refs=False)  # skip the bib regex on the hot import path

LAMBDA_0 = _R.value("lambda_mu_decay")   # s^-1  (= 4.552e5)
E_F_MEV = _R.value("E_fusion")           # MeV   (= 17.6)
E_MU_GEV_DEFAULT = _R.value("E_mu_cost")  # GeV   (= 5.0)
