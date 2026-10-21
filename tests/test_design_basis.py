"""DESIGN.md's Method and Basis paragraphs against the record they cite.

DESIGN.md ranks the next experiments over a calibration chain pinned in `openmucf.design`: the omega_s0, R
and lambda_c boxes calibrate used before they were widened, under `openmucf.calibrate.model`'s likelihood.
Its Method paragraph attributes that chain's omega_s0/R correlation to the prior-sensitivity sweep of
CALIBRATION.md as released in v1.3.0, and its Basis paragraph gives the correlation the corrected sweep
prints for the same omega_s0 and R boxes. Both figures are read here from the calibration record as the
calibration's gate admits it (the released print from ``shipped_v1_3_0``, the corrected one from the sweep
rows), independently of the generator's own reading, and every figure the two paragraphs print must be the
record's or the design's pins. The design basis check traces the models used by the base posterior and
refits and compares their prior supports and observed likelihood inputs with the published basis. No NUTS
runs.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from openmucf import calibrate, design

REPO = Path(__file__).resolve().parents[1]
ROW = "R=legacy/os0=legacy/lc=narrow"
RELEASED = "CALIBRATION.md|Prior-sensitivity sweep (weak-prior mode; 4 chains x 1000)|" + ROW + "|"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _cells():
    """The calibration record as its gate admits it, read from the repository root."""
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        return _load("scripts/generate_calibration.py", "_gen_calibration_basis").admit_calibration_cells()
    finally:
        os.chdir(cwd)


def _thaw(node):
    """A mutable copy of the admitted (read-only) record, for the drills."""
    if isinstance(node, Mapping):
        return {k: _thaw(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_thaw(v) for v in node]
    return node


def _sweep_row(data):
    rows = [r for r in data["sweep"] if r["config"] == ROW]
    return rows[0] if len(rows) == 1 else None


def _paragraphs(design_md: str) -> str:
    """The Method and Basis paragraphs of DESIGN.md, whitespace-normalised."""
    text = design_md.replace("\r\n", "\n")
    start = text.index("**Method (no new physics).**")
    basis = text.index("**Basis: the likelihood before the anchor-basis correction.**")
    end = text.find("\n\n", basis)
    return " ".join(text[start:end].split())


def _box(lo: float, hi: float, fmt: str) -> str:
    return f"({lo:{fmt}}, {hi:{fmt}})"


def basis_failures(design_md: str, data) -> list[str]:
    """Each figure the Method and Basis paragraphs must print and do not, from the record ``data`` (the
    admitted calibration cells) and the design's pins; and each mismatch between the boxes the cited sweep
    rows ran and the design's pinned omega_s0 and R boxes, which is what makes them the rows for "these
    boxes"."""
    par = _paragraphs(design_md)
    released_corr = data["shipped_v1_3_0"][RELEASED + "corr"]
    released_boxes = data["shipped_v1_3_0"][RELEASED + "boxes"]
    row = _sweep_row(data)
    if row is None:
        return [f"the record does not hold exactly one sweep row {ROW}"]
    pins = {"os0": (design.BASE_PRIOR[1], design.BASE_PRIOR[2]), "R": (design._R_LO, design._R_HI)}
    bad = []
    # the cited rows ran the design's omega_s0 and R boxes, released and corrected alike
    for name, (lo, hi) in pins.items():
        if f"{name}[{lo:.2f},{hi:.2f}]" not in released_boxes:
            bad.append(f"the released row's boxes {released_boxes!r} are not the design's {name} box")
    if (list(row["os0_box"]), list(row["R_box"])) != (list(pins["os0"]), list(pins["R"])):
        bad.append(f"the corrected sweep row's boxes {row['os0_box']}, {row['R_box']} are not the design's")
    edge = re.search(r"lc\[[^,\]]+,([^\]]+)\]", released_boxes)
    obs = design._OBS
    need = {
        "the design's boxes": (
            f"Uniform omega_s0 {_box(*pins['os0'], '.2f')}, Uniform R {_box(*pins['R'], '.2f')} and Uniform "
            f"lambda_c {_box(design._LAMBDA_C_LO, design._LAMBDA_C_HI, '.2g')} s^-1"
        ),
        "the design's likelihood": (
            f"the effective sticking {obs['omega_s_eff_obs']:g} +- {obs['omega_s_eff_sd']:g} % taken as the "
            f"whole loss per cycle and the yield {obs['xmu_obs']:g} +- {obs['xmu_sd']:g}, with no cycle-rate "
            "measurement"
        ),
        "the released correlation": (
            f"as released in v1.3.0 printed as +{released_corr} for these boxes (its row {ROW},"
        ),
        "the released box edge": f"printed at one significant figure as {edge.group(1) if edge else '?'})",
        "the corrected likelihood's yield": f"the yield carries its printed +-{calibrate.OBS_W['xmu_sd']:g}.",
        "the corrected sweep's box": f"cycle-rate box {_box(row['lc_box'][0], row['lc_box'][1], '.2g')} s^-1",
        "the corrected sweep's correlation": f"an omega_s0/R correlation of {row['corr']:.2f}:",
    }
    bad += [f"{what}: {text!r} not in DESIGN.md" for what, text in need.items() if text not in par]
    return bad


def _design_md() -> str:
    return (REPO / "DESIGN.md").read_text(encoding="utf-8")


def test_design_prints_the_record_it_cites():
    assert basis_failures(_design_md(), _cells().data) == []


def test_the_cited_figures_differ():
    """The released print and the corrected sweep's are different figures, so the drills tell them apart."""
    data = _cells().data
    assert data["shipped_v1_3_0"][RELEASED + "corr"] != f"{_sweep_row(data)['corr']:.2f}"


@pytest.mark.parametrize("old, new", [
    ("printed as +0.84 for these boxes", "printed as +0.85 for these boxes"),
    ("an omega_s0/R correlation of 0.82:", "an omega_s0/R correlation of 0.85:"),
    ("cycle-rate box (8e+07, 2e+08) s^-1", "cycle-rate box (8e+07, 1.6e+08) s^-1"),
    ("its printed +-10.", "its printed +-12."),
    ("Uniform lambda_c (8e+07, 1.6e+08) s^-1", "Uniform lambda_c (8e+07, 2e+08) s^-1"),
])
def test_basis_drill_refuses_an_altered_figure(old, new):
    """Drill: one figure of the Method or Basis paragraph altered in a copy of DESIGN.md is refused."""
    doc = " ".join(_design_md().split())
    assert old in doc, old
    assert basis_failures(doc.replace(old, new), _cells().data)


def test_basis_drill_refuses_an_altered_record():
    """Drill: the record's released print or its corrected sweep row altered under an unchanged DESIGN.md is
    refused, and so is a sweep row that ran other boxes."""
    base = _thaw(_cells().data)
    for edit in (
        lambda d: d["shipped_v1_3_0"].__setitem__(RELEASED + "corr", "0.85"),
        lambda d: _sweep_row(d).__setitem__("corr", 0.85),
        lambda d: _sweep_row(d).__setitem__("os0_box", [0.5, 1.2]),
    ):
        data = _thaw(base)
        edit(data)
        assert data != base
        assert basis_failures(_design_md(), data)


# ------------------------------------------------------------------ the models the sampler actually receives
# The checks above read the published figures against the design's declared pins. These read them against the
# models `openmucf.design.base_posterior` and `openmucf.design._refit_sd` actually hand to the sampler: the
# sampler boundary (numpyro's MCMC and NUTS, as `openmucf.calibrate` and the refit take them) is replaced by a
# stand-in that traces the received model with numpyro's handlers and stops before any chain is drawn.
class _Received(Exception):
    def __init__(self, sites):
        super().__init__("the sampler received a model")
        self.sites = sites


class _TraceNUTS:
    def __init__(self, model, *args, **kwargs):
        self.model = model


class _TraceMCMC:
    def __init__(self, kernel, *args, **kwargs):
        self.model = kernel.model

    def run(self, rng_key, *args, extra_fields=(), **kwargs):
        from numpyro import handlers

        trace = handlers.trace(handlers.seed(self.model, rng_seed=0)).get_trace(*args, **kwargs)
        sites = {}
        for name, site in trace.items():
            if site["type"] != "sample":
                continue
            fn = site["fn"]
            rec = {"dist": type(fn).__name__, "observed": bool(site["is_observed"])}
            for attr in ("low", "high", "loc", "scale"):
                if hasattr(fn, attr) and getattr(fn, attr) is not None:
                    rec[attr] = float(getattr(fn, attr))
            if rec["observed"]:
                rec["value"] = float(site["value"])
            sites[name] = rec
        raise _Received(sites)


def received_models(monkeypatch) -> dict:
    """``{model name: {site: description}}`` for the base posterior and every refit of every registered
    candidate in both classes, each traced at the sampler boundary."""
    import numpyro.infer

    for mod in (calibrate, numpyro.infer):
        monkeypatch.setattr(mod, "MCMC", _TraceMCMC)
        monkeypatch.setattr(mod, "NUTS", _TraceNUTS)
    out = {}
    calls = {"base_posterior": lambda: design.base_posterior()}
    for cid in ("C1", "C2", "C3", "C4"):
        for cls in ("constant", "inflated"):
            calls[f"refit {cid} {cls}"] = (
                lambda c=cid, k=cls: design._refit_sd(design._resolve(c), k, 1.0, 1.0, 0))
    for name, call in calls.items():
        try:
            call()
        except _Received as got:
            out[name] = got.sites
        else:
            raise AssertionError(f"{name}: the sampler boundary was not reached")
    return out


#: (quantity, the published wording's pattern in the Method paragraph, how a received model renders it)
_RECEIVED = (
    ("omega_s0 prior", r"Uniform omega_s0 \(([^)]*)\)",
     lambda s: f"{s['omega_s0_pct']['low']:.2f}, {s['omega_s0_pct']['high']:.2f}"),
    ("R prior", r"Uniform R \(([^)]*)\)", lambda s: f"{s['R']['low']:.2f}, {s['R']['high']:.2f}"),
    ("lambda_c prior", r"Uniform lambda_c \(([^)]*)\) s\^-1",
     lambda s: f"{s['lambda_c']['low']:.2g}, {s['lambda_c']['high']:.2g}"),
    ("effective sticking observation", r"the effective sticking (\S+ \+- \S+) %",
     lambda s: f"{s['obs_ose']['value']:g} +- {s['obs_ose']['scale']:g}"),
    ("yield observation", r"the yield (\S+ \+- \S+), with",
     lambda s: f"{s['obs_xmu']['value']:g} +- {s['obs_xmu']['scale']:g}"),
)


def model_basis_failures(models: dict, design_md: str, slots: dict) -> list[str]:
    """Each way a received model departs from the basis DESIGN.md publishes and the generator renders
    (`basis_slots`): a prior support, an observed value or scale, a likelihood that is not two independent
    Normals, or an observation of the cycle rate. Each failure names the received and the published value."""
    par = _paragraphs(design_md)
    rendered = {
        "omega_s0 prior": f"{slots['os0_lo']}, {slots['os0_hi']}",
        "R prior": f"{slots['R_lo']}, {slots['R_hi']}",
        "lambda_c prior": f"{slots['lc_lo']}, {slots['lc_hi']}",
        "effective sticking observation": f"{slots['obs']} +- {slots['obs_sd']}",
        "yield observation": f"{slots['xmu']} +- {slots['xmu_sd']}",
    }
    bad = []
    if "with no cycle-rate measurement" not in par:
        bad.append("DESIGN.md does not say the base likelihood has no cycle-rate measurement")
    for name, sites in models.items():
        observed = {k for k, v in sites.items() if v["observed"]}
        allowed = {"obs_ose", "obs_xmu"} | ({"obs_design"} if name.startswith("refit") else set())
        if observed != allowed:
            bad.append(f"{name}: the sampler received observations {sorted(observed)}, the published basis "
                       f"has {sorted(allowed)} (no cycle-rate measurement)")
            continue
        for site in ("obs_ose", "obs_xmu"):
            if sites[site]["dist"] != "Normal":
                bad.append(f"{name}: {site} is a {sites[site]['dist']}, not an independent Normal")
        for quantity, pattern, render in _RECEIVED:
            got = render(sites)
            m = re.search(pattern, par)
            published = m.group(1) if m else None
            if got != published:
                bad.append(
                    f"{name}: {quantity}: the sampler received ({got}), DESIGN.md publishes ({published})"
                )
            if got != rendered[quantity]:
                bad.append(f"{name}: {quantity}: the sampler received ({got}), the generator renders "
                           f"({rendered[quantity]})")
    return bad


def _slots():
    return _load("scripts/generate_design.py", "_gen_design_basis").basis_slots(_cells())


def test_the_sampled_models_are_the_published_basis(monkeypatch):
    """The base posterior and every refit hand the sampler the priors and observations DESIGN.md states."""
    models = received_models(monkeypatch)
    assert len(models) == 9 and all(models.values()), sorted(models)
    assert model_basis_failures(models, _design_md(), _slots()) == []


@pytest.mark.parametrize("label, drift, named", [
    ("the base yield sigma", ("calibrate", {"xmu_sd": 10.0}), "base_posterior: yield observation"),
    ("the base cycle-rate box", ("calibrate", {"lambda_c_prior": (0.8e8, 2.0e8)}),
     "base_posterior: lambda_c prior"),
    ("the refit yield sigma", ("design_obs", {"xmu_sd": 10.0}), "refit C1 constant: yield observation"),
    ("the refit cycle-rate box", ("design_lc", 2.0e8), "refit C1 constant: lambda_c prior"),
])
def test_model_basis_drill_refuses_a_drifted_model(monkeypatch, label, drift, named):
    """Drill: with DESIGN.md, the design pins and the admitted record unchanged, a model handed to the sampler
    with a different yield sigma or cycle-rate box -- in the base posterior's call, or in the refits -- is
    refused, and the failure names the value received and the value published."""
    slots = _slots()
    kind, arg = drift
    if kind == "calibrate":
        real = calibrate.run_mcmc

        def drifted(*a, **k):
            return real(*a, **{**k, **arg})

        monkeypatch.setattr(calibrate, "run_mcmc", drifted)
    elif kind == "design_obs":
        monkeypatch.setattr(design, "_OBS", {**design._OBS, **arg})
    else:
        monkeypatch.setattr(design, "_LAMBDA_C_HI", arg)
    failures = model_basis_failures(received_models(monkeypatch), _design_md(), slots)
    hits = [f for f in failures if f.startswith(named)]
    assert hits, (label, failures)
    assert "received" in hits[0] and "publishes" in hits[0], hits[0]
