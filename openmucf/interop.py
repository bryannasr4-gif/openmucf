"""openmucf.interop -- GEANT4 / external-tool interoperability (v1 stub).

Honors the pre-registered interop contract (PRE_REGISTRATION.md, "GEANT4 interop contract"):

  * EXPORT the differentiable OpenMuCF rates -- omega_s^eff(phi, T) and lambda_dtmu(E, phi, T, F) --
    as plain numeric tables (CSV/JSON) plus a callable API that a GEANT4 muonic-atom run can consume.
  * INGEST an external spectrum (e.g. neutron time-of-flight, mu-He sticking X-rays) as validation
    data, returned in a normalized form for comparison against the engine.
  * NEVER re-implement the particle transport GEANT4 already does. OpenMuCF is the rate / kinetics /
    UQ layer; this module is only the thin data bridge.

This is a v1 stub: it *fixes the on-disk formats and the callable contract* so the downstream GEANT4
coupling (Phase 5) is a wiring exercise, not a redesign. One honest caveat is baked in: in v1,
``omega_s^eff`` is condition-independent (= omega_s0*(1-R) from the ledger), so its (phi, T) table is
flat by construction -- the (phi, T, c_t) dependence is exactly what the Phase-3 surrogate fills in.
The table *shape* is already Phase-3-ready.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from . import formation
from .analytic import effective_sticking
from .rates import omega_fraction

# ---------------------------------------------------------------------------
# Export: rate tables + a callable API for GEANT4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateTable:
    """A named rate sampled on a 1-D or 2-D grid, ready for CSV/JSON export.

    ``values`` is a plain (nested) Python-float list matching ``axis_grids``:
    1-D -> ``values[i]``; 2-D -> ``values[i][j]`` (row index = first axis).
    """

    name: str
    axis_names: tuple[str, ...]
    axis_grids: tuple[tuple[float, ...], ...]
    values: object
    unit: str
    note: str = ""

    @property
    def ndim(self) -> int:
        return len(self.axis_names)

    def to_json(self, path) -> Path:
        """Serialize to JSON (axes + values + metadata)."""
        from . import __version__

        path = Path(path)
        payload = {
            "openmucf_version": __version__,
            "name": self.name,
            "unit": self.unit,
            "axis_names": list(self.axis_names),
            "axis_grids": [list(g) for g in self.axis_grids],
            "values": self.values,
            "note": self.note,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def to_csv(self, path) -> Path:
        """Serialize to long-format CSV (one grid point per row) -- the tool-agnostic form."""
        path = Path(path)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*self.axis_names, self.name])
            if self.ndim == 1:
                (g0,) = self.axis_grids
                for i, x0 in enumerate(g0):
                    w.writerow([x0, self.values[i]])
            elif self.ndim == 2:
                g0, g1 = self.axis_grids
                for i, x0 in enumerate(g0):
                    for j, x1 in enumerate(g1):
                        w.writerow([x0, x1, self.values[i][j]])
            else:  # pragma: no cover - v1 exports are 1-D or 2-D only
                raise ValueError(f"CSV export supports 1-D/2-D tables, got ndim={self.ndim}")
        return path


def _build_2d(g0, g1, fn):
    return [[float(fn(x0, x1)) for x1 in g1] for x0 in g0]


def _build_1d(g0, fn):
    return [float(fn(x0)) for x0 in g0]


def _ledger_omega_s_eff(rates, use_legacy_sticking: bool = False) -> float:
    """omega_s^eff = omega_s0*(1-R) from the ledger (a bare fraction)."""
    os0 = omega_fraction(rates["omega_s0_legacy" if use_legacy_sticking else "omega_s0"])
    return float(effective_sticking(os0, rates.value("R_col")))


def export_omega_s_eff(rates, T_grid, phi_grid, use_legacy_sticking: bool = False) -> RateTable:
    """omega_s^eff(phi, T) table. v1: flat in (phi, T) -- see module docstring (Phase-3 fills it in)."""
    ose = _ledger_omega_s_eff(rates, use_legacy_sticking)
    grids = (tuple(float(p) for p in phi_grid), tuple(float(t) for t in T_grid))
    values = _build_2d(grids[0], grids[1], lambda phi, T: ose)
    return RateTable(
        name="omega_s_eff",
        axis_names=("phi", "T"),
        axis_grids=grids,
        values=values,
        unit="fraction",
        note="v1: condition-independent (omega_s0*(1-R)); (phi,T,c_t) dependence arrives in Phase 3.",
    )


def export_lambda_dtmu_thermal(T_grid, phi_grid, F: int = 1, eta: float = 1.0) -> RateTable:
    """Thermally-averaged lambda_dtmu(phi, T) at fixed hyperfine F [s^-1]."""
    grids = (tuple(float(p) for p in phi_grid), tuple(float(t) for t in T_grid))
    values = _build_2d(grids[0], grids[1], lambda phi, T: formation.lambda_dtmu(T, phi, F, eta))
    return RateTable(
        name="lambda_dtmu_thermal",
        axis_names=("phi", "T"),
        axis_grids=grids,
        values=values,
        unit="s^-1",
        note=f"resonance-averaged v1 formation model, F={F}, eta={eta}.",
    )


def export_lambda_dtmu_energy(E_grid, F: int = 1) -> RateTable:
    """Energy-resolved lambda_dtmu(E) at fixed hyperfine F [s^-1] (the Vesman resonances)."""
    g0 = tuple(float(e) for e in E_grid)
    values = _build_1d(g0, lambda E: formation.lambda_dtmu_energy(E, F))
    return RateTable(
        name="lambda_dtmu_energy",
        axis_names=("E_eV",),
        axis_grids=(g0,),
        values=values,
        unit="s^-1",
        note=f"energy-resolved Vesman resonances, F={F}.",
    )


def export_all(
    rates,
    outdir,
    T_grid=(100.0, 200.0, 300.0, 500.0, 800.0),
    phi_grid=(1.0, 1.2, 1.45),
    E_grid=None,
    F: int = 1,
    eta: float = 1.0,
    fmt: str = "both",
) -> dict:
    """Write the standard export bundle to ``outdir`` and return a ``{name: [paths]}`` map.

    ``fmt`` is ``"csv"``, ``"json"``, or ``"both"``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if E_grid is None:
        E_grid = [i * 2.0 / 199 for i in range(200)]  # 0..2 eV, 200 points (no numpy dep)

    tables = [
        export_omega_s_eff(rates, T_grid, phi_grid),
        export_lambda_dtmu_thermal(T_grid, phi_grid, F=F, eta=eta),
        export_lambda_dtmu_energy(E_grid, F=F),
    ]
    written: dict = {}
    for t in tables:
        paths = []
        if fmt in ("csv", "both"):
            paths.append(t.to_csv(outdir / f"{t.name}.csv"))
        if fmt in ("json", "both"):
            paths.append(t.to_json(outdir / f"{t.name}.json"))
        written[t.name] = paths
    return written


def geant4_callables(rates, use_legacy_sticking: bool = False, eta: float = 1.0) -> dict:
    """Return plain-Python float callables a GEANT4 embedding can query in-process.

    ``omega_s_eff(phi, T)`` and ``lambda_dtmu(E=None, phi=1.0, T=300.0, F=1)`` -- passing ``E`` gives
    the energy-resolved rate, omitting it gives the thermally-averaged rate. Values are host-side
    floats (not traced arrays), which is what an external C++/GEANT4 caller needs.
    """
    ose = _ledger_omega_s_eff(rates, use_legacy_sticking)

    def omega_s_eff(phi=1.0, T=300.0):  # noqa: ARG001 - v1 is condition-independent (Phase-3 fills in)
        return ose

    def lambda_dtmu(E=None, phi=1.0, T=300.0, F=1):
        if E is not None:
            return float(formation.lambda_dtmu_energy(E, F))
        return float(formation.lambda_dtmu(T, phi, F, eta))

    return {"omega_s_eff": omega_s_eff, "lambda_dtmu": lambda_dtmu}


# ---------------------------------------------------------------------------
# Ingest: external validation spectra (e.g. GEANT4 / experiment output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrum:
    """A 1-D validation spectrum ingested from an external tool (transport stays in GEANT4)."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    kind: str
    source: str = ""
    x_unit: str = ""
    y_unit: str = ""

    def __len__(self) -> int:
        return len(self.x)

    def area(self) -> float:
        """Trapezoidal integral of y over x."""
        return sum(
            0.5 * (self.y[i] + self.y[i + 1]) * (self.x[i + 1] - self.x[i]) for i in range(len(self.x) - 1)
        )

    def normalized(self) -> Spectrum:
        """Return a copy area-normalized to unit integral (for shape comparison)."""
        a = self.area()
        if a <= 0.0:
            raise ValueError(f"cannot normalize spectrum with non-positive area {a}")
        y = tuple(v / a for v in self.y)
        return Spectrum(self.x, y, self.kind, self.source, self.x_unit, self.y_unit)


def ingest_spectrum(
    path,
    kind: str = "neutron_tof",
    source: str = "",
    x_unit: str = "",
    y_unit: str = "",
    delimiter: str | None = None,
) -> Spectrum:
    """Parse a two-column numeric spectrum (CSV or whitespace) into a :class:`Spectrum`.

    Blank lines and ``#`` comments are skipped; a non-numeric header row is tolerated. Only the first
    two columns are read. This is validation-data ingest only -- OpenMuCF never runs the transport.
    """
    xs: list[float] = []
    ys: list[float] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(delimiter) if delimiter else (line.split(",") if "," in line else line.split())
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:  # header / non-numeric row -> skip
            continue
    if not xs:
        raise ValueError(f"no numeric two-column rows found in {path}")
    return Spectrum(tuple(xs), tuple(ys), kind=kind, source=source, x_unit=x_unit, y_unit=y_unit)
