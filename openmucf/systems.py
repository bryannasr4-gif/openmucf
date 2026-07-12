"""openmucf.systems -- the full differentiable energy-balance graph + the "Q Rosetta stone".

A strict SUPERSET of :class:`openmucf.energy.EnergyChain` (which stays FROZEN -- v1 public API).
:class:`SystemChain` exposes every node of the wall-plug -> muon -> fusion(+breeding) -> blanket ->
thermal -> electric -> recirculation chain as a named, differentiable knob, and adds two explicit,
FLAGGED, default-OFF factors the v1 chain folded away:

  * ``breeding_credit_MeV`` -- an additive tritium-breeding / blanket energy credit per fusion
    (default 0; a flagged column-equivalent, never a silent default);
  * ``recirc_fraction`` -- a recirculating-power fraction that reduces net electrical output
    (default 0; v1-degenerate bookkeeping).

At the defaults ``breeding_credit_MeV=0``/``recirc_fraction=0`` the graph is the v1 EnergyChain to
machine precision -- the G-legacy degenerate special case and no-tuning anchor.

The **Rosetta stone** (:class:`QBasis` + :data:`BASES` + :func:`rosetta_table`) converts the several Q
conventions in the muCF literature into ONE comparable reference basis (the efficiency-free scientific
gain ``q_sci``): v1's scientific gain ``q_sci_v1`` and net-electrical gain ``q_net_v1``, the
Kelly-Hart-Rose 2021 electrical gain ``kelly_Q_elec``, and a Yin-Kou-Chen-style efficiency-free gain
``ykc_efficiency_free``. It shows that a "Q=1.87" in one paper and a "14 %" in another are the same
physics viewed through different accounting -- the table IS the deliverable.

Out of ``__all__`` (a submodule like ``calibrate``/``forecast``). The engine functions ``q_sci``/``q_net``
are pure ``jax.numpy`` (differentiable; no host-side branching on traced values); all shipped scalars are
CLOSED-FORM algebra (no solver/MCMC), hence byte-stable and printed at full precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from .constants import E_F_MEV
from .constants import E_MU_GEV_DEFAULT as E_MU_GEV


@dataclass(frozen=True)
class SystemChain:
    """Differentiable muCF energy-balance graph; supersets :class:`openmucf.energy.EnergyChain`.

    Field defaults reproduce the v1 EnergyChain exactly (``E_mu_GeV`` from the ledger via
    ``openmucf.constants``; ``eta_acc=0.30`` KEPT -- the Kelly PSI-measured 0.18 is a documented FINDING,
    not a silent default change). ``breeding_credit_MeV`` and ``recirc_fraction`` are the two new flagged
    knobs, both default-off so the extended graph is degenerate-equal to v1.
    """

    E_mu_GeV: float = E_MU_GEV
    eta_acc: float = 0.30  # electrical -> muon-beam (wall-plug); optimistic v1 default (KEPT)
    eta_thermal: float = 0.40  # thermal -> electric
    blanket_M: float = 1.0  # blanket energy multiplication (>= 1)
    E_fusion_MeV: float = E_F_MEV  # D-T kinetic energy per fusion (17.6 MeV)
    breeding_credit_MeV: float = 0.0  # FLAGGED additive tritium-breeding credit per fusion (default 0)
    recirc_fraction: float = 0.0  # recirculating-power fraction of gross electric (default 0)

    @property
    def E_mu_MeV(self) -> float:
        """Muon-production beam energy in MeV."""
        return self.E_mu_GeV * 1.0e3

    @property
    def E_per_fusion_MeV(self):
        """Total thermal energy released per catalysed fusion = D-T kinetic + flagged breeding credit.

        A single named graph node feeding both ``q_sci`` and ``q_net``. At ``breeding_credit_MeV=0`` this
        is exactly ``E_fusion_MeV`` (= v1's ``E_f_MeV``).
        """
        return self.E_fusion_MeV + self.breeding_credit_MeV

    def _net_efficiency_factor(self):
        """The electrical-chain multiplier converting scientific gain -> net-electrical gain:
        ``blanket_M * eta_thermal * eta_acc * (1 - recirc_fraction)``. Degenerate v1 factor at defaults.
        """
        return self.blanket_M * self.eta_thermal * self.eta_acc * (1.0 - self.recirc_fraction)

    def breakeven_xmu_sci(self) -> float:
        """X_mu at ``q_sci = 1`` (= ``E_mu_MeV / E_per_fusion_MeV``; ~284 at defaults). Closed form."""
        return self.E_mu_MeV / self.E_per_fusion_MeV

    def breakeven_xmu_net(self) -> float:
        """X_mu at ``q_net = 1`` (~2367 at defaults; ~3946 at ``eta_acc=0.18`` -- the self-correction
        finding). Closed form; linear in ``1/eta_acc``.
        """
        return self.E_mu_MeV / (self.E_per_fusion_MeV * self._net_efficiency_factor())


def q_sci(chain: SystemChain, x_mu):
    """Scientific gain: total nuclear energy released per muon-production beam energy (efficiency-free).

    ``q_sci = x_mu * E_per_fusion_MeV / E_mu_MeV``. Supersets ``EnergyChain.Q_sci``: at the default
    ``breeding_credit_MeV=0`` this is v1's ``x_mu * E_f / E_mu`` exactly. Carries NO efficiency factors
    and NO blanket multiplication -- exactly as v1 kept those out of ``Q_sci``. Differentiable in ``x_mu``.
    """
    x = jnp.asarray(x_mu, dtype=jnp.float64)
    return x * chain.E_per_fusion_MeV / chain.E_mu_MeV


def q_net(chain: SystemChain, x_mu):
    """Net-electrical gain through the full documented graph (differentiable):

        ``q_net = x_mu * E_per_fusion_MeV * blanket_M * eta_thermal * eta_acc * (1 - recirc_fraction)
                  / E_mu_MeV``.

    Supersets ``EnergyChain.Q_net_electrical`` (``breeding_credit_MeV=0``, ``recirc_fraction=0`` -> byte
    identical). Monotone increasing in ``x_mu``, ``eta_acc``, ``eta_thermal``, ``blanket_M`` and
    ``breeding_credit_MeV``; decreasing in ``E_mu_GeV`` and ``recirc_fraction`` (checked by ``jax.grad``).
    """
    x = jnp.asarray(x_mu, dtype=jnp.float64)
    return x * chain.E_per_fusion_MeV * chain._net_efficiency_factor() / chain.E_mu_MeV


# --------------------------------------------------------------------------------------------------
# Kelly-Hart-Rose 2021 electrical-gain convention (an external published Q basis)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KellyElectrical:
    """Kelly, Hart & Rose 2021 (J. Phys. Energy 3, 035003, open access) electrical-gain chain.

    Their Eq. (2): ``Q_elec = ((F*eta_mu + H*eta_rec)/B) * eta_acc * eta_heat`` with, from their Table 1
    (deuteron -> tungsten; 150 fusions/muon; their highest-Q config, Q=1.87; 5000-beam-particle sim):
    ``F`` = fusion heat = row (A) = 2991 MeV (17.6 MeV D-T kinetic + 8.4 MeV fusion-neutron tritium
    breeding = 26.0 MeV/fusion, folded per the paper); ``H`` = recoverable heat = rows (B)-(E) =
    2664+23+526+530 = 3743 MeV; ``B`` = beam energy = row (G) = 3606 MeV. Efficiencies (their text):
    ``eta_mu=0.50`` ("arbitrary but reasonable" muon-delivery), ``eta_rec=1.00`` (100 % recoverable-heat
    capture), ``eta_acc=0.18`` (PSI 590 MeV accelerator), ``eta_heat=0.60`` (heat->electricity, ">60%").
    Fusion heat scales with fusions/muon (H, B fixed) so ``Q_elec`` is AFFINE in ``x_mu`` -- their
    figure-3 curve. Every number was read from the OA paper this session; nothing is tuned.
    """

    F_fusion_MeV_ref: float = 2991.0  # Table 1 (A): fusion heat / beam particle at x_mu_ref fusions/muon
    H_recover_MeV: float = 3743.0  # rows (B)-(E): 2664 + 23 + 526 + 530
    B_beam_MeV: float = 3606.0  # row (G): beam energy / beam particle (= 3.61 GeV)
    eta_mu: float = 0.50
    eta_rec: float = 1.00
    eta_acc: float = 0.18
    eta_heat: float = 0.60
    x_mu_ref: float = 150.0  # Jones 1986 record fusions/muon (the headline operating point)
    E_fusion_kinetic_MeV: float = E_F_MEV  # 17.6 MeV pure D-T (for the scientific-gain reference)
    E_mu_GeV: float = 4.70  # 3.61 GeV beam / 0.77 muons per beam particle (Table 1)

    def q_elec(self, x_mu: float | None = None) -> float:
        """Kelly Eq. (2) electrical gain at ``x_mu`` fusions/muon (default: the 150-fusion anchor).

        Closed-form; returns a Python float. At ``x_mu=150`` this is the faithful reproduction of Kelly's
        Table-1 config (0.1569); the paper's 14 % HEADLINE is the lower figure-3 *curve* value (see the
        SYSTEMS.md finding -- an internal Table-1-vs-curve gap, reported not tuned).
        """
        x = self.x_mu_ref if x_mu is None else float(x_mu)
        F = self.F_fusion_MeV_ref * (x / self.x_mu_ref)  # fusion heat scales with fusions/muon
        return ((F * self.eta_mu + self.H_recover_MeV * self.eta_rec) / self.B_beam_MeV) * (
            self.eta_acc * self.eta_heat
        )

    def q_sci(self, x_mu: float | None = None) -> float:
        """Kelly's efficiency-free scientific gain (the reference basis): ``x_mu * 17.6 / E_mu``."""
        x = self.x_mu_ref if x_mu is None else float(x_mu)
        return x * self.E_fusion_kinetic_MeV / (self.E_mu_GeV * 1.0e3)


# One shared instance -- Kelly's published operating point; all its numbers are paper-cited.
KELLY = KellyElectrical()

# Kelly's effective electrical multiplier at his operating point: Q_elec / scientific-gain. Bundles the
# heat boost (breeding + recoverable-particle recovery) and the electrical efficiencies into one factor
# so the basis converts cleanly to the reference. Derived from his cited primitives, never hard-coded.
_KELLY_MULT: float = KELLY.q_elec() / KELLY.q_sci()


# --------------------------------------------------------------------------------------------------
# The Q Rosetta stone: convert every convention to one comparable reference basis (q_sci)
# --------------------------------------------------------------------------------------------------
REFERENCE_BASIS = "q_sci_v1"  # the efficiency-free scientific gain is the common yardstick


@dataclass(frozen=True)
class QBasis:
    """A published Q convention + how to express it in the common reference basis (``q_sci``).

    ``convert_to_reference(value, chain)`` maps a Q reported in THIS basis (at ``chain``) to the
    reference scientific gain. Each conversion is exact algebra (or, for Kelly, his cited electrical
    multiplier), so every basis round-trips its own native value back to the reference.
    """

    name: str
    description: str
    convert_to_reference: Callable


def _native_ykc(chain: SystemChain, x_mu):
    """Yin-Kou-Chen-style efficiency-free, credit-free fusion gain: ``x_mu * E_fusion_MeV / E_mu_MeV``
    (pure D-T fusion energy per beam energy, no efficiencies, no breeding credit)."""
    x = jnp.asarray(x_mu, dtype=jnp.float64)
    return x * chain.E_fusion_MeV / chain.E_mu_MeV


def _to_ref_identity(value, chain: SystemChain):
    return value


def _to_ref_from_net(value, chain: SystemChain):
    """Divide out the electrical chain: ``q_net / (blanket_M*eta_thermal*eta_acc*(1-recirc)) = q_sci``."""
    return value / chain._net_efficiency_factor()


def _to_ref_from_ykc(value, chain: SystemChain):
    """Restore the flagged breeding credit: pure-fusion gain * (E_per_fusion / E_fusion) = q_sci."""
    return value * (chain.E_per_fusion_MeV / chain.E_fusion_MeV)


def _to_ref_from_kelly(value, chain: SystemChain):
    """Divide out Kelly's cited electrical multiplier to recover the scientific gain."""
    return value / _KELLY_MULT


BASES: dict[str, QBasis] = {
    "q_sci_v1": QBasis(
        "q_sci_v1",
        "v1 scientific gain: nuclear energy released per muon-production beam energy "
        "(x_mu*E_per_fusion/E_mu). The reference basis.",
        _to_ref_identity,
    ),
    "q_net_v1": QBasis(
        "q_net_v1",
        "v1 net-electrical gain through the full efficiency chain "
        "(q_sci * blanket_M * eta_thermal * eta_acc * (1-recirc_fraction)).",
        _to_ref_from_net,
    ),
    "kelly_Q_elec": QBasis(
        "kelly_Q_elec",
        "Kelly-Hart-Rose 2021 electrical gain Q_elec (their Eq. 2 + Table 1; open access, "
        "DOI 10.1088/2515-7655/abfb4b). Converted via their cited electrical multiplier.",
        _to_ref_from_kelly,
    ),
    "ykc_efficiency_free": QBasis(
        "ykc_efficiency_free",
        "Yin-Kou-Chen-style efficiency-free fusion gain (x_mu*E_fusion/E_mu; no efficiencies, no "
        "breeding credit). With the shared E_mu convention this equals q_sci -- the simplest identity.",
        _to_ref_from_ykc,
    ),
}


def _native_value(basis_name: str, chain: SystemChain, x_mu):
    """Native Q value for ``basis_name`` at (chain, x_mu). Kelly uses its own published chain."""
    if basis_name == "q_sci_v1":
        return q_sci(chain, x_mu)
    if basis_name == "q_net_v1":
        return q_net(chain, x_mu)
    if basis_name == "ykc_efficiency_free":
        return _native_ykc(chain, x_mu)
    if basis_name == "kelly_Q_elec":
        return KELLY.q_elec(x_mu)
    raise KeyError(basis_name)  # pragma: no cover


def rosetta_table(x_mu_grid) -> list[dict]:
    """One row per X_mu: each basis's native Q and the common efficiency-free scientific-gain reference.

    The three SystemChain-native bases (``q_sci_v1``, ``q_net_v1``, ``ykc_efficiency_free``) are
    evaluated on the v1-default :class:`SystemChain`; each collapses to the SAME ``reference_q_sci`` under
    its ``convert_to_reference``. ``kelly_Q_elec`` is Kelly-Hart-Rose's published electrical convention on
    THEIR chain (an external anchor); its reference is Kelly's own scientific gain. Every value is
    closed-form algebra computed with pure-Python floats (NOT through the XLA graph) so the shipped table
    is byte-stable cross-architecture -- the table is deterministic and IS the deliverable.
    """
    chain = SystemChain()
    epf = chain.E_per_fusion_MeV  # Python float at defaults
    emu = chain.E_mu_MeV
    netf = chain._net_efficiency_factor()
    efus = chain.E_fusion_MeV
    rows: list[dict] = []
    for x in x_mu_grid:
        xr = float(x)
        qsci = xr * epf / emu
        rows.append(
            {
                "x_mu": xr,
                "q_sci_v1": qsci,
                "q_net_v1": xr * epf * netf / emu,
                "kelly_Q_elec": KELLY.q_elec(xr),
                "ykc_efficiency_free": xr * efus / emu,
                "reference_q_sci": qsci,
            }
        )
    return rows
