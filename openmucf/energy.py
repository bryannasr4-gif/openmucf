"""openmucf.energy -- transparent muCF energy balance (scientific AND net-electrical).

Two explicit layers, so an auditor sees every assumption rather than a single opaque Q:

  scientific gain   Q_sci = X_mu * E_f / E_mu
                    (fusion energy released / muon-production *beam* energy; breakeven X_mu = E_mu/E_f ~ 284)

  net-electrical    Q_net = X_mu * E_f * eta_thermal * M * eta_acc / E_mu
                    where eta_acc   = electrical -> muon-beam (wall-plug) efficiency,
                          eta_thermal = thermal -> electric conversion,
                          M         = blanket energy multiplication (>= 1; 1 for pure muCF,
                                      > 1 for a fission/breeding hybrid).

The net-electrical breakeven is FAR above the scientific one -- this module exists to make that
honesty unavoidable. Every factor is a named knob with a documented default; nothing is hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import E_F_MEV
from .constants import E_MU_GEV_DEFAULT as E_MU_GEV


@dataclass(frozen=True)
class EnergyChain:
    E_f_MeV: float = E_F_MEV
    E_mu_GeV: float = E_MU_GEV
    eta_acc: float = 0.30  # electrical -> muon-beam (wall-plug); optimistic
    eta_thermal: float = 0.40  # thermal -> electric
    blanket_M: float = 1.0  # energy multiplication (>= 1)

    @property
    def E_mu_MeV(self) -> float:
        return self.E_mu_GeV * 1.0e3

    def Q_sci(self, x_mu):
        """Scientific gain: fusion energy out / muon-production beam energy in."""
        return x_mu * self.E_f_MeV / self.E_mu_MeV

    def Q_net_electrical(self, x_mu):
        """Net-electrical gain through the full documented efficiency chain."""
        return (x_mu * self.E_f_MeV * self.eta_thermal * self.blanket_M * self.eta_acc) / self.E_mu_MeV

    def breakeven_xmu_sci(self) -> float:
        """X_mu at Q_sci = 1 (~284 for 5 GeV)."""
        return self.E_mu_MeV / self.E_f_MeV

    def breakeven_xmu_net(self) -> float:
        """X_mu at Q_net = 1 -- the brutal, honest target."""
        return self.E_mu_MeV / (self.E_f_MeV * self.eta_thermal * self.blanket_M * self.eta_acc)
