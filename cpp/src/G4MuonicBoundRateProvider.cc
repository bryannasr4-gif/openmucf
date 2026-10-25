// G4MuonicBoundRateProvider -- see the header for what the pair is, where it is used and its domain.
#include "G4MuonicBoundRateProvider.hh"

#include "G4Exception.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <atomic>
#include <cmath>
#include <sstream>
#include <string>

namespace {

// Set on the master thread before any consumer runs; read by every thread afterwards.
std::atomic<bool> g_enabled{false};
// Set by the first Consult, on whichever thread forms the first rate pair.
std::atomic<bool> g_frozen{false};

void Change(bool value) {
  if (g_frozen.load() && g_enabled.load() != value) {
    G4Exception("G4MuonicBoundRateProvider::Change", "G4MuonicBoundRates002", FatalException,
                "G4HadronicParameters::SetEnableMuonicBoundRates was called after a consumer had formed a "
                "bound-muon rate pair; muonic atoms already constructed keep the lifetimes they were built "
                "with, so the switch can be set only before the first one");
    return;
  }
  g_enabled.store(value);
}

bool PositiveFinite(G4double x) { return std::isfinite(x) && x > 0.; }

}  // namespace

void G4MuonicBoundRateProvider::Enable() { Change(true); }

void G4MuonicBoundRateProvider::Disable() { Change(false); }

G4bool G4MuonicBoundRateProvider::IsEnabled() { return g_enabled.load(); }

G4bool G4MuonicBoundRateProvider::IsFrozen() { return g_frozen.load(); }

G4bool G4MuonicBoundRateProvider::Consult() {
  g_frozen.store(true);
  return g_enabled.load();
}

G4bool G4MuonicBoundRateProvider::InDomain(G4int Z, G4int A) { return Z >= 2 && Z <= 13 && A >= Z; }

G4double G4MuonicBoundRateProvider::LowZDecayRate(G4int Z, G4double muonMass, G4double nuclearMass) {
  const G4double freeMuonDecayRate = 0.45517005 / microsecond;
  G4double lambda = 1.0;
  const G4double x = Z * fine_structure_const;
  lambda -= x * x * (0.5 + 0.06 * muonMass / nuclearMass);
  lambda *= freeMuonDecayRate;
  return lambda;
}

G4MuonicBoundRateProvider::Status G4MuonicBoundRateProvider::Rates(G4int Z, G4int A, G4double muonMass,
                                                                   G4double nuclearMass,
                                                                   G4MuonicBoundRates& rates) {
  if (!Consult()) return Status::kDisabled;
  if (!InDomain(Z, A)) return Status::kUnsupported;
  G4double capture = 0.;
  G4double decay = 0.;
  const char* why = nullptr;
  if (!PositiveFinite(nuclearMass)) {
    why = "the nuclear mass is not finite and positive";
  } else if (!PositiveFinite(muonMass)) {
    why = "the muon mass is not finite and positive";
  } else {
    capture = G4MuonicAtomHelper::GetMuonCaptureRate(Z, A);
    decay = LowZDecayRate(Z, muonMass, nuclearMass);
    if (!std::isfinite(capture) || capture < 0.) {
      why = "the capture rate is negative or not finite";
    } else if (!PositiveFinite(decay)) {
      why = "the decay rate is not finite and positive";
    }
    // A finite capture >= 0 and a finite decay > 0 (at most the free rate) give a finite, positive sum.
  }
  if (why != nullptr) {
    std::ostringstream message;
    message.precision(17);
    message << "Z=" << Z << " A=" << A << ": " << why << " (nuclear mass " << nuclearMass << " MeV, muon mass "
            << muonMass << " MeV, capture " << capture << " /ns, decay " << decay << " /ns)";
    G4Exception("G4MuonicBoundRateProvider::Rates", "G4MuonicBoundRates001", FatalException,
                message.str().c_str());
    return Status::kRefused;
  }
  rates.capture = capture;
  rates.decay = decay;
  return Status::kCommon;
}

const char* G4MuonicBoundRateProvider::Prescription() {
  return "capture: G4MuonicAtomHelper::GetMuonCaptureRate(Z, A); decay: the Z < 14 expression of "
         "G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, nuclMass); domain 2 <= Z <= 13, A >= Z";
}
