// G4MuonicBoundRateProvider -- see the header for what the pair is, where it is used and its domain.
#include "G4MuonicBoundRateProvider.hh"

#include "G4Exception.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4MuonicDataOverlay.hh"
#include "G4MuonicDataTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <atomic>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
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

const char* const kCompiled = "compiled";

// The pair profile the frozen overlay selected, or "compiled" with the data option off or none selected.
std::string PairProfile() {
  if (!G4MuonicDataTable::IsEnabled()) return kCompiled;
  const std::string& profile = G4MuonicDataOverlay::Config().d1PairProfile;
  return profile.empty() ? std::string(kCompiled) : profile;
}

// A pair profile selected with the common-rate switch off: the requested evaluated pairs would never reach a
// consumer. Fatal G4MuonicBoundRates003 the first time; should an installed handler suppress the abort, every
// later consultation throws rather than let a consumer run its own pair for the requested data.
void RefusePairWithoutSwitch(const std::string& profile) {
  const std::string text = "G4MUONICDATA_D1_PAIR_PROFILE selects the bound-rate pair profile '" + profile +
                           "' but G4HadronicParameters::SetEnableMuonicBoundRates(true) was not called before the "
                           "first consumer formed a rate pair; the evaluated pairs need both options";
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true)) {
    G4Exception("G4MuonicBoundRateProvider::Consult", "G4MuonicBoundRates003", FatalException, text.c_str());
  }
  throw std::runtime_error(text);
}

// Why a pair tuple cannot be used, or nullptr. Rates in ns^-1.
const char* PairProblem(G4double tau, G4double q, G4double freeTau, G4double& capture, G4double& decay) {
  if (!PositiveFinite(tau) || !PositiveFinite(freeTau)) return "tau or free_tau is not finite and positive";
  if (!std::isfinite(q) || !(q > 0.) || q > 1.) return "q is outside (0, 1]";
  decay = q / freeTau;
  capture = 1. / tau - decay;
  if (!PositiveFinite(decay) || decay > 1. / freeTau) return "the decay rate is outside 0 < decay <= 1/free_tau";
  if (!std::isfinite(capture) || capture < 0.) return "the capture rate 1/tau - q/free_tau is negative or not finite";
  return nullptr;
}

}  // namespace

void G4MuonicBoundRateProvider::Enable() { Change(true); }

void G4MuonicBoundRateProvider::Disable() { Change(false); }

G4bool G4MuonicBoundRateProvider::IsEnabled() { return g_enabled.load(); }

G4bool G4MuonicBoundRateProvider::IsFrozen() { return g_frozen.load(); }

G4bool G4MuonicBoundRateProvider::Consult() {
  g_frozen.store(true);
  const bool enabled = g_enabled.load();
  if (!enabled) {
    const std::string profile = PairProfile();
    if (profile != kCompiled) RefusePairWithoutSwitch(profile);
  }
  return enabled;
}

G4bool G4MuonicBoundRateProvider::InDomain(G4int Z, G4int A) { return Z >= 2 && Z <= 13 && A >= Z; }

G4double G4MuonicBoundRateProvider::LowZDecayRate(G4int Z, G4double muonMass, G4double nuclearMass) {
  if (Z < 2 || Z > 13 || !PositiveFinite(muonMass) || !PositiveFinite(nuclearMass)) {
    return std::numeric_limits<G4double>::quiet_NaN();
  }
  const G4double eta = muonMass / nuclearMass;
  if (!std::isfinite(eta) || eta >= 0.5) return std::numeric_limits<G4double>::quiet_NaN();
  const G4double freeMuonDecayRate = 0.45517005 / microsecond;
  G4double lambda = 1.0;
  const G4double x = Z * fine_structure_const;
  lambda -= x * x * (0.5 - eta);
  lambda *= freeMuonDecayRate;
  return lambda;
}

G4MuonicBoundRateProvider::Status G4MuonicBoundRateProvider::Rates(G4int Z, G4int A, G4double muonMass,
                                                                   G4double nuclearMass,
                                                                   G4MuonicBoundRates& rates) {
  if (!Consult()) return Status::kDisabled;
  // An evaluated pair row takes precedence: both rates come from its one tuple.
  if (G4MuonicDataTable::IsEnabled()) {
    const G4MuonicDataOverlay::Resolution row = G4MuonicDataOverlay::ResolvePair(Z, A);
    if (row.values != nullptr) {
      G4double capture = 0., decay = 0.;
      const G4double tau = row.values[0] * ns, q = row.values[1], freeTau = row.values[2] * ns;
      if (row.count != 3) {
        G4Exception("G4MuonicBoundRateProvider::Rates", "G4MuonicBoundRates001", FatalException,
                    "the bound-rate pair row does not carry the three values tau q free_tau");
        return Status::kRefused;
      }
      if (const char* why = PairProblem(tau, q, freeTau, capture, decay)) {
        std::ostringstream message;
        message.precision(17);
        message << "Z=" << Z << " A=" << A << " under pair profile " << row.profile << ": " << why << " (tau " << tau
                << " ns, q " << q << ", free_tau " << freeTau << " ns)";
        G4Exception("G4MuonicBoundRateProvider::Rates", "G4MuonicBoundRates001", FatalException,
                    message.str().c_str());
        return Status::kRefused;
      }
      rates.capture = capture;
      rates.decay = decay;
      return Status::kCommon;
    }
  }
  if (!InDomain(Z, A)) return Status::kUnsupported;
  G4double capture = 0.;
  G4double decay = 0.;
  const char* why = nullptr;
  if (!PositiveFinite(nuclearMass)) {
    why = "the nuclear mass is not finite and positive";
  } else if (!PositiveFinite(muonMass)) {
    why = "the muon mass is not finite and positive";
  } else if (!std::isfinite(muonMass / nuclearMass) || muonMass / nuclearMass >= 0.5) {
    why = "the mass ratio eta is not finite or is >= 1/2";
  } else {
    capture = G4MuonicAtomHelper::GetMuonCaptureRate(Z, A);
    decay = LowZDecayRate(Z, muonMass, nuclearMass);
    if (!std::isfinite(capture) || capture < 0.) {
      why = "the capture rate is negative or not finite";
    } else if (!PositiveFinite(decay) || decay > 0.45517005 / microsecond) {
      why = "the decay rate is outside 0 < decay <= free";
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

std::string G4MuonicBoundRateProvider::Describe(G4int Z, G4int A) {
  std::ostringstream out;
  const bool data = G4MuonicDataTable::IsEnabled();
  const std::string pairProfile = PairProfile();
  std::string captureProfile = kCompiled;
  if (data) captureProfile = G4MuonicDataOverlay::Config().d1Profile;
  out << "Z=" << Z << " A=" << A << " switch=" << (IsEnabled() ? "on" : "off") << " data=" << (data ? "on" : "off")
      << " capture_profile=" << captureProfile << " pair_profile=" << pairProfile;
  if (!IsEnabled()) {
    out << " answer=consumer";
    return out.str();
  }
  const G4MuonicDataOverlay::Resolution row =
      data ? G4MuonicDataOverlay::ResolvePair(Z, A) : G4MuonicDataOverlay::Resolution();
  if (row.values != nullptr) {
    out << " answer=pair-row row=" << row.resolvedZ << "-" << row.resolvedA << " q_convention=evaluated:" << pairProfile
        << " fallback=none";
  } else if (InDomain(Z, A)) {
    out << " answer=common-formula q_convention=leading:low-z-inclusive-tree-linear-recoil-v1 fallback="
        << (pairProfile == kCompiled ? "none" : "formula-no-pair-row");
  } else {
    out << " answer=consumer fallback=outside-common-domain";
  }
  return out.str();
}

const char* G4MuonicBoundRateProvider::Prescription() {
  return "low-z-inclusive-tree-linear-recoil-v1; "
         "capture: G4MuonicAtomHelper::GetMuonCaptureRate(Z, A); "
         "decay: free * [1 - (Z alpha)^2 * (1/2 - eta)], eta = muMass / nuclearMass; "
         "domain 2 <= Z <= 13, A >= Z, finite positive masses, eta < 1/2; "
         "leading binding, linear eta prescription; physical expansion eta << 1; no accuracy certificate; "
         "an evaluated pair row of a selected pair profile takes precedence: d = q/free_tau, c = 1/tau - d";
}
