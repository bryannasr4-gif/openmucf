// harvest_bound_rates.cc -- what a Geant4 build carrying cpp/patches/ gives each consumer of the bound-rate pair, over
// the provider's domain and its edges, with the two opt-ins as the command line sets them.
//
//   harvest_bound_rates <dataset on|off> <bound rates on|off>
//
// The dataset opt-in reads the profile G4MUONICDATA_D1_PROFILE selects, as an application's would. For each Z of
// 1..14 and each A of Z..3Z, and for (82, 208), one line, every value as an exact hexadecimal float (%a):
//
//   K <Z> <A> <G4Nucleus::AtomicMass(A, Z)> <G4NucleiProperties::GetNuclearMass(A, Z)>
//     <G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A)> <G4MuonicAtomHelper::GetMuonCaptureRate(Z, A)>
//     <status> <capture> <decay> <lifetime> <DIO lifetime> <NC lifetime> <codes>
//
// <status> (0 common, 1 disabled, 2 unsupported, 3 refused), <capture> and <decay> come from
// G4MuonicBoundRateProvider::Rates(Z, A, muon mass, AtomicMass(A, Z)), the call the stopping route makes (`x`
// unless common); the three lifetimes from the muonic atom G4IonTable::GetMuonicAtom(Z, A) constructs, which the
// helper builds through the provider when the switch is on; <codes> lists the G4Exception codes raised while the
// line was formed, or `-`. An exception handler records each code; it lets a G4MuonicBoundRates001 refusal return
// to its caller, so the sweep continues past a key the provider refuses, and aborts on any other fatal code.
//
// Then one line LIFECYCLE <enabled before> <frozen> <enabled after> <codes>: in the unlocked pre-initialisation
// state, after the sweep has frozen the switch, G4HadronicParameters::SetEnableMuonicBoundRates is called with
// the opposite value; the provider must refuse it with G4MuonicBoundRates002 and keep its state.
#include "G4Alpha.hh"
#include "G4AntiNeutrinoE.hh"
#include "G4Deuteron.hh"
#include "G4Electron.hh"
#include "G4GenericIon.hh"
#include "G4GenericMuonicAtom.hh"
#include "G4HadronicParameters.hh"
#include "G4He3.hh"
#include "G4IonTable.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtom.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4MuonicBoundRateProvider.hh"
#include "G4Neutron.hh"
#include "G4NeutrinoMu.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4Proton.hh"
#include "G4StateManager.hh"
#include "G4Triton.hh"
#include "G4VExceptionHandler.hh"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

class Recorder : public G4VExceptionHandler {
 public:
  G4bool Notify(const char*, const char* code, G4ExceptionSeverity severity, const char*) override {
    codes.emplace_back(code);
    const bool fatal = severity == FatalException || severity == FatalErrorInArgument;
    return fatal && std::strcmp(code, "G4MuonicBoundRates001") != 0 && std::strcmp(code, "G4MuonicBoundRates002") != 0;
  }
  std::string Take() {
    std::string out;
    for (const auto& c : codes) out += (out.empty() ? "" : ",") + c;
    codes.clear();
    return out.empty() ? "-" : out;
  }
  std::vector<std::string> codes;
};

std::string Hex(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%a", x);
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: harvest_bound_rates <dataset on|off> <bound rates on|off>\n");
    return 2;
  }
  const bool dataset = std::strcmp(argv[1], "on") == 0;
  const bool switch_on = std::strcmp(argv[2], "on") == 0;
  auto* recorder = new Recorder;
  G4StateManager::GetStateManager()->SetExceptionHandler(recorder);
  G4Proton::Proton(); G4Neutron::Neutron(); G4Deuteron::Deuteron();
  G4Triton::Triton(); G4Alpha::Alpha(); G4He3::He3();
  G4Electron::Electron(); G4AntiNeutrinoE::AntiNeutrinoE(); G4NeutrinoMu::NeutrinoMu();
  for (G4ParticleDefinition* generic : {static_cast<G4ParticleDefinition*>(G4GenericIon::GenericIon()),
                                        static_cast<G4ParticleDefinition*>(G4GenericMuonicAtom::GenericMuonicAtom())}) {
    if (generic->GetProcessManager() == nullptr) generic->SetProcessManager(new G4ProcessManager(generic));
  }
  G4ParticleTable::GetParticleTable()->SetReadiness();
  if (dataset) G4HadronicParameters::Instance()->SetEnableMuonicData(true);
  if (switch_on) G4HadronicParameters::Instance()->SetEnableMuonicBoundRates(true);
  std::printf("CONFIG dataset %s switch %d frozen %d codes %s\n", argv[1], G4MuonicBoundRateProvider::IsEnabled(),
              G4MuonicBoundRateProvider::IsFrozen(), recorder->Take().c_str());
  const double muMass = G4MuonMinus::MuonMinus()->GetPDGMass();
  std::vector<std::pair<int, int>> keys;
  for (int Z = 1; Z <= 14; ++Z) {
    for (int A = Z; A <= 3 * Z; ++A) keys.emplace_back(Z, A);
  }
  keys.emplace_back(82, 208);
  const G4Nucleus nucleus;
  for (const auto& [Z, A] : keys) {
    const double massAtomic = nucleus.AtomicMass(A, Z);
    const double massNuclei = G4NucleiProperties::GetNuclearMass(A, Z);
    const double cBound = G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A);
    const double cHelper = G4MuonicAtomHelper::GetMuonCaptureRate(Z, A);
    G4MuonicBoundRates rates;
    const auto status = G4MuonicBoundRateProvider::Rates(Z, A, muMass, massAtomic, rates);
    const bool common = status == G4MuonicBoundRateProvider::Status::kCommon;
    const auto* atom = static_cast<const G4MuonicAtom*>(G4IonTable::GetIonTable()->GetMuonicAtom(Z, A));
    if (atom == nullptr) {
      std::fprintf(stderr, "harvest_bound_rates: no muonic atom for Z=%d A=%d\n", Z, A);
      return 2;
    }
    std::printf("K %d %d %s %s %s %s %d %s %s %s %s %s %s\n", Z, A, Hex(massAtomic).c_str(), Hex(massNuclei).c_str(),
                Hex(cBound).c_str(), Hex(cHelper).c_str(), static_cast<int>(status),
                common ? Hex(rates.capture).c_str() : "x", common ? Hex(rates.decay).c_str() : "x",
                Hex(atom->GetPDGLifeTime()).c_str(), Hex(atom->GetDIOLifeTime()).c_str(),
                Hex(atom->GetNCLifeTime()).c_str(), recorder->Take().c_str());
  }
  const bool before = G4MuonicBoundRateProvider::IsEnabled();
  const bool frozen = G4MuonicBoundRateProvider::IsFrozen();
  G4HadronicParameters::Instance()->SetEnableMuonicBoundRates(!before);
  std::printf("LIFECYCLE %d %d %d %s\n", before, frozen, G4MuonicBoundRateProvider::IsEnabled(),
              recorder->Take().c_str());
  return 0;
}
