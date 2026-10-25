// repro_bound_rates.cc -- the capture and decay rates a stock Geant4 build gives a negative muon in the 1s state
// of a muonic atom on its two stopped-muon routes, read through public interfaces only.
//
//   stopping route  G4MuonMinusBoundDecay::ApplyYourself draws its delay and branch from
//                   GetMuonCaptureRate(Z, A) and GetMuonDecayRate(Z, A, muMass, G4Nucleus::AtomicMass(A, Z));
//   atomic route    G4IonTable::GetMuonicAtom(Z, A) builds the muonic atom through
//                   G4MuonicAtomHelper::ConstructMuonicAtom, which stores its lifetime and its DIO and
//                   nuclear-capture lifetimes; G4MuonicAtomDecay draws the delay and branch from them.
//
// For each (Z, A) on the command line (default: the nine nuclei below) it prints one line of named fields, every
// value as an exact hexadecimal float (%a) followed by a decimal rendering:
//
//   Z A mass_atomic mass_nuclei capture_stopping capture_helper decay_stopping decay_helper
//   lifetime_stopping lifetime_atom dio_lifetime_atom nc_lifetime_atom p_capture_stopping p_dio_atom
//
// mass_atomic is G4Nucleus::AtomicMass(A, Z), the mass the stopping caller passes; mass_nuclei is
// G4NucleiProperties::GetNuclearMass(A, Z). lifetime_stopping is 1 / (capture + decay) as the stopping caller
// sums them; the three atom lifetimes are the ones the constructed G4MuonicAtom returns. Each route's branch test
// compares a uniform number with one ratio, printed as formed there: p_capture_stopping is capture / (capture +
// decay), the stopping caller choosing capture below it; p_dio_atom is (1/dio) / (1/dio + 1/nc), the muonic atom's
// decay choosing DIO below it. Rates are in ns^-1 and masses in MeV, Geant4's internal units. The file contains no decay or capture
// formula: every rate comes out of the Geant4 libraries it is linked against.
#include "G4Alpha.hh"
#include "G4Deuteron.hh"
#include "G4Electron.hh"
#include "G4GenericIon.hh"
#include "G4GenericMuonicAtom.hh"
#include "G4He3.hh"
#include "G4IonTable.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtom.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4Neutron.hh"
#include "G4NeutrinoMu.hh"
#include "G4AntiNeutrinoE.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4Proton.hh"
#include "G4Triton.hh"
#include "G4Version.hh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

void Field(const char* name, double value) { std::printf(" %s=%a(%.17g)", name, value, value); }

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::pair<int, int>> nuclei;
  if (argc > 1) {
    if (argc % 2 != 1) {
      std::fprintf(stderr, "usage: repro_bound_rates [Z A]...\n");
      return 2;
    }
    for (int i = 1; i + 1 < argc; i += 2) nuclei.emplace_back(std::atoi(argv[i]), std::atoi(argv[i + 1]));
  } else {
    nuclei = {{6, 12}, {6, 13}, {7, 14}, {8, 16}, {8, 18}, {2, 4}, {1, 1}, {13, 27}, {14, 28}};
  }
  // G4NucleiProperties caches six particle masses from the particle table; without them a nucleus outside its
  // mass tables gets a wrong mass. The muonic atom's decay table names the electron and the two neutrinos.
  G4Proton::Proton(); G4Neutron::Neutron(); G4Deuteron::Deuteron();
  G4Triton::Triton(); G4Alpha::Alpha(); G4He3::He3();
  G4Electron::Electron(); G4AntiNeutrinoE::AntiNeutrinoE(); G4NeutrinoMu::NeutrinoMu();
  G4GenericIon::GenericIon(); G4GenericMuonicAtom::GenericMuonicAtom();
  const G4double muMass = G4MuonMinus::MuonMinus()->GetPDGMass();
  for (const char* name : {"neutron", "deuteron", "triton", "alpha", "He3", "proton", "GenericIon", "GenericMuonicAtom"}) {
    if (G4ParticleTable::GetParticleTable()->FindParticle(name) == nullptr) {
      std::fprintf(stderr, "repro_bound_rates: the particle table lacks %s\n", name);
      return 2;
    }
  }
  // G4IonTable creates an ion only once G4GenericIon has a process manager, which a physics list's
  // InitializeProcessManager gives every particle; this program has no physics list, so it gives the two generic
  // particles theirs the same way.
  for (G4ParticleDefinition* generic : {static_cast<G4ParticleDefinition*>(G4GenericIon::GenericIon()),
                                        static_cast<G4ParticleDefinition*>(G4GenericMuonicAtom::GenericMuonicAtom())}) {
    if (generic->GetProcessManager() == nullptr) generic->SetProcessManager(new G4ProcessManager(generic));
  }
  G4ParticleTable::GetParticleTable()->SetReadiness();
  std::printf("VERSION %s\n", G4VERSION_TAG);
  std::printf("MU_MASS %a(%.17g)\n", muMass, muMass);
  const G4Nucleus nucleus;
  for (const auto& [Z, A] : nuclei) {
    const G4double massAtomic = nucleus.AtomicMass(A, Z);
    const G4double massNuclei = G4NucleiProperties::GetNuclearMass(A, Z);
    const G4double cStop = G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A);
    const G4double dStop = G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, massAtomic);
    const G4double cHelp = G4MuonicAtomHelper::GetMuonCaptureRate(Z, A);
    const G4double dHelp = G4MuonicAtomHelper::GetMuonDecayRate(Z);
    const G4double lambdaStop = cStop + dStop;
    const auto* atom = static_cast<const G4MuonicAtom*>(G4IonTable::GetIonTable()->GetMuonicAtom(Z, A));
    if (atom == nullptr) {
      std::fprintf(stderr, "repro_bound_rates: no muonic atom for Z=%d A=%d\n", Z, A);
      return 2;
    }
    const G4double l1 = 1. / atom->GetDIOLifeTime();
    const G4double l2 = 1. / atom->GetNCLifeTime();
    std::printf("NUCLEUS Z=%d A=%d", Z, A);
    Field("mass_atomic", massAtomic);
    Field("mass_nuclei", massNuclei);
    Field("capture_stopping", cStop);
    Field("capture_helper", cHelp);
    Field("decay_stopping", dStop);
    Field("decay_helper", dHelp);
    Field("lifetime_stopping", 1. / lambdaStop);
    Field("lifetime_atom", atom->GetPDGLifeTime());
    Field("dio_lifetime_atom", atom->GetDIOLifeTime());
    Field("nc_lifetime_atom", atom->GetNCLifeTime());
    Field("p_capture_stopping", cStop / lambdaStop);
    Field("p_dio_atom", l1 / (l1 + l2));
    std::printf("\n");
  }
  return 0;
}
