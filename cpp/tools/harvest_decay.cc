// harvest_decay.cc -- the rate at which a negative muon bound in a muonic atom decays, as the two routes of a
// Geant4 build compute it, with the inputs one of them is passed.
//
//   K mu_mass <%a>          the negative muon's mass, G4MuonMinus::MuonMinus()->GetPDGMass()
//   K alpha <%a>            CLHEP::fine_structure_const
//   K microsecond <%a>      CLHEP::microsecond
//   H <Z> <%a>              G4MuonicAtomHelper::GetMuonDecayRate(Z), for Z = -1..121
//   B <Z> <A> <%a> <%a>     the nuclear mass G4Nucleus::AtomicMass(A, Z) returns, and
//                           G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, mu_mass, that mass), the call
//                           G4MuonMinusBoundDecay::ApplyYourself makes: for Z = 1..13 at A = Z..3Z, and for
//                           Z = 14..120 at A = Z..300
//
// Every value is printed with %a, the exact hexadecimal float. Nothing else is written to stdout.
#include "G4Alpha.hh"
#include "G4Deuteron.hh"
#include "G4He3.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4Neutron.hh"
#include "G4Nucleus.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4Triton.hh"

#include <cstdio>
#include <initializer_list>

int main() {
  // G4NucleiProperties caches six particle masses from the particle table; without them a nucleus outside
  // its mass tables gets a wrong mass.
  G4Proton::Proton(); G4Neutron::Neutron(); G4Deuteron::Deuteron();
  G4Triton::Triton(); G4Alpha::Alpha(); G4He3::He3();
  for (const char* name : {"neutron", "deuteron", "triton", "alpha", "He3", "proton"}) {
    if (G4ParticleTable::GetParticleTable()->FindParticle(name) == nullptr) {
      std::fprintf(stderr, "harvest_decay: the particle table lacks %s\n", name);
      return 2;
    }
  }
  const G4double muMass = G4MuonMinus::MuonMinus()->GetPDGMass();
  std::printf("K mu_mass %a\n", muMass);
  std::printf("K alpha %a\n", CLHEP::fine_structure_const);
  std::printf("K microsecond %a\n", CLHEP::microsecond);
  for (int Z = -1; Z <= 121; ++Z) std::printf("H %d %a\n", Z, G4MuonicAtomHelper::GetMuonDecayRate(Z));
  const G4Nucleus nucleus;
  for (int Z = 1; Z <= 120; ++Z) {
    const int lastA = Z < 14 ? 3 * Z : 300;
    for (int A = Z; A <= lastA; ++A) {
      const G4double mass = nucleus.AtomicMass(A, Z);
      std::printf("B %d %d %a %a\n", Z, A, mass, G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, mass));
    }
  }
  return 0;
}
