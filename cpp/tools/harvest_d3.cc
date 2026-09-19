// harvest_d3.cc -- the muonic-atom energies as a Geant4 build computes them, without the dataset
// opt-in: the helper's K energy for every Z of the sweep box, and one capture cascade for every
// (Z, A) of the box with A >= Z, each at its own fixed seed.
//
//   K <Z> <%a>                                     G4MuonicAtomHelper::GetKShellEnergy(G4double(Z))
//   C <Z> <A> <L0> ... <L13> <n> <s1> ... <sn> edep <%a>
//
// L0..L13 are the cascade's level energies after ApplyYourself, n the number of secondaries, and
// each s the secondary's kinetic energy as `e:<%a>` (electron) or `g:<%a>` (photon), or the
// particle's name for any other definition; edep is the energy the cascade deposits. Every value
// is printed with %a, the exact hexadecimal float. Nothing else is written to stdout.
//
// Built with HARVEST_D3_NO_MAIN defined, this file provides HarvestD3() and no main, for
// harvest_d3_overlay.cc to include.
#include "G4Alpha.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4EmCaptureCascade.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4He3.hh"
#include "G4MuonMinus.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4Neutron.hh"
#include "G4Nucleus.hh"
#include "G4Proton.hh"
#include "G4ThreeVector.hh"
#include "G4Triton.hh"
#include "Randomize.hh"

#include <cstdio>

// The cascade's level array is private. It is read through the standard explicit-instantiation
// access idiom: a friend function defined inside a class template returns the template's
// pointer-to-member argument, and the explicit instantiation below names the private member --
// an explicit instantiation ignores access checks, so the friend can hand the pointer out.
struct CascadeLevels {
  typedef G4double (G4EmCaptureCascade::*type)[14];
  friend type LevelsOf(CascadeLevels);
};
template <typename Tag, typename Tag::type Member>
struct AccessPrivate {
  friend typename Tag::type LevelsOf(Tag) { return Member; }
};
template struct AccessPrivate<CascadeLevels, &G4EmCaptureCascade::fLevelEnergy>;

void HarvestD3() {
  // G4NucleiProperties caches six particle masses from the particle table. Without them, a key with Z equal to A
  // outside its mass tables gets a nuclear mass of zero.
  G4Proton::Proton(); G4Neutron::Neutron(); G4Deuteron::Deuteron();
  G4Triton::Triton(); G4Alpha::Alpha(); G4He3::He3();
  for (int Z = 1; Z <= 120; ++Z) std::printf("K %d %a\n", Z, G4MuonicAtomHelper::GetKShellEnergy(G4double(Z)));
  // Owned by the hadronic interaction registry from construction on, so never deleted here.
  G4EmCaptureCascade* cascade = new G4EmCaptureCascade();
  const G4ParticleDefinition* electron = G4Electron::Electron();
  const G4ParticleDefinition* gamma = G4Gamma::Gamma();
  for (int Z = 1; Z <= 120; ++Z) {
    for (int A = Z; A <= 300; ++A) {
      G4Random::setTheSeed(1000L * Z + A);
      G4DynamicParticle muon(G4MuonMinus::MuonMinus(), G4ThreeVector(0., 0., 1.), 0.);
      G4HadProjectile projectile(muon);
      G4Nucleus nucleus(A, Z);
      G4HadFinalState* result = cascade->ApplyYourself(projectile, nucleus);
      std::printf("C %d %d", Z, A);
      const G4double* levels = cascade->*LevelsOf(CascadeLevels());
      for (int i = 0; i < 14; ++i) std::printf(" %a", levels[i]);
      const int n = static_cast<int>(result->GetNumberOfSecondaries());
      std::printf(" %d", n);
      for (int i = 0; i < n; ++i) {
        G4DynamicParticle* particle = result->GetSecondary(i)->GetParticle();
        const G4ParticleDefinition* definition = particle->GetDefinition();
        if (definition == electron) {
          std::printf(" e:%a", particle->GetKineticEnergy());
        } else if (definition == gamma) {
          std::printf(" g:%a", particle->GetKineticEnergy());
        } else {
          std::printf(" %s", definition->GetParticleName().c_str());
        }
        delete particle;
      }
      std::printf(" edep %a\n", result->GetLocalEnergyDeposit());
    }
  }
}

#ifndef HARVEST_D3_NO_MAIN
int main() {
  HarvestD3();
  return 0;
}
#endif
