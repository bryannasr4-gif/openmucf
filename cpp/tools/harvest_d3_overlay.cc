// harvest_d3.cc with the dataset opt-in switched on, then the helper's two-argument K energy over
// the same box, built against a Geant4 install carrying the overlay patch:
//
//   every line harvest_d3.cc prints, then
//   KA <Z> <A> <%a>                                G4MuonicAtomHelper::GetKShellEnergy(G4double(Z), A)
#define HARVEST_D3_NO_MAIN
#include "harvest_d3.cc"

#include "G4HadronicParameters.hh"

int main() {
  G4HadronicParameters::Instance()->SetEnableMuonicData(true);
  HarvestD3();
  for (int Z = 1; Z <= 120; ++Z)
    for (int A = Z; A <= 300; ++A)
      std::printf("KA %d %d %a\n", Z, A, G4MuonicAtomHelper::GetKShellEnergy(G4double(Z), A));
  return 0;
}
