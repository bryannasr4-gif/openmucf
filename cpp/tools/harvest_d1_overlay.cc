// harvest_d1.cc with the dataset opt-in switched on: the same sweep through
// G4MuonMinusBoundDecay, built against a Geant4 install carrying the overlay patch.
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicDataTable.hh"
#include <cstdio>
int main() {
  G4MuonicDataTable::Enable();
  for (int Z = 1; Z <= 120; ++Z)
    for (int A = 1; A <= 300; ++A)
      std::printf("%d %d %a\n", Z, A, G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A));
  for (int Z = 0; Z <= 101; ++Z)
    std::printf("ZEFF %d %a\n", Z, G4MuonMinusBoundDecay::GetMuonZeff(Z));
  return 0;
}
