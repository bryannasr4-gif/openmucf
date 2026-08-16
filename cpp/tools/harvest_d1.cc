#include "G4MuonMinusBoundDecay.hh"
#include <cstdio>
int main() {
  for (int Z = 1; Z <= 120; ++Z)
    for (int A = 1; A <= 300; ++A)
      std::printf("%d %d %a\n", Z, A, G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A));
  for (int Z = 0; Z <= 101; ++Z)
    std::printf("ZEFF %d %a\n", Z, G4MuonMinusBoundDecay::GetMuonZeff(Z));
  return 0;
}
