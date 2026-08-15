// Companion to harvest_d1.cc for the inputs the sweep deliberately excludes.
//
// The sweep starts at Z = 1 and A = 1 because degenerate inputs return non-finite values, and a
// NaN has no single bit pattern to hash. They still have to be recorded: what Geant4 does at
// Z = 0 and A = 0 is a registered finding about the seam, and the oracle is where a committed
// measurement belongs. Printed by CLASSIFICATION rather than as a hexfloat, for the same reason
// they are not in the digest.
#include "G4MuonMinusBoundDecay.hh"
#include <cmath>
#include <cstdio>

static const char* classify(double x) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x > 0 ? "+inf" : "-inf";
  if (x < 0) return "negative";
  if (x == 0) return "zero";
  return "positive";
}

int main() {
  const int probes[][2] = {{0, 12}, {0, 1}, {1, 0}, {-1, 12}};
  for (const auto& probe : probes) {
    double rate = G4MuonMinusBoundDecay::GetMuonCaptureRate(probe[0], probe[1]);
    std::printf("RATE %d %d %s %a\n", probe[0], probe[1], classify(rate), rate);
  }
  for (int Z : {-5, 0, 101, 120})
    std::printf("ZEFFCLAMP %d %a\n", Z, G4MuonMinusBoundDecay::GetMuonZeff(Z));
  return 0;
}
