// Stand-in for Geant4's G4MuonicAtomHelper.hh: the one function the bound-rate provider calls, the
// helper's capture resolver. The definition lives in bound_rate_check.cc, which counts the calls and
// returns values the check chooses.
#ifndef G4MUONICATOMHELPER_HH
#define G4MUONICATOMHELPER_HH

#include "G4Types.hh"

class G4MuonicAtomHelper {
 public:
  static G4double GetMuonCaptureRate(G4int Z, G4int A);
};

#endif
