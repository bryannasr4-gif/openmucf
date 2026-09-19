// G4MuonicDataOverlay -- the Geant4-facing glue between the G4MuonicData dataset reader and the
// compiled-in muon-capture and muonic-atom energy code paths.
//
// The reader (G4MuonicDataTable) knows nothing about Geant4. This class adds what a consumer inside
// Geant4 needs: dataset discovery through G4FindDataDir, a fatal G4Exception when the opt-in was
// requested but the dataset is absent or invalid, and keyed lookups returning a pointer to the
// stored value, or nullptr when the table has no record for that key. Nothing here is reached
// unless G4MuonicDataTable::Enable() has been called: every caller tests IsEnabled() first, so the
// default build performs no lookup and no I/O.
#ifndef G4MUONICDATAOVERLAY_HH
#define G4MUONICDATAOVERLAY_HH

#include "G4MuonicDataTable.hh"
#include "G4Types.hh"

class G4MuonicDataOverlay {
 public:
  // The dataset, loaded once on first use, or nullptr after a suppressed fatal exception.
  static const G4MuonicDataTable* Table();
  // The `value` field of the nuclear_capture_rate record keyed (Z, A), in the dataset's own units
  // as declared by its #UNITS line, or nullptr when the table has no such record.
  static const G4double* Rate(G4int Z, G4int A);
  // The `value` field of the muon_zeff record keyed Z, or nullptr when the table has no such record.
  static const G4double* Zeff(G4int Z);
  // The `value` field (keV) of the k_shell_energy record keyed (Z, A), else of the natural row
  // (Z, 0), or nullptr when the selected profile carries no such table or record.
  static const G4double* KShell(G4int Z, G4int A);
  // The `e2` field (keV) of the level_energy record keyed (Z, A), else of the natural row (Z, 0),
  // with `count` set to the number of consecutive columns e2, e3, ... the table declares; nullptr
  // and a count of zero when the selected profile carries no such table or record.
  static const G4double* Levels(G4int Z, G4int A, G4int& count);
};

#endif
