// Stand-in for Geant4's G4Types.hh: the three scalar types the overlay glue names, so the glue
// compiles and runs in CI without a Geant4 installation.
#ifndef G4TYPES_HH
#define G4TYPES_HH

typedef double G4double;
typedef int G4int;
typedef bool G4bool;

#endif
