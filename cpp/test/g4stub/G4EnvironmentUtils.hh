// Stand-in for Geant4's G4EnvironmentUtils.hh: the data-directory lookup the overlay glue calls.
// The definition lives in overlay_check.cc.
#ifndef G4ENVIRONMENTUTILS_HH
#define G4ENVIRONMENTUTILS_HH

const char* G4FindDataDir(const char* name);

#endif
