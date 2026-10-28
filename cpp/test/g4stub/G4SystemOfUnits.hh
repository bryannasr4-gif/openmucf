// Stand-in for Geant4's G4SystemOfUnits.hh: the two units the bound-rate provider names. Geant4's
// internal time unit is the nanosecond, so a nanosecond is 1 and a microsecond is 1000.
#ifndef G4SYSTEMOFUNITS_HH
#define G4SYSTEMOFUNITS_HH

static constexpr double ns = 1.;
static constexpr double microsecond = 1000.;

#endif
