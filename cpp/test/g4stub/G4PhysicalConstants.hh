// Stand-in for Geant4's G4PhysicalConstants.hh: the one constant the bound-rate provider names, with
// the bits CLHEP's definition evaluates to in a Geant4 build, as the decay-rate oracle
// (data/g4/d1/d1_decay_rate.oracle, line `K alpha`) records them; bound_rate_check.cc compares the two.
#ifndef G4PHYSICALCONSTANTS_HH
#define G4PHYSICALCONSTANTS_HH

static constexpr double fine_structure_const = 0x1.de3d429d88bb3p-8;

#endif
