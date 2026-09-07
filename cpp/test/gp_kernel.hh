// The Goulard-Primakoff fallback of the D1 dataset, evaluated in C++ in the association order
// `DATASET_D1.md` section 2 declares -- and the contraction detector every parity claim runs first.
//
// `gp_kernel.cc` is the only translation unit that evaluates the model. It is compiled once with
// contraction disabled (namespace `gp_off`, the conforming evaluation) and, on GCC/Clang, a second
// time with contraction enabled (namespace `gp_fast`) for the F-3 producer, which measures how far a
// contracted build moves the same expression. `GP_KERNEL_NS` selects the namespace per compilation.
#ifndef G4MUONICDATA_GP_KERNEL_HH
#define G4MUONICDATA_GP_KERNEL_HH

#include <cmath>
#include <cstddef>
#include <vector>

#include "G4MuonicDataTable.hh"

// The eight `#FALLBACK` inputs, parsed from the directive with the reader's own number parser.
struct GpCoefficients {
  double b0a = 0.0;
  double b0b = 0.0;
  double b0c = 0.0;
  double t1 = 0.0;
  double xmu_coeff = 0.0;
  double mix = 0.0;
  long zmin = 0;
  long zmax = 0;
};

// `ok == false` is the domain error a conforming evaluation reports for Z < 1 or A < 1; the value
// is meaningful only when `ok` is true.
struct GpResult {
  bool ok = false;
  double value = 0.0;
};

// `Fallback` is the Goulard-Primakoff expression itself for (Z, A) -- what F-3 measures at every
// point of the box. `Rate` is the capture rate in 1/ns as Geant4 computes it: a table hit is
// `value / 1000.0`, a miss is `Fallback`. `zeff` is indexed by Z (the `muon_zeff` table, Z = 0 ..
// its last key); `capture` is the `nuclear_capture_rate` table and `value_index` the position of
// its `value` column among the float columns.
namespace gp_off {
GpResult Fallback(int Z, int A, const GpCoefficients& c, const std::vector<double>& zeff);
GpResult Rate(int Z, int A, const GpCoefficients& c, const std::vector<double>& zeff,
              const G4MuonicDataTable::Table& capture, std::size_t value_index);
}
namespace gp_fast {
GpResult Fallback(int Z, int A, const GpCoefficients& c, const std::vector<double>& zeff);
GpResult Rate(int Z, int A, const GpCoefficients& c, const std::vector<double>& zeff,
              const G4MuonicDataTable::Table& capture, std::size_t value_index);
}

// The contraction self-test (V-01). With a = 1 + 2^-30, b = 1 - 2^-30 and c = -1, `a * b + c` is
// exactly 0 when the product is rounded before the addition and -2^-60 when the two are fused.
// Every operand is volatile so the compiler cannot fold it; the function is inline so each
// binary measures the flags IT was compiled with.
inline bool ContractionDetected() {
  const double epsilon = std::ldexp(1.0, -30);
  volatile double a = 1.0 + epsilon;
  volatile double b = 1.0 - epsilon;
  volatile double c = -1.0;
  volatile double result = a * b + c;
  return result != 0.0;
}

// The exact message a contracted binary prints before exiting non-zero.
inline const char* ContractionMessage() {
  return "floating-point contraction detected; this binary cannot make a parity claim";
}

#endif
