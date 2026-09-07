// The Goulard-Primakoff fallback, in the shape of the upstream statement it reproduces
// (`third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc`, `GetMuonCaptureRate`). The evaluation
// order is normative (`DATASET_D1.md` section 2): `*` and `+` associate left to right, `2 * (A - Z)`
// is integer arithmetic before it meets the double, and the expression must not be contracted --
// this file is compiled with `-ffp-contract=off` for the conforming evaluation.

#include "gp_kernel.hh"

#include <cmath>

#ifndef GP_KERNEL_NS
#define GP_KERNEL_NS gp_off
#endif

namespace GP_KERNEL_NS {

GpResult Fallback(int Z, int A, const GpCoefficients& c, const std::vector<double>& zeff) {
  GpResult result;
  // The model is declared for Z >= 1 and A >= 1. Outside that box a conforming evaluation reports
  // a domain error rather than reproducing the non-finite or negative value the library returns.
  if (Z < 1 || A < 1) return result;

  const double b0a = c.b0a;
  const double b0b = c.b0b;
  const double b0c = c.b0c;
  const double t1 = c.t1;
  const double xmu_coeff = c.xmu_coeff;
  const double mix = c.mix;
  const long zc = Z < c.zmin ? c.zmin : (Z > c.zmax ? c.zmax : Z);
  const double r1 = zeff[static_cast<std::size_t>(zc)];
  const double zeff2 = r1 * r1;
  const double xmu = zeff2 * xmu_coeff;
  const double a2ze = 0.5 * double(A) / double(Z);
  const double r2 = 1.0 - xmu;
  const double lambda = t1 * zeff2 * zeff2 * (r2 * r2) * (1.0 - (1.0 - xmu) * mix) *
                        (a2ze * b0a + 1.0 - (a2ze - 1.0) * b0b -
                         double(2 * (A - Z) + std::fabs(a2ze - 1.0)) * b0c / double(A * 4));
  result.ok = true;
  result.value = lambda;
  return result;
}

GpResult Rate(int Z, int A, const GpCoefficients& c, const std::vector<double>& zeff,
              const G4MuonicDataTable::Table& capture, std::size_t value_index) {
  // Geant4's own scan of the table first: a listed (Z, A) is the tabulated value in 1/us, so
  // `/ 1000.0` gives 1/ns.
  if (const G4MuonicDataTable::Table::Record* hit = capture.Lookup({static_cast<long>(Z), static_cast<long>(A)})) {
    GpResult result;
    result.ok = true;
    result.value = hit->floats[value_index] / 1000.0;
    return result;
  }
  return Fallback(Z, A, c, zeff);
}

}  // namespace GP_KERNEL_NS
