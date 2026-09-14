// The hexfloat parser's subnormal path, exercised on every platform the validator builds on.
//
// The committed oracle carries no subnormal field, so until this check existed the branch of
// `hexfloat.hh` that parses and renders `0x0.<hex>p-1022` ran under no test. Each field below is one
// of the grammar's own spellings (`cpp/tools/build_oracle.py`, `hexfloat.hh`): the smallest and the
// largest subnormal, a negative subnormal, the smallest normal, and the two zeros. For each: the
// whole rule (`hexfloat::Problem`) accepts it, `hexfloat::Canonical` renders it back to the same
// bytes, and the parsed double is bit-identical to the reference `std::ldexp` builds from the
// literal's own mantissa and exponent -- so the expectation is computed, never typed as a decimal.
// One line per field; the first failure exits non-zero. The `FORCE_STREAM` matrix entries run it
// over the hand-rolled parser as well as over `std::from_chars`.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "hexfloat.hh"

namespace {

struct Case {
  const char* field;   // the grammar's spelling
  double expected;     // built by std::ldexp from that spelling's mantissa and exponent
  int classification;  // what std::fpclassify must report
};

std::uint64_t Bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

const char* ClassName(int classification) {
  switch (classification) {
    case FP_SUBNORMAL: return "subnormal";
    case FP_NORMAL: return "normal";
    case FP_ZERO: return "zero";
    default: return "other";
  }
}

}  // namespace

int main() {
  const Case cases[] = {
      {"0x0.0000000000001p-1022", std::ldexp(1.0, -1074), FP_SUBNORMAL},
      {"0x0.fffffffffffffp-1022", std::ldexp(std::ldexp(1.0, 52) - 1.0, -1074), FP_SUBNORMAL},
      {"-0x0.8p-1022", -std::ldexp(0.5, -1022), FP_SUBNORMAL},
      {"0x1p-1022", std::ldexp(1.0, -1022), FP_NORMAL},
      {"0x0p+0", 0.0, FP_ZERO},
      {"-0x0p+0", -0.0, FP_ZERO},
  };
#if G4MUONICDATA_HEX_FROM_CHARS
  std::printf("parser=from_chars\n");
#else
  std::printf("parser=hand-rolled\n");
#endif
  for (const Case& c : cases) {
    double value = 0.0;
    const std::string problem = hexfloat::Problem(c.field, value);
    const std::string rendered = problem.empty() ? hexfloat::Canonical(value) : std::string("(not rendered)");
    const bool accepted = problem.empty();
    const bool round_trips = rendered == c.field;
    const bool same_bits = Bits(value) == Bits(c.expected);
    const bool same_sign = std::signbit(value) == std::signbit(c.expected);
    const bool same_class = std::fpclassify(value) == c.classification;
    const bool ok = accepted && round_trips && same_bits && same_sign && same_class;
    std::printf("%s %s accepted=%s rendered=%s bits=%s sign=%s class=%s/%s%s%s\n", ok ? "PASS" : "FAIL", c.field,
                accepted ? "yes" : "no", rendered.c_str(), same_bits ? "equal" : "DIFFER", same_sign ? "equal" : "DIFFER",
                ClassName(std::fpclassify(value)), ClassName(c.classification), problem.empty() ? "" : " problem=",
                problem.c_str());
    if (!ok) return 1;
  }
  return 0;
}
