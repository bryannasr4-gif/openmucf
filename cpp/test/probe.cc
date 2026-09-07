// The toolchain probe: compiler identity, the floating-point `from_chars` feature macro, the
// number-parsing path this build compiled, and the contraction detector. It is the CI job's first
// step and refuses (non-zero exit) when the binary's own arithmetic is contracted.

#include <charconv>
#include <cstdio>
#include <string>

#include "G4MuonicDataTable.hh"
#include "gp_kernel.hh"

int main() {
#if defined(_MSC_FULL_VER)
  std::printf("compiler=MSVC _MSC_FULL_VER=%d\n", static_cast<int>(_MSC_FULL_VER));
#elif defined(__VERSION__)
  std::printf("compiler=%s\n", __VERSION__);
#else
  std::printf("compiler=unknown\n");
#endif
#if defined(__cpp_lib_to_chars)
  std::printf("__cpp_lib_to_chars=%ld\n", static_cast<long>(__cpp_lib_to_chars));
#else
  std::printf("__cpp_lib_to_chars=undefined\n");
#endif
  std::printf("parser=%s\n", G4MuonicDataTable::ParserPath().c_str());
  const bool contracted = ContractionDetected();
  std::printf("contracted=%s\n", contracted ? "YES" : "NO");
  if (contracted) {
    std::printf("%s\n", ContractionMessage());
    return 1;
  }
  return 0;
}
