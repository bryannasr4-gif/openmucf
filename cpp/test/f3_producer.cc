// The F-3 producer: the same model, compiled twice in one binary -- `gp_off` with contraction
// disabled and `gp_fast` with contraction enabled (plus FMA on x86-64) -- the expression evaluated
// at every point of the sweep box the oracle header names (the table hits are not consulted: F-3 is
// about the expression), and the difference summarised as the figures `DATASET_D1.md` F-3 states:
// points, points that differ, points differing by more than 1 ulp, the maximum ulp distance and
// where it occurs. `check_f3.py` compares this output with the document.
//
// Usage: g4muonicdata_f3 <dataset-dir> --oracle <file> [--output <file>]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "G4MuonicDataTable.hh"
#include "gp_kernel.hh"

#ifndef G4MUONICDATA_CXX_ID
#define G4MUONICDATA_CXX_ID "unknown"
#endif

namespace {

struct Box {
  long z_min = 0, z_max = 0, a_min = 0, a_max = 0;
};

bool ParseLong(const std::string& s, long& out) {
  if (s.empty()) return false;
  std::size_t i = 0;
  const bool negative = s[0] == '-';
  if (negative) ++i;
  if (i >= s.size()) return false;
  long value = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    value = value * 10 + (s[i] - '0');
    if (value > 1000000) return false;
  }
  out = negative ? -value : value;
  return true;
}

// `# sweep             Z 1..120 x A 1..300 = 36000 points, row-major, Z ascending outermost`
bool ParseSweepHeader(const std::string& oracle_path, Box& box) {
  std::ifstream in(oracle_path, std::ios::binary);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, 7, "# sweep") != 0) continue;
    const std::size_t z = line.find('Z');
    const std::size_t x = line.find(" x A ");
    const std::size_t eq = line.find(" = ");
    if (z == std::string::npos || x == std::string::npos || eq == std::string::npos) return false;
    auto range = [](const std::string& text, long& lo, long& hi) {
      const std::size_t dots = text.find("..");
      if (dots == std::string::npos) return false;
      return ParseLong(text.substr(0, dots), lo) && ParseLong(text.substr(dots + 2), hi);
    };
    return range(line.substr(z + 2, x - (z + 2)), box.z_min, box.z_max) &&
           range(line.substr(x + 5, eq - (x + 5)), box.a_min, box.a_max);
  }
  return false;
}

bool FmaAvailable() {
#if defined(__x86_64__) || defined(__amd64__)
  return __builtin_cpu_supports("fma") != 0;
#elif defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

std::int64_t OrderedBits(double x) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &x, sizeof bits);
  return (bits >> 63) ? -static_cast<std::int64_t>(bits & 0x7fffffffffffffffull) : static_cast<std::int64_t>(bits);
}

std::uint64_t UlpDistance(double a, double b) {
  const std::int64_t oa = OrderedBits(a), ob = OrderedBits(b);
  return oa > ob ? static_cast<std::uint64_t>(oa - ob) : static_cast<std::uint64_t>(ob - oa);
}

bool LoadCoefficients(const G4MuonicDataTable::Table& capture, GpCoefficients& c) {
  const std::string* fallback = capture.Directive("FALLBACK");
  if (!fallback) return false;
  std::vector<std::string> words;
  std::size_t i = 0;
  while (i < fallback->size()) {
    while (i < fallback->size() && (fallback->at(i) == ' ' || fallback->at(i) == '\t')) ++i;
    std::size_t j = i;
    while (j < fallback->size() && fallback->at(j) != ' ' && fallback->at(j) != '\t') ++j;
    if (j > i) words.push_back(fallback->substr(i, j - i));
    i = j;
  }
  if (words.empty() || words[0] != "goulard_primakoff") return false;
  int seen = 0;
  for (std::size_t k = 1; k < words.size(); ++k) {
    const std::size_t eq = words[k].find('=');
    if (eq == std::string::npos) return false;
    const std::string name = words[k].substr(0, eq), text = words[k].substr(eq + 1);
    double value = 0.0;
    if (!G4MuonicDataTable::ParseDouble(text, value).empty()) return false;
    long integer = 0;
    if (name == "b0a") c.b0a = value;
    else if (name == "b0b") c.b0b = value;
    else if (name == "b0c") c.b0c = value;
    else if (name == "t1") c.t1 = value;
    else if (name == "xmu_coeff") c.xmu_coeff = value;
    else if (name == "mix") c.mix = value;
    else if (name == "zmin" && ParseLong(text, integer)) c.zmin = integer;
    else if (name == "zmax" && ParseLong(text, integer)) c.zmax = integer;
    else return false;
    ++seen;
  }
  return seen == 8;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dataset, oracle, output;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--oracle" && i + 1 < argc) oracle = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output = argv[++i];
    else if (dataset.empty()) dataset = arg;
    else { std::fprintf(stderr, "usage: g4muonicdata_f3 <dataset-dir> --oracle <file> [--output <file>]\n"); return 2; }
  }
  if (dataset.empty() || oracle.empty()) {
    std::fprintf(stderr, "usage: g4muonicdata_f3 <dataset-dir> --oracle <file> [--output <file>]\n");
    return 2;
  }
  auto emit = [&](const std::string& line) {
    std::printf("%s\n", line.c_str());
    if (!output.empty()) {
      std::ofstream out(output, std::ios::binary);
      out << line << "\n";
    }
  };
  if (ContractionDetected()) {
    std::printf("%s\n", ContractionMessage());
    return 1;
  }
  if (!FmaAvailable()) {
    emit("F3 SKIPPED: no FMA on this CPU");
    return 0;
  }
  try {
    const G4MuonicDataTable tables = G4MuonicDataTable::Load(dataset);
    const G4MuonicDataTable::Table* capture = tables.Find("nuclear_capture_rate");
    const G4MuonicDataTable::Table* zeff_table = tables.Find("muon_zeff");
    if (!capture || !zeff_table) { std::fprintf(stderr, "dataset lacks nuclear_capture_rate or muon_zeff\n"); return 1; }
    GpCoefficients c;
    if (!LoadCoefficients(*capture, c)) { std::fprintf(stderr, "cannot parse #FALLBACK\n"); return 1; }
    std::vector<double> zeff;
    for (const G4MuonicDataTable::Table::Record& record : zeff_table->records) {
      if (record.keys.size() != 1 || record.keys[0] != static_cast<long>(zeff.size())) {
        std::fprintf(stderr, "muon_zeff keys are not 0..N contiguous\n");
        return 1;
      }
      zeff.push_back(record.floats[0]);
    }
    Box box;
    if (!ParseSweepHeader(oracle, box)) { std::fprintf(stderr, "cannot read the sweep box from %s\n", oracle.c_str()); return 1; }

    std::uint64_t points = 0, differ = 0, over_one = 0, max_ulp = 0;
    long max_z = 0, max_a = 0;
    double max_rel = 0.0;
    for (long z = box.z_min; z <= box.z_max; ++z) {
      for (long a = box.a_min; a <= box.a_max; ++a) {
        const GpResult off = gp_off::Fallback(static_cast<int>(z), static_cast<int>(a), c, zeff);
        const GpResult fast = gp_fast::Fallback(static_cast<int>(z), static_cast<int>(a), c, zeff);
        if (!off.ok || !fast.ok) { std::fprintf(stderr, "domain error inside the sweep box at (%ld, %ld)\n", z, a); return 1; }
        ++points;
        const std::uint64_t ulp = UlpDistance(off.value, fast.value);
        if (ulp == 0) continue;
        ++differ;
        if (ulp > 1) ++over_one;
        if (ulp > max_ulp) { max_ulp = ulp; max_z = z; max_a = a; }
        if (off.value != 0.0) {
          const double rel = std::fabs(off.value - fast.value) / std::fabs(off.value);
          if (rel > max_rel) max_rel = rel;
        }
      }
    }
    char line[256];
    std::snprintf(line, sizeof line, "F3 cxx=%s points=%llu differ=%llu over1ulp=%llu max_ulp=%llu at=(%ld,%ld) max_rel=%.3e",
                  G4MUONICDATA_CXX_ID, static_cast<unsigned long long>(points), static_cast<unsigned long long>(differ),
                  static_cast<unsigned long long>(over_one), static_cast<unsigned long long>(max_ulp), max_z, max_a, max_rel);
    emit(line);
    return 0;
  } catch (const G4MuonicDataTable::Error& error) {
    std::fprintf(stderr, "%s\n", error.what().c_str());
    return 1;
  }
}
