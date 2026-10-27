// G4MuonicDataOverlay -- implementation. See the header for what this class is for.
#include "G4MuonicDataOverlay.hh"

#include "G4EnvironmentUtils.hh"
#include "G4Exception.hh"
#include "G4MuonicDataSemantics.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Index = std::vector<double>::size_type;
using Table = G4MuonicDataTable::Table;
using Origin = G4MuonicDataOverlay::Origin;
using Resolution = G4MuonicDataOverlay::Resolution;

// The reserved token of a seam with no tables: every request there falls through to the consumer's
// compiled-in code, and a configuration in which both seams carry it searches no directory.
const char* const kCompiledProfile = "compiled";
const char* const kCaptureTable = "nuclear_capture_rate";
const char* const kZeffTable = "muon_zeff";
const char* const kKShellTable = "k_shell_energy";
const char* const kLevelTable = "level_energy";
// The evaluated bound-rate pair table and its three value columns, in the order the provider reads them.
const char* const kPairTable = "bound_rate_primitives";
const char* const kPairColumns[] = {"tau", "q", "free_tau"};
const char* const kPairUnits[] = {"ns", "1", "ns"};
const char* const kRepresentative = "most_abundant_and_listed";
// The number of consecutive muonic-cascade levels a patched build assembles.
const G4int kCascadeLevels = 14;

// One table a lookup reads, once it has been selected and checked.
struct Selected {
  const Table* table = nullptr;
  // The position of the first value column inside Record::floats: #COLUMNS minus the key columns.
  Index value = 0;
  // How many consecutive doubles from that position the lookup hands back.
  G4int count = 0;
  // The table's `#VALIDITY` `A:` assignment, which names what its `A = 0` records mean.
  std::string convention;
};

// Everything the environment selected, read once.
struct State {
  G4MuonicDataOverlay::Configuration config;
  G4MuonicDataTable dataset;
  bool loaded = false;
  Selected capture, zeff, kshell, levels;
  // The pair table selected by G4MUONICDATA_D1_PAIR_PROFILE, independently of the capture profile.
  Selected pairs;
  bool d1Evaluated = false;
  // Non-empty when the semantic layer refused the dataset and an installed handler suppressed the
  // abort: every entry point then throws instead of answering from a dataset no consumer may read.
  std::string refusal;
};

// The index of `name` among the non-key columns, or false when the table declares no such column.
bool FindColumn(const Table& table, const std::string& name, Index& index) {
  Index position = 0;
  for (const std::string& column : table.columns) {
    if (column == name) {
      index = position;
      return true;
    }
    if (column != "Z" && column != "A") ++position;
  }
  return false;
}

// The space- and tab-separated tokens of a directive value.
std::vector<std::string> Tokens(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  for (char c : text) {
    if (c == ' ' || c == '\t') {
      if (!current.empty()) {
        out.push_back(current);
        current.clear();
      }
    } else {
      current += c;
    }
  }
  if (!current.empty()) out.push_back(current);
  return out;
}

// The value a `name<delimiter>value` directive assigns to `name`, or "" when it assigns none.
std::string Assigned(const Table& table, const char* keyword, const std::string& name, char delimiter) {
  const std::string* value = table.Directive(keyword);
  if (value == nullptr) return "";
  const std::string prefix = name + delimiter;
  for (const std::string& token : Tokens(*value)) {
    if (token.size() > prefix.size() && token.compare(0, prefix.size(), prefix) == 0) {
      return token.substr(prefix.size());
    }
  }
  return "";
}

// True for a level column name: `e` followed by digits only.
bool IsLevelColumn(const std::string& column) {
  return column.size() > 1 && column[0] == 'e' &&
         std::all_of(column.begin() + 1, column.end(), [](char c) { return c >= '0' && c <= '9'; });
}

void Fatal(const char* code, const std::string& text) {
  G4Exception("G4MuonicDataOverlay", code, FatalException, text.c_str());
}

// The unit each value column a lookup reads must carry, by table.
struct UnitRule {
  const char* table;
  const char* column;
  const char* unit;
};
const UnitRule kUnitRules[] = {
    {kCaptureTable, "value", "1e6/s"},
    {kZeffTable, "value", "dimensionless"},
    {kKShellTable, "value", "keV"},
};
const char* const kLevelUnit = "keV";

// False, after a fatal exception naming the file, table, column and unit, when `column` of `table`
// is absent or its #UNITS assignment is not `unit`.
bool CheckUnit(const std::string& dir, const Table& table, const std::string& column, const std::string& unit) {
  Index index = 0;
  if (!FindColumn(table, column, index)) {
    Fatal("G4MuonicData003", "dataset at " + dir + ": file " + table.file + ", #TABLE " + table.name +
                                 " declares no column '" + column + "' (unit " + unit + ") in #COLUMNS");
    return false;
  }
  const std::string declared = Assigned(table, "UNITS", column, '=');
  if (declared != unit) {
    Fatal("G4MuonicData003", "dataset at " + dir + ": file " + table.file + ", #TABLE " + table.name +
                                 ", column '" + column + "' has unit '" + declared + "' in #UNITS; the lookup reads it as '" +
                                 unit + "'");
    return false;
  }
  return true;
}

// The value of an environment variable, or "" when it is unset or empty -- an empty variable is
// unset, and nothing is trimmed or corrected.
std::string Variable(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

// How a seam's profile was named, which decides what the dataset must carry for it.
enum class Source { PerSeam, Legacy, Default };

struct Choice {
  std::string profile;
  Source source = Source::Default;
};

Choice Choose(const char* variable, const std::string& legacy, const char* fallback) {
  Choice choice;
  const std::string named = Variable(variable);
  if (!named.empty()) {
    choice.profile = named;
    choice.source = Source::PerSeam;
  } else if (!legacy.empty()) {
    choice.profile = legacy;
    choice.source = Source::Legacy;
  } else {
    choice.profile = fallback;
    choice.source = Source::Default;
  }
  return choice;
}

// Select one table of a profile, check the unit of every column a lookup reads, and record where
// its values sit. False after a fatal exception.
bool Select(const State& state, const std::string& dir, const std::string& profile, const char* name,
            Selected& selected) {
  const Table* table = state.dataset.Find(profile, name);
  if (table == nullptr) return true;  // the caller decides whether that is admissible
  Index index = 0;
  const bool level = std::string(name) == kLevelTable;
  if (level) {
    for (const std::string& column : table->columns) {
      if (IsLevelColumn(column) && !CheckUnit(dir, *table, column, kLevelUnit)) return false;
    }
    if (!CheckUnit(dir, *table, "e2", kLevelUnit)) return false;
    FindColumn(*table, "e2", index);
    G4int count = 0;
    for (std::vector<std::string>::const_iterator it = std::find(table->columns.begin(), table->columns.end(), "e2");
         it != table->columns.end(); ++it, ++count) {
      if (*it != "e" + std::to_string(count + 2)) break;
    }
    selected.count = count;
  } else {
    if (!FindColumn(*table, "value", index)) {
      Fatal("G4MuonicData003", "dataset at " + dir + " has no 'value' column in #TABLE " + name);
      return false;
    }
    for (const UnitRule& rule : kUnitRules) {
      if (std::string(rule.table) == name && !CheckUnit(dir, *table, rule.column, rule.unit)) return false;
    }
    selected.count = 1;
  }
  selected.table = table;
  selected.value = index;
  selected.convention = Assigned(*table, "VALIDITY", "A", ':');
  return true;
}

// A refused pair configuration: fatal, and, should an installed handler suppress the abort, recorded so that
// every entry point throws -- requested evaluated pairs never silently become the compiled-in pair.
void RefusePairs(State& state, const char* code, const std::string& text) {
  Fatal(code, text);
  state.refusal = text;
}

// ---- binding the selected pair table to its provenance ------------------------------------------------------
// The selected evaluated profile's rows are a scientific roster, not merely well-formed records: before any
// consumer can use the table, the live parsed record keys must be exactly the row keys of the table's Layer-2
// provenance file, whose SHA-256 must be the table's `#SOURCEDIGEST`. A substituted, moved or removed row, a
// missing or altered provenance file, or an altered digest is refused.

// SHA-256 (FIPS 180-4) of `data`, lowercase hexadecimal.
std::string Sha256Hex(const std::string& data) {
  static const std::uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::string m = data;
  const std::uint64_t bits = static_cast<std::uint64_t>(data.size()) * 8u;
  m.push_back(static_cast<char>(0x80));
  while (m.size() % 64 != 56) m.push_back('\0');
  for (int i = 7; i >= 0; --i) m.push_back(static_cast<char>((bits >> (8 * i)) & 0xffu));
  auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (std::size_t chunk = 0; chunk < m.size(); chunk += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = 0;
      for (int j = 0; j < 4; ++j) w[i] = (w[i] << 8) | static_cast<unsigned char>(m[chunk + 4 * i + j]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + mj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  static const char* const hex = "0123456789abcdef";
  std::string out;
  for (std::uint32_t v : h) {
    for (int i = 7; i >= 0; --i) out.push_back(hex[(v >> (4 * i)) & 0xfu]);
  }
  return out;
}

// A minimal reader of the canonical Layer-2 JSON: the keys of the top-level "rows" object, or false when the text
// is not a JSON object carrying one.
struct JsonCursor {
  const std::string& s;
  std::size_t i = 0;
  void Space() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
  }
  bool String(std::string* out) {
    Space();
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    std::string value;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        if (i + 1 >= s.size()) return false;
        value.push_back(s[i + 1]);
        i += 2;
        continue;
      }
      value.push_back(s[i++]);
    }
    if (i >= s.size()) return false;
    ++i;
    if (out != nullptr) *out = value;
    return true;
  }
  bool Value(int depth, std::vector<std::string>* keys) {
    Space();
    if (depth > 64 || i >= s.size()) return false;
    const char c = s[i];
    if (c == '"') return String(nullptr);
    if (c == '{' || c == '[') {
      const char close = c == '{' ? '}' : ']';
      ++i;
      Space();
      if (i < s.size() && s[i] == close) {
        ++i;
        return true;
      }
      while (true) {
        if (c == '{') {
          std::string key;
          if (!String(&key)) return false;
          Space();
          if (i >= s.size() || s[i] != ':') return false;
          ++i;
          if (keys != nullptr) keys->push_back(key);
          if (!Value(depth + 1, nullptr)) return false;
        } else if (!Value(depth + 1, nullptr)) {
          return false;
        }
        Space();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == close) {
          ++i;
          return true;
        }
        return false;
      }
    }
    const std::size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != ' ' && s[i] != '\n' &&
           s[i] != '\r' && s[i] != '\t') {
      ++i;
    }
    return i > start;
  }
};

bool ProvenanceRowKeys(const std::string& text, std::vector<std::string>& keys) {
  JsonCursor cursor{text};
  cursor.Space();
  if (cursor.i >= text.size() || text[cursor.i] != '{') return false;
  ++cursor.i;
  bool found = false;
  while (true) {
    std::string key;
    if (!cursor.String(&key)) return false;
    cursor.Space();
    if (cursor.i >= text.size() || text[cursor.i] != ':') return false;
    ++cursor.i;
    if (key == "rows") {
      cursor.Space();
      if (cursor.i >= text.size() || text[cursor.i] != '{' || found) return false;
      std::vector<std::string> rows;
      // the keys of the rows object, read as the keys of its first level
      JsonCursor inner{text, cursor.i};
      if (!inner.Value(1, &rows)) return false;
      cursor.i = inner.i;
      keys = rows;
      found = true;
    } else if (!cursor.Value(1, nullptr)) {
      return false;
    }
    cursor.Space();
    if (cursor.i < text.size() && text[cursor.i] == ',') {
      ++cursor.i;
      continue;
    }
    if (cursor.i < text.size() && text[cursor.i] == '}') {
      ++cursor.i;
      cursor.Space();
      return found && cursor.i == text.size();
    }
    return false;
  }
}

// Why the selected pair table is not bound to its provenance, or "" when it is.
std::string PairProvenanceProblem(const Table& table) {
  const std::string suffix = ".g4dat";
  if (table.file.size() <= suffix.size() || table.file.compare(table.file.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return "the table file is not a .g4dat file";
  }
  const std::string sibling = table.file.substr(0, table.file.size() - suffix.size()) + ".prov.json";
  std::ifstream in(sibling, std::ios::binary);
  if (!in) return "its provenance file " + sibling + " cannot be read";
  std::ostringstream bytes;
  bytes << in.rdbuf();
  const std::string text = bytes.str();
  const std::string* declared = table.Directive("SOURCEDIGEST");
  if (declared == nullptr || *declared != Sha256Hex(text)) {
    return "the SHA-256 of " + sibling + " is not the table's #SOURCEDIGEST";
  }
  std::vector<std::string> rows;
  if (!ProvenanceRowKeys(text, rows)) return "the provenance file " + sibling + " carries no readable rows object";
  std::vector<std::string> records;
  for (const Table::Record& record : table.records) {
    if (record.keys.size() != 2) return "a record does not carry the two keys Z and A";
    records.push_back(std::to_string(record.keys[0]) + "-" + std::to_string(record.keys[1]));
  }
  std::sort(rows.begin(), rows.end());
  std::sort(records.begin(), records.end());
  if (rows != records || std::adjacent_find(rows.begin(), rows.end()) != rows.end()) {
    return "its records are not the rows its provenance file admits";
  }
  return "";
}

// Select the bound-rate pair table of `profile`: it must exist, and its three value columns must be the
// contiguous `tau q free_tau` with units `ns`, `1`, `ns`. False after a refusal.
bool SelectPairs(State& state, const std::string& dir, const std::string& profile, const std::string& present) {
  const Table* table = state.dataset.Find(profile, kPairTable);
  if (table == nullptr) {
    RefusePairs(state, "G4MuonicData004",
                "G4MUONICDATA_D1_PAIR_PROFILE names '" + profile + "' but the dataset at " + dir +
                    " carries no #TABLE " + kPairTable + " under that '#PROFILE'; the profiles present are: " + present);
    return false;
  }
  Index first = 0;
  for (G4int i = 0; i < 3; ++i) {
    if (!CheckUnit(dir, *table, kPairColumns[i], kPairUnits[i])) {
      state.refusal = "the bound-rate pair table at " + dir + " does not carry tau q free_tau in ns, 1, ns";
      return false;
    }
    Index index = 0;
    FindColumn(*table, kPairColumns[i], index);
    if (i == 0) {
      first = index;
    } else if (index != first + static_cast<Index>(i)) {
      RefusePairs(state, "G4MuonicData003", "dataset at " + dir + ": file " + table->file + ", #TABLE " + kPairTable +
                                                " does not declare tau q free_tau as consecutive columns");
      return false;
    }
  }
  const std::string unbound = PairProvenanceProblem(*table);
  if (!unbound.empty()) {
    RefusePairs(state, "G4MuonicData005",
                "dataset at " + dir + ": the bound-rate pair table " + table->file + " of the selected profile '" +
                    profile + "' is refused: " + unbound);
    return false;
  }
  state.pairs.table = table;
  state.pairs.value = first;
  state.pairs.count = 3;
  state.pairs.convention = Assigned(*table, "VALIDITY", "A", ':');
  return true;
}

// Read the environment, discover and read the dataset, apply the semantic layer, and select the
// tables of each seam. Any refusal raises a FatalException; should an installed handler suppress
// the abort, the state is left with no tables, and a semantic refusal is recorded so that every
// entry point throws rather than answering from a dataset no consumer may read.
void BuildOrRefuse(State& state) {
  const std::string legacy = Variable("G4MUONICDATA_PROFILE");
  const Choice d1 = Choose("G4MUONICDATA_D1_PROFILE", legacy, G4MuonicDataTable::kParityProfile);
  const Choice d3 = Choose("G4MUONICDATA_D3_PROFILE", legacy, kCompiledProfile);
  // The pair profile has its own variable and no legacy or default source: unset is `compiled`.
  const std::string pairNamed = Variable("G4MUONICDATA_D1_PAIR_PROFILE");
  const std::string pairProfile = pairNamed.empty() ? std::string(kCompiledProfile) : pairNamed;
  state.config.d1Profile = d1.profile;
  state.config.d3Profile = d3.profile;
  state.config.d1PairProfile = pairProfile;
  const bool d1Compiled = d1.profile == kCompiledProfile;
  const bool d3Compiled = d3.profile == kCompiledProfile;
  const bool pairCompiled = pairProfile == kCompiledProfile;
  if (d1Compiled && d3Compiled && pairCompiled) return;  // no discovery and no I/O

  const char* found = G4FindDataDir("G4MUONICDATA");
  // With a pair profile selected, a dataset that cannot be found or read is recorded as refused too, so that a
  // suppressed abort never leaves the requested evaluated pairs replaced by the compiled-in code.
  if (found == nullptr) {
    const std::string text =
        "G4MuonicDataTable::Enable() was called but no dataset directory was found: G4MUONICDATA "
        "is unset and GEANT4_DATA_DIR resolved none. Export G4MUONICDATA=<directory holding "
        "the *.g4dat files>.";
    Fatal("G4MuonicData001", text);
    if (!pairCompiled) state.refusal = text;
    return;
  }
  const std::string dir(found);
  try {
    state.dataset = G4MuonicDataTable::Load(dir);
  } catch (const G4MuonicDataTable::Error& e) {
    const std::string text = "dataset at " + dir + " rejected: " + e.what();
    Fatal("G4MuonicData002", text);
    if (!pairCompiled) state.refusal = text;
    return;
  }
  state.loaded = true;
  state.config.datasetDirectory = dir;
  if (!state.dataset.Tables().empty()) {
    const std::string* version = state.dataset.Tables().front().Directive("VERSION");
    if (version != nullptr) state.config.datasetVersion = *version;
  }

  // Every loaded table, including the profiles this run does not read through, before any of them
  // is selected: a dataset that does not mean what a consumer may read is refused whole.
  const std::vector<G4MuonicDataSemantics::Issue> issues = G4MuonicDataSemantics::Validate(state.dataset);
  if (!issues.empty()) {
    const G4MuonicDataSemantics::Issue& first = issues.front();
    const std::string text = "dataset at " + dir + " means what no consumer may read: " + first.code + " in " +
                             (first.file.empty() ? "#PROFILE " + first.profile : "file " + first.file) +
                             (first.table.empty() ? "" : ", #TABLE " + first.table) + ": " + first.text;
    Fatal("G4MuonicData005", text);
    state.refusal = text;
    return;
  }

  const std::vector<std::string> profiles = state.dataset.Profiles();
  std::string present;
  for (const std::string& name : profiles) present += (present.empty() ? "" : ", ") + name;
  // A profile named by the one variable that spans both seams must exist; a seam it carries no
  // table for resolves to the compiled-in code for that seam.
  if (legacy != kCompiledProfile && (d1.source == Source::Legacy || d3.source == Source::Legacy) &&
      std::find(profiles.begin(), profiles.end(), legacy) == profiles.end()) {
    Fatal("G4MuonicData004", "G4MUONICDATA_PROFILE names '" + legacy + "' but no file in the dataset at " +
                                 dir + " declares that '#PROFILE'; the profiles present are: " + present);
    return;
  }
  // A bound-rate pair profile is selected by its own variable only. Named by the variable that spans both
  // seams, it would resolve to a capture seam with no table and fall back to the compiled-in capture code with
  // no pair at all; that configuration is refused rather than silently compiled.
  if (!legacy.empty() && legacy != kCompiledProfile && state.dataset.Find(legacy, kPairTable) != nullptr) {
    RefusePairs(state, "G4MuonicData004",
                "G4MUONICDATA_PROFILE names '" + legacy + "', a bound-rate pair profile of the dataset at " + dir +
                    "; a pair profile is selected only by G4MUONICDATA_D1_PAIR_PROFILE");
    return;
  }
  if (!pairCompiled && !SelectPairs(state, dir, pairProfile, present)) return;

  if (!d1Compiled) {
    if (!Select(state, dir, d1.profile, kCaptureTable, state.capture)) return;
    if (!Select(state, dir, d1.profile, kZeffTable, state.zeff)) return;
    const bool parity = d1.profile == G4MuonicDataTable::kParityProfile;
    if (d1.source == Source::PerSeam && state.capture.table == nullptr) {
      Fatal("G4MuonicData004", "G4MUONICDATA_D1_PROFILE names '" + d1.profile + "' but the dataset at " + dir +
                                   " carries no #TABLE " + kCaptureTable + " under that '#PROFILE'; the profiles present are: " +
                                   present);
      return;
    }
    if (parity) {
      for (const Selected* selected : {&state.capture, &state.zeff}) {
        if (selected->table == nullptr) {
          const char* name = selected == &state.capture ? kCaptureTable : kZeffTable;
          Fatal("G4MuonicData003", "dataset at " + dir + " lacks #TABLE " + name + " under #PROFILE " +
                                       G4MuonicDataTable::kParityProfile);
          return;
        }
      }
    }
    state.d1Evaluated = !parity;
  }
  if (!d3Compiled) {
    if (!Select(state, dir, d3.profile, kKShellTable, state.kshell)) return;
    if (!Select(state, dir, d3.profile, kLevelTable, state.levels)) return;
    if (d3.source == Source::PerSeam && (state.kshell.table == nullptr || state.levels.table == nullptr)) {
      Fatal("G4MuonicData004", "G4MUONICDATA_D3_PROFILE names '" + d3.profile + "' but the dataset at " + dir +
                                   " carries no #TABLE " +
                                   (state.kshell.table == nullptr ? kKShellTable : kLevelTable) +
                                   " under that '#PROFILE'; the profiles present are: " + present);
      return;
    }
  }
}

// A seam that ended with no table of its own -- because its profile carries none, or because the
// configuration was refused before it could be selected -- is the compiled-in code for that seam. The pair
// profile is its own selection: resetting the capture seam to compiled leaves it, and its table, alone; a pair
// profile that ended with no table is `compiled` too.
void Build(State& state) {
  BuildOrRefuse(state);
  if (state.capture.table == nullptr && state.zeff.table == nullptr) {
    state.config.d1Profile = kCompiledProfile;
    state.d1Evaluated = false;
  }
  if (state.kshell.table == nullptr && state.levels.table == nullptr) state.config.d3Profile = kCompiledProfile;
  if (state.pairs.table == nullptr) state.config.d1PairProfile = kCompiledProfile;
}

const State& Frozen() {
  static const State* const state = [] {
    State* built = new State;
    Build(*built);
    return built;
  }();
  return *state;
}

// The frozen state, or nullptr with the opt-in off. Throws when the semantic layer refused the
// dataset and the abort was suppressed.
const State* Access() {
  if (!G4MuonicDataTable::IsEnabled()) return nullptr;
  const State& state = Frozen();
  if (!state.refusal.empty()) throw std::runtime_error(state.refusal);
  return &state;
}

Resolution Resolve(const Selected& selected, const std::string& profile, G4int Z, G4int A) {
  Resolution resolution;
  resolution.requestedZ = Z;
  resolution.requestedA = A;
  resolution.profile = profile;
  if (selected.table == nullptr) return resolution;
  // The legacy natural rung is used exactly when the resolved profile for this seam is `parity`.
  const bool legacy = profile == G4MuonicDataTable::kParityProfile;
  const Table::Record* record = selected.table->Lookup({Z, A});
  Origin origin = Origin::Exact;
  G4int resolvedA = A;
  if (record == nullptr) {
    if (!legacy || A == 0) return resolution;
    record = selected.table->Lookup({Z, 0});
    if (record == nullptr) return resolution;
    origin = Origin::LegacyNaturalFallback;
    resolvedA = 0;
  } else if (A == 0 && !legacy) {
    origin = selected.convention == kRepresentative ? Origin::ExplicitRepresentative : Origin::ExplicitNatural;
  }
  resolution.values = &record->floats[selected.value];
  resolution.count = selected.count;
  resolution.resolvedZ = Z;
  resolution.resolvedA = resolvedA;
  resolution.origin = origin;
  return resolution;
}

Resolution ResolveSingle(const Selected& selected, const std::string& profile, G4int Z) {
  Resolution resolution;
  resolution.requestedZ = Z;
  resolution.profile = profile;
  if (selected.table == nullptr) return resolution;
  const Table::Record* record = selected.table->Lookup({Z});
  if (record == nullptr) return resolution;
  resolution.values = &record->floats[selected.value];
  resolution.count = selected.count;
  resolution.resolvedZ = Z;
  resolution.origin = Origin::Exact;
  return resolution;
}

Resolution Off(G4int Z, G4int A) {
  Resolution resolution;
  resolution.requestedZ = Z;
  resolution.requestedA = A;
  resolution.profile = kCompiledProfile;
  return resolution;
}

}  // namespace

void G4MuonicDataOverlay::Initialize() {
  if (!G4MuonicDataTable::IsEnabled()) return;
  const State& state = Frozen();
  if (!state.refusal.empty()) throw std::runtime_error(state.refusal);
}

const G4MuonicDataOverlay::Configuration& G4MuonicDataOverlay::Config() {
  static const Configuration off = [] {
    Configuration configuration;
    configuration.d1Profile = kCompiledProfile;
    configuration.d3Profile = kCompiledProfile;
    configuration.d1PairProfile = kCompiledProfile;
    return configuration;
  }();
  if (!G4MuonicDataTable::IsEnabled()) return off;
  return Frozen().config;
}

const G4MuonicDataTable* G4MuonicDataOverlay::Table() {
  const State* state = Access();
  return state == nullptr || !state->loaded ? nullptr : &state->dataset;
}

G4MuonicDataOverlay::Resolution G4MuonicDataOverlay::ResolveRate(G4int Z, G4int A) {
  const State* state = Access();
  return state == nullptr ? Off(Z, A) : Resolve(state->capture, state->config.d1Profile, Z, A);
}

G4MuonicDataOverlay::Resolution G4MuonicDataOverlay::ResolvePair(G4int Z, G4int A) {
  const State* state = Access();
  if (state == nullptr) {
    Resolution off = Off(Z, A);
    return off;
  }
  // Isotope keys only: a natural-composition request has no pair, and no other rung is consulted.
  if (A <= 0) {
    Resolution none = Off(Z, A);
    none.profile = state->config.d1PairProfile;
    return none;
  }
  return Resolve(state->pairs, state->config.d1PairProfile, Z, A);
}

G4MuonicDataOverlay::Resolution G4MuonicDataOverlay::ResolveZeff(G4int Z) {
  const State* state = Access();
  return state == nullptr ? Off(Z, 0) : ResolveSingle(state->zeff, state->config.d1Profile, Z);
}

G4MuonicDataOverlay::Resolution G4MuonicDataOverlay::ResolveKShell(G4int Z, G4int A) {
  const State* state = Access();
  return state == nullptr ? Off(Z, A) : Resolve(state->kshell, state->config.d3Profile, Z, A);
}

G4MuonicDataOverlay::Resolution G4MuonicDataOverlay::ResolveLevels(G4int Z, G4int A) {
  const State* state = Access();
  return state == nullptr ? Off(Z, A) : Resolve(state->levels, state->config.d3Profile, Z, A);
}

const G4double* G4MuonicDataOverlay::Rate(G4int Z, G4int A) { return ResolveRate(Z, A).values; }

const G4double* G4MuonicDataOverlay::Zeff(G4int Z) { return ResolveZeff(Z).values; }

const G4double* G4MuonicDataOverlay::KShell(G4int Z, G4int A) { return ResolveKShell(Z, A).values; }

const G4double* G4MuonicDataOverlay::Levels(G4int Z, G4int A, G4int& count) {
  const Resolution resolution = ResolveLevels(Z, A);
  count = resolution.count;
  return resolution.values;
}

void G4MuonicDataOverlay::CheckComputedCaptureRate(G4int Z, G4int A, G4double rate) {
  const State* state = Access();
  if (state == nullptr || !state->d1Evaluated) return;
  if (!std::isfinite(rate)) {
    Fatal("G4MuonicData005", "under #PROFILE " + state->config.d1Profile + " the dataset has no nuclear capture rate for (Z, A) = (" +
                                 std::to_string(Z) + ", " + std::to_string(A) +
                                 ") and the compiled-in model computed one that is not a finite rate");
    return;
  }
  if (rate <= 0.) {
    // Reported once per process, from whichever thread meets it first.
    static std::atomic<bool> reported(false);
    if (!reported.exchange(true)) {
      const std::string text = "under #PROFILE " + state->config.d1Profile + " the dataset has no nuclear capture rate for (Z, A) = (" +
                               std::to_string(Z) + ", " + std::to_string(A) +
                               ") and the compiled-in model computed one that is not positive; it is used as computed, as under parity, "
                               "and no later such rate is reported";
      G4Exception("G4MuonicDataOverlay", "G4MuonicData006", JustWarning, text.c_str());
    }
  }
}

void G4MuonicDataOverlay::CheckCascadeLevels(G4int Z, G4int A, const G4double* levels, G4int count) {
  const State* state = Access();
  if (state == nullptr || levels == nullptr) return;
  const std::string where = " for (Z, A) = (" + std::to_string(Z) + ", " + std::to_string(A) + ")";
  if (count != kCascadeLevels) {
    Fatal("G4MuonicData005", "the muonic cascade assembled " + std::to_string(count) + " level(s)" + where +
                                 ", not " + std::to_string(kCascadeLevels));
    return;
  }
  for (G4int i = 0; i < count; ++i) {
    if (!std::isfinite(levels[i]) || levels[i] <= 0.) {
      Fatal("G4MuonicData005", "the muonic cascade assembled a level that is not a positive finite energy" + where);
      return;
    }
    if (i > 0 && !(levels[i - 1] > levels[i])) {
      Fatal("G4MuonicData005", "the muonic cascade assembled levels that do not fall" + where);
      return;
    }
  }
}
