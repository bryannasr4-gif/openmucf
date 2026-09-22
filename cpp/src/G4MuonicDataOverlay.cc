// G4MuonicDataOverlay -- implementation. See the header for what this class is for.
#include "G4MuonicDataOverlay.hh"

#include "G4EnvironmentUtils.hh"
#include "G4Exception.hh"
#include "G4MuonicDataSemantics.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

// Read the environment, discover and read the dataset, apply the semantic layer, and select the
// tables of each seam. Any refusal raises a FatalException; should an installed handler suppress
// the abort, the state is left with no tables, and a semantic refusal is recorded so that every
// entry point throws rather than answering from a dataset no consumer may read.
void BuildOrRefuse(State& state) {
  const std::string legacy = Variable("G4MUONICDATA_PROFILE");
  const Choice d1 = Choose("G4MUONICDATA_D1_PROFILE", legacy, G4MuonicDataTable::kParityProfile);
  const Choice d3 = Choose("G4MUONICDATA_D3_PROFILE", legacy, kCompiledProfile);
  state.config.d1Profile = d1.profile;
  state.config.d3Profile = d3.profile;
  const bool d1Compiled = d1.profile == kCompiledProfile;
  const bool d3Compiled = d3.profile == kCompiledProfile;
  if (d1Compiled && d3Compiled) return;  // no discovery and no I/O

  const char* found = G4FindDataDir("G4MUONICDATA");
  if (found == nullptr) {
    Fatal("G4MuonicData001",
          "G4MuonicDataTable::Enable() was called but no dataset directory was found: G4MUONICDATA "
          "is unset and GEANT4_DATA_DIR resolved none. Export G4MUONICDATA=<directory holding "
          "the *.g4dat files>.");
    return;
  }
  const std::string dir(found);
  try {
    state.dataset = G4MuonicDataTable::Load(dir);
  } catch (const G4MuonicDataTable::Error& e) {
    Fatal("G4MuonicData002", "dataset at " + dir + " rejected: " + e.what());
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
  if ((d1.source == Source::Legacy || d3.source == Source::Legacy) &&
      std::find(profiles.begin(), profiles.end(), legacy) == profiles.end()) {
    Fatal("G4MuonicData004", "G4MUONICDATA_PROFILE names '" + legacy + "' but no file in the dataset at " +
                                 dir + " declares that '#PROFILE'; the profiles present are: " + present);
    return;
  }

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
// configuration was refused before it could be selected -- is the compiled-in code for that seam.
void Build(State& state) {
  BuildOrRefuse(state);
  if (state.capture.table == nullptr && state.zeff.table == nullptr) {
    state.config.d1Profile = kCompiledProfile;
    state.d1Evaluated = false;
  }
  if (state.kshell.table == nullptr && state.levels.table == nullptr) state.config.d3Profile = kCompiledProfile;
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
  if (!std::isfinite(rate) || rate <= 0.) {
    Fatal("G4MuonicData005", "under #PROFILE " + state->config.d1Profile + " the dataset has no nuclear capture rate for (Z, A) = (" +
                                 std::to_string(Z) + ", " + std::to_string(A) +
                                 ") and the compiled-in model computed one that is not a positive finite rate");
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
