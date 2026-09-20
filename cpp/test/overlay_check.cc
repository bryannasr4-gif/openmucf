// overlay_check -- the Geant4-facing glue (G4MuonicDataOverlay) run without Geant4, against the
// stand-in headers under g4stub/, and every answer it gives checked against a resolution made here.
//
//   g4muonicdata_overlay_check [--off] [--init] [--setenv NAME=VALUE]... <query>...
//   query:  rate Z A | zeff Z | kshell Z A | levels Z A
//           checkrate Z A positive|zero|negative|nan|inf
//           checklevels falling|flat|negative|short
//
// The stand-ins are defined in this file: G4FindDataDir reads the environment variable it is given,
// and G4Exception prints `G4Exception <origin> <code> <description>` and, for a FatalException,
// exits with status 3 unless OVERLAY_CHECK_SUPPRESS_FATAL is set -- the behaviour of an exception
// handler that suppresses the abort.
//
// The opt-in is set here, before anything else, because the glue answers nothing without it;
// `--off` leaves it clear, and then no line may report a lookup and no directory may be read.
// `--init` freezes the configuration before the queries run, and `--setenv` changes the
// environment after that point, so a frozen configuration can be told from a re-read one.
//
// One line per query through the glue:
//   CONFIG d1=<profile> d3=<profile> dir=<none|set> version=<value|none>
//   TABLE <none|loaded>
//   RATE Z A <%a|none> <profile> <origin>      ZEFF Z <%a|none> <profile> <origin>
//   KSHELL Z A <%a|none> <profile> <origin>    LEVELS Z A <count> <%a>...|none <profile> <origin>
//   CHECKRATE Z A <token> returned             CHECKLEVELS <token> returned
// Then, unless `--off`, the directory G4MUONICDATA named at start-up is loaded again here, each
// query is resolved independently -- the two seams chosen from the environment as it stood at
// start-up, the legacy natural rung used exactly under `parity`, the columns by name -- and one
// line says `MATCH <query>` or `MISMATCH <query>: <detail>`. The exit status is 1 when any line
// says MISMATCH.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "G4EnvironmentUtils.hh"
#include "G4Exception.hh"
#include "G4MuonicDataOverlay.hh"
#include "G4MuonicDataTable.hh"

const char* G4FindDataDir(const char* name) { return std::getenv(name); }

void G4Exception(const char* originOfException, const char* exceptionCode, G4ExceptionSeverity severity,
                 const char* description) {
  std::printf("G4Exception %s %s %s\n", originOfException, exceptionCode, description);
  std::fflush(stdout);
  if (severity == FatalException && std::getenv("OVERLAY_CHECK_SUPPRESS_FATAL") == nullptr) std::exit(3);
}

namespace {

using Origin = G4MuonicDataOverlay::Origin;

const char* const kCompiledProfile = "compiled";
const char* const kRepresentative = "most_abundant_and_listed";
const char* const kCaptureTable = "nuclear_capture_rate";
const char* const kZeffTable = "muon_zeff";
const char* const kKShellTable = "k_shell_energy";
const char* const kLevelTable = "level_energy";
const int kCascadeLevels = 14;

struct Query {
  std::string kind;
  long z = 0;
  long a = 0;
  std::string token;
  std::string text;
};

// An answer: nothing, or the values found, with the rung and profile that produced them.
struct Answer {
  bool found = false;
  std::vector<double> values;
  Origin origin = Origin::Compiled;
  std::string profile;
};

const char* Name(Origin origin) {
  switch (origin) {
    case Origin::Exact: return "exact";
    case Origin::ExplicitNatural: return "explicit_natural";
    case Origin::ExplicitRepresentative: return "explicit_representative";
    case Origin::LegacyNaturalFallback: return "legacy_natural";
    case Origin::Compiled: break;
  }
  return "compiled";
}

std::string Render(const Answer& answer, bool with_count) {
  std::string out;
  if (!answer.found) {
    out = "none";
  } else {
    if (with_count) out = std::to_string(answer.values.size());
    for (double v : answer.values) {
      char buffer[64];
      std::snprintf(buffer, sizeof buffer, "%a", v);
      out += (out.empty() ? "" : " ") + std::string(buffer);
    }
  }
  return out + " " + (answer.profile.empty() ? std::string(kCompiledProfile) : answer.profile) + " " + Name(answer.origin);
}

Answer FromResolution(const G4MuonicDataOverlay::Resolution& resolution) {
  Answer answer;
  answer.origin = resolution.origin;
  answer.profile = resolution.profile;
  if (resolution.values != nullptr) {
    answer.found = true;
    answer.values.assign(resolution.values, resolution.values + resolution.count);
  }
  return answer;
}

Answer ThroughGlue(const Query& q) {
  const G4int z = static_cast<G4int>(q.z);
  const G4int a = static_cast<G4int>(q.a);
  if (q.kind == "rate") return FromResolution(G4MuonicDataOverlay::ResolveRate(z, a));
  if (q.kind == "zeff") return FromResolution(G4MuonicDataOverlay::ResolveZeff(z));
  if (q.kind == "kshell") return FromResolution(G4MuonicDataOverlay::ResolveKShell(z, a));
  return FromResolution(G4MuonicDataOverlay::ResolveLevels(z, a));
}

// The environment as it stood before any --setenv, which is what the glue is held to.
struct Environment {
  std::string directory, legacy, d1, d3;
  bool directorySet = false;
};

std::string Read(const char* name, bool& set) {
  const char* value = std::getenv(name);
  set = value != nullptr;
  return value == nullptr ? std::string() : std::string(value);
}

// The profile each seam resolves to, made here from the environment: a per-seam variable, else the
// one that spans both, else `parity` for the capture seam and the compiled-in code for the other.
std::string Chosen(const std::string& seam, const std::string& legacy, const char* fallback) {
  if (!seam.empty()) return seam;
  if (!legacy.empty()) return legacy;
  return fallback;
}

// A seam whose profile carries neither of its two tables is the compiled-in code for that seam,
// which is what the glue reports for it.
std::string Effective(const G4MuonicDataTable* dataset, const std::string& profile, bool d1_seam) {
  if (dataset == nullptr || profile == kCompiledProfile) return kCompiledProfile;
  const char* first = d1_seam ? kCaptureTable : kKShellTable;
  const char* second = d1_seam ? kZeffTable : kLevelTable;
  if (dataset->Find(profile, first) == nullptr && dataset->Find(profile, second) == nullptr) {
    return kCompiledProfile;
  }
  return profile;
}

// The `NAME:VALUE` assignment of `#VALIDITY`, read the way the glue reads it.
std::string Convention(const G4MuonicDataTable::Table& table) {
  const std::string* value = table.Directive("VALIDITY");
  if (value == nullptr) return "";
  std::string token;
  std::string out;
  std::string text = *value;
  text += ' ';
  for (char c : text) {
    if (c == ' ' || c == '\t') {
      if (token.compare(0, 2, "A:") == 0 && token.size() > 2) out = token.substr(2);
      token.clear();
    } else {
      token += c;
    }
  }
  return out;
}

// The resolution made here, without the glue: the legacy natural rung is used exactly when the
// seam's profile is `parity`; under any other profile a request with A > 0 is answered exactly or
// not at all, and the (Z, 0) record answers only a request for A = 0.
Answer Independently(const G4MuonicDataTable* dataset, const std::string& d1, const std::string& d3, const Query& q) {
  Answer answer;
  const bool d1_seam = q.kind == "rate" || q.kind == "zeff";
  answer.profile = d1_seam ? d1 : d3;
  if (dataset == nullptr || answer.profile == kCompiledProfile) return answer;
  const char* name = q.kind == "rate" ? kCaptureTable
                     : q.kind == "zeff" ? kZeffTable
                     : q.kind == "kshell" ? kKShellTable
                                          : kLevelTable;
  const G4MuonicDataTable::Table* table = dataset->Find(answer.profile, name);
  if (table == nullptr) return answer;
  const bool two_key = q.kind != "zeff";
  const bool parity = answer.profile == G4MuonicDataTable::kParityProfile;
  const G4MuonicDataTable::Table::Record* record =
      two_key ? table->Lookup({q.z, q.a}) : table->Lookup({q.z});
  Origin origin = Origin::Exact;
  if (record == nullptr) {
    if (!two_key || !parity || q.a == 0) return answer;
    record = table->Lookup({q.z, 0});
    if (record == nullptr) return answer;
    origin = Origin::LegacyNaturalFallback;
  } else if (two_key && q.a == 0 && !parity) {
    origin = Convention(*table) == kRepresentative ? Origin::ExplicitRepresentative : Origin::ExplicitNatural;
  }
  // The non-key columns, in #COLUMNS order, are the record's floats.
  std::vector<std::string> columns;
  for (const std::string& column : table->columns) {
    if (column != "Z" && column != "A") columns.push_back(column);
  }
  answer.origin = origin;
  if (q.kind != "levels") {
    for (std::size_t i = 0; i < columns.size(); ++i) {
      if (columns[i] == "value") {
        answer.found = true;
        answer.values.push_back(record->floats[i]);
      }
    }
    if (!answer.found) answer.origin = Origin::Compiled;
    return answer;
  }
  std::size_t first = 0;
  while (first < columns.size() && columns[first] != "e2") ++first;
  if (first == columns.size()) {
    answer.origin = Origin::Compiled;
    return answer;
  }
  answer.found = true;
  for (std::size_t i = first, n = 2; i < columns.size() && columns[i] == "e" + std::to_string(n); ++i, ++n) {
    answer.values.push_back(record->floats[i]);
  }
  return answer;
}

bool Same(const Answer& left, const Answer& right) {
  if (left.found != right.found || left.values.size() != right.values.size()) return false;
  if (left.origin != right.origin || left.profile != right.profile) return false;
  for (std::size_t i = 0; i < left.values.size(); ++i) {
    if (std::memcmp(&left.values[i], &right.values[i], sizeof(double)) != 0) return false;
  }
  return true;
}

bool ParseLong(const char* text, long& out) {
  char* end = nullptr;
  out = std::strtol(text, &end, 10);
  return end != text && *end == '\0';
}

// A computed capture rate of the shape the token names, built here so no value is spelled out.
bool ComputedRate(const std::string& token, double& out) {
  if (token == "positive") out = 1.;
  else if (token == "zero") out = 0.;
  else if (token == "negative") out = -1.;
  else if (token == "nan") out = std::nan("");
  else if (token == "inf") out = HUGE_VAL;
  else return false;
  return true;
}

// A cascade level array of the shape the token names.
bool CascadeLevels(const std::string& token, std::vector<double>& out) {
  out.clear();
  const int n = token == "short" ? kCascadeLevels - 1 : kCascadeLevels;
  if (token != "falling" && token != "flat" && token != "negative" && token != "short") return false;
  for (int i = 0; i < n; ++i) {
    if (token == "flat") out.push_back(1.);
    else if (token == "negative") out.push_back(static_cast<double>(-1 - i));
    else out.push_back(static_cast<double>(n - i));
  }
  return true;
}

void PutEnvironment(const std::string& assignment) {
  const std::string::size_type at = assignment.find('=');
  if (at == std::string::npos) return;
  const std::string name = assignment.substr(0, at), value = assignment.substr(at + 1);
#ifdef _WIN32
  _putenv_s(name.c_str(), value.c_str());
#else
  setenv(name.c_str(), value.c_str(), 1);
#endif
}

void Usage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--off] [--init] [--setenv NAME=VALUE]... (rate Z A | zeff Z | kshell Z A | "
               "levels Z A | checkrate Z A TOKEN | checklevels TOKEN)...\n",
               program);
}

int Run(int argc, char** argv) {
  bool off = false, initialise = false;
  std::vector<std::string> later;
  int i = 1;
  for (; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--off") off = true;
    else if (option == "--init") initialise = true;
    else if (option == "--setenv" && i + 1 < argc) later.push_back(argv[++i]);
    else break;
  }

  std::vector<Query> queries;
  for (; i < argc;) {
    Query q;
    q.kind = argv[i];
    int operands = 0;
    if (q.kind == "zeff") operands = 1;
    else if (q.kind == "rate" || q.kind == "kshell" || q.kind == "levels") operands = 2;
    else if (q.kind == "checkrate") operands = 3;
    else if (q.kind == "checklevels") operands = 1;
    else {
      Usage(argv[0]);
      return 2;
    }
    if (i + operands >= argc) {
      Usage(argv[0]);
      return 2;
    }
    if (q.kind == "checklevels") {
      q.token = argv[i + 1];
      q.text = q.kind + " " + q.token;
    } else {
      if (!ParseLong(argv[i + 1], q.z) || (operands >= 2 && !ParseLong(argv[i + 2], q.a))) {
        Usage(argv[0]);
        return 2;
      }
      if (operands == 3) q.token = argv[i + 3];
      q.text = q.kind + " " + std::to_string(q.z) + (operands >= 2 ? " " + std::to_string(q.a) : "") +
               (q.token.empty() ? "" : " " + q.token);
    }
    queries.push_back(q);
    i += operands + 1;
  }
  if (queries.empty()) {
    Usage(argv[0]);
    return 2;
  }

  Environment environment;
  environment.directory = Read("G4MUONICDATA", environment.directorySet);
  bool ignored = false;
  environment.legacy = Read("G4MUONICDATA_PROFILE", ignored);
  environment.d1 = Read("G4MUONICDATA_D1_PROFILE", ignored);
  environment.d3 = Read("G4MUONICDATA_D3_PROFILE", ignored);

  if (!off) G4MuonicDataTable::Enable();
  if (initialise) G4MuonicDataOverlay::Initialize();
  for (const std::string& assignment : later) PutEnvironment(assignment);

  const G4MuonicDataOverlay::Configuration& configuration = G4MuonicDataOverlay::Config();
  std::printf("CONFIG d1=%s d3=%s dir=%s version=%s\n", configuration.d1Profile.c_str(),
              configuration.d3Profile.c_str(), configuration.datasetDirectory.empty() ? "none" : "set",
              configuration.datasetVersion.empty() ? "none" : configuration.datasetVersion.c_str());
  std::printf("TABLE %s\n", G4MuonicDataOverlay::Table() == nullptr ? "none" : "loaded");
  std::fflush(stdout);

  std::vector<Answer> glue;
  std::vector<std::size_t> lookups;
  for (std::size_t q = 0; q < queries.size(); ++q) {
    const Query& query = queries[q];
    if (query.kind == "checkrate") {
      double rate = 0.;
      if (!ComputedRate(query.token, rate)) {
        Usage(argv[0]);
        return 2;
      }
      G4MuonicDataOverlay::CheckComputedCaptureRate(static_cast<G4int>(query.z), static_cast<G4int>(query.a), rate);
      std::printf("CHECKRATE %ld %ld %s returned\n", query.z, query.a, query.token.c_str());
      std::fflush(stdout);
      continue;
    }
    if (query.kind == "checklevels") {
      std::vector<double> levels;
      if (!CascadeLevels(query.token, levels)) {
        Usage(argv[0]);
        return 2;
      }
      G4MuonicDataOverlay::CheckCascadeLevels(0, 0, levels.data(), static_cast<G4int>(levels.size()));
      std::printf("CHECKLEVELS %s returned\n", query.token.c_str());
      std::fflush(stdout);
      continue;
    }
    glue.push_back(ThroughGlue(query));
    lookups.push_back(q);
    const char* label = query.kind == "rate" ? "RATE"
                        : query.kind == "zeff" ? "ZEFF"
                        : query.kind == "kshell" ? "KSHELL"
                                                 : "LEVELS";
    std::printf("%s %s %s\n", label, query.text.substr(query.kind.size() + 1).c_str(),
                Render(glue.back(), query.kind == "levels").c_str());
    std::fflush(stdout);
  }

  // With the opt-in clear the glue may read nothing, so nothing is read here either: every answer
  // must already be the empty one.
  if (off) {
    bool wrong = false;
    for (std::size_t n = 0; n < glue.size(); ++n) {
      const Query& query = queries[lookups[n]];
      if (!glue[n].found && glue[n].origin == Origin::Compiled && glue[n].profile == kCompiledProfile) {
        std::printf("MATCH %s\n", query.text.c_str());
      } else {
        wrong = true;
        std::printf("MISMATCH %s: the opt-in is clear and the glue answered %s\n", query.text.c_str(),
                    Render(glue[n], true).c_str());
      }
    }
    return wrong ? 1 : 0;
  }

  G4MuonicDataTable loaded;
  const G4MuonicDataTable* dataset = nullptr;
  if (environment.directorySet) {
    try {
      loaded = G4MuonicDataTable::Load(environment.directory);
      dataset = &loaded;
    } catch (const G4MuonicDataTable::Error&) {
      dataset = nullptr;
    }
  }
  const std::string d1 = Effective(dataset, Chosen(environment.d1, environment.legacy, G4MuonicDataTable::kParityProfile), true);
  const std::string d3 = Effective(dataset, Chosen(environment.d3, environment.legacy, kCompiledProfile), false);
  bool mismatch = false;
  for (std::size_t n = 0; n < glue.size(); ++n) {
    const Query& query = queries[lookups[n]];
    const Answer expected = Independently(dataset, d1, d3, query);
    if (Same(glue[n], expected)) {
      std::printf("MATCH %s\n", query.text.c_str());
    } else {
      mismatch = true;
      std::printf("MISMATCH %s: glue %s, resolved here %s\n", query.text.c_str(), Render(glue[n], true).c_str(),
                  Render(expected, true).c_str());
    }
  }
  return mismatch ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::runtime_error& error) {
    std::printf("REFUSED %s\n", error.what());
    return 4;
  }
}
