// overlay_check -- the Geant4-facing glue (G4MuonicDataOverlay) run without Geant4, against the
// stand-in headers under g4stub/, and every answer it gives checked against a resolution made here.
//
//   g4muonicdata_overlay_check <query>...
//   query:  rate Z A | zeff Z | kshell Z A | levels Z A
//
// The stand-ins are defined in this file: G4FindDataDir reads the environment variable it is given,
// and G4Exception prints `G4Exception <origin> <code> <description>` and, for a FatalException,
// exits with status 3 unless OVERLAY_CHECK_SUPPRESS_FATAL is set -- the behaviour of an exception
// handler that suppresses the abort, after which the glue answers every lookup with nothing.
//
// First every query goes through the glue, one line each:
//   RATE Z A <%a|none>   ZEFF Z <%a|none>   KSHELL Z A <%a|none>   LEVELS Z A <count> <%a>...|none
// Then the directory G4MUONICDATA names is loaded again here with G4MuonicDataTable::Load, each
// query is resolved independently -- the profile G4MUONICDATA_PROFILE names (parity when unset or
// empty), the exact key and then (Z, 0) on a two-key table, the column by name -- and one line says
// `MATCH <query>` or `MISMATCH <query>: <detail>`. The exit status is 1 when any line says MISMATCH.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Query {
  std::string kind;
  long z = 0;
  long a = 0;
  std::string text;
};

// An answer: nothing, or the values found (one for rate, zeff and kshell; the level columns for levels).
struct Answer {
  bool found = false;
  std::vector<double> values;
};

std::string Render(const Answer& answer, bool with_count) {
  if (!answer.found) return "none";
  std::string out;
  if (with_count) out = std::to_string(answer.values.size());
  for (double v : answer.values) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%a", v);
    out += (out.empty() ? "" : " ") + std::string(buffer);
  }
  return out;
}

Answer FromPointer(const G4double* value) {
  Answer answer;
  if (value != nullptr) {
    answer.found = true;
    answer.values.push_back(*value);
  }
  return answer;
}

Answer ThroughGlue(const Query& q) {
  const G4int z = static_cast<G4int>(q.z);
  const G4int a = static_cast<G4int>(q.a);
  if (q.kind == "rate") return FromPointer(G4MuonicDataOverlay::Rate(z, a));
  if (q.kind == "zeff") return FromPointer(G4MuonicDataOverlay::Zeff(z));
  if (q.kind == "kshell") return FromPointer(G4MuonicDataOverlay::KShell(z, a));
  G4int count = 0;
  const G4double* levels = G4MuonicDataOverlay::Levels(z, a, count);
  Answer answer;
  if (levels != nullptr) {
    answer.found = true;
    answer.values.assign(levels, levels + count);
  }
  return answer;
}

// The resolution made here, without the glue.
Answer Independently(const G4MuonicDataTable* dataset, const std::string& profile, const Query& q) {
  Answer answer;
  if (dataset == nullptr) return answer;
  const char* name = q.kind == "rate" ? "nuclear_capture_rate"
                     : q.kind == "zeff" ? "muon_zeff"
                     : q.kind == "kshell" ? "k_shell_energy"
                                          : "level_energy";
  const G4MuonicDataTable::Table* table = dataset->Find(profile, name);
  if (table == nullptr) return answer;
  const bool two_key = q.kind != "zeff";
  const G4MuonicDataTable::Table::Record* record =
      two_key ? table->Lookup({q.z, q.a}) : table->Lookup({q.z});
  if (record == nullptr && two_key) record = table->Lookup({q.z, 0});
  if (record == nullptr) return answer;
  // The non-key columns, in #COLUMNS order, are the record's floats.
  std::vector<std::string> columns;
  for (const std::string& column : table->columns) {
    if (column != "Z" && column != "A") columns.push_back(column);
  }
  if (q.kind != "levels") {
    for (std::size_t i = 0; i < columns.size(); ++i) {
      if (columns[i] == "value") {
        answer.found = true;
        answer.values.push_back(record->floats[i]);
      }
    }
    return answer;
  }
  std::size_t first = 0;
  while (first < columns.size() && columns[first] != "e2") ++first;
  if (first == columns.size()) return answer;
  answer.found = true;
  for (std::size_t i = first, n = 2; i < columns.size() && columns[i] == "e" + std::to_string(n); ++i, ++n) {
    answer.values.push_back(record->floats[i]);
  }
  return answer;
}

bool Same(const Answer& left, const Answer& right) {
  if (left.found != right.found || left.values.size() != right.values.size()) return false;
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

}  // namespace

int main(int argc, char** argv) {
  std::vector<Query> queries;
  for (int i = 1; i < argc;) {
    Query q;
    q.kind = argv[i];
    const int keys = q.kind == "zeff" ? 1 : 2;
    if ((q.kind != "rate" && q.kind != "zeff" && q.kind != "kshell" && q.kind != "levels") || i + keys >= argc ||
        !ParseLong(argv[i + 1], q.z) || (keys == 2 && !ParseLong(argv[i + 2], q.a))) {
      std::fprintf(stderr, "usage: %s (rate Z A | zeff Z | kshell Z A | levels Z A)...\n", argv[0]);
      return 2;
    }
    q.text = q.kind + " " + std::to_string(q.z) + (keys == 2 ? " " + std::to_string(q.a) : "");
    queries.push_back(q);
    i += keys + 1;
  }
  if (queries.empty()) {
    std::fprintf(stderr, "usage: %s (rate Z A | zeff Z | kshell Z A | levels Z A)...\n", argv[0]);
    return 2;
  }

  std::vector<Answer> glue;
  for (const Query& q : queries) {
    glue.push_back(ThroughGlue(q));
    std::string label = q.kind == "rate" ? "RATE" : q.kind == "zeff" ? "ZEFF" : q.kind == "kshell" ? "KSHELL" : "LEVELS";
    std::printf("%s %s %s\n", label.c_str(), q.text.substr(q.kind.size() + 1).c_str(),
                Render(glue.back(), q.kind == "levels").c_str());
    std::fflush(stdout);
  }

  G4MuonicDataTable loaded;
  const G4MuonicDataTable* dataset = nullptr;
  const char* dir = std::getenv("G4MUONICDATA");
  if (dir != nullptr) {
    try {
      loaded = G4MuonicDataTable::Load(dir);
      dataset = &loaded;
    } catch (const G4MuonicDataTable::Error&) {
      dataset = nullptr;
    }
  }
  const char* requested = std::getenv("G4MUONICDATA_PROFILE");
  const std::string profile = (requested == nullptr || *requested == '\0') ? std::string(G4MuonicDataTable::kParityProfile)
                                                                           : std::string(requested);
  bool mismatch = false;
  for (std::size_t i = 0; i < queries.size(); ++i) {
    const Answer expected = Independently(dataset, profile, queries[i]);
    if (Same(glue[i], expected)) {
      std::printf("MATCH %s\n", queries[i].text.c_str());
    } else {
      mismatch = true;
      std::printf("MISMATCH %s: glue %s, resolved here %s\n", queries[i].text.c_str(),
                  Render(glue[i], true).c_str(), Render(expected, true).c_str());
    }
  }
  return mismatch ? 1 : 0;
}
