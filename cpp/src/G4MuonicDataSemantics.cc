// G4MuonicDataSemantics -- implementation. See the header for what each code stands for.
//
// No value is ever rendered into a message: the reader parses numbers without consulting the
// process locale, and `std::to_string(double)` would not. A message names keys, columns and
// directives, which are integers and text.
#include "G4MuonicDataSemantics.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using Table = G4MuonicDataTable::Table;
using Issue = G4MuonicDataSemantics::Issue;

const char* const kDatasetName = "G4MuonicData";
const char* const kCaptureTable = "nuclear_capture_rate";
const char* const kZeffTable = "muon_zeff";
const char* const kKShellTable = "k_shell_energy";
const char* const kLevelTable = "level_energy";
const char* const kD1Seam = "d1_nuclear_capture";
const char* const kD3Seam = "d3_transitions";
const char* const kRateUnit = "1e6/s";
const char* const kZeffUnit = "dimensionless";
const char* const kEnergyUnit = "keV";
// The `#VALIDITY` `A:` conventions this layer knows (FORMAT_SPEC.md section 6).
const char* const kListed = "listed";
const char* const kNatural = "natural_and_listed";
const char* const kRepresentative = "most_abundant_and_listed";
// The key domain: every record is about a nuclide, or about an element under the `A = 0` rung.
const long kMinZ = 1;
const long kMaxZ = 120;
const long kMaxA = 300;
const long kMaxZeffZ = 100;
// `level_energy` declares `e2 .. eN` and `u2 .. uN` with N in this inclusive range.
const std::size_t kMinLevelIndex = 2;
const std::size_t kMaxLevelIndex = 14;

void Add(std::vector<Issue>& issues, const char* code, const Table& table, const std::string& text) {
  Issue issue;
  issue.code = code;
  issue.file = table.file;
  issue.profile = table.profile;
  issue.table = table.name;
  issue.text = text;
  issues.push_back(issue);
}

void AddProfile(std::vector<Issue>& issues, const char* code, const std::string& profile, const std::string& text) {
  Issue issue;
  issue.code = code;
  issue.profile = profile;
  issue.text = text;
  issues.push_back(issue);
}

std::string Value(const Table& table, const char* keyword) {
  const std::string* value = table.Directive(keyword);
  return value == nullptr ? std::string() : *value;
}

std::string Quote(const std::string& text) { return "'" + text + "'"; }

std::string Join(const std::vector<std::string>& parts) {
  std::string out;
  for (const std::string& part : parts) out += (out.empty() ? "" : " ") + part;
  return out;
}

std::string Key(const std::vector<long>& keys) {
  std::string out;
  for (long part : keys) out += (out.empty() ? "" : "-") + std::to_string(part);
  return out;
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

// The `name<delimiter>value` assignments of a directive value. Returns "" and fills `out`, or the
// reason the value is not a set of assignments.
std::string Assignments(const std::string& text, char delimiter, std::map<std::string, std::string>& out) {
  out.clear();
  for (const std::string& token : Tokens(text)) {
    const std::string::size_type at = token.find(delimiter);
    if (at == std::string::npos || at == 0 || at + 1 == token.size()) {
      return "assignment " + Quote(token) + " is not NAME" + std::string(1, delimiter) + "VALUE";
    }
    const std::string name = token.substr(0, at);
    if (out.find(name) != out.end()) return "assigns " + Quote(name) + " twice";
    out[name] = token.substr(at + 1);
  }
  return "";
}

bool Integer(const std::string& text, long& out) {
  if (text.empty() || text.size() > 9) return false;
  long value = 0;
  for (char c : text) {
    if (c < '0' || c > '9') return false;
    value = value * 10 + (c - '0');
  }
  out = value;
  return true;
}

// `MIN-MAX`, inclusive, non-negative, MIN <= MAX.
bool Range(const std::string& text, long& low, long& high) {
  const std::string::size_type at = text.find('-');
  if (at == std::string::npos) return false;
  return Integer(text.substr(0, at), low) && Integer(text.substr(at + 1), high) && low <= high;
}

// The `#COLUMNS` sequence `level_energy` declares for N, and the N a given sequence implies.
std::vector<std::string> LevelColumns(std::size_t n) {
  std::vector<std::string> columns;
  columns.push_back("Z");
  columns.push_back("A");
  for (std::size_t i = kMinLevelIndex; i <= n; ++i) columns.push_back("e" + std::to_string(i));
  for (std::size_t i = kMinLevelIndex; i <= n; ++i) columns.push_back("u" + std::to_string(i));
  return columns;
}

bool LevelShape(const Table& table, std::size_t& n) {
  if (table.columns.size() < 4 || table.columns.size() % 2 != 0) return false;
  n = (table.columns.size() - 2) / 2 + 1;
  if (n < kMinLevelIndex || n > kMaxLevelIndex) return false;
  return table.columns == LevelColumns(n);
}

// The `#COLUMNS` sequence of the fixed-shape tables, by `#TABLE` name; empty for `level_energy`,
// whose sequence depends on the N it declares, and for a name no consumer reads.
std::vector<std::string> FixedColumns(const std::string& name) {
  std::vector<std::string> columns;
  if (name == kCaptureTable || name == kKShellTable) {
    columns.push_back("Z");
    columns.push_back("A");
    columns.push_back("value");
    columns.push_back("unc");
  } else if (name == kZeffTable) {
    columns.push_back("Z");
    columns.push_back("value");
  }
  return columns;
}

const char* SeamOf(const std::string& name) {
  if (name == kCaptureTable || name == kZeffTable) return kD1Seam;
  if (name == kKShellTable || name == kLevelTable) return kD3Seam;
  return "";
}

const char* UnitOf(const std::string& name) {
  if (name == kCaptureTable) return kRateUnit;
  if (name == kZeffTable) return kZeffUnit;
  return kEnergyUnit;
}

// S001, S002, S003, S004 and S005 over one table. `version` is the `#VERSION` the first loaded
// file declares. Every rule reports its first offending row only, so one broken column does not
// bury the rest.
void CheckTable(const Table& table, const std::string& version, std::vector<Issue>& issues) {
  // ---- S001: this dataset, one version.
  const std::string dataset_name = Value(table, "DATASET");
  if (dataset_name != kDatasetName) {
    Add(issues, "S001", table, "#DATASET is " + Quote(dataset_name) + ", not " + Quote(kDatasetName));
  }
  const std::string declared = Value(table, "VERSION");
  if (declared.empty()) {
    Add(issues, "S001", table, "#VERSION is empty");
  } else if (declared != version) {
    Add(issues, "S001", table,
        "#VERSION is " + Quote(declared) + "; the first loaded file declares " + Quote(version));
  }

  // ---- S002: a table name a consumer reads, in its seam, with the columns that name implies.
  const char* const seam = SeamOf(table.name);
  if (*seam == '\0') {
    Add(issues, "S002", table, "'#TABLE " + table.name + "' is not a table this dataset defines");
    return;
  }
  std::vector<std::string> expected = FixedColumns(table.name);
  if (table.name == kLevelTable) {
    std::size_t n = 0;
    if (!LevelShape(table, n)) {
      Add(issues, "S002", table,
          "#COLUMNS " + Quote(Join(table.columns)) + "; " + kLevelTable + " declares Z A e2 .. eN u2 .. uN with N from " +
              std::to_string(kMinLevelIndex) + " to " + std::to_string(kMaxLevelIndex));
      return;
    }
    expected = LevelColumns(n);
  }
  const std::string declared_seam = Value(table, "SEAM");
  if (declared_seam != seam) {
    Add(issues, "S002", table, "#SEAM is " + Quote(declared_seam) + "; " + table.name + " sits in seam " + Quote(seam));
  }
  if (table.columns != expected) {
    Add(issues, "S002", table, "#COLUMNS " + Quote(Join(table.columns)) + "; " + table.name + " declares " + Quote(Join(expected)));
    return;
  }
  if (table.records.empty()) {
    Add(issues, "S002", table, "declares no record");
  }

  const bool two_key = table.name != kZeffTable;
  // The non-key columns, in `#COLUMNS` order, are the record's floats.
  std::vector<std::string> values(expected.begin() + (two_key ? 2 : 1), expected.end());
  const char* const unit = UnitOf(table.name);

  // ---- S004: the unit of every value column, as the lookups read it.
  std::map<std::string, std::string> units;
  const std::string units_problem = Assignments(Value(table, "UNITS"), '=', units);
  if (!units_problem.empty()) {
    Add(issues, "S004", table, "#UNITS " + units_problem);
  } else {
    for (const std::string& column : values) {
      const std::map<std::string, std::string>::const_iterator at = units.find(column);
      const std::string assigned = at == units.end() ? std::string() : at->second;
      if (assigned != unit) {
        Add(issues, "S004", table,
            "#UNITS assigns " + Quote(column) + " the unit " + Quote(assigned) + "; the lookup reads it as " + Quote(unit));
        break;
      }
    }
  }

  // ---- S003: the key domain, and a `#VALIDITY` convention this layer knows.
  std::map<std::string, std::string> validity;
  const std::string validity_problem = Assignments(Value(table, "VALIDITY"), ':', validity);
  std::string convention;
  bool ranged = false;
  long low = 0, high = 0;
  if (!validity_problem.empty()) {
    Add(issues, "S003", table, "#VALIDITY " + validity_problem);
  } else {
    for (std::map<std::string, std::string>::const_iterator it = validity.begin(); it != validity.end(); ++it) {
      if (it->first != "Z" && it->first != "A") {
        Add(issues, "S003", table, "#VALIDITY assigns " + Quote(it->first) + ", which this layer has no convention for");
      }
    }
    const std::map<std::string, std::string>::const_iterator z = validity.find("Z");
    if (z == validity.end()) {
      Add(issues, "S003", table, "#VALIDITY assigns no 'Z'");
    } else if (z->second != kListed) {
      if (!Range(z->second, low, high)) {
        Add(issues, "S003", table,
            "#VALIDITY assigns 'Z:" + z->second + "', which is neither 'listed' nor an inclusive MIN-MAX");
      } else {
        ranged = true;
      }
    }
    const std::map<std::string, std::string>::const_iterator a = validity.find("A");
    if (!two_key) {
      if (a != validity.end()) {
        Add(issues, "S003", table, std::string(kZeffTable) + " has no mass-number column, and #VALIDITY assigns 'A:" + a->second + "'");
      }
    } else if (a == validity.end()) {
      Add(issues, "S003", table, "#VALIDITY assigns no 'A'");
    } else if (a->second != kListed && a->second != kNatural && a->second != kRepresentative) {
      Add(issues, "S003", table,
          "#VALIDITY assigns 'A:" + a->second + "', which is none of 'listed', '" + kNatural + "' and '" + kRepresentative + "'");
    } else {
      convention = a->second;
    }
  }
  bool reported_z = false, reported_a = false, reported_range = false, reported_natural = false;
  for (const Table::Record& record : table.records) {
    if (record.keys.size() != (two_key ? 2u : 1u)) continue;
    const long z = record.keys[0];
    if (!reported_z && (two_key ? (z < kMinZ || z > kMaxZ) : (z < 0 || z > kMaxZeffZ))) {
      Add(issues, "S003", table, "row " + Key(record.keys) + " has a Z outside the key domain");
      reported_z = true;
    }
    if (!reported_range && ranged && (z < low || z > high)) {
      Add(issues, "S003", table, "row " + Key(record.keys) + " has a Z outside '#VALIDITY Z:" + std::to_string(low) + "-" + std::to_string(high) + "'");
      reported_range = true;
    }
    if (!two_key) continue;
    const long a = record.keys[1];
    if (!reported_a && a != 0 && (a < z || a > kMaxA)) {
      Add(issues, "S003", table, "row " + Key(record.keys) + " has a mass number below its Z or above " + std::to_string(kMaxA));
      reported_a = true;
    }
    if (!reported_natural && a == 0 && convention == kListed) {
      Add(issues, "S003", table, "row " + Key(record.keys) + " carries A = 0 under 'A:" + kListed + "'");
      reported_natural = true;
    }
  }

  // ---- S005: the value domain of each quantity.
  const std::size_t level_values = table.name == kLevelTable ? values.size() / 2 : 0;
  bool reported_finite = false, reported_positive = false, reported_uncertainty = false, reported_zeff = false;
  for (const Table::Record& record : table.records) {
    if (record.floats.size() != values.size()) continue;
    for (std::size_t i = 0; i < record.floats.size(); ++i) {
      const double v = record.floats[i];
      if (!reported_finite && !std::isfinite(v)) {
        Add(issues, "S005", table, "row " + Key(record.keys) + " column " + Quote(values[i]) + " is not finite");
        reported_finite = true;
      }
      const bool uncertainty = table.name == kLevelTable ? i >= level_values : values[i] == "unc";
      if (uncertainty) {
        if (!reported_uncertainty && !(v >= 0.)) {
          Add(issues, "S005", table, "row " + Key(record.keys) + " column " + Quote(values[i]) + " is a negative uncertainty");
          reported_uncertainty = true;
        }
        continue;
      }
      if (table.name == kZeffTable) {
        const long z = record.keys.empty() ? 0 : record.keys[0];
        const bool sentinel = z == 0 && table.profile == G4MuonicDataTable::kParityProfile;
        const bool ok = sentinel ? v == 0. : (v > 0. && v <= static_cast<double>(z));
        if (!reported_zeff && !ok) {
          Add(issues, "S005", table,
              sentinel ? "row " + Key(record.keys) + " is the Z = 0 sentinel and is not zero"
                       : "row " + Key(record.keys) + " has an effective charge outside (0, Z]");
          reported_zeff = true;
        }
        continue;
      }
      if (!reported_positive && !(v > 0.)) {
        Add(issues, "S005", table, "row " + Key(record.keys) + " column " + Quote(values[i]) + " is not positive");
        reported_positive = true;
      }
    }
  }
}

// S006 over one profile's two energy tables.
void CheckPair(const std::string& profile, const Table& kshell, const Table& levels, std::vector<Issue>& issues) {
  std::size_t n = 0;
  if (kshell.columns != FixedColumns(kKShellTable) || !LevelShape(levels, n)) return;  // S002 said so already
  std::map<std::vector<long>, const Table::Record*> by_key;
  std::set<std::vector<long>> kshell_keys, level_keys;
  for (const Table::Record& record : levels.records) {
    level_keys.insert(record.keys);
    by_key[record.keys] = &record;
  }
  for (const Table::Record& record : kshell.records) kshell_keys.insert(record.keys);
  if (kshell_keys != level_keys) {
    std::vector<std::vector<long>> differing;
    std::set_symmetric_difference(kshell_keys.begin(), kshell_keys.end(), level_keys.begin(), level_keys.end(),
                                  std::back_inserter(differing));
    AddProfile(issues, "S006", profile,
               std::string(kKShellTable) + " and " + kLevelTable + " differ first at key " + Key(differing.front()) +
                   ", carried by " + (kshell_keys.count(differing.front()) != 0 ? kKShellTable : kLevelTable) + " only");
    return;
  }
  std::map<std::string, std::string> kshell_validity, level_validity;
  if (Assignments(Value(kshell, "VALIDITY"), ':', kshell_validity).empty() &&
      Assignments(Value(levels, "VALIDITY"), ':', level_validity).empty()) {
    const std::map<std::string, std::string>::const_iterator k = kshell_validity.find("A");
    const std::map<std::string, std::string>::const_iterator l = level_validity.find("A");
    const std::string left = k == kshell_validity.end() ? std::string() : k->second;
    const std::string right = l == level_validity.end() ? std::string() : l->second;
    if (left != right) {
      AddProfile(issues, "S006", profile,
                 std::string(kKShellTable) + " assigns 'A:" + left + "' and " + kLevelTable + " assigns 'A:" + right + "'");
    }
  }
  for (const Table::Record& record : kshell.records) {
    const std::map<std::vector<long>, const Table::Record*>::const_iterator at = by_key.find(record.keys);
    if (at == by_key.end() || record.floats.empty() || at->second->floats.size() < n - 1) continue;
    double previous = record.floats[0];
    const std::string* below = nullptr;
    const std::string binding = "value";
    for (std::size_t i = 0; i + 1 < n; ++i) {
      const double e = at->second->floats[i];
      if (!(previous > e)) {
        below = &levels.columns[2 + i];
        break;
      }
      previous = e;
    }
    if (below == nullptr && !(previous > 0.)) below = &levels.columns[n];
    if (below != nullptr) {
      AddProfile(issues, "S006", profile,
                 "at key " + Key(record.keys) + " the energies do not fall from " + Quote(binding) + " through " + Quote(*below));
      return;
    }
  }
}

}  // namespace

std::vector<G4MuonicDataSemantics::Issue> G4MuonicDataSemantics::Validate(const G4MuonicDataTable& dataset) {
  std::vector<Issue> issues;
  std::string version;
  for (const Table& table : dataset.Tables()) {
    const std::string declared = Value(table, "VERSION");
    if (!declared.empty()) {
      version = declared;
      break;
    }
  }
  for (const Table& table : dataset.Tables()) CheckTable(table, version, issues);
  const std::vector<std::string> profiles = dataset.Profiles();
  for (const std::string& profile : profiles) {
    const Table* kshell = dataset.Find(profile, kKShellTable);
    const Table* levels = dataset.Find(profile, kLevelTable);
    if (kshell == nullptr && levels == nullptr) continue;
    if (kshell == nullptr || levels == nullptr) {
      AddProfile(issues, "S006", profile,
                 std::string("carries '#TABLE ") + (kshell != nullptr ? kKShellTable : kLevelTable) + "' but not '#TABLE " +
                     (kshell != nullptr ? kLevelTable : kKShellTable) + "'");
      continue;
    }
    CheckPair(profile, *kshell, *levels, issues);
  }
  return issues;
}
