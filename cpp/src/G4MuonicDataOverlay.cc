// G4MuonicDataOverlay -- implementation. See the header for what this class is for.
#include "G4MuonicDataOverlay.hh"

#include "G4EnvironmentUtils.hh"
#include "G4Exception.hh"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Index = std::vector<double>::size_type;
using Table = G4MuonicDataTable::Table;

struct LoadedDataset {
  G4MuonicDataTable dataset;
  const Table* capture = nullptr;
  const Table* zeff = nullptr;
  const Table* kshell = nullptr;
  const Table* levels = nullptr;
  // Positions inside Record::floats: #COLUMNS minus the integer key columns.
  Index capture_value = 0;
  Index zeff_value = 0;
  Index kshell_value = 0;
  Index levels_e2 = 0;
  // How many consecutive columns e2, e3, ... the level table declares.
  G4int levels_count = 0;
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

// The unit `#UNITS` assigns to `column`, or "" when it assigns none.
std::string UnitOf(const Table& table, const std::string& column) {
  const std::string* units = table.Directive("UNITS");
  if (units == nullptr) return "";
  const std::string prefix = column + "=";
  std::string::size_type start = 0;
  while (start < units->size()) {
    std::string::size_type end = units->find(' ', start);
    if (end == std::string::npos) end = units->size();
    const std::string token = units->substr(start, end - start);
    if (token.compare(0, prefix.size(), prefix) == 0) return token.substr(prefix.size());
    start = end + 1;
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
    {"nuclear_capture_rate", "value", "1e6/s"},
    {"muon_zeff", "value", "dimensionless"},
    {"k_shell_energy", "value", "keV"},
};
const char* const kLevelTable = "level_energy";
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
  const std::string declared = UnitOf(table, column);
  if (declared != unit) {
    Fatal("G4MuonicData003", "dataset at " + dir + ": file " + table.file + ", #TABLE " + table.name +
                                 ", column '" + column + "' has unit '" + declared + "' in #UNITS; the lookup reads it as '" +
                                 unit + "'");
    return false;
  }
  return true;
}

// Resolve, read and check the dataset once. Any failure raises a FatalException and, should the
// installed exception handler suppress the abort, returns nullptr so every lookup misses.
const LoadedDataset* Load() {
  // Every lookup goes through the one profile G4MUONICDATA_PROFILE names; unset or empty selects
  // the parity profile.
  const char* requested = std::getenv("G4MUONICDATA_PROFILE");
  const std::string profile = (requested == nullptr || *requested == '\0')
                                  ? std::string(G4MuonicDataTable::kParityProfile)
                                  : std::string(requested);
  const char* found = G4FindDataDir("G4MUONICDATA");
  if (found == nullptr) {
    Fatal("G4MuonicData001",
          "G4MuonicDataTable::Enable() was called but no dataset directory was found: G4MUONICDATA "
          "is unset and GEANT4_DATA_DIR resolved none. Export G4MUONICDATA=<directory holding "
          "the *.g4dat files>.");
    return nullptr;
  }
  const std::string dir(found);
  LoadedDataset* loaded = new LoadedDataset;
  try {
    loaded->dataset = G4MuonicDataTable::Load(dir);
  } catch (const G4MuonicDataTable::Error& e) {
    Fatal("G4MuonicData002", "dataset at " + dir + " rejected: " + e.what());
    delete loaded;
    return nullptr;
  }
  const std::vector<std::string> profiles = loaded->dataset.Profiles();
  if (std::find(profiles.begin(), profiles.end(), profile) == profiles.end()) {
    std::string present;
    for (const std::string& name : profiles) present += (present.empty() ? "" : ", ") + name;
    Fatal("G4MuonicData004", "G4MUONICDATA_PROFILE names '" + profile + "' but no file in the dataset at " +
                                dir + " declares that '#PROFILE'; the profiles present are: " + present);
    delete loaded;
    return nullptr;
  }
  for (const UnitRule& rule : kUnitRules) {
    const std::string name(rule.table);
    const Table* table = loaded->dataset.Find(profile, name);
    const bool d1 = name == "nuclear_capture_rate" || name == "muon_zeff";
    if (table == nullptr) {
      // The parity profile carries every D1 table; another profile may lack one, and no profile
      // needs a D3 table: a table the profile lacks falls through to the compiled-in code for
      // every key.
      if (!d1 || profile != G4MuonicDataTable::kParityProfile) continue;
      Fatal("G4MuonicData003", "dataset at " + dir + " lacks #TABLE " + name + " under #PROFILE " + G4MuonicDataTable::kParityProfile);
      delete loaded;
      return nullptr;
    }
    Index index = 0;
    if (!FindColumn(*table, rule.column, index)) {
      Fatal("G4MuonicData003", "dataset at " + dir + " has no 'value' column in #TABLE " + name);
      delete loaded;
      return nullptr;
    }
    if (!CheckUnit(dir, *table, rule.column, rule.unit)) {
      delete loaded;
      return nullptr;
    }
    if (name == "nuclear_capture_rate") {
      loaded->capture = table;
      loaded->capture_value = index;
    } else if (name == "muon_zeff") {
      loaded->zeff = table;
      loaded->zeff_value = index;
    } else {
      loaded->kshell = table;
      loaded->kshell_value = index;
    }
  }
  if (const Table* levels = loaded->dataset.Find(profile, kLevelTable)) {
    // Every level column carries the unit the lookup reads, and the first of them is e2.
    for (const std::string& column : levels->columns) {
      if (IsLevelColumn(column) && !CheckUnit(dir, *levels, column, kLevelUnit)) {
        delete loaded;
        return nullptr;
      }
    }
    if (!CheckUnit(dir, *levels, "e2", kLevelUnit)) {
      delete loaded;
      return nullptr;
    }
    Index e2 = 0;
    FindColumn(*levels, "e2", e2);
    const std::vector<std::string>::const_iterator at = std::find(levels->columns.begin(), levels->columns.end(), "e2");
    G4int count = 0;
    for (std::vector<std::string>::const_iterator it = at; it != levels->columns.end(); ++it, ++count) {
      if (*it != "e" + std::to_string(count + 2)) break;
    }
    loaded->levels = levels;
    loaded->levels_e2 = e2;
    loaded->levels_count = count;
  }
  return loaded;
}

const LoadedDataset* Dataset() {
  static const LoadedDataset* const loaded = Load();
  return loaded;
}

}  // namespace

const G4MuonicDataTable* G4MuonicDataOverlay::Table() {
  const LoadedDataset* loaded = Dataset();
  return loaded == nullptr ? nullptr : &loaded->dataset;
}

const G4double* G4MuonicDataOverlay::Rate(G4int Z, G4int A) {
  const LoadedDataset* loaded = Dataset();
  if (loaded == nullptr || loaded->capture == nullptr) return nullptr;
  const G4MuonicDataTable::Table::Record* record = loaded->capture->LookupNatural({Z, A});
  return record == nullptr ? nullptr : &record->floats[loaded->capture_value];
}

const G4double* G4MuonicDataOverlay::Zeff(G4int Z) {
  const LoadedDataset* loaded = Dataset();
  if (loaded == nullptr || loaded->zeff == nullptr) return nullptr;
  const G4MuonicDataTable::Table::Record* record = loaded->zeff->Lookup({Z});
  return record == nullptr ? nullptr : &record->floats[loaded->zeff_value];
}

const G4double* G4MuonicDataOverlay::KShell(G4int Z, G4int A) {
  const LoadedDataset* loaded = Dataset();
  if (loaded == nullptr || loaded->kshell == nullptr) return nullptr;
  const G4MuonicDataTable::Table::Record* record = loaded->kshell->LookupNatural({Z, A});
  return record == nullptr ? nullptr : &record->floats[loaded->kshell_value];
}

const G4double* G4MuonicDataOverlay::Levels(G4int Z, G4int A, G4int& count) {
  count = 0;
  const LoadedDataset* loaded = Dataset();
  if (loaded == nullptr || loaded->levels == nullptr) return nullptr;
  const G4MuonicDataTable::Table::Record* record = loaded->levels->LookupNatural({Z, A});
  if (record == nullptr) return nullptr;
  count = loaded->levels_count;
  return &record->floats[loaded->levels_e2];
}
