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

struct LoadedDataset {
  G4MuonicDataTable dataset;
  const G4MuonicDataTable::Table* capture = nullptr;
  const G4MuonicDataTable::Table* zeff = nullptr;
  // Position of the `value` column inside Record::floats: #COLUMNS minus the integer key columns.
  Index capture_value = 0;
  Index zeff_value = 0;
};

// The index of `value` among the non-key columns, or false when the table declares no such column.
bool FindValueColumn(const G4MuonicDataTable::Table& table, Index& index) {
  Index position = 0;
  for (const std::string& column : table.columns) {
    if (column == "value") {
      index = position;
      return true;
    }
    if (column != "Z" && column != "A") ++position;
  }
  return false;
}

void Fatal(const char* code, const std::string& text) {
  G4Exception("G4MuonicDataOverlay", code, FatalException, text.c_str());
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
  const char* const names[] = {"nuclear_capture_rate", "muon_zeff"};
  for (const char* name : names) {
    const G4MuonicDataTable::Table* table = loaded->dataset.Find(profile, name);
    if (table == nullptr) {
      // The parity profile carries every table; another profile may lack one, and that table
      // then falls through to the compiled-in code for every key.
      if (profile != G4MuonicDataTable::kParityProfile) continue;
      Fatal("G4MuonicData003", "dataset at " + dir + " lacks #TABLE " + name + " under #PROFILE " + G4MuonicDataTable::kParityProfile);
      delete loaded;
      return nullptr;
    }
    Index index = 0;
    if (!FindValueColumn(*table, index)) {
      Fatal("G4MuonicData003", "dataset at " + dir + " has no 'value' column in #TABLE " + name);
      delete loaded;
      return nullptr;
    }
    if (table->name == "nuclear_capture_rate") {
      loaded->capture = table;
      loaded->capture_value = index;
    } else {
      loaded->zeff = table;
      loaded->zeff_value = index;
    }
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
