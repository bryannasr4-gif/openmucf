// G4MuonicDataSemantics -- what a G4MuonicData dataset must mean, over and above what its grammar
// allows.
//
// The reader (G4MuonicDataTable) enforces the Layer-1 grammar: E001-E016 say whether a file is a
// well-formed `.g4dat` document. A file can be well-formed and still describe something no
// consumer can use -- a binding energy below the level it sits above, a capture rate that is
// negative, a `#TABLE` name nothing reads, a `#UNITS` assignment a lookup would misread, two files
// of one profile that disagree about which nuclides they cover. Those are semantic rules, and they
// live here so that the standalone validator and the Geant4-facing glue apply exactly the same
// ones to exactly the same bytes.
//
// Nothing here depends on Geant4: standard library plus the reader. `double` and `int`, not
// `G4double` and `G4int`, for that reason.
//
// The codes, and the one sentence each stands for:
//   S001  identity and version: every table names this dataset, and one `#VERSION` spans the load.
//   S002  shape: the `#TABLE` name, its `#SEAM`, its `#COLUMNS` sequence, and that it has records.
//   S003  keys and validity: the key domain, and a `#VALIDITY` convention this layer knows.
//   S004  units: the `#UNITS` assignment of every value column, as the lookups read it.
//   S005  values: finite, and in the domain the quantity has (a rate and an energy are positive,
//         an uncertainty is not negative, an effective charge does not exceed Z).
//   S006  the D3 pair: one profile's two energy tables cover one key set, in one falling order.
#ifndef G4MUONICDATASEMANTICS_HH
#define G4MUONICDATASEMANTICS_HH

#include "G4MuonicDataTable.hh"

#include <string>
#include <vector>

class G4MuonicDataSemantics {
 public:
  // One violation: the code above, the file and the `#PROFILE` / `#TABLE` it was found in, and a
  // sentence naming what is wrong. `file`, `profile` and `table` are empty for an issue that is
  // about a profile rather than one table.
  struct Issue {
    std::string code, file, profile, table, text;
  };

  // Every issue in `dataset`, in loaded-file order and, within a file, in record order; the
  // profile-level S006 issues follow the per-table ones, in bytewise profile order. An empty
  // vector is the only acceptable result for a dataset a consumer may read.
  static std::vector<Issue> Validate(const G4MuonicDataTable& dataset);
};

#endif
