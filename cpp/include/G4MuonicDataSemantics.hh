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
//   S003  keys and validity: the key domain, and a `#VALIDITY` convention this layer knows (the
//         bound-rate pair table carries isotope keys only, `A:listed`).
//   S004  units: the `#UNITS` assignment of every value column, as the lookups read it.
//   S005  values: finite, and in the domain the quantity has (a rate and an energy are positive,
//         an uncertainty is not negative, an effective charge does not exceed Z, a pair tuple's q
//         lies in (0, 1] and its listed stopped-route transform satisfies BoundPairProblem).
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

  // The consumer transform of one bound-rate pair tuple (tau and free_tau in ns, q dimensionless), in the order
  // the consumers evaluate it. The tuple checks below cover rates, lifetimes, probability, reciprocal read-back
  // and waiting-time expressions with multiplier 745. All listed quantities must be finite and in their stated
  // domains except the infinite capture lifetime stored for zero capture. Absolute track times and kinematic
  // GetMeanFreePath returns are outside these checks; nothing is clipped.
  //   tau and free_tau finite and positive, q finite in (0, 1];
  //   the decay rate d = q / free_tau finite and in (0, 1/free_tau] (a quotient that underflows is refused);
  //   1/tau finite, and the capture rate c = 1/tau - d finite and not negative (c = 0 is admitted);
  //   the total rate c + d finite and positive. The consumer lifetime `1/(capture + decay)` must be finite
  //   and positive;
  //   the capture probability c / (c + d) in [0, 1];
  //   the partial lifetimes the atomic route stores: 1/d finite, and 1/c finite when c > 0 (c = 0 stores an
  //   infinite capture lifetime, read back as a zero capture rate, so capture is never chosen); the rates
  //   read back from them, 1/(1/c) + 1/(1/d), finite and positive;
  //   a sampled waiting time: kExponentialVariateMax times the lifetime, and over the total rate, finite.
  // Returns nullptr and sets `capture` and `decay` for an admitted tuple, otherwise the reason. The semantic
  // rule S005 and the Geant4 provider call this one function, so they cannot disagree.
  static const char* BoundPairProblem(double tau, double q, double free_tau, double& capture, double& decay);

  static constexpr double kExponentialVariateMax = 745.;
};

#endif
