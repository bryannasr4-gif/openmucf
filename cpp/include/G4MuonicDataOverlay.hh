// G4MuonicDataOverlay -- the Geant4-facing glue between the G4MuonicData dataset reader and the
// compiled-in muon-capture and muonic-atom energy code paths.
//
// The reader (G4MuonicDataTable) knows nothing about Geant4. This class adds what a consumer inside
// Geant4 needs: dataset discovery through G4FindDataDir, the semantic layer
// (G4MuonicDataSemantics) applied to every loaded table before any of them is selected, a fatal
// G4Exception when the opt-in was requested but the dataset is absent, invalid or does not mean
// what a consumer may read, and keyed lookups that say which rung answered. With the opt-in off
// (G4MuonicDataTable::Enable() not called) every entry point below returns without reading the
// dataset or performing I/O; CheckComputedCaptureRate is the one the patched code calls even then.
//
// Two seams are selected independently. G4MUONICDATA_D1_PROFILE names the profile the nuclear
// capture seam reads through and G4MUONICDATA_D3_PROFILE the one the muonic-transition seam reads
// through; G4MUONICDATA_PROFILE names both where neither is set, and `parity` (D1) with the
// compiled-in code (D3) is what an unset environment selects. `compiled` is a reserved token: a
// seam that resolves to it has no tables and every request there falls through to the consumer's
// own code. The environment is read once, when the opt-in is set, and not again.
//
// A third selection is independent of both: G4MUONICDATA_D1_PAIR_PROFILE names the profile whose
// `bound_rate_primitives` table carries evaluated bound-muon rate pairs, one tuple (tau, q, free_tau) per
// isotope, which G4MuonicBoundRateProvider gives both stopped-muon routes in place of the common pair for the
// keys it holds. Unset, it is `compiled` (no pair table). It does not replace the capture profile: keys the
// pair table does not hold keep the D1 profile's capture rate. G4MUONICDATA_PROFILE never selects it, and a
// G4MUONICDATA_PROFILE naming a pair profile is refused (G4MuonicData004), as is a pair profile that carries
// no pair table. Pair lookups are by isotope only.
//
// The legacy natural-composition rung -- a request for (Z, A) answered by the element's (Z, 0)
// record -- is used exactly when the resolved profile for that seam is `parity`. Under any
// evaluated profile a request with A > 0 is answered from that isotope or not at all, and the
// (Z, 0) record is reachable only by asking for A = 0.
#ifndef G4MUONICDATAOVERLAY_HH
#define G4MUONICDATAOVERLAY_HH

#include "G4MuonicDataTable.hh"
#include "G4Types.hh"

#include <string>

class G4MuonicDataOverlay {
 public:
  // Which rung answered a request, or that none did.
  enum class Origin {
    Exact,                  // the record keyed by exactly the nuclide asked for
    ExplicitNatural,        // the (Z, 0) record of a table under `A:natural_and_listed`, asked for
    ExplicitRepresentative, // the (Z, 0) record of a table under `A:most_abundant_and_listed`, asked for
    LegacyNaturalFallback,  // the (Z, 0) record reached from a request for (Z, A > 0), `parity` only
    Compiled                // no record: the consumer's compiled-in code answers
  };

  // One answer. `values` points at `count` consecutive stored doubles, or is null with `count` 0
  // and both resolved keys 0 when nothing matched -- that record, with the requested keys kept, is
  // the whole of what "the dataset does not support this nuclide" means here.
  struct Resolution {
    const G4double* values = nullptr;
    G4int count = 0;
    G4int requestedZ = 0, requestedA = 0, resolvedZ = 0, resolvedA = 0;
    Origin origin = Origin::Compiled;
    std::string profile;
  };

  // What the environment selected, frozen when the opt-in was set. Both profiles are `compiled`
  // and both strings empty with the opt-in off.
  // `fallbackDeclarationExecuted` says
  // whether a `#FALLBACK` declaration was evaluated: it is always false, because a declaration is
  // documentary here and no expression of one is interpreted.
  struct Configuration {
    std::string datasetDirectory, datasetVersion, d1Profile, d3Profile;
    bool fallbackDeclarationExecuted = false;
    // The pair profile G4MUONICDATA_D1_PAIR_PROFILE selected, `compiled` when none.
    std::string d1PairProfile;
  };

  // Read the environment, discover and read the dataset, and freeze the configuration. Called once
  // when the opt-in is set, in unlocked pre-initialisation on the master thread. Calling it again
  // reuses the frozen configuration, as does a lookup made without it.
  static void Initialize();
  static const Configuration& Config();

  // The dataset, or nullptr.
  static const G4MuonicDataTable* Table();

  // The `value` field of the nuclear_capture_rate record for this nuclide, in the dataset's own
  // units as declared by its #UNITS line, or nullptr when nothing matched.
  static const G4double* Rate(G4int Z, G4int A);
  // The `value` field of the muon_zeff record keyed Z, or nullptr when nothing matched.
  static const G4double* Zeff(G4int Z);
  // The `value` field (keV) of the k_shell_energy record for this nuclide, or nullptr.
  static const G4double* KShell(G4int Z, G4int A);
  // The `e2` field (keV) of the level_energy record for this nuclide, with `count` set to the
  // number of consecutive columns e2, e3, ... the table declares; nullptr and a count of zero when
  // nothing matched.
  static const G4double* Levels(G4int Z, G4int A, G4int& count);

  // The same four lookups, with the rung that answered and the profile it was read through.
  static Resolution ResolveRate(G4int Z, G4int A);
  // The bound-rate pair tuple for this isotope under the pair profile: `values` points at tau (ns), q and
  // free_tau (ns), `count` 3; null for A <= 0, for a key the table does not hold, and with no pair profile.
  static Resolution ResolvePair(G4int Z, G4int A);
  static Resolution ResolveZeff(G4int Z);
  static Resolution ResolveKShell(G4int Z, G4int A);
  static Resolution ResolveLevels(G4int Z, G4int A);

  // A capture rate the consumer computed itself, once the dataset had no record for the nuclide.
  // A no-op unless the effective D1 profile is an evaluated one: under `parity` and under the
  // compiled-in seam the documented Goulard-Primakoff output, negative values included, stands.
  // Under an evaluated profile a rate that is not finite is fatal (G4MuonicData005); a finite rate
  // that is not positive stands as computed, and the first such rate in the process raises the
  // warning G4MuonicData006.
  static void CheckComputedCaptureRate(G4int Z, G4int A, G4double rate);
  // The assembled muonic-cascade level array, after at least one lookup overlaid part of it: the
  // whole array must be finite, positive and strictly falling, boundary between overlaid and
  // hydrogen-like values included.
  static void CheckCascadeLevels(G4int Z, G4int A, const G4double* levels, G4int count);
};

#endif
