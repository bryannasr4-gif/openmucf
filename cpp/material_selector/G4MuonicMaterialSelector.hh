// G4MuonicMaterialSelector -- opt-in measured capture vectors for the negative-muon capturing-element selector.
//
// Geant4 chooses the element whose nucleus takes a stopped negative muon in G4ElementSelector::SelectZandA, with
// weights factor(Z) * Z times the atom number density. This provider replaces those weights, for explicitly
// bound materials only, by a measured probability vector: a `material_vector` table of seam d2_atomic_capture
// (data/g4/d2/measured/). openmucf/g4/d2_material.py is the reference implementation of the rules below.
//
//   * Off unless the caller enables it. Disabled, the installed selector calls the stock
//     G4ElementSelector::SelectZandA exactly once; nothing in this provider reads a file unless the caller calls
//     G4MuonicMaterialCatalogue::Load.
//   * Binding is explicit: Bind(material, profile, formula, acknowledgeProvisional). The request is refused --
//     G4Exception G4MuonicData101, message "ENTRY_REFUSED: ...", nothing bound -- unless the acknowledgement is
//     given, the formula's counts are the entry's, the material's elements are exactly the entry's (matched by Z,
//     not by position; no extra element, hydrogen included), every atom number density is finite and positive,
//     and each density ratio equals the counts' ratio within 1e-9 relative, or within 1e-4 for exactly the named
//     Geant4 NIST oxides defined by weight fraction, each bound to the entry of its own formula.
//   * Only an unbound material falls back to the stock selector, silently and with one call.
//   * A bound material is sampled with the stock cumulative search over the bound probabilities in the
//     material's element order: one uniform draw for the element, then the stock isotope draw (one draw iff the
//     element has several isotopes); the target nucleus is set as the stock selector sets it. Stock element
//     selection is never called again for that material.
//   * The bindings are frozen before the run and only read afterwards. G4MuonicMaterialSelectorPhysics, registered
//     after the stopping physics, installs a fresh selector into the negative muon's muMinusCaptureAtRest and
//     into any G4MuonMinusAtomicCapture of every thread that builds processes. The process takes ownership of the
//     selector it is given; the installer keeps no pointer to it. If no stopping process exists when the
//     installer runs, it raises G4MuonicData103 instead of running with nothing installed.
#ifndef G4MUONICMATERIALSELECTOR_HH
#define G4MUONICMATERIALSELECTOR_HH

#include "G4ElementSelector.hh"
#include "G4MuonMinusAtomicCapture.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4Version.hh"

#include <atomic>
#include <map>
#include <string>
#include <vector>

class G4Material;

#if defined(G4MUONIC_ATOMIC_SETTER_SHIM)
// G4MuonMinusAtomicCapture declares SetElementSelector inline and no file of Geant4 11.4.2 or 11.5.0.beta defines
// it. The build enables this definition only after a link probe has shown the definition missing and the
// header's bytes are among those recorded in supported_headers.txt; a revision that defines it builds without
// the shim. The definition is the one G4HadronStoppingProcess gives its own setter.
#if !(G4VERSION_NUMBER == 1142 || G4VERSION_NUMBER == 1150)
#error "the G4MuonMinusAtomicCapture::SetElementSelector shim is limited to Geant4 11.4.2 and 11.5.0.beta"
#endif
inline void G4MuonMinusAtomicCapture::SetElementSelector(G4ElementSelector* ptr) {
  if (fElementSelector != ptr) {
    delete fElementSelector;
    fElementSelector = ptr;
  }
}
#endif

class G4MuonicMaterialCatalogue {
 public:
  struct Entry {
    std::string profile, formula;
    std::vector<int> z;      // ascending
    std::vector<double> p;   // probability per element, Z order
    std::vector<int> n;      // atoms per formula unit, Z order
  };
  // Read every d2_atomic_capture material_vector table of the directory; a violation raises G4MuonicData100,
  // and so does a table whose units are not the dimensionless P=1 n=1, whose scope is not
  // hydrogen_free_nominal_solid or whose status is not SOURCE_REPORTED_PROVISIONAL.
  static G4MuonicMaterialCatalogue Load(const std::string& directory);
  const Entry* Find(const std::string& profile) const;
  const std::vector<Entry>& Entries() const { return entries_; }

 private:
  std::vector<Entry> entries_;
};

class G4MuonicMaterialBindings {
 public:
  struct Bound {
    const G4MuonicMaterialCatalogue::Entry* entry = nullptr;
    std::vector<double> weights;  // in the material's element order
    double tolerance = 0.;
  };
  explicit G4MuonicMaterialBindings(const G4MuonicMaterialCatalogue* catalogue) : catalogue_(catalogue) {}
  // Bind or refuse. On refusal: G4Exception G4MuonicData101 (FatalErrorInArgument), nothing bound, false returned
  // when an exception handler lets execution continue.
  bool Bind(const G4Material* material, const std::string& profile, const std::string& formula,
            bool acknowledgeProvisional);
  void Freeze() { frozen_ = true; }
  bool Frozen() const { return frozen_; }
  const Bound* Find(const G4Material* material) const;
  std::size_t Size() const { return bound_.size(); }
  static double RatioTolerance(const G4Material* material, const std::string& formula);

 private:
  bool Refuse(const std::string& why) const;
  const G4MuonicMaterialCatalogue* catalogue_;
  std::map<const G4Material*, Bound> bound_;
  bool frozen_ = false;
};

class G4MuonicMaterialSelector : public G4ElementSelector {
 public:
  G4MuonicMaterialSelector(const G4MuonicMaterialBindings* bindings, bool enabled);
  ~G4MuonicMaterialSelector() override;
  const G4Element* SelectZandA(const G4Track& track, G4Nucleus* target) override;
  // The running sums this selector forms for a bound material (empty when unbound or disabled).
  std::vector<double> Cumulative(const G4Material* material) const;
  bool Enabled() const { return enabled_; }
  // Instances constructed and destroyed so far, process-wide: the ownership drill reads them.
  static long Constructed() { return constructed_.load(); }
  static long Destroyed() { return destroyed_.load(); }

 private:
  const G4MuonicMaterialBindings* bindings_;
  bool enabled_;
  std::vector<double> prob_;
  static std::atomic<long> constructed_, destroyed_;
};

class G4MuonicMaterialSelectorPhysics : public G4VPhysicsConstructor {
 public:
  // enabled == false installs selectors that delegate to the stock selector; the bindings may then be null.
  G4MuonicMaterialSelectorPhysics(const G4MuonicMaterialBindings* bindings, bool enabled);
  void ConstructParticle() override {}
  // Called in every thread that builds processes: installs a new selector into the negative muon's
  // muMinusCaptureAtRest (a G4HadronStoppingProcess) and into any G4MuonMinusAtomicCapture.
  void ConstructProcess() override;
  int Installed() const { return installed_.load(); }

 protected:
  virtual G4ElementSelector* CreateSelector() const;
  const G4MuonicMaterialBindings* bindings_;
  bool enabled_;

 private:
  std::atomic<int> installed_{0};  // ConstructProcess runs concurrently in workers
};

#endif
