// G4MuonicMaterialSelector -- see the header for the rules. The drill macros (G4MUONIC_DRILL_*) exist only for the
// mutation drills that show the delivery, activation, draw-count, ownership and worker gates can fail; no build
// that delivers physics defines them.
#include "G4MuonicMaterialSelector.hh"

#include "G4Element.hh"
#include "G4Exception.hh"
#include "G4HadronStoppingProcess.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonicDataTable.hh"
#include "G4Nucleus.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

std::atomic<long> G4MuonicMaterialSelector::constructed_{0};
std::atomic<long> G4MuonicMaterialSelector::destroyed_{0};

namespace {

[[noreturn]] void CatalogueError(const std::string& what) {
  G4Exception("G4MuonicMaterialCatalogue::Load", "G4MuonicData100", FatalException, what.c_str());
  throw std::runtime_error(what);  // reached only if a handler lets execution continue
}

const char* kSymbols[] = {"H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
                          "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                          "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
                          "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
                          "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
                          "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
                          "Pa", "U"};

// Atoms per formula unit by Z, each element named once; ok == false on anything else.
std::map<int, int> FormulaCounts(const std::string& formula, bool& ok) {
  std::map<int, int> out;
  ok = false;
  static const std::regex part("([A-Z][a-z]?)([0-9]*)");
  std::size_t pos = 0;
  for (auto it = std::sregex_iterator(formula.begin(), formula.end(), part); it != std::sregex_iterator(); ++it) {
    if (static_cast<std::size_t>(it->position()) != pos) return out;
    int z = 0;
    for (int i = 0; i < 92; ++i)
      if ((*it)[1].str() == kSymbols[i]) z = i + 1;
    const std::string digits = (*it)[2].str();
    if (z == 0 || out.count(z) || (!digits.empty() && digits[0] == '0')) return out;
    out[z] = digits.empty() ? 1 : std::stoi(digits);
    pos += it->length();
  }
  ok = !formula.empty() && pos == formula.size();
  return out;
}

std::map<std::string, std::string> Tokens(const std::string& validity) {
  std::map<std::string, std::string> out;
  std::istringstream in(validity);
  std::string item;
  static const std::regex token("^([A-Za-z_][A-Za-z0-9_]*):(\\S+)$");
  while (in >> item) {
    std::smatch m;
    if (!std::regex_match(item, m, token)) CatalogueError("#VALIDITY item '" + item + "' is not NAME:VALUE");
    if (!out.emplace(m[1].str(), m[2].str()).second) CatalogueError("#VALIDITY repeats " + m[1].str());
  }
  return out;
}

// The named Geant4 NIST oxides, defined by weight fraction, and the formula of the entry each may bind to.
const std::map<std::string, std::string>& NistPolicy() {
  static const std::map<std::string, std::string> policy = {
      {"G4_ALUMINUM_OXIDE", "Al2O3"}, {"G4_CALCIUM_OXIDE", "CaO"},    {"G4_FERRIC_OXIDE", "Fe2O3"},
      {"G4_LEAD_OXIDE", "PbO"},       {"G4_MAGNESIUM_OXIDE", "MgO"}, {"G4_SILICON_DIOXIDE", "SiO2"},
      {"G4_TITANIUM_DIOXIDE", "TiO2"}};
  return policy;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ catalogue
G4MuonicMaterialCatalogue G4MuonicMaterialCatalogue::Load(const std::string& directory) {
  G4MuonicMaterialCatalogue cat;
  G4MuonicDataTable tables;
  try {
    tables = G4MuonicDataTable::Load(directory);
  } catch (const G4MuonicDataTable::Error& e) {
    CatalogueError("catalogue " + directory + ": " + e.what());
  }
  std::set<std::string> seen;
  for (const auto& t : tables.Tables()) {
    const std::string* seam = t.Directive("SEAM");
    if (seam == nullptr || *seam != "d2_atomic_capture") continue;
    if (t.name != "material_vector") CatalogueError(t.profile + ": table " + t.name + " is not material_vector");
    if (t.columns != std::vector<std::string>{"Z", "P", "n"}) CatalogueError(t.profile + ": columns");
    const std::string* validity = t.Directive("VALIDITY");
    auto tok = Tokens(validity ? *validity : "");
    std::set<std::string> got;
    for (const auto& kv : tok) got.insert(kv.first);
    if (got != std::set<std::string>{"form", "basis", "scope", "status", "material"} || tok["form"] != "vector" ||
        tok["basis"] != "atom")
      CatalogueError(t.profile + ": #VALIDITY");
    Entry e;
    e.profile = t.profile;
    e.formula = tok["material"];
    if (!seen.insert(e.profile).second) CatalogueError("profile " + e.profile + " appears twice");
    if (t.records.size() < 2) CatalogueError(e.profile + ": fewer than two elements");
    long double sum = 0.L;
    for (const auto& r : t.records) {
      const double p = r.floats.at(0), n = r.floats.at(1);
      if (!(std::isfinite(p) && p > 0. && p < 1.)) CatalogueError(e.profile + ": a probability outside (0, 1)");
      if (!(std::isfinite(n) && n >= 1. && n == std::floor(n) && n < 1e6)) CatalogueError(e.profile + ": a count");
      e.z.push_back(static_cast<int>(r.keys.at(0)));
      e.p.push_back(p);
      e.n.push_back(static_cast<int>(n));
      sum += p;
    }
    if (std::fabs(static_cast<double>(sum) - 1.) > 4. * DBL_EPSILON) CatalogueError(e.profile + ": P sum");
    bool ok = false;
    const auto counts = FormulaCounts(e.formula, ok);
    std::map<int, int> mine;
    for (std::size_t i = 0; i < e.z.size(); ++i) mine[e.z[i]] = e.n[i];
    if (!ok || counts != mine) CatalogueError(e.profile + ": counts differ from the formula " + e.formula);
    cat.entries_.push_back(e);
  }
  return cat;
}

const G4MuonicMaterialCatalogue::Entry* G4MuonicMaterialCatalogue::Find(const std::string& profile) const {
  for (const auto& e : entries_)
    if (e.profile == profile) return &e;
  return nullptr;
}

// ------------------------------------------------------------------------------------------------ bindings
bool G4MuonicMaterialBindings::Refuse(const std::string& why) const {
  G4Exception("G4MuonicMaterialBindings::Bind", "G4MuonicData101", FatalErrorInArgument,
              ("ENTRY_REFUSED: " + why + "; nothing was bound").c_str());
  return false;
}

double G4MuonicMaterialBindings::RatioTolerance(const G4Material* material, const std::string& formula) {
  const auto it = NistPolicy().find(material->GetName());
  return (it != NistPolicy().end() && it->second == formula) ? 1e-4 : 1e-9;
}

bool G4MuonicMaterialBindings::Bind(const G4Material* material, const std::string& profile,
                                    const std::string& formula, bool acknowledgeProvisional) {
  if (frozen_) return Refuse("the bindings are frozen");
  if (material == nullptr || catalogue_ == nullptr) return Refuse("no material or no catalogue");
  if (!acknowledgeProvisional) return Refuse("the provisional, source-reported status was not acknowledged");
  if (bound_.count(material)) return Refuse("material " + material->GetName() + " is already bound");
  const auto* e = catalogue_->Find(profile);
  if (e == nullptr) return Refuse("no entry " + profile);
  bool ok = false;
  const auto counts = FormulaCounts(formula, ok);
  std::map<int, int> want;
  for (std::size_t i = 0; i < e->z.size(); ++i) want[e->z[i]] = e->n[i];
  if (!ok || counts != want) return Refuse("the formula " + formula + " differs from the entry's " + e->formula);
  const std::size_t n = material->GetNumberOfElements();
  std::vector<int> zm;
  std::set<int> zs;
  for (std::size_t i = 0; i < n; ++i) {
    zm.push_back(material->GetElement(static_cast<G4int>(i))->GetZasInt());
    if (!zs.insert(zm.back()).second) return Refuse("an element repeats");
  }
  std::set<int> ze(e->z.begin(), e->z.end());
  if (zs != ze || n != e->z.size()) return Refuse("the material's elements are not the entry's");
  const G4double* dens = material->GetAtomicNumDensityVector();
  for (std::size_t i = 0; i < n; ++i)
    if (!(std::isfinite(dens[i]) && dens[i] > 0.)) return Refuse("an atom number density is not finite and positive");
  const double tol = RatioTolerance(material, e->formula);
  for (std::size_t i = 1; i < n; ++i) {
    const double ratio = (dens[i] / dens[0]) / (static_cast<double>(want[zm[i]]) / want[zm[0]]);
    if (!(std::fabs(ratio - 1.) <= tol)) return Refuse("an atom-number-density ratio differs from the counts");
  }
  Bound bd;
  bd.entry = e;
  bd.tolerance = tol;
  for (std::size_t i = 0; i < n; ++i) {
#if defined(G4MUONIC_DRILL_POSITIONAL)
    const std::size_t k = i;  // drill: match by position instead of by Z
#else
    const std::size_t k = std::find(e->z.begin(), e->z.end(), zm[i]) - e->z.begin();
#endif
    bd.weights.push_back(e->p[k]);
  }
  bound_.emplace(material, bd);
  return true;
}

const G4MuonicMaterialBindings::Bound* G4MuonicMaterialBindings::Find(const G4Material* material) const {
  const auto it = bound_.find(material);
  return it == bound_.end() ? nullptr : &it->second;
}

// ------------------------------------------------------------------------------------------------ selector
G4MuonicMaterialSelector::G4MuonicMaterialSelector(const G4MuonicMaterialBindings* bindings, bool enabled)
    : bindings_(bindings), enabled_(enabled) {
  ++constructed_;
}

G4MuonicMaterialSelector::~G4MuonicMaterialSelector() { ++destroyed_; }

std::vector<double> G4MuonicMaterialSelector::Cumulative(const G4Material* material) const {
  std::vector<double> out;
  const auto* b = (enabled_ && bindings_) ? bindings_->Find(material) : nullptr;
  if (b == nullptr) return out;
  double sum = 0.;
  for (double w : b->weights) {
    sum += w;
    out.push_back(sum);
  }
  return out;
}

const G4Element* G4MuonicMaterialSelector::SelectZandA(const G4Track& track, G4Nucleus* target) {
#if defined(G4MUONIC_DRILL_FORCE_ACTIVE)
  const bool active = true;
#else
  const bool active = enabled_;
#endif
  const G4Material* mat = track.GetMaterial();
  const G4MuonicMaterialBindings::Bound* b = (active && bindings_) ? bindings_->Find(mat) : nullptr;
#if defined(G4MUONIC_DRILL_ENABLED_DELEGATES)
  b = nullptr;
#endif
  if (b == nullptr) return G4ElementSelector::SelectZandA(track, target);

  const std::size_t n = mat->GetNumberOfElements();
  const G4ElementVector* elements = mat->GetElementVector();
  std::size_t i = 0;
  if (n > 1) {
    if (prob_.size() < n) prob_.resize(n, 0.);
    G4double sum = 0.;
    for (i = 0; i < n; ++i) {
      sum += b->weights[i];
      prob_[i] = sum;
    }
    sum *= G4UniformRand();
    for (i = 0; i < n; ++i) {
      if (sum <= prob_[i]) break;
    }
    if (i == n) i = n - 1;
  }
  const G4Element* elm = (*elements)[i];
  const G4int Z = elm->GetZasInt();
  // the stock isotope rule, unchanged
  const G4IsotopeVector* isv = elm->GetIsotopeVector();
  const std::size_t ni = isv->size();
  i = 0;
  if (ni > 1) {
    const G4double* ab = elm->GetRelativeAbundanceVector();
    G4double y = G4UniformRand();
    for (i = 0; i < ni; ++i) {
      y -= ab[i];
      if (y <= 0.0) break;
    }
    if (i == ni) i = ni - 1;
  }
#if defined(G4MUONIC_DRILL_EXTRA_ENABLED_DRAW)
  (void)G4UniformRand();
#endif
  const G4int A = elm->GetIsotope(static_cast<G4int>(i))->GetN();
  target->SetParameters(A, Z);
  return elm;
}

// ------------------------------------------------------------------------------------------------ installer
G4MuonicMaterialSelectorPhysics::G4MuonicMaterialSelectorPhysics(const G4MuonicMaterialBindings* bindings,
                                                                 bool enabled)
    : G4VPhysicsConstructor("G4MuonicMaterialSelector"), bindings_(bindings), enabled_(enabled) {
  if (enabled_ && (bindings_ == nullptr || !bindings_->Frozen())) {
    G4Exception("G4MuonicMaterialSelectorPhysics", "G4MuonicData102", FatalException,
                "an enabled selector needs frozen bindings");
  }
}

G4ElementSelector* G4MuonicMaterialSelectorPhysics::CreateSelector() const {
  return new G4MuonicMaterialSelector(bindings_, enabled_);
}

void G4MuonicMaterialSelectorPhysics::ConstructProcess() {
#if defined(G4MUONIC_DRILL_MASTER_ONLY)
  if (!G4Threading::IsMasterThread()) return;
#endif
  auto* list = G4MuonMinus::MuonMinus()->GetProcessManager()->GetProcessList();
  int here = 0;
  for (std::size_t k = 0; k < list->size(); ++k) {
    G4VProcess* p = (*list)[k];
    if (auto* s = dynamic_cast<G4HadronStoppingProcess*>(p); s && s->GetProcessName() == "muMinusCaptureAtRest") {
      G4ElementSelector* selector = CreateSelector();
      s->SetElementSelector(selector);
#if defined(G4MUONIC_DRILL_INSTALLER_DELETES)
      // drill: the installer keeps a deleting pointer to a selector the process also owns
      static thread_local std::vector<std::unique_ptr<G4ElementSelector>> held;
      held.emplace_back(selector);
#endif
      ++here;
    } else if (auto* a = dynamic_cast<G4MuonMinusAtomicCapture*>(p)) {
      a->SetElementSelector(CreateSelector());
      ++here;
    }
  }
  if (here == 0) {
    G4Exception("G4MuonicMaterialSelectorPhysics::ConstructProcess", "G4MuonicData103", FatalException,
                "no negative-muon stopping or atomic-capture process exists yet: register this constructor after "
                "the stopping physics");
  }
  installed_ += here;
}
