// harvest_material.cc -- what G4MuonicMaterialSelector delivers at Geant4's real capture call sites.
//
// A negative muon starts at rest at the centre of a box of one material, on one of two routes: "stopping" (QBBC;
// G4HadronStoppingProcess::AtRestDoIt selects the nucleus, then hands it to G4MuonMinusBoundDecay) or "atomic"
// (QBBC with the stopping physics' muon-minus capture off, G4MuonMinusAtomicCapture and G4MuonicAtomDecayPhysics;
// AtRestDoIt selects the nucleus of the muonic atom it creates). Run managers: Serial, or Tasking with --threads
// workers. The provider is registered last; --enable absent registers none.
//
// Modes:
//   probe     read the catalogue; for every entry build its nominal material from Geant4's NIST elements in
//             ascending-Z and in descending-Z order, bind it, and print the atom densities, the bound weights,
//             the running sums and the counts of a midpoint ladder of --draws draws; then print, for each named
//             NIST oxide, its atom densities and the binding outcome against the entry of its formula.
//   ladder    one event per ladder index k = event id: the selector's uniform draws (and only those) come from a
//             per-thread queue holding u_k = (k + 1/2)/N then 1/2, swapped in around the selector call; every other
//             draw comes from the run's engine, reseeded per event. Secondaries are killed after the capture step.
//             Records per worker the selected (Z, A), the nucleus handed onward (stopping: the nucleus the bound
//             decay receives; atomic: the muonic atom's base ion), the draws consumed against the stock rule and
//             the selector calls.
//   identity  full transport, --events events seeded seed base + event id; per event, the ordered numerical trace
//             of every track start and step (hexfloat energies and times, processes, secondaries) with the engine
//             state hash folded in at every point, written sorted by event id for a byte comparison.
//   bind      the binding's refusal cases, with an exception handler that records G4MuonicData101.
//   teardown  Tasking, three runs, then the run manager deleted: the selectors constructed and destroyed.
//   order     a modular list whose provider is registered before (or, as the control, after) the stopping
//             physics: the installer must raise G4MuonicData103 when it finds nothing to install into.
#include "G4MuonicMaterialSelector.hh"

#include "G4Box.hh"
#include "G4DecayPhysics.hh"
#include "G4Element.hh"
#include "G4EmStandardPhysics.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4HadronStoppingProcess.hh"
#include "G4HadronicParameters.hh"
#include "G4IonTable.hh"
#include "G4Isotope.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusAtomicCapture.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtom.hh"
#include "G4MuonicAtomDecayPhysics.hh"
#include "G4NistManager.hh"
#include "G4Nucleus.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4RunManagerFactory.hh"
#include "G4StateManager.hh"
#include "G4Step.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VExceptionHandler.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "QBBC.hh"
#include "Randomize.hh"

#include <CLHEP/Random/MixMaxRng.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string mode, route = "stopping", manager = "serial", material = "al2o3", catalogue, profile = "none",
                    formula = "none", enable = "absent", data = "off", order = "after", out;
  int threads = 2, events = 100;
  long draws = 1L << 20, seed_base = 17;
};

Options g_opt;

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i += 2) {
    const std::string k(argv[i]);
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
    const std::string v(argv[i + 1]);
    if (k == "--mode") o.mode = v;
    else if (k == "--route") o.route = v;
    else if (k == "--manager") o.manager = v;
    else if (k == "--material") o.material = v;
    else if (k == "--catalogue") o.catalogue = v;
    else if (k == "--profile") o.profile = v;
    else if (k == "--formula") o.formula = v;
    else if (k == "--enable") o.enable = v;
    else if (k == "--data") o.data = v;
    else if (k == "--order") o.order = v;
    else if (k == "--out") o.out = v;
    else if (k == "--threads") o.threads = std::stoi(v);
    else if (k == "--events") o.events = std::stoi(v);
    else if (k == "--draws") o.draws = std::stol(v);
    else if (k == "--seed-base") o.seed_base = std::stol(v);
    else throw std::runtime_error("unknown argument " + k);
  }
  const std::set<std::string> modes = {"probe", "ladder", "identity", "bind", "teardown", "order"};
  const std::set<std::string> materials = {"al2o3", "al2o3_rev", "pbo_nist", "pbo_rev"};
  const std::set<std::string> enables = {"absent", "off", "off_bound", "on", "on_unbound"};
  if (!modes.count(o.mode) || (o.route != "stopping" && o.route != "atomic") ||
      (o.manager != "serial" && o.manager != "tasking") || !materials.count(o.material) || !enables.count(o.enable) ||
      (o.data != "off" && o.data != "on") || (o.order != "after" && o.order != "before") || o.out.empty() ||
      o.threads < 1 || o.draws < 1 || o.events < 1 || o.seed_base < 1)
    throw std::runtime_error("invalid options");
  return o;
}

std::string Hex(double x) {
  char b[64];
  std::snprintf(b, sizeof(b), "%a", x);
  return b;
}

std::uint64_t EngineHash() {
  const std::vector<unsigned long> words = G4Random::getTheEngine()->put();
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned long w : words)
    for (int b = 0; b < 8; ++b) {
      h ^= static_cast<std::uint64_t>((w >> (8 * b)) & 0xffUL);
      h *= 1099511628211ULL;
    }
  return h;
}

G4Element* Nist(int z) { return G4NistManager::Instance()->FindOrBuildElement(z); }

// A material of the formula's counts from NIST elements, in ascending (or descending) Z order.
G4Material* ByCounts(const std::string& name, const std::vector<std::pair<int, int>>& counts, bool descending) {
  if (auto* m = G4Material::GetMaterial(name, false)) return m;
  auto ordered = counts;
  std::sort(ordered.begin(), ordered.end());
  if (descending) std::reverse(ordered.begin(), ordered.end());
  auto* m = new G4Material(name, 1. * g / cm3, static_cast<G4int>(ordered.size()), kStateSolid, 293.15 * kelvin);
  for (const auto& zc : ordered) m->AddElement(Nist(zc.first), zc.second);
  return m;
}

G4Material* BuildMaterial(const std::string& kind) {
  if (kind == "al2o3") return ByCounts("harvest_Al2O3", {{13, 2}, {8, 3}}, true);   // Al, O
  if (kind == "al2o3_rev") return ByCounts("harvest_Al2O3_rev", {{13, 2}, {8, 3}}, false);  // O, Al
  if (kind == "pbo_nist") return G4NistManager::Instance()->FindOrBuildMaterial("G4_LEAD_OXIDE");
  return ByCounts("harvest_PbO_rev", {{82, 1}, {8, 1}}, true);  // Pb, O: the NIST material lists O first
}

std::string FormulaOf(const std::string& kind) { return kind.rfind("al2o3", 0) == 0 ? "Al2O3" : "PbO"; }
std::string ProfileOf(const std::string& kind) {
  return kind.rfind("al2o3", 0) == 0 ? "stanislaus1987-al2o3" : "stanislaus1987-pbo";
}

// ---------------------------------------------------------------------------------------------- scripted draws
class QueueEngine final : public CLHEP::HepRandomEngine {
 public:
  void Fill(double first) {
    values_[0] = first;
    values_[1] = 0.5;
    cursor_ = 0;
  }
  int Consumed() const { return cursor_; }
  double flat() override { return cursor_ < 2 ? values_[cursor_++] : (++cursor_, 0.5); }
  void flatArray(const int n, double* a) override {
    for (int i = 0; i < n; ++i) a[i] = flat();
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[] = "Config.conf") const override {}
  void restoreStatus(const char[] = "Config.conf") override {}
  void showStatus() const override {}
  std::string name() const override { return "HarvestQueue"; }

 private:
  double values_[2] = {0.5, 0.5};
  int cursor_ = 0;
};

struct Tally {
  std::map<std::pair<int, int>, long> selected, onward;
  long calls = 0, draw_mismatch = 0, onward_mismatch = 0, events = 0;
};

std::mutex g_mutex;
std::map<int, Tally> g_tally;                     // worker id -> tally
std::map<int, std::vector<std::string>> g_trace;  // event id -> trace lines

struct EventState {
  int sel_z = 0, sel_a = 0, onward_z = 0, onward_a = 0, selector_calls = 0;
  std::uint64_t fold = 1469598103934665603ULL;
  std::vector<std::string> lines;
};
G4ThreadLocal EventState* t_event = nullptr;
G4ThreadLocal Tally* t_tally = nullptr;
G4ThreadLocal QueueEngine* t_queue = nullptr;

EventState& Ev() {
  if (t_event == nullptr) t_event = new EventState;
  return *t_event;
}
Tally& Ta() {
  if (t_tally == nullptr) t_tally = new Tally;
  return *t_tally;
}

void Fold(std::uint64_t h) {
  for (int b = 0; b < 8; ++b) {
    Ev().fold ^= (h >> (8 * b)) & 0xffULL;
    Ev().fold *= 1099511628211ULL;
  }
}

template <class Base>
class Scripted : public Base {
 public:
  using Base::Base;
  const G4Element* SelectZandA(const G4Track& track, G4Nucleus* target) override {
    if (t_queue == nullptr) t_queue = new QueueEngine;
    const int id = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    t_queue->Fill((static_cast<double>(id) + 0.5) / static_cast<double>(g_opt.draws));
    CLHEP::HepRandomEngine* previous = G4Random::getTheEngine();
    G4Random::setTheEngine(t_queue);
    const G4Element* e = Base::SelectZandA(track, target);
    G4Random::setTheEngine(previous);
    const auto* mat = track.GetMaterial();
    const int expected = (mat->GetNumberOfElements() > 1 ? 1 : 0) + (e->GetIsotopeVector()->size() > 1 ? 1 : 0);
    if (t_queue->Consumed() != expected) ++Ta().draw_mismatch;
    ++Ta().calls;
    Ev().sel_z = target->GetZ_asInt();
    Ev().sel_a = target->GetA_asInt();
    ++Ev().selector_calls;
    return e;
  }
};

class SpyBoundDecay : public G4MuonMinusBoundDecay {
 public:
  G4HadFinalState* ApplyYourself(const G4HadProjectile& p, G4Nucleus& n) override {
    Ev().onward_z = n.GetZ_asInt();
    Ev().onward_a = n.GetA_asInt();
    return G4MuonMinusBoundDecay::ApplyYourself(p, n);
  }
};

void InstallSpy() {
  auto* list = G4MuonMinus::MuonMinus()->GetProcessManager()->GetProcessList();
  for (std::size_t k = 0; k < list->size(); ++k)
    if (auto* s = dynamic_cast<G4HadronStoppingProcess*>((*list)[k]); s && s->GetProcessName() == "muMinusCaptureAtRest")
      s->SetBoundDecay(new SpyBoundDecay());
}

// The provider installer, with the ladder's scripted wrapper and the stopping route's bound-decay spy.
class HarnessPhysics : public G4MuonicMaterialSelectorPhysics {
 public:
  HarnessPhysics(const G4MuonicMaterialBindings* b, bool enabled, bool scripted)
      : G4MuonicMaterialSelectorPhysics(b, enabled), scripted_(scripted) {}
  void ConstructProcess() override {
    G4MuonicMaterialSelectorPhysics::ConstructProcess();
    if (scripted_) InstallSpy();
  }

 protected:
  G4ElementSelector* CreateSelector() const override {
    if (scripted_) return new Scripted<G4MuonicMaterialSelector>(bindings_, enabled_);
    return G4MuonicMaterialSelectorPhysics::CreateSelector();
  }

 private:
  bool scripted_;
};

// --enable absent in ladder mode: the stock selector, scripted the same way, installed per thread.
class StockLadderPhysics : public G4VPhysicsConstructor {
 public:
  StockLadderPhysics() : G4VPhysicsConstructor("HarvestStockLadder") {}
  void ConstructParticle() override {}
  void ConstructProcess() override {
    auto* list = G4MuonMinus::MuonMinus()->GetProcessManager()->GetProcessList();
    for (std::size_t k = 0; k < list->size(); ++k) {
      if (auto* s = dynamic_cast<G4HadronStoppingProcess*>((*list)[k]); s && s->GetProcessName() == "muMinusCaptureAtRest")
        s->SetElementSelector(new Scripted<G4ElementSelector>());
      else if (auto* a = dynamic_cast<G4MuonMinusAtomicCapture*>((*list)[k]))
        a->SetElementSelector(new Scripted<G4ElementSelector>());
    }
    InstallSpy();
  }
};

class AtomicCapturePhysics : public G4VPhysicsConstructor {
 public:
  AtomicCapturePhysics() : G4VPhysicsConstructor("MuAtomicCapture") {}
  void ConstructParticle() override { G4MuonMinus::MuonMinus(); }
  void ConstructProcess() override {
    G4MuonMinus::MuonMinus()->GetProcessManager()->AddRestProcess(new G4MuonMinusAtomicCapture());
  }
};

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* logical = new G4LogicalVolume(new G4Box("world", 1. * m, 1. * m, 1. * m), BuildMaterial(g_opt.material),
                                        "world");
    return new G4PVPlacement(nullptr, G4ThreeVector(), logical, "world", nullptr, false, 0);
  }
};

class Primary : public G4VUserPrimaryGeneratorAction {
 public:
  Primary() : gun_(1) {
    gun_.SetParticleDefinition(G4MuonMinus::MuonMinus());
    gun_.SetParticleEnergy(0.);
    gun_.SetParticlePosition(G4ThreeVector());
    gun_.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    gun_.SetParticleTime(0.);
  }
  void GeneratePrimaries(G4Event* event) override {
    G4Random::setTheSeed(g_opt.seed_base + event->GetEventID());
    gun_.GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun gun_;
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return (g_opt.mode == "ladder" && t->GetParentID() > 0) ? fKill : fUrgent;
  }
};

class Event : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event*) override { Ev() = EventState(); }
  void EndOfEventAction(const G4Event* event) override {
    EventState& e = Ev();
    if (g_opt.mode == "ladder") {
      Tally& t = Ta();
      ++t.events;
      if (e.selector_calls == 1) {
        ++t.selected[{e.sel_z, e.sel_a}];
        ++t.onward[{e.onward_z, e.onward_a}];
        if (e.sel_z != e.onward_z || e.sel_a != e.onward_a) ++t.onward_mismatch;
      } else {
        ++t.onward_mismatch;
      }
    } else if (g_opt.mode == "identity") {
      Fold(EngineHash());
      e.lines.push_back("R " + std::to_string(event->GetEventID()) + " " + std::to_string(e.fold));
      std::lock_guard<std::mutex> lock(g_mutex);
      g_trace[event->GetEventID()] = e.lines;
    }
  }
};

class Run : public G4UserRunAction {
 public:
  void EndOfRunAction(const G4Run*) override {
    if (t_tally == nullptr) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    Tally& g = g_tally[G4Threading::G4GetThreadId()];
    for (const auto& kv : t_tally->selected) g.selected[kv.first] += kv.second;
    for (const auto& kv : t_tally->onward) g.onward[kv.first] += kv.second;
    g.calls += t_tally->calls;
    g.draw_mismatch += t_tally->draw_mismatch;
    g.onward_mismatch += t_tally->onward_mismatch;
    g.events += t_tally->events;
    *t_tally = Tally();
  }
};

class Tracking : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* t) override {
    if (g_opt.mode != "identity") return;
    Fold(EngineHash());
    const auto* c = t->GetCreatorProcess();
    Ev().lines.push_back("T " + std::to_string(t->GetTrackID()) + " " + std::to_string(t->GetParentID()) + " " +
                         std::to_string(t->GetParticleDefinition()->GetPDGEncoding()) + " " +
                         (c ? std::string(c->GetProcessName()) : std::string("-")) + " " + Hex(t->GetKineticEnergy()) + " " +
                         Hex(t->GetGlobalTime()) + " " + std::to_string(Ev().fold));
  }
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const auto* track = step->GetTrack();
    const auto* sec = step->GetSecondaryInCurrentStep();
    if (g_opt.mode == "ladder" && g_opt.route == "atomic" && track->GetParticleDefinition() == G4MuonMinus::MuonMinus()) {
      for (const auto* s : *sec)
        if (const auto* atom = dynamic_cast<const G4MuonicAtom*>(s->GetParticleDefinition())) {
          Ev().onward_z = atom->GetBaseIon()->GetAtomicNumber();
          Ev().onward_a = atom->GetBaseIon()->GetAtomicMass();
        }
    }
    if (g_opt.mode != "identity") return;
    Fold(EngineHash());
    const auto* post = step->GetPostStepPoint();
    const auto* p = post->GetProcessDefinedStep();
    std::string line = "S " + std::to_string(track->GetTrackID()) + " " + std::to_string(track->GetCurrentStepNumber()) +
                       " " + (p ? std::string(p->GetProcessName()) : std::string("-")) + " " + Hex(post->GetKineticEnergy()) + " " +
                       Hex(post->GetGlobalTime()) + " " + Hex(post->GetPosition().x()) + " " +
                       std::to_string(sec->size()) + " " + std::to_string(Ev().fold);
    for (const auto* s : *sec)
      line += " " + std::to_string(s->GetParticleDefinition()->GetPDGEncoding()) + ":" + Hex(s->GetKineticEnergy());
    Ev().lines.push_back(line);
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void BuildForMaster() const override { SetUserAction(new Run); }
  void Build() const override {
    SetUserAction(new Primary);
    SetUserAction(new Event);
    SetUserAction(new Run);
    SetUserAction(new Tracking);
    SetUserAction(new Stepping);
    SetUserAction(new Stacking);
  }
};

class Recorder : public G4VExceptionHandler {
 public:
  G4bool Notify(const char* origin, const char* code, G4ExceptionSeverity, const char* text) override {
    std::lock_guard<std::mutex> lock(g_mutex);
    codes.push_back(code);
    std::printf("EXCEPTION %s %s %s\n", origin, code, text);
    return false;  // record and continue: the drills observe the refusal instead of aborting
  }
  std::vector<std::string> codes;
};

std::string Drills() {
  std::string d;
#ifdef G4MUONIC_DRILL_ENABLED_DELEGATES
  d += " enabled_delegates";
#endif
#ifdef G4MUONIC_DRILL_EXTRA_ENABLED_DRAW
  d += " extra_enabled_draw";
#endif
#ifdef G4MUONIC_DRILL_FORCE_ACTIVE
  d += " force_active";
#endif
#ifdef G4MUONIC_DRILL_MASTER_ONLY
  d += " master_only";
#endif
#ifdef G4MUONIC_DRILL_POSITIONAL
  d += " positional";
#endif
#ifdef G4MUONIC_DRILL_INSTALLER_DELETES
  d += " installer_deletes";
#endif
  return d.empty() ? " none" : d;
}

void WriteConfig(std::ostream& f) {
  f << "VERSION " << G4VERSION_TAG << "\nVERSION_NUMBER " << G4VERSION_NUMBER << "\nMODE " << g_opt.mode
    << "\nROUTE " << g_opt.route << "\nMANAGER " << g_opt.manager << "\nTHREADS " << g_opt.threads << "\nMATERIAL "
    << g_opt.material << "\nENABLE " << g_opt.enable << "\nDATA " << g_opt.data << "\nPROFILE " << g_opt.profile
    << "\nDRAWS " << g_opt.draws << "\nEVENTS " << g_opt.events << "\nSEED_BASE " << g_opt.seed_base << "\nDRILLS"
    << Drills()
#ifdef G4MUONIC_ATOMIC_SETTER_SHIM
    << "\nATOMIC_SETTER shim"
#else
    << "\nATOMIC_SETTER upstream"
#endif
    << '\n';
}

// Midpoint-ladder counts of the stock cumulative search over `weights`, by bisection on each boundary.
std::vector<long> LadderCounts(const std::vector<double>& weights, long draws) {
  std::vector<double> cum;
  double sum = 0.;
  for (double w : weights) cum.push_back(sum += w);
  auto pick = [&](double u) {
    const double x = cum.back() * u;
    std::size_t i = 0;
    for (; i < cum.size(); ++i)
      if (x <= cum[i]) break;
    return std::min(i, cum.size() - 1);
  };
  std::vector<long> bounds = {0};
  for (std::size_t idx = 0; idx + 1 < weights.size(); ++idx) {
    long lo = 0, hi = draws;
    while (lo < hi) {
      const long mid = lo + (hi - lo) / 2;
      if (pick((static_cast<double>(mid) + 0.5) / static_cast<double>(draws)) <= idx) lo = mid + 1;
      else hi = mid;
    }
    bounds.push_back(lo);
  }
  bounds.push_back(draws);
  std::vector<long> out;
  for (std::size_t i = 0; i + 1 < bounds.size(); ++i) out.push_back(bounds[i + 1] - bounds[i]);
  return out;
}

int Probe(const G4MuonicMaterialCatalogue& cat) {
  std::ofstream f(g_opt.out + "/probe.txt");
  WriteConfig(f);
  auto* rec = new Recorder;
  G4StateManager::GetStateManager()->SetExceptionHandler(rec);
  for (const auto& e : cat.Entries()) {
    std::vector<std::pair<int, int>> counts;
    for (std::size_t i = 0; i < e.z.size(); ++i) counts.emplace_back(e.z[i], e.n[i]);
    for (const bool descending : {false, true}) {
      auto* mat = ByCounts("probe_" + e.formula + (descending ? "_desc" : "_asc"), counts, descending);
      G4MuonicMaterialBindings b(&cat);
      const bool ok = b.Bind(mat, e.profile, e.formula, true);
      b.Freeze();
      G4MuonicMaterialSelector sel(&b, true);
      const auto cum = sel.Cumulative(mat);
      f << "ENTRY " << e.profile << ' ' << (descending ? "desc" : "asc") << ' ' << (ok ? "BOUND" : "BIND_REFUSED");
      if (ok) {
        f << " tolerance " << Hex(b.Find(mat)->tolerance) << " counts";
        for (long c : LadderCounts(b.Find(mat)->weights, g_opt.draws)) f << ' ' << c;
      }
      f << '\n';
      const G4double* d = mat->GetAtomicNumDensityVector();
      for (std::size_t i = 0; ok && i < mat->GetNumberOfElements(); ++i)
        f << "P " << e.profile << ' ' << (descending ? "desc" : "asc") << ' ' << i << ' '
          << mat->GetElement(static_cast<G4int>(i))->GetZasInt() << ' ' << Hex(d[i]) << ' '
          << Hex(b.Find(mat)->weights[i]) << ' ' << Hex(cum[i]) << '\n';
    }
  }
  const std::vector<std::pair<std::string, std::string>> nist = {
      {"G4_ALUMINUM_OXIDE", "Al2O3"}, {"G4_CALCIUM_OXIDE", "CaO"},    {"G4_FERRIC_OXIDE", "Fe2O3"},
      {"G4_LEAD_OXIDE", "PbO"},       {"G4_MAGNESIUM_OXIDE", "MgO"}, {"G4_SILICON_DIOXIDE", "SiO2"},
      {"G4_TITANIUM_DIOXIDE", "TiO2"}};
  for (const auto& nf : nist) {
    auto* mat = G4NistManager::Instance()->FindOrBuildMaterial(nf.first);
    std::string profile = "stanislaus1987-" + nf.second;
    std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) { return std::tolower(c); });
    G4MuonicMaterialBindings b(&cat);
    rec->codes.clear();
    const bool ok = b.Bind(mat, profile, nf.second, true);
    f << "NIST " << nf.first << ' ' << profile << ' ' << (ok ? "BOUND" : "BIND_REFUSED") << ' '
      << (rec->codes.empty() ? "-" : rec->codes.back()) << " tolerance "
      << Hex(G4MuonicMaterialBindings::RatioTolerance(mat, nf.second));
    const G4double* d = mat->GetAtomicNumDensityVector();
    for (std::size_t i = 0; i < mat->GetNumberOfElements(); ++i)
      f << ' ' << mat->GetElement(static_cast<G4int>(i))->GetZasInt() << ':' << Hex(d[i]);
    f << '\n';
  }
  // the isotopes of the ladder materials' elements, for the stock isotope draw at u = 1/2
  for (const int z : {8, 13, 82}) {
    const G4Element* e = Nist(z);
    const G4double* ab = e->GetRelativeAbundanceVector();
    f << "ISOTOPES " << z;
    for (std::size_t k = 0; k < e->GetNumberOfIsotopes(); ++k)
      f << ' ' << e->GetIsotope(static_cast<G4int>(k))->GetN() << ':' << Hex(ab[k]);
    f << '\n';
  }
  return 0;
}

int Bind(const G4MuonicMaterialCatalogue& cat) {
  auto* rec = new Recorder;
  G4StateManager::GetStateManager()->SetExceptionHandler(rec);
  std::ofstream f(g_opt.out + "/bind.txt");
  WriteConfig(f);
  auto* al2o3 = BuildMaterial("al2o3");
  auto* al2o3_rev = BuildMaterial("al2o3_rev");
  auto* pbo = BuildMaterial("pbo_nist");
  auto* al = ByCounts("bind_Al", {{13, 1}}, false);                               // short: oxygen missing
  auto* al2o3h = ByCounts("bind_Al2O3H", {{13, 2}, {8, 3}, {1, 1}}, false);       // long: hydrogen added
  auto* alo = ByCounts("bind_AlO", {{13, 1}, {8, 1}}, false);                     // the wrong atom ratio
  auto* weight = new G4Material("bind_Al2O3_by_weight", 3.97 * g / cm3, 2, kStateSolid);  // a weight-fraction
  weight->AddElement(Nist(13), 0.529251);                                          // Al2O3 not on the policy
  weight->AddElement(Nist(8), 0.470749);
  struct Case { std::string name, profile, formula; G4Material* m; bool ack; };
  const std::vector<Case> cases = {
      {"control", "stanislaus1987-al2o3", "Al2O3", al2o3, true},
      {"control_reversed_order", "stanislaus1987-al2o3", "Al2O3", al2o3_rev, true},
      {"control_nist_lead_oxide", "stanislaus1987-pbo", "PbO", pbo, true},
      {"short_vector", "stanislaus1987-al2o3", "Al2O3", al, true},
      {"long_vector", "stanislaus1987-al2o3", "Al2O3", al2o3h, true},
      {"mismatched_formula", "stanislaus1987-al2o3", "AlO", al2o3, true},
      {"wrong_atom_ratio", "stanislaus1987-al2o3", "Al2O3", alo, true},
      {"weight_fraction_not_on_policy", "stanislaus1987-al2o3", "Al2O3", weight, true},
      {"nist_on_another_entry", "stanislaus1987-pbo2", "PbO2", pbo, true},
      {"not_acknowledged", "stanislaus1987-al2o3", "Al2O3", al2o3, false},
      {"unknown_profile", "stanislaus1987-xx", "Al2O3", al2o3, true},
  };
  for (const auto& c : cases) {
    G4MuonicMaterialBindings b(&cat);
    rec->codes.clear();
    const bool ok = b.Bind(c.m, c.profile, c.formula, c.ack);
    f << "CASE " << c.name << " result=" << (ok ? "BOUND" : "BIND_REFUSED")
      << " exception=" << (rec->codes.empty() ? "-" : rec->codes.back()) << " bound_count=" << b.Size() << '\n';
  }
  {
    G4MuonicMaterialBindings b(&cat);
    b.Freeze();
    rec->codes.clear();
    const bool ok = b.Bind(al2o3, "stanislaus1987-al2o3", "Al2O3", true);
    f << "CASE frozen result=" << (ok ? "BOUND" : "BIND_REFUSED")
      << " exception=" << (rec->codes.empty() ? "-" : rec->codes.back()) << " bound_count=" << b.Size() << '\n';
  }
  return 0;
}

G4VModularPhysicsList* Physics(G4MuonicMaterialBindings* bindings) {
  if (g_opt.mode == "order") {
    auto* list = new G4VModularPhysicsList;
    list->RegisterPhysics(new G4EmStandardPhysics);
    list->RegisterPhysics(new G4DecayPhysics);
    auto* provider = new G4MuonicMaterialSelectorPhysics(bindings, true);
    if (g_opt.order == "before") list->RegisterPhysics(provider);
    list->RegisterPhysics(new G4StoppingPhysics);
    if (g_opt.order == "after") list->RegisterPhysics(provider);
    return list;
  }
  auto* physics = new QBBC;
  if (g_opt.route == "atomic") {
    physics->ReplacePhysics(new G4StoppingPhysics("stopping", 1, false));
    physics->RegisterPhysics(new AtomicCapturePhysics);
    physics->RegisterPhysics(new G4MuonicAtomDecayPhysics);
  }
  if (g_opt.enable != "absent") {
    physics->RegisterPhysics(new HarnessPhysics(bindings, g_opt.enable == "on" || g_opt.enable == "on_unbound",
                                                g_opt.mode == "ladder"));
  } else if (g_opt.mode == "ladder") {
    physics->RegisterPhysics(new StockLadderPhysics);
  }
  return physics;
}

void Precreate(const G4Material* m) {
  if (g_opt.route != "atomic") return;
  for (std::size_t i = 0; i < m->GetNumberOfElements(); ++i) {
    const G4Element* e = m->GetElement(static_cast<G4int>(i));
    for (std::size_t k = 0; k < e->GetNumberOfIsotopes(); ++k)
      G4IonTable::GetIonTable()->GetMuonicAtom(e->GetZasInt(), e->GetIsotope(static_cast<G4int>(k))->GetN());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    g_opt = Parse(argc, argv);
    std::filesystem::create_directories(g_opt.out);
    G4Random::setTheEngine(new CLHEP::MixMaxRng);
    G4MuonicMaterialCatalogue cat;
    const bool needs_catalogue = g_opt.mode == "probe" || g_opt.mode == "bind" || g_opt.mode == "order" ||
                                 g_opt.enable == "on" || g_opt.enable == "off_bound" || g_opt.enable == "on_unbound";
    if (needs_catalogue) cat = G4MuonicMaterialCatalogue::Load(g_opt.catalogue);
    if (g_opt.mode == "probe") return Probe(cat);
    if (g_opt.mode == "bind") return Bind(cat);
    Recorder* rec = nullptr;
    if (g_opt.mode == "order") {
      rec = new Recorder;
      G4StateManager::GetStateManager()->SetExceptionHandler(rec);
    }
    auto* manager = G4RunManagerFactory::CreateRunManager(
        (g_opt.manager == "serial" || g_opt.mode == "order") ? G4RunManagerType::SerialOnly
                                                              : G4RunManagerType::TaskingOnly);
    if (g_opt.manager == "tasking" && g_opt.mode != "order") manager->SetNumberOfThreads(g_opt.threads);
#ifdef G4MUONIC_HARNESS_DATA_SWITCH
    if (g_opt.data == "on") G4HadronicParameters::Instance()->SetEnableMuonicData(true);
#else
    if (g_opt.data == "on") throw std::runtime_error("--data on needs an installation with the data switch");
#endif
    manager->SetUserInitialization(new Detector);
    G4MuonicMaterialBindings* bindings = nullptr;
    if (needs_catalogue) {
      bindings = new G4MuonicMaterialBindings(&cat);
      const bool bind_now = g_opt.mode == "order" || g_opt.enable == "on" || g_opt.enable == "off_bound";
      if (bind_now && !bindings->Bind(BuildMaterial(g_opt.material), ProfileOf(g_opt.material),
                                      FormulaOf(g_opt.material), true))
        throw std::runtime_error("binding refused");
      bindings->Freeze();
    }
    manager->SetUserInitialization(Physics(bindings));
    manager->SetUserInitialization(new Actions);
    manager->Initialize();
    std::ofstream f(g_opt.out + "/" + g_opt.mode + ".txt");
    WriteConfig(f);
    if (g_opt.mode == "order") {
      f << "INSTALL_ORDER " << g_opt.order << " exceptions";
      for (const auto& c : rec->codes) f << ' ' << c;
      f << (rec->codes.empty() ? " none" : "") << '\n';
      return 0;
    }
    Precreate(BuildMaterial(g_opt.material));
    const int runs = g_opt.mode == "teardown" ? 3 : 1;
    for (int r = 0; r < runs; ++r)
      manager->BeamOn(static_cast<G4int>(g_opt.mode == "ladder" ? g_opt.draws : g_opt.events));
    f << "RUN_MANAGER " << (manager->GetRunManagerType() == G4RunManager::sequentialRM ? "serial" : "worker-based")
      << '\n';
    if (g_opt.mode == "ladder") {
      const G4Material* lm = BuildMaterial(g_opt.material);
      const G4double* ld = lm->GetAtomicNumDensityVector();
      for (std::size_t i = 0; i < lm->GetNumberOfElements(); ++i)
        f << "ELEMENT " << i << ' ' << lm->GetElement(static_cast<G4int>(i))->GetZasInt() << ' ' << Hex(ld[i])
          << '\n';
      for (const auto& w : g_tally) {
        f << "WORKER " << w.first << " events " << w.second.events << " calls " << w.second.calls << " draw_mismatch "
          << w.second.draw_mismatch << " onward_mismatch " << w.second.onward_mismatch << '\n';
        for (const auto& kv : w.second.selected)
          f << "SELECTED " << w.first << ' ' << kv.first.first << ' ' << kv.first.second << ' ' << kv.second << '\n';
        for (const auto& kv : w.second.onward)
          f << "ONWARD " << w.first << ' ' << kv.first.first << ' ' << kv.first.second << ' ' << kv.second << '\n';
      }
    } else if (g_opt.mode == "identity") {
      for (const auto& ev : g_trace)
        for (const auto& line : ev.second) f << ev.first << ' ' << line << '\n';
    }
    const long before = G4MuonicMaterialSelector::Constructed() - G4MuonicMaterialSelector::Destroyed();
    delete manager;
    f << "SELECTORS constructed " << G4MuonicMaterialSelector::Constructed() << " destroyed "
      << G4MuonicMaterialSelector::Destroyed() << " live_before_teardown " << before << '\n';
    f.close();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "harvest_material: %s\n", e.what());
    return 2;
  }
}
