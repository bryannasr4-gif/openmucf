// harvest_mixture.cc -- what a natural-composition target gives a negative muon stopped at rest in it, read at
// the Geant4 call sites and at the transported output.
//
// One single-threaded process transports negative muons that start at rest at t = 0 at the centre of a box of
// the NIST material G4_<symbol> (one element, the natural composition G4NistManager builds for it), on one of
// two routes:
//
//   stopping  QBBC as built, whose G4MuonMinusCapture runs G4ElementSelector::SelectZandA and then
//             G4MuonMinusBoundDecay::ApplyYourself (rates, delay and branch);
//   atomic    QBBC with G4StoppingPhysics built without negative-muon capture, G4MuonMinusAtomicCapture for
//             the negative muon (its selector draws the nucleus of the muonic atom) and G4MuonicAtomDecay for
//             the muonic atom, registered as G4MuonicAtomDecayPhysics registers it (delay and branch).
//
// With --hooks on, each call site is reached through a class derived from the stock one, which records what
// it is asked and what it returns and otherwise calls the stock method once and returns its result:
//
//   SpySelector          : G4ElementSelector       the (Z, A) the stock SelectZandA set on the nucleus;
//   SpyBoundDecay        : G4MuonMinusBoundDecay   (Z, A), GetMuonCaptureRate(Z, A) and
//                                                  GetMuonDecayRate(Z, A, muMass, AtomicMass(A, Z)) with the
//                                                  arguments ApplyYourself passes them, the projectile's
//                                                  global time before and after the stock call, the status;
//   SpyMuonicAtomDecay   : G4MuonicAtomDecay       before the stock call: (Z, A) of the atom's nucleus, the
//                                                  atom's lifetime, DIO and NC lifetimes, the interaction-length
//                                                  counter and the delay counter * lifetime formed as the stock
//                                                  DecayIt forms it (the stock DIO branch clears the counter, so
//                                                  it is read first); after it: the branch, DIO when the
//                                                  returned change carries the anti_nu_e of the DIO channel.
//
// G4MuonMinusAtomicCapture declares SetElementSelector inline and no file of either revision defines it; this
// file supplies the definition G4HadronStoppingProcess gives its own setter (replace and delete the old one).
// With --hooks off the stock classes run with no derived class anywhere. Every random number is drawn by stock
// code; the classes between SPY-BEGIN and SPY-END call no random-number function.
//
// Output (--out <dir>): config.txt; stops.txt, one line per event (hexadecimal floats, %a):
//   stopping: <event> <selector Z> <selector A> <Z> <A> <capture rate> <decay rate> <t before> <t after>
//             <status 1=alive 0=stopAndKill> <outcome> <DIO electron time or ->
//   atomic:   <event> <selector Z> <selector A> <Z> <A> <lifetime> <DIO lifetime> <NC lifetime> <counter>
//             <delay> <atom time> <branch D|N> <outcome> <DIO electron time or ->
// where <outcome> D or N and the electron time come from the transported secondaries alone (the stopping
// route's DIO-labelled secondaries; the muonic atom decay's secondaries carrying an anti_nu_e). With --hooks
// off the call-site fields are '-'. With --identity on, records.txt also carries, per event, the transport
// lines of g4muonic_transport.cc (T, S, C) and an R line folding the engine-state hash taken at every track
// start, every step end and the event end; summary.txt gives the hook and base call counts and, inside each
// hook, whether the engine state at hook entry equals the state at base entry and the state at base return
// equals the state at hook exit.
#include "G4Box.hh"
#include "G4DecayTable.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4ElementSelector.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4GenericMuonicAtom.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadronStoppingProcess.hh"
#include "G4Isotope.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusAtomicCapture.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtom.hh"
#include "G4MuonicAtomDecay.hh"
#include "G4MuonicAtomDecayPhysics.hh"
#include "G4NistManager.hh"
#include "G4Nucleus.hh"
#include "G4ParticleChange.hh"
#include "G4ParticleGun.hh"
#include "G4PhysListUtil.hh"
#include "G4PhysicsListHelper.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4PVPlacement.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "QBBC.hh"
#include "Randomize.hh"

#ifdef HARVEST_MIXTURE_PATCHED
#include "G4HadronicParameters.hh"
#include "G4MuonicDataOverlay.hh"
#endif

#include <CLHEP/Random/MixMaxRng.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string route;
  std::string symbol;
  std::string out;
  std::string opt_in = "off";
  int z = 0;
  int events = 0;
  long seed_base = 0;
  bool hooks = false;
  bool identity = false;
  int drill_a = 0;
};

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value(argv[++i]);
    if (key == "--route") o.route = value;
    else if (key == "--symbol") o.symbol = value;
    else if (key == "--Z") o.z = std::stoi(value);
    else if (key == "--events") o.events = std::stoi(value);
    else if (key == "--seed-base") o.seed_base = std::stol(value);
    else if (key == "--out") o.out = value;
    else if (key == "--opt-in") o.opt_in = value;
    else if (key == "--hooks") o.hooks = value == "on";
    else if (key == "--identity") o.identity = value == "on";
    else if (key == "--drill-a") o.drill_a = std::stoi(value);
    else throw std::runtime_error("unknown argument " + key);
  }
  if ((o.route != "stopping" && o.route != "atomic") || o.symbol.empty() || o.z <= 0 || o.events <= 0 ||
      o.seed_base <= 0 || o.out.empty() || (o.opt_in != "on" && o.opt_in != "off")) {
    throw std::runtime_error("invalid route, element, event count, seed, output or opt-in");
  }
#ifndef HARVEST_MIXTURE_PATCHED
  if (o.opt_in == "on") throw std::runtime_error("opt-in requires a patched build");
#endif
  return o;
}

std::string Hex(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%a", value);
  return buf;
}

// FNV-1a over the words of the engine's state; reading the state draws no random number.
std::uint64_t EngineHash() {
  const std::vector<unsigned long> words = G4Random::getTheEngine()->put();
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned long w : words) {
    for (int b = 0; b < 8; ++b) {
      h ^= static_cast<std::uint64_t>((w >> (8 * b)) & 0xffUL);
      h *= 1099511628211ULL;
    }
  }
  return h;
}

// What the call sites and the transported output recorded for the current event.
struct EventRecord {
  int selector_calls = 0, decay_calls = 0, dio_electrons = 0;
  int sel_z = 0, sel_a = 0, z = 0, a = 0;
  double rate_c = 0., rate_d = 0., t_before = 0., t_after = 0.;
  int status = -1;
  double tau = 0., tau_dio = 0., tau_nc = 0., counter = 0., delay = 0., atom_time = 0.;
  char branch = '-';
  char outcome = '-';
  double electron_time = 0.;
  std::uint64_t rng = 1469598103934665603ULL;
  long rng_points = 0;
};

struct Counters {
  long hook_calls = 0, base_calls = 0, entry_mismatch = 0, exit_mismatch = 0, returned_other = 0;
};

EventRecord g_event;
Counters g_count;
bool g_identity = false;
int g_drill_a = 0;

void Fold(std::uint64_t h) {
  for (int b = 0; b < 8; ++b) {
    g_event.rng ^= (h >> (8 * b)) & 0xffULL;
    g_event.rng *= 1099511628211ULL;
  }
  ++g_event.rng_points;
}

struct HookFrame {
  std::uint64_t entry = 0;
  void Enter() {
    ++g_count.hook_calls;
    if (g_identity) entry = EngineHash();
  }
  void BeforeBase() {
    ++g_count.base_calls;
    if (g_identity && EngineHash() != entry) ++g_count.entry_mismatch;
  }
  std::uint64_t AfterBase() const { return g_identity ? EngineHash() : 0; }
  void Exit(std::uint64_t after_base) const {
    if (g_identity && EngineHash() != after_base) ++g_count.exit_mismatch;
  }
};

// SPY-BEGIN
class SpySelector : public G4ElementSelector {
 public:
  const G4Element* SelectZandA(const G4Track& track, G4Nucleus* target) override {
    HookFrame frame;
    frame.Enter();
    frame.BeforeBase();
    const G4Element* element = G4ElementSelector::SelectZandA(track, target);
    const std::uint64_t after = frame.AfterBase();
    ++g_event.selector_calls;
    g_event.sel_z = target->GetZ_asInt();
    g_event.sel_a = target->GetA_asInt();
    frame.Exit(after);
    return element;
  }
};

class SpyBoundDecay : public G4MuonMinusBoundDecay {
 public:
  G4HadFinalState* ApplyYourself(const G4HadProjectile& projectile, G4Nucleus& nucleus) override {
    HookFrame frame;
    frame.Enter();
    const int z = nucleus.GetZ_asInt();
    const int a = nucleus.GetA_asInt();
    const double mu_mass = G4MuonMinus::MuonMinus()->GetPDGMass();
    const double rate_c = GetMuonCaptureRate(z, a);
    const double rate_d = GetMuonDecayRate(z, a, mu_mass, nucleus.AtomicMass(a, z));
    const double before = projectile.GetGlobalTime();
#ifdef HARVEST_MIXTURE_DRILL_EARLY_RETURN
    if (a == g_drill_a) {
      drill_result.Clear();
      drill_result.SetStatusChange(isAlive);
      ++g_event.decay_calls;
      return &drill_result;
    }
#endif
    frame.BeforeBase();
    G4HadFinalState* result = G4MuonMinusBoundDecay::ApplyYourself(projectile, nucleus);
    const std::uint64_t after = frame.AfterBase();
#ifdef HARVEST_MIXTURE_DRILL_EXTRA_DRAW
    (void)G4UniformRand();
#endif
    ++g_event.decay_calls;
    g_event.z = z;
    g_event.a = a;
    g_event.rate_c = rate_c;
    g_event.rate_d = rate_d;
    g_event.t_before = before;
    g_event.t_after = projectile.GetGlobalTime();
    g_event.status = result->GetStatusChange() == isAlive ? 1 : 0;
    frame.Exit(after);
    return result;
  }

 private:
  G4HadFinalState drill_result;
};

class SpyMuonicAtomDecay : public G4MuonicAtomDecay {
 public:
  SpyMuonicAtomDecay() : G4MuonicAtomDecay(nullptr) {}
  G4VParticleChange* AtRestDoIt(const G4Track& track, const G4Step& step) override {
    return Spy(track, step, true);
  }
  G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step) override {
    return Spy(track, step, false);
  }

 private:
  G4VParticleChange* Spy(const G4Track& track, const G4Step& step, bool at_rest) {
    HookFrame frame;
    frame.Enter();
    const auto* atom = static_cast<const G4MuonicAtom*>(track.GetDefinition());
    const auto* ion = atom->GetBaseIon();
    const int z = ion->GetAtomicNumber();
    const int a = ion->GetAtomicMass();
    const double tau = GetMeanLifeTime(track, nullptr);
    const double counter = theNumberOfInteractionLengthLeft;
    const double delay = theNumberOfInteractionLengthLeft * tau;
    const double atom_time = track.GetGlobalTime();
#ifdef HARVEST_MIXTURE_DRILL_EARLY_RETURN
    if (a == g_drill_a) {
      drill_change.Initialize(track);
      drill_change.ProposeTrackStatus(fStopAndKill);
      ++g_event.decay_calls;
      return &drill_change;
    }
#endif
    frame.BeforeBase();
    G4VParticleChange* change = at_rest ? G4MuonicAtomDecay::AtRestDoIt(track, step)
                                        : G4MuonicAtomDecay::PostStepDoIt(track, step);
    const std::uint64_t after = frame.AfterBase();
#ifdef HARVEST_MIXTURE_DRILL_EXTRA_DRAW
    (void)G4UniformRand();
#endif
    char branch = 'N';
    for (G4int i = 0; i < change->GetNumberOfSecondaries(); ++i) {
      if (change->GetSecondary(i)->GetDefinition()->GetPDGEncoding() == -12) branch = 'D';
    }
    ++g_event.decay_calls;
    g_event.z = z;
    g_event.a = a;
    g_event.tau = tau;
    g_event.tau_dio = atom->GetDIOLifeTime();
    g_event.tau_nc = atom->GetNCLifeTime();
    g_event.counter = counter;
    g_event.delay = delay;
    g_event.atom_time = atom_time;
    g_event.branch = branch;
    frame.Exit(after);
    return change;
  }

  G4ParticleChange drill_change;
};
// SPY-END

class SpyMuonicAtomDecayPhysics : public G4VPhysicsConstructor {
 public:
  SpyMuonicAtomDecayPhysics() : G4VPhysicsConstructor("G4MuonicAtomDecay") {
    G4PhysListUtil::InitialiseParameters();
    SetVerboseLevel(1);
  }
  void ConstructParticle() override { G4GenericMuonicAtom::GenericMuonicAtom(); }
  void ConstructProcess() override {
    G4PhysicsListHelper* helper = G4PhysicsListHelper::GetPhysicsListHelper();
    const G4int level = helper->GetVerboseLevel();
    if (!helper->RegisterProcess(new SpyMuonicAtomDecay(), G4GenericMuonicAtom::GenericMuonicAtom())) {
      throw std::runtime_error("RegisterProcess failed for G4GenericMuonicAtom");
    }
    helper->SetVerboseLevel(level);
  }
};

class AtomicCapturePhysics : public G4VPhysicsConstructor {
 public:
  explicit AtomicCapturePhysics(bool hooks) : G4VPhysicsConstructor("MuAtomicCapture"), hooks_(hooks) {}
  void ConstructParticle() override { G4MuonMinus::MuonMinus(); }
  void ConstructProcess() override {
    auto* capture = new G4MuonMinusAtomicCapture();
    if (hooks_) capture->SetElementSelector(new SpySelector());
    G4MuonMinus::MuonMinus()->GetProcessManager()->AddRestProcess(capture);
  }

 private:
  bool hooks_;
};

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Options& options) : o(options) {}
  G4VPhysicalVolume* Construct() override {
    material = G4NistManager::Instance()->FindOrBuildMaterial("G4_" + o.symbol);
    if (material == nullptr || material->GetNumberOfElements() != 1 ||
        material->GetElement(0)->GetZasInt() != o.z) {
      throw std::runtime_error("G4_" + o.symbol + " is not a single-element material of Z " + std::to_string(o.z));
    }
    auto* solid = new G4Box("world", 1. * m, 1. * m, 1. * m);
    auto* logical = new G4LogicalVolume(solid, material, "world");
    return new G4PVPlacement(nullptr, G4ThreeVector(), logical, "world", nullptr, false, 0);
  }
  G4Material* material = nullptr;

 private:
  const Options& o;
};

struct Output {
  std::ofstream stops;
  std::ofstream records;
  int dio_model = -1;
};

class Primary : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Primary(const Options& options) : o(options), gun(1) {
    gun.SetParticleDefinition(G4MuonMinus::MuonMinus());
    gun.SetParticleEnergy(0.);
    gun.SetParticlePosition(G4ThreeVector());
    gun.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    gun.SetParticleTime(0.);
  }
  void GeneratePrimaries(G4Event* event) override {
    G4Random::setTheSeed(o.seed_base + event->GetEventID());
    gun.GeneratePrimaryVertex(event);
  }

 private:
  const Options& o;
  G4ParticleGun gun;
};

class Event : public G4UserEventAction {
 public:
  Event(const Options& options, Output& output) : o(options), out(output) {}
  void BeginOfEventAction(const G4Event*) override { g_event = EventRecord(); }
  void EndOfEventAction(const G4Event* event) override {
    const EventRecord& e = g_event;
    if (g_identity) {
      Fold(EngineHash());
      out.records << "R " << event->GetEventID() << ' ' << e.rng_points << ' ' << std::hex << e.rng << std::dec
                  << '\n';
    }
    const bool hooked = o.hooks;
    if (hooked && (e.selector_calls != 1 || e.decay_calls != 1)) {
      throw std::runtime_error("event " + std::to_string(event->GetEventID()) + ": " +
                               std::to_string(e.selector_calls) + " selector and " +
                               std::to_string(e.decay_calls) + " decay calls");
    }
    if (e.dio_electrons > 1 || (e.outcome == 'D') != (e.dio_electrons == 1)) {
      throw std::runtime_error("event " + std::to_string(event->GetEventID()) + ": transported DIO electrons " +
                               std::to_string(e.dio_electrons) + " with outcome " + e.outcome);
    }
    auto& s = out.stops;
    s << event->GetEventID();
    if (hooked) {
      s << ' ' << e.sel_z << ' ' << e.sel_a << ' ' << e.z << ' ' << e.a;
      if (o.route == "stopping") {
        s << ' ' << Hex(e.rate_c) << ' ' << Hex(e.rate_d) << ' ' << Hex(e.t_before) << ' ' << Hex(e.t_after)
          << ' ' << e.status;
      } else {
        s << ' ' << Hex(e.tau) << ' ' << Hex(e.tau_dio) << ' ' << Hex(e.tau_nc) << ' ' << Hex(e.counter) << ' '
          << Hex(e.delay) << ' ' << Hex(e.atom_time) << ' ' << e.branch;
      }
    } else {
      const int fields = o.route == "stopping" ? 9 : 11;
      for (int i = 0; i < fields; ++i) s << " -";
    }
    s << ' ' << (e.outcome == '-' ? 'N' : e.outcome) << ' '
      << (e.dio_electrons == 1 ? Hex(e.electron_time) : std::string("-")) << '\n';
  }

 private:
  const Options& o;
  Output& out;
};

class Tracking : public G4UserTrackingAction {
 public:
  explicit Tracking(Output& output) : out(output) {}
  void PreUserTrackingAction(const G4Track* track) override {
    if (!g_identity) return;
    Fold(EngineHash());
    const auto* creator = track->GetCreatorProcess();
    const int model = track->GetCreatorModelID();
    out.records << "T " << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() << ' '
                << track->GetTrackID() << ' ' << track->GetParentID() << ' '
                << track->GetParticleDefinition()->GetPDGEncoding() << ' '
                << (creator == nullptr ? "-" : creator->GetProcessName()) << ' '
                << (model < 0 ? std::string("-") : std::string(G4PhysicsModelCatalog::GetModelNameFromID(model)))
                << ' ' << Hex(track->GetKineticEnergy()) << ' ' << Hex(track->GetGlobalTime()) << '\n';
  }

 private:
  Output& out;
};

class Stepping : public G4UserSteppingAction {
 public:
  Stepping(const Options& options, Output& output) : o(options), out(output) {}
  void UserSteppingAction(const G4Step* step) override {
    if (g_identity) Fold(EngineHash());
    const auto* track = step->GetTrack();
    const auto* definition = track->GetParticleDefinition();
    const bool muon = definition == G4MuonMinus::MuonMinus();
    const bool atom = dynamic_cast<const G4MuonicAtom*>(definition) != nullptr;
    if (!muon && !atom) return;
    const auto* secondaries = step->GetSecondaryInCurrentStep();
    // the transported outcome of the stop, from the secondaries alone
    if (o.route == "stopping" && muon) {
      for (const auto* secondary : *secondaries) {
        if (secondary->GetCreatorModelID() != out.dio_model) continue;
        g_event.outcome = 'D';
        if (secondary->GetParticleDefinition()->GetPDGEncoding() == 11) {
          ++g_event.dio_electrons;
          g_event.electron_time = secondary->GetGlobalTime();
        }
      }
    } else if (o.route == "atomic" && atom && !secondaries->empty()) {
      bool dio = false;
      for (const auto* secondary : *secondaries) {
        if (secondary->GetParticleDefinition()->GetPDGEncoding() == -12) dio = true;
      }
      g_event.outcome = dio ? 'D' : 'N';
      if (dio) {
        for (const auto* secondary : *secondaries) {
          if (secondary->GetParticleDefinition()->GetPDGEncoding() != 11) continue;
          ++g_event.dio_electrons;
          g_event.electron_time = secondary->GetGlobalTime();
        }
      }
    }
    if (!g_identity) return;
    const auto* post = step->GetPostStepPoint();
    const auto* process = post->GetProcessDefinedStep();
    const int event = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    out.records << "S " << event << ' ' << track->GetTrackID() << ' ' << track->GetCurrentStepNumber() << ' '
                << (process == nullptr ? "-" : process->GetProcessName()) << ' ' << Hex(post->GetKineticEnergy())
                << ' ' << Hex(post->GetGlobalTime()) << ' ' << secondaries->size() << '\n';
    int k = 0;
    for (const auto* secondary : *secondaries) {
      const int model = secondary->GetCreatorModelID();
      out.records << "C " << event << ' ' << track->GetTrackID() << ' ' << k++ << ' '
                  << secondary->GetParticleDefinition()->GetPDGEncoding() << ' '
                  << (model < 0 ? std::string("-") : std::string(G4PhysicsModelCatalog::GetModelNameFromID(model)))
                  << ' ' << Hex(secondary->GetKineticEnergy()) << ' ' << Hex(secondary->GetGlobalTime()) << '\n';
    }
  }

 private:
  const Options& o;
  Output& out;
};

class Actions : public G4VUserActionInitialization {
 public:
  Actions(const Options& options, Output& output) : o(options), out(output) {}
  void Build() const override {
    SetUserAction(new Primary(o));
    SetUserAction(new Event(o, out));
    SetUserAction(new Tracking(out));
    SetUserAction(new Stepping(o, out));
  }

 private:
  const Options& o;
  Output& out;
};

// Install the stopping route's hooks on the negative muon's G4MuonMinusCapture.
void HookStoppingRoute() {
  auto* processes = G4MuonMinus::MuonMinus()->GetProcessManager()->GetProcessList();
  int hooked = 0;
  for (std::size_t i = 0; i < processes->size(); ++i) {
    auto* stopping = dynamic_cast<G4HadronStoppingProcess*>((*processes)[i]);
    if (stopping == nullptr || stopping->GetProcessName() != "muMinusCaptureAtRest") continue;
    stopping->SetElementSelector(new SpySelector());
    stopping->SetBoundDecay(new SpyBoundDecay());
    ++hooked;
  }
  if (hooked != 1) throw std::runtime_error("expected one muMinusCaptureAtRest process, found " + std::to_string(hooked));
}

void WriteConfig(const Options& o, Detector* detector) {
  std::ofstream file(o.out + "/config.txt");
  if (!file) throw std::runtime_error("cannot open config.txt");
  file << "VERSION " << G4VERSION_TAG << '\n';
  file << "ENGINE MixMaxRng\n";
  file << "RUN_MANAGER serial\n";
  file << "ROUTE " << o.route << '\n';
  file << "HOOKS " << (o.hooks ? "on" : "off") << '\n';
  file << "IDENTITY " << (o.identity ? "on" : "off") << '\n';
  file << "OPT_IN " << o.opt_in << '\n';
  file << "SEED_BASE " << o.seed_base << '\n';
  file << "EVENTS " << o.events << '\n';
#if defined(HARVEST_MIXTURE_DRILL_EXTRA_DRAW)
  file << "DRILL extra_draw\n";
#elif defined(HARVEST_MIXTURE_DRILL_EARLY_RETURN)
  file << "DRILL early_return " << o.drill_a << '\n';
#else
  file << "DRILL none\n";
#endif
  auto* processes = G4MuonMinus::MuonMinus()->GetProcessManager()->GetProcessList();
  file << "PROCESSES";
  for (std::size_t i = 0; i < processes->size(); ++i) file << ' ' << (*processes)[i]->GetProcessName();
  file << '\n';
  auto* material = detector->material;
  const auto* element = material->GetElement(0);
  file << "MATERIAL " << material->GetName() << ' ' << element->GetZasInt() << ' ' << element->GetNumberOfIsotopes();
  for (std::size_t i = 0; i < element->GetNumberOfIsotopes(); ++i) {
    file << ' ' << element->GetIsotope(static_cast<G4int>(i))->GetN() << ' ' << Hex(element->GetRelativeAbundanceVector()[i]);
  }
  file << '\n';
#ifdef HARVEST_MIXTURE_PATCHED
  const auto& config = G4MuonicDataOverlay::Config();
  file << "CONFIG " << config.d1Profile << ' ' << config.d3Profile << ' '
       << (config.datasetDirectory.empty() ? "none" : "set") << ' '
       << (config.datasetVersion.empty() ? "none" : config.datasetVersion) << '\n';
#else
  file << "CONFIG pristine\n";
#endif
  file << "SUMMARY hook_calls " << g_count.hook_calls << " base_calls " << g_count.base_calls << " entry_mismatch "
       << g_count.entry_mismatch << " exit_mismatch " << g_count.exit_mismatch << '\n';
}

}  // namespace

// Declared inline in G4MuonMinusAtomicCapture.hh and defined by no file of either revision; this is the
// definition G4HadronStoppingProcess gives its own SetElementSelector.
inline void G4MuonMinusAtomicCapture::SetElementSelector(G4ElementSelector* ptr) {
  if (fElementSelector != ptr) {
    delete fElementSelector;
    fElementSelector = ptr;
  }
}

int main(int argc, char** argv) {
  try {
    const Options options = Parse(argc, argv);
    g_identity = options.identity;
    g_drill_a = options.drill_a;
    std::filesystem::create_directories(options.out);
    Output output;
    output.stops.open(options.out + "/stops.txt");
    if (options.identity) output.records.open(options.out + "/records.txt");
    if (!output.stops || (options.identity && !output.records)) throw std::runtime_error("cannot open outputs");
    G4Random::setTheEngine(new CLHEP::MixMaxRng);
    auto* manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
#ifdef HARVEST_MIXTURE_PATCHED
    if (options.opt_in == "on") G4HadronicParameters::Instance()->SetEnableMuonicData(true);
#endif
    auto* detector = new Detector(options);
    manager->SetUserInitialization(detector);
    auto* physics = new QBBC;
    if (options.route == "atomic") {
      physics->ReplacePhysics(new G4StoppingPhysics("stopping", 1, false));
      physics->RegisterPhysics(new AtomicCapturePhysics(options.hooks));
      if (options.hooks) physics->RegisterPhysics(new SpyMuonicAtomDecayPhysics());
      else physics->RegisterPhysics(new G4MuonicAtomDecayPhysics());
    }
    manager->SetUserInitialization(physics);
    manager->SetUserInitialization(new Actions(options, output));
    manager->Initialize();
    output.dio_model = G4PhysicsModelCatalog::GetModelID("model_muMinusCaptureAtRest_DIO");
    if (options.route == "stopping" && output.dio_model < 0) throw std::runtime_error("no DIO model id");
    if (options.route == "stopping" && options.hooks) HookStoppingRoute();
    manager->BeamOn(options.events);
    output.stops.close();
    if (options.identity) {
      std::ofstream summary(options.out + "/summary.txt");
      summary << "hook_calls " << g_count.hook_calls << "\nbase_calls " << g_count.base_calls << "\nentry_mismatch "
              << g_count.entry_mismatch << "\nexit_mismatch " << g_count.exit_mismatch << '\n';
    }
    WriteConfig(options, detector);
    delete manager;
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "harvest_mixture: %s\n", error.what());
    return 2;
  }
}
