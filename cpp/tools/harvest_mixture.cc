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
//
// Options added for single-isotope targets, nonzero start times and the bound-rate provider; each defaults to the
// behaviour above, and with every one at its default the outputs above are unchanged and no other file is written:
//
//   --isotope <A>          a box of one isotope of the element (a G4Element of that one G4Isotope, at the density of
//                          G4_<symbol>) instead of the natural composition; 0, the default, keeps G4_<symbol>
//   --t0 <ns>              the muon's global start time (default 0)
//   --event-offset <i0>    event k of this process is event i0 + k: seeded with seed-base + i0 + k and written with
//                          that number (default 0)
//   --bound-rates on|off   G4HadronicParameters::SetEnableMuonicBoundRates(true) before the physics list is built
//                          (only in a build against a library that carries the bound-rate provider; default off)
//   --bound-record on|off  write bound.txt, one line per event (needs --hooks on; default off)
//   --rng-states on|off    write rng.txt: the engine's whole serialized state at the end of every event (default off)
//   --synthetic-capture-scale <s>   trace builds only: multiply the capture rate the provider returns by s before
//                          the consumer receives it -- a synthetic fixture that shows that the returned pair reaches
//                          the consumer, never a physics configuration (default off)
// Any of them away from its default also writes extended.txt, which records them and, in a trace build, where each
// traced function resolves.
//
// bound.txt (hexadecimal floats; `x` where the build does not observe a field):
//   stopping  S <event> <Z> <A> <t before> <t after> <outcome> P <calls> <other calls> <status> <capture> <decay>
//               M <model calls> <k> <model-local time>*k N <n> <transported time>*n E <DIO electron time or ->
//   atomic    A <event> <Z> <A> <atom creation time> <lifetime> <DIO lifetime> <NC lifetime> <counter> <delay>
//               <atom time> <branch> <outcome> M <model calls> <k> <model-local time>*k N <n> <transported time>*n
//               E <DIO electron time or ->
//   provider  C <event> <caller> <Z> <A> <status> <capture> <decay>, before the event line, for each call the
//               provider receives from the muonic-atom helper or from a caller other than the two consumers
// With the bound record on, bound_summary.txt gives the sufficient statistics this process accumulated itself, a
// second path to the ones a summary computes from bound.txt: the event count, the number of events the consumer
// sent to nuclear capture (the stopping route's alive status, the atomic route's branch N), and the compensated
// (Neumaier) sums of the elapsed time T over all events, the capture events and the decay events, where T is
// <t after> - <t before> on the stopping route and <delay> on the atomic route.
// P counts the calls G4MuonMinusBoundDecay::ApplyYourself makes to G4MuonicBoundRateProvider::Rates in the event,
// with the status (0 common, 1 disabled, 2 unsupported, 3 refused) and the pair of the last one. M is the nuclear
// capture model of the route (G4CascadeInterface on the stopping route, G4MuMinusCapturePrecompound on the atomic
// route) called with a negative muon: the number of calls and the local time of each secondary its last call
// returned, read before the caller adds any time. N lists the transported times of the capture products, in the
// order the caller created them. The P, M and C fields come from a trace build (-DHARVEST_MIXTURE_TRACE), which
// defines those three functions in this executable and exports them, so that the Geant4 libraries' own calls bind
// to them; each calls the library's definition (dlsym RTLD_NEXT) once and returns its result unchanged, apart from
// the synthetic fixture. They draw no random number.
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
#ifdef HARVEST_MIXTURE_BOUND_RATES
#include "G4MuonicBoundRateProvider.hh"
#endif
#if defined(HARVEST_MIXTURE_TRACE) && !defined(HARVEST_MIXTURE_BOUND_RATES)
#error "a trace build traces the bound-rate provider: define HARVEST_MIXTURE_BOUND_RATES as well"
#endif
#if defined(HARVEST_MIXTURE_BOUND_RATES) && !defined(HARVEST_MIXTURE_PATCHED)
#error "the bound-rate provider comes with the muonic-data patch: define HARVEST_MIXTURE_PATCHED as well"
#endif
#ifdef HARVEST_MIXTURE_TRACE
#include "G4CascadeInterface.hh"
#include "G4MuMinusCapturePrecompound.hh"
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <CLHEP/Random/MixMaxRng.h>
#include <cmath>
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
  int isotope = 0;
  double t0 = 0.;
  long event_offset = 0;
  bool bound_rates = false;
  bool bound_record = false;
  bool rng_states = false;
  double synthetic_capture_scale = 0.;
  bool Extended() const {
    return isotope != 0 || t0 != 0. || event_offset != 0 || bound_rates || bound_record || rng_states ||
           synthetic_capture_scale != 0.;
  }
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
    else if (key == "--isotope") o.isotope = std::stoi(value);
    else if (key == "--t0") o.t0 = std::stod(value);
    else if (key == "--event-offset") o.event_offset = std::stol(value);
    else if (key == "--bound-rates") o.bound_rates = value == "on";
    else if (key == "--bound-record") o.bound_record = value == "on";
    else if (key == "--rng-states") o.rng_states = value == "on";
    else if (key == "--synthetic-capture-scale") o.synthetic_capture_scale = std::stod(value);
    else throw std::runtime_error("unknown argument " + key);
  }
  if (o.isotope < 0 || (o.isotope > 0 && o.isotope < o.z) || !std::isfinite(o.t0) || o.t0 < 0. ||
      o.event_offset < 0 || (o.bound_record && !o.hooks) || !std::isfinite(o.synthetic_capture_scale) ||
      o.synthetic_capture_scale < 0.) {
    throw std::runtime_error("invalid isotope, start time, event offset, bound record or synthetic scale");
  }
#ifndef HARVEST_MIXTURE_BOUND_RATES
  if (o.bound_rates) throw std::runtime_error("--bound-rates on requires a build with the bound-rate provider");
#endif
#ifndef HARVEST_MIXTURE_TRACE
  if (o.synthetic_capture_scale != 0.) throw std::runtime_error("--synthetic-capture-scale requires a trace build");
#endif
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
  // the bound record
  double atom_create = 0.;
  bool atom_seen = false;
  std::vector<double> products;  // transported times of the capture products, in creation order
  int rate_calls = 0, rate_other = 0, rate_status = -1;
  double pair_c = 0., pair_d = 0.;
  int model_calls = 0;
  std::vector<double> model_times;  // local times of the secondaries of the last model call
  std::vector<std::string> provider_lines;
};

struct Counters {
  long hook_calls = 0, base_calls = 0, entry_mismatch = 0, exit_mismatch = 0, returned_other = 0;
};

// A Neumaier-compensated sum.
struct CompensatedSum {
  double sum = 0., carry = 0.;
  void Add(double x) {
    const double t = sum + x;
    carry += std::fabs(sum) >= std::fabs(x) ? (sum - t) + x : (x - t) + sum;
    sum = t;
  }
  double Value() const { return sum + carry; }
};

struct BoundTotals {
  long events = 0, captures = 0;
  CompensatedSum all, capture, decay;
};

EventRecord g_event;
Counters g_count;
bool g_identity = false;
int g_drill_a = 0;
long g_event_offset = 0;
double g_synthetic_capture_scale = 0.;
long g_current_event = -1;
bool g_bound_record = false;
BoundTotals g_totals;

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

#ifdef HARVEST_MIXTURE_TRACE
// The library definition a traced function forwards to: the next definition after this executable's.
void* NextDefinition(const char* name) {
  void* next = dlsym(RTLD_NEXT, name);
  if (next == nullptr) throw std::runtime_error(std::string("no library definition of ") + name);
  return next;
}

const char* const kRatesSymbol = "_ZN25G4MuonicBoundRateProvider5RatesEiiddR18G4MuonicBoundRates";
const char* const kBertiniSymbol = "_ZN18G4CascadeInterface13ApplyYourselfERK15G4HadProjectileR9G4Nucleus";
const char* const kPrecoSymbol = "_ZN27G4MuMinusCapturePrecompound13ApplyYourselfERK15G4HadProjectileR9G4Nucleus";

// Which consumer called: the nearest exported symbol before the return address.
std::string Caller(void* address) {
  Dl_info info;
  if (dladdr(address, &info) == 0 || info.dli_sname == nullptr) return "unknown";
  const std::string name(info.dli_sname);
  if (name.find("G4MuonMinusBoundDecay") != std::string::npos) return "bound";
  if (name.find("G4MuonicAtomHelper") != std::string::npos) return "helper";
  return "other:" + name;
}

// The model-local times of the secondaries a capture model returned, before any caller adds a time.
void RecordModel(const G4HadProjectile& projectile, const G4HadFinalState* result) {
  if (projectile.GetDefinition() != G4MuonMinus::MuonMinus() || result == nullptr) return;
  ++g_event.model_calls;
  g_event.model_times.clear();
  for (std::size_t i = 0; i < result->GetNumberOfSecondaries(); ++i) {
    g_event.model_times.push_back(result->GetSecondary(static_cast<G4int>(i))->GetTime());
  }
}
#endif
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
    if (o.isotope > 0) {
      const std::string name = o.symbol + std::to_string(o.isotope);
      auto* isotope = new G4Isotope(name, o.z, o.isotope);
      auto* element = new G4Element(name, o.symbol, 1);
      element->AddIsotope(isotope, 1.);
      auto* single = new G4Material("U_" + name, material->GetDensity(), 1, material->GetState(),
                                    material->GetTemperature(), material->GetPressure());
      single->AddElement(element, 1.);
      material = single;
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
  std::ofstream bound;
  std::ofstream rng;
  int dio_model = -1;
  int nc_model = -1;
};

class Primary : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Primary(const Options& options) : o(options), gun(1) {
    gun.SetParticleDefinition(G4MuonMinus::MuonMinus());
    gun.SetParticleEnergy(0.);
    gun.SetParticlePosition(G4ThreeVector());
    gun.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    gun.SetParticleTime(o.t0 * ns);
  }
  void GeneratePrimaries(G4Event* event) override {
    G4Random::setTheSeed(o.seed_base + o.event_offset + event->GetEventID());
    gun.GeneratePrimaryVertex(event);
  }

 private:
  const Options& o;
  G4ParticleGun gun;
};

class Event : public G4UserEventAction {
 public:
  Event(const Options& options, Output& output) : o(options), out(output) {}
  void BeginOfEventAction(const G4Event* event) override {
    g_event = EventRecord();
    g_current_event = o.event_offset + event->GetEventID();
  }
  void EndOfEventAction(const G4Event* event) override {
    const EventRecord& e = g_event;
    const long id = o.event_offset + event->GetEventID();
    if (g_identity) {
      Fold(EngineHash());
      out.records << "R " << id << ' ' << e.rng_points << ' ' << std::hex << e.rng << std::dec << '\n';
    }
    const bool hooked = o.hooks;
    if (hooked && (e.selector_calls != 1 || e.decay_calls != 1)) {
      throw std::runtime_error("event " + std::to_string(id) + ": " +
                               std::to_string(e.selector_calls) + " selector and " +
                               std::to_string(e.decay_calls) + " decay calls");
    }
    if (e.dio_electrons > 1 || (e.outcome == 'D') != (e.dio_electrons == 1)) {
      throw std::runtime_error("event " + std::to_string(id) + ": transported DIO electrons " +
                               std::to_string(e.dio_electrons) + " with outcome " + e.outcome);
    }
    auto& s = out.stops;
    s << id;
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
    if (o.bound_record) WriteBound(id, e);
    if (o.rng_states) {
      out.rng << "E " << id;
      for (unsigned long w : G4Random::getTheEngine()->put()) out.rng << ' ' << w;
      out.rng << '\n';
    }
  }

  void WriteBound(long id, const EventRecord& e) {
    const bool stopping = o.route == "stopping";
    const double elapsed = stopping ? e.t_after - e.t_before : e.delay;
    const bool captured = stopping ? e.status == 1 : e.branch == 'N';
    ++g_totals.events;
    g_totals.all.Add(elapsed);
    if (captured) {
      ++g_totals.captures;
      g_totals.capture.Add(elapsed);
    } else {
      g_totals.decay.Add(elapsed);
    }
    auto& b = out.bound;
    for (const auto& line : e.provider_lines) b << line << '\n';
    const char outcome = e.outcome == '-' ? 'N' : e.outcome;
    if (o.route == "stopping") {
      b << "S " << id << ' ' << e.z << ' ' << e.a << ' ' << Hex(e.t_before) << ' ' << Hex(e.t_after) << ' '
        << outcome;
#ifdef HARVEST_MIXTURE_TRACE
      const bool common = e.rate_status == 0;
      b << " P " << e.rate_calls << ' ' << e.rate_other << ' ' << e.rate_status << ' '
        << (common ? Hex(e.pair_c) : std::string("x")) << ' ' << (common ? Hex(e.pair_d) : std::string("x"));
#else
      b << " P x";
#endif
    } else {
      b << "A " << id << ' ' << e.z << ' ' << e.a << ' ' << (e.atom_seen ? Hex(e.atom_create) : std::string("x"))
        << ' ' << Hex(e.tau) << ' ' << Hex(e.tau_dio) << ' ' << Hex(e.tau_nc) << ' ' << Hex(e.counter) << ' '
        << Hex(e.delay) << ' ' << Hex(e.atom_time) << ' ' << e.branch << ' ' << outcome;
    }
#ifdef HARVEST_MIXTURE_TRACE
    b << " M " << e.model_calls << ' ' << e.model_times.size();
    for (double t : e.model_times) b << ' ' << Hex(t);
#else
    b << " M x";
#endif
    b << " N " << e.products.size();
    for (double t : e.products) b << ' ' << Hex(t);
    b << " E " << (e.dio_electrons == 1 ? Hex(e.electron_time) : std::string("-")) << '\n';
  }

 private:
  const Options& o;
  Output& out;
};

class Tracking : public G4UserTrackingAction {
 public:
  explicit Tracking(Output& output) : out(output) {}
  void PreUserTrackingAction(const G4Track* track) override {
    if (g_bound_record && dynamic_cast<const G4MuonicAtom*>(track->GetParticleDefinition()) != nullptr) {
      g_event.atom_create = track->GetGlobalTime();
      g_event.atom_seen = true;
    }
    if (!g_identity) return;
    Fold(EngineHash());
    const auto* creator = track->GetCreatorProcess();
    const int model = track->GetCreatorModelID();
    out.records << "T " << g_current_event << ' ' << track->GetTrackID() << ' ' << track->GetParentID() << ' '
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
        if (g_bound_record && secondary->GetCreatorModelID() == out.nc_model) {
          g_event.products.push_back(secondary->GetGlobalTime());
        }
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
      if (!dio && g_bound_record) {
        for (const auto* secondary : *secondaries) g_event.products.push_back(secondary->GetGlobalTime());
      }
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
    const long event = g_current_event;
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

// The record of options away from their defaults and, in a trace build, where each traced function resolves.
void WriteExtended(const Options& o) {
  std::ofstream file(o.out + "/extended.txt");
  if (!file) throw std::runtime_error("cannot open extended.txt");
  file << "ISOTOPE " << o.isotope << '\n';
  file << "T0_NS " << Hex(o.t0) << '\n';
  file << "EVENT_OFFSET " << o.event_offset << '\n';
  file << "BOUND_RATES " << (o.bound_rates ? "on" : "off") << '\n';
  file << "BOUND_RECORD " << (o.bound_record ? "on" : "off") << '\n';
  file << "RNG_STATES " << (o.rng_states ? "on" : "off") << '\n';
  file << "SYNTHETIC_CAPTURE_SCALE " << (o.synthetic_capture_scale != 0. ? Hex(o.synthetic_capture_scale) : "off")
       << '\n';
#ifdef HARVEST_MIXTURE_BOUND_RATES
  file << "PROVIDER enabled " << G4MuonicBoundRateProvider::IsEnabled() << " frozen "
       << G4MuonicBoundRateProvider::IsFrozen() << " parameters "
       << G4HadronicParameters::Instance()->EnableMuonicBoundRates() << '\n';
#else
  file << "PROVIDER none\n";
#endif
#ifdef HARVEST_MIXTURE_TRACE
  file << "TRACE on\n";
  for (const char* name : {kRatesSymbol, kBertiniSymbol, kPrecoSymbol}) {
    Dl_info self{}, next{};
    void* here = dlsym(RTLD_DEFAULT, name);
    void* there = dlsym(RTLD_NEXT, name);
    dladdr(here, &self);
    dladdr(there, &next);
    file << "INTERPOSE " << name << " default " << (self.dli_fname ? self.dli_fname : "?") << " next "
         << (next.dli_fname ? next.dli_fname : "?") << '\n';
  }
#else
  file << "TRACE off\n";
#endif
}

#ifdef HARVEST_MIXTURE_TRACE
// Every traced function must resolve to this executable, and its next definition to a Geant4 library.
void CheckInterposition() {
  char self[4096];
  const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n <= 0) throw std::runtime_error("cannot read /proc/self/exe");
  self[n] = '\0';
  for (const char* name : {kRatesSymbol, kBertiniSymbol, kPrecoSymbol}) {
    Dl_info here{}, next{};
    if (dladdr(dlsym(RTLD_DEFAULT, name), &here) == 0 || here.dli_fname == nullptr ||
        std::string(here.dli_fname) != self) {
      throw std::runtime_error(std::string("traced function does not resolve to this executable: ") + name);
    }
    if (dladdr(dlsym(RTLD_NEXT, name), &next) == 0 || next.dli_fname == nullptr ||
        std::string(next.dli_fname).find("libG4") == std::string::npos) {
      throw std::runtime_error(std::string("traced function has no Geant4 library definition: ") + name);
    }
  }
}
#endif

}  // namespace

#ifdef HARVEST_MIXTURE_TRACE
// The three traced functions: exported from this executable (the build passes --export-dynamic-symbol for each),
// so the libraries' calls bind here; each calls the library's definition once and returns its result.
G4MuonicBoundRateProvider::Status G4MuonicBoundRateProvider::Rates(G4int Z, G4int A, G4double muonMass,
                                                                   G4double nuclearMass, G4MuonicBoundRates& rates) {
  using Fn = Status (*)(G4int, G4int, G4double, G4double, G4MuonicBoundRates&);
  static const Fn next = reinterpret_cast<Fn>(NextDefinition(kRatesSymbol));
  void* const address = __builtin_return_address(0);
  const Status status = next(Z, A, muonMass, nuclearMass, rates);
  if (status == Status::kCommon && g_synthetic_capture_scale != 0.) rates.capture *= g_synthetic_capture_scale;
  const std::string caller = Caller(address);
  const bool common = status == Status::kCommon;
  if (caller == "bound") {
    ++g_event.rate_calls;
    g_event.rate_status = static_cast<int>(status);
    g_event.pair_c = common ? rates.capture : 0.;
    g_event.pair_d = common ? rates.decay : 0.;
  } else {
    if (caller != "helper") ++g_event.rate_other;
    g_event.provider_lines.push_back("C " + std::to_string(g_current_event) + ' ' + caller + ' ' + std::to_string(Z) +
                                     ' ' + std::to_string(A) + ' ' + std::to_string(static_cast<int>(status)) + ' ' +
                                     (common ? Hex(rates.capture) : std::string("x")) + ' ' +
                                     (common ? Hex(rates.decay) : std::string("x")));
  }
  return status;
}

G4HadFinalState* G4CascadeInterface::ApplyYourself(const G4HadProjectile& projectile, G4Nucleus& nucleus) {
  using Fn = G4HadFinalState* (*)(G4CascadeInterface*, const G4HadProjectile&, G4Nucleus&);
  static const Fn next = reinterpret_cast<Fn>(NextDefinition(kBertiniSymbol));
  G4HadFinalState* result = next(this, projectile, nucleus);
  RecordModel(projectile, result);
  return result;
}

G4HadFinalState* G4MuMinusCapturePrecompound::ApplyYourself(const G4HadProjectile& projectile, G4Nucleus& nucleus) {
  using Fn = G4HadFinalState* (*)(G4MuMinusCapturePrecompound*, const G4HadProjectile&, G4Nucleus&);
  static const Fn next = reinterpret_cast<Fn>(NextDefinition(kPrecoSymbol));
  G4HadFinalState* result = next(this, projectile, nucleus);
  RecordModel(projectile, result);
  return result;
}
#endif

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
    g_event_offset = options.event_offset;
    g_synthetic_capture_scale = options.synthetic_capture_scale;
    g_bound_record = options.bound_record;
#ifdef HARVEST_MIXTURE_TRACE
    CheckInterposition();
#endif
    std::filesystem::create_directories(options.out);
    Output output;
    output.stops.open(options.out + "/stops.txt");
    if (options.identity) output.records.open(options.out + "/records.txt");
    if (options.bound_record) output.bound.open(options.out + "/bound.txt");
    if (options.rng_states) output.rng.open(options.out + "/rng.txt");
    if (!output.stops || (options.identity && !output.records) || (options.bound_record && !output.bound) ||
        (options.rng_states && !output.rng)) {
      throw std::runtime_error("cannot open outputs");
    }
    G4Random::setTheEngine(new CLHEP::MixMaxRng);
    auto* manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
#ifdef HARVEST_MIXTURE_PATCHED
    if (options.opt_in == "on") G4HadronicParameters::Instance()->SetEnableMuonicData(true);
#endif
#ifdef HARVEST_MIXTURE_BOUND_RATES
    if (options.bound_rates) {
      G4HadronicParameters::Instance()->SetEnableMuonicBoundRates(true);
      if (!G4MuonicBoundRateProvider::IsEnabled()) throw std::runtime_error("the bound-rate switch did not take");
    }
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
    output.nc_model = G4PhysicsModelCatalog::GetModelID("model_muMinusCaptureAtRest_NuclearCapture");
    if (options.route == "stopping" && options.bound_record && output.nc_model < 0) {
      throw std::runtime_error("no nuclear-capture model id");
    }
    if (options.route == "stopping" && options.hooks) HookStoppingRoute();
    manager->BeamOn(options.events);
    output.stops.close();
    if (options.identity) {
      std::ofstream summary(options.out + "/summary.txt");
      summary << "hook_calls " << g_count.hook_calls << "\nbase_calls " << g_count.base_calls << "\nentry_mismatch "
              << g_count.entry_mismatch << "\nexit_mismatch " << g_count.exit_mismatch << '\n';
    }
    WriteConfig(options, detector);
    if (options.Extended()) WriteExtended(options);
    if (options.bound_record) {
      output.bound.close();
      std::ofstream totals(options.out + "/bound_summary.txt");
      totals << "events " << g_totals.events << "\ncaptures " << g_totals.captures << "\nsum_all "
             << Hex(g_totals.all.Value()) << "\nsum_capture " << Hex(g_totals.capture.Value()) << "\nsum_decay "
             << Hex(g_totals.decay.Value()) << '\n';
      if (!totals) throw std::runtime_error("cannot write bound_summary.txt");
    }
    if (options.rng_states) output.rng.close();
    delete manager;
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "harvest_mixture: %s\n", error.what());
    return 2;
  }
}
