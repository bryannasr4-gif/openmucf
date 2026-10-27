// Standalone current-provider checks with g4stub headers. Each mode is a fresh process.
// current checks static limit and independent linear slopes; oracle-delta compares the change from
// the archived legacy getter, not equality to it. disabled/enabled/refuse check the unchanged switch,
// domain, forwarding and fatal/untouched-output contracts. The pair modes run the real overlay over a
// dataset directory (G4MUONICDATA): pair/pair-order check the evaluated row's precedence, its exact pair
// and the untouched keys in either option order; pair-compiled checks the pair selector at `compiled`;
// pair-switch-off checks the refusal of a pair profile without the switch; pair-config-refused checks that
// a refused pair configuration raises its code and then answers nothing. The G4Exception stand-in records
// a code and returns, as a handler that suppresses the abort would. No Geant4 or physical-accuracy claim.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "G4Exception.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4MuonicBoundRateProvider.hh"
#include "G4MuonicDataOverlay.hh"
#include "G4MuonicDataTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

namespace {

std::map<std::pair<int, int>, double> g_planted;
int g_resolver_calls = 0;
std::vector<std::string> g_codes;
int g_failures = 0;
int g_checks = 0;

void Expect(bool ok, const std::string& detail) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL %s\n", detail.c_str());
  }
}

bool SameBits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

std::string Hex(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%a", x);
  return buf;
}

using P = G4MuonicBoundRateProvider;

const char* Name(P::Status s) {
  switch (s) {
    case P::Status::kCommon: return "kCommon";
    case P::Status::kDisabled: return "kDisabled";
    case P::Status::kUnsupported: return "kUnsupported";
    case P::Status::kRefused: return "kRefused";
  }
  return "?";
}

// A muon mass and nuclear masses in MeV of the right size; the provider never looks them up itself.
constexpr double kMu = 105.6583715;
double Mass(int A) { return 931.494 * A; }

int OracleDelta(const char* path) {
  std::ifstream in(path);
  if (!in) {
    Expect(false, std::string("cannot open ") + path);
    return 1;
  }
  std::map<std::string, double> constants;
  std::vector<std::tuple<int, int, double, double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string tag;
    fields >> tag;
    if (tag == "K") {
      std::string name, value;
      fields >> name >> value;
      constants[name] = std::strtod(value.c_str(), nullptr);
    } else if (tag == "B") {
      int z = 0, a = 0;
      std::string mass, rate;
      fields >> z >> a >> mass >> rate;
      if (mass != "-") rows.emplace_back(z, a, std::strtod(mass.c_str(), nullptr), std::strtod(rate.c_str(), nullptr));
    }
  }
  Expect(constants.count("mu_mass") == 1 && constants.count("alpha") == 1 && constants.count("microsecond") == 1,
         "the oracle lacks one of its K lines");
  Expect(SameBits(constants["alpha"], fine_structure_const),
         "fine_structure_const " + Hex(fine_structure_const) + " != oracle alpha " + Hex(constants["alpha"]));
  Expect(SameBits(constants["microsecond"], microsecond),
         "microsecond " + Hex(microsecond) + " != oracle microsecond " + Hex(constants["microsecond"]));
  const double mu = constants["mu_mass"];
  int compared = 0;
  for (const auto& [z, a, mass, rate] : rows) {
    if (z < 2 || z >= 14) continue;
    const double got = P::LowZDecayRate(z, mu, mass);
    const double z2 = std::pow(z * fine_structure_const, 2);
    const double expected_delta = 1.06 * z2 * (mu / mass) * (0.45517005 / microsecond);
    Expect(std::abs((got - rate) - expected_delta) < 2e-19,
           "legacy-to-current delta Z=" + std::to_string(z) + " A=" + std::to_string(a));
    ++compared;
  }
  std::printf("COMPARED %d legacy oracle rates in current domain\n", compared);
  Expect(compared > 0, "no legacy oracle key in current domain");
  return 0;
}

int Current() {
  const double free = 0.45517005 / microsecond;
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double huge = std::numeric_limits<double>::max();
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Expect(std::string(P::Prescription()).find("low-z-inclusive-tree-linear-recoil-v1") == 0,
         "missing stable prescription identifier");
  for (int z : {2, 6, 13}) {
    const double z2 = std::pow(z * fine_structure_const, 2);
    const double q0 = P::LowZDecayRate(z, tiny, huge) / free;
    Expect(std::abs(q0 - (1. - z2 / 2.)) < 3e-16, "static limit");
    double previous = q0;
    for (double eta : {0.001, 0.01, 0.125, 0.375}) {
      const double q = P::LowZDecayRate(z, 1., 1. / eta) / free;
      // Affine slope from the static value distinguishes no-recoil, legacy and resummed mutants.
      Expect(std::abs((q - q0) / (z2 * eta) - 1.) < 2e-9, "linear positive recoil slope");
      Expect(q > previous && q <= 1., "monotone suppression");
      previous = q;
    }
    Expect(std::isfinite(P::LowZDecayRate(z, std::nextafter(0.5, 0.), 1.)), "just below eta boundary");
  }
  for (int z : {-1, 0, 1, 14, 1000000})
    Expect(std::isnan(P::LowZDecayRate(z, kMu, Mass(12))), "unsupported direct Z");
  for (double bad : {0., -1., nan, inf, -inf}) {
    Expect(std::isnan(P::LowZDecayRate(6, bad, Mass(12))), "invalid direct muon mass");
    Expect(std::isnan(P::LowZDecayRate(6, kMu, bad)), "invalid direct nuclear mass");
  }
  for (const auto& masses : std::vector<std::pair<double, double>>{
           {0.5, 1.}, {std::nextafter(0.5, 1.), 1.}, {1., 1.}, {huge, tiny}})
    Expect(std::isnan(P::LowZDecayRate(6, masses.first, masses.second)), "invalid direct eta");
  Expect(g_codes.empty() && g_resolver_calls == 0 && !P::IsFrozen(),
         "direct evaluator has switch/capture/exception side effects");
  return 0;
}

int Disabled() {
  Expect(!P::IsEnabled() && !P::IsFrozen(), "the switch is not off and unfrozen at start");
  G4MuonicBoundRates r;
  r.capture = -7.;
  r.decay = -8.;
  g_planted[{6, 12}] = 3.807e-05;
  const P::Status s = P::Rates(6, 12, kMu, Mass(12), r);
  Expect(s == P::Status::kDisabled, std::string("disabled Rates returned ") + Name(s));
  Expect(r.capture == -7. && r.decay == -8., "disabled Rates wrote its output");
  Expect(g_resolver_calls == 0, "disabled Rates called the capture resolver");
  Expect(P::Rates(6, 12, std::numeric_limits<double>::quiet_NaN(), 0., r) == P::Status::kDisabled,
         "disabled Rates did not bypass invalid inputs");
  Expect(r.capture == -7. && r.decay == -8. && g_resolver_calls == 0 && g_codes.empty(),
         "disabled invalid inputs had side effects");
  Expect(P::IsFrozen(), "the first consultation did not freeze the switch");
  Expect(!P::Consult(), "Consult reports the switch on");
  P::Enable();
  Expect(g_codes == std::vector<std::string>{"G4MuonicBoundRates002"}, "Enable after the freeze raised " +
                                                                          std::to_string(g_codes.size()) + " codes");
  Expect(!P::IsEnabled(), "Enable after the freeze changed the switch");
  P::Disable();
  Expect(g_codes.size() == 1, "a Disable that changes nothing raised a code");
  return 0;
}

int Enabled() {
  P::Enable();
  Expect(P::IsEnabled() && !P::IsFrozen(), "Enable did not set the switch, or froze it");
  struct Case {
    int z, a;
    P::Status want;
  };
  const std::vector<Case> cases = {
      {1, 1, P::Status::kUnsupported},  {1, 2, P::Status::kUnsupported},   {2, 1, P::Status::kUnsupported},
      {2, 2, P::Status::kCommon},       {2, 4, P::Status::kCommon},        {6, 0, P::Status::kUnsupported},
      {6, 5, P::Status::kUnsupported},  {6, 12, P::Status::kCommon},       {8, 18, P::Status::kCommon},
      {13, 27, P::Status::kCommon},     {13, 39, P::Status::kCommon},      {14, 28, P::Status::kUnsupported},
      {82, 208, P::Status::kUnsupported}, {0, 0, P::Status::kUnsupported}, {-1, 5, P::Status::kUnsupported},
  };
  for (const auto& c : cases) {
    const double planted = 1.0e-4 + 1.0e-7 * c.z + 1.0e-10 * c.a;  // distinct per key
    g_planted[{c.z, c.a}] = planted;
    const int calls_before = g_resolver_calls;
    G4MuonicBoundRates r;
    r.capture = -1.;
    r.decay = -2.;
    const P::Status s = P::Rates(c.z, c.a, kMu, Mass(c.a > 0 ? c.a : 1), r);
    const std::string key = "Z=" + std::to_string(c.z) + " A=" + std::to_string(c.a);
    Expect(s == c.want, key + " returned " + Name(s) + ", expected " + Name(c.want));
    Expect(P::InDomain(c.z, c.a) == (c.want == P::Status::kCommon), key + " InDomain disagrees with the status");
    if (c.want == P::Status::kCommon) {
      Expect(g_resolver_calls == calls_before + 1, key + " did not call the capture resolver once");
      Expect(SameBits(r.capture, planted), key + " capture " + Hex(r.capture) + " is not the resolver's " + Hex(planted));
      const double decay = P::LowZDecayRate(c.z, kMu, Mass(c.a));
      Expect(SameBits(r.decay, decay), key + " decay " + Hex(r.decay) + " is not LowZDecayRate " + Hex(decay));
    } else {
      Expect(g_resolver_calls == calls_before, key + " called the capture resolver outside the domain");
      Expect(r.capture == -1. && r.decay == -2., key + " wrote its output outside the domain");
    }
  }
  Expect(P::IsFrozen(), "Rates did not freeze the switch");
  Expect(g_codes.empty(), "a valid key raised a code");
  P::Disable();
  Expect(g_codes == std::vector<std::string>{"G4MuonicBoundRates002"}, "Disable after the freeze did not raise 002");
  Expect(P::IsEnabled(), "Disable after the freeze changed the switch");
  P::Enable();
  Expect(g_codes.size() == 1, "an Enable that changes nothing raised a code");
  return 0;
}

int Refuse() {
  P::Enable();
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  struct Case {
    const char* label;
    double capture, muon, nuclear;
    bool refused;
  };
  const std::vector<Case> cases = {
      {"negative capture", -1.0e-6, kMu, Mass(12), true},
      {"NaN capture", nan, kMu, Mass(12), true},
      {"infinite capture", inf, kMu, Mass(12), true},
      {"negative infinite capture", -inf, kMu, Mass(12), true},
      {"zero nuclear mass", 3.8e-5, kMu, 0., true},
      {"negative nuclear mass", 3.8e-5, kMu, -Mass(12), true},
      {"NaN nuclear mass", 3.8e-5, kMu, nan, true},
      {"infinite nuclear mass", 3.8e-5, kMu, inf, true},
      {"zero muon mass", 3.8e-5, 0., Mass(12), true},
      {"negative muon mass", 3.8e-5, -kMu, Mass(12), true},
      {"NaN muon mass", 3.8e-5, nan, Mass(12), true},
      {"infinite muon mass", 3.8e-5, inf, Mass(12), true},
      {"negative infinite muon mass", 3.8e-5, -inf, Mass(12), true},
      {"negative infinite nuclear mass", 3.8e-5, kMu, -inf, true},
      {"eta boundary", 3.8e-5, 0.5, 1., true},
      {"eta just above", 3.8e-5, std::nextafter(0.5, 1.), 1., true},
      {"eta overflow", 3.8e-5, std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::denorm_min(), true},
      {"eta far above boundary", 3.8e-5, kMu, 1.0e-3, true},
      {"zero capture", 0., kMu, Mass(12), false},
      {"eta just below", 3.8e-5, std::nextafter(0.5, 0.), 1., false},
      {"static underflow", 3.8e-5, std::numeric_limits<double>::denorm_min(),
                           std::numeric_limits<double>::max(), false},
  };
  for (const auto& c : cases) {
    g_planted[{6, 12}] = c.capture;
    g_codes.clear();
    const int calls_before = g_resolver_calls;
    G4MuonicBoundRates r;
    r.capture = -1.;
    r.decay = -2.;
    const P::Status s = P::Rates(6, 12, c.muon, c.nuclear, r);
    const bool valid_masses = std::isfinite(c.muon) && c.muon > 0. && std::isfinite(c.nuclear) &&
                              c.nuclear > 0. && std::isfinite(c.muon / c.nuclear) &&
                              c.muon / c.nuclear < 0.5;
    Expect(g_resolver_calls == calls_before + (valid_masses ? 1 : 0),
           std::string(c.label) + " unexpected capture resolver calls");
    if (c.refused) {
      Expect(s == P::Status::kRefused, std::string(c.label) + " returned " + Name(s));
      Expect(g_codes == std::vector<std::string>{"G4MuonicBoundRates001"}, std::string(c.label) + " raised " +
                                                                                std::to_string(g_codes.size()) +
                                                                                " codes, not G4MuonicBoundRates001");
      Expect(r.capture == -1. && r.decay == -2., std::string(c.label) + " wrote its output");
    } else {
      Expect(s == P::Status::kCommon, std::string(c.label) + " returned " + Name(s));
      Expect(g_codes.empty(), std::string(c.label) + " raised a code");
      Expect(r.capture == c.capture && r.decay > 0. && r.decay <= 0.45517005 / microsecond,
             std::string(c.label) + " did not forward capture with 0 < decay <= free");
    }
  }
  return 0;
}

// The evaluated row of the dataset's pair table, read with the reader alone: (Z, A, tau, q, free_tau).
struct Row {
  int z = 0, a = 0;
  double tau = 0., q = 0., free_tau = 0.;
};

Row PairRow() {
  const char* dir = std::getenv("G4MUONICDATA");
  const char* profile = std::getenv("G4MUONICDATA_D1_PAIR_PROFILE");
  Row row;
  if (dir == nullptr || profile == nullptr) {
    Expect(false, "G4MUONICDATA and G4MUONICDATA_D1_PAIR_PROFILE must be set");
    return row;
  }
  const G4MuonicDataTable dataset = G4MuonicDataTable::Load(dir);
  const G4MuonicDataTable::Table* table = dataset.Find(profile, "bound_rate_primitives");
  Expect(table != nullptr && !table->records.empty(), "the dataset carries no pair row");
  if (table == nullptr || table->records.empty()) return row;
  const auto& record = table->records.front();
  row.z = static_cast<int>(record.keys[0]);
  row.a = static_cast<int>(record.keys[1]);
  row.tau = record.floats[0];
  row.q = record.floats[1];
  row.free_tau = record.floats[2];
  return row;
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// The pair row answers both rates, exactly; another key keeps the formula pair and its capture resolver;
// a natural-composition request has no pair; the capture profile's own table is still read.
int Pair(bool dataFirst) {
  if (dataFirst) {
    G4MuonicDataTable::Enable();
    G4MuonicDataOverlay::Initialize();
    P::Enable();
  } else {
    P::Enable();
    G4MuonicDataTable::Enable();
    G4MuonicDataOverlay::Initialize();
  }
  const Row row = PairRow();
  const std::string profile = std::getenv("G4MUONICDATA_D1_PAIR_PROFILE");
  Expect(G4MuonicDataOverlay::Config().d1PairProfile == profile, "the overlay did not select the pair profile");
  G4MuonicBoundRates r;
  const int calls = g_resolver_calls;
  const P::Status s = P::Rates(row.z, row.a, kMu, Mass(row.a), r);
  Expect(s == P::Status::kCommon, std::string("the pair key returned ") + Name(s));
  const double decay = row.q / row.free_tau;
  const double capture = 1. / row.tau - decay;
  Expect(SameBits(r.decay, decay), "pair decay " + Hex(r.decay) + " is not q/free_tau " + Hex(decay));
  Expect(SameBits(r.capture, capture), "pair capture " + Hex(r.capture) + " is not 1/tau - q/free_tau " + Hex(capture));
  Expect(std::fabs((r.capture + r.decay) * row.tau - 1.) <= 1e-15, "the pair does not sum to 1/tau");
  Expect(g_resolver_calls == calls, "the pair key consulted the capture resolver");
  const std::string described = P::Describe(row.z, row.a);
  Expect(Contains(described, "answer=pair-row row=" + std::to_string(row.z) + "-" + std::to_string(row.a)) &&
             Contains(described, "pair_profile=" + profile) && Contains(described, "q_convention=evaluated:"),
         "readout of the pair key: " + described);
  std::printf("DESCRIBE %s\n", described.c_str());
  // another key in the formula domain keeps the formula pair
  const int other_z = row.z == 12 ? 11 : 12, other_a = 2 * other_z;
  g_planted[{other_z, other_a}] = 4.8e-4;
  G4MuonicBoundRates o;
  const int calls_other = g_resolver_calls;
  Expect(P::Rates(other_z, other_a, kMu, Mass(other_a), o) == P::Status::kCommon, "the formula key was not answered");
  Expect(g_resolver_calls == calls_other + 1 && SameBits(o.capture, 4.8e-4), "the formula key lost its capture");
  Expect(SameBits(o.decay, P::LowZDecayRate(other_z, kMu, Mass(other_a))), "the formula key lost its decay");
  const std::string other = P::Describe(other_z, other_a);
  Expect(Contains(other, "answer=common-formula") && Contains(other, "fallback=formula-no-pair-row"),
         "readout of the formula key: " + other);
  std::printf("DESCRIBE %s\n", other.c_str());
  // no natural-composition alias
  G4MuonicBoundRates n;
  n.capture = -1.;
  Expect(P::Rates(row.z, 0, kMu, Mass(row.a), n) == P::Status::kUnsupported && n.capture == -1.,
         "a natural-composition request reached the pair");
  Expect(G4MuonicDataOverlay::ResolvePair(row.z, 0).values == nullptr, "ResolvePair answered A = 0");
  // the capture profile is not replaced: its table still answers its own keys (a `compiled` capture seam
  // has no table to answer from)
  const std::string capture_profile = G4MuonicDataOverlay::Config().d1Profile;
  const G4MuonicDataOverlay::Resolution capture_row = G4MuonicDataOverlay::ResolveRate(row.z, row.a);
  if (capture_profile == "compiled") {
    Expect(capture_row.values == nullptr, "a compiled capture seam answered from a table");
  } else {
    Expect(capture_row.values != nullptr && capture_row.profile == capture_profile,
           "the capture profile's own row is no longer answered under the pair profile");
  }
  Expect(g_codes.empty(), "a valid pair configuration raised a code");
  return 0;
}

// The pair selector at `compiled`: the formula pair, exactly as without the dataset's pair table.
int PairCompiled() {
  P::Enable();
  G4MuonicDataTable::Enable();
  G4MuonicDataOverlay::Initialize();
  Expect(G4MuonicDataOverlay::Config().d1PairProfile == "compiled", "the pair selector is not compiled");
  g_planted[{13, 27}] = 6.985e-4;
  G4MuonicBoundRates r;
  Expect(P::Rates(13, 27, kMu, Mass(27), r) == P::Status::kCommon, "the formula key was not answered");
  Expect(SameBits(r.capture, 6.985e-4) && SameBits(r.decay, P::LowZDecayRate(13, kMu, Mass(27))),
         "with the pair selector compiled the pair is not the formula pair");
  const std::string described = P::Describe(13, 27);
  Expect(Contains(described, "pair_profile=compiled") && Contains(described, "fallback=none"), described);
  std::printf("DESCRIBE %s\n", described.c_str());
  Expect(g_codes.empty(), "the compiled pair selector raised a code");
  return 0;
}

// A pair profile with the switch off: refused at the first consultation, and every later one throws.
int PairSwitchOff() {
  G4MuonicDataTable::Enable();
  G4MuonicDataOverlay::Initialize();
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool threw = false;
    try {
      P::Consult();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Expect(threw, "a consultation with the switch off and a pair profile did not refuse");
  }
  Expect(g_codes == std::vector<std::string>{"G4MuonicBoundRates003"}, "the refusal did not raise 003 once");
  return 0;
}

// A refused pair configuration: its code is raised, and afterwards the provider answers nothing.
int PairConfigRefused(const std::string& code) {
  G4MuonicDataTable::Enable();
  bool threw = false;
  try {
    G4MuonicDataOverlay::Initialize();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "the refused configuration did not throw after the suppressed abort");
  Expect(!g_codes.empty() && g_codes.front() == code, "the refused configuration raised " +
                                                          (g_codes.empty() ? std::string("nothing") : g_codes.front()));
  P::Enable();
  G4MuonicBoundRates r;
  r.capture = -1.;
  bool answered = false;
  try {
    P::Rates(13, 27, kMu, Mass(27), r);
    answered = true;
  } catch (const std::runtime_error&) {
  }
  Expect(!answered && r.capture == -1., "a refused pair configuration still answered a rate pair");
  return 0;
}

// A selected evaluated row that cannot be used, wherever it is refused (at load by the semantic layer, or by the
// provider when it forms the rates): one of `codes` (comma-separated) is raised, and with the abort suppressed every
// later consultation throws; no call returns a status that lets a consumer keep its own pair.
int PairRowRefused(const std::string& codes) {
  G4MuonicDataTable::Enable();
  try {
    G4MuonicDataOverlay::Initialize();
  } catch (const std::runtime_error&) {
  }
  P::Enable();
  for (int attempt = 0; attempt < 3; ++attempt) {
    G4MuonicBoundRates r;
    r.capture = -1.;
    bool returned = false;
    try {
      const P::Status s = P::Rates(13, 27, kMu, Mass(27), r);
      returned = true;
      std::printf("RETURNED %s%s", Name(s), "\n");
    } catch (const std::runtime_error&) {
    }
    Expect(!returned && r.capture == -1., "a refused selected row returned a status a consumer may fall back on");
  }
  bool known = !g_codes.empty();
  for (const std::string& code : g_codes) known = known && ("," + codes + ",").find("," + code + ",") != std::string::npos;
  Expect(known, "the refusal raised another code than " + codes);
  return 0;
}

}  // namespace

const char* G4FindDataDir(const char* name) { return std::getenv(name); }

G4double G4MuonicAtomHelper::GetMuonCaptureRate(G4int Z, G4int A) {
  ++g_resolver_calls;
  const auto it = g_planted.find({Z, A});
  return it == g_planted.end() ? std::numeric_limits<double>::quiet_NaN() : it->second;
}

void G4Exception(const char* originOfException, const char* exceptionCode, G4ExceptionSeverity severity,
                 const char* description) {
  std::printf("G4Exception %s %s %d %s\n", originOfException, exceptionCode, static_cast<int>(severity),
              description);
  // Every code the provider raises is fatal; a code raised at another severity is recorded as such.
  g_codes.emplace_back(severity == FatalException ? std::string(exceptionCode)
                                                  : std::string(exceptionCode) + ":not-fatal");
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "oracle-delta" && argc == 3) {
    OracleDelta(argv[2]);
  } else if (mode == "current") {
    Current();
  } else if (mode == "disabled") {
    Disabled();
  } else if (mode == "enabled") {
    Enabled();
  } else if (mode == "refuse") {
    Refuse();
  } else if (mode == "pair") {
    Pair(false);
  } else if (mode == "pair-order") {
    Pair(true);
  } else if (mode == "pair-compiled") {
    PairCompiled();
  } else if (mode == "pair-switch-off") {
    PairSwitchOff();
  } else if (mode == "pair-config-refused" && argc == 3) {
    PairConfigRefused(argv[2]);
  } else if (mode == "pair-row-refused" && argc == 3) {
    PairRowRefused(argv[2]);
  } else {
    std::printf("usage: g4muonicdata_bound_rate_check oracle-delta <file> | current | disabled | enabled | refuse | "
                "pair | pair-order | pair-compiled | pair-switch-off | pair-config-refused <code> | pair-row-refused <codes>\n");
    return 2;
  }
  if (g_failures > 0) return 1;
  std::printf("OK %s %d\n", mode.c_str(), g_checks);
  return 0;
}
