// bound_rate_check -- the bound-rate provider (G4MuonicBoundRateProvider) run without Geant4, against the
// stand-in headers under g4stub/, in one of four modes, each its own process because the provider's switch
// freezes at the first consultation:
//
//   g4muonicdata_bound_rate_check oracle <d1_decay_rate.oracle>
//       LowZDecayRate(Z, mu_mass, mass) for every `B Z A mass rate` line of the oracle below Z = 14 equals
//       the rate a compiled Geant4 build returned, bit for bit, with the oracle's muon mass; the stand-in's
//       fine-structure constant and microsecond equal the oracle's `K alpha` and `K microsecond`.
//   g4muonicdata_bound_rate_check disabled
//       with the switch never set: Rates returns kDisabled and leaves its output alone, the capture
//       resolver is not called, the first consultation freezes the switch and a later Enable raises
//       G4MuonicBoundRates002 and changes nothing.
//   g4muonicdata_bound_rate_check enabled
//       with the switch set: the domain edges (Z = 1, 2, 13, 14; A = 0, Z - 1, Z), the pair for keys inside
//       it -- capture exactly what the resolver returned, decay exactly LowZDecayRate -- and, once frozen,
//       a Disable that raises G4MuonicBoundRates002 and a repeated Enable that raises nothing.
//   g4muonicdata_bound_rate_check refuse
//       with the switch set: each invalid input inside the domain raises G4MuonicBoundRates001, returns
//       kRefused and leaves the output alone -- a negative, a NaN and an infinite capture rate, a zero, a
//       negative, a NaN and an infinite nuclear mass, a zero muon mass, and a nuclear mass small enough to
//       make the decay rate negative -- while a zero capture rate is accepted.
//
// The stand-ins are defined here: G4MuonicAtomHelper::GetMuonCaptureRate returns the value the check
// planted for the key and counts its calls; G4Exception records each code (marked when its severity is not
// FatalException) and returns, which is what an exception handler that suppresses the abort does. Every failed expectation prints `FAIL <detail>`; the
// exit status is 1 if any did, else 0 after `OK <mode> <checks>`.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "G4Exception.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4MuonicBoundRateProvider.hh"
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

int Oracle(const char* path) {
  std::ifstream in(path);
  if (!in) {
    std::printf("FAIL cannot open %s\n", path);
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
    if (z >= 14) continue;
    const double got = P::LowZDecayRate(z, mu, mass);
    Expect(SameBits(got, rate), "Z=" + std::to_string(z) + " A=" + std::to_string(a) + " LowZDecayRate " + Hex(got) +
                                    " != oracle " + Hex(rate));
    ++compared;
  }
  std::printf("COMPARED %d oracle rates below Z = 14\n", compared);
  Expect(compared > 0, "no oracle rate below Z = 14");
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
      {"zero nuclear mass", 3.8e-5, kMu, 0., true},
      {"negative nuclear mass", 3.8e-5, kMu, -Mass(12), true},
      {"NaN nuclear mass", 3.8e-5, kMu, nan, true},
      {"infinite nuclear mass", 3.8e-5, kMu, inf, true},
      {"zero muon mass", 3.8e-5, 0., Mass(12), true},
      {"negative decay rate", 3.8e-5, kMu, 1.0e-3, true},
      {"zero capture", 0., kMu, Mass(12), false},
  };
  for (const auto& c : cases) {
    g_planted[{6, 12}] = c.capture;
    g_codes.clear();
    G4MuonicBoundRates r;
    r.capture = -1.;
    r.decay = -2.;
    const P::Status s = P::Rates(6, 12, c.muon, c.nuclear, r);
    if (c.refused) {
      Expect(s == P::Status::kRefused, std::string(c.label) + " returned " + Name(s));
      Expect(g_codes == std::vector<std::string>{"G4MuonicBoundRates001"}, std::string(c.label) + " raised " +
                                                                                std::to_string(g_codes.size()) +
                                                                                " codes, not G4MuonicBoundRates001");
      Expect(r.capture == -1. && r.decay == -2., std::string(c.label) + " wrote its output");
    } else {
      Expect(s == P::Status::kCommon, std::string(c.label) + " returned " + Name(s));
      Expect(g_codes.empty(), std::string(c.label) + " raised a code");
      Expect(r.capture == 0. && r.decay > 0., std::string(c.label) + " did not give capture 0 and decay > 0");
    }
  }
  return 0;
}

}  // namespace

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
  if (mode == "oracle" && argc == 3) {
    Oracle(argv[2]);
  } else if (mode == "disabled") {
    Disabled();
  } else if (mode == "enabled") {
    Enabled();
  } else if (mode == "refuse") {
    Refuse();
  } else {
    std::printf("usage: g4muonicdata_bound_rate_check oracle <file> | disabled | enabled | refuse\n");
    return 2;
  }
  if (g_failures > 0) return 1;
  std::printf("OK %s %d\n", mode.c_str(), g_checks);
  return 0;
}
