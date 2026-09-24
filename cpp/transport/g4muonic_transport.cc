#include "G4Alpha.hh"
#include "G4Box.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4Electron.hh"
#include "G4Element.hh"
#include "G4EmCaptureCascade.hh"
#include "G4Event.hh"
#include "G4Gamma.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadronicParameters.hh"
#include "G4He3.hh"
#include "G4Isotope.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusAtomicCapture.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtom.hh"
#include "G4MuonicAtomDecayPhysics.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4Neutron.hh"
#include "G4Nucleus.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4ProcessManager.hh"
#include "G4Proton.hh"
#include "G4PVPlacement.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4Triton.hh"
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

#ifdef G4MUONIC_TRANSPORT_PATCHED
#include "G4MuonicDataOverlay.hh"
#endif

#include <CLHEP/Random/MixMaxRng.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
  std::string route;
  std::string opt_in;
  std::string out;
  int z = 0;
  int a = 0;
  int events = 0;
  int threads = 0;
  long seed_base = 0;
  long seed_stride = 0;
  bool config_only = false;
};

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--config-only") {
      o.config_only = true;
      continue;
    }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value(argv[++i]);
    if (key == "--route") o.route = value;
    else if (key == "--opt-in") o.opt_in = value;
    else if (key == "--Z") o.z = std::stoi(value);
    else if (key == "--A") o.a = std::stoi(value);
    else if (key == "--events") o.events = std::stoi(value);
    else if (key == "--threads") o.threads = std::stoi(value);
    else if (key == "--seed-base") o.seed_base = std::stol(value);
    else if (key == "--seed-stride") o.seed_stride = std::stol(value);
    else if (key == "--out") o.out = value;
    else throw std::runtime_error("unknown argument " + key);
  }
  if ((o.route != "bound_decay" && o.route != "muonic_atom_helper") ||
      (o.opt_in != "on" && o.opt_in != "off") || o.z <= 0 || o.a < o.z ||
      o.events <= 0 || o.threads <= 0 || o.out.empty() || o.seed_base <= 0 || o.seed_stride <= 0) {
    throw std::runtime_error("invalid route, mode, target, event count, thread count, seed or output");
  }
#ifndef G4MUONIC_TRANSPORT_PATCHED
  if (o.opt_in == "on") throw std::runtime_error("opt-in requires a patched build");
#endif
#ifndef G4MULTITHREADED
  if (o.threads > 1) throw std::runtime_error("this Geant4 installation lacks multithreading");
#endif
  return o;
}

std::string Hex(double value) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%a", value);
  return buf;
}

std::string Model(int id) {
  return id < 0 ? "-" : std::string(G4PhysicsModelCatalog::GetModelNameFromID(id));
}

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Options& options) : o(options) {}
  G4VPhysicalVolume* Construct() override {
    auto* isotope = new G4Isotope("target_isotope", o.z, o.a);
    auto* element = new G4Element("target_element", "target", 1);
    element->AddIsotope(isotope, 100. * perCent);
    material = new G4Material("target_material", 1. * g / cm3, 1);
    material->AddElement(element, 1.);
    auto* solid = new G4Box("world", 1. * m, 1. * m, 1. * m);
    auto* logical = new G4LogicalVolume(solid, material, "world");
    return new G4PVPlacement(nullptr, G4ThreeVector(), logical, "world", nullptr, false, 0);
  }
  G4Material* material = nullptr;

 private:
  const Options& o;
};

class AtomicCapturePhysics : public G4VPhysicsConstructor {
 public:
  AtomicCapturePhysics() : G4VPhysicsConstructor("MuAtomicCapture") {}
  void ConstructParticle() override { G4MuonMinus::MuonMinus(); }
  void ConstructProcess() override {
    G4MuonMinus::MuonMinus()->GetProcessManager()->AddRestProcess(new G4MuonMinusAtomicCapture());
  }
};

struct Recorder {
  explicit Recorder(const std::string& dir) {
    const int id = G4Threading::G4GetThreadId();
    file.open(dir + "/records-" + std::to_string(id) + ".txt");
    if (!file) throw std::runtime_error("cannot open worker record file");
  }
  std::ofstream file;
  int event = -1;
  int tracks = 0;
  long seed = 0;
};

class Primary : public G4VUserPrimaryGeneratorAction {
 public:
  Primary(const Options& options, std::shared_ptr<Recorder> record)
      : o(options), r(std::move(record)), gun(1) {
    gun.SetParticleDefinition(G4MuonMinus::MuonMinus());
    gun.SetParticleEnergy(0.);
    gun.SetParticlePosition(G4ThreeVector());
    gun.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
  }
  void GeneratePrimaries(G4Event* event) override {
    r->event = event->GetEventID();
    r->seed = o.seed_base + o.seed_stride * 0 + r->event;
    G4Random::setTheSeed(r->seed);
    gun.GeneratePrimaryVertex(event);
  }

 private:
  const Options& o;
  std::shared_ptr<Recorder> r;
  G4ParticleGun gun;
};

class Event : public G4UserEventAction {
 public:
  explicit Event(std::shared_ptr<Recorder> record) : r(std::move(record)) {}
  void BeginOfEventAction(const G4Event*) override { r->tracks = 0; }
  void EndOfEventAction(const G4Event* event) override {
    r->file << "E " << event->GetEventID() << ' ' << r->seed << ' ' << r->tracks << '\n';
    r->file.flush();
  }

 private:
  std::shared_ptr<Recorder> r;
};

class Tracking : public G4UserTrackingAction {
 public:
  explicit Tracking(std::shared_ptr<Recorder> record) : r(std::move(record)) {}
  void PreUserTrackingAction(const G4Track* track) override {
    ++r->tracks;
    const auto* creator = track->GetCreatorProcess();
    r->file << "T " << r->event << ' ' << track->GetTrackID() << ' ' << track->GetParentID()
            << ' ' << track->GetParticleDefinition()->GetPDGEncoding() << ' '
            << (creator == nullptr ? "-" : creator->GetProcessName()) << ' '
            << Model(track->GetCreatorModelID()) << ' ' << Hex(track->GetKineticEnergy())
            << ' ' << Hex(track->GetGlobalTime()) << '\n';
  }

 private:
  std::shared_ptr<Recorder> r;
};

class Stepping : public G4UserSteppingAction {
 public:
  explicit Stepping(std::shared_ptr<Recorder> record) : r(std::move(record)) {}
  void UserSteppingAction(const G4Step* step) override {
    const auto* track = step->GetTrack();
    const auto* definition = track->GetParticleDefinition();
    if (definition != G4MuonMinus::MuonMinus() && dynamic_cast<const G4MuonicAtom*>(definition) == nullptr) return;
    const auto* post = step->GetPostStepPoint();
    const auto* process = post->GetProcessDefinedStep();
    r->file << "S " << r->event << ' ' << track->GetTrackID() << ' ' << track->GetCurrentStepNumber()
            << ' ' << (process == nullptr ? "-" : process->GetProcessName()) << ' '
            << Hex(post->GetKineticEnergy()) << ' ' << Hex(post->GetGlobalTime()) << ' '
            << step->GetSecondaryInCurrentStep()->size() << '\n';
    int k = 0;
    for (const auto* secondary : *step->GetSecondaryInCurrentStep()) {
      r->file << "C " << r->event << ' ' << track->GetTrackID() << ' ' << k++ << ' '
              << secondary->GetParticleDefinition()->GetPDGEncoding() << ' '
              << Model(secondary->GetCreatorModelID()) << ' ' << Hex(secondary->GetKineticEnergy())
              << ' ' << Hex(secondary->GetGlobalTime()) << '\n';
    }
    r->file.flush();
  }

 private:
  std::shared_ptr<Recorder> r;
};

class Actions : public G4VUserActionInitialization {
 public:
  explicit Actions(const Options& options) : o(options) {}
  void Build() const override {
    auto record = std::make_shared<Recorder>(o.out);
    SetUserAction(new Primary(o, record));
    SetUserAction(new Event(record));
    SetUserAction(new Tracking(record));
    SetUserAction(new Stepping(record));
  }

 private:
  const Options& o;
};

struct CascadeLevels {
  using type = G4double (G4EmCaptureCascade::*)[14];
  friend type LevelsOf(CascadeLevels);
};
template <typename Tag, typename Tag::type Member>
struct AccessPrivate {
  friend typename Tag::type LevelsOf(Tag) { return Member; }
};
template struct AccessPrivate<CascadeLevels, &G4EmCaptureCascade::fLevelEnergy>;

void ParticlesOrExit() {
  for (const char* name : {"neutron", "deuteron", "triton", "alpha", "He3", "proton"}) {
    if (G4ParticleTable::GetParticleTable()->FindParticle(name) == nullptr) {
      std::fprintf(stderr, "g4muonic_transport: the particle table lacks %s\n", name);
      std::exit(2);
    }
  }
}

#ifdef G4MUONIC_TRANSPORT_PATCHED
const char* Origin(G4MuonicDataOverlay::Origin origin) {
  switch (origin) {
    case G4MuonicDataOverlay::Origin::Exact: return "exact";
    case G4MuonicDataOverlay::Origin::ExplicitNatural: return "explicit_natural";
    case G4MuonicDataOverlay::Origin::ExplicitRepresentative: return "explicit_representative";
    case G4MuonicDataOverlay::Origin::LegacyNaturalFallback: return "legacy_natural_fallback";
    case G4MuonicDataOverlay::Origin::Compiled: return "compiled";
  }
  return "unknown";
}
void Resolution(std::ofstream& file, const char* name, G4MuonicDataOverlay::Resolution r) {
  file << "RES " << name << ' ' << r.requestedZ << ' ' << r.requestedA << ' '
       << Origin(r.origin) << ' ' << r.profile << ' ' << r.count << '\n';
}
#endif

void WriteConfig(const Options& o, Detector* detector) {
  std::ofstream file(o.out + "/config.txt");
  if (!file) throw std::runtime_error("cannot open config.txt");
  file << "VERSION " << G4VERSION_TAG << '\n';
  file << "ENGINE MixMaxRng\n";
  const auto* muon = G4MuonMinus::MuonMinus();
  auto* processes = muon->GetProcessManager()->GetProcessList();
  file << "PROCESSES";
  for (int i = 0; i < processes->size(); ++i) file << ' ' << (*processes)[i]->GetProcessName();
  file << '\n';
  auto* material = detector->material;
  file << "MATERIAL " << material->GetNumberOfElements();
  for (const auto* element : *material->GetElementVector()) {
    file << ' ' << element->GetNumberOfIsotopes();
    for (int i = 0; i < element->GetNumberOfIsotopes(); ++i) {
      auto* isotope = element->GetIsotope(i);
      file << ' ' << isotope->GetZ() << ' ' << isotope->GetN()
           << ' ' << Hex(element->GetRelativeAbundanceVector()[i]);
    }
  }
  file << '\n';
  file << "PARTICLES ok\n";
#ifdef G4MUONIC_TRANSPORT_PATCHED
  const auto& config = G4MuonicDataOverlay::Config();
  file << "CONFIG " << config.d1Profile << ' ' << config.d3Profile << ' '
       << (config.datasetDirectory.empty() ? "none" : "set") << ' '
       << (config.datasetVersion.empty() ? "none" : config.datasetVersion) << '\n';
  file << "DIRECTORY " << (config.datasetDirectory.empty() ? "none" : config.datasetDirectory) << '\n';
  Resolution(file, "rate", G4MuonicDataOverlay::ResolveRate(o.z, o.a));
  Resolution(file, "zeff", G4MuonicDataOverlay::ResolveZeff(o.z));
  Resolution(file, "kshell", G4MuonicDataOverlay::ResolveKShell(o.z, o.a));
  Resolution(file, "levels", G4MuonicDataOverlay::ResolveLevels(o.z, o.a));
#endif
  file << "RATE_BD " << Hex(G4MuonMinusBoundDecay::GetMuonCaptureRate(o.z, o.a)) << '\n';
  file << "ZEFF_BD " << Hex(G4MuonMinusBoundDecay::GetMuonZeff(o.z)) << '\n';
  file << "RATE_HELPER " << Hex(G4MuonicAtomHelper::GetMuonCaptureRate(o.z, o.a)) << '\n';
  file << "ZEFF_HELPER " << Hex(G4MuonicAtomHelper::GetMuonZeff(o.z)) << '\n';
  file << "K1 " << Hex(G4MuonicAtomHelper::GetKShellEnergy(o.z)) << '\n';
#ifdef G4MUONIC_TRANSPORT_PATCHED
  file << "KA " << Hex(G4MuonicAtomHelper::GetKShellEnergy(o.z, o.a)) << '\n';
#endif
  auto* cascade = new G4EmCaptureCascade();
  G4Random::setTheSeed(o.seed_base);
  G4DynamicParticle particle(G4MuonMinus::MuonMinus(), G4ThreeVector(0., 0., 1.), 0.);
  G4HadProjectile projectile(particle);
  G4Nucleus nucleus(o.a, o.z);
  auto* result = cascade->ApplyYourself(projectile, nucleus);
  file << "L";
  const G4double* levels = cascade->*LevelsOf(CascadeLevels());
  for (int i = 0; i < 14; ++i) file << ' ' << Hex(levels[i]);
  file << '\n';
  for (int i = 0; i < result->GetNumberOfSecondaries(); ++i) {
    delete result->GetSecondary(i)->GetParticle();
  }
  result->Clear();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = Parse(argc, argv);
    std::filesystem::create_directories(options.out);
    G4Random::setTheEngine(new CLHEP::MixMaxRng);
    auto* manager = G4RunManagerFactory::CreateRunManager();
#ifdef G4MULTITHREADED
    manager->SetNumberOfThreads(options.threads);
#endif
#ifdef G4MUONIC_TRANSPORT_PATCHED
    if (options.opt_in == "on") G4HadronicParameters::Instance()->SetEnableMuonicData(true);
#endif
    auto* detector = new Detector(options);
    manager->SetUserInitialization(detector);
    auto* physics = new QBBC;
    if (options.route == "muonic_atom_helper") {
      physics->ReplacePhysics(new G4StoppingPhysics("stopping", 1, false));
      physics->RegisterPhysics(new AtomicCapturePhysics());
      physics->RegisterPhysics(new G4MuonicAtomDecayPhysics());
    }
    manager->SetUserInitialization(physics);
    manager->SetUserInitialization(new Actions(options));
    manager->Initialize();
    ParticlesOrExit();
    if (!options.config_only) manager->BeamOn(options.events);
    WriteConfig(options, detector);
    delete manager;
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "g4muonic_transport: %s\n", error.what());
    return 2;
  }
}
