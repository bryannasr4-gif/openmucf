// Compile against an installed Geant4 to harvest selector counts and at-rest bindings.
#include "G4AntiProton.hh"
#include "G4Box.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4ElementSelector.hh"
#include "G4HadronStoppingProcess.hh"
#include "G4Isotope.hh"
#include "G4KaonMinus.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusAtomicCapture.hh"
#include "G4MuonicAtomDecayPhysics.hh"
#include "G4NistManager.hh"
#include "G4Nucleus.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysListFactory.hh"
#include "G4PionMinus.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VProcess.hh"
#include "G4VPhysicsConstructor.hh"
#include "Randomize.hh"

#include <array>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace {
constexpr int kDraws = 1 << 20;

class QueueEngine final : public CLHEP::HepRandomEngine {
 public:
  void Fill(double first) { values = {first, 0.5}; cursor = 0; }
  int Consumed() const { return cursor; }
  double flat() override {
    if (cursor >= static_cast<int>(values.size())) throw std::runtime_error("selector requested an extra draw");
    return values[cursor++];
  }
  void flatArray(const int size, double* array) override {
    for (int i = 0; i < size; ++i) array[i] = flat();
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[] = "Config.conf") const override {}
  void restoreStatus(const char[] = "Config.conf") override {}
  void showStatus() const override {}
  std::string name() const override { return "SelectorQueue"; }

 private:
  std::vector<double> values;
  int cursor = 0;
};

G4Material* Custom(int z) {
  const int other_z = z == 2 ? 10 : 2;
  const int other_a = z == 2 ? 20 : 4;
  const std::string name = "zscan-" + std::to_string(z);
  auto* first_iso = new G4Isotope(name + "-isotope", z, 3 * z, 3 * z * g / mole);
  auto* first = new G4Element(name + "-element", name + "-symbol", 1);
  first->AddIsotope(first_iso, 100. * perCent);
  auto* second_iso = new G4Isotope(name + "-reference-isotope", other_z, other_a, other_a * g / mole);
  auto* second = new G4Element(name + "-reference", name + "-reference-symbol", 1);
  second->AddIsotope(second_iso, 100. * perCent);
  auto* material = new G4Material(name, 1. * g / cm3, 2);
  material->AddElement(first, 1);
  material->AddElement(second, 1);
  return material;
}

void HarvestMaterial(G4Material* material, QueueEngine* engine, G4ElementSelector* selector) {
  if (!material || material->GetNumberOfElements() < 1) throw std::runtime_error("cannot build material");
  auto* particle = new G4DynamicParticle(G4MuonMinus::MuonMinus(), G4ThreeVector(0., 0., 1.), 0.);
  G4Track track(particle, 0., G4ThreeVector());
  G4Step step;
  step.GetPreStepPoint()->SetMaterial(material);
  track.SetStep(&step);
  const auto* elements = material->GetElementVector();
  const auto* densities = material->GetAtomicNumDensityVector();
  std::vector<int> counts(material->GetNumberOfElements(), 0);
  std::vector<int> half(material->GetNumberOfElements(), -1);
  int total_draws = 0;
  for (int k = 0; k < kDraws; ++k) {
    engine->Fill((static_cast<double>(k) + 0.5) / kDraws);
    G4Nucleus nucleus;
    const G4Element* selected = selector->SelectZandA(track, &nucleus);
    std::size_t index = 0;
    while (index < counts.size() && (*elements)[index] != selected) ++index;
    if (index == counts.size()) throw std::runtime_error("selector returned an unknown element");
    const int expected = (counts.size() > 1 ? 1 : 0) + (selected->GetIsotopeVector()->size() > 1 ? 1 : 0);
    if (engine->Consumed() != expected) throw std::runtime_error("selector draw count differs");
    total_draws += expected;
    ++counts[index];
    if (half[index] == -1) half[index] = nucleus.GetA_asInt();
    else if (half[index] != nucleus.GetA_asInt()) throw std::runtime_error("fixed isotope draw changed A");
  }
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const G4Element* element = (*elements)[i];
    std::printf("M\t%s\t%zu\t%d\t%a\t", material->GetName().c_str(), i,
                element->GetZasInt(), densities[i]);
    const auto* isotopes = element->GetIsotopeVector();
    for (std::size_t j = 0; j < isotopes->size(); ++j) {
      if (j) std::printf(";");
      std::printf("%d", (*isotopes)[j]->GetN());
    }
    std::printf("\t");
    const auto* abundance = element->GetRelativeAbundanceVector();
    for (std::size_t j = 0; j < isotopes->size(); ++j) {
      if (j) std::printf(";");
      std::printf("%a", abundance[j]);
    }
    // A zero-count element still has a fixed isotope selected at draw 0.5.
    if (half[i] == -1) {
      double remaining = 0.5;
      for (std::size_t j = 0; j < isotopes->size(); ++j) {
        remaining -= abundance[j];
        if (remaining <= 0.) { half[i] = (*isotopes)[j]->GetN(); break; }
      }
    }
    std::printf("\t%d\t%d\t%d\n", half[i], counts[i], total_draws);
  }
}

void Selector() {
  G4MuonMinus::MuonMinus();
  auto* nist = G4NistManager::Instance();
  std::vector<G4Material*> materials;
  for (const char* name : {"G4_TEFLON", "G4_LITHIUM_FLUORIDE", "G4_POLYVINYL_CHLORIDE",
                           "G4_SODIUM_IODIDE", "G4_WATER", "G4_POLYETHYLENE", "G4_SILICON_DIOXIDE"}) {
    auto* material = nist->FindOrBuildMaterial(name);
    if (!material) throw std::runtime_error(std::string("cannot build ") + name);
    materials.push_back(material);
  }
  for (int z = 1; z <= 100; ++z) materials.push_back(Custom(z));
  auto* engine = new QueueEngine;
  G4Random::setTheEngine(engine);
  G4ElementSelector selector;
  for (auto* material : materials) HarvestMaterial(material, engine, &selector);
}

class Detector final : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* material = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
    auto* solid = new G4Box("world", 1. * m, 1. * m, 1. * m);
    auto* logical = new G4LogicalVolume(solid, material, "world");
    return new G4PVPlacement(nullptr, G4ThreeVector(), logical, "world", nullptr, false, 0);
  }
};

class AtomicCapturePhysics final : public G4VPhysicsConstructor {
 public:
  AtomicCapturePhysics() : G4VPhysicsConstructor("MuAtomicCapture") {}
  void ConstructParticle() override { G4MuonMinus::MuonMinus(); }
  void ConstructProcess() override {
    G4MuonMinus::MuonMinus()->GetProcessManager()->AddRestProcess(new G4MuonMinusAtomicCapture());
  }
};

std::string ClassName(const G4VProcess* process) {
  int status = 0;
  std::unique_ptr<char, decltype(&std::free)> demangled(
      abi::__cxa_demangle(typeid(*process).name(), nullptr, nullptr, &status), &std::free);
  return status == 0 ? demangled.get() : typeid(*process).name();
}

void Bindings(const std::string& name) {
  auto* manager = new G4RunManager;
  manager->SetUserInitialization(new Detector);
  G4PhysListFactory factory;
  const bool helper = name == "QBBC+helper";
  auto* physics = factory.GetReferencePhysList(helper ? "QBBC" : name);
  if (!physics) throw std::runtime_error("unknown physics list " + name);
  if (helper) {
    physics->ReplacePhysics(new G4StoppingPhysics("stopping", 1, false));
    physics->RegisterPhysics(new AtomicCapturePhysics());
    physics->RegisterPhysics(new G4MuonicAtomDecayPhysics());
  }
  manager->SetUserInitialization(physics);
  manager->Initialize();
  const std::array<G4ParticleDefinition*, 4> particles = {
      G4MuonMinus::MuonMinus(), G4PionMinus::PionMinus(),
      G4KaonMinus::KaonMinus(), G4AntiProton::AntiProton()};
  for (auto* particle : particles) {
    auto* process_manager = particle->GetProcessManager();
    if (!process_manager) throw std::runtime_error("particle has no process manager");
    auto* vector = process_manager->GetAtRestProcessVector();
    if (!vector || vector->size() == 0) {
      std::printf("B\t%s\t-\t-\tfalse\tfalse\n", particle->GetParticleName().c_str());
      continue;
    }
    for (std::size_t i = 0; i < vector->size(); ++i) {
      const auto* process = (*vector)[i];
      std::printf("B\t%s\t%s\t%s\t%s\t%s\n", particle->GetParticleName().c_str(),
                  process->GetProcessName().c_str(), ClassName(process).c_str(),
                  dynamic_cast<const G4HadronStoppingProcess*>(process) ? "true" : "false",
                  dynamic_cast<const G4MuonMinusAtomicCapture*>(process) ? "true" : "false");
    }
  }
  delete manager;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "selector") Selector();
    else if (argc == 3 && std::string(argv[1]) == "bindings") Bindings(argv[2]);
    else throw std::runtime_error("usage: harvest_d2 selector | bindings <physics list>");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "harvest_d2: %s\n", error.what());
    return 2;
  }
}
