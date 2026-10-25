// Transport confirmation of the clock_probe process-boundary checks.
#include "G4Box.hh"
#include "G4Element.hh"
#include "G4Event.hh"
#include "G4GenericMuonicAtom.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadronStoppingProcess.hh"
#include "G4HadronicInteraction.hh"
#include "G4Isotope.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusAtomicCapture.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4MuonicAtom.hh"
#include "G4MuonicAtomDecay.hh"
#include "G4MuonicAtomDecayPhysics.hh"
#include "G4MuMinusCapturePrecompound.hh"
#include "G4Nucleus.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicsListHelper.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessTable.hh"
#include "G4RunManagerFactory.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "QBBC.hh"
#include "Randomize.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
bool atomic=false, spy=true;
double t0=0., boundWait=0.;
bool boundDecay=false;
int eventID() { return G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID(); }
struct Detector : G4VUserDetectorConstruction {
  G4VPhysicalVolume* Construct() override {
    auto* el=new G4Element("Pb208","Pb",1);
    el->AddIsotope(new G4Isotope("Pb208",82,208),1.);
    auto* mat=new G4Material("Pb208",11.35*g/cm3,1); mat->AddElement(el,1.);
    auto* lv=new G4LogicalVolume(new G4Box("world",m,m,m),mat,"world");
    return new G4PVPlacement(nullptr,{},lv,"world",nullptr,false,0);
  }
};
struct SpyBound : G4HadronicInteraction {
  G4MuonMinusBoundDecay* inner=new G4MuonMinusBoundDecay;
  SpyBound() : G4HadronicInteraction("ClockSpyBound") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile& p,G4Nucleus& n) override {
    auto* r=inner->ApplyYourself(p,n); boundWait=p.GetGlobalTime();
    boundDecay=r->GetStatusChange()==stopAndKill; return r;
  }
};
struct SpyNC : G4HadronicInteraction {
  G4MuMinusCapturePrecompound* inner=new G4MuMinusCapturePrecompound;
  SpyNC() : G4HadronicInteraction("ClockSpyAtomicCapture") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile& p,G4Nucleus& n) override {
    auto* r=inner->ApplyYourself(p,n);
    for(std::size_t i=0;i<r->GetNumberOfSecondaries();++i)
      std::printf("MODEL_LOCAL %d %zu %a\n",eventID(),i,r->GetSecondary(i)->GetTime()/ns);
    return r;
  }
};
struct AtomicPhysics : G4VPhysicsConstructor {
  AtomicPhysics() : G4VPhysicsConstructor("AtomicCapture") {}
  void ConstructParticle() override { G4MuonMinus::MuonMinus(); }
  void ConstructProcess() override {
    G4MuonMinus::MuonMinus()->GetProcessManager()->AddRestProcess(new G4MuonMinusAtomicCapture);
  }
};
struct AtomicDecayPhysics : G4VPhysicsConstructor {
  AtomicDecayPhysics() : G4VPhysicsConstructor("AtomicDecay") {}
  void ConstructParticle() override { G4GenericMuonicAtom::GenericMuonicAtom(); }
  void ConstructProcess() override {
    G4PhysicsListHelper::GetPhysicsListHelper()->RegisterProcess(
      new G4MuonicAtomDecay(new SpyNC),G4GenericMuonicAtom::GenericMuonicAtom());
  }
};
struct Gun : G4VUserPrimaryGeneratorAction {
  G4ParticleGun gun{1};
  Gun() { gun.SetParticleDefinition(G4MuonMinus::MuonMinus()); gun.SetParticleEnergy(0.); gun.SetParticleTime(t0); }
  void GeneratePrimaries(G4Event* e) override { gun.GeneratePrimaryVertex(e); }
};
struct Tracking : G4UserTrackingAction {
  void PreUserTrackingAction(const G4Track* t) override {
    if(!dynamic_cast<const G4MuonicAtom*>(t->GetDefinition())) return;
    std::printf("ATOM_START %d %a\n",eventID(),t->GetGlobalTime()/ns);
    auto* list=t->GetDefinition()->GetProcessManager()->GetProcessList();
    for(int i=0;i<list->size();++i)
      if(auto* p=dynamic_cast<G4MuonicAtomDecay*>((*list)[i])) p->SetVerboseLevel(2);
  }
};
struct Steps : G4UserSteppingAction {
  void UserSteppingAction(const G4Step* s) override {
    const auto* track=s->GetTrack();
    const bool atom=dynamic_cast<const G4MuonicAtom*>(track->GetDefinition());
    if(!atom && track->GetTrackID()!=1) return;
    const auto* products=s->GetSecondaryInCurrentStep();
    if(products->empty()) return;
    if(!atomic && spy)
      std::printf("BOUND_WAIT %d %a branch %d\n",eventID(),boundWait/ns,boundDecay);
    for(std::size_t i=0;i<products->size();++i) {
      const auto* t=(*products)[i];
      const auto name=G4PhysicsModelCatalog::GetModelNameFromID(t->GetCreatorModelID());
      const char* phase=atom ? "atom_daughter" : dynamic_cast<const G4MuonicAtom*>(t->GetDefinition()) ? "atom" :
        name.find("EMCascade")!=std::string::npos ? "em" :
        name.find("NuclearCapture")!=std::string::npos ? "capture" :
        name.find("DIO")!=std::string::npos ? "decay" : "other";
      std::printf("PRODUCT %d %zu %s phase %s time %a energy %a\n",eventID(),i,
        t->GetDefinition()->GetParticleName().c_str(),phase,t->GetGlobalTime()/ns,t->GetKineticEnergy()/MeV);
    }
  }
};
struct Actions : G4VUserActionInitialization {
  void Build() const override { SetUserAction(new Gun); SetUserAction(new Tracking); SetUserAction(new Steps); }
};
}
int main(int argc,char** argv) {
  if(argc!=6) { std::fprintf(stderr,"usage: clock_transport atomic|stopping spy|plain t0_ns events seed\n"); return 2; }
  atomic=std::strcmp(argv[1],"atomic")==0; spy=std::strcmp(argv[2],"spy")==0;
  t0=std::atof(argv[3])*ns; G4Random::setTheSeed(std::atol(argv[5]));
  G4cout.precision(17); // includes the stock atom-decay sampled-wait diagnostic
  auto* rm=G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Detector);
  auto* physics=new QBBC(0);
  if(atomic) {
    physics->ReplacePhysics(new G4StoppingPhysics("stopping",0,false));
    physics->RegisterPhysics(new AtomicPhysics);
    if(spy) physics->RegisterPhysics(new AtomicDecayPhysics);
    else physics->RegisterPhysics(new G4MuonicAtomDecayPhysics);
  }
  rm->SetUserInitialization(physics); rm->SetUserInitialization(new Actions); rm->Initialize();
  if(!atomic && spy) {
    auto* p=dynamic_cast<G4HadronStoppingProcess*>(G4ProcessTable::GetProcessTable()->FindProcess(
      "muMinusCaptureAtRest",G4MuonMinus::MuonMinus()));
    if(!p) return 3;
    p->SetBoundDecay(new SpyBound);
  }
  rm->BeamOn(std::atoi(argv[4])); delete rm;
}
