// Focused consumer-boundary probe: actual Geant4 AtRestDoIt, before scheduling.
// Synthetic interactions isolate clock composition; stock interactions check
// matched-input branch, kinematics and RNG state independently of transport order.
#include "G4Box.hh"
#include "G4Element.hh"
#include "G4Gamma.hh"
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
#include "G4MuonicAtomHelper.hh"
#include "G4IonTable.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4Nucleus.hh"
#include "G4PVPlacement.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessTable.hh"
#include "G4RunManagerFactory.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4VUserDetectorConstruction.hh"
#include "QBBC.hh"
#include "Randomize.hh"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <cstdint>

namespace {
G4Material* material;
struct Detector : G4VUserDetectorConstruction {
  G4VPhysicalVolume* Construct() override {
    auto* el = new G4Element("Pb208", "Pb", 1);
    el->AddIsotope(new G4Isotope("Pb208", 82, 208), 1.);
    material = new G4Material("Pb208", 11.35*g/cm3, 1);
    material->AddElement(el, 1.);
    auto* lv = new G4LogicalVolume(new G4Box("world",m,m,m), material,"world");
    return new G4PVPlacement(nullptr,{},lv,"world",nullptr,false,0);
  }
};
struct AtomicPhysics : G4VPhysicsConstructor {
  AtomicPhysics() : G4VPhysicsConstructor("AtomicCapture") {}
  void ConstructParticle() override { G4MuonMinus::MuonMinus(); }
  void ConstructProcess() override {
    G4MuonMinus::MuonMinus()->GetProcessManager()->AddRestProcess(new G4MuonMinusAtomicCapture);
  }
};

double waitTime=0., modelTime=0.;
bool decayBranch=false;
struct SpyBound : G4HadronicInteraction {
  G4MuonMinusBoundDecay* inner=new G4MuonMinusBoundDecay;
  SpyBound() : G4HadronicInteraction("ClockSpyBound") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile& p,G4Nucleus& n) override {
    auto* r=inner->ApplyYourself(p,n);
    waitTime=p.GetGlobalTime(); decayBranch=r->GetStatusChange()==stopAndKill;
    return r;
  }
};
struct Cascade : G4HadronicInteraction {
  G4HadFinalState result;
  Cascade() : G4HadronicInteraction("ClockCascade") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile&,G4Nucleus&) override {
    result.Clear();
    result.AddSecondary(new G4DynamicParticle(G4Gamma::Gamma(),{1,0,0},0.01*MeV));
    result.GetSecondary(0)->SetTime(0.375*ns); // legitimate positive cascade delay
    return &result;
  }
};
struct Bound : G4HadronicInteraction {
  G4HadFinalState result;
  Bound() : G4HadronicInteraction("ClockBound") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile& p,G4Nucleus&) override {
    result.Clear();
    const_cast<G4HadProjectile&>(p).SetGlobalTime(waitTime);
    result.SetStatusChange(decayBranch ? stopAndKill : isAlive);
    if(decayBranch) {
      result.AddSecondary(new G4DynamicParticle(G4Gamma::Gamma(),{0,1,0},0.02*MeV));
      result.GetSecondary(0)->SetTime(waitTime);
    }
    return &result;
  }
};
struct Nuclear : G4HadronicInteraction {
  G4HadFinalState result;
  Nuclear() : G4HadronicInteraction("ClockNuclear") {
    // Fixture is a time probe, deliberately not a conserving nuclear model.
    SetEnergyMomentumCheckLevels(DBL_MAX,DBL_MAX);
    SetMinEnergy(0.); SetMaxEnergy(1.e10*MeV);
  }
  G4HadFinalState* ApplyYourself(const G4HadProjectile&,G4Nucleus&) override {
    result.Clear(); result.SetStatusChange(stopAndKill);
    decayBranch=false;
    result.AddSecondary(new G4DynamicParticle(G4Gamma::Gamma(),{0,0,1},0.03*MeV));
    result.GetSecondary(0)->SetTime(modelTime);
    return &result;
  }
};
struct ProbeAtomDecay : G4MuonicAtomDecay {
  ProbeAtomDecay() : G4MuonicAtomDecay(new Nuclear) {}
  void SetFixtureInteractionLengths(double value) { theNumberOfInteractionLengthLeft=value; }
};
}

int main(int argc,char** argv) {
  if(argc<4) { std::fprintf(stderr,"usage: clock_probe atomic|stopping|fixture|atom_decay t0_ns samples\n"); return 2; }
  const bool atomic=std::strcmp(argv[1],"atomic")==0;
  const bool fixture=std::strcmp(argv[1],"fixture")==0;
  const bool atomDecay=std::strcmp(argv[1],"atom_decay")==0;
  const double t0=std::atof(argv[2])*ns;
  const int samples=std::atoi(argv[3]);
  G4Random::setTheSeed(12345);
  auto* rm=G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Detector);
  auto* physics=new QBBC(0);
  if(atomic || atomDecay) {
    physics->ReplacePhysics(new G4StoppingPhysics("stopping",0,false));
    physics->RegisterPhysics(new AtomicPhysics);
    physics->RegisterPhysics(new G4MuonicAtomDecayPhysics);
  }
  rm->SetUserInitialization(physics); rm->Initialize();
  G4VProcess* process=nullptr;
  ProbeAtomDecay* decayProcess=nullptr;
  G4ParticleDefinition* primary=G4MuonMinus::MuonMinus();
  if(atomDecay) {
    primary=G4IonTable::GetIonTable()->GetMuonicAtom(82,208);
    decayProcess=new ProbeAtomDecay;
    process=decayProcess;
  } else if(atomic) {
    auto* list=G4MuonMinus::MuonMinus()->GetProcessManager()->GetProcessList();
    for(int i=0;i<list->size();++i)
      if(dynamic_cast<G4MuonMinusAtomicCapture*>((*list)[i])) process=(*list)[i];
  } else if(fixture) {
    auto* p=new G4HadronStoppingProcess("muMinusCaptureAtRest");
    p->SetEmCascade(new Cascade); p->SetBoundDecay(new Bound);
    p->RegisterMe(new Nuclear); p->SetEnergyMomentumCheckLevels(DBL_MAX,DBL_MAX);
    p->PreparePhysicsTable(*G4MuonMinus::MuonMinus());
    process=p;
  } else {
    process=G4ProcessTable::GetProcessTable()->FindProcess("muMinusCaptureAtRest",G4MuonMinus::MuonMinus());
    auto* stopping=dynamic_cast<G4HadronStoppingProcess*>(process);
    if(!stopping) return 4;
    stopping->SetBoundDecay(new SpyBound);
  }
  if(!process) return 3;
  // BeamOn normally performs this before transport. This probe calls DoIt
  // directly, so initialize the process model labels explicitly.
  if(!fixture && !atomDecay) process->PreparePhysicsTable(*G4MuonMinus::MuonMinus());
  G4Nucleus rateNucleus;
  std::printf("RATES stopping_capture %a stopping_decay %a atomic_capture %a atomic_decay %a\n",
    G4MuonMinusBoundDecay::GetMuonCaptureRate(82,208),
    G4MuonMinusBoundDecay::GetMuonDecayRate(82,208,G4MuonMinus::MuonMinus()->GetPDGMass(),rateNucleus.AtomicMass(208,82)),
    G4MuonicAtomHelper::GetMuonCaptureRate(82,208),G4MuonicAtomHelper::GetMuonDecayRate(82));
  auto* navigator=G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  navigator->LocateGlobalPointAndSetup({});
  const double waits[]={0.,0.25,1.,2.,123.};
  const double locals[]={-3.,-1.,-0.1,0.,0.125,3.};
  const int cases=atomDecay ? 5*6+samples : fixture ? 2*5*6+100 : samples;
  std::uint32_t generated=71819;
  // Reset after initialization: matched variants receive the same process inputs.
  G4Random::setTheSeed(12345);
  for(int i=0;i<cases;++i) {
    waitTime=waits[(i/6)%5]*ns; modelTime=locals[i%6]*ns; decayBranch=i>=30;
    if(atomic) { waitTime=0.; modelTime=0.; decayBranch=false; }
    if(fixture && i>=60) {
      // Generated finite witnesses independent of the physics random engine.
      generated=1664525u*generated+1013904223u;
      waitTime=(generated%100001u)/10000.*ns;
      generated=1664525u*generated+1013904223u;
      modelTime=(int(generated%60001u)-30000)/10000.*ns;
      decayBranch=(i%2)==0;
    }
    G4Track track(new G4DynamicParticle(primary,{0,0,1},0.),t0,{});
    track.SetTrackID(1); track.SetTrackStatus(fStopButAlive);
    track.SetTouchableHandle(navigator->CreateTouchableHistoryHandle());
    G4Step step; step.SetTrack(&track); track.SetStep(&step);
    step.GetPreStepPoint()->SetMaterial(material);
    if(atomDecay) {
      G4ForceCondition condition=NotForced;
      const double mean=decayProcess->GetMeanLifeTime(track,&condition);
      const double declared=waitTime;
      const bool controlled=i<30;
      if(controlled) G4Random::setTheSeed(12345);
      decayProcess->SetFixtureInteractionLengths(controlled ? declared/mean : -1.);
      // Observe the actual process waiting time independently of its products.
      waitTime=decayProcess->AtRestGetPhysicalInteractionLength(track,&condition);
      std::printf("WAIT %d mode %s declared %a mean %a observed %a\n",i,
        controlled ? "controlled" : "sampled",declared/ns,mean/ns,waitTime/ns);
      decayBranch=true; // the supplied nuclear interaction clears this on capture
    }
    auto* result=process->AtRestDoIt(track,step);
    std::printf("CASE %s %d t0 %a wait %a local %a branch %d count %d status %d edep %a\n",
      argv[1],i,t0/ns,waitTime/ns,modelTime/ns,decayBranch,result->GetNumberOfSecondaries(),
      int(result->GetTrackStatus()),result->GetLocalEnergyDeposit()/MeV);
    for(int j=0;j<result->GetNumberOfSecondaries();++j) {
      auto* t=result->GetSecondary(j); const auto p=t->GetMomentum();
      const auto name=G4PhysicsModelCatalog::GetModelNameFromID(t->GetCreatorModelID());
      const char* phase=atomDecay ? (decayBranch ? "decay" : "capture") :
        dynamic_cast<const G4MuonicAtom*>(t->GetDefinition()) ? "atom" : atomic ? "em" :
        name.find("EMCascade")!=std::string::npos ? "em" :
        name.find("NuclearCapture")!=std::string::npos ? "capture" :
        name.find("DIO")!=std::string::npos ? "decay" : "other";
      std::printf("SEC %d %d %s time %a energy %a px %a py %a pz %a weight %a model %d phase %s\n",
        i,j,t->GetDefinition()->GetParticleName().c_str(),t->GetGlobalTime()/ns,t->GetKineticEnergy()/MeV,
        p.x()/MeV,p.y()/MeV,p.z()/MeV,t->GetWeight(),t->GetCreatorModelID(),phase);
      delete t;
    }
    result->Clear();
    // Full engine state, without drawing a variate. Exact equality is required
    // across matched process calls; downstream transport is deliberately absent.
    std::ostringstream state; G4Random::getTheEngine()->put(state);
    std::string s=state.str(); for(char& c:s) if(c=='\n') c=' ';
    std::printf("RNG %d %s\n",i,s.c_str());
  }
  // Processes/interactions are registry owned. Normal run-manager teardown.
  delete rm;
}
