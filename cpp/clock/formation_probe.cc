// Actual muon precompound wrapper with independent timing metadata inputs.
#include "G4Box.hh"
#include "G4Element.hh"
#include "G4Fragment.hh"
#include "G4Gamma.hh"
#include "G4HadProjectile.hh"
#include "G4Isotope.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuMinusCapturePrecompound.hh"
#include "G4MuonicAtomHelper.hh"
#include "G4Nucleus.hh"
#include "G4PVPlacement.hh"
#include "G4ReactionProduct.hh"
#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include "G4VPreCompoundModel.hh"
#include "G4VUserDetectorConstruction.hh"
#include "QBBC.hh"
#include "Randomize.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {
int caseID=0;
struct Detector : G4VUserDetectorConstruction {
  G4VPhysicalVolume* Construct() override {
    auto* element=new G4Element("Pb208","Pb",1);
    element->AddIsotope(new G4Isotope("Pb208",82,208),1.);
    auto* material=new G4Material("Pb208",11.35*g/cm3,1);
    material->AddElement(element,1.);
    auto* volume=new G4LogicalVolume(new G4Box("world",m,m,m),material,"world");
    return new G4PVPlacement(nullptr,{},volume,"world",nullptr,false,0);
  }
};
struct Controlled : G4VPreCompoundModel {
  int calls=0;
  Controlled() : G4VPreCompoundModel(nullptr,"ControlledFormationMetadata") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile&,G4Nucleus&) override { return nullptr; }
  void DeExciteModelDescription(std::ostream&) const override {}
  G4ReactionProductVector* DeExcite(G4Fragment& fragment) override {
    ++calls;
    std::printf("FRAGMENT %d creation %a\n",caseID,fragment.GetCreationTime()/ns);
    auto* products=new G4ReactionProductVector;
    const double tofs[]={-1.,0.,1.,7.};
    const double formations[]={-2.,0.,0.125,3.};
    for(double tof:tofs) for(double formation:formations) {
      const double energy=(products->size()+1)*0.001*MeV;
      auto* rp=new G4ReactionProduct(G4Gamma::Gamma());
      rp->SetMomentum({0.,0.,energy}); rp->SetKineticEnergy(energy);
      rp->SetTOF(tof*ns); rp->SetFormationTime(formation*ns);
      std::printf("META %d %zu name gamma tof %a formation %a\n",caseID,products->size(),
        rp->GetTOF()/ns,rp->GetFormationTime()/ns);
      products->push_back(rp);
    }
    return products;
  }
};
}

int main(int argc,char** argv) {
  if(argc!=4) { std::fprintf(stderr,"usage: formation_probe controlled|default|light origin_ns calls\n"); return 2; }
  const bool stock=std::strcmp(argv[1],"default")==0;
  const bool light=std::strcmp(argv[1],"light")==0;
  const double origin=std::atof(argv[2])*ns;
  const int count=light ? 4 : std::atoi(argv[3]);
  auto* rm=G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Detector); rm->SetUserInitialization(new QBBC(0)); rm->Initialize();
  auto* controlled=stock ? nullptr : new Controlled;
  auto* model=new G4MuMinusCapturePrecompound(controlled);
  G4Random::setTheSeed(12345);
  const int lightZ[]={1,1,2,2}, lightA[]={1,2,3,4};
  for(caseID=0;caseID<count;++caseID) {
    const int z=light ? lightZ[caseID] : 82, a=light ? lightA[caseID] : 208;
    G4DynamicParticle particle(G4MuonMinus::MuonMinus(),{0,0,1},0.);
    G4HadProjectile projectile(particle);
    projectile.SetGlobalTime(origin); projectile.SetBoundEnergy(G4MuonicAtomHelper::GetKShellEnergy(z));
    G4Nucleus nucleus; nucleus.SetParameters(a,z);
    const int before=controlled ? controlled->calls : 0;
    std::printf("INPUT %d mode %s origin %a Z %d A %d\n",caseID,argv[1],origin/ns,z,a);
    auto* result=model->ApplyYourself(projectile,nucleus);
    std::printf("FINAL %d count %zu status %d edep %a dependency_calls %d\n",caseID,
      result->GetNumberOfSecondaries(),int(result->GetStatusChange()),result->GetLocalEnergyDeposit()/MeV,
      controlled ? controlled->calls-before : -1);
    for(std::size_t j=0;j<result->GetNumberOfSecondaries();++j) {
      auto* secondary=result->GetSecondary(j); const auto* dp=secondary->GetParticle(); const auto p=dp->GetMomentum();
      std::printf("SECONDARY %d %zu name %s time %a energy %a px %a py %a pz %a weight %a\n",caseID,j,
        dp->GetDefinition()->GetParticleName().c_str(),secondary->GetTime()/ns,dp->GetKineticEnergy()/MeV,
        p.x()/MeV,p.y()/MeV,p.z()/MeV,secondary->GetWeight());
    }
    result->Clear();
    std::ostringstream state; G4Random::getTheEngine()->put(state);
    std::string text=state.str(); for(char& c:text) if(c=='\n') c=' ';
    std::printf("ENGINE %d %s\n",caseID,text.c_str());
  }
  delete rm;
}
