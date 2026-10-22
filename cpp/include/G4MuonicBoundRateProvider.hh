// G4MuonicBoundRateProvider -- one capture and decay rate pair for a negative muon in the 1s state of a
// muonic atom, given to both stopped-muon routes of Geant4 when an application asks for it.
//
// Geant4 forms the two rates of a bound negative muon in two places, with two decay prescriptions:
//
//   G4MuonMinusBoundDecay::ApplyYourself    draws the stopping route's delay and branch from
//                                           GetMuonCaptureRate(Z, A) and
//                                           GetMuonDecayRate(Z, A, muMass, nuclear mass);
//   G4MuonicAtomHelper::ConstructMuonicAtom stores the atomic route's lifetime and its DIO and
//                                           nuclear-capture lifetimes from GetMuonCaptureRate(Z, A)
//                                           and GetMuonDecayRate(Z), which G4MuonicAtomDecay reads.
//
// The two decay rates differ: below Z = 14 the first is free * [1 - (Z alpha)^2 (0.5 + 0.06 m_mu / M)],
// the comment beside it attributing the form to H. C. von Baeyer and D. Leiter, Phys. Rev. A 19, 1371
// (1979); the second is free * [1 - 2.5 (Zeff alpha)^2] for every Z but hydrogen, its comment
// attributing it to N. C. Mukhopadhyay, Phys. Rep. 30, 1 (1977). With the switch
// G4HadronicParameters::SetEnableMuonicBoundRates(true), both consumers take the pair this class
// returns instead, over the domain below:
//
//   capture  G4MuonicAtomHelper::GetMuonCaptureRate(Z, A), the helper's capture resolver as it stands
//            (its compiled-in table, its Goulard-Primakoff fallback and, when the G4MuonicData opt-in
//            is on, the dataset row it reads first), called, not copied;
//   decay    LowZDecayRate(Z, muMass, nuclearMass): the stopping route's expression below Z = 14, with
//            its literals and its order of operations.
//
// The domain is an implementation domain, not a statement of physical accuracy: 2 <= Z <= 13 and
// A >= Z. Hydrogen (whose atomic-route decay rate is a separate tabulated value), Z >= 14 (where the
// stopping route uses a different fit) and a natural-composition request (A = 0) are outside it:
// Rates returns kUnsupported and each consumer keeps the pair it formed itself. For a key inside the
// domain, a nuclear or muon mass that is not finite and positive, a capture rate that is negative or
// not finite, or a decay rate that is not finite and positive is refused with the fatal G4Exception
// G4MuonicBoundRates001; nothing is clipped. (A finite capture rate >= 0 and a finite decay rate > 0,
// which never exceeds the free rate, then have a finite, positive sum.)
//
// The switch is off by default and the G4MuonicData opt-in (SetEnableMuonicData) does not set it.
// With it off, Consult returns false and the consumers run their own code unchanged. The first
// Consult -- the first time either consumer forms a rate pair -- freezes the switch: a later attempt to
// change it raises the fatal G4Exception G4MuonicBoundRates002, because muonic atoms already
// constructed keep the lifetimes they were built with. Set it on the master thread before the physics
// list is built; worker threads only read it.
//
// The static legacy getters G4MuonMinusBoundDecay::GetMuonDecayRate and
// G4MuonicAtomHelper::GetMuonDecayRate are unchanged and are not affected by the switch: they report
// each route's own prescription, not the pair a consumer receives with the switch on.
//
// Rates are in Geant4's internal units (inverse time, ns^-1), masses in MeV.
#ifndef G4MUONICBOUNDRATEPROVIDER_HH
#define G4MUONICBOUNDRATEPROVIDER_HH

#include "G4Types.hh"

struct G4MuonicBoundRates {
  G4double capture = 0.;  // nuclear capture rate, ns^-1
  G4double decay = 0.;    // decay-in-orbit rate, ns^-1
};

class G4MuonicBoundRateProvider {
 public:
  enum class Status {
    kCommon,       // the pair is the common pair; the consumer uses it
    kDisabled,     // the switch is off; the consumer keeps its own pair
    kUnsupported,  // outside the domain; the consumer keeps its own pair
    kRefused       // a fatal G4MuonicBoundRates001 was raised; the consumer keeps its own pair
  };

  // The switch. Enable and Disable are called by G4HadronicParameters::SetEnableMuonicBoundRates
  // only; once a consumer has consulted the provider, a call that would change the state raises the
  // fatal G4MuonicBoundRates002 and leaves the state as it was.
  static void Enable();
  static void Disable();
  static G4bool IsEnabled();
  static G4bool IsFrozen();

  // Called by each consumer before it uses its pair: freezes the switch and returns whether it is on.
  static G4bool Consult();

  // Whether (Z, A) lies in the implementation domain: 2 <= Z <= 13 and A >= Z.
  static G4bool InDomain(G4int Z, G4int A);

  // The stopping route's decay rate below Z = 14, in ns^-1:
  //   free = 0.45517005 / microsecond; f = 1 - (Z alpha)^2 (0.5 + 0.06 muMass / nuclearMass); f * free,
  // grouped as G4MuonMinusBoundDecay::GetMuonDecayRate groups it. Standard library only.
  static G4double LowZDecayRate(G4int Z, G4double muonMass, G4double nuclearMass);

  // The pair for (Z, A): kCommon with `rates` set, or another status with `rates` untouched.
  static Status Rates(G4int Z, G4int A, G4double muonMass, G4double nuclearMass, G4MuonicBoundRates& rates);

  // What the pair is made of, for a consumer or an application that reports it.
  static const char* Prescription();
};

#endif
