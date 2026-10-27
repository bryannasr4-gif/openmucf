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
//   decay    LowZDecayRate(Z, muMass, nuclearMass): free * [1 - (Z alpha)^2 (1/2 - eta)],
//            eta = muMass/nuclearMass. This is leading binding and linear eta in the fully inclusive
//            tree width of an unpolarized Coulomb 1s muon and a point spinless spectator. The finite-M
//            derivation and independent Born asymptotic check are in docs/low-z-bound-muon-decay.md.
//            Higher recoil/binding, radiative, nuclear-size and spin effects have no assigned error.
//            It is not an exact reduced-mass width or a physical-accuracy certificate.
//
// The domain is an implementation domain, not a statement of physical accuracy: 2 <= Z <= 13 and
// A >= Z. Hydrogen (whose atomic-route decay rate is a separate tabulated value), Z >= 14 (where the
// stopping route uses a different fit) and a natural-composition request (A = 0) are outside it:
// Rates returns kUnsupported and each consumer keeps the pair it formed itself. For a key inside the
// domain, a nuclear or muon mass that is not finite and positive, a capture rate that is negative or
// not finite, eta not finite or >= 1/2, or a decay rate outside 0 < decay <= free is refused with
// the fatal G4Exception
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
// Evaluated pair rows. With the G4MuonicData opt-in on and G4MUONICDATA_D1_PAIR_PROFILE naming a profile
// that carries a `bound_rate_primitives` table (G4MuonicDataOverlay), a key that table holds is answered
// from its one tuple (tau, q, free_tau): decay = q / free_tau, capture = 1/tau - decay, so the two rates of
// both routes come from one realization and capture + decay = 1/tau. The row takes precedence over the
// formula above and is not limited to its domain. Other keys retain the selected capture profile and the
// existing route-specific formula or fallback. A row whose tuple is not finite and positive, has q outside
// (0, 1], or whose decay rate, 1/tau or capture rate is not finite or out of domain (a decay rate that
// underflows to zero, a negative capture rate) is refused with G4MuonicBoundRates001; the provider is then
// poisoned: should a handler suppress the abort, that call and every later consultation throw, so no consumer
// falls back to its own pair. The decay rate is checked against the row's own 1/free_tau, not the formula's
// free normalization. A pair profile with the switch off is refused at the
// first consultation with the fatal G4MuonicBoundRates003, whichever of the two options was set first;
// should a handler suppress the abort, every later consultation throws. Describe(Z, A) names, per key, the
// capture profile, the pair profile, the answering row or formula and its decay-factor convention.
//
// The static legacy getters G4MuonMinusBoundDecay::GetMuonDecayRate and
// G4MuonicAtomHelper::GetMuonDecayRate are unchanged and are not affected by the switch: they report
// each route's own prescription, not the pair a consumer receives with the switch on.
//
// Rates are in Geant4's internal units (inverse time, ns^-1), masses in MeV.
#ifndef G4MUONICBOUNDRATEPROVIDER_HH
#define G4MUONICBOUNDRATEPROVIDER_HH

#include "G4Types.hh"

#include <string>

struct G4MuonicBoundRates {
  G4double capture = 0.;  // nuclear capture rate, ns^-1
  G4double decay = 0.;    // decay-in-orbit rate, ns^-1
};

class G4MuonicBoundRateProvider {
 public:
  enum class Status {
    kCommon,       // the consumer uses `rates`: an evaluated pair row, or else the common formula pair
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

  // Called by each consumer before it uses its pair: freezes the switch and returns whether it is on. With
  // the switch off and a pair profile selected it refuses (G4MuonicBoundRates003).
  static G4bool Consult();

  // Whether (Z, A) lies in the implementation domain: 2 <= Z <= 13 and A >= Z.
  static G4bool InDomain(G4int Z, G4int A);

  // Checked current prescription in ns^-1, standard library only:
  //   free = 0.45517005 / microsecond; Q = 1 - (Z alpha)^2 * (1/2 - eta).
  // Unsupported Z, nonpositive/nonfinite masses, nonfinite eta or eta >= 1/2 return quiet_NaN.
  // Underflow of a positive mass ratio to zero gives the static limit. The eta guard enforces
  // suppression consistency only; the physical expansion requires eta << 1 and has no error bound here.
  static G4double LowZDecayRate(G4int Z, G4double muonMass, G4double nuclearMass);

  // The pair for (Z, A): kCommon with `rates` set, or another status with `rates` untouched.
  static Status Rates(G4int Z, G4int A, G4double muonMass, G4double nuclearMass, G4MuonicBoundRates& rates);

  // What the pair is made of, for a consumer or an application that reports it.
  static const char* Prescription();

  // One line for (Z, A): switch, data option, capture profile, pair profile, and which answer a consumer
  // receives (pair row, common formula or its own pair) with its decay-factor convention. Changes no state.
  static std::string Describe(G4int Z, G4int A);
};

#endif
