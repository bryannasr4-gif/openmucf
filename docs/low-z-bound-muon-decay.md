# Leading inclusive recoil in low-Z bound-muon decay

The experimental common-rate provider uses

\[
 Q_{\rm lin}=1-\frac{z^2}{2}+z^2\eta,\qquad
 z=Z\alpha,\quad \eta=m/M.
\]

This is a leading-binding, linear-recoil tree prescription. It is not an
all-orders reduced-mass width. Its implementation domain, `2 <= Z <= 13`,
`A >= Z`, finite positive masses and `eta < 1/2`, is a consistency domain;
physical use of the expansion requires `eta << 1`. No numerical physical
uncertainty or real-isotope accuracy has been established. Higher binding,
higher recoil, radiative and nuclear effects remain unassigned.

The stable prescription identifier is `low-z-inclusive-tree-linear-recoil-v1`.
The default-off switch, capture resolver and legacy getters are unchanged.
Historical `bound_rates_*` policies and harvests describe the former getter
prescription. Their oracle equality and numerical results do not verify this
new formula. `openmucf/g4/d1_linear_recoil.py` is the current deterministic
evaluator; `d1_boundrates.Reference` retains its historical meaning.

## Observable and normalization

Use units with hbar=c=1. A negative muon of mass m occupies an unpolarized
Coulomb 1s state with a point scalar nucleus of mass M and charge +Ze. The atom
is at rest. The nucleus has no weak interaction, and there are no occupied
electron states. The observable is the loss width through `mu -> e nu nubar`,
summed over spectator momentum, both neutrinos and the **entire** allowed
positive-energy daughter-electron spectrum, including bound states. There is
no electron-energy cut. Nuclear capture is a separate channel.

The observable is photon inclusive. Here tree means zeroth order in the
additional radiative QED expansion: Coulomb binding is resummed, and source
current/recoil terms at the stated binding order are retained. Real radiative
decay and its virtual partners occur at the next radiative order. Their
hard-photon/soft-electron region is outside this tree argument.

Normalize to the free tree width with the same daughter-electron mass:

\[
 \Gamma_f={G_F^2m^5\over192\pi^3}f(r^2),\quad r=m_e/m,
 \qquad f(x)=1-8x+8x^3-x^4-12x^2\log x.
\]

The coefficient sought is quadratic in z and linear in eta about eta=0.
The fixed-r limit assumes r<1 separated from vanishing energy release. A
second hierarchy is r=kappa*z with bounded kappa, including r=0. Statements
about these asymptotic coefficients are not finite-parameter error bounds.

The implementation multiplies Q_lin by the existing numerical free-rate
normalization `0.45517005 / microsecond`; it does not recalculate that constant
from G_F or claim that multiplying it supplies radiative bound-state matching.

## Off-shell cut, spectator pole and Ward partner

For a massless daughter the unpolarized weak absorptive self-energy is
proportional to `slash(p)*(p^2)^2`. With daughter mass it has the factor
`(p^2)^2*f(m_e^2/p^2)`. Let

\[
 a={d\log\Gamma_f(m)\over d\log m}
   =5-2r^2{f'(r^2)\over f(r^2)}.
\]

Expanding the weak cut about the muon mass shell produces a redundant term
`a*(slash(p)-m)/m`, in addition to the number and kinetic terms. To see its
normalization explicitly at r=0, write p^0=m+omega. Through the required order,

```
(p^2)^2/m^4 = 1 + 4*omega/m - 2*p_vec^2/m^2,
(u_dagger*gamma^0*slash(p)*u)/(m*u_dagger*u)
             = 1 + omega/m - p_vec^2/m^2.
```

The product gives coefficients 5 and -3. On a free moving mass shell,
omega=p_vec^2/(2m), so the width is `1-p_vec^2/(2m^2)`, fixing the
state-normalization convention.

In the atom put the uncut nucleus on its positive-energy pole. With
`mu=m*M/(m+M)` and `E_bind=-mu*z^2/2`,

```
omega = E_bind - p_N^2/(2*M),    p_N = -p_mu,
omega*psi = [p_mu^2/(2*m) + V_C]*psi,    V_C=-z/R.
```

The second equation follows from the two-body Schrodinger equation. Omitting
the spectator kinetic energy while retaining E_bind would violate it. The
off-shell cut is therefore

\[
 Q_{\rm off}=1-{\langle p_\mu^2\rangle\over2m^2}
                 +a{\langle V_C\rangle\over m}.
\]

Attach the source photon to every charged segment of the same weak forward
amplitude. Its electromagnetic vertex obeys

\[
 q_\nu\Lambda^\nu(p+q,p)=e_\mu[\Sigma(p+q)-\Sigma(p)].
\]

Gauge completion pairs the off-shell term with `a*(i slash(D)-m)`; a field
redefinition removes this equation-of-motion operator. Its scalar binding
insertion gives `delta Q_vertex=-a*<V_C>/m`. The cancellation is fixed by
matching, not a choice of unrelated corrections. It applies to a normalized
soft wavefunction and the same finite-mass spectator pole. Thus

\[
 Q_{\rm local}=1-\langle p_\mu^2\rangle/(2m^2).
\]

For r=0 an explicit linear-eta audit is

```
<p^2>/m^2 = z^2*(1-2*eta) + O(z^2*eta^2),
<V_C>/m   = -z^2*(1-eta) + O(z^2*eta^2),
<omega>/m = -z^2/2 + O(z^2*eta^2),
Q_off     = 1 - 11*z^2/2 + 6*eta*z^2,
delta Q_vertex = 5*z^2 - 5*eta*z^2,
Q_total   = 1 - z^2/2 + eta*z^2.
```

The `+5*z^2` term is static; its recoil slope is -5. The finite-mass extension
here must also exclude a promoted nonlocal final-electron contribution.

## Complete cuts and hard endpoints

After integrating the two free neutrinos while retaining their total momentum
K, insertion of complete final electron+nucleus states gives

\[
 \Gamma=\sum_{ab}\int d^3K\,
 \langle i|J_a(K)^\dagger
 F_{ab,K}(E_i-H_f(K))J_b(K)|i\rangle.
\]

Here E_i=m+E_bind and H_f has the nuclear rest mass subtracted. Its free form
is `E_e(p)+(-K-p)^2/(2M)`, so the spectator momentum is not replaced by the
electron momentum. Its full spectral measure includes allowed bound states.
The neutrino tensor F is polynomial in its energy and K times the light-cone
step function. The current matrices remain in J: the functional calculus acts
on each scalar F component and does not commute matrices through H_f.
Negative-energy virtual states in a Dirac representation belong in matching,
not in the physical positive-energy final-state sum.

Set `G(E)=F(E_i-E)`. For a scalar or momentum-dependent potential matrix
insertion,

```
delta G(H)_ab = V_ab * [G(E_a)-G(E_b)]/(E_a-E_b),
[G(E_a)-G(E_b)]/(E_a-E_b) = integral_0^1 dt G'(E_b+t*(E_a-E_b)).
```

The exact energy difference, including spectator recoil, is

```
Delta = E_e(p+q)-E_e(p) + [(P_N-q)^2-P_N^2]/(2*M).
```

Both cut placements have this same Delta. The underlying identity is

```
Im[1/(x+i0)/(x+Delta+i0)]
  = -pi*[delta(x)-delta(x+Delta)]/Delta.
```

At coincidence it is a delta derivative, not an uncancelled eikonal pole.
Sum the cuts before expanding q^0, q_parallel or 1/M. Recoil differentiation
of the hard smooth weight gives its next derivative, not a squared pole.
In particular `q^0 ~ (mu*z)^2/M` supplies no inverse atomic scale.

The light-cone theta derivatives are endpoint delta distributions and must
be retained. A sharp-boundary divided difference can be of size 1/Delta
pointwise in a layer of thickness Delta; its integrated limit is finite.
The hard upper electron endpoint has a regular phase-space Jacobian. Keeping
its shifted boundary yields that distributional limit. This is not a
pointwise smoothness claim about a theta function. The soft-electron endpoint
requires separate power counting below.

Higher potential insertions yield higher divided differences; coincident
limits are derivatives divided by factorials, with simplex integral
representations. The complete cut sum removes repeated scattering poles.
One hard-region scalar insertion scales as `<V>/m ~ z^2`; two as
`<V^2>/m^2 ~ z^4`, finite in the Coulomb 1s state. Initial Coulomb ladders are
already in psi. This argument uses the full inclusive spectral measure; it
does not apply to continuum-only widths or spectra with cuts.

## Soft electrons, source currents and operator scope

For a relativistic soft daughter with E_e of order k=mu*z, the tree spectrum
has `dGamma/dE_e = O(Gamma_f*E_e^2/m^3)`. A width-k slice is O(z^3).
The neutrino tensor is hard and regular: two neutrinos can carry energy of
order m with nearly cancelling momenta. For r=kappa*z, the rescaled soft
problem has finite functions of bounded kappa; `V/E_e=O(z)` and
`(k^2/M)/E_e=O(eta*z)`. Neither Coulomb response nor recoil promotes it to
eta*z^2. A wavefunction change is relative O(eta).

At fixed positive r, `p_e ~ k` is nonrelativistic. The measure remains
`d^3p_e=O(k^3)`, the Coulomb parameter z/v_e is finite as z tends to zero,
and recoil divided by electron kinetic energy is O(m_e/M). The narrower
threshold region `p_e ~ m_e*z` can require nonperturbative Coulomb exchange,
but its full spectral weight is still O((m_e*z)^3), beyond z^2.

Allowed electron bound states have local density of order `(mu_eN*z)^3`.
Integration over neutrino total momentum uses Parseval on the orbital overlap;
the s-wave level sum converges as `sum n^-3`, with extra overlap factors for
other partial waves. Their total scale is at most O(r^3*z^3) times functions
of r; for r=kappa*z it is O(z^6). Recoil supplies no inverse z. At r=0 the
subcritical point-Coulomb problem has no massive atomic electron bound
spectrum. These separate hierarchies imply no uniform numerical remainder.

For the heavy scalar source use the canonical Lagrangian
`N_dagger*(iD_t+D^2/(2M))*N`. The irreducible absorptive operator enumeration,
after subtracting Coulomb iterations and redundant terms, is:

| Class | Contribution through the requested order |
|---|---|
| Muon number | Coefficient Gamma_f; expectation exactly one. Disconnected spectator kinetic insertions cancel state normalization. |
| Muon time derivative | Redundant; the cut/Ward cancellation above removes its independent imaginary A0 coefficient. |
| Muon kinetic term | Surviving scalar at dimension five, coefficient -1/2 fixed by the free moving width. |
| Source A0 current | `(P'+P)^0/(2*sqrt(E_P'*E_P))` has no linear-1/M correction. Kinetic recoil is already in spectator poles and spectral energies. |
| Source spatial current | `Ze*(P'+P)/(2M)` is O(eta*z). Before inclusive matching a hard electron current gives the weaker bound `delta H/m=O(eta*z^3)`; no eikonal promotion survives. The matched spin-independent muon current brings another velocity, yielding O(eta*z^4). |
| Source seagull | Canonical reduction cancels the apparent A0 squared term against antiparticle/normalization terms. Check `E=Ze*A0+sqrt(M^2+(P-Ze*A)^2)`: no A0 squared through 1/M. The surviving transverse A squared term requires extra transverse couplings. |
| Source state normalization | Relativistic factors start at P_N squared/M squared, hence eta squared. Other reductions must transform their kernels and normalization together. |
| Magnetic/spin-orbit | Scalar source has no intrinsic spin; unpolarized 1s removes leading muon-spin/orbital terms. Even without those zeros, the moving-source field is O(eta*z^4) in the width. |
| Electric gradient/Darwin | `<Laplacian V>/m^3=4*(mu/m)^3*z^4`; the light-electron threshold was counted separately. |
| Higher derivatives/two fields | p^4/m^4 and V squared/m squared are O(z^4). UV contact pieces require matching, not divergent matrix-element estimates. |
| Muon-nucleus contact | No source weak interaction. A connected hard EM remainder needs at least one factor z and a pair density of order k cubed: O(z^4), with first recoil correction O(eta*z^4). |
| Occupied-electron weak contact | Normal-ordered initial expectation vanishes; there is no spectator electron. It cannot replace the final soft-electron calculation. |
| Hard source transfer | Resolves separation 1/m and belongs to the density-suppressed contact/derivative terms. |
| Potential/ultrasoft source transfer | Included in canonical source dynamics, Ward pairing and complete cut differences, with the light-cone boundary retained. |

Finite nuclear size and intrinsic nuclear spin are outside this point-scalar
model. Their corrections, nuclear structure, occupied electrons and radiative
matching have not been assigned numerical errors. A local operator list alone
would not suffice without the preceding nonlocal and endpoint analysis.

## Applying the leading prescription to nuclear inputs

The central prescription uses a nuclear rest-mass parameter. Its interpretation
is a leading Coulomb model, rather than a complete prediction for a particular
isotope or material. For a localized nucleus, take its radius, electromagnetic
moments and excitation gaps fixed in the small-z expansion, with k*R_C small.

An arbitrary-spin nuclear Hamiltonian has charge and kinetic terms and a
magnetic term proportional to `g_N*Ze*I.B/(2M)`. Here that equation defines the
magnetic-factor convention. The charge monopole and kinetic term do not depend
on nuclear spin. The absorptive muon magnetic operator is allowed, but its
nuclear matrix element scales as `e*B_N/m^2 = O(g_N*eta*z^4)`, including its
contact part. Hyperfine radial changes likewise start beyond quadratic binding
order. Different hyperfine populations can strongly affect nuclear capture;
this decay argument does not evaluate that channel.

For a finite charge distribution,
`F_C(q^2)=1-q^2*R_C^2/6+...` gives
`delta V=(2*pi/3)*z*R_C^2*delta^3(R)`. Its expectation is of order
`mu^3*z^4*R_C^2`. It therefore changes the kinetic matrix element beyond the
matched z-squared recoil order. These are formal counts at fixed nuclear
parameters, not numerical error bounds. A higher binding power can be as large
as or larger than the recoil term at a physical isotope. Correcting the leading
coefficient does not establish a smaller total physical error.

The arbitrary-spin operator basis in [Eides and Martin, equations 1-3
(2010)](https://arxiv.org/pdf/1005.5737v2), and the independent magnetic and
charge-radius coefficients in [Paz, equations 2-4
(2015)](https://arxiv.org/pdf/1503.07216), supply the operator framework. Neither
paper is cited as a calculation of this muon-decay coefficient; the binding
power counts here are an application of that framework.

This model still assumes no initially occupied electrons. Screening can change
the muon kinetic state; its size and mass-ratio hierarchy are not evaluated
here. No assertion that every electron-configuration correction has a higher
power of z is needed or made. Nuclear structure, radiative effects and other
omitted terms remain without numerical uncertainty assignments.

## Independent first-/second-Born asymptotic check

A direct massless-electron check can keep k independent of z and m until the
last step. This checks the final-electron Coulomb coefficient without using
the Ward cancellation to obtain it. Define

```
phi_k(P)=8*sqrt(pi)*k^(5/2)/(P^2+k^2)^2,
Psi_k,s(P)=phi_k(P)*(chi_s; sigma.P/(2*m)*chi_s)/sqrt(1+k^2/(4*m^2)),
V_lambda(q)=-4*pi*z/(q^2+lambda^2),
R(E;p)=(E-alpha.p+i0)^(-1), A=R_dagger, S0=(A-R)/(2*pi*i),
S1=R*V*S0+S0*V*A,
S2=R*V*R*V*S0+R*V*S0*V*A+S0*V*A*V*A.
```

The upper profile has norm one with measure d^3P/(2pi)^3. Its canonical
lower component is retained. The complete source is normalized; its radial
density is `k^3*exp(-2*k*R)/pi`. The three terms in S2 include both
second-Born interference terms as well as the first-Born square.

For `B^a=gamma^0*gamma^a*(1-gamma5)`, take
`J_a,s(K;P)=B_a*Psi_k,s(P+K)`. With metric (+---), the integrated neutrino
tensor is

```
T_ab(Q)=(Q_a*Q_b-Q^2*g_ab)/(3*pi)*theta(Q^0)*theta(Q^2),
Gamma_n=(G_F^2/2)*integral_K integral_0^m dE T_ab((m-E,K))
             * (1/2)*sum_s <J_b,s(K)|S_n(E)|J_a,s(K)>.
```

All momentum integrals use d^3p/(2pi)^3. Resolvents retain both energy signs;
the physical cut has E>0. These equations specify the finite-parameter
integrals, which are **not evaluated here**.

The direct local trace uses a rest spin-averaged muon and an electron along
the third axis. With `rho_mu=diag(1/2,1/2,0,0)` and
`rho_e=(1+alpha^3)/2`, matrix multiplication gives

```
3*pi*Tr[rho_e*B^a*rho_mu*B^(b)dagger]*T_ab = W*(3*W-4*p),
integral_0^(W/2) dp p^2*W*(3*W-4*p) = W^5/16.
```

Its derivative is `3*W^4/16` from the bulk plus `2*W^4/16` from the moving
endpoint, giving 5. The first local insertion is `-5*<V_lambda>/m`.
The second local insertion is `10*<V_lambda^2>/m^2`; it alone is insufficient
to justify an interchange of limits. The following bounds control that step.

Write electron momenta `p_j=P_j-K`, energies `e_j=|p_j|`, signed nodes
`a_j=sigma_j*e_j`, and projectors `(1+sigma_j*alpha.p_j/e_j)/2`.
For n=1,2 the exact complete cut E integral is

```
sum_j F_ab(a_j,K)/product_(l!=j)(a_j-a_l),
F_ab(E,K)=T_ab((m-E,K))*theta(E).
```

Projector products keep their order; partial fractions act only on scalar
denominators. Coincident positive nodes give the first derivative on a segment
or second derivative on the unit simplex (area 1/2). F' contains the upper
endpoint delta; F'' contains its derivative. Mixed signs retain the E=0
projection discontinuity rather than assuming smoothness there.

Useful exact atomic estimates, uniform in lambda>=0, are

```
C_k(q)=integral_P phi(P)*phi(P+q)=16*k^4/(q^2+4*k^2)^2,
integral_q |V_lambda(q)|*C_k(q) <= z*k,
integral_q |q|*|V_lambda(q)|*C_k(q) <= 4*z*k^2/pi,
||V_lambda*phi||_2 <= sqrt(2)*z*k,    ||p*phi||_2=k,
<V_lambda>=-4*z*k^3/(2*k+lambda)^2,
<V_lambda^2>=2*z^2*k^3/(k+lambda) <= 2*z^2*k^2,
||p^2*phi||_2=sqrt(5)*k^2.
```

Choose a fixed L=delta*m, with sufficiently small constant delta independent
of k,z,lambda. For external |P_j|<L the integrated first-Born kernel has one
bounded momentum derivative: its soft electron singularities are at worst
1/e_j squared, integrable in three dimensions. Near the hard upper endpoint,
`m-|K|-[(1-t)*e_0+t*e_1]` has a nonzero radial derivative. The kernel differs
from its zero-momentum value by at most `C_delta*m^3*(|P_0|+|P_1|)`.
Cauchy on the unmodulated profile bounds the first remainder by
`C_delta*z*k^2/m^2`. It also bounds lower-component insertions. No false
estimate of `||p*exp(-i*K.R)*phi||` by k is used.

At second order the endpoint is
`f=m-r-sum_j t_j*e_j`, K=r*n, with radial coarea factor
`D=1+sum_j t_j*(r-n.P_j)/e_j>=0`. For D bounded below, radial integration
of delta and delta prime costs only hard scales. When D is small, select a
node with t_j>=1/3 and group nodes linked by transfers below a fixed hard
scale. Its electron direction is close to n. Translate that entire hard
cluster along a fixed cone direction. Then `|df/du|>=c/3`; all internal soft
transfers are fixed, all crossing edges are hard, and every translated source
endpoint has hard P. Derivatives cost 1/L or 1/m, never 1/k or 1/lambda.

Use overlapping cutoffs with thresholds L and 2L and enlarged translation
intervals of length cL. Apply `|w'|<=C*w/L` to the **unpartitioned** positive
envelope `w=phi(P_0)*|V_01|*|V_12|*phi(P_2)`, while bounding cutoff derivatives
absolutely by C/L. The cutoff's logarithmic derivative need not be bounded.
On each enlarged interval,
`w(u_endpoint)<=exp(C*c)/(c*L)*integral du w(u)`.
This trace-to-volume estimate survives the one integration by parts for delta
prime; matrix/tensor factors are bounded in norm, not assumed positive.
Finite edge patterns and angular cones give a finite overlapping cover.

For soft external sources and a backward hard internal node, set
`s=t_0+t_2`, `a=1+de_1/dr` and `b=1-n.P_1/|P_1|`. Solid angle is proportional
to db, and
`a*(2-a)=(|P_1|/e_1)^2*b*(2-b)` makes a,b uniformly comparable on the hard
backward chart. Thus D>=c(s+b), and the worst majorant integrates as
`integral_0^1 ds integral_0^1 db s/(s+b)^2=log(2)`.
This uses a geometric majorant, not an angular Jacobian evaluated after an
angle-dependent endpoint root.

Mixed-sign cut denominators are energy sums. At first order they are bounded
by C*m squared/(e_i+e_j); at second order by products with worst behavior
C*m squared/(e_i*e_l). Their K integrals are locally finite by Cauchy and
`integral_ball d^3K/|K-P|^2<=C*R`. Separated hard energies are first split by
partial fractions; the close pair is only a first divided difference. Use a
finite multiscale gap partition at `delta*m`, `delta*m/3` and `delta*m/9`:
three nodes yield either an all-close group or a separated group at one of
these fixed hard fractions. This avoids assuming that one close/separated
threshold is a dichotomy; it changes partition constants rather than powers
of k. If all
momenta are soft the upper endpoint is inactive and its polynomial is regular.

Soft electrons with hard K require a separate carrier chart. First separate
hard energies so that the endpoint interpolation involves only the soft
cluster. Then translate K and all P_j of that cluster together: every soft
p_j, signed denominator, projector and internal transfer stays fixed, while
the endpoint derivative from |K| is bounded away from zero on a cone. Hard
nodes appear only in smooth separated denominators. Without this separation
the all-node simplex would not have a unit Jacobian. Source endpoints in this
cluster are hard tails; crossing edges are hard. No soft derivative is taken.

Overlapping infrared subclusters are integrable: for two insertions three
soft electron vectors have dimension nine, Coulomb degree -4 and signed
energy degree at worst -2, giving net +3. Two soft vectors have dimension six,
at most one internal Coulomb edge and even allowing two energy denominators
have net at least +2. One soft vector has at worst 1/e squared; a soft transfer
alone has 1/q squared. Two soft transfers have degree -4 in six dimensions.
The first-order cases are weaker. This checks proper subclusters as well as
the simultaneous infrared limit. Two hard-source endpoints supply k^5.

Finally the exact tails obey

```
integral_(q>L) |V_lambda|*C_k <= 32*z*k^4/(3*pi*L^3),
||1_(P>L)*phi||_2^2 <= 32*k^5/(5*pi*L^5).
```

For two nearly cancelling hard transfers and |Q|<L/2,
`integral_(q>L) (4*pi*z/q^2)*(4*pi*z/|Q-q|^2) <=32*z^2/L`.
Multiplication by `(integral phi)^2=k^3/pi` supplies further suppression.
The full unscreened convolution is `2*pi^2*z^2/|Q|`, whose source average is
`2*z^2*k^2`. Lower components are bounded by the norms above and Hardy on
unmodulated profiles: `||V_lambda*|p|phi||<=2*z*||p^2 phi||`. Their second-order
weights are O(z squared*k cubed/m) and O(z squared*k^4/m squared). No p cubed
norm or differentiated soft Coulomb factor is required.

These chart and tail bounds give, with unspecified finite constants,

```
Gamma_1/Gamma_f = 5*z*k/m + O(z*k^2/m^2) + o(z*k/m),
Gamma_2/Gamma_f = O(z^2*k^2/m^2).
```

The limits are ordered: sum complete R/A cuts and take their physical
resolvent boundary value at fixed k>0 and lambda>0; remove screening with
lambda/k tending to zero using the uniform bounds; then let z tend to zero
with k/m=xi*z for fixed positive finite xi. Only afterwards set k=mu*z.
The first term becomes `5*z^2*(mu/m)` with slope `-5*z^2` in eta, while the
second is o(z squared). This is a first-/second-Born asymptotic check with a
specified canonical source and massless electron. It supplies no evaluated
finite-z integral, numerical remainder, higher-Born bound or finite-eta theorem.

This calculation independently checks the static final-electron Coulomb vertex
ingredient. It does not independently derive the complete +1 recoil coefficient.
Using the matching relation k=mu*z gives its contribution
`5*z^2-5*eta*z^2`; the separate off-shell/spectator calculation gives the
`+6*eta*z^2` term. Their combination, rather than the Born calculation alone,
gives the linear recoil result below.

## Result and comparison with primary literature

The inclusive absorptive Hamiltonian through the matched orders is

\[
 H_{\rm abs}=-{i\Gamma_f\over2}\int d^3x\,
 h^\dagger[1-\boldsymbol\pi_\mu^2/(2m^2)]h+
 \text{terms beyond these orders}.
\]

The Coulomb expectation gives

\[
 Q_{\rm tree}=1-z^2/2+z^2\eta+O(z^2\eta^2)
              +\text{binding terms beyond quadratic order}.
\]

The exact kinetic expectation `1-z^2/[2*(1+eta)^2]` is a useful intermediate
identity, but the present matching establishes only its linear eta slope in
the complete width. Other order-z squared*eta squared terms have not been
matched. Conservative region counts allow binding remainders beginning beyond
quadratic order, with logarithms; no finite coefficient bound is assigned.
This is an EFT matching and asymptotic argument, not a mathematical
finite-error certificate. It does not establish a completed physical model.

[Czarnecki, Lepage and Marciano, *Muonium Decay*, Appendix A, equations
A1-A15 (1999)](https://arxiv.org/pdf/hep-ph/9908439v3) provide the off-shell/
vertex Ward pairing, the nonrelativistic operator structure, moving-muon
matching and number normalization. The finite-heavy-scalar reduction,
inclusive spectral-region analysis and linear finite-M conclusion above are
the derivation here; that paper is not cited as an explicit finite-M
adjudication. The daughter-mass dependence cancels through a(r) within the
specified hierarchies, rather than being silently discarded.

[Von Baeyer and Leiter, *Physical Review A* **19**, 1371 (1979), equations
27-30](https://doi.org/10.1103/PhysRevA.19.1371) concern the same defined
spin-independent point-charge tree observable in the massless-electron case.
Their final recoil coefficient is -0.06, contrary to the +1 obtained here.
Their +5*z squared term is static. Their displayed modified initial state
and final recoil result do not expose the dynamical integral sufficiently to
identify a particular arithmetic or approximation error. No located erratum
or reconstruction of their unprinted working is claimed. The algebraic
change from that legacy formula is `delta Q=1.06*z^2*eta`; this is a model
difference, not an uncertainty estimate.
