# Harvest drivers — proving the text is what the binary does

Vendoring `G4MuonMinusBoundDecay.cc` proves the D1 dataset matches Geant4's **source**. It does not
prove the compiler turned those decimal literals into the doubles we think it did, nor that our
Python reference implementation reproduces the compiled function. C++ does not even *require* a
correctly-rounded decimal literal ([lex.fcon]).

These two drivers close that gap by measurement, once. The measurement is then committed as
`data/g4/d1/d1_gp_sweep.oracle`, so CI can check the parity claim on every platform **with no Geant4
present** — because the Python reference reproduces the compiled library exactly, the committed
digest is a check anyone can run.

| Driver | What it harvests |
|---|---|
| `harvest_d1.cc` | `GetMuonCaptureRate(Z, A)` over Z 1..120 × A 1..300, then `GetMuonZeff(Z)` for Z 0..101 |
| `harvest_d1_degenerate.cc` | the inputs the sweep excludes: `Z = 0`, `A = 0`, `Z < 0`, and the `zeff` clamp at both ends |

The second driver exists because the oracle commits those four rows, and a committed harvested
artifact whose producing driver is not committed is exactly the reproducibility hole vendoring the
source was meant to close. They are separate because the sweep must stay a clean numeric box: the
degenerate inputs return non-finite values, a NaN has no single bit pattern to hash, and folding
them into the digest would make it undefined.

## Building and running

```sh
source ~/geant4/install/bin/geant4.sh
g++ -O2 harvest_d1.cc            -o harvest_d1            $(geant4-config --cflags --libs)
g++ -O2 harvest_d1_degenerate.cc -o harvest_d1_degenerate $(geant4-config --cflags --libs)
./harvest_d1 > sweep.txt
./harvest_d1_degenerate > degenerate.txt
```

Values are printed with `%a` — the exact hexadecimal float — so nothing is lost to decimal
rounding on the way out. Consumers compare **parsed values**, never the printed strings, so no
`%a`-versus-`float.hex()` formatting question can arise.

## From harvest to oracle

The drivers print raw lines; `build_oracle.py` is the step between those and the committed file, and
it is committed for the same reason they are — the principle above applies to the whole chain, not
only to its C++ half. It passes every harvested value through untouched, and computes exactly two
things: the sha256 over the harvested doubles, and *which* harvested rows are echoed into the
diagnostic subset (every table hit, every Z's first negative A, the corners of the box).

```sh
python cpp/tools/build_oracle.py \
    --sweep sweep.txt --degenerate degenerate.txt \
    --build "Ubuntu 26.04 (WSL2), x86_64, g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0," \
    --build "Geant4 11.4.2 RelWithDebInfo (-O2 -g -DNDEBUG), no -ffp-contract setting," \
    --build "FMA absent from the baseline ISA so contraction is impossible" \
    -o data/g4/d1/d1_gp_sweep.oracle
```

That invocation, on the build named in it, reproduces the committed
`data/g4/d1/d1_gp_sweep.oracle` **byte for byte** from a clean rebuild of both drivers.

The build description is an argument rather than a constant on purpose: the header records the build
that produced the values, so whoever runs the harvest states it, and a header cannot come to claim a
build nobody used. Re-running this script does **not** re-pin anything — if the digest it computes
differs from the committed one, see below.

## The build is part of the measurement

The recorded build for the committed oracle:

```
Ubuntu 26.04 (WSL2), x86_64
g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0
Geant4 11.4.2, CMAKE_BUILD_TYPE=RelWithDebInfo (-O2 -g -DNDEBUG)
no -ffp-contract setting; FMA absent from the baseline ISA, so contraction is impossible
```

That last line is not bookkeeping. Compiling the **identical** Goulard–Primakoff expression with
contraction enabled — which is the compiler default wherever FMA exists, including the *baseline*
ISA on aarch64 — moves the result by up to **2980 ulp**, with 14668 of the 36000 swept points
differing. Geant4's own build sets no `-ffp-contract` flag, so two conforming Geant4 builds of one
source compute different muon capture rates: physically negligible at ~7e-13 relative, formally
fatal to any unqualified "bit-identical" claim.

Consequences, all of which the dataset already carries:

* every parity statement names its build (this file, and the oracle header);
* the `goulard_primakoff` model contract **forbids contraction** for a conforming evaluation
  (`DATASET_D1.md`);
* a C++ validator reproducing this dataset must compile `-ffp-contract=off`, or it will disagree
  with the dataset on any FMA-capable target;
* Python is contraction-free by construction — CPython rounds every operation separately — which is
  why the reference implementation *is* the contract rather than merely obeying it.

## Re-harvesting

The digest in the oracle is **pre-registered**: on the build above the value is determined, so a
disagreement means the build, the driver, or the environment is not what this file describes. That
is a stop-and-diagnose, never a re-pin. Do not adjust the reference implementation to make a
mismatch go away — the reference implementation is the thing being tested.
