# MuDirac 1.3.0 patch

`mudirac-1.3.0-p1.patch` is one cumulative patch to MuDirac 1.3.0 (Sturniolo and Hillier). The D3 tables of the
`mudirac130p1` profile are computed with it (`DATASET_D3.md`). It does two things:
- it repairs numerics: the Uehling and finite-size Coulomb quadratures and the precision of the printed energies;
- it makes the Hamiltonian consistent: one nuclear charge density for both the Coulomb and the Uehling potential, the
  Fermi parameter solved from the intended rms radius, the nuclear mass in the reduced mass, and the CODATA 2022
  Hartree energy on the printed energies.

The patch is offered under MuDirac's MIT License ([`LICENSE`](LICENSE)). That covers the MuDirac lines it carries and
the lines it adds.

## Build
The patch applies to the MuDirac 1.3.0 release archive `mudirac-1.3.0.tar.gz`, sha256
`e50e346c3b9dedb29c11685d42fa9042ee92fd89240aaf07c8fd2342187bece0`.

```sh
echo "e50e346c3b9dedb29c11685d42fa9042ee92fd89240aaf07c8fd2342187bece0  mudirac-1.3.0.tar.gz" | sha256sum -c -
tar -xzf mudirac-1.3.0.tar.gz
cd mudirac-1.3.0
patch -p1 < /path/to/third_party/mudirac/mudirac-1.3.0-p1.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mudirac
```

The binary is `build/bin/mudirac`. It reads one input file, as MuDirac 1.3.0 does.

## Keywords
Each keyword the patch adds defaults to one of three things:
- off;
- MuDirac 1.3.0's own value;
- for `uehling_shell_nodes` and `fermi2_rms_fm`, a value read only when another of these keywords is set.

| keyword | default | effect |
|---|---|---|
| `uehling_stable` | FALSE | the uniform-sphere Uehling kernel in a cancellation-free form (trapezoid rule in u) |
| `uehling_gauss` | FALSE | the same kernel by Gauss-Legendre quadrature in theta (u = sin theta), with `uehling_steps` nodes |
| `fermi2_csteps` | 5000 | the size of the hidden logarithmic grid of the 2pF Coulomb potential |
| `fermi2_hermite` | FALSE | the 2pF Coulomb moments by 4-point Gauss-Legendre quadrature per interval, read back by cubic Hermite interpolation |
| `state_header_precision` | -1 | the significant digits of the energy header of every state file (-1: the stream default) |
| `domain_in_eps`, `domain_out_eps` | 1e-5 | the inner and outer integration-domain tolerances, which MuDirac 1.3.0 fixes at 1e-5 |
| `uehling_density` | sphere | `fermi2`: the Uehling potential of the run's own 2pF charge density, the one the Coulomb potential uses |
| `uehling_shell_nodes` | 12 | the Gauss-Legendre nodes per panel of the 2pF Uehling superposition |
| `fermi2_rms_exact` | FALSE | the 2pF parameter c solved from the exact 2pF moments, so that the density's rms radius is `fermi2_rms_fm` |
| `fermi2_rms_fm` | -1 | the run's intended rms charge radius, in fm |
| `nuclear_mass` | FALSE | the reduced mass from the nuclear mass M_atom u - Z m_e + B_e(Z)/c^2, with B_e from AME2020 II, eqs. 1-2 |
| `ev_codata2022` | FALSE | printed energies in eV through the CODATA 2022 Hartree energy, 27.211386245981 eV |

The patched build refuses an input, with a non-zero exit status and a message, in each of these cases:
- `uehling_density` is neither `sphere` nor `fermi2`;
- `fermi2_rms_exact` is set without both `fermi2_rms_fm` and `radius`;
- `radius / sqrt(5/3)` and `fermi2_rms_fm` disagree by more than 1e-12 relative;
- `uehling_density: fermi2` is used without `nuclear_model: FERMI2`, or with a non-zero `uehling_lowcut`;
- `ev_codata2022` is combined with `devel_debug_task` or `optimise_fermi_parameters`.

If the intended rms radius is at or below the smallest rms a 2pF density of the run's thickness can have, the build
keeps MuDirac's own c and logs a warning.

## Default output
With default keywords the patched build printed the same output as MuDirac 1.3.0, byte for byte, on the six inputs in
[`default_check/`](default_check/). The comparison covered every `.out` file a run writes. It covers those six inputs
and nothing broader.
