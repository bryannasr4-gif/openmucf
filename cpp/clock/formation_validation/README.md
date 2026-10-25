# Test-only formation-time observers and comparators

The complete runnable two-revision commands are in [the clock README](../README.md).
This directory supplies every additional source and hook its formation checker
requires. It does not add a production runtime option or change the clock patch.

`prepare.py` copies four files from the selected pristine Geant4 source tree into
a fresh output directory, applies the complete clock patch, then applies the
matching `trace-g4-*.patch`. Each trace patch only adds const-getter observations
around the existing reaction-product loop. It prints the incoming fragment origin,
independent TOF/FormationTime metadata and the selected local time.

The new comparator uses that traced formation getter. The old comparator replaces
exactly that statement with `fTime = time0 + rp->GetTOF();`; zero uses
`fTime = time0;`. They remove the now-unused explicit algorithm include to reproduce
the tested source bodies exactly. A valid zero formation value never falls back
to TOF in the corrected variant.

`formation_observer.cc` is the component probe plus its case-ID hook.
`audit_transport.cc` is the transport probe plus its event-ID hook and passive
end-of-event full RNG-state observation. Their executable source bodies equal the
tested bodies byte for byte. The comparator CMake differs only in the observer
source basename. `tested_sources.json` pins those bodies and the generated
old/new/zero model files for both named revisions. Preparation normalizes copied
source text to LF; it never edits the input source tree.

`run.py` invokes this CMake build with the three enclosing clock-process sources
and one model variant per executable. It builds the same wrapper/transport matrix:
four controlled calls, fifty default calls and four light-target calls at each of
0/250/1000 ns; then 120 atomic-route events at 250 ns, seed 12345. Raw output goes to
`OUTPUT_ROOT/logs/<revision>/`, ready for `../check_formation.py`. Build/run commands
are bounded by timeouts. The helper refuses an existing per-revision output tree.
Use the ordinary Geant4 dataset configuration for the matching installation.

The trace patches, including their Geant4 context lines, explicitly point to the
existing [Geant4 Software License](../../patches/LICENSE), also copied verbatim as
[LICENSE](LICENSE) here. The observer applications use the repository license.

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).
