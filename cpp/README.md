# The C++ reader, its validator and the harvest tooling

`include/` and `src/` hold `G4MuonicDataTable`, a standard-library-only C++17 reader for the Layer 1
`.g4dat` grammar; nothing in it depends on Geant4. `test/` is the CMake project root: the standalone
validator, the toolchain probe and the F-3 producer, none of which needs a Geant4 installation.
`tools/` holds the Geant4-linked harvest drivers and `build_oracle.py`, the step between a harvest
and the committed oracle. `patches/` is the overlay: the patches that let a Geant4 build read the
dataset behind an explicit opt-in.

Build and run the validator from the repository root:

```sh
cmake -S cpp/test -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

CI builds and runs it on Linux, macOS and Windows. How a reader finds the dataset at run time is
specified in `FORMAT_SPEC.md` section 5 and the rules for C and C++ readers in section 6; the opt-in
is described in `cpp/patches/README.md`.
