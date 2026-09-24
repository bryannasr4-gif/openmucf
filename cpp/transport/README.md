# The transport workflow

This directory builds each Geant4 revision the patches under `cpp/patches/` name from a clean export of its tag, as released and with both patches applied, under the same configuration, and transports negative muons stopped in single-isotope targets through both builds.

`matrix.json` names the revisions, the targets, the routes a stopped muon takes into the patched code, the modes and the thread counts.

Each event is seeded from its own number, and each track it creates is written with its energy and time as exact hexadecimal floating-point values, so that the records of different runs can be compared byte for byte.

`transport.py check` requires the unpatched build, the patched build with the dataset opt-in off and the patched build with the opt-in on and no profile selected to write identical records, and requires the records of each thread count the matrix names to be identical once sorted.

Under the profiles the matrix selects, it requires each transition of the muonic cascade to carry the difference of the level energies it connects, and a transition between tabulated levels to carry the difference of their table values, within the tolerance the script states.

For each target it requires the capture rate, looked up by nuclide, the effective charge, looked up by element, and the K-shell energy, looked up by nuclide, to equal the value the selected profile's table gives for that key under the lookup rules `cpp/patches/README.md` states, or, where the profile gives none, the unpatched build's value.

`transport.py cases` runs the profile-selection, discovery and refusal cases, and a build whose cascade does not call its level check, which shows that the refusal of a level array that does not fall comes from that call.

`evidence/manifest.json` records the sha256 of each patch, of the CMake cache, of each executable and of each library it loads; `evidence/cells.csv` records the result of each check; and a test holds the manifest to the patches, harness and dataset committed beside it.

```sh
python cpp/transport/transport.py stage-dataset --out <dir>
python3 cpp/transport/transport.py farm --tag <tag> --physics-data <dir> --dataset <dir> --work <dir>
python3 cpp/transport/transport.py preserved --tag <tag> --install <dir> --work <dir>
python3 cpp/transport/transport.py throughput --tag <tag> --work <dir> --events <events>
python3 cpp/transport/transport.py build --tag <tag> --source <dir> --kind <kind> --work <dir>
python3 cpp/transport/transport.py run --tag <tag> --mode <mode> --work <dir>
python3 cpp/transport/transport.py cases --work <dir>
python3 cpp/transport/transport.py manifest --work <dir> --out cpp/transport/evidence/manifest.json
python3 cpp/transport/transport.py check --work <dir> --out cpp/transport/evidence/cells.csv
```
