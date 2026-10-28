# Historical transport patch sources

`index.json` locates the two provider patches whose SHA-256 values are recorded in
`../manifest.json`. Each snapshot is the exact committed blob at `source_commit`
and the original `path`; `git_blob` records its Git object identity. The intake
checked each blob against the existing manifest SHA-256 before copying it here.
These are archived patches from this repository's own code history.

The historical manifest, cell results and build identities are unchanged. Their
provider patches predate the current linear recoil implementation. These results
therefore describe those earlier builds; they do not validate the current patches.
The two registration patches are archived the same way: the evaluated pair table
later joined the dataset, which changed the archive digest those patches register.
Their entries name their own `source_commit`. The historical dataset identity is
rebuilt as the current dataset without the pair table
(`scripts/generate_g4data.py`, `build_dataset_artifacts(pair_table=False)`).

The historical manifest test resolves the explicit `(path, sha256)` entries in the
index before checking the bytes. A missing or corrupted indexed snapshot fails;
it does not fall back to the current patch. Unindexed patches must still match the
manifest at their original paths. `cpp/transport/transport.py` continues to build
from the current `cpp/patches/` files; these snapshots are provenance only.
