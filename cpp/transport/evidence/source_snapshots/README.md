# Historical transport patch sources

`index.json` locates the two provider patches whose SHA-256 values are recorded in
`../manifest.json`. Each snapshot is the exact committed blob at `source_commit`
and the original `path`; `git_blob` records its Git object identity. The intake
checked each blob against the existing manifest SHA-256 before copying it here.
These are archived patches from this repository's own code history.

They are the patches of an earlier run of the workflow, whose manifest and cell
results the repository's history keeps; their provider patches predate the current
linear recoil implementation, and their registration patches register the archive
as it stood before the evaluated pair table joined it. Their entries name their own
`source_commit`. The committed `../manifest.json` and `../cells.csv` now record a
run with the current patches under `cpp/patches/` and the shipped dataset archive,
so none of its entries resolves to a snapshot.

The historical manifest test resolves the explicit `(path, sha256)` entries in the
index before checking the bytes. A missing or corrupted indexed snapshot fails;
it does not fall back to the current patch. Unindexed patches must still match the
manifest at their original paths. `cpp/transport/transport.py` continues to build
from the current `cpp/patches/` files; these snapshots are provenance only.
