"""openmucf.g4.sources -- structural extractors for the upstream sources a dataset reproduces.

A ``parity`` profile claims to reproduce some other implementation's compiled-in numbers
bit-for-bit. The only way to make that claim checkable is to derive the dataset **from the upstream
source text itself**, at build time, every time -- so this package parses vendored upstream files
and hands the generator records it read rather than records anybody typed.

Two rules govern everything here, and both exist because the alternative silently produces a dataset
that is wrong in a way no test notices:

* **structural, never positional.** A parser keyed to line numbers still parses at the next upstream
  release; it just extracts the wrong thing. Every extractor here anchors on the *declaration* it
  wants and brace-matches to its end, so a moved construct is found and a renamed or deleted one is
  a loud error.
* **no count is written down.** Counts are ``len()`` of what was parsed. The record count of a
  compiled-in table is exactly the kind of fact that gets copied into a document once, drifts, and
  is then never re-derived; ``tests/test_g4parity.py`` walks this package's AST and fails if a
  literal count appears in it.

Standard library only, and no import of the kinetics modules -- the same fence
``openmucf/g4/__init__.py`` states and ``tests/test_g4spec.py`` enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["bibkeys", "d1_nuclear_capture"]

#: A BibTeX entry key, as ``openmucf.rates.bibkeys()`` reads them.
_BIBKEY_PATTERN = re.compile(r"@\w+\{([^,]+),")


def bibkeys(bib_path: Path) -> set[str]:
    """Every citation key defined in a BibTeX file.

    Duplicated from :func:`openmucf.rates.bibkeys` on purpose rather than imported: this subpackage
    is fenced off from the rest of ``openmucf`` so that the data layer can be lifted into its own
    distribution, and a three-line regex is a cheaper price than an import edge that would have to
    be unpicked later. The duplication cannot drift -- ``tests/test_g4parity.py`` asserts the two
    agree on the shipped bibliography.
    """
    return set(_BIBKEY_PATTERN.findall(Path(bib_path).read_text(encoding="utf-8")))
