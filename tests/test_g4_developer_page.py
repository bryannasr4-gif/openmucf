"""`docs/geant4-developers.md` held to the tree: its paragraphs and bullets carry links, the opener and
the names bullet link each document their clauses come from, its relative links resolve, the patches,
opt-in call and table directories its commands name are the ones the repository ships, the clauses
listed in `COPIED` stand both on the page and in their sources, and `selector_vs_primary.csv` and
`incompatible_groups.csv` carry the gating and the empty intersections the page states."""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "geant4-developers.md"
PATCHES = REPO / "cpp" / "patches"
LINK = re.compile(r"\]\(([^)\s]+)\)")

#: Clauses the page copies, each with the document it copies it from.
COPIED = (
    (
        "runs exactly its unpatched code",
        "cpp/patches/README.md",
    ),
    (
        "unpatched build and the patched build with the opt-in off, or on with no profile selected, "
        "write identical records",
        "CHANGELOG.md",
    ),
    (
        "no single shell energy can lie within the band of every line it stands for",
        "DATASET_D3.md",
    ),
    (
        "adopted after the selector's ratio had already been compared with the rows it gates",
        "DATASET_D2.md",
    ),
    (
        "the difference may come from such transfer rather than from the weights, and this "
        "comparison does not tell them apart",
        "DATASET_D2.md",
    ),
    (
        "constructs a new muonic atom before it takes the ion table's lock",
        "cpp/transport/README.md",
    ),
    (
        "pending discussion with the Geant4 collaboration",
        "README.md",
    ),
    (
        "none is fitted away",
        "DATASET_D3.md",
    ),
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def _rows(relative: str) -> list[dict[str, str]]:
    with (REPO / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_every_relative_link_on_the_page_resolves() -> None:
    targets = [t for t in LINK.findall(_page()) if not re.match(r"[a-z][a-z0-9+.-]*:", t)]
    assert targets, "the page links nowhere -- the pattern is broken, not the page"
    for target in targets:
        assert "#" not in target, f"{target}: a fragment is not checked here"
        resolved = (PAGE.parent / target).resolve()
        assert resolved.is_relative_to(REPO.resolve()) and resolved.exists(), target


def test_every_paragraph_and_bullet_of_the_page_carries_a_link() -> None:
    prose = re.sub(r"^```.*?^```", "", _page(), flags=re.M | re.S)
    blocks = [line for line in prose.splitlines() if line.strip() and not line.startswith("#")]
    assert blocks, "the page has no prose -- the split is broken, not the page"
    for line in blocks:
        assert LINK.search(line), line


#: For the opener and the names bullet, which join clauses from several documents, the documents
#: whose clauses they join.
REQUIRED_LINKS = {
    "This repository ships": {"../paper/muonic-data/paper.md", "../cpp/patches/README.md"},
    "- **Names, registration.**": {"../README.md", "../FORMAT_SPEC.md", "../cpp/patches/README.md"},
}


def test_the_opener_and_the_names_bullet_link_each_document_their_clauses_come_from() -> None:
    lines = _page().splitlines()
    for start, required in REQUIRED_LINKS.items():
        (line,) = [line for line in lines if line.startswith(start)]
        assert required <= set(LINK.findall(line)), (start, required - set(LINK.findall(line)))


def test_the_patches_the_page_names_are_the_patches_the_repository_ships() -> None:
    named = set(re.findall(r"g4-v[0-9A-Za-z.]+-muonicdata\.patch", _page()))
    shipped = {path.name for path in PATCHES.glob("*-muonicdata.patch")}
    assert shipped and named == shipped, (named, shipped)


def test_the_opt_in_line_on_the_page_is_the_call_the_transport_harness_makes() -> None:
    block = re.search(r"^```cpp\n(.+?)\n```", _page(), re.M | re.S)
    assert block, "the page carries no C++ block"
    call = block.group(1).split("//")[0].strip()
    harness = (REPO / "cpp" / "transport" / "g4muonic_transport.cc").read_text(encoding="utf-8")
    assert call and call in harness, call
    for patch in sorted(PATCHES.glob("*-muonicdata.patch")):
        text = patch.read_text(encoding="utf-8")
        assert re.search(r"^\+\s*void SetEnableMuonicData\(\s*G4bool val\s*\);", text, re.M), patch.name


def test_the_dataset_command_copies_the_table_directories_the_validator_reads() -> None:
    command = next(line for line in _page().splitlines() if line.startswith("mkdir G4MuonicData"))
    copied = set(re.findall(r"openmucf/(data/g4/\w+)/\*\.g4dat", command))
    cmake = (REPO / "cpp" / "test" / "CMakeLists.txt").read_text(encoding="utf-8")
    named_dir = r'set\(G4MUONICDATA_D\d_DIR "\$\{G4MUONICDATA_REPO_ROOT\}/(data/g4/\w+)"\)'
    validator = set(re.findall(named_dir, cmake))
    tables = (REPO / "data" / "g4").glob("*/*.g4dat")
    holding = {path.parent.relative_to(REPO).as_posix() for path in tables}
    assert copied and copied == validator == holding, (copied, validator, holding)


def test_the_clauses_the_page_copies_stand_in_their_sources() -> None:
    page = _collapsed(_page())
    for clause, source in COPIED:
        assert clause in page, clause
        assert clause in _collapsed((REPO / source).read_text(encoding="utf-8")), (clause, source)


def test_the_selector_comparison_says_what_the_page_says() -> None:
    rows = _rows("data/g4/d2/selector_vs_primary.csv")
    assert rows
    assert {row["gated"] for row in rows} == {"false"}
    second = [row for row in rows if row["method_gated"] == "true"]
    assert second
    assert {row["method"] for row in second} == {"lifetime"}
    assert {row["method_result"] for row in second} == {"outside"}
    assert {row["class"] for row in second} <= {"d2-selector-oxides", "d2-selector-other-compounds"}


def test_every_multi_line_group_of_the_cascade_comparison_has_an_empty_intersection() -> None:
    groups = _rows("data/g4/d3/incompatible_groups.csv")
    assert groups
    assert {group["empty"] for group in groups} == {"true"}
