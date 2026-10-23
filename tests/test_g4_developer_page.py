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
        "it does not tell whether the difference arises in the weights or in the steps by which the muon "
        "reaches that element",
        "DATASET_D2.md",
    ),
    (
        "an abundance-weighted first-order estimate against the measurement the table itself encodes",
        "DATASET_D1.md",
    ),
    (
        "answers only a nucleus whose Z and A both match a record, and the Goulard–Primakoff formula "
        "answers every other",
        "DATASET_D1.md",
    ),
    (
        "a target of natural composition reaches an element's record only on the stops that draw the "
        "record's own mass number",
        "DATASET_D1.md",
    ),
    (
        "The comparison is of the rates the code returns for a single nucleus, not of a transported "
        "simulation of a target",
        "DATASET_D1.md",
    ),
    (
        "can rest on the same mean life, so the band errs wide",
        "DATASET_D1.md",
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
    (
        "the tables leave out nuclear polarization, the self energy, vacuum polarization beyond the "
        "Uehling potential, recoil beyond the reduced mass, electron screening, hyperfine structure and "
        "nuclear charge distributions other than 2pF",
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
    "This repository ships": {"../paper/muonic-data/paper.md"},
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
    assert {row["class"] for row in second} == {"d2-selector-oxides", "d2-selector-other-compounds"}


#: The variable, the patch path and the header the page names, as the page's own commands spell them.
EXPORTED = re.compile(r"export (\w+)=")
APPLIED = re.compile(r'"\$PWD/openmucf/(cpp/patches/[\w.-]+\.patch)"')
INCLUDED = re.compile(r'#include "(\w+\.hh)"')


def names_problems(page: str) -> list[str]:
    """What the page names that the repository does not use: the variable it exports against the one the
    glue's `G4FindDataDir` call reads, each patch it applies against the files under `cpp/patches/`, and the
    header its include comment names against the file in which every behaviour patch declares the setter."""
    problems = []
    glue = (REPO / "cpp" / "src" / "G4MuonicDataOverlay.cc").read_text(encoding="utf-8")
    (read,) = re.findall(r'G4FindDataDir\("(\w+)"\)', glue)
    exported = EXPORTED.findall(page)
    if exported != [read]:
        problems.append(f"the page exports {exported}; the glue reads {read}")
    applied = APPLIED.findall(page)
    if not applied or not all((REPO / path).is_file() for path in applied):
        problems.append(f"the page applies {applied}, not files of the repository")
    declared = set()
    for patch in sorted(PATCHES.glob("*-muonicdata.patch")):
        for section in patch.read_text(encoding="utf-8").split("\ndiff --git ")[1:]:
            if re.search(r"^\+\s*void SetEnableMuonicData\(\s*G4bool val\s*\);", section, re.M):
                declared.add(section.split()[0].rsplit("/", 1)[1])
    included = INCLUDED.findall(page)
    if len(declared) != 1 or included != sorted(declared):
        problems.append(f"the page includes {included}; the patches declare the setter in {sorted(declared)}")
    return problems


def test_the_variable_patch_paths_and_header_the_page_names_are_the_repository_s() -> None:
    assert names_problems(_page()) == []


def test_drill_a_wrong_variable_patch_path_or_header_on_the_page_is_named() -> None:
    page = _page()
    for old, new, word in (
        ("export G4MUONICDATA=", "export G4MUONICDATA_DIR=", "exports"),
        ("openmucf/cpp/patches/g4-v11.4.2", "openmucf/cpp/patch/g4-v11.4.2", "applies"),
        ('#include "G4HadronicParameters.hh"', '#include "G4HadronicProcessStore.hh"', "includes"),
    ):
        assert page.count(old) == 1, old
        assert any(word in problem for problem in names_problems(page.replace(old, new))), old


def test_every_multi_line_group_of_the_cascade_comparison_has_an_empty_intersection() -> None:
    groups = _rows("data/g4/d3/incompatible_groups.csv")
    assert groups
    assert {group["empty"] for group in groups} == {"true"}
