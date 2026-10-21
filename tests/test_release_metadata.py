"""The citation metadata describe the release whose version they carry, and the README says what is not in it.

`CITATION.cff`, `codemeta.json` and `.zenodo.json` carry one release version (and the CFF its DOI). Their
descriptions may name only the D3 profiles that the change log's section for that release names. The
README's citation paragraph, which describes the default branch, binds each D3 profile it names to an
explicit clause: the released clause ("Release v<version> holds ...", with the files' version) or the
unreleased clause ("... in no release yet ..."), each profile in a "`<profile>` profile at dataset version
<x.y.z>" pair. The released pairs must be the release section's profiles at the release's dataset version
(the topmost "`#VERSION` becomes <x.y.z>" of that section); the unreleased pairs, the profiles only
[Unreleased] names at the dataset version of the tables on the tree (the single `#VERSION` of
`data/g4/d3/*.g4dat`). No PyYAML: the CFF's version line and abstract block are read with the regular
expressions below.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PROFILE = re.compile(r"\bmudirac130\w*")
PAIR = re.compile(r"`(mudirac130\w*)` profile at dataset version (\d+\.\d+\.\d+)")
RELEASED_CLAUSE = re.compile(r"^Release v(\d+\.\d+\.\d+) holds\b")
UNRELEASED_CLAUSE = re.compile(r"\bin no release yet\b")
BECOMES = re.compile(r"`#VERSION` becomes (\d+\.\d+\.\d+)")


def cff_version_and_abstract(cff: str) -> tuple[str, str]:
    version = re.search(r"^version:\s*(\S+)\s*$", cff, re.M).group(1)
    block = re.search(r"^abstract: >-\n((?:  .*\n)+)", cff, re.M)
    return version, " ".join(line.strip() for line in block.group(1).splitlines())


def changelog_section(changelog: str, head: str) -> str:
    m = re.search(rf"^## \[{re.escape(head)}\][^\n]*\n(.*?)(?=^## \[)", changelog, re.M | re.S)
    return m.group(1) if m else ""


def readme_citation_paragraph(readme: str) -> str:
    return re.search(r"^## How to cite\n(.*?)(?=^## )", readme, re.M | re.S).group(1)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+", " ".join(text.split())) if s]


def tree_dataset_version(root: Path = REPO) -> str:
    """The dataset version of the D3 tables on the tree: the single `#VERSION` of data/g4/d3/*.g4dat."""
    found = set()
    for p in sorted((root / "data" / "g4" / "d3").glob("*.g4dat")):
        m = re.search(r"^#VERSION\s+(\S+)\s*$", p.read_text(encoding="utf-8"), re.M)
        found.add(m.group(1) if m else None)
    assert len(found) == 1 and None not in found, f"the D3 tables carry more than one #VERSION: {found}"
    return found.pop()


def release_failures(cff, codemeta, zenodo, changelog, readme_par, tree_version) -> list[str]:
    """Each way the three descriptions or the README paragraph depart from the change log and the tree."""
    probs = []
    version, abstract = cff_version_and_abstract(cff)
    versions = {
        "CITATION.cff": version,
        "codemeta.json version": codemeta["version"],
        "codemeta.json softwareVersion": codemeta["softwareVersion"],
        ".zenodo.json": zenodo["version"],
    }
    if len(set(versions.values())) != 1:
        probs.append(f"versions differ: {versions}")
    rel_sec, unrel_sec = changelog_section(changelog, version), changelog_section(changelog, "Unreleased")
    if not rel_sec:
        probs.append(f"the change log has no section for release {version}")
    released = set(PROFILE.findall(rel_sec))
    unreleased_only = set(PROFILE.findall(unrel_sec)) - released
    becomes = BECOMES.findall(rel_sec)
    release_dv = becomes[0] if becomes else None
    if release_dv is None:
        probs.append(f"the change log's section {version} states no dataset version")
    for name, text in (("CITATION.cff abstract", abstract), ("codemeta.json", codemeta["description"]),
                       (".zenodo.json", zenodo["description"])):
        named = set(PROFILE.findall(text))
        if not named:
            probs.append(f"{name}: names no D3 profile")
        if named - released:
            probs.append(f"{name}: names {sorted(named - released)}, which release {version} does not hold")
    got: dict[str, set] = {"released": set(), "unreleased": set()}
    for s in sentences(readme_par):
        named = PROFILE.findall(s)
        if not named:
            continue
        rel, unrel = RELEASED_CLAUSE.search(s), UNRELEASED_CLAUSE.search(s)
        if bool(rel) == bool(unrel):
            probs.append(f"README: a sentence naming {sorted(set(named))} is not exactly one of the released "
                         f"and the unreleased clause: {s!r}")
            continue
        if rel and rel.group(1) != version:
            probs.append(f"README: the released clause names v{rel.group(1)}, the files carry {version}")
        pairs = PAIR.findall(s)
        if sorted(p for p, _ in pairs) != sorted(named):
            loose = sorted(set(named) - {p for p, _ in pairs}) or sorted(named)
            probs.append(f"README: {loose} named outside a profile-and-dataset-version pair: {s!r}")
        got["released" if rel else "unreleased"] |= set(pairs)
    want = {"released": {(p, release_dv) for p in released},
            "unreleased": {(p, tree_version) for p in unreleased_only}}
    for cls in ("released", "unreleased"):
        if got[cls] != want[cls]:
            probs.append(
                f"README: the {cls} clause gives {sorted(got[cls])}, the records give {sorted(want[cls])}"
            )
    both = {p for p, _ in got["released"]} & {p for p, _ in got["unreleased"]}
    if both:
        probs.append(f"README: {sorted(both)} classified both released and unreleased")
    return probs


def _texts():
    def read(rel):
        return (REPO / rel).read_text(encoding="utf-8")

    return {
        "cff": read("CITATION.cff"),
        "codemeta": json.loads(read("codemeta.json")),
        "zenodo": json.loads(read(".zenodo.json")),
        "changelog": read("CHANGELOG.md"),
        "readme_par": readme_citation_paragraph(read("README.md")),
        "tree_version": tree_dataset_version(),
    }


def test_the_metadata_describe_their_release_and_the_readme_what_is_unreleased():
    assert release_failures(**_texts()) == []


def test_each_clause_has_something_to_bind():
    """The check is not vacuous: the release section names a profile and a dataset version, [Unreleased]
    names a profile the release does not, and the README paragraph has one released and one unreleased
    clause."""
    t = _texts()
    version, _ = cff_version_and_abstract(t["cff"])
    rel = set(PROFILE.findall(changelog_section(t["changelog"], version)))
    unrel = set(PROFILE.findall(changelog_section(t["changelog"], "Unreleased"))) - rel
    assert rel and unrel and BECOMES.search(changelog_section(t["changelog"], version))
    kinds = [bool(RELEASED_CLAUSE.search(s)) for s in sentences(t["readme_par"]) if PROFILE.search(s)]
    assert sorted(kinds) == [False, True], kinds


def _pdrill(old, new):
    return lambda t: dict(t, readme_par=t["readme_par"].replace(old, new))


def _append(sentence):
    return lambda t: dict(t, readme_par=t["readme_par"].rstrip("\n") + " " + sentence + "\n")


def _swap(a, b):
    return lambda t: dict(t, readme_par=t["readme_par"].replace(a, "@").replace(b, a).replace("@", b))


#: (name, how the drill alters the corrected texts): each must be refused
DRILLS = [
    ("CFF abstract names the unreleased profile",
     lambda t: dict(t, cff=t["cff"].replace("tables of the mudirac130 profile",
                                            "tables of the mudirac130p1 profile"))),
    ("codemeta names the unreleased profile",
     lambda t: dict(t, codemeta=dict(t["codemeta"], description=t["codemeta"]["description"].replace(
         "mudirac130 profile", "mudirac130p1 profile")))),
    ("zenodo version differs", lambda t: dict(t, zenodo=dict(t["zenodo"], version="1.4.0"))),
    ("README drops the unreleased profile", _pdrill("`mudirac130p1`", "the new")),
    ("README says nothing is unreleased", _pdrill("in no release yet", "released")),
    ("README calls the released profile unreleased",
     _append("The `mudirac130` profile is in no release yet.")),
    ("README calls the released profile, with its version, unreleased",
     _append("The `mudirac130` profile at dataset version 0.5.1 is in no release yet.")),
    ("README moves the released profile into the unreleased clause",
     lambda t: dict(t, readme_par=" ".join(t["readme_par"].split()).replace(
         "Release v1.3.0 holds the D3 tables of the `mudirac130` profile at dataset version 0.5.1.",
         "Release v1.3.0 holds the D3 tables.").replace(
         "among them the `mudirac130p1`",
         "among them the `mudirac130` profile at dataset version 0.5.1 and the `mudirac130p1`"))),
    ("README gives the released profile the unreleased dataset version",
     _pdrill("dataset version 0.5.1", "dataset version 0.6.0")),
    ("README swaps the two dataset versions", _swap("0.5.1", "0.6.0")),
    ("main's dataset version differs from the README", lambda t: dict(t, tree_version="0.6.1")),
    ("the released clause names another release", _pdrill("Release v1.3.0", "Release v1.2.0")),
]


@pytest.mark.parametrize("name, drill", DRILLS, ids=[d[0] for d in DRILLS])
def test_release_drill_is_refused(name, drill):
    """Drill: each alteration of the corrected texts changes them, and is refused."""
    t = _texts()
    drilled = drill(t)
    assert drilled != t, name
    assert release_failures(**drilled), name
