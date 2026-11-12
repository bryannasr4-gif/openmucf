"""The citation metadata describe the release whose version they carry, and the README says what is not in it.

`CITATION.cff`, `codemeta.json` and `.zenodo.json` carry one release version. The D3 profiles a release
holds are the `#PROFILE` values of the D3 tables the tree ships (`data/g4/d3/*.g4dat`); the change log's
section for that release must name each of them and may name others, such as a profile it replaced. The
three descriptions may name only held profiles. The README's citation paragraph, which describes the default
branch, binds each D3 profile it names to an explicit clause: the released clause ("Release v<version>
holds ...", with the dataset version of that release) or the unreleased clause ("... in no release yet ..."),
each profile in a "`<profile>` profile at dataset version <x.y.z>" pair. The released pairs must be the held
profiles at the release's dataset version (the topmost "`#VERSION` becomes <x.y.z>" of its section); the
unreleased pairs, the profiles that [Unreleased] names and the release does not hold, at the dataset version
of the tables on the tree (the single `#VERSION` of `data/g4/d3/*.g4dat`). There is an unreleased clause
exactly when [Unreleased] names such a profile, and without one the tables on the tree carry the release's
dataset version. No PyYAML: the CFF's version line and abstract block are read with the regular expressions
below.
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


def d3_tables(root: Path = REPO) -> dict[str, str]:
    """The D3 tables the tree ships, by name, as text."""
    d3 = root / "data" / "g4" / "d3"
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(d3.glob("*.g4dat"))}


def directive(tables: dict[str, str], name: str) -> set:
    found = set()
    for text in tables.values():
        m = re.search(rf"^#{name}\s+(\S+)\s*$", text, re.M)
        found.add(m.group(1) if m else None)
    return found


def tree_dataset_version(tables: dict[str, str]) -> str:
    """The dataset version of the D3 tables on the tree: the single `#VERSION` of data/g4/d3/*.g4dat."""
    found = directive(tables, "VERSION")
    assert len(found) == 1 and None not in found, f"the D3 tables carry more than one #VERSION: {found}"
    return found.pop()


def held_profiles(tables: dict[str, str]) -> set:
    """The D3 profiles the shipped tables carry (their `#PROFILE` values)."""
    found = directive(tables, "PROFILE")
    assert None not in found and found, f"a D3 table carries no #PROFILE: {found}"
    return found


def release_failures(cff, codemeta, zenodo, changelog, readme_par, tables) -> list[str]:
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
    held, tree_version = held_profiles(tables), tree_dataset_version(tables)
    rel_sec, unrel_sec = changelog_section(changelog, version), changelog_section(changelog, "Unreleased")
    if not rel_sec:
        probs.append(f"the change log has no section for release {version}")
    unnamed = held - set(PROFILE.findall(rel_sec))
    if unnamed:
        probs.append(f"the change log's section {version} does not name {sorted(unnamed)}")
    unreleased_only = set(PROFILE.findall(unrel_sec)) - held
    becomes = BECOMES.findall(rel_sec)
    release_dv = becomes[0] if becomes else None
    if release_dv is None:
        probs.append(f"the change log's section {version} states no dataset version")
    for name, text in (("CITATION.cff abstract", abstract), ("codemeta.json", codemeta["description"]),
                       (".zenodo.json", zenodo["description"])):
        named = set(PROFILE.findall(text))
        if not named:
            probs.append(f"{name}: names no D3 profile")
        if named - held:
            probs.append(f"{name}: names {sorted(named - held)}, which release {version} does not hold")
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
    want = {"released": {(p, release_dv) for p in held},
            "unreleased": {(p, tree_version) for p in unreleased_only}}
    for cls in ("released", "unreleased"):
        if got[cls] != want[cls]:
            probs.append(
                f"README: the {cls} clause gives {sorted(got[cls])}, the records give {sorted(want[cls])}"
            )
    if not unreleased_only and release_dv is not None and tree_version != release_dv:
        probs.append(f"the tables on the tree carry dataset version {tree_version}, release {version} holds "
                     f"{release_dv}, and [Unreleased] names no profile")
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
        "tables": d3_tables(),
    }


def test_the_metadata_describe_their_release_and_the_readme_what_is_unreleased():
    assert release_failures(**_texts()) == []


def test_each_clause_has_something_to_bind():
    """The check is not vacuous: the tree ships D3 tables with a profile, the release section names it and
    a dataset version, the README paragraph has one released clause, and it has an unreleased clause exactly
    when [Unreleased] names a profile the release does not hold."""
    t = _texts()
    version, _ = cff_version_and_abstract(t["cff"])
    held = held_profiles(t["tables"])
    rel_sec = changelog_section(t["changelog"], version)
    assert held and held <= set(PROFILE.findall(rel_sec)) and BECOMES.search(rel_sec)
    unrel = set(PROFILE.findall(changelog_section(t["changelog"], "Unreleased"))) - held
    kinds = [bool(RELEASED_CLAUSE.search(s)) for s in sentences(t["readme_par"]) if PROFILE.search(s)]
    assert sorted(kinds) == ([False, True] if unrel else [True]), kinds


def _par(fn):
    return lambda t: dict(t, readme_par=fn(t["readme_par"]))


def _append(sentence):
    return _par(lambda p: p.rstrip("\n") + " " + sentence + "\n")


def _release(t) -> str:
    return cff_version_and_abstract(t["cff"])[0]


def _held(t) -> str:
    return sorted(held_profiles(t["tables"]))[0]


def _dv(t) -> str:
    return BECOMES.findall(changelog_section(t["changelog"], _release(t)))[0]


def _retired(t) -> str:
    """A profile the release section names that the tree does not ship (the one it replaced)."""
    rel = set(PROFILE.findall(changelog_section(t["changelog"], _release(t))))
    return sorted(rel - held_profiles(t["tables"]))[0]


def _description(field, fn):
    return lambda t: dict(t, **{field: dict(t[field], description=fn(t, t[field]["description"]))})


def _bump(v: str) -> str:
    major, minor, patch = (int(x) for x in v.split("."))
    return f"{major}.{minor}.{patch + 1}"


def _profile_moved(t):
    """One D3 table's `#PROFILE` changed while the README is not."""
    name = sorted(t["tables"])[0]
    text = re.sub(r"^(#PROFILE\s+)(\S+)", r"\1\2x", t["tables"][name], count=1, flags=re.M)
    return dict(t, tables=dict(t["tables"], **{name: text}))


def _section_drops_held(t):
    """The release section names every profile but the held one (which the tables, the README and the
    descriptions still name)."""
    sec = changelog_section(t["changelog"], _release(t))
    return dict(t, changelog=t["changelog"].replace(sec, sec.replace(_held(t), f"{_retired(t)}q")))


#: (name, how the drill alters the corrected texts): each must be refused. Every drill reads the release
#: version, the profiles and the dataset versions from the texts; none is typed.
DRILLS = [
    ("CFF abstract names the retired profile",
     lambda t: dict(t, cff=t["cff"].replace(f"{_held(t)} profile", f"{_retired(t)} profile", 1))),
    ("codemeta names the retired profile",
     _description("codemeta", lambda t, d: d.replace(f"{_held(t)} profile", f"{_retired(t)} profile", 1))),
    ("zenodo names the retired profile",
     _description("zenodo", lambda t, d: d.replace(f"{_held(t)} profile", f"{_retired(t)} profile", 1))),
    ("codemeta names no profile", _description("codemeta", lambda t, d: PROFILE.sub("the", d))),
    ("zenodo version differs", lambda t: dict(t, zenodo=dict(t["zenodo"], version=_bump(_release(t))))),
    ("README drops the held profile", lambda t: _par(lambda p: p.replace(f"`{_held(t)}`", "the new"))(t)),
    ("README's released clause names the retired profile",
     lambda t: _par(lambda p: p.replace(f"`{_held(t)}` profile", f"`{_retired(t)}` profile"))(t)),
    ("README calls the held profile unreleased",
     lambda t: _append(f"The `{_held(t)}` profile is in no release yet.")(t)),
    ("README calls the held profile, with its version, unreleased",
     lambda t: _append(f"The `{_held(t)}` profile at dataset version {_dv(t)} is in no release yet.")(t)),
    ("README keeps an unreleased clause while [Unreleased] names none",
     lambda t: _append(f"The `{_retired(t)}` profile at dataset version {_dv(t)} is in no release yet.")(t)),
    ("README gives the held profile another dataset version",
     lambda t: _par(lambda p: p.replace(f"dataset version {_dv(t)}", f"dataset version {_bump(_dv(t))}"))(t)),
    ("the tree's dataset version differs from the README's",
     lambda t: dict(t, tables={k: re.sub(r"^(#VERSION\s+)\S+", rf"\g<1>{_bump(_dv(t))}", v, flags=re.M)
                               for k, v in t["tables"].items()})),
    ("a table's #PROFILE changed while the README is not", _profile_moved),
    ("the released clause names another release",
     lambda t: _par(lambda p: p.replace(f"Release v{_release(t)}", f"Release v{_bump(_release(t))}"))(t)),
    ("the change log has no section for the release",
     lambda t: dict(t, changelog=t["changelog"].replace(f"## [{_release(t)}]", f"## [{_release(t)}-gone]"))),
    ("the release section names the held profile nowhere", _section_drops_held),
]


@pytest.mark.parametrize("name, drill", DRILLS, ids=[d[0] for d in DRILLS])
def test_release_drill_is_refused(name, drill):
    """Drill: each alteration of the corrected texts changes them, and is refused."""
    t = _texts()
    drilled = drill(t)
    assert drilled != t, name
    assert release_failures(**drilled), name
