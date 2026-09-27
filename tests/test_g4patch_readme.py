"""T-155, T-156 -- what `cpp/patches/README.md` says a patched function does, held to both patches.

The README describes what each behaviour patch makes a Geant4 function do with the dataset opt-in
off and on. Prose of that kind goes stale when a patch gains code the prose never mentions, so these
tests read the `+` lines of both behaviour patches and require the README to name what they find:

* **T-155** -- the calls into `G4MuonicDataOverlay` that the patch's hunks in existing Geant4 files
  make on an added line outside every `if (G4MuonicDataTable::IsEnabled())` block are exactly the
  calls the README names as `G4MuonicDataOverlay::<name>`, and the README names the function each is
  made in. The `CheckComputedCaptureRate` calls are made in exactly the functions the README names
  `...::GetMuonCaptureRate`, and each passes, as its last argument, the value its function returns on
  the next line. Every definition the patch renames (T-72 holds the renames to the patch) is named in
  the README under its new name.
* **T-156** -- the `G4MuonicData` codes the patches raise are exactly the codes the README names, and
  the README paragraph that names `CheckComputedCaptureRate` names exactly the codes its body raises.

The drills plant each defect in an in-memory copy -- of a patch, parsed, or of the README -- and
require the check to name it.
"""

from __future__ import annotations

import re

import pytest
import test_g4overlay as overlay

IS_ENABLED = b"if (G4MuonicDataTable::IsEnabled())"
OVERLAY_CALL = re.compile(rb"G4MuonicDataOverlay::(\w+)\s*\(")
README_CALL = re.compile(r"G4MuonicDataOverlay::(\w+)")
README_FUNCTION = re.compile(r"`(\w+::\w+)`")
CONTEXT = re.compile(rb"^@@ [^@]* @@ .*?(\w+::\w+)\(")
DEFINITION = re.compile(rb"^[A-Za-z_][^;]*?\b(\w+::\w+)\s*\([^;]*$")
RETURN = re.compile(rb"^\s*return\s+(\w+)\s*;\s*$")
CODE = re.compile(r"G4MuonicData\d{3}")
CAPTURE_RATE = "::GetMuonCaptureRate"
CHECK = "CheckComputedCaptureRate"


def readme() -> str:
    return overlay.README.read_text(encoding="utf-8")


def parsed(tag: str) -> dict[str, overlay.FilePatch]:
    behaviour, _, _ = overlay.FAMILIES[tag]
    return overlay.by_new_path(overlay.parse_patch(behaviour.read_bytes()))


def arguments(content: bytes, start: int) -> list[bytes]:
    """The top-level comma-separated arguments of the call whose `(` sits at `start`."""
    depth, current, out = 0, b"", []
    for i in range(start, len(content)):
        ch = content[i : i + 1]
        if ch == b"(":
            depth += 1
            if depth == 1:
                continue
        elif ch == b")":
            depth -= 1
            if depth == 0:
                out.append(current.strip())
                return out
        elif ch == b"," and depth == 1:
            out.append(current.strip())
            current = b""
            continue
        current += ch
    raise AssertionError(f"unbalanced call in {content!r}")


def calls_outside_opt_in(
    files: dict[str, overlay.FilePatch],
) -> list[tuple[str, str, str, list[bytes], bytes]]:
    """(path, function, called name, arguments, next line) for every call into `G4MuonicDataOverlay`
    on an added line of a file that existed before the patch, outside every
    `if (G4MuonicDataTable::IsEnabled())` block. Braces are tracked over the added lines of each
    hunk, character by character, so a block that opens and closes on one line is seen; the
    function is the last definition an added line of the hunk opens, else the hunk header's context."""
    out = []
    for path, file in sorted(files.items()):
        if file.old_path == b"/dev/null":
            continue
        for hunk in file.hunks:
            match = CONTEXT.match(hunk.header)
            function = match.group(1).decode() if match else ""
            stack: list[bool] = []
            pending = False
            for i, (marker, content, _) in enumerate(hunk.lines):
                if marker != b"+":
                    continue
                definition = DEFINITION.match(content)
                if definition:
                    function = definition.group(1).decode()
                events = [(m.start(), 0, m) for m in re.finditer(re.escape(IS_ENABLED), content)]
                events += [(m.start(), 1, m) for m in OVERLAY_CALL.finditer(content)]
                events += [(j, 2, None) for j in range(len(content)) if content[j : j + 1] in (b"{", b"}")]
                for position, kind, m in sorted(events, key=lambda e: (e[0], e[1])):
                    if kind == 0:
                        pending = True
                    elif kind == 1 and not any(stack):
                        following = hunk.lines[i + 1][1] if i + 1 < len(hunk.lines) else b""
                        args = arguments(content, m.end() - 1)
                        out.append((path, function, m.group(1).decode(), args, following))
                    elif kind == 2 and content[position : position + 1] == b"{":
                        stack.append(pending)
                        pending = False
                    elif kind == 2:
                        assert stack, (
                            f"{path}: hunk {hunk.header.decode()}: an added '}}' closes no added '{{'"
                        )
                        stack.pop()
    return out


def check_calls_outside_opt_in(tag: str, files: dict[str, overlay.FilePatch], text: str) -> None:
    calls = calls_outside_opt_in(files)
    made = {name for _, _, name, _, _ in calls}
    named = set(README_CALL.findall(text))
    assert made == named, (
        f"{tag}: calls into G4MuonicDataOverlay outside an opt-in test: {sorted(made)}; "
        f"the README names {sorted(named)}"
    )
    for path, function, name, _, _ in calls:
        method = function.rpartition("::")[2]
        assert method and re.search(rf"\b{method}\b", text), (
            f"{tag}: {path}: {name} is called in {function or 'no function'}, which the README does not name"
        )
    checks = [call for call in calls if call[2] == CHECK]
    capture = {name for name in README_FUNCTION.findall(text) if name.endswith(CAPTURE_RATE)}
    functions = {function for _, function, _, _, _ in checks}
    assert functions == (capture if checks else set()), (
        f"{tag}: {CHECK} is called outside an opt-in test in {sorted(functions)}; "
        f"the README's capture-rate functions are {sorted(capture)}"
    )
    for path, function, name, args, following in checks:
        returned = RETURN.match(following)
        assert returned and args and args[-1] == returned.group(1), (
            f"{tag}: {path}: {function} passes {args!r} to {name}, and its next line is {following!r}"
        )


def check_renames_named(text: str) -> None:
    for path, (_, new) in overlay.RENAMES.items():
        name = re.search(rb"::(\w+)\(", new).group(1).decode()
        assert f"`{name}`" in text, f"{path}: the README does not name the renamed definition {name}"


def glue_added_text(files: dict[str, overlay.FilePatch]) -> str:
    return "\n".join(content.decode() for _, content in overlay.added_lines(files[overlay.GLUE_CC]))


def codes_raised(files: dict[str, overlay.FilePatch]) -> set[str]:
    return {
        code
        for file in files.values()
        for _, content in overlay.added_lines(file)
        for code in CODE.findall(content.decode())
    }


def check_codes_named(tag: str, files: dict[str, overlay.FilePatch], text: str) -> None:
    raised, named = codes_raised(files), set(CODE.findall(text))
    assert raised == named, f"{tag}: the patch raises {sorted(raised)}; the README names {sorted(named)}"


def check_check_codes(tag: str, files: dict[str, overlay.FilePatch], text: str) -> None:
    glue = glue_added_text(files)
    start = glue.index(f"void G4MuonicDataOverlay::{CHECK}(")
    end = glue.index("\n}", start)
    body = set(CODE.findall(glue[start:end]))
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if CHECK in p]
    stated = {code for p in paragraphs for code in CODE.findall(p)}
    assert body and body == stated, (
        f"{tag}: {CHECK} raises {sorted(body)}; the README paragraphs naming it name {sorted(stated)}"
    )


# T-155 / T-156 -- the checks on the committed files
# ----------------------------------------------------


@overlay.family
def test_t155_the_calls_a_patched_function_makes_outside_its_opt_in_test_are_the_ones_the_readme_names(
    tag: str,
):
    check_calls_outside_opt_in(tag, parsed(tag), readme())


def test_t155_every_definition_the_patch_renames_is_named_in_the_readme():
    check_renames_named(readme())


@overlay.family
def test_t156_the_codes_the_patch_raises_are_the_codes_the_readme_names(tag: str):
    check_codes_named(tag, parsed(tag), readme())


@overlay.family
def test_t156_the_readme_names_the_code_the_computed_rate_check_raises(tag: str):
    check_check_codes(tag, parsed(tag), readme())


# The drills
# ----------


def replaced(
    tag: str, old: bytes, new: bytes, expected: int, which: int | None = None
) -> dict[str, overlay.FilePatch]:
    """The family's behaviour patch, parsed after replacing `old` by `new` -- every occurrence, or
    only occurrence number `which` -- once `old` is shown to occur exactly `expected` times."""
    behaviour, _, _ = overlay.FAMILIES[tag]
    raw = behaviour.read_bytes()
    assert raw.count(old) == expected, (tag, old, raw.count(old))
    if which is None:
        raw = raw.replace(old, new)
    else:
        at = -1
        for _ in range(which + 1):
            at = raw.index(old, at + 1)
        raw = raw[:at] + new + raw[at + len(old) :]
    return overlay.by_new_path(overlay.parse_patch(raw))


CALL_LINE = b"  G4MuonicDataOverlay::CheckComputedCaptureRate(Z, A, lambda);"


@overlay.family
def test_t155_drill_a_second_call_outside_the_opt_in_test_is_named(tag: str):
    files = replaced(
        tag, CALL_LINE, b"  G4MuonicDataOverlay::CheckCascadeLevels(Z, A, nullptr, 0);", 2, which=1
    )
    with pytest.raises(AssertionError, match="CheckCascadeLevels"):
        check_calls_outside_opt_in(tag, files, readme())


@overlay.family
def test_t155_drill_the_call_moved_inside_the_opt_in_test_is_named(tag: str):
    inside = b"  if (G4MuonicDataTable::IsEnabled()) { " + CALL_LINE.strip() + b" }"
    files = replaced(tag, CALL_LINE, inside, 2)
    with pytest.raises(AssertionError, match=CHECK):
        check_calls_outside_opt_in(tag, files, readme())


@overlay.family
def test_t155_drill_a_readme_that_no_longer_names_the_call_is_refused(tag: str):
    text = readme()
    assert f"G4MuonicDataOverlay::{CHECK}" in text
    with pytest.raises(AssertionError, match=CHECK):
        check_calls_outside_opt_in(tag, parsed(tag), text.replace(f"G4MuonicDataOverlay::{CHECK}", CHECK))


@overlay.family
def test_t155_drill_a_check_of_another_value_than_the_one_returned_is_named(tag: str):
    files = replaced(
        tag, CALL_LINE, b"  G4MuonicDataOverlay::CheckComputedCaptureRate(Z, A, 0.);", 2, which=0
    )
    with pytest.raises(AssertionError, match=r"passes .*0\."):
        check_calls_outside_opt_in(tag, files, readme())


@overlay.family
def test_t155_drill_a_call_made_in_a_function_the_readme_does_not_call_a_capture_rate_function(tag: str):
    files = parsed(tag)
    headers = [
        hunk
        for file in files.values()
        for hunk in file.hunks
        if any(marker == b"+" and CALL_LINE in content for marker, content, _ in hunk.lines)
    ]
    assert headers, tag
    headers[-1].header = headers[-1].header.replace(b"::GetMuonCaptureRate(", b"::GetMuonZeff(")
    with pytest.raises(AssertionError, match="GetMuonZeff"):
        check_calls_outside_opt_in(tag, files, readme())


@overlay.family
def test_t155_drill_a_readme_that_no_longer_names_the_setters_call_is_refused(tag: str):
    text = readme()
    assert text.count("G4MuonicDataOverlay::Initialize") == 1
    with pytest.raises(AssertionError, match="Initialize"):
        check_calls_outside_opt_in(
            tag, parsed(tag), text.replace("G4MuonicDataOverlay::Initialize", "the glue's own")
        )


@overlay.family
def test_t155_drill_a_call_made_in_a_function_the_readme_does_not_name_is_named(tag: str):
    files = replaced(
        tag, b"G4HadronicParameters::SetEnableMuonicData(", b"G4HadronicParameters::ArmMuonicData(", 1
    )
    with pytest.raises(AssertionError, match="ArmMuonicData"):
        check_calls_outside_opt_in(tag, files, readme())


def test_t155_drill_a_readme_that_no_longer_names_the_renamed_definition_is_refused():
    ((_, (_, new)),) = overlay.RENAMES.items()
    name = re.search(rb"::(\w+)\(", new).group(1).decode()
    text = readme()
    assert f"`{name}`" in text
    with pytest.raises(AssertionError, match=name):
        check_renames_named(text.replace(f"`{name}`", "the private form"))


@overlay.family
def test_t156_drill_a_code_the_readme_does_not_name_is_named(tag: str):
    files = replaced(tag, b'Fatal("G4MuonicData004"', b'Fatal("G4MuonicData006"', 3, which=0)
    with pytest.raises(AssertionError, match="G4MuonicData006"):
        check_codes_named(tag, files, readme())


@overlay.family
def test_t156_drill_a_code_the_readme_drops_is_named(tag: str):
    text = readme()
    assert "G4MuonicData003" in text
    with pytest.raises(AssertionError, match="G4MuonicData003"):
        check_codes_named(tag, parsed(tag), text.replace("G4MuonicData003", "G4MuonicData"))


@overlay.family
def test_t156_drill_the_check_raising_another_code_is_refused(tag: str):
    files = replaced(
        tag, b'Fatal("G4MuonicData005", "under #PROFILE "', b'Fatal("G4MuonicData004", "under #PROFILE "', 1
    )
    check_codes_named(tag, files, readme())
    with pytest.raises(AssertionError, match=CHECK):
        check_check_codes(tag, files, readme())


@overlay.family
def test_t156_drill_the_readme_naming_another_code_beside_the_check_is_refused(tag: str):
    text = readme()
    paragraph = next(p for p in re.split(r"\n\s*\n", text) if CHECK in p)
    assert paragraph.count("G4MuonicData005") == 1, paragraph
    changed = text.replace(paragraph, paragraph.replace("G4MuonicData005", "G4MuonicData004"))
    check_codes_named(tag, parsed(tag), changed)
    with pytest.raises(AssertionError, match=CHECK):
        check_check_codes(tag, parsed(tag), changed)
