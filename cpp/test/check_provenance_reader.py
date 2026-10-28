"""Differential check of the compiled provenance reader (G4MuonicDataOverlay::ProvenanceRowKeys).

Every document of a corpus -- the shipped provenance file, re-serialized, reordered, CRLF and
Unicode-metadata variants of it, fixtures that try to bind another roster through escaped keys, invalid
escapes, invalid literals and numbers, invalid UTF-8, structural damage, nesting at the depth limit, and
seeded random byte edits -- is read by the compiled reader and by Python's json module. A document is admitted
when json.loads accepts the UTF-8 text and it is one object with exactly one "rows" member whose value is an
object; the roster is that object's keys in document order. The reader may differ from that in one direction
only, and only for a documented intended refusal:

* escaped-key: an object key spelled with a backslash (the generator never escapes a key; decoding one would
  let a key stand for another);
* nonstandard-constant: NaN, Infinity or -Infinity, which json.loads accepts and RFC 8259 does not;
* depth: containers nested deeper than 64.

Any other disagreement, a document the reader admits that Python refuses, or a different roster fails the
check.

usage: check_provenance_reader.py <g4muonicdata_bound_rate_check> <shipped .prov.json> <fixture dir>
       <work dir>
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import subprocess
import sys

MAX_DEPTH = 64
#: Characters allowed on one reader command line; Windows refuses a command line longer than 32,767.
COMMAND_LINE_BUDGET = 24_000
STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


class Members(list):  # type: ignore[type-arg]
    """An object's (key, value) pairs in document order."""


def depth(value: object) -> int:
    if isinstance(value, Members):
        return 1 + max((depth(v) for _, v in value), default=0)
    if isinstance(value, list):
        return 1 + max((depth(v) for v in value), default=0)
    return 0


def refuse_constant(name: str) -> float:
    raise ValueError(f"nonstandard constant {name}")


def python_reading(data: bytes) -> tuple[list[str] | None, list[str]]:
    """(roster or None, the intended-refusal reasons that apply). The reasons are only consulted when the
    reader refuses a document Python admits."""
    reasons: list[str] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, reasons
    try:
        doc = json.loads(text, object_pairs_hook=Members)
    except (ValueError, RecursionError):
        return None, reasons
    if not isinstance(doc, Members):
        return None, reasons
    rows = [v for k, v in doc if k == "rows"]
    if len(rows) != 1 or not isinstance(rows[0], Members):
        return None, reasons
    try:
        json.loads(text, parse_constant=refuse_constant)
    except ValueError:
        reasons.append("nonstandard-constant")
    # json.loads accepted the text, so every quoted token is a string; one followed by ':' is a key
    for match in STRING.finditer(text):
        if "\\" in match.group() and text[match.end() :].lstrip(" \t\n\r").startswith(":"):
            reasons.append("escaped-key")
            break
    if depth(doc) > MAX_DEPTH:
        reasons.append("depth")
    return [k for k, _ in rows[0]], reasons


def nest(levels: int) -> str:
    return "[" * levels + "]" * levels


def corpus(shipped: pathlib.Path, fixtures: pathlib.Path) -> dict[str, bytes]:
    base = shipped.read_bytes()
    doc = json.loads(base)
    out: dict[str, bytes] = {"shipped": base}
    for path in sorted(fixtures.glob("*.prov.json")):
        out[f"fixture-{path.name}"] = path.read_bytes()
    out["reserialized-compact"] = json.dumps(doc, separators=(",", ":")).encode()
    out["reserialized-indent1"] = json.dumps(doc, indent=1).encode()
    out["reordered"] = json.dumps(dict(reversed(list(doc.items()))), indent=2).encode()
    out["sorted-keys"] = json.dumps(doc, indent=2, sort_keys=True).encode()
    out["crlf"] = base.replace(b"\n", b"\r\n")
    out["tabs-and-trailing-space"] = base.replace(b"  ", b"\t") + b" \r\n\t"
    note = "".join(
        map(chr, (0x6D, 0x75, 0xB5, 0x20, 0x2013, 0x20, 0xFC, 0x20, 0x65E5, 0x672C, 0x20, 0x1F600))
    )
    unicode_doc = dict(doc, note=note, escaped="a/b" + chr(34) + "c" + chr(10) + chr(0xB5))
    out["unicode-metadata-raw"] = json.dumps(unicode_doc, ensure_ascii=False, indent=2).encode()
    out["unicode-metadata-escaped"] = json.dumps(unicode_doc, ensure_ascii=True, indent=2).encode()
    out["surrogate-pair-escape"] = json.dumps(dict(doc, note="\U0001f600"), ensure_ascii=True).encode()
    out["valid-value-escapes"] = base.replace(
        b'"G4MuonicData"', b'"G4\\/\\"\\\\\\b\\f\\n\\r\\t\\u00B5\\u00b5Data"', 1
    )
    out["lone-surrogate-escape"] = base.replace(b'"dataset": "G4MuonicData"', b'"dataset": "\\ud800"')
    numbers = ["0", "-0", "1.5", "-1.5e-3", "1E+2", "1e999", "123456789012345678901234567890"]
    out["numbers"] = (
        json.dumps(dict(doc, numbers=0))
        .replace('"numbers": 0', '"numbers": [' + ", ".join(numbers) + "]")
        .encode()
    )
    out["literals"] = json.dumps(dict(doc, flags=[True, False, None])).encode()
    out["depth-64"] = json.dumps(doc).replace('"dataset"', '"deep": ' + nest(63) + ', "dataset"', 1).encode()
    out["depth-65"] = json.dumps(doc).replace('"dataset"', '"deep": ' + nest(64) + ', "dataset"', 1).encode()
    head = base.split(b'"rows"', 1)
    edits = {
        "escaped-rows-key": b'"\\u0072ows"',
        "escaped-cr-rows-key": b'"\\rows"',
        "escaped-solidus-key": b'"\\/rows"',
    }
    for name, key in edits.items():
        out[name] = head[0] + key + head[1]
    out["escaped-metadata-key"] = base.replace(b'"dataset"', b'"d\\u0061taset"', 1)
    out["escaped-row-key"] = base.replace(b'"13-27"', b'"13\\u002d27"', 1)
    for name, token in {
        "invalid-escape-x": b"\\x41",
        "invalid-escape-quote": b"\\'",
        "short-unicode": b"\\u12",
        "bad-hex-unicode": b"\\u12G4",
        "raw-tab": b"\t",
        "raw-newline": b"\n",
        "raw-nul": b"\x00",
        "raw-del": b"\x7f",
    }.items():
        out[f"value-{name}"] = base.replace(b'"G4MuonicData"', b'"G4Muonic' + token + b'Data"', 1)
    for name, token in {
        "tru": b"tru",
        "True": b"True",
        "nul": b"nul",
        "nulll": b"nulll",
        "leading-zero": b"01",
        "plus": b"+1",
        "dot-only": b"1.",
        "bare-dot": b".5",
        "exp-only": b"1e",
        "minus": b"-",
        "hex": b"0x10",
        "NaN": b"NaN",
        "Infinity": b"Infinity",
        "neg-Infinity": b"-Infinity",
        "word": b"openmucf",
        "single-quoted": b"'x'",
    }.items():
        out[f"primitive-{name}"] = base.replace(b'"G4MuonicData"', token, 1)
    structural = {
        "trailing-comma": base.replace(b'"openmucf"\n  ]', b'"openmucf",\n  ]', 1),
        "missing-comma": base.replace(b'"G4MuonicData",', b'"G4MuonicData"', 1),
        "trailing-garbage": base + b"x",
        "two-documents": base + base,
        "bom": b"\xef\xbb\xbf" + base,
        "overlong-utf8": base.replace(b"G4MuonicData", b"G4\xc0\xafMuonicData", 1),
        "encoded-surrogate": base.replace(b"G4MuonicData", b"G4\xed\xa0\x80MuonicData", 1),
        "beyond-unicode": base.replace(b"G4MuonicData", b"G4\xf5\x80\x80\x80MuonicData", 1),
        "lone-continuation": base.replace(b"G4MuonicData", b"G4\x80MuonicData", 1),
        "truncated-utf8": base.replace(b"G4MuonicData", b"G4\xe2\x82MuonicData", 1),
        "duplicate-rows": base.replace(b'"profile"', b'"rows": {}, "profile"', 1),
        "rows-list": head[0] + b'"rows": [], "withheld"' + head[1],
        "rows-absent": head[0] + b'"withheld"' + head[1],
        "rows-string": head[0] + b'"rows": "13-27", "withheld"' + head[1],
        "empty-rows": head[0] + b'"rows": {}, "withheld"' + head[1],
        "duplicate-row-key": base.replace(b'"rows": {', b'"rows": {"13-27": {}, ', 1),
        "top-array": b"[" + base + b"]",
        "empty": b"",
        "whitespace-only": b" \n",
        "unterminated-string": base[: base.index(b"G4MuonicData") + 4],
    }
    out.update(structural)
    rng = random.Random(4172)
    alphabet = b'{}[]:,"\\ \t\r\nu0123456789aeEfnrstl-+.' + bytes(range(0x80, 0x100, 7))
    for k in range(4000):
        data = bytearray(base)
        for _ in range(rng.randint(1, 3)):
            at = rng.randrange(len(data))
            op = rng.random()
            byte = alphabet[rng.randrange(len(alphabet))]
            if op < 0.4:
                data[at] = byte
            elif op < 0.7:
                data.insert(at, byte)
            else:
                del data[at]
        out[f"random-{k:04d}"] = bytes(data)
    return out


def batches(head: list[str], paths: list[pathlib.Path], budget: int = COMMAND_LINE_BUDGET) -> list[list[str]]:
    """Split ``paths`` into consecutive batches whose command line ``head + batch`` stays within ``budget``
    characters, counting each argument with a separating space and two quotes. A batch holds at least one
    path, so a single path longer than the budget still runs, alone."""
    base = sum(len(arg) + 3 for arg in head)
    out: list[list[str]] = []
    batch: list[str] = []
    size = base
    for path in map(str, paths):
        cost = len(path) + 3
        if batch and size + cost > budget:
            out.append(batch)
            batch, size = [], base
        batch.append(path)
        size += cost
    if batch:
        out.append(batch)
    return out


def main() -> int:
    reader, shipped, fixtures, work = (pathlib.Path(a) for a in sys.argv[1:5])
    work.mkdir(parents=True, exist_ok=True)
    documents = corpus(shipped, fixtures)
    paths = []
    for name, data in documents.items():
        path = work / f"{name}.json"
        path.write_bytes(data)
        paths.append(path)
    answers: list[str] = []
    head = [str(reader), "provenance-rows"]
    for batch in batches(head, paths):
        run = subprocess.run(
            [*head, *batch],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            print(f"FAIL: reader rc {run.returncode}")
            return 1
        answers += [line for line in run.stdout.splitlines() if line.startswith(("ACCEPT", "REFUSE"))]
    if len(answers) != len(documents):
        print(f"FAIL: {len(answers)} answers for {len(documents)} documents")
        return 1
    silent = 0
    tally: dict[str, int] = {"agree-accept": 0, "agree-refuse": 0}
    for (name, data), answer in zip(documents.items(), answers, strict=True):
        roster, reasons = python_reading(data)
        if answer == "REFUSE":
            if roster is None:
                tally["agree-refuse"] += 1
            elif reasons:
                tally[reasons[0]] = tally.get(reasons[0], 0) + 1
            else:
                silent += 1
                print(f"SILENT MISMATCH {name}: the reader refuses what Python admits with roster {roster}")
            continue
        keys = [bytes.fromhex(word[1:]).decode("utf-8") for word in answer.split()[1:]]
        if roster is None or reasons or keys != roster:
            silent += 1
            print(f"SILENT MISMATCH {name}: the reader admits {keys}; Python {roster} {reasons}")
        else:
            tally["agree-accept"] += 1
    print(
        f"documents {len(documents)} "
        + " ".join(f"{k} {v}" for k, v in sorted(tally.items()))
        + f" silent-mismatches {silent}"
    )
    required = (
        "fixture-escape-decoy.prov.json",
        "fixture-full-generated-metadata-decoy.prov.json",
        "fixture-missing-actual-rows.prov.json",
    )
    for name, answer in zip(documents, answers, strict=True):
        if name in required and answer != "REFUSE":
            print(f"FAIL: {name} is not refused")
            silent += 1
    if [a for n, a in zip(documents, answers, strict=True) if n == "shipped"] != ["ACCEPT x31332d3237"]:
        print("FAIL: the shipped provenance is not read as the one row 13-27")
        silent += 1
    return 1 if silent or any(name not in documents for name in required) else 0


if __name__ == "__main__":
    sys.exit(main())
