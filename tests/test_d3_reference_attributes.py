"""A checkout must preserve the digested primitive evidence bytes on every platform."""
import subprocess
from pathlib import Path


def test_reference_evidence_has_no_checkout_translation():
    root = Path(__file__).resolve().parents[1]
    files = sorted(p.relative_to(root).as_posix() for p in (root / 'data/g4/d3/reference').rglob('*')
                   if p.is_file())
    assert files
    command = ['git', '--no-optional-locks', '-C', str(root), 'check-attr', 'text', '--', *files]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [f'{path}: text: unset' for path in files]
