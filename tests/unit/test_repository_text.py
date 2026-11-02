"""Text files of the repository contain no control characters.

Such characters arise, for example, when LaTeX such as ``\\big`` or ``\\vert``
passes through a non-raw string, where ``\\b`` and ``\\v`` are escape
sequences. Sphinx does not warn about them; MathJax silently renders broken
formulas. Tab, line feed and carriage return are allowed.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TEXT_SUFFIXES = {".md", ".py", ".rst", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".bib"}
SKIP_DIRS = {".git", ".venv", ".nox", "_build", "generated", "__pycache__"}
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def text_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if SKIP_DIRS.intersection(relative.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def test_text_files_are_found():
    files = [p.relative_to(ROOT).as_posix() for p in text_files()]
    if "pyproject.toml" not in files:  # pragma: no cover - installed-wheel job
        pytest.skip("not running from a source checkout")
    assert "docs/theory/schemes/euler_maruyama.md" in files


def test_no_control_characters_in_text_files():
    offending = []
    for path in text_files():
        # Split at line feeds only: str.splitlines() would treat U+000B and
        # U+000C as line breaks and hide them.
        lines = path.read_text(encoding="utf-8").split("\n")
        for number, line in enumerate(lines, 1):
            for match in CONTROL.finditer(line):
                offending.append(
                    f"{path.relative_to(ROOT).as_posix()}:{number}: "
                    f"U+{ord(match.group()):04X} before {line[match.end() :][:12]!r}"
                )
    assert not offending, "control characters found:\n" + "\n".join(offending)
