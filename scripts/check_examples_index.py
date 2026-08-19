"""Guard that README's Examples table lists exactly the example scripts on disk.

The README carries a "## Examples" table with one row per runnable script in
``examples/``. It drifts silently: a new ``examples/*.py`` lands without a row,
or a row outlives the file it points to. This check compares the set of
``examples/*.py`` files (excluding ``__init__.py``) against the set of
``examples/<name>.py`` paths referenced as rows under the "## Examples" heading,
and fails loudly on any mismatch.

Run from the repo root::

    python scripts/check_examples_index.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
README = REPO_ROOT / "README.md"

# Matches an examples/<name>.py path anywhere on a table row line.
ROW_PATH = re.compile(r"examples/([A-Za-z0-9_./-]+\.py)")


def files_on_disk() -> set[str]:
    """Every examples/*.py filename on disk except __init__.py."""
    return {p.name for p in EXAMPLES_DIR.glob("*.py") if p.name != "__init__.py"}


def files_in_table() -> set[str]:
    """Every examples/<name>.py referenced as a row in the README Examples table."""
    lines = README.read_text(encoding="utf-8").splitlines()
    in_table = False
    referenced: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            # Enter the section on the Examples heading; leave it on the next.
            in_table = stripped == "## Examples"
            continue
        if not in_table:
            continue
        # Only count genuine table rows (lines that start a Markdown table cell).
        if not stripped.startswith("|"):
            continue
        for match in ROW_PATH.finditer(line):
            referenced.add(match.group(1))
    return referenced


def main() -> int:
    on_disk = files_on_disk()
    in_table = files_in_table()

    if on_disk == in_table:
        print(f"examples index OK ({len(on_disk)} examples)")
        return 0

    missing_from_table = sorted(on_disk - in_table)
    no_file = sorted(in_table - on_disk)

    if missing_from_table:
        print("missing from README table:")
        for name in missing_from_table:
            print(f"  examples/{name}")
    if no_file:
        print("in table but no file:")
        for name in no_file:
            print(f"  examples/{name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
