"""In-README anchor link-check.

Every `](#anchor)` link in the README must resolve to a heading. The slug is
computed with GitHub's algorithm — critically, each space becomes its own hyphen
(no collapsing), so a heading like `## Voice adapters (STT -> LLM -> TTS)` yields
`voice-adapters-stt--llm--tts` with *double* hyphens. A naive checker that
collapses `--` to `-` reports those live anchors as dead (a false positive); this
one does not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
READMES = [REPO_ROOT / "README.md", REPO_ROOT / "js" / "README.md"]


def github_slug(heading: str) -> str:
    """Match GitHub's heading-anchor slugifier (github-slugger)."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)  # keep word chars, whitespace, hyphens
    return text.replace(" ", "-")  # each space -> one hyphen; NO collapse


def heading_anchors(markdown: str) -> set[str]:
    slugs: list[str] = [
        github_slug(m) for m in re.findall(r"^#{1,6}\s+(.*)$", markdown, re.M)
    ]
    # GitHub disambiguates repeated headings with -1, -2, ... suffixes.
    seen: dict[str, int] = {}
    resolved: set[str] = set()
    for slug in slugs:
        if slug in seen:
            seen[slug] += 1
            resolved.add(f"{slug}-{seen[slug]}")
        else:
            seen[slug] = 0
            resolved.add(slug)
    return resolved


def internal_links(markdown: str) -> list[str]:
    return re.findall(r"\]\(#([\w-]+)\)", markdown)


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_readme_internal_anchors_resolve(path: Path) -> None:
    markdown = path.read_text(encoding="utf-8")
    anchors = heading_anchors(markdown)
    dead = sorted({link for link in internal_links(markdown) if link not in anchors})
    assert not dead, f"{path.name}: dead in-README anchors: {dead}"
