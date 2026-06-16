"""Pull floe-guard's public adoption metrics into a single snapshot.

No dependencies — stdlib only. Reads public APIs:
  - GitHub repo stats (stars, forks, watchers)         — no auth needed
  - GitHub traffic (views, clones)                     — needs a token with push access
  - PyPI downloads (last day / week / month)           — no auth needed (pypistats.org)

Usage:
    python scripts/metrics.py
    GITHUB_TOKEN=ghp_xxx python scripts/metrics.py   # also fetch traffic (views/clones)

These are the runbook KPIs (star velocity, forks, install velocity). Run it on a
schedule and diff snapshots to get velocity. It only reports real numbers — if a
source is unreachable it says so rather than guessing.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

REPO = "Floe-Labs/floe-guard"
PACKAGE = "floe-guard"


def _get(
    url: str, headers: dict[str, str] | None = None, timeout: float = 15.0
) -> dict[str, object]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted hosts)
        return json.loads(resp.read().decode())


def github_repo() -> dict[str, object]:
    try:
        d = _get(
            f"https://api.github.com/repos/{REPO}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "floe-guard-metrics"},
        )
        return {
            "stars": d.get("stargazers_count"),
            "forks": d.get("forks_count"),
            "watchers": d.get("subscribers_count"),
            "open_issues": d.get("open_issues_count"),
        }
    except (urllib.error.URLError, ValueError) as e:
        return {"error": f"github repo stats unavailable: {e}"}


def github_traffic() -> dict[str, object]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return {"note": "set GITHUB_TOKEN (push access) to fetch views/clones"}
    hdr = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "floe-guard-metrics",
    }
    out: dict[str, object] = {}
    for kind in ("views", "clones"):
        try:
            d = _get(f"https://api.github.com/repos/{REPO}/traffic/{kind}", headers=hdr)
            out[kind] = {"count": d.get("count"), "uniques": d.get("uniques")}
        except (urllib.error.URLError, ValueError) as e:
            out[kind] = {"error": str(e)}
    return out


def pypi_downloads() -> dict[str, object]:
    try:
        d = _get(
            f"https://pypistats.org/api/packages/{PACKAGE}/recent",
            headers={"User-Agent": "floe-guard-metrics"},
        )
        return d.get("data", {"error": "no data field"})
    except (urllib.error.URLError, ValueError) as e:
        return {"error": f"pypistats unavailable (new packages take ~1 day to appear): {e}"}


def main() -> None:
    snapshot = {
        "repo": REPO,
        "package": PACKAGE,
        "github": github_repo(),
        "traffic": github_traffic(),
        "pypi_downloads_recent": pypi_downloads(),
    }
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
