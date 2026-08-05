"""`floe_guard.__version__` must match what the package actually shipped as.

The published 0.11.0 wheel reports `__version__ == "0.10.0"`, because the value
was a hand-maintained literal that had to be bumped in lockstep with
pyproject.toml and was missed. Anyone reading the version for a bug report, a
telemetry tag or a compatibility check got the previous release.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import pytest

import floe_guard


def test_version_matches_installed_package_metadata():
    try:
        installed = package_version("floe-guard")
    except PackageNotFoundError:
        pytest.skip("floe-guard is not installed; nothing to compare against")
    assert floe_guard.__version__ == installed


def test_version_is_a_non_empty_string():
    assert isinstance(floe_guard.__version__, str)
    assert floe_guard.__version__
