"""Shared test fixtures.

The autouse cache fixture is the load-bearing one. Every reader in this codebase is
wrapped in ``functools.lru_cache``, most at ``maxsize=1``, and those caches are
process-global. Today that is invisible because every test opens the same study. The
moment one test opens a second study, every later test in the same process silently
receives the first study's data — and the failure looks like a refactor bug rather
than a fixture bug.

Rather than hard-code the list (14 functions today, and the migration moves all of
them), the fixture walks ``sys.modules`` and clears anything exposing ``cache_clear``.
A module that was never imported has no cache to go stale, so iterating loaded modules
is both complete and self-maintaining.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Package prefixes whose caches are ours to clear. Keep in sync with the package
# layout; the migration renames these under ``panoscope.`` (IMPLEMENTATION.md §1).
_OWNED_PREFIXES = ("agent", "ui", "pipeline", "panoscope", "app")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "data" / "studies"

DEMO_ID = "xenium_hbreast_sample1"
MINI_ID = "demo_mini"


def _owned_modules():
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        root = name.split(".", 1)[0]
        if root in _OWNED_PREFIXES:
            yield mod


def clear_all_caches() -> int:
    """Clear every ``lru_cache`` reachable from a loaded first-party module.

    Returns the number of caches cleared, so a test can assert the fixture is
    actually finding them rather than silently no-opping.
    """
    cleared = 0
    for mod in _owned_modules():
        for attr in list(vars(mod).values()):
            clear = getattr(attr, "cache_clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:  # noqa: BLE001 - a broken cache must not break the suite
                    continue
                cleared += 1
    return cleared


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Clear first-party caches before and after every test.

    Before, so a test never inherits state. After, so a test that deliberately
    switches study does not leak into the next one.
    """
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture
def datasets_root() -> Path:
    return DATASETS_ROOT


@pytest.fixture
def demo_id() -> str:
    return DEMO_ID


@pytest.fixture
def mini_id() -> str:
    """The 3-cluster fixture study.

    This is the second study the plurality gate needs (IMPLEMENTATION.md §7, Phase D).
    It is deliberately tiny so it can be committed and run in CI.
    """
    return MINI_ID


@pytest.fixture
def tmp_tree(tmp_path: Path) -> Path:
    """An empty datasets root, for pipeline writes that must not touch the repo."""
    root = tmp_path / "datasets"
    root.mkdir()
    return root
