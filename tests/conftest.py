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

The second load-bearing fixture is the data-tree guard at the bottom of this file. It
fails the session if ``data/studies/`` changed while the suite ran. Its own rationale
is down there; what belongs up here is that the two are a pair -- the cache fixture is
what lets a test point the code at a tmp tree, and the guard is what notices when that
did not take.
"""

from __future__ import annotations

import hashlib
import subprocess
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


@pytest.fixture(autouse=True)
def _isolate_context(tmp_path_factory, monkeypatch):
    """Point the local knowledge layer at a tmp dir for every test.

    ``context/`` holds the biologist's own decisions, corrections, research profile
    and user memory. It is gitignored and local-only, and nothing was redirecting
    it: the suite read the REAL one. On this machine that meant every run saw a
    genuine ``celltype_override`` note, so ``composed_verdicts`` and the Summary
    table were tested against one person's saved judgement -- while a fresh clone,
    which has no ``context/`` at all, exercised a different path. That is the same
    shape as the gitignored input that made the suite red on a clone, and it fails
    in the direction that looks green.

    A per-test directory rather than a session one: a test that writes a note must
    not change what the next test sees. Nothing is deleted -- the real directory is
    simply not addressed.
    """
    monkeypatch.setenv("PANOSCOPE_CONTEXT_DIR",
                       str(tmp_path_factory.mktemp("context")))


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


@pytest.fixture(scope="session")
def _studies_copy(tmp_path_factory) -> Path:
    """A writable copy of ``data/studies/``, made once for the whole session.

    Some tests drive the real app through ``streamlit.testing.v1.AppTest``, and the
    app PERSISTS as it renders: ``ui/state._persist_threads`` writes
    ``interp/chat_threads.json`` on the first opening message and the Summary page
    writes ``interp/review_state.json``. Those are the biologist's local state and
    writing them is correct behaviour -- the bug was only ever the tree it happened
    in. Five modules rendered against the committed demo study, so every suite run
    left two untracked files in ``data/studies/`` that the next ``panoscope build``
    would read.

    A copy rather than a redirect to an empty tmp root, because these tests have to
    render a REAL study: an empty root gives an app with nothing to draw, which is a
    different test. A copy rather than hardlinks or symlinks, because the writes
    that matter here truncate an existing file and would reach the original through
    the shared inode. ~57 MB once per session, against a ~50 s suite.
    """
    from shutil import copytree

    dest = tmp_path_factory.mktemp("data_root")
    copytree(DATASETS_ROOT, dest / "studies")
    return dest


@pytest.fixture
def app_tree(_studies_copy: Path, monkeypatch) -> Path:
    """Point the whole stack at :func:`_studies_copy` for the duration of one test.

    ``layout.clear_cache()`` AFTER the setenv, never before: ``resolve_data_root``
    is an ``lru_cache`` keyed on the environment as it stood at its first call, and
    the autouse cache fixture runs before this one, so without the explicit clear
    the setenv redirects nothing at all.
    """
    from panoscope.data import layout

    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(_studies_copy))
    layout.clear_cache()
    clear_all_caches()
    yield _studies_copy / "studies"
    layout.clear_cache()


@pytest.fixture
def tmp_tree(tmp_path: Path) -> Path:
    """An empty datasets root, for pipeline writes that must not touch the repo."""
    root = tmp_path / "datasets"
    root.mkdir()
    return root


# --------------------------------------------------------------------------- #
# The data-tree guard.
#
# ``data/studies/`` is a committed, read-only INPUT to the suite. Twice in three
# rounds a test wrote into it anyway, and both times ``git status`` was silent,
# because every runtime path under ``interp/`` is gitignored on purpose
# (summary_edits.json, review_state.json, chat_threads.json are the biologist's
# local state, not build artifacts). A dirty tree that git reports as clean is
# invisible until someone rebuilds a study on top of it.
#
# So this cannot be implemented with ``git status`` -- seeing the gitignored paths
# is the entire point. It walks the tree itself.
# --------------------------------------------------------------------------- #
_TREE_BASELINE: dict[str, str] | None = None
_TREE_RESIDUE: list[str] | None = None


def snapshot_tree(root: Path = DATASETS_ROOT) -> dict[str, str]:
    """``{path relative to root: content digest}`` for every file under ``root``.

    Content, not merely the path list, and the cost was measured before choosing.
    On this tree (914 files, 57.4 MB): hashing 212 ms, walking and stat-ing alone
    3.6 ms, against a 47.5 s suite -- two snapshots are under 1% of the run. The
    weaker comparison would buy that 1% back and open a hole the suite has already
    fallen into once: the second leak was a probe string written over
    ``xenium_hbreast_sample1/interp/summary_edits.json``, a path that can legitimately
    already be on disk, so a path-set comparison reports it clean.

    An absent root snapshots as ``{}``. That is what the out-of-tree CI job sees --
    it copies ``tests/`` to a temp dir and moves the checkout aside -- and ``{}``
    compares equal to ``{}``, so the guard no-ops there instead of failing a job
    that has no data tree to protect.
    """
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.blake2b(
                p.read_bytes(), digest_size=16
            ).hexdigest()
        elif p.is_dir() and not any(p.iterdir()):
            # An EMPTY directory is recorded; a populated one is not, because the
            # file inside it already names the leak and recording both would report
            # one write as three paths. The empty case is its own shape and it is
            # the one this snapshot used to miss: ``ui/chat_store.save_all``
            # mkdir -p's ``<study>/interp/`` and then swallows the write (it is
            # fail-soft by design, so losing the transcript cannot break the app),
            # which leaves a directory under a study id that exists nowhere and not
            # one file to notice it by.
            out[str(p.relative_to(root)) + "/"] = "<empty dir>"
    return out


_IGNORED_NAMES = frozenset({".DS_Store"})


def untracked_paths(root: Path = DATASETS_ROOT) -> list[str]:
    """Files under ``root`` that git does not track -- i.e. residue, at ANY time.

    **The guard's second question, and the one it did not ask.** ``diff_tree``
    compares the tree against itself and so reports a leak only while it is being
    made: once the residue is on disk, every later run starts and ends with it
    present and the guard says the tree is clean. That is not hypothetical -- four
    fixes in one round were reported against a green suite whose green was only
    obtainable from a tree that already carried the leak, and the one guard watching
    for it had been silenced by it.

    ``git ls-files`` is the right oracle for THIS question and the wrong one for the
    other: the runtime files are gitignored on purpose, so ``git status`` cannot see
    them, but ``git ls-files`` still knows the tree is supposed to be 910 committed
    files and nothing else. ``.DS_Store`` is excluded because macOS writes it into
    any directory a Finder window has visited and no test put it there.

    An empty list when git is unavailable or ``root`` is outside a checkout: the
    out-of-tree CI job has no repository to ask, and a guard that fails there is a
    guard someone removes.
    """
    if not root.is_dir():
        return []
    try:
        tracked = set(subprocess.run(
            ["git", "ls-files", "-z", str(root)],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30,
        ).stdout.split("\0"))
    except (OSError, subprocess.SubprocessError):
        return []
    if not tracked - {""}:
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name not in _IGNORED_NAMES
        and str(p.relative_to(PROJECT_ROOT)) not in tracked
    )


def diff_tree(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Offending paths as ``+ added`` / ``- removed`` / ``~ modified``, sorted.

    Attribution is by path, which is the row identity here; a digest of the whole
    tree would say "something changed" and name nothing, and a guard that cannot
    name the file it is complaining about gets switched off.
    """
    lines = [f"+ {p}" for p in sorted(set(after) - set(before))]
    lines += [f"- {p}" for p in sorted(set(before) - set(after))]
    lines += [f"~ {p}" for p in sorted(p for p in set(before) & set(after)
                                       if before[p] != after[p])]
    return lines


_TREE_DIRTY = """\
The suite wrote into data/studies/. That tree is a committed, read-only input.

{paths}

  (+ added, - removed, ~ modified -- paths are relative to data/studies/)

`git status` will NOT show you this: everything a run writes under interp/ is
gitignored on purpose, so the working copy looks clean while carrying state that
the next `panoscope build` will read.

Nine times out of ten the cause is a test that redirected the data root with

    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(tmp_path))

and did not then call

    from panoscope.data import layout; layout.clear_cache()

AFTER the setenv. `layout.resolve_data_root` is an lru_cache keyed on the
environment as it was at first call, so without that clear the code under test
keeps resolving the real repo tree. conftest's autouse cache fixture runs BEFORE
your setenv, not after it, so it cannot do this for you.

Note what `clear_cache()` does not reach: a module that froze a root at ITS OWN
import time -- a module-level ``X = cfg.DATA_DIR_PATH / ...`` -- writes to the real
tree no matter what the environment says afterwards. If the path above persists
with the clear in place, look for one of those.

Use the `tmp_tree` fixture for anything that writes, and delete what this run left
behind before you commit."""


_TREE_RESIDUE_MSG = """\
data/studies/ was ALREADY dirty when this run started.

{paths}

  (paths are relative to data/studies/; every one is untracked by git)

This is not something this run did -- it is what an earlier run left behind, and
while it sits there the change-based half of this guard has nothing to report: the
tree starts and ends identical, so a leaked study reads as a clean tree. Four fixes
in one round were verified against a suite that was green for exactly this reason.

`git status` will not show you these: the runtime paths under interp/ are gitignored
on purpose. Delete them, then re-run. If they come straight back, the test that
writes them is reading the real data root -- see the message above this one.
"""


def pytest_configure(config):
    """Take the baseline before collection, not at first test.

    Collection imports every test module, and a module-level probe is as capable of
    writing as a test body is. A session-scoped fixture's setup runs after all of
    that, so the baseline would already contain anything collection wrote.
    """
    global _TREE_BASELINE, _TREE_RESIDUE
    _TREE_BASELINE = snapshot_tree()
    _TREE_RESIDUE = untracked_paths()


@pytest.fixture(scope="session", autouse=True)
def _guard_data_tree():
    """Fail the session if data/studies/ changed while the suite ran.

    Order-independent by construction: it compares the whole tree against the whole
    tree and reports paths, so it neither depends on nor imposes an order. Under
    pytest-xdist (not installed here, and the CI workflow does not use it) this runs
    once per worker against a baseline each worker took at its own start, so a leak
    from any worker is caught by every worker that outlives it -- the report names
    the path, never the test, so cross-worker attribution does not arise.
    """
    yield
    changed = diff_tree(_TREE_BASELINE or {}, snapshot_tree())
    problems = []
    if changed:
        problems.append(
            _TREE_DIRTY.format(paths="\n".join("    " + c for c in changed))
        )
    # Reported even when nothing changed, which is the point: residue already on
    # disk makes the comparison above structurally blind to itself.
    if _TREE_RESIDUE:
        problems.append(
            _TREE_RESIDUE_MSG.format(
                paths="\n".join("    " + c for c in _TREE_RESIDUE)
            )
        )
    if problems:
        pytest.fail("\n\n".join(problems), pytrace=False)


@pytest.fixture
def tree_guard():
    """This module itself, for the guard's own test file.

    A fixture rather than an ``import conftest``: the name this module lands under
    in ``sys.modules`` differs between the normal run (``tests.conftest``, because
    ``tests/`` is a package and pytest puts the rootdir on sys.path) and the
    out-of-tree CI job, which copies ``tests/`` elsewhere and moves the checkout
    aside. A test file that guesses the name fails at collection in one of the two.
    This used to rest on ``pytest.ini``'s ``pythonpath = .``; that setting is gone
    (the config now lives in ``pyproject.toml``) and the reasoning is unchanged.
    """
    return sys.modules[__name__]
