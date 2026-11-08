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
import os
import shutil
import socket
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

# Package prefixes whose caches are ours to clear. Keep in sync with the package
# layout; the migration renames these under ``panoscope.`` (IMPLEMENTATION.md §1).
_OWNED_PREFIXES = ("agent", "ui", "pipeline", "panoscope", "app")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "data" / "studies"

DEMO_ID = "xenium_hbreast_sample1"
MINI_ID = "demo_mini"

# Property tests (tests/test_properties.py). Per commit: derandomised, 200 examples,
# no example database, so the suite is reproducible and writes no .hypothesis/.
# Nightly: HYPOTHESIS_PROFILE=deep, random, 2000 examples; a failure it finds
# becomes an @example in the commit suite.
try:
    from hypothesis import settings as _hyp

    _hyp.register_profile("commit", derandomize=True, max_examples=200,
                          database=None, deadline=None)
    _hyp.register_profile("deep", max_examples=2000, database=None, deadline=None)
    _hyp.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "commit"))
except ImportError:  # dev extra not installed; test_properties.py skips itself
    pass


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
    # Same for fetched reference data: a source cached on this machine must not
    # change a canonical list the suite asserts on.
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR",
                       str(tmp_path_factory.mktemp("reference")))
    # Injection beats the environment, and a previous test may have set one.
    from panoscope.agent import tools as _tools

    _tools.set_memory_base_dir(None)


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
def demo(demo_id: str, datasets_root: Path):
    """The committed demo study, opened read-only."""
    from panoscope.data.study import Study

    return Study.open(demo_id, data_root=datasets_root.parent).require()


def copy_studies(dest: Path) -> Path:
    """Copy ``data/studies/`` to ``dest``, symlinking only each study's ``viz/``.

    ``viz/`` is 50 MB of precomputed frames nothing writes; everything else (~1.5 MB)
    is copied, because a symlinked ``inputs/`` or ``interp/`` is writable through the
    link and a write would land in the repo.
    """
    for study in DATASETS_ROOT.iterdir():
        if not study.is_dir():
            continue
        shutil.copytree(study, dest / study.name, ignore=shutil.ignore_patterns("viz"))
        if (study / "viz").is_dir():
            (dest / study.name / "viz").symlink_to(study / "viz", target_is_directory=True)
    return dest


@pytest.fixture(scope="session")
def _studies_copy(tmp_path_factory) -> Path:
    """A writable copy of ``data/studies/``, made once for the whole session.

    Only the browser server uses this now (one process for a whole module); every
    in-process test gets its own tree from :func:`app_tree`.

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
def app_tree(tmp_path: Path, monkeypatch) -> Path:
    """Point the whole stack at a per-test copy of ``data/studies/`` (:func:`copy_studies`).

    Per test, not per session: the app persists as it renders (decisions, sign-offs,
    chat threads), and a session-shared tree let one test's decision leak into the
    next -- ``test_interactive_journey.py`` before
    ``test_decisions.py::test_the_bundled_demo_opens_with_every_cluster_open`` failed.

    ``layout.clear_cache()`` AFTER the setenv, never before: ``resolve_data_root``
    is an ``lru_cache`` keyed on the environment as it stood at its first call, and
    the autouse cache fixture runs before this one, so without the explicit clear
    the setenv redirects nothing at all.
    """
    from panoscope.data import layout

    root = tmp_path / "data_root"
    copy_studies(root / "studies")
    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(root))
    layout.clear_cache()
    clear_all_caches()
    yield root / "studies"
    layout.clear_cache()


@pytest.fixture
def isolated(app_tree: Path, tmp_path: Path, monkeypatch):
    """The journey-walk harness: a per-test data tree, HOME, no model key, and Europe
    PMC id lookups answered from ``.ids`` (a dict the test fills: pid -> SourceRecord).

    The context and reference dirs are already per-test (autouse ``_isolate_context``).
    """
    from panoscope.agent import loop
    from panoscope.corpus import exchange

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PANOSCOPE_DATASET", raising=False)
    monkeypatch.setattr(loop, "has_model_key", lambda: False)
    ids: dict = {}
    monkeypatch.setattr(exchange, "resolve_id", lambda pid, retrieved: ids.get(pid))
    return SimpleNamespace(root=app_tree, ids=ids)


@pytest.fixture
def study_builder(tmp_path: Path, monkeypatch):
    """``make_study`` bound to a fresh data root the whole stack points at.

    ``study_builder("s", labels=False)`` writes ``<root>/studies/s`` and returns it;
    ``study_builder.root`` is the data root. See tests/fixtures/studies.py.
    """
    from functools import partial

    from panoscope.data import layout
    from tests.fixtures.studies import make_study

    root = tmp_path / "built_root"
    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(root))
    layout.clear_cache()
    clear_all_caches()
    build = partial(make_study, root)
    build.root = root
    yield build
    layout.clear_cache()


@pytest.fixture
def fake_model(monkeypatch):
    """A scripted Anthropic client behind the REAL agent loop (tests/fixtures/fake_model.py)."""
    from tests.fixtures.fake_model import install

    return install(monkeypatch)


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch) -> list:
    """Block AF_INET/AF_INET6 connects in every test not marked ``live`` or ``browser``.

    A blocked connect raises ``OSError``, exactly what an offline laptop gives, so
    fail-soft code degrades as it would there. AF_UNIX stays open for subprocess
    tests. Request this fixture by name to assert on the attempted addresses.
    """
    attempts: list = []
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("browser"):
        return attempts
    real, real_ex = socket.socket.connect, socket.socket.connect_ex

    def _guard(fn):
        def connect(self, address, *a):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                attempts.append(address)
                raise OSError(f"network blocked in the offline suite: {address!r}")
            return fn(self, address, *a)
        return connect

    monkeypatch.setattr(socket.socket, "connect", _guard(real))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(real_ex))
    return attempts


# --------------------------------------------------------------------------- #
# Requirements traceability (tests/requirements.toml).
#
# ``@pytest.mark.req("ARCH-9.5-01")`` names the spec rows a test covers; an id not
# in the registry is a usage error at collection. ``gap=True`` makes the test a
# strict xfail that must fail on an AssertionError: it turns red the day the gap is
# built, and a fixture error or typo cannot hide inside it. Built / gap is derived
# from the tests, never stored in the registry.
# --------------------------------------------------------------------------- #
REQS_PATH = Path(__file__).with_name("requirements.toml")
_DEFAULT_OFF = ("browser", "budget")  # deselected unless -m names them


def load_requirements() -> dict[str, dict]:
    with REQS_PATH.open("rb") as f:
        return {r["id"]: r for r in tomllib.load(f)["req"]}


def pytest_addoption(parser):
    parser.addoption("--trace-report", metavar="PATH",
                     help="write a requirement -> tests table (markdown) to PATH")
    parser.addoption("--trace-strict", action="store_true",
                     help="fail when a requirement has no test and no deferred/decision status")


def pytest_collection_modifyitems(config, items):
    reqs = load_requirements()
    bad = []
    for item in items:
        for m in item.iter_markers("req"):
            if unknown := [i for i in m.args if i not in reqs]:
                bad.append(f"  {item.nodeid}: {unknown}")
            if m.kwargs.get("gap"):
                item.add_marker(pytest.mark.xfail(
                    strict=True, raises=AssertionError, reason=f"{', '.join(m.args)}: not built"))
    if bad:
        raise pytest.UsageError("requirement ids not in tests/requirements.toml:\n" + "\n".join(bad))
    markexpr = config.getoption("markexpr") or ""
    off = [n for n in _DEFAULT_OFF if n not in markexpr]
    keep = [i for i in items if not any(i.get_closest_marker(n) for n in off)]
    if len(keep) < len(items):
        config.hook.pytest_deselected(items=[i for i in items if i not in keep])
        items[:] = keep
    if os.environ.get("PANOSCOPE_TEST_ORDER") == "reverse":
        items.reverse()


def pytest_collection_finish(session):
    path, strict = (session.config.getoption("--trace-report"),
                    session.config.getoption("--trace-strict"))
    if not (path or strict):
        return
    reqs = load_requirements()
    tests, built = defaultdict(set), set()
    for item in session.items:
        for m in item.iter_markers("req"):
            for i in m.args:
                tests[i].add(item.nodeid.split("[")[0])
                if not m.kwargs.get("gap"):
                    built.add(i)

    def state(i):
        return "built" if i in built else "gap" if i in tests else reqs[i].get("status", "none")

    counts = defaultdict(int)
    for i in reqs:
        counts[state(i)] += 1
    if path:
        out = [f"# Traceability: {len(reqs)} requirements",
               "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), "",
               "| id | state | requirement | tests |", "|---|---|---|---|"]
        for i, r in reqs.items():
            cell = "<br>".join(sorted(tests.get(i, ()))) or r.get("reason", "")
            out.append(f"| {i} | {state(i)} | {r['text'].replace('|', '/')} | {cell} |")
        Path(path).write_text("\n".join(out) + "\n")
    untested = [i for i in reqs if state(i) not in ("built", "gap", "deferred", "decision")]
    if strict and untested:
        raise pytest.UsageError(f"{len(untested)} requirements have no test: {', '.join(untested)}")


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
