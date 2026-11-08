"""Structural safety net for the v2 migration (IMPLEMENTATION.md §7, Phase A.5).

Two properties, both about *import* and neither about behaviour. The existing
suite exercises behaviour through a handful of entry points, so most modules are
only ever reached transitively: a botched bulk import rewrite (§7 Phase B moves
~120 import lines across 83 files) can leave a module unimportable while every
behavioural test still passes. These two tests close that gap.

1. ``test_every_first_party_module_imports_cleanly`` — walk every first-party
   module and import it. Catches a broken rewrite instantly, by module name.
2. ``test_import_does_not_read_dataset_files`` — importing the package must not
   read a built dataset off disk. This one FAILS today, on purpose, and is
   marked ``xfail(strict=True)``; see its docstring for what must change.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Top-level packages to walk. Written as *candidates* rather than a fixed list so
# the Phase B rename (agent/ + ui/ + pipeline/ -> panoscope/) needs no edit here;
# a candidate that does not exist is ignored, and _MIN_MODULES below is what stops
# an empty walk from passing silently.
_CANDIDATE_PACKAGES = ("agent", "ui", "pipeline", "panoscope")

# ``app.py`` is a Streamlit script, not a library module: its module body runs the
# whole UI render. It *is* importable with no server running (ui/data_access.py's
# lazy decorator shim falls through to an identity decorator, and Streamlit runs
# the body in "bare mode"), verified rather than assumed — but importing it in the
# pytest process would execute st.set_page_config and friends inside a process that
# other UI tests share, so it gets its own subprocess below.
_APP_MODULE = "panoscope.app"

# A module may be unimportable ONLY because one of these optional distributions is
# absent (the extras in IMPLEMENTATION.md §8: app / agent / export / spatial). A
# skip fires only when the module named by the ImportError is in this set AND is
# genuinely not installed. Every other failure — a first-party module, a core
# dependency, a name the bulk rewrite got wrong — is a hard failure. Today all of
# these are installed in .venv, so nothing is skipped and the test really runs.
_OPTIONAL_DISTRIBUTIONS = frozenset(
    {
        "streamlit", "plotly",         # [app]
        "anthropic", "mcp", "dotenv",  # [agent]
        "docx", "fpdf",                # [export]
        "spatialdata", "zarr",         # [spatial]
    }
)

# 54 first-party modules today (agent/ 20, ui/ 22, pipeline/ 12). The floor only
# has to be high enough that a walk which silently found nothing cannot pass.
_MIN_MODULES = 40


def _first_party_modules() -> list[str]:
    """Every importable first-party module name, packages included.

    ``walk_packages`` imports each package to read its ``__path__``; a package that
    fails to import is recorded by name (via ``onerror``) so the import loop below
    re-raises it with a real traceback instead of losing it inside the walk.
    """
    names: set[str] = set()
    for pkg_name in _CANDIDATE_PACKAGES:
        if importlib.util.find_spec(pkg_name) is None:
            continue
        pkg = importlib.import_module(pkg_name)
        names.add(pkg_name)
        path = getattr(pkg, "__path__", None)
        if path is None:  # a module, not a package
            continue
        names.update(
            info.name
            for info in pkgutil.walk_packages(path, prefix=f"{pkg_name}.", onerror=names.add)
        )
    return sorted(names)


def _missing_optional(exc: BaseException) -> str | None:
    """The optional distribution this ImportError blames, or None.

    None means "not an optional-extra problem" — i.e. a real breakage.
    """
    if not isinstance(exc, ImportError):
        return None
    top = (getattr(exc, "name", None) or "").split(".")[0]
    if top in _OPTIONAL_DISTRIBUTIONS and importlib.util.find_spec(top) is None:
        return top
    return None


@pytest.mark.req("IMPL-7-02")
def test_every_first_party_module_imports_cleanly() -> None:
    """Every module under agent/, ui/ and pipeline/ imports without raising.

    Failures are collected rather than raised one at a time, so a broken import
    rewrite reports every module it damaged in a single run.
    """
    modules = _first_party_modules()
    assert len(modules) >= _MIN_MODULES, (
        f"import-surface walk found only {len(modules)} module(s): {modules}. "
        f"Expected >= {_MIN_MODULES}. Either the package layout moved (update "
        f"_CANDIDATE_PACKAGES) or the walk is silently finding nothing."
    )

    failures: list[str] = []
    skipped: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - any failure to import is a failure
            missing = _missing_optional(exc)
            if missing is not None:
                skipped.append(f"{name} (optional dependency {missing!r} not installed)")
                continue
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failures.append(f"  {name}: {detail}")

    assert not failures, (
        f"{len(failures)} of {len(modules)} first-party module(s) failed to import:\n"
        + "\n".join(failures)
    )
    if skipped:
        # One line, so `pytest -rs` shows the whole list: its short summary prints
        # only the first line of a skip reason, and a hidden list is exactly the
        # "silently skipped a real breakage" failure mode this test exists to avoid.
        pytest.skip(
            "optional extras not installed, so these modules were NOT import-checked: "
            + "; ".join(skipped)
        )


def test_streamlit_entrypoint_imports_headless() -> None:
    """``panoscope/app.py`` imports with no Streamlit server running.

    Run in a subprocess because importing it executes the full page render; the
    point is only that the import graph is intact, not that the render is silent.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, sys.argv[1]); import {_APP_MODULE}",
         str(PROJECT_ROOT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"`import {_APP_MODULE}` failed headless (exit {proc.returncode}):\n"
        f"{proc.stderr[-3000:]}"
    )


# --------------------------------------------------------------------------- #
# Import-time filesystem I/O
# --------------------------------------------------------------------------- #
# Child program for the test below. Passed to `python -c`, so argv[1] is the repo
# root. It installs an audit hook before importing anything first-party, then
# reports every file under data/ that the import opened.
_NO_IO_CHILD = (
    "import json, sys\n"
    "ROOT = sys.argv[1]\n"
    "DATA = ROOT + '/data'\n"
    "sys.path.insert(0, ROOT)\n"
    "reads = []\n"
    "sys.addaudithook(\n"
    "    lambda event, args: reads.append(args[0])\n"
    "    if event == 'open' and args and isinstance(args[0], str) and args[0].startswith(DATA)\n"
    "    else None\n"
    ")\n"
    "import panoscope.agent.config\n"   # reads cluster_key.json + markers_top.csv + panel.parquet
    "import panoscope.pipeline.v1_verdicts\n"
    "print('PANOSCOPE_DATASET_READS ' + json.dumps(sorted(set(reads))))\n"
)

# A dataset id with no tree under data/studies/. It is the only lever available:
# agent/config.py derives PROJECT_ROOT from __file__ and reads PANOSCOPE_DATASET,
# PANOSCOPE_MODEL and (in tools.py) PANOSCOPE_CONTEXT_DIR — there is no
# PANOSCOPE_DATA_ROOT to point at a tmp dir, which is itself part of what Phase C
# has to fix.
_ABSENT_DATASET = "no_such_dataset_for_import_surface_test"


# Flipped from xfail(strict=True) to a hard assert at the end of Phase C, as its
# original reason promised. What changed: config.py's dataset-derived globals moved
# behind a PEP 562 __getattr__ resolving against the current Study, and verdict.py's
# module-scope _assert_off_panel_map() -- which read panel.parquet at import to
# validate a map nothing used -- was deleted and became a test.
def test_import_does_not_read_dataset_files(tmp_path: Path) -> None:
    """Importing the package reads nothing that depends on a built dataset.

    XFAIL TODAY — verified failure, with PANOSCOPE_DATASET set to an id that has no
    built tree::

        PANOSCOPE_DATASET=demo_mini .venv/bin/python -c "from agent import config"
        FileNotFoundError: [config] cluster key missing:
            .../data/jazzpanda/cluster_key.json

    What must change for this to pass, and then for the ``xfail`` marker to be
    deleted and this to become a hard assert (end of Phase C):

    * ``agent/config.py`` lines 130-132 call ``_load_cluster_key()``,
      ``_derive_cluster_order()`` and ``_derive_demo_markers()`` at module scope;
      all three read the active dataset's ``cluster_key.json`` / ``markers_top.csv``
      / ``panel.parquet`` and raise when the tree is not built. They must become
      lazy — §7 Phase C's PEP 562 module ``__getattr__`` over a frozen ``Study`` —
      so the values resolve at call time, not import time.
    * ``agent/verdict.py:111`` runs ``_assert_off_panel_map()`` at module scope,
      which calls ``data.panel_contains`` and so reads ``panel.parquet``. §7 Phase B
      deletes it along with the duplicated canonical-marker map it validates.
    * ``ui/data_access.py`` resolves ``_MARKER_EXPR_PARQUET`` / ``_DENSITY_INDEX``
      with ``Path.exists()`` at module scope. Those are stats rather than opens, so
      the audit hook below does not see them, but they are the same import-time
      coupling and should move behind the same lazy resolution.

    A subprocess is mandatory: a module cannot be un-imported, so this property is
    unobservable in a process that has already imported ``agent.config``.
    """
    env = dict(os.environ)
    env["PANOSCOPE_DATASET"] = _ABSENT_DATASET
    env["PANOSCOPE_CONTEXT_DIR"] = str(tmp_path / "context")  # keep writes out of the repo

    proc = subprocess.run(
        [sys.executable, "-c", _NO_IO_CHILD, str(PROJECT_ROOT)],
        cwd=str(tmp_path),  # not the repo root: importing must not depend on cwd either
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, (
        f"importing agent.config / pipeline.v1_verdicts with PANOSCOPE_DATASET="
        f"{_ABSENT_DATASET!r} failed (exit {proc.returncode}); import must not "
        f"require a built dataset:\n{proc.stderr[-3000:]}"
    )

    marker = "PANOSCOPE_DATASET_READS "
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker)]
    assert lines, f"child produced no read report; stdout was:\n{proc.stdout[-3000:]}"
    reads = json.loads(lines[-1][len(marker):])
    assert reads == [], (
        "importing the package opened dataset files:\n  " + "\n  ".join(reads)
    )
