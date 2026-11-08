"""Browser E2E (plan Stage 7): the real app in headless Chromium, marker ``browser``.

Deselected by default. To run::

    pip install -e ".[browser]" && playwright install chromium
    pytest -m browser -q

Every server here runs on an ephemeral port against its OWN copy of
``data/studies/`` and its own context dir: the app persists as it renders
(decisions, sign-offs, chat threads, notes), so a shared tree leaks one test's
clicks into the next, and the committed tree must stay byte-unchanged.
``browser``/``page`` are pytest-playwright's fixtures; nothing imports Playwright
at collection, so the default suite collects these files without it.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.conftest import DEMO_ID, PROJECT_ROOT, copy_studies

APP = PROJECT_ROOT / "panoscope" / "app.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def app_env(tmp: Path) -> dict[str, str]:
    """A fresh data root (copy of the studies) + context dir + HOME, no model key."""
    copy_studies(tmp / "data" / "studies")
    (tmp / "ctx").mkdir()
    env = {**os.environ, "PANOSCOPE_DATA_ROOT": str(tmp / "data"),
           "PANOSCOPE_DATASET": DEMO_ID, "PANOSCOPE_CONTEXT_DIR": str(tmp / "ctx"),
           "HOME": str(tmp)}
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def seed(env: dict, code: str) -> None:
    """Write state the way the app would, in the server's environment, before it starts."""
    subprocess.run([sys.executable, "-c", code], env=env, cwd=PROJECT_ROOT,
                   check=True, capture_output=True)


@contextmanager
def serve(env: dict, launcher: str | None = None):
    """Start the app (or ``python -m <launcher>``, which starts it itself) headless."""
    port = _free_port()
    head = [sys.executable, "-m", launcher] if launcher else \
        [sys.executable, "-m", "streamlit", "run", str(APP)]
    proc = subprocess.Popen(
        head + ["--server.port", str(port), "--server.headless", "true",
                "--browser.gatherUsageStats", "false"],
        env=env, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            if proc.poll() is not None:
                pytest.fail("the app exited while starting")
            try:
                with urllib.request.urlopen(f"{base}/_stcore/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.25)
        else:
            pytest.fail(f"the app never answered on {port}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The demo app, one per module, on its own copy of the study tree."""
    with serve(app_env(tmp_path_factory.mktemp("app"))) as base:
        yield base


def settle(page, timeout: int = 45_000) -> None:
    """Wait until Streamlit has finished a script run (it streams over a websocket,
    so ``load`` fires long before content, and a rerun keeps the old DOM meanwhile).
    Assert an expected outcome with :func:`see`, which waits for it; this only lets
    a negative check ("nothing replayed") read a finished page."""
    page.wait_for_timeout(800)            # a click's rerun has not started yet
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=timeout)
    page.wait_for_function(
        "() => document.body.innerText.replace(/\\s+/g,' ').trim().length > 300",
        timeout=timeout)
    page.wait_for_function(
        "() => !document.querySelector(\"[data-testid='stStatusWidget']\")"
        " && !document.querySelector('[data-stale=\"true\"]')", timeout=timeout)
    page.wait_for_timeout(500)


def see(page, pattern: str, timeout: int = 30_000) -> None:
    """Wait until the page's text matches ``pattern`` (a case-blind regex), else fail naming it."""
    from playwright.sync_api import expect

    expect(page.locator("body")).to_contain_text(re.compile(pattern, re.IGNORECASE), timeout=timeout)


def body(page) -> str:
    return " ".join(page.inner_text("body").split())
