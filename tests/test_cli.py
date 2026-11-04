"""The ``panoscope`` console script.

Four verbs, so four concerns worth a test: the study id resolves in the documented
order, ``explain`` prints the stored numbers and nothing else, ``doctor`` never leaks
a credential, and ``run`` launches *this* interpreter's Streamlit rather than
whatever is first on PATH. ``build`` is covered by ``test_pipeline.py``; the test
here only pins that the verb forwards its arguments.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

from panoscope import cli
from panoscope.data import layout

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(*argv: str) -> int:
    return cli.main(list(argv))


# --------------------------------------------------------------------------- #
# packaging
# --------------------------------------------------------------------------- #
def test_console_script_points_at_this_module():
    """A typo here is only discovered by a user, after an install."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["scripts"]["panoscope"] == "panoscope.cli:main"
    assert pyproject["project"]["requires-python"] == ">=3.11"  # types/ uses StrEnum


def test_runtime_package_data_is_inside_the_package():
    """skills/ and assets/ must live under panoscope/ or they vanish in site-packages.

    Name-agnostic on purpose: the invariant is that the runtime-read files sit inside
    the installed package, not what the skills happen to be called this week.
    """
    pkg = Path(cli.__file__).resolve().parent
    assert list(pkg.glob("skills/*/SKILL.md")), f"no SKILL.md under {pkg / 'skills'}"
    assert (pkg / "assets" / "panoscope_logo.png").is_file()


# --------------------------------------------------------------------------- #
# study resolution
# --------------------------------------------------------------------------- #
def test_study_id_prefers_the_flag_then_the_env(monkeypatch, demo_id, mini_id):
    monkeypatch.setenv(layout.ENV_STUDY, mini_id)
    assert cli._resolve_study_id(demo_id) == demo_id
    assert cli._resolve_study_id(None) == mini_id


def test_study_id_with_nothing_to_go_on_names_what_is_available(monkeypatch, tmp_path):
    monkeypatch.delenv(layout.ENV_STUDY, raising=False)
    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path))
    layout.clear_cache()
    with pytest.raises(SystemExit) as exc:
        cli._resolve_study_id(None)
    assert "--study" in str(exc.value)
    layout.clear_cache()


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
def test_explain_prints_the_stored_numbers(capsys, demo_id, datasets_root):
    stored = json.loads(
        (datasets_root / demo_id / "interp" / "clusters" / "c9.json").read_text("utf-8")
    )
    assert _run("explain", "c9", "--study", demo_id) == 0
    out = capsys.readouterr().out

    assert stored["cell_type"] in out
    assert stored["confidence"] in out
    assert f"verify        {str(stored['verify']).upper()}" in out
    for ev in stored["evidence"]:
        assert ev["gene"] in out
        assert f"{ev['glm_coef']:.3f}" in out
    # The never-measured count is stated, never implied by a blank. It is the
    # RECORD's count and not the v1 file's: the v1 view carries only the panel-frame
    # refusals (core/compat.V1_UNREPRESENTABLE), so the two deliberately differ --
    # and every gene the v1 file DOES call off-panel must be named as such here.
    assert re.search(r"never measured \(\d+\) . absence is NOT evidence", out)
    for note in stored["offpanel_notes"]:
        assert f"- {note['gene']}: outside panel:" in out


def test_explain_json_is_the_record_verbatim(capsys, demo_id, datasets_root):
    assert _run("explain", "c1", "--study", demo_id, "--json") == 0
    emitted = json.loads(capsys.readouterr().out)
    stored = json.loads(
        (datasets_root / demo_id / "interp" / "clusters" / "c1.json").read_text("utf-8")
    )
    assert emitted == stored


def test_explain_on_an_unknown_unit_lists_the_units_that_exist(demo_id):
    with pytest.raises(SystemExit) as exc:
        _run("explain", "c99", "--study", demo_id)
    message = str(exc.value)
    assert "c99" in message and "c1" in message


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def test_doctor_reports_presence_but_never_the_secret(tmp_path, capsys, monkeypatch):
    secret = "sk-ant-NOTAREALKEY-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    # Run from a directory with no .env. ``doctor`` now loads one before reporting,
    # because the agent does -- it used to print "absent" for every key one line
    # above its own ".env in cwd: present", on a machine where the app would go
    # live. This test passed only because of that defect, so it has to stop
    # reading the developer's real .env to keep meaning what it says.
    monkeypatch.chdir(tmp_path)

    assert _run("doctor") == 0
    out = capsys.readouterr().out

    assert "ANTHROPIC_API_KEY    present" in out
    assert "NCBI_API_KEY         absent" in out
    # Not the value, and not a prefix of it either.
    assert secret not in out
    for n in range(4, len(secret)):
        assert secret[:n] not in out


def test_doctor_reports_resolved_paths_and_studies(capsys, demo_id):
    assert _run("doctor") == 0
    out = capsys.readouterr().out
    assert str(layout.resolve_data_root()) in out
    assert demo_id in out
    assert "pandas=" in out


# --------------------------------------------------------------------------- #
# run / build
# --------------------------------------------------------------------------- #
def test_run_launches_streamlit_in_this_interpreter(monkeypatch, demo_id):
    seen: dict = {}

    def fake_call(cmd, env=None):
        seen["cmd"] = cmd
        seen["env"] = env
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert _run("run", "--study", demo_id) == 0

    cmd = seen["cmd"]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(cmd[4]) == Path(cli.__file__).resolve().parent / "app.py"
    assert seen["env"][layout.ENV_STUDY] == demo_id


def test_build_forwards_the_study_and_the_notes_flag(monkeypatch, tmp_path, demo_id):
    import panoscope.pipeline.run as pipeline_run

    seen: dict = {}

    def fake_run(dataset_id=None, root=None, notes=False):
        seen.update(dataset_id=dataset_id, notes=notes)
        return tmp_path

    monkeypatch.setattr(pipeline_run, "run", fake_run)
    assert _run("build", "--study", demo_id, "--notes") == 0
    assert seen == {"dataset_id": demo_id, "notes": True}


def test_explain_refuses_a_store_with_no_capability_frame(tmp_path, demo_id) -> None:
    """``explain`` BUILDS the record, so a missing frame makes it smaller, not absent.

    With the panel removed, ``build`` refuses and names the file, while
    ``explain`` printed a complete-looking record -- "band Very High", "verify
    FALSE" -- and a frames line carrying detect/annotation/clusters and no panel
    at all. The panel-absence rule rests on that frame, and CLAUDE.md documents
    this command as the way to read the interpretation record.

    Run as a SUBPROCESS, which is how the command is invoked and the only way to
    get a clean read: in-process, first-party lru_caches hold the frame resolved
    by an earlier test in this file, so the same assertion passed in isolation
    and failed in file order.
    """
    import os
    import shutil
    import subprocess
    import sys

    root = tmp_path / "data"
    shutil.copytree(layout.resolve_data_root(), root)
    removed = []
    for rel in (("studies", demo_id, "inputs", "panel.parquet"),
                ("panels", "panel.parquet")):
        panel = root.joinpath(*rel)
        if panel.is_file():
            panel.rename(panel.with_suffix(".parquet.bak"))
            removed.append(panel.name)
    assert removed, "the fixture store has no panel to remove"

    env = {**os.environ, "PANOSCOPE_DATA_ROOT": str(root)}
    done = subprocess.run(
        [sys.executable, "-m", "panoscope.cli", "explain", "c1", "--study", demo_id],
        env=env, capture_output=True, text=True)
    output = done.stdout + done.stderr

    assert done.returncode != 0, f"explain rendered a record anyway:\n{output[:600]}"
    assert "panel.parquet" in output, output[:600]
    assert "band" not in done.stdout.lower(), (
        f"a band was printed for a store with no capability frame:\n{done.stdout[:400]}")

    # ...and the healthy store still renders.
    ok = subprocess.run(
        [sys.executable, "-m", "panoscope.cli", "explain", "c1", "--study", demo_id],
        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "band" in ok.stdout.lower()
