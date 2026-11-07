"""The loading screen states numbers, so the numbers have to be DERIVED, and this
study's.

``ui/onboarding._run_progress`` ended on the literal
``"Interpretation ready. 9 clusters, 280-gene panel."`` -- the demo's cluster count
and the demo's panel size, shown on every study's hand-off. That literal was replaced
by a read of the study's ``manifest.json`` (``clusters`` / ``panel_gene_count``),
which is per-study and still an assertion: a stored count is a CLAIM about the files,
written once at build time, and a manifest built before the control-probe filter says
13 on a study whose panel file holds 12 -- while every caption under every plot on the
same screen says 12. CLAUDE.md: panel size is dynamic across studies, derive it from
the file, never assert a constant.

Four things are pinned, in descending order of what they buy:

1. **A stored count does not get to speak.** The manifest is made to disagree with
   the files, and the line must follow the FILES. This is the half that bites: every
   study on disk has a manifest that agrees with its own files, so a test using only
   those would pass just as well on the version this replaces.
2. The two real studies' lines are each their own, checked against their input files
   read directly here -- not against the expression the implementation uses.
3. The two lines differ, which the literal would also have failed.
4. An underivable number is left out of the sentence rather than defaulted, down to
   the bare "Interpretation ready."; and the count is never borrowed from the bundled
   demo's marker table by a study that has none of its own.

Every tree this file writes is built under ``tmp_path``; nothing is written under
``data/studies/``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from panoscope.agent import config as cfg
from panoscope.agent.data import is_control
from panoscope.data.study import Study
from panoscope.ui import onboarding

DEMO = "xenium_hbreast_sample1"
FIXTURE = "synthetic_multisample_fixture"


def _ready_for(study: Study | str) -> str:
    with cfg.use_study(study):
        return onboarding._ready_line()


def _ground_truth(store: Study) -> tuple[int, int]:
    """(clusters, panel genes) read straight off this study's input files.

    Deliberately NOT ``cfg.CLUSTER_ORDER`` / ``da.panel_names()``: asserting the
    implementation against the expression the implementation evaluates would pin
    nothing. These are the files, read here.

    Control probes are dropped through the platform adapter's own list, because a
    panel gene count that counts them is the error ``agent/data.load_panel`` exists
    to fix -- and the fixture's ``panel.parquet`` holds exactly one such row, which
    is what makes a manifest "built before the control filter" reproducible here.
    """
    markers = pd.read_csv(store.inputs / "markers_top.csv")
    clusters = markers.loc[markers["top_cluster"] != "NoSig", "top_cluster"]
    panel = pd.read_parquet(store.inputs / "panel.parquet")["gene"].astype(str)
    return clusters.nunique(), int((~panel.map(is_control)).sum())


@pytest.mark.parametrize("study_id", [DEMO, FIXTURE])
def test_the_ready_line_carries_the_open_studys_own_numbers(study_id: str) -> None:
    store = Study.open(study_id)
    if not store.exists:
        pytest.skip(f"study {study_id!r} is not on disk")
    n_clusters, n_panel = _ground_truth(store)
    assert _ready_for(study_id) == (
        f"Interpretation ready. {n_clusters} clusters, {n_panel}-gene panel."
    )


def test_the_two_studies_do_not_get_the_same_line() -> None:
    """The half the literal failed: one sentence, shown on both studies."""
    for study_id in (DEMO, FIXTURE):
        if not Study.open(study_id).exists:
            pytest.skip(f"study {study_id!r} is not on disk")
    assert _ready_for(DEMO) != _ready_for(FIXTURE)


def _study_with_manifest(tmp_path: Path, manifest_patch: dict, *, inputs=True) -> Study:
    """A copy of the fixture under ``tmp_path``, with its manifest edited.

    Copied rather than pointed at, so the edited manifest never touches the repo;
    ``Study(id=..., data_root=tmp_path)`` carries the root explicitly, so no
    environment variable and no ``layout`` cache is involved.
    """
    src = Study.open(FIXTURE)
    dst = tmp_path / "studies" / "edited"
    shutil.copytree(src.dir, dst)
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(manifest_patch)
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if not inputs:
        shutil.rmtree(dst / "inputs")
    return Study(id="edited", data_root=tmp_path)


def test_a_stored_count_does_not_get_to_speak(tmp_path: Path) -> None:
    """THE discriminating input: the manifest disagrees with the files.

    A manifest built before the control-probe filter says 13 where the panel file
    holds 12, and a manifest written before a cluster was merged over-counts the
    clusters. A reader of ``panel_gene_count`` / ``clusters`` prints 13 and 5 here --
    numbers no file on disk holds, under a screen whose own plot captions say 12.

    13 is not an arbitrary wrong number: the fixture's ``panel.parquet`` really does
    hold 13 rows, one of them a control probe, so a manifest written by a run that
    counted rows rather than genes says 13 -- the exact shape this guards.
    """
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    store = _study_with_manifest(
        tmp_path, {"panel_gene_count": 13, "clusters": ["c1", "c2", "c3", "c4", "c5"]}
    )
    n_clusters, n_panel = _ground_truth(store)
    assert (n_clusters, n_panel) == (3, 12)  # the files, and they differ from the claim
    assert _ready_for(store) == "Interpretation ready. 3 clusters, 12-gene panel."


def test_an_underivable_number_is_left_out_rather_than_defaulted(tmp_path: Path) -> None:
    """``unknown`` is first-class and the sentence shrinks around it.

    With no inputs of its own the study can derive neither number, and the line must
    shrink to the bare statement -- NOT borrow the bundled demo's 9 clusters and 280
    genes, which is what a reader that falls back to the legacy flat files does.
    """
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    store = _study_with_manifest(
        tmp_path, {"panel_gene_count": 13, "clusters": ["c1", "c2", "c3"]}, inputs=False
    )
    assert _ready_for(store) == "Interpretation ready."


def test_one_unreadable_file_costs_only_its_own_half_of_the_sentence(tmp_path: Path) -> None:
    """The two numbers are read independently: a missing panel file must not take the
    cluster count down with it, and vice versa."""
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    store = _study_with_manifest(tmp_path, {})
    (store.inputs / "panel.parquet").unlink()
    assert _ready_for(store) == "Interpretation ready. 3 clusters."


def test_a_folder_path_builds_a_study_this_session_opens_not_yet_called(tmp_path, monkeypatch) -> None:
    """Path box -> checklist -> platform and tissue -> build -> the session's study."""
    from streamlit.testing.v1 import AppTest

    import panoscope
    from panoscope.agent import loop as agent_loop
    from panoscope.data import layout
    from panoscope.ui import state

    demo = Study.open(DEMO).inputs
    folder = tmp_path / "mystudy"
    folder.mkdir()
    for name in ("markers_top.csv", "panel.parquet"):  # no labels, no manifest
        shutil.copyfile(demo / name, folder / name)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(layout.ENV_DATA_ROOT, raising=False)
    layout.clear_cache()

    def no_model(*_a, **_k):
        raise RuntimeError("no model key")

    monkeypatch.setattr(agent_loop, "chat", no_model)
    monkeypatch.setattr(agent_loop, "has_model_key", lambda: False)
    from panoscope.reference import sources as ref_sources

    monkeypatch.setattr(ref_sources, "uberon_candidates",
                        lambda text: [("UBERON:0000310", "breast")])

    at = AppTest.from_file(str(Path(panoscope.__file__).parent / "app.py"), default_timeout=300)
    at.session_state["active_page"] = "upload"
    at.run()
    at.text_input(key="ob_folder").input(str(folder)).run()
    at.selectbox(key="ob_platform").select("xenium").run()
    assert "tissue" in at.code[0].value.split("BLOCKING")[1].split("Optional")[0]
    at.text_input(key="ob_f_tissue").input("human breast cancer")
    at.button(key="ob_apply").click().run()
    assert "tissue             human breast cancer  (stated" in at.code[0].value
    assert "no proposal: no model key" in "\n".join(c.value for c in at.caption)
    at.button(key="ob_uberon_find").click().run()
    at.selectbox(key="ob_uberon_pick").select(("UBERON:0000310", "breast")).run()
    at.button(key="ob_uberon_use").click().run()
    at.button(key="ob_build").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # Next the corpus is offered; with no key it is refused, and annotation opens anyway.
    captions = "\n".join(c.value for c in at.caption)
    assert "corpus build refused: no model key" in captions
    assert not [b for b in at.button if b.key == "ob_corpus_build"]
    at.button(key="ob_corpus_skip").click().run()
    assert "corpus not built — cannot settle" in "\n".join(m.value for m in at.markdown)

    at.run()  # a later rerun starts from a fresh context: only the session names the study
    root = (tmp_path / "home" / ".panoscope" / "data").resolve()
    sid = onboarding._stage(folder).id
    assert sid.startswith("mystudy-")
    assert at.session_state["active_page"] == "summary"
    assert at.session_state[state.K_STUDY] == (str(root), sid)
    built = Study(id=sid, data_root=root)
    assert built.manifest["tissue"] == "human breast cancer" and len(built.units) == 9
    assert built.manifest["stated"] == ["platform", "tissue", "tissue_id", "tissue_label"]
    assert (built.manifest["tissue_id"], built.manifest["tissue_label"]) == ("UBERON:0000310", "breast")
    assert (at.query_params["study"], at.query_params["root"]) == ([sid], [str(root)])
    assert "not yet called" in "\n".join(m.value for m in at.markdown)
    at.session_state["active_page"] = "markers"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "xenium · human breast cancer" in "\n".join(m.value for m in at.markdown)  # ours, not the demo's "Xenium"


def test_opening_a_study_drops_the_last_studys_ui_state(monkeypatch) -> None:
    """A demo chat thread must not follow the biologist into their own study."""
    from types import SimpleNamespace

    from panoscope.ui import state

    ss = {state.K_CHAT_THREAD: {"c1": ["a demo message"]}, state.K_INITIALIZED: True}
    monkeypatch.setattr(state, "_ss", lambda: ss)
    monkeypatch.setattr(state, "apply_session_study", lambda: None)
    state.open_study(SimpleNamespace(data_root=Path("/r"), id="mine"))
    assert ss == {state.K_STUDY: ("/r", "mine")}


def test_a_refresh_keeps_the_study_the_url_names_and_a_bad_one_falls_back(tmp_path) -> None:
    """A fresh session reads ?study=&root=; one naming no study opens the demo."""
    from streamlit.testing.v1 import AppTest

    import panoscope
    from panoscope.ui import state

    root = Study.open(FIXTURE).data_root
    for sid, want in ((FIXTURE, (str(root), FIXTURE)), (f"../studies/{FIXTURE}", None), ("nope", None)):
        at = AppTest.from_file(str(Path(panoscope.__file__).parent / "app.py"), default_timeout=300)
        at.query_params.update(page="summary", study=sid, root=str(root))
        at.run()
        assert at.session_state[state.K_STUDY] == want, sid
        assert ("study" in at.query_params) is bool(want)


def test_staging_keeps_two_same_named_folders_apart_and_links_samples(tmp_path, monkeypatch) -> None:
    from panoscope.data import layout

    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path / "root"))
    a, b = tmp_path / "a" / "run", tmp_path / "b" / "run"
    for d in (a, b):
        (d / "samples" / "s1").mkdir(parents=True)
        (d / "samples" / "s2").mkdir()
        (d / "samples" / "s1" / "markers_top.csv").write_text("gene\n")
    (a / "reference").mkdir()
    (a / "reference" / "lab_markers.csv").write_text("cell_type,gene,tissue\nT cell,CD3E,t\n")
    sa, sb = onboarding._stage(a), onboarding._stage(b)
    lab = sa.dir / "inputs" / "reference" / "lab_markers.csv"
    assert lab.is_file() and not lab.is_symlink() and not (sb.dir / "inputs" / "reference").exists()
    assert sa.id != sb.id
    assert (sa.inputs / "samples" / "s1" / "markers_top.csv").is_file()
    assert sa.manifest["samples"] == ["s1", "s2"]


def test_user_studies_default_to_the_home_root_unless_the_env_names_one(tmp_path, monkeypatch) -> None:
    from panoscope.data import layout

    (tmp_path / "data").mkdir()  # a ./data in the cwd is not where user studies go
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(layout.ENV_DATA_ROOT, raising=False)
    layout.clear_cache()
    assert layout.user_data_root() == (tmp_path / "home" / ".panoscope" / "data").resolve()
    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path / "elsewhere"))
    assert layout.user_data_root() == (tmp_path / "elsewhere").resolve()


def test_the_demo_chat_transcript_is_written_under_the_user_root_not_the_repo(tmp_path, monkeypatch):
    from panoscope.data import layout
    from panoscope.ui import chat_store

    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path / "user"))
    try:
        layout.set_session_root(layout._package_repo_root() / "data")
        assert chat_store._path().is_relative_to((tmp_path / "user").resolve())
        layout.set_session_root(tmp_path / "elsewhere")
        assert chat_store._path().is_relative_to((tmp_path / "elsewhere").resolve())
    finally:
        layout.set_session_root(None)
