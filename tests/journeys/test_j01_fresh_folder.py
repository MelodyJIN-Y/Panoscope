"""J1: a fresh folder, no labels, no model key (journeys S1)."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from tests.journeys.flow import DEMO_INPUTS, call_it, defer, folder_with, go, paste, upload, declare
from tests.ui.screen import plain, raw_markdown, run, screen_text


@pytest.mark.req("PLAN-P1.1-01", "PLAN-P1.2-01", "ARCH-6.3-04", "PLAN-P1C-01",
                 "PLAN-P3.4-02", "CLAUDE-HR-04")
def test_a_fresh_labelless_folder_is_listed_built_called_and_never_reads_thin(
        isolated, tmp_path):
    """Given a postdoc's folder holding markers_top.csv, panel.parquet, a stray
    notes_from_core.txt and a stray table saved as .tsv, no labels and no model key,
    When they paste the path, pick xenium, declare the tissue, build and start,
    Then every file is listed, the two strays as not recognised by any method;
    only tissue blocks the build; every cluster opens not yet called;
    And when they call c1 themselves with no corpus built, the literature line on
    their call says it cannot settle, never that the literature is thin (Ph3.4)."""
    folder = folder_with(tmp_path, "my_breast_run", {
        "markers_top.csv": DEMO_INPUTS / "markers_top.csv",
        "panel.parquet": DEMO_INPUTS / "panel.parquet",
        "notes_from_core.txt": "sectioned 2024-03-02\n",
        "copy_of_markers.tsv": "gene\ttop_cluster\nERBB2\tc1\n",
    })
    panel = pd.read_parquet(folder / "panel.parquet")["gene"].nunique()
    clusters = sorted({c for c in pd.read_csv(folder / "markers_top.csv")["top_cluster"]
                       if c != "NoSig"}, key=lambda c: int(c[1:]))

    at = paste(upload(), folder)
    listed = {m.value.split("`")[1]: m.value for m in at.markdown if m.value.startswith("- `")}
    assert set(listed) == {"markers_top.csv", "notes_from_core.txt", "copy_of_markers.tsv"}
    assert "jazzPanda" in listed["markers_top.csv"]
    for stray in ("notes_from_core.txt", "copy_of_markers.tsv"):
        assert "not recognised by any registered method" in listed[stray]
    checklist = at.code[0].value
    assert re.search(rf"panel:\S+ \({panel}\)", checklist), checklist
    assert re.search(rf"clusters:\S+ \({len(clusters)}\)", checklist), checklist
    blocking = checklist.split("BLOCKING", 1)[1].split("Optional", 1)[0]
    assert re.findall(r"\[ \] (\w+)", blocking) == ["tissue"]
    for f in ("fixation", "segmentation", "disease_state"):
        assert f'propagates as "{f} unknown"' in checklist
    assert [w.value for w in at.warning] == ["Blocked until declared: tissue"]
    assert "ob_build" not in {b.key for b in at.button}

    at = declare(at, tissue="human breast cancer")
    assert "stated by the biologist" in at.code[0].value
    run(at.button(key="ob_build").click())
    at = run(at.button(key="ob_corpus_skip").click())

    at = go(at, "markers", "c1")
    rail = {b.key: b.label for b in at.button if (b.key or "").startswith("rail_")}
    assert sorted(rail, key=lambda k: int(k[6:])) == [f"rail_{c}" for c in clusters]
    assert "cannot settle" in screen_text(at)
    run(at.toggle(key="rv_reveal_c1").set_value(True))
    assert at.button(key="rv_accept_c1").disabled           # no proposal to accept

    at = call_it(at, "c1", free="Luminal tumour epithelium")
    line = next(plain(m.value) for m in at.markdown if "literature check on your call" in m.value)
    assert not re.search(r"\bthin\b", line), line
    assert "cannot settle" in line, line
    for c in clusters[1:]:
        at = defer(at, c)
    at = go(at, "summary")
    assert len(at.download_button) == 2


@pytest.mark.req("PLAN-P1.1-01")
def test_every_result_like_file_is_listed_and_only_a_csv_is_read(tmp_path):
    """recognise_inputs lists .csv/.tsv/.txt/.xlsx at the top level; a non-CSV is None
    (no method reads it), and the panel or a subfolder is not a result file."""
    from panoscope import methods

    for name in ("a.tsv", "b.txt", "c.xlsx", "panel.parquet"):
        (tmp_path / name).write_text("gene\tx\n")
    (tmp_path / "markers_top.csv").write_bytes((DEMO_INPUTS / "markers_top.csv").read_bytes())
    (tmp_path / "samples").mkdir()
    got = {p.name: (m.name if m else None) for p, m in methods.recognise_inputs(tmp_path).items()}
    assert got == {"a.tsv": None, "b.txt": None, "c.xlsx": None, "markers_top.csv": "jazzpanda_spatial_glm"}
