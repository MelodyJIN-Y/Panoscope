"""The Markers / review screen, as the biologist sees it (plan Stage 5, 5.1-5.12, 5.28, 5.30).

Every expected value comes from the study's own input files or the spec, never from the
module that renders it. Gaps are strict xfails asserted on the SCREEN: the day the
feature lands on the page, the test turns red until its ``gap=True`` is removed.
"""
from __future__ import annotations

import csv
import json
import re
import shutil

import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import (evidence_rows, figure_values, open_app, raw_markdown, run,
                             screen_text, widget_keys)


@pytest.fixture
def demo(app_tree):
    """The per-test copy of the demo study, and its committed input rows."""
    study = app_tree / DEMO_ID
    with (study / "inputs" / "markers_top.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (study / "interp" / "verdicts.csv").open() as f:
        verdicts = {r["cluster"]: r for r in csv.DictReader(f)}
    with (study / "inputs" / "reference" / "lab_markers.csv").open() as f:
        types = sorted({r["cell_type"] for r in csv.DictReader(f)})
    return type("Demo", (), dict(dir=study, rows=rows, verdicts=verdicts, types=types))


def _record(study, unit):
    return json.loads((study / "interp" / "records" / "assignment" / f"{unit}.json").read_text())


def _frozen(study) -> dict:
    """The build's outputs a decision must never touch: band, verify, confidence_score."""
    return {p.name: p.read_bytes() for p in
            [study / "interp" / "verdicts.csv", *(study / "interp" / "records").rglob("*.json")]}


def _decisions() -> dict:
    from panoscope.pipeline import store

    return store.load_decisions(DEMO_ID)


# --------------------------------------------------------------------------- #
# 5.10  The four states, visibly distinct (and never a zero)
# --------------------------------------------------------------------------- #
_STATES = {  # state -> (glyph, word) from interface-design §2.3 / CLAUDE.md
    "observed": ("●", "observed"),
    "tested_absent": ("◐", "measured, not here"),
    "never_measured": ("⊘", "never measured"),
}


@pytest.mark.req("ARCH-13.4-01", "UI-2.3-01", "UI-2.3-02")
def test_each_state_row_carries_its_glyph_its_class_and_its_word_and_never_a_zero(demo):
    """c1: the reported genes are observed with a number; the NoSig genes and the
    off-panel canonical marker are never measured; the rest of the panel was tested and
    not reported here. Membership is read from markers_top.csv."""
    at = open_app("markers", cluster="c1")
    rows = evidence_rows(at)
    reported = {r["gene"] for r in demo.rows if r["top_cluster"] == "c1"}
    nosig = {r["gene"] for r in demo.rows if r["top_cluster"] == "NoSig"}
    by_state: dict[str, set] = {s: set() for s in _STATES}
    for key, row in rows.items():
        gene = key.rsplit(":", 1)[1]
        state = next(s for s in _STATES if f"s-{s}" in row["html"])
        glyph, word = _STATES[state]
        assert f"{glyph} {word}" in row["text"], (gene, row["text"])
        by_state[state].add(gene)
        if state == "observed":
            assert re.search(r"\d+\.\d\d", row["cells"][0]), (gene, row["cells"])
            continue
        # Not a number, not a blank, not a zero: the hatch and the word stand in for it.
        assert 'class="pano-sf-hatch"' in row["html"], gene
        for cell in row["cells"]:
            assert cell and not re.fullmatch(r"0(\.0+)?|nan|None|NaN", cell), (gene, cell)
            assert not re.search(r"\d+\.\d+", cell), (gene, cell)
    assert by_state["observed"] == reported
    assert nosig < by_state["never_measured"]
    assert by_state["never_measured"] - nosig == {"KRT18"}      # canonical, off the panel
    assert by_state["tested_absent"] and not by_state["tested_absent"] & (reported | nosig)
    md = raw_markdown(at)
    assert f"NEVER MEASURED - {len(nosig)} on panel, no usable signal" in md
    assert "NEVER MEASURED - 1 off panel" in md


@pytest.mark.req("ARCH-13.4-01", "UI-2.3-01", gap=True)
def test_an_off_panel_row_and_a_no_signal_row_say_different_words_on_the_row(demo):
    """ADIPOQ (on panel, no usable signal) and KRT18 (off the panel) are both never
    measured, for different reasons. The reason must be a visible word on the row,
    not only the block heading above it or a tooltip (today it is only in ``title=``)."""
    rows = evidence_rows(open_app("markers", cluster="c1"))
    krt18, adipoq = rows["gene|g=gene:KRT18"]["cells"], rows["gene|g=gene:ADIPOQ"]["cells"]
    status = 3                                       # Gene, glm_coef, pearson, Status
    assert krt18[status] != adipoq[status], (krt18[status], adipoq[status])


# --------------------------------------------------------------------------- #
# 5.8 / 5.11  The bin control: default 50 µm, changes the picture, never a value
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-1.3-01")
def test_the_bin_control_opens_on_50_um(demo):
    at = open_app("markers", cluster="c1", markers="ERBB2")
    assert at.radio(key="bin_um_global").value == 50
    assert "pano_density_gene_ERBB2_50" in figure_values(at)
    assert "ERBB2 transcript detections, pre-segmentation - 50 um" in screen_text(at)


def _filled(trace) -> int:
    return sum(v is not None and v == v for row in trace["z"] for v in row)


@pytest.mark.req("UI-2.8-01", "CLAUDE-IF-01", "ARCH-13.2-02")
def test_changing_the_bin_changes_the_density_picture_and_no_value(demo):
    """The hexes must re-aggregate (a dead control would pass the invariance half);
    every other line on the screen and every other figure must stay identical."""
    at = open_app("markers", cluster="c1", markers="ERBB2")
    text0, figs0, rows0 = screen_text(at).splitlines(), figure_values(at), evidence_rows(at)
    run(at.radio(key="bin_um_global").set_value(100))
    text1, figs1 = screen_text(at).splitlines(), figure_values(at)

    assert "pano_density_gene_ERBB2_50" in figs0 and "pano_density_gene_ERBB2_100" in figs1
    d0, d1 = figs0.pop("pano_density_gene_ERBB2_50"), figs1.pop("pano_density_gene_ERBB2_100")
    assert _filled(d1[0]) < _filled(d0[0])            # 4x the area: fewer, coarser bins
    assert figs1 == figs0                             # cell map, UMAPs, dot plot
    assert evidence_rows(at) == rows0
    # Only the density caption moved, and it names the new bin.
    changed = (set(text0) ^ set(text1))
    assert changed and all("um square bins" in ln for ln in changed), changed
    assert any("- 100 um square bins" in ln for ln in text1)


# --------------------------------------------------------------------------- #
# 5.12  Click-back
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-13.5-01")
def test_every_evidence_row_resolves_to_a_record_decision_that_names_its_frame(demo):
    """A row's ``data-claim`` lands on the record's own decision for that subject, and
    that decision states the conditions it was made under (the capability frame)."""
    at = open_app("markers", cluster="c1")
    refs: dict[str, list] = {}
    for d in _record(demo.dir, "c1")["decisions"]:
        for r in d["refs"]:
            refs.setdefault(r, []).append(d)
    rows = evidence_rows(at)
    assert len(rows) == len(demo.rows) + 1           # the panel + the off-panel canonical
    for key in rows:
        assert key in refs, key
        assert any(d["rule"].startswith("measure:")
                   and re.search(r"\b(panel|detect):", " ".join(d["because"]))
                   for d in refs[key]), (key, refs[key])


@pytest.mark.req("ARCH-13.5-01")
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BUG (needs a product decision): the reference candidates' "
                          "off-panel chips (COL1A1, DCN, ...) carry a data-claim no record "
                          "decision or row on the page resolves")
def test_every_anchor_on_the_page_resolves_to_the_record(demo):
    at = open_app("markers", cluster="c1")
    refs = {r for d in _record(demo.dir, "c1")["decisions"] for r in d["refs"]}
    anchors = {a for a in re.findall(r'data-claim="([^"]*)"', raw_markdown(at))
               if not a.startswith("note:")}
    assert anchors <= refs, sorted(anchors - refs)


@pytest.mark.req("ARCH-13.5-01", gap=True)
def test_a_rows_authority_and_conditions_are_visible_not_only_a_tooltip(demo):
    """'Every sentence resolves to its claim, with authority and conditions visible.'
    Today the frame lives only in the ``title=`` hover; nothing on the page shows it."""
    at = open_app("markers", cluster="c1")
    because = next(d["because"][0] for d in _record(demo.dir, "c1")["decisions"]
                   if d["refs"] == ["gene|g=gene:ERBB2"])
    assert because in screen_text(at), because


# --------------------------------------------------------------------------- #
# 5.30  The proposal names the runner-up and what separated them
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.5-01", "UI-2.1-02")
def test_the_proposal_header_names_the_runner_up_and_the_gene_with_its_number(demo):
    """c3: T cells' CD4 also peaks in c3 (markers_top.csv), so T cells is the rejected
    alternative and CD4's glm_coef is the number shown beside it."""
    cd4 = next(r for r in demo.rows if r["gene"] == "CD4")
    assert cd4["top_cluster"] == "c3"
    at = open_app("markers", cluster="c3")
    assert "rejected:" not in screen_text(at)          # hidden until tapped
    run(at.toggle(key="rv_reveal_c3").set_value(True))
    line = next(ln for ln in screen_text(at).splitlines() if "rejected:" in ln)
    assert "rejected: T Cells" in line
    assert f"CD4 (glm_coef {float(cd4['glm_coef']):.2f})" in line, line


# --------------------------------------------------------------------------- #
# 5.2-5.6  The commit bar
# --------------------------------------------------------------------------- #
def _flag(demo, flagged: bool) -> str:
    return next(c for c, v in demo.verdicts.items() if (v["verify"] == "TRUE") is flagged)


@pytest.mark.req("UI-5.2-04", "UI-5.6-01")
def test_an_unflagged_accept_is_one_tap_writes_no_note_and_leaves_the_band_alone(demo):
    from panoscope.agent import memory, tools

    c = _flag(demo, False)
    before = _frozen(demo.dir)
    at = open_app("markers", cluster=c)
    run(at.toggle(key=f"rv_reveal_{c}").set_value(True))
    run(at.button(key=f"rv_accept_{c}").click())
    got = _decisions()[c]
    assert (got["state"], got["label"]) == ("ACCEPTED", demo.verdicts[c]["cell_type"])
    assert memory.read_notes(tools.memory_base_dir()) == []
    assert "decision · accepted" in screen_text(at)
    assert _frozen(demo.dir) == before


@pytest.mark.req("UI-5.2-04", gap=True)
def test_accepting_a_flagged_call_takes_a_second_tap_and_drafts_a_validation_note(demo):
    """One tap on a ⚑ call must not commit it: it drafts a validation note to confirm."""
    from panoscope.agent import memory, tools

    c = _flag(demo, True)
    at = open_app("markers", cluster=c)
    run(at.toggle(key=f"rv_reveal_{c}").set_value(True))
    run(at.button(key=f"rv_accept_{c}").click())
    assert c not in _decisions(), "one tap committed a flagged call"
    assert [n.type for n in memory.read_notes(tools.memory_base_dir())] == ["validation"]


@pytest.mark.req("UI-5.2-02", gap=True)
def test_a_decided_cluster_reopens_in_one_tap_and_the_band_is_untouched(demo):
    from panoscope.agent import memory, tools

    c = _flag(demo, False)
    before = _frozen(demo.dir)
    at = open_app("markers", cluster=c)
    run(at.toggle(key=f"rv_reveal_{c}").set_value(True))
    run(at.button(key=f"rv_accept_{c}").click())
    reopen = [b for b in at.button if "reopen" in (b.label or "").lower()]
    assert reopen, "a decided cluster offers no Reopen"
    run(reopen[0].click())
    assert "decision · open" in screen_text(at)
    kinds = [e["kind"] for e in memory.read_decisions(tools.memory_base_dir())]
    assert kinds[-1] == "decision_reopened"
    assert _frozen(demo.dir) == before


@pytest.mark.req("UI-5.2-05", "SKM-S6-01", gap=True)
def test_calling_it_yourself_offers_cluster_dataset_or_lab_scope(demo):
    """Today the scope is a fixed line, 'this cluster (c1)'."""
    at = open_app("markers", cluster="c1")
    run(at.toggle(key="rv_calling_c1").set_value(True))
    words = " ".join(str(o).lower() for kind in ("radio", "selectbox", "button_group")
                     for w in at.get(kind) for o in (w.options or ()))
    assert re.search(r"\bdataset\b", words) and re.search(r"\blab\b", words), words


@pytest.mark.req("UI-5.3-05", gap=True)
def test_the_commit_bar_is_disabled_while_a_chat_note_draft_is_open(demo):
    from panoscope.agent import memory

    draft = memory.draft_note(claim="c1 is Stromal (the biologist's call)", scope="cluster",
                              basis="own_validation", cluster="c1",
                              subject_cell_type="Stromal", dataset=DEMO_ID)
    at = open_app("markers", cluster="c1", pending_draft={"c1": draft})
    run(at.toggle(key="rv_reveal_c1").set_value(True))
    run(at.toggle(key="rv_calling_c1").set_value(True))
    run(at.selectbox(key="rv_pick_c1").select(demo.types[0]))
    assert at.button(key="rv_save_c1").disabled
    assert at.button(key="rv_accept_c1").disabled


# --------------------------------------------------------------------------- #
# 5.1 / 5.7  What the reference may and may not do on the screen
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-5.3-09", "ARCH-6.4-02", gap=True)
def test_the_picker_offers_the_intakes_expected_types_beside_the_candidates_never_as_rivals(demo):
    """The biologist told intake to expect plasma cells. The picker offers it beside the
    reference candidates; it never becomes a candidate row, a rival or the band."""
    man = demo.dir / "manifest.json"
    m = json.loads(man.read_text())
    m["background"] = {"expected_cell_types": [{"value": "Plasma_Cells", "provenance": "stated"}]}
    man.write_text(json.dumps(m))
    at = open_app("markers", cluster="c1")
    run(at.toggle(key="rv_reveal_c1").set_value(True))
    run(at.toggle(key="rv_calling_c1").set_value(True))
    options = at.selectbox(key="rv_pick_c1").options
    assert set(demo.types) <= set(options)
    assert "Plasma_Cells" in options
    md = raw_markdown(at)
    cand = md[md.index("reference canonical markers"):md.index("literature, per candidate")]
    assert "Plasma" not in cand
    assert "Plasma" not in "".join(ln for ln in screen_text(at).splitlines()
                                   if "rejected:" in ln or "confidence" in ln)


@pytest.mark.req("UI-7-01", gap=True)
def test_what_a_reference_source_could_not_narrow_is_shown(demo):
    from panoscope.agent import config as cfg
    from panoscope.reference import view

    scope = view.for_study(cfg.current_study()).scope
    assert scope                                       # the demo's PanglaoDB is not species-narrowed
    text = screen_text(open_app("markers", cluster="c1"))
    assert all(s in text for s in scope), scope


@pytest.mark.req("ARCH-7.3-01")
def test_a_label_with_no_reference_node_never_reads_does_not_exist(demo):
    at = open_app("markers", cluster="c1")
    run(at.toggle(key="rv_calling_c1").set_value(True))
    run(at.text_input(key="rv_free_c1").input("Fooblast"))
    run(at.button(key="rv_save_c1").click())
    card = next(ln for ln in screen_text(at).splitlines() if "Your call" in ln)
    assert "Fooblast" in card and "reference data" in card, card
    assert "does not exist" not in screen_text(at)


# --------------------------------------------------------------------------- #
# 5.9  Search lands on a never-measured row
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.3-04", gap=True)
def test_searching_an_off_panel_gene_lands_on_its_never_measured_row(demo):
    at = open_app("markers", cluster="c1")
    boxes = [t for t in at.text_input if "search" in (t.label or "").lower()]
    assert boxes, "the Markers page has no search box"
    run(boxes[0].input("COL1A1"))
    row = evidence_rows(at).get("gene|g=gene:COL1A1")
    assert row and "never measured" in row["text"] and "off panel" in screen_text(at)


# --------------------------------------------------------------------------- #
# 5.28  A half-read study: degraded spec
# --------------------------------------------------------------------------- #
@pytest.fixture
def half_read(tmp_path, monkeypatch):
    """demo_mini whose method file has columns no registered method recognises."""
    from panoscope.data.study import Study

    root = tmp_path / "half_root"
    study = root / "studies" / "half"
    shutil.copytree(Study.open("demo_mini").dir, study)
    shutil.rmtree(study / "interp")
    p = study / "inputs" / "markers_top.csv"
    lines = p.read_text().splitlines()
    lines[0] = "feature,grp,score_x,r1,r2,r3,anno"
    p.write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return (str(root), "half"), lines


@pytest.mark.req("UI-3.1-01")
def test_a_half_read_study_draws_no_plot_offers_no_pin_and_cannot_be_signed_off(half_read):
    study, _ = half_read
    at = open_app("markers", study=study)
    assert not at.get("plotly_chart")
    assert not [k for k in widget_keys(at) if k.startswith(("sfsel_", "rv_"))]
    at = open_app("summary", study=study)
    assert not at.get("download_button")
    assert "nothing to review or sign off" in screen_text(at)


@pytest.mark.req("UI-3.1-01", gap=True)
def test_a_half_read_study_shows_its_columns_verbatim_with_every_row_unknown(half_read):
    """Degraded spec: a provenance-only header, the file's own columns, unrounded."""
    study, lines = half_read
    text = screen_text(open_app("markers", study=study))
    assert all(col in text for col in lines[0].split(",")), text[:400]
    assert lines[1].split(",")[2] in text                # verbatim, not rounded
    assert "unknown" in text
