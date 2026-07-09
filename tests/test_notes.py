"""Notes stage: the confident-floor guard on the deterministic fallback.

The adversarial review confirmed a grounding-floor violation: when the model's
prose is unusable and the note falls back to a deterministic clause, the note
must NOT keep the model's top search citation. Stapling a real PMID onto a
fallback clause the paper does not support is a mismatched citation — worse than
none, per the project's core principle. ``_resolve_note`` is the single point
that both note-builders (cell-type and per-gene) route through; these tests pin
its behavior so the violation cannot regress.
"""

from __future__ import annotations

from types import SimpleNamespace

from pipeline import run as pipeline_run
from pipeline.stages.notes import _fallback_summary, _looks_meta, _resolve_note


def _cite(pmid: str = "12345"):
    return SimpleNamespace(pmid=pmid, title="A real paper", authors="Doe J", year=2021)


def test_keeps_citation_when_model_text_survives():
    summary, pmid, citation, used_fb = _resolve_note(
        "KRT7 is a luminal epithelial keratin marking glandular identity.",
        _cite("33416175"),
        "fallback clause",
    )
    assert not used_fb
    assert summary.startswith("KRT7")
    assert pmid == "33416175"
    assert citation["pmid"] == "33416175"


def test_drops_citation_on_meta_fallback():
    # Meta-commentary about the citation choice -> deterministic fallback -> the
    # real-but-unrelated PMID must be dropped, not stapled to the fallback clause.
    fb = _fallback_summary("Stromal", ["LUM", "POSTN", "DCN"])
    summary, pmid, citation, used_fb = _resolve_note(
        "Neither paper is a strong fit; this citation resolves cleanly but is off-topic.",
        _cite("999"),
        fb,
    )
    assert used_fb
    assert summary == fb
    assert pmid is None
    assert citation is None


def test_drops_citation_on_empty_text():
    summary, pmid, citation, used_fb = _resolve_note("", _cite("111"), "Fallback.")
    assert used_fb and pmid is None and citation is None and summary == "Fallback."


def test_no_model_citation_is_fine():
    summary, pmid, citation, used_fb = _resolve_note(
        "EPCAM is an epithelial adhesion molecule marking carcinoma identity.",
        None,
        "fallback",
    )
    assert not used_fb and pmid is None and citation is None and summary.startswith("EPCAM")


def test_looks_meta_catches_review_examples():
    assert _looks_meta("Neither paper is a strong fit for stromal identity.")
    assert _looks_meta("The PubMed connector is returning no hits.")
    assert not _looks_meta("LUM is a small leucine-rich proteoglycan in stroma.")


def test_read_tree_frame_prefers_csv_else_none(tmp_path):
    import pandas as pd

    vdir = tmp_path / "viz"
    vdir.mkdir()
    assert pipeline_run._read_tree_frame(vdir, "cells") is None  # neither present

    pd.DataFrame({"cell_id": [1, 2, 3]}).to_csv(vdir / "cells.csv", index=False)
    got = pipeline_run._read_tree_frame(vdir, "cells")
    assert got is not None and len(got) == 3
