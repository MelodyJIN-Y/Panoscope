"""J6: a collaborator sends only an enrichment table, another only DE output (S6).

Every folder shape must OPEN in the app, not just build: S6 shipped a first screen
that crashed because its only test checked the build."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.journeys.flow import DEMO_INPUTS, build_and_start, declare, folder_with, go, paste, upload
from tests.ui.screen import screen_text

_ROOT = Path(__file__).resolve().parents[2] / "data"
HALLMARK = _ROOT / "jazzpanda" / "hbc_sp1_top10_hallmark_test_statistic.csv"
SYNTH = _ROOT / "studies" / "synthetic_multisample_fixture" / "inputs"


def _listed(at) -> dict[str, str]:
    return {m.value.split("`")[1]: m.value for m in at.markdown if m.value.startswith("- `")}


@pytest.mark.req("PLAN-P1.1-01", "PLAN-P1.5-01", "ARCH-13.1-02", "PLAN-P4.4-01",
                 "CLAUDE-TK-04", "PLAN-T5-01")
def test_an_enrichment_only_folder_builds_from_any_name_and_opens_summary_and_pathways(
        isolated, tmp_path):
    """Given a folder with only jazzPanda CGST hallmark output saved as hallmark.csv,
    and the panel,
    When the collaborator pastes it, declares the tissue, builds and starts,
    Then the file is listed as the enrichment method's, the build reads that same file
    (no demand for enrichment.csv), Summary opens without an exception, and Pathways
    opens on a cluster with its census -- no marker workflow, so nothing to call first
    (decision 2: the lock applies only when a marker workflow exists)."""
    folder = folder_with(tmp_path, "enrich_only", {"hallmark.csv": HALLMARK,
                                                   "panel.parquet": DEMO_INPUTS / "panel.parquet"})
    first = sorted(set(pd.read_csv(HALLMARK)["cluster"]), key=lambda c: int(c[1:]))[0]
    at = paste(upload(), folder)
    assert "(enrichment-interpretation)" in _listed(at)["hallmark.csv"]
    at = build_and_start(declare(at, tissue="human breast cancer"))
    assert not at.error, [e.value for e in at.error]

    at = go(at, "summary")
    at = go(at, "pathways", first)
    text = screen_text(at)
    assert "Call this cluster first" not in text
    assert " enriched · " in text and " sets" in text, text[:600]


@pytest.mark.req("PLAN-P1.5-01", "ARCH-13.1-02", "PLAN-T5-01")
def test_a_de_only_folder_builds_and_opens_summary_with_the_stray_tsv_listed(
        isolated, tmp_path):
    """Given a folder with only edgeR output, the panel, and a stray copy saved as .tsv,
    When the collaborator pastes it, declares the tissue, builds and starts,
    Then the .tsv is listed as not recognised, and Summary, Markers and Pathways open
    without an exception (the DE screen itself is the 5.22 gap in tests/ui/test_shell.py)."""
    folder = folder_with(tmp_path, "de_only", {"de_edger.csv": SYNTH / "de_edger.csv",
                                               "panel.parquet": SYNTH / "panel.parquet",
                                               "de_seurat.tsv": SYNTH / "de_edger.csv"})
    at = paste(upload(), folder)
    listed = _listed(at)
    assert "not recognised by any registered method" in listed["de_seurat.tsv"]
    assert "(de-interpretation)" in listed["de_edger.csv"]
    at = build_and_start(declare(at, tissue="synthetic"))
    assert not at.error, [e.value for e in at.error]
    for page in ("summary", "markers", "pathways"):
        assert screen_text(go(at, page)).strip(), page
