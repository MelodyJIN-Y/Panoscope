"""J16: a collaborator's marker table Panoscope only half understands."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.journeys.flow import DEMO_INPUTS, declare, folder_with, paste, upload
from tests.ui.screen import run, screen_text

RENAMED = {"gene": "Gene", "top_cluster": "cluster_id", "glm_coef": "score"}


@pytest.fixture
def renamed_folder(tmp_path):
    table = pd.read_csv(DEMO_INPUTS / "markers_top.csv").rename(columns=RENAMED)
    (tmp_path / "src").mkdir()
    table.to_csv(tmp_path / "src" / "markers_top.csv", index=False)
    return folder_with(tmp_path, "renamed", {"markers_top.csv": tmp_path / "src" / "markers_top.csv",
                                             "panel.parquet": DEMO_INPUTS / "panel.parquet"})


def _try_build(folder):
    at = declare(paste(upload(), folder), tissue="human breast cancer")
    listed = {m.value.split("`")[1]: m.value for m in at.markdown if m.value.startswith("- `")}
    if "ob_build" in {b.key for b in at.button}:
        at = run(at.button(key="ob_build").click())
    return at, listed


@pytest.mark.req("ARCH-9.2-01", "PLAN-P1.1-01", "PLAN-T5-01")
def test_a_renamed_marker_table_is_refused_by_name_never_guessed_and_never_a_crash(
        isolated, renamed_folder):
    """Given a marker table whose columns were renamed (Gene, cluster_id, score),
    When the collaborator pastes the folder and tries to build,
    Then the table is listed as not recognised by any registered method, and no build
    produces an interpretation from a guessed column mapping: the screen says, in plain
    words, what is missing -- no traceback."""
    at, listed = _try_build(renamed_folder)
    assert "not recognised by any registered method" in listed["markers_top.csv"]
    text = screen_text(at)
    assert at.error or at.warning, text[:400]
    assert "Traceback" not in text
    assert "ob_corpus_skip" not in {b.key for b in at.button}     # nothing was interpreted


@pytest.mark.req("UI-3.1-01", "ARCH-13.1-02", gap=True)
def test_a_renamed_marker_table_opens_the_degraded_screen_naming_columns_read(
        isolated, renamed_folder):
    """Then it still opens a degraded surface: a provenance-only header, the columns
    shown verbatim, every row 'unknown', no plots and no sign-off (interface §3.1)."""
    at, _ = _try_build(renamed_folder)
    assert "ob_corpus_skip" in {b.key for b in at.button}, screen_text(at)[:400]
