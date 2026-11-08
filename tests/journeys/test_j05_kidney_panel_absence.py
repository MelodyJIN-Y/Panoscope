"""J5: panel absence at maximum severity -- kidney on a panel built for breast (S5).

The kidney stand-in: the demo's real 280-gene panel declared as human kidney, with a
lab list of real kidney canonical markers almost all off the panel."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import clear_all_caches
from tests.fixtures.studies import KIDNEY_LAB_MARKERS, make_kidney_standin
from tests.journeys.flow import call_it
from tests.ui.screen import open_app, plain, screen_text

_BREAST = ("Tumor", "HER2", "Myoepithelial", "Luminal", "Stromal")


@pytest.fixture
def kidney(study_builder, offline_user):
    from panoscope.pipeline import run as pipeline_run

    d = make_kidney_standin(study_builder.root)
    pipeline_run.run(d.name)
    clear_all_caches()                    # as the app does after a build
    panel = set(pd.read_parquet(d / "inputs" / "panel.parquet")["gene"])
    return d, panel


def _card(at) -> str:
    return " ".join(plain(m.value) for m in at.markdown if 'class="pano-rv-assess"' in m.value)


@pytest.mark.req("CLAUDE-HR-01", "ARCH-9.5-01", "PLAN-P2.2-01", "UI-5.3-09")
def test_a_kidney_call_on_an_off_panel_type_reads_low_verify_and_no_breast_label_leaks(kidney):
    """Given a kidney section run on a breast panel, labelless, with the lab's kidney list,
    When the nephrologist opens c1,
    Then the candidate types and the label picker are the lab's kidney types only;
    When they call c1 Proximal tubule, none of whose markers is on the panel,
    Then their call reads Low with the verify flag: nothing could have supported it."""
    d, panel = kidney
    kidney_types = {ct for ct, _ in KIDNEY_LAB_MARKERS}
    pt = [g for ct, g in KIDNEY_LAB_MARKERS if ct == "Proximal tubule"]
    assert pt and not set(pt) & panel

    at = open_app("markers", cluster="c1", study=(str(d.parent.parent), d.name))
    text = screen_text(at)
    assert "Podocyte" in text and not [b for b in _BREAST if b in text]
    at.toggle(key="rv_calling_c1").set_value(True).run()
    options = set(at.selectbox(key="rv_pick_c1").options)
    assert "Proximal tubule" in options and options <= kidney_types, options

    at = call_it(at, "c1", pick="Proximal tubule")
    card = _card(at)
    assert "Proximal tubule" in card and "Low" in card and "verify" in card, card


@pytest.mark.req("CLAUDE-HR-01", "ARCH-9.5-01", gap=True)
def test_the_kidney_call_card_says_k_of_n_and_that_nothing_on_the_panel_could_support_it(
        kidney):
    """Then the card says why: 0 of N canonical markers measurable, nothing on this
    panel could have supported or refuted it -- not just 'no canonical driver', which
    reads the same as a tested absence."""
    d, panel = kidney
    pt = [g for ct, g in KIDNEY_LAB_MARKERS if ct == "Proximal tubule"]
    at = call_it(open_app("markers", cluster="c1", study=(str(d.parent.parent), d.name)),
                 "c1", pick="Proximal tubule")
    card = _card(at)
    assert f"0 of {len(pt)}" in card, card
    assert "could have supported" in card
