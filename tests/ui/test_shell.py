"""The shell around every workflow screen: the claims drawer, the multi-sample strip,
the class-constant banner, the DE screen (plan Stage 5: 5.21-5.24).

Most of these are strict-xfail gaps asserted on the screen the biologist uses."""
from __future__ import annotations

import pytest

from tests.ui.screen import open_app, screen_text

_FIXTURE = "synthetic_multisample_fixture"


def _labels(at) -> list[str]:
    return [str(w.label) for k in ("button", "toggle", "radio", "selectbox", "button_group",
                                   "segmented_control") for w in at.get(k)]


# --------------------------------------------------------------------------- #
# 5.21  A claims drawer lists the claims used, with authority
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-13.1-01", "ARCH-10.2-02", gap=True)
def test_the_shell_has_a_claims_drawer_listing_claims_with_their_authority(app_tree):
    at = open_app("markers", cluster="c1")
    assert any("claims" in lb.lower() for lb in _labels(at)), _labels(at)[:20]


# --------------------------------------------------------------------------- #
# 5.23  Two samples: one primary view plus a context strip that is the switcher
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-13.7-01", "UI-4.1-01", "UI-4.6-01", gap=True)
def test_a_two_sample_study_shows_a_context_strip_that_switches_sample(app_tree):
    """The fixture's manifest lists sampleA and sampleB; each must be a tappable
    thumbnail in the strip, not equal small multiples and not absent."""
    at = open_app("markers", cluster="c1", study=(str(app_tree.parent), _FIXTURE))
    labels = " ".join(_labels(at))
    assert "sampleA" in labels and "sampleB" in labels, labels[:400]


# --------------------------------------------------------------------------- #
# 5.24  Class-level constants: one banner, and never "% conditional"
# --------------------------------------------------------------------------- #
def _every_page(app_tree):
    from panoscope.pipeline import decisions as D
    from panoscope.ui import data_access as da

    D.accept("c1", da.verdict_for("c1").cell_type, by="tester", at="2026-11-09T00:00:00")
    yield "markers", open_app("markers", cluster="c1")
    yield "pathways", open_app("pathways", cluster="c1")
    yield "summary", open_app("summary")


@pytest.mark.req("ARCH-7.5-03")
def test_no_page_reports_a_percent_conditional(app_tree):
    for page, at in _every_page(app_tree):
        text = screen_text(at).lower()
        assert "% conditional" not in text and "percent conditional" not in text, page


@pytest.mark.req("ARCH-7.5-03", "CORP-A8", gap=True)
def test_the_class_level_constant_renders_once_as_a_banner(app_tree):
    """Curated and literature claims get a provenance stamp, not a verdict (§7.5);
    that class-level fact is said once, not per row."""
    at = open_app("markers", cluster="c1")
    hits = [line for line in screen_text(at).splitlines()
            if "curated" in line.lower() and "provenance" in line.lower()]
    assert len(hits) == 1, hits


# --------------------------------------------------------------------------- #
# 5.22  A DE study opens a COMPARISON screen
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.7-01", "UI-2.7-02", "UI-2.7-03", gap=True)
def test_a_de_study_opens_a_contrasts_screen_with_the_confound_checklist_and_four_blocks(
        app_tree):
    """The fixture carries inputs/de_edger.csv, so Contrasts has a result to show."""
    assert (app_tree / _FIXTURE / "inputs" / "de_edger.csv").is_file()
    at = open_app("contrasts", study=(str(app_tree.parent), _FIXTURE))
    text = screen_text(at).lower()
    for item in ("panel scoping", "replication", "composition", "cross-method"):
        assert item in text, item
    for block in ("measured", "not testable"):
        assert block in text, block
