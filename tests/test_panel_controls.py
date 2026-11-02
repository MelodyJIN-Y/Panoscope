"""Negative-control probes are not panel genes, and the count has one source.

Xenium writes its controls into ``panel.parquet`` beside the genes.
``XeniumAdapter.frames`` dropped them and ``agent.data.load_panel`` did not, so the
capability frame said one number and every consumer of the tidy reader said another:
``manifest.panel_gene_count`` (``pipeline/run.py`` counts these rows),
``agent.enrichment._panel_size`` and so every "...on this N-gene panel" caveat, and
``ui.conditionality._panel_scope``, the line printed under every plot. All three are
coverage denominators, which is the one place a panel size is load-bearing.

The bundled demo's panel was pre-stripped, so the disagreement was unobservable on it
and only appears on a store whose panel file still carries a control row -- the
multi-sample fixture does, and a real 10x panel ships dozens.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import enrichment as E
from panoscope.data.study import Study
from panoscope.platforms.xenium import XeniumAdapter, is_control_probe
from panoscope.ui import conditionality as C

FIXTURE = "synthetic_multisample_fixture"
CONTROL = "NegControlProbe_00001"


@pytest.fixture
def store() -> Study:
    s = Study.open(FIXTURE)
    if not s.exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    return s


def test_the_control_row_is_still_in_the_file(store: Study) -> None:
    """The guard for this whole module: a fixture quietly cleaned of its control row
    would make every assertion below pass for the wrong reason."""
    import pandas as pd

    raw = pd.read_parquet(store.inputs / "panel.parquet", columns=["gene"])
    assert CONTROL in set(raw["gene"].astype(str))


def test_the_reader_and_the_capability_frame_agree_on_the_assay_size(store: Study) -> None:
    """One assay, one number. The frame was right; the reader was one row larger."""
    frame = next(f for f in XeniumAdapter().frames(store) if f.name.startswith("panel:"))
    with cfg.use_study(store):
        panel = data.load_panel()
    assert len(panel) == len(frame.members) == 12
    assert set(panel["gene"].astype(str)) == set(frame.members)


def test_a_control_probe_is_not_on_the_panel(store: Study) -> None:
    """``panel_contains`` is THE absence primitive; a control answering True to it
    means a probe can never be reported as off-panel."""
    with cfg.use_study(store):
        assert data.panel_contains(CONTROL) is False
        assert data.panel_annotation(CONTROL) is None
        assert data.panel_contains("SYNG01") is True
    assert is_control_probe(CONTROL) and not is_control_probe("SYNG01")


def test_no_coverage_denominator_counts_a_control_probe(store: Study) -> None:
    """The three places the inflated count surfaced, checked where they are read."""
    with cfg.use_study(store):
        assert E._panel_size() == 12
        assert "12-gene panel" in E._panel_scope_caveat("HALLMARK_X", 2, 30)
        assert "a 12-gene targeted panel" in C._panel_scope()
        # What pipeline/run.py writes into manifest.panel_gene_count.
        assert int(len(data.load_panel())) == store.manifest["panel_gene_count"] == 12


def test_the_demo_panel_is_unchanged(demo_id: str) -> None:
    """The demo panel carries no control rows, so a correct filter is a no-op there.
    If this moves, the filter is dropping a real gene and the golden would move too."""
    with cfg.use_study(Study.open(demo_id)):
        assert len(data.load_panel()) == 280
