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

import pandas as pd
import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import enrichment as E
from panoscope.data.study import Study
from panoscope.platforms.xenium import XeniumAdapter, is_control_probe
from panoscope.types import measured as M
from panoscope.types.subject import Entity, EntityKind, subject
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


# --------------------------------------------------------------------------- #
# A control probe is MEASURED. Two ways that was being denied.
# --------------------------------------------------------------------------- #
def _store_with_a_measured_control(src: Study, root, *, named: str) -> Study:
    """``src``, copied, with its control row named ``named`` AND a row for it in the
    method output -- a real unit, the largest statistic in the file.

    Two constructions in one, because the defect needed both to show its worst face
    and last round built them separately. A vendor who writes ``NEGCONTROLPROBE_00001``
    is shipping the same probe and nothing stops a kidney ``panel.parquet`` arriving
    that way; and a control reaches ``markers_top.csv`` with a number on it for the
    ordinary reason that the method wrote its output over every probe the instrument
    ran. Built under ``tmp_path`` because the fixture on disk must stay byte-identical
    -- and because a test that writes into ``data/studies/`` is gitignored into silence.

    The statistic is deliberately the largest in the table: if the lattice ever admits
    this row, it does not merely appear, it drives the band.
    """
    import shutil

    data_root = root / "data"
    (data_root / "studies").mkdir(parents=True)
    shutil.copytree(src.dir, data_root / "studies" / src.id)
    study = data_root / "studies" / src.id

    pq = study / "inputs" / "panel.parquet"
    df = pd.read_parquet(pq)
    assert (df["gene"].astype(str) == CONTROL).sum() == 1
    df.loc[df["gene"].astype(str) == CONTROL, "gene"] = named
    df.to_parquet(pq, index=False)

    csv = study / "inputs" / "markers_top.csv"
    markers = pd.read_csv(csv)
    assert not markers["gene"].astype(str).str.upper().str.startswith("NEGCONTROL").any()
    markers.loc[len(markers)] = {
        "gene": named, "top_cluster": "c1", "glm_coef": 99.0, "pearson": 0.99,
        "max_gg_corr": 0.99, "max_gc_corr": 0.99, "cell_type": "Synthetic Type A",
    }
    markers.to_csv(csv, index=False)

    return Study.open(src.id, data_root=data_root)


@pytest.fixture(params=[CONTROL, CONTROL.upper()], ids=["as_written", "mixed_case"])
def control_store(request, store: Study, tmp_path) -> tuple[Study, str]:
    """The cross product: the whole section below runs in both cases.

    Last round's two tests each pinned one half -- the case-insensitivity test never
    called ``panel_lookup`` and the ``panel_lookup`` test never used a mixed-case
    store -- so the combination, which is where the frame and the readers actually
    disagreed, was untested. Parametrising the fixture is what makes the hard half
    cost nothing to cover.
    """
    return _store_with_a_measured_control(store, tmp_path, named=request.param), request.param


def test_a_control_is_excluded_from_every_gene_frame_in_any_case(
    control_store: tuple[Study, str],
) -> None:
    """One rule, one place: the capability frames and the tidy reader agree because
    they call the same function, upper-cased like every other gene comparison here.

    ``platforms.xenium.is_control_probe`` was the one case-SENSITIVE comparison in the
    package, and it was the one the frames used, so on a mixed-case panel file
    ``panel:<id>`` admitted the control and read 13 for a 12-gene assay while
    ``load_panel`` read 12. ``_detectable_genes`` filtered nothing at all, so
    ``detect:<id>`` admitted it in *either* case and went 9 -> 10.

    The numbers below are asserted against the constructed file -- 12 gene rows and
    one control row, written three lines up -- and never against another caller of the
    rule. A cross-check between two users of one broken rule passes while both are
    wrong, which is exactly how this survived a round.

    Both frames refusing is also what enforces the exclusion: ``_state_for`` requires
    *every* frame of a kind to admit an entity, so with the control outside both there
    is no store, and no case, in which it can be ``OBSERVED`` (and carry its count
    forward as a spatial marker) or ``TESTED_ABSENT`` (and argue against a cell type).
    """
    st, named = control_store
    raw = pd.read_parquet(st.inputs / "panel.parquet", columns=["gene"])["gene"].astype(str)
    assert named in set(raw) and len(raw) == 13   # the file really still has it

    frames = XeniumAdapter().frames(st)
    panel = next(f for f in frames if f.name.startswith("panel:"))
    detect = next(f for f in frames if f.name.startswith("detect:"))

    assert named not in panel.members and len(panel.members) == 12
    assert named not in detect.members and len(detect.members) == 9

    with cfg.use_study(st):
        assert len(data.load_panel()) == 12
        assert set(data.load_panel()["gene"].astype(str)) == set(panel.members)
        assert data.panel_contains(named) is False
        assert data.is_control(named) is True
        # The three coverage denominators, read where they are read.
        assert E._panel_size() == 12
        assert "a 12-gene targeted panel" in C._panel_scope()
        assert "12-gene panel" in E._panel_scope_caveat("HALLMARK_X", 2, 30)


def test_a_control_can_reach_neither_state_that_does_anything(
    control_store: tuple[Study, str],
) -> None:
    """The failure the lattice exists to make unrepresentable, pinned on both sides.

    The control sits in ``markers_top.csv`` with the largest statistic in the file, so
    ``reported`` contains it -- which is the live situation, not a contrived one. With
    the frames admitting it, that row resolves ``OBSERVED`` and ``carries_number``, and
    a negative control's count becomes a spatial marker with a number on it; with it
    absent from the method output instead, it resolves ``TESTED_ABSENT`` and its
    silence argues against a cell type. Neither may ever happen, in any case.

    A real panel gene is asserted beside it so the test fails if the frames simply
    stopped admitting anything.
    """
    st, named = control_store
    world = M.MeasuredWorld(XeniumAdapter().frames(st))

    def resolve(gene_id: str, reported: frozenset[str]):
        subj = subject("gene", g=Entity(EntityKind.GENE, gene_id))
        return M.resolve(subj, world, reported)

    assert resolve(named, frozenset({named})).carries_number is False
    assert resolve(named, frozenset()).absence_is_evidence is False
    assert resolve("SYNG01", frozenset({"SYNG01"})).carries_number is True
    assert resolve("SYNG01", frozenset()).absence_is_evidence is True


def test_panel_lookup_names_a_control_in_any_case(
    control_store: tuple[Study, str],
) -> None:
    """The sentence the biologist reads, on the store that used to break it.

    ``load_panel`` dropping the control row made ``panel_annotation`` return None, which
    sent ``tools.panel_lookup`` down its off-panel branch: a ``Source``-backed
    "never measured" about a probe whose entire function is to be measured -- a
    fabricated measurement status with click-back intact. That was fixed for a
    correctly-cased store only; on a mixed-case one the probe was back ON the panel, so
    the tool reported a negative control as a measured panel gene with an annotation.

    On an off-target panel this is the one fact that matters: the controls are how you
    decide whether a flat kidney-resident signal is biology or the assay floor.
    """
    from panoscope.agent import tools

    st, named = control_store
    with cfg.use_study(st):
        env = tools.panel_lookup(named)
        assert env["ok"]
        assert env["data"]["on_panel"] is False   # the denominator does not move
        assert env["data"]["control"] is True
        assert env["data"]["annotation"] == "negative control"
        detail = env["sources"][0]["detail"]
        assert "never measured" not in detail
        assert "control probe" in detail and "negative control" in detail

        # The third answer must not blunt the real one: a genuinely off-panel gene
        # still gets the never-measured sentence.
        off = tools.panel_lookup("COL1A1")
        assert off["data"]["on_panel"] is False and off["data"]["control"] is False
        assert "never measured" in off["sources"][0]["detail"]

        # ...and a control is still not a PANEL annotation, in any case.
        assert data.control_annotation(named) == "negative control"
        assert data.panel_annotation(named) is None


def test_control_annotation_is_reachable_and_only_for_controls(store: Study) -> None:
    """Dropped from the panel, not from the file -- that is what keeps the sentence true."""
    with cfg.use_study(store):
        assert data.control_annotation(CONTROL) == "negative control"
        assert data.control_annotation(CONTROL.lower()) == "negative control"
        assert data.control_annotation("SYNG01") is None   # a panel gene is not a control
        assert data.control_annotation("COL1A1") is None   # nor is an off-panel gene
        assert data.panel_annotation(CONTROL) is None      # still not a PANEL annotation
