"""Panel-coverage guarantee — the "8 of 200 genes" penalty, tested WITHOUT the private
jazzPanda enrichment result.

``test_enrichment.py`` pins the same engine but is SKIPPED on a fresh clone / CI (it is
gated on the gitignored jazzPanda CSV existing), so on the public repo the flagship
panel-coverage rule would otherwise have no test coverage at all. These tests drive the
deterministic logic with synthetic gene-set rows, so they run everywhere: a set with
almost none of its genes measured earns the low-coverage caveat and forces ``verify``;
a well-covered set does not; and an untestable cluster reads as untestable rather than
as an empty, confident-looking row.

**There is no band here any more, and that is the point.** These tests used to pin a
confidence ladder (``_band_cluster``) over a CHARACTERISATION — a unit-level band on a
task with no label space. ``ClusterEnrichment`` carries no band now; the header is the
per-item census, and :func:`test_a_cluster_enrichment_has_nowhere_to_put_a_band` pins
the absence structurally, so a later refactor cannot quietly reintroduce one as a
default-valued field that nothing draws but every committed record carries.

The sibling guarantee — the marker-level panel-ABSENCE rule (an off-panel gene's absence
is not evidence against a cell type, and never moves the band) — lives in
``test_verdict.py``, which is not data-gated.
"""
from __future__ import annotations

import dataclasses

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E
from panoscope.types.subject import EntityKind

# The method the synthetic rows claim to come from (jazzPanda enrichment).
_METHOD = E._JZ_SCORE_KIND  # "jazzpanda_enrichment"


def _frame_genes(n: int, *, admitted: bool = True) -> list[str]:
    """``n`` real symbols every gene frame admits (or, with ``admitted=False``, rejects).

    The leading edge is filtered through the study's capability frames, so a
    synthetic row full of invented symbols like ``G0, G1, ...`` would have every
    driver rejected and collapse to ``n_leading_edge == 0`` — the fixture would then
    be exercising the rejection path in every test that meant to isolate something
    else. Real symbols are read off the frames instead of hard-coded, because the
    panel is per study and this file must not assert a gene list.

    ``admitted=False`` returns panel genes the DETECTABILITY frame rejects — on the
    demo, 49 of 280. That is the production shape of a rejected driver: a gene the
    panel designed for and the measurement could not usably see, not a typo.

    This still runs on a fresh clone. The artifact these tests avoid is the
    gitignored jazzPanda RESULT; ``inputs/panel.parquet`` and
    ``inputs/markers_top.csv``, which the frames are read from, are committed.
    """
    frames = E._gene_world().for_kind(EntityKind.GENE)
    sets = [set(f.members) for f in frames]
    pool = set.intersection(*sets) if admitted else (set.union(*sets) - set.intersection(*sets))
    members = sorted(pool)
    assert len(members) >= n, (
        f"the open study declares only {len(members)} genes "
        f"{'every' if admitted else 'not every'} gene frame admits; this fixture "
        f"needs {n}"
    )
    return members[:n]


def _row(*, geneset_size: int, panel_hits: int, n_leading_edge: int = 10,
         test_statistic: float = 15.0, q: float = 0.001) -> dict:
    """A synthetic result row, as ``_load_jazzpanda_rows`` hands it on.

    ``panel_hits`` goes in under :data:`agent.enrichment.PANEL_HITS_IN_FRAME`, NOT
    under ``n_overlap``: the engine refuses the result's own overlap count, and a
    fixture still offering one would be exercising a code path that no longer
    exists. Coverage = panel_hits / geneset_size.

    ``n_leading_edge`` is kept >= 5 by default so the low-coverage rule stays
    isolated from the separate tiny-leading-edge one, and the genes are real ones
    the frames admit so that the leading-edge filter is a no-op here (see
    :func:`_frame_genes`) — this fixture pins the GATES, not the filter.
    """
    genes = ",".join(_frame_genes(n_leading_edge))
    return {
        "gene_set": "HALLMARK_TEST",
        "geneset_size": geneset_size,
        E.PANEL_HITS_IN_FRAME: panel_hits,
        "genes_selected": genes,
        "test_statistic": test_statistic,
        "p_value": 0.001,
        "p_adj_bh": q,
        "gc_corr": 0.80,
    }


def test_low_coverage_earns_caveat_and_exact_fraction():
    # 8 of 200 genes measured -> 4% coverage, below the 5% floor.
    pe = E._pathway_from_jz_row(_row(geneset_size=200, panel_hits=8))
    assert pe.panel_coverage == 8 / 200
    assert "very low panel coverage" in pe.caveats
    # The deterministic panel-scoped caveat names the real fraction — never invented.
    assert pe.panel_scope_caveat.startswith("Panel-scoped:")
    assert "8 of 200" in pe.panel_scope_caveat
    # ...and says USABLY MEASURED, not "on the panel": panel_hits is the count every
    # gene frame admits, so a panel gene with no usable signal is not in it.
    assert "usably measured" in pe.panel_scope_caveat


def test_adequate_coverage_is_not_flagged():
    pe = E._pathway_from_jz_row(_row(geneset_size=40, panel_hits=20))  # 50%
    assert pe.panel_coverage == 0.5
    assert "very low panel coverage" not in pe.caveats


def test_a_row_without_the_frame_checked_count_is_refused():
    """The soft fallback this engine must never grow back.

    A row carrying only the result's own ``n_overlap`` is a row whose footprint was
    never checked against the capability frames. Reading it would credit a set with
    genes the measurement never usably saw -- the panel-absence rule broken in the
    arithmetic, and silently, because every downstream number would still look
    well-formed. It raises instead.
    """
    row = _row(geneset_size=200, panel_hits=8)
    del row[E.PANEL_HITS_IN_FRAME]
    row["n_overlap"] = 8                      # the tempting, wrong number
    with pytest.raises(KeyError, match=E.PANEL_HITS_IN_FRAME):
        E._pathway_from_jz_row(row)


def test_low_coverage_forces_verify_without_inventing_a_band():
    """Coverage below the floor is what ``verify`` is for, now that no band exists.

    Same strong score; only coverage differs. The well-covered cluster is not
    flagged, the thin one is. This is the rule that used to express itself as a
    one-band demotion; it did not disappear with the band, it moved to the only
    output a task with no label space can carry it in.
    """
    well = E._cluster_enrichment(
        "c1", (E._pathway_from_jz_row(_row(geneset_size=40, panel_hits=20)),), _METHOD)
    thin = E._cluster_enrichment(
        "c1", (E._pathway_from_jz_row(_row(geneset_size=200, panel_hits=8)),), _METHOD)
    assert well.verify is False
    assert thin.verify is True
    # ...and both surfaced their set: verify flags the call, it is not a tier gate.
    assert well.census == (1, 0, 0) and thin.census == (1, 0, 0)


@pytest.mark.req("IMPL-10-01")
def test_a_cluster_enrichment_has_nowhere_to_put_a_band():
    """The no-band invariant, pinned on the type rather than on one value.

    ``methods/jazzpanda_cgst.rubric`` raises and ``types/outputs.Finding`` declares no
    band field; this is the v1 object's half of the same guarantee. Asserting over the
    FIELD NAMES is deliberate: a band that came back as a default-valued field would
    still be written into every committed record by ``dataclasses.asdict``, which is
    exactly how the deleted one survived being taken off the screen.
    """
    ce = E._cluster_enrichment(
        "c1", (E._pathway_from_jz_row(_row(geneset_size=40, panel_hits=20)),), _METHOD)
    names = {f.name for f in dataclasses.fields(ce)}
    assert not (names & {"band", "confidence", "confidence_score", "demotions"}), names
    assert "confidence" not in dataclasses.asdict(ce)


@pytest.mark.req("UI-2.6-01")
def test_the_census_counts_every_scored_set_exactly_once():
    """supported + suggestive + untestable == everything scored. No set goes missing.

    The untestable count is the one that matters: those rows are neither exported nor
    surfaced, so an undercount would let a reader see "nothing else was running"
    where the truth is "nothing else could be asked".
    """
    pathways = (
        E._pathway_from_jz_row(_row(geneset_size=40, panel_hits=20, q=0.0)),
        E._pathway_from_jz_row(_row(geneset_size=40, panel_hits=20, q=cfg.ENRICH_Q_MAX)),
        E._pathway_from_jz_row(_row(geneset_size=200, panel_hits=1)),
        E._pathway_from_jz_row(_row(geneset_size=200, panel_hits=50, q=0.99)),
    )
    ce = E._cluster_enrichment("c1", pathways, _METHOD)
    assert ce.census == (1, 1, 2)
    assert sum(ce.census) == len(ce.all_tested) == 4
    assert ce.census_line == "1 supported · 1 suggestive · 2 untestable"


# --------------------------------------------------------------------------- #
# The TIER gates — the q-value logic that otherwise only runs locally.
#
# ``test_enrichment.py`` exercises these against the real jazzPanda result, but
# that whole module is skipped on a fresh clone / CI (the result CSV is
# gitignored), and the golden hash cannot see two of them on the demo data:
#
#   * ``ENRICH_Q_MAX`` / ``SUGGESTIVE_Q_MAX`` — the demo study has no gene set in
#     the suggestive window, so the suggestive branch never executes there
#   * ``MIN_PANEL_HITS`` — ``n_leading_edge <= panel_hits`` always holds (the
#     leading edge is a SUBSET of the measured set genes), so while
#     ``MIN_LEADING_EDGE >= MIN_PANEL_HITS`` the leading-edge gate subsumes the
#     panel-hits gate and nothing real can trip it on its own
#
# Synthetic rows can hold the two apart, which is the only way to pin each gate
# independently. These run everywhere.
# --------------------------------------------------------------------------- #
def test_panel_hits_gate_is_what_makes_a_set_untestable():
    """Below MIN_PANEL_HITS -> untestable, however good the q and the leading edge.

    The row is deliberately impossible in real data (a leading edge larger than the
    measured footprint) so the panel-hits gate is isolated from the leading-edge one.
    """
    too_thin = E._pathway_from_jz_row(
        _row(geneset_size=200, panel_hits=cfg.MIN_PANEL_HITS - 1, n_leading_edge=10, q=0.0)
    )
    assert too_thin.panel_hits < cfg.MIN_PANEL_HITS
    assert too_thin.tier == "untestable"

    just_enough = E._pathway_from_jz_row(
        _row(geneset_size=200, panel_hits=cfg.MIN_PANEL_HITS, n_leading_edge=10, q=0.0)
    )
    assert just_enough.tier == "enriched"


def test_leading_edge_gate_is_independent_of_panel_hits():
    """Plenty measured, too few driving -> untestable. The 1-gene-artifact rule."""
    thin_edge = E._pathway_from_jz_row(
        _row(geneset_size=200, panel_hits=50, n_leading_edge=cfg.MIN_LEADING_EDGE - 1, q=0.0)
    )
    assert thin_edge.panel_hits >= cfg.MIN_PANEL_HITS
    assert thin_edge.tier == "untestable"


@pytest.mark.req("SKE-S1-01")
def test_q_value_tiers_at_both_thresholds():
    """enriched < ENRICH_Q_MAX <= suggestive <= SUGGESTIVE_Q_MAX < untestable.

    Both boundaries are pinned, including which side of each is inclusive — a
    silent flip of ``<`` to ``<=`` would promote a set the floor refuses.
    """
    def tier(q: float) -> str:
        return E._pathway_from_jz_row(
            _row(geneset_size=200, panel_hits=50, n_leading_edge=10, q=q)
        ).tier

    assert tier(0.0) == "enriched"
    assert tier(cfg.ENRICH_Q_MAX / 2) == "enriched"
    # the enriched bar is strict: q == ENRICH_Q_MAX is NOT enriched
    assert tier(cfg.ENRICH_Q_MAX) == "suggestive"
    assert tier(cfg.SUGGESTIVE_Q_MAX) == "suggestive"
    # above the suggestive bar nothing is surfaced
    assert tier(cfg.SUGGESTIVE_Q_MAX + 0.01) == "untestable"


def test_missing_q_value_is_untestable_never_assumed_significant():
    """No q -> untestable. The floor never promotes a set on a missing number."""
    row = _row(geneset_size=200, panel_hits=50, n_leading_edge=10)
    row["p_adj_bh"] = ""
    pe = E._pathway_from_jz_row(row)
    assert pe.q_value is None
    assert pe.tier == "untestable"


@pytest.mark.req("SKE-S1-02")
def test_untestable_sets_are_never_surfaced_and_force_verify():
    """A cluster whose only sets are untestable: nothing surfaced, verify TRUE.

    The CHARACTERISATION workflow carries no unit-level band, so the census is how an
    untestable cluster must read — "0 supported · 0 suggestive · 2 untestable", not
    an empty, confident-looking row.
    """
    pathways = (
        E._pathway_from_jz_row(
            _row(geneset_size=200, panel_hits=50, n_leading_edge=1, q=0.0)
        ),
        E._pathway_from_jz_row(
            _row(geneset_size=200, panel_hits=50, n_leading_edge=10, q=0.99)
        ),
    )
    ce = E._cluster_enrichment("c1", pathways, _METHOD)
    assert ce.enriched == () and ce.suggestive == ()
    assert ce.all_tested == pathways          # the audit keeps every tested set
    assert ce.top_theme is None
    assert ce.census_line == "0 supported · 0 suggestive · 2 untestable"
    assert ce.verify is True


# --------------------------------------------------------------------------- #
# The leading-edge filter — the same capability frames that recount ``panel_hits``,
# applied to the genes the record NAMES and ``ui/enrichment_spatial`` PLOTS.
#
# ``tests/test_enrichment_frame_count.py`` pins this on the real result; these two
# run everywhere, and they are where the floor interaction is isolated.
# --------------------------------------------------------------------------- #
def test_a_selected_gene_the_frames_reject_is_not_a_driver():
    """Rejected drivers are not counted, not named — and not silently dropped.

    The record used to say "only 3 of 36 genes were usably measured" and then print
    four driving genes, two of which the detectability frame rejects: one record,
    two disagreeing accounts of the same measurement. Both ways of being rejected
    are covered here -- a panel gene with no usable signal (the production shape)
    and a symbol no frame has ever heard of -- because "every frame admits it" is
    the rule, not "the panel admits it".

    The count of what was refused is NOT asserted on the row: it is a fact about the
    raw result that a record rebuilt from disk cannot recover, so it is reported at
    build time instead (``pipeline/stages/enrichment._report_divergence``).
    """
    rejected = _frame_genes(1, admitted=False)[0]
    row = _row(geneset_size=200, panel_hits=50, n_leading_edge=4, q=0.0)
    kept = row["genes_selected"].split(",")
    row["genes_selected"] = ",".join([*kept, rejected, "NOT_A_GENE_ON_ANY_PANEL"])

    pe = E._pathway_from_jz_row(row)
    assert pe.leading_edge == tuple(kept)
    assert rejected not in pe.leading_edge
    assert pe.n_leading_edge == 4
    assert pe.tier == "enriched"          # still over the floor; only the list shrank


def test_frame_rejection_can_push_a_set_under_the_leading_edge_floor():
    """The floor is hit more often now, and that is the floor working.

    A set whose drivers are mostly undetectable was never testable to that depth,
    so it is UNTESTABLE — the state that argues nothing — and not an enriched call
    resting on genes that produced no usable signal. The caveat says why, so the
    row cannot read as "this program is simply not running here".
    """
    row = _row(geneset_size=200, panel_hits=50,
               n_leading_edge=cfg.MIN_LEADING_EDGE - 1, q=0.0)
    kept = row["genes_selected"].split(",")
    row["genes_selected"] = ",".join([*kept, *_frame_genes(3, admitted=False)])

    pe = E._pathway_from_jz_row(row)
    assert pe.n_leading_edge == cfg.MIN_LEADING_EDGE - 1 < cfg.MIN_LEADING_EDGE
    assert pe.tier == "untestable"
    # ...and nothing about it reads as a measured absence: the row the surface draws
    # for this state is UNDERPOWERED, which argues nothing.
    assert "tiny leading edge" in pe.caveats


@pytest.mark.req("SKE-S2-01")
def test_the_denominator_travels_into_the_chat_bubble_too(monkeypatch) -> None:
    """The same claim, the same authority, one surface carrying its conditionality.

    ``ui/report.py`` names K of N for every program in the exported paragraph,
    because a set scored on 5 of its 100 genes reads exactly like one scored on
    90 without it. The Pathways chat opened with "Top programs - <name>: ..." and
    no denominator at all; on the real demo that program is scored on 5 of 100.

    SYNTHETIC, like the rest of this file: the jazzPanda enrichment result is
    gitignored, and a fresh clone must still run this guarantee rather than skip
    it.
    """
    from panoscope.ui import enrichment_conversation as ec

    ce = E._cluster_enrichment(
        "c1",
        (E._pathway_from_jz_row(_row(geneset_size=100, panel_hits=5, q=0.0)),),
        _METHOD,
    )
    assert ce.enriched, "the fixture must produce an enriched program to name"
    monkeypatch.setattr(ec, "_ce", lambda _c: ce)
    monkeypatch.setattr(ec.da, "pathway_note", lambda _c, _s: {})

    text, _pmids, _verify = ec._opening_text("c1")
    assert "Top programs" in text, text
    assert "[5/100 set genes on panel]" in text, (
        f"the program is named with no denominator:\n{text}"
    )
