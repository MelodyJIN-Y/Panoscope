"""Property tests: the universals the spec states, checked for ANY input (plan Stage 2).

Two rules for every property here (TEST_PLAN §2.5):

* **No vacuous passes.** Each property counts how often its perturbation actually
  reached the code under test and asserts a floor on that share after the run
  (:func:`_run`). "Nothing raises a band" passes trivially if the addition never gets
  past the driver gate; the floor is what makes the pass mean something.
* **Deterministic per commit.** The ``commit`` profile (conftest) is derandomised, 200
  examples, no database. ``HYPOTHESIS_PROFILE=deep`` runs 2000 random examples nightly.
  The few properties that build a study on disk run a fixed fraction of the profile's
  examples (:data:`HEAVY`).

Gene vocabularies are real: on-panel names from the demo's 280-gene panel, and
off-panel names from real kidney markers the breast panel never carried. Expected
values come from the inputs and the spec, never from the module under test.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, event, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from panoscope.core import measure as core_measure  # noqa: E402
from panoscope.core.concordance import Replicate, concordance  # noqa: E402
from panoscope.core.interpret import interpret  # noqa: E402
from panoscope.core.verdict import (  # noqa: E402
    BANDS, FLAG_LOCALISES_ELSEWHERE, FLAG_LOW_PEARSON, FLAG_NOT_UNIQUE, PEARSON, assess,
)
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance  # noqa: E402
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence, UngroundedStatistic  # noqa: E402
from panoscope.types.measured import (  # noqa: E402
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.outputs import Finding, TaskKind  # noqa: E402
from panoscope.types.subject import (  # noqa: E402
    EntityKind, Gate, cell_type, gene, gene_set, subject, unit,
)
from panoscope.workflows import enrichment as wf_enrichment  # noqa: E402
from panoscope.workflows import marker as wf_marker  # noqa: E402

# Real demo-panel genes (asserted against the panel file below) and real kidney
# markers that the breast panel never measured.
ON_PANEL = ("ABCC11", "ANGPT2", "BANK1", "CCL5", "CD3G", "CTLA4", "ERBB2", "FLNB",
            "HAVCR2", "ITGAM", "KRT14", "MMP12", "PDGFRA", "PRF1", "RUNX1", "TPD52")
OFF_PANEL = ("NPHS1", "NPHS2", "UMOD", "LRP2", "AQP2", "SLC12A1")
CELL_TYPES = ("Tumor", "Macrophages", "T cells", "B cells", "Podocyte")
UNITS = ("c1", "c2", "c3", "c4")
RANK = {s: i for i, s in enumerate((MeasuredState.NEVER_MEASURED, MeasuredState.UNDERPOWERED,
                                    MeasuredState.TESTED_ABSENT, MeasuredState.OBSERVED))}

#: Examples for the properties that write and build a study on disk.
HEAVY = settings(max_examples=max(10, settings().max_examples // 8),
                 suppress_health_check=[HealthCheck.function_scoped_fixture,
                                        HealthCheck.too_slow])


def _run(prop, floors: dict[str, float]) -> Counter:
    """Run a ``@given`` property that takes a ``hits`` Counter, then hold the floors.

    ``hits["n"]`` counts examples; each other key is a path the example reached.
    ``floors`` maps a key to the minimum share of examples that must reach it.
    """
    hits: Counter = Counter()
    prop(hits)
    for key, share in floors.items():
        assert hits["n"] and hits[key] >= share * hits["n"], (
            f"vacuous: {key!r} reached in {hits[key]} of {hits['n']} examples, "
            f"floor {share:.0%}")
    return hits


def _ev(items: tuple[EvidenceItem, ...], u: str = "c1") -> NormalisedEvidence:
    panel = Frame(EntityKind.GENE, "panel:prop", frozenset(ON_PANEL))
    return NormalisedEvidence(
        unit=u, unit_kind="cluster", unit_subject=subject("unit", u=unit(u)),
        unit_measured=Measured(MeasuredState.OBSERVED, "clusters:prop"),
        module="jazzpanda", items=items, reported=(panel,),
        conditions=Conditions(
            assay=Condition("xenium", Provenance.READ),
            modality=Condition("spatial_transcriptomics", Provenance.READ),
            tissue=Condition("UBERON:0000310", Provenance.STATED),
            method=Condition("jazzpanda", Provenance.STATED),
            segmentation=UNKNOWN, fixation=UNKNOWN, frames=(panel.ref(),)))


def _band(b: str) -> int:
    """Higher is LOWER confidence: index into the ladder, Very High first."""
    return BANDS.index(b)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
coefs = st.floats(-15, 15, allow_nan=False, allow_infinity=False)


@st.composite
def marker_items(draw, min_size=1):
    """A unit's marker items in method order: observed rows with a glm_coef, pearson
    and specificity flags, plus some tested-absent rows (no number)."""
    genes = draw(st.lists(st.sampled_from(ON_PANEL), min_size=min_size, max_size=7,
                          unique=True))
    out = []
    for g in genes:
        if draw(st.integers(0, 5)) == 0:
            out.append(EvidenceItem(subject("gene", g=gene(g)), None, "glm_coef",
                                    Measured(MeasuredState.TESTED_ABSENT, "panel:prop")))
            continue
        flags = draw(st.lists(st.sampled_from(
            (FLAG_LOW_PEARSON, FLAG_LOCALISES_ELSEWHERE, FLAG_NOT_UNIQUE)), unique=True,
            max_size=2))
        out.append(EvidenceItem(
            subject("gene", g=gene(g)), draw(coefs), "glm_coef",
            Measured(MeasuredState.OBSERVED, "panel:prop"),
            corroborating=((PEARSON, draw(st.floats(-1, 1))),), flags=tuple(flags)))
    return tuple(out)


def _absent(g: str) -> EvidenceItem:
    """An off-panel canonical marker, adjudicated by the real lattice."""
    world = MeasuredWorld((Frame(EntityKind.GENE, "panel:prop", frozenset(ON_PANEL)),))
    return core_measure.measure(core_measure.Reading(subject("gene", g=gene(g)), None,
                                                     "glm_coef"), world, frozenset()).value


# --------------------------------------------------------------------------- #
def test_the_vocabularies_are_real_demo_panel_genes():
    import pandas as pd

    panel = set(pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "studies"
                                / "xenium_hbreast_sample1" / "inputs" / "panel.parquet")["gene"])
    assert set(ON_PANEL) <= panel and not set(OFF_PANEL) & panel


# --------------------------------------------------------------------------- #
# 2.1  statistic is None iff not OBSERVED
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-MS-01", "CLAUDE-MS-03")
def test_a_statistic_exists_iff_the_subject_was_observed_at_construction():
    @given(st.sampled_from(MeasuredState), st.one_of(st.none(), coefs))
    def prop(hits, state, stat):
        hits["n"] += 1
        ok = (stat is not None) == (state is MeasuredState.OBSERVED)
        try:
            item = EvidenceItem(subject("gene", g=gene("ERBB2")), stat, "glm_coef",
                                Measured(state, "panel:prop"))
        except UngroundedStatistic:
            assert not ok
            hits["refused"] += 1
            return
        assert ok and item.statistic == stat
        hits["built"] += 1

    _run(prop, {"refused": 0.2, "built": 0.2})


@pytest.mark.req("CLAUDE-MS-01", "CLAUDE-MS-02", "CLAUDE-MS-03")
def test_measuring_any_reading_keeps_a_number_only_where_observed():
    """Through the real adjudication: a reported number survives only for an OBSERVED
    subject and is then the method's own value; UNDERPOWERED withholds it too."""
    @given(st.sampled_from(ON_PANEL + OFF_PANEL), st.sets(st.sampled_from(ON_PANEL)),
           st.booleans(), st.booleans(), st.one_of(st.none(), coefs),
           st.integers(0, 3).map(lambda k: k == 0))
    def prop(hits, g, panel, in_frame, said, stat, thin):
        hits["n"] += 1
        # The frame admits g only if it is a real panel gene; "reported" is the rows
        # the method gave a number to.
        panel = panel | {g} if in_frame and g in ON_PANEL else panel - {g}
        reported = {g} if said and stat is not None else set()
        world = MeasuredWorld((Frame(EntityKind.GENE, "panel:prop", frozenset(panel)),)
                              if panel else ())
        cov = (Coverage("cells", 1, 10, 5),) if thin else ()
        item = core_measure.measure(core_measure.Reading(
            subject("gene", g=gene(g)), stat, "glm_coef", coverage=cov), world,
            frozenset(reported)).value
        state = item.measured.state
        hits[state.value] += 1
        event(state.value)
        assert (item.statistic is not None) == (state is MeasuredState.OBSERVED
                                                and stat is not None)
        if item.statistic is not None:
            assert item.statistic == stat
        assert item.measured.absence_is_evidence == (state is MeasuredState.TESTED_ABSENT)
        if g not in panel:
            assert state is MeasuredState.NEVER_MEASURED

    _run(prop, {s.value: 0.03 for s in MeasuredState})


# --------------------------------------------------------------------------- #
# 2.2  the meet over required roles
# --------------------------------------------------------------------------- #
_SINGLE = {EntityKind.GENE: ("gene", "g"), EntityKind.CELL_TYPE: ("cell_type", "ct"),
           EntityKind.UNIT: ("unit", "u")}


@st.composite
def composites(draw):
    def g():
        return gene(draw(st.sampled_from(ON_PANEL + OFF_PANEL)))

    def c():
        return cell_type(draw(st.sampled_from(CELL_TYPES)))

    shape = draw(st.sampled_from(("gene_pair", "cell_type_pair", "adjacency",
                                  "interaction", "niche")))
    if shape == "gene_pair":
        return subject(shape, a=g(), b=g())
    if shape == "cell_type_pair":
        return subject(shape, a=c(), b=c())
    if shape == "adjacency":
        return subject(shape, centre=c(), neighbour=c())
    if shape == "interaction":     # ligand-receptor, four required roles
        return subject(shape, ligand=g(), receptor=g(), source=c(), target=c())
    return subject("niche", niche=unit(draw(st.sampled_from(UNITS))),
                   component=[c() for _ in range(draw(st.integers(1, 3)))])


@pytest.mark.req("ARCH-9.5-01", "CLAUDE-MS-01")
def test_a_composite_is_the_meet_of_its_required_roles_and_names_the_blocker():
    @given(composites(),
           st.sets(st.sampled_from(ON_PANEL)), st.sets(st.sampled_from(CELL_TYPES)),
           st.sets(st.sampled_from(UNITS)), st.sets(st.sampled_from(
               ON_PANEL + OFF_PANEL + CELL_TYPES + UNITS)))
    def prop(hits, subj, genes, cts, units, reported):
        hits["n"] += 1
        frames = tuple(Frame(k, f"{k.value}:prop", frozenset(m)) for k, m in (
            (EntityKind.GENE, genes), (EntityKind.CELL_TYPE, cts), (EntityKind.UNIT, units))
            if m)
        world, rep = MeasuredWorld(frames), frozenset(reported)
        got = core_measure.resolve(subj, world, rep).value
        single = {}
        for _r, e in subj.roles_with_gate(Gate.REQUIRED):
            shape, role = _SINGLE[e.kind]
            single[e] = core_measure.resolve(subject(shape, **{role: e}), world,
                                             rep).value.state
        worst = min(RANK[s] for s in single.values())
        assert RANK[got.state] == worst            # never better than any role
        blockers = {e for e, s in single.items() if RANK[s] == worst}
        if got.state is MeasuredState.OBSERVED:
            assert got.blocked_by == ()
        else:
            assert set(got.blocked_by) == blockers   # names what pulled it down
            assert all(str(e) in " ".join(got.because) for e in blockers)
        if len(set(single.values())) > 1:
            hits["mixed"] += 1
        hits[subj.shape] += 1

    _run(prop, {"mixed": 0.2, "interaction": 0.1, "niche": 0.1})


# --------------------------------------------------------------------------- #
# 2.3  nothing raises a band
# --------------------------------------------------------------------------- #
@pytest.mark.req("IMPL-6-01", "ARCH-18-08", "ARCH-2.2-01", "ARCH-9.10-01")
def test_replicates_and_off_panel_reference_never_raise_a_band_or_clear_verify():
    """Base: the band off the method's items alone. Then add what every other layer
    can add -- a second section's concordance (§9.10) and off-panel canonical markers
    from the reference -- and the band may only hold or fall, verify only be set."""
    @given(marker_items(), st.sets(st.sampled_from(ON_PANEL)),
           st.lists(st.sampled_from(OFF_PANEL), unique=True, max_size=4),
           st.data())
    def prop(hits, items, canon, off, data):
        # The leading observed marker is canonical, so the ladder has a driver to move.
        canon = frozenset(canon) | {next((i.subject.one("g").id for i in items
                                          if i.statistic is not None), "ERBB2")}
        base = assess(items, canon).value
        # The second section: same subjects, some dropped, some signs flipped.
        second = tuple(
            dataclasses.replace(i, statistic=-i.statistic) if (
                i.statistic is not None and data.draw(st.booleans())) else i
            for i in items if data.draw(st.integers(0, 3)))
        hits["n"] += 1    # after the last draw: an aborted draw is not an example
        rep = concordance("c1", (Replicate("s1", items), Replicate("s2", second)),
                          joint=True).value.coverage
        more = items + tuple(_absent(g) for g in off)
        after = assess(more, canon | frozenset(off), rep).value
        assert _band(after.band) >= _band(base.band)
        assert after.verify or not base.verify
        # The driver's number is the method's number; no layer supplies another.
        if after.driver is not None:
            assert after.driver in items
        if base.driver is not None and ((rep.n and not rep.ok) or off):
            hits["reached"] += 1
        if _band(after.band) > _band(base.band):
            hits["lowered"] += 1

    _run(prop, {"reached": 0.5, "lowered": 0.05})


def _seed(root: Path, pairs: list[dict]) -> None:
    p = root / "claims" / "canonical_markers_seed.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pairs": pairs, "summary": {"total": len(pairs)}}))


def _numbers(built) -> dict:
    """Everything a claim must not move: band, verify, rungs, coverage, N, values."""
    return {u: (b.verdict.band, b.verdict.verify, b.verdict.demotions,
                str(b.verdict.coverage), len(b.evidence.items),
                tuple((i.subject.key, i.statistic, i.measured.state)
                      for i in b.evidence.items))
            for u, b in built.items()}


@pytest.mark.req("IMPL-6-01", "CLAUDE-RD-01", "CORP-B1", "CORP-A27")
def test_lab_claims_with_any_support_grade_or_off_panel_flag_move_no_number(study_builder):
    """A curated claim is reference data: random seeds of (cell type, gene) pairs with
    any support grade and any ``off_panel`` column reach the claim gate and change no
    band, verify, rung, coverage, N or statistic."""
    from panoscope.data.study import Study
    from panoscope.pipeline import records
    from tests.conftest import clear_all_caches

    study_builder("s", clusters=["c1", "c3", "c5"])
    labels = sorted({m["cell_type"] for m in Study.open("s").unit_labels.values()})
    seed = study_builder.root / "claims" / "canonical_markers_seed.json"
    clear_all_caches()
    base = _numbers(records.build_all(Study.open("s")))
    assert not seed.exists()

    # Real pairs and their real citations from the committed seed (never a made-up
    # PMID). Randomised: the support grade (a cited pair is supported or weak, an
    # uncited one none), the off_panel column, and the label it is about.
    real = json.loads((Path(__file__).resolve().parents[1] / "data" / "claims"
                       / "canonical_markers_seed.json").read_text())["pairs"]

    @st.composite
    def pair(draw):
        p = dict(draw(st.sampled_from(real)))
        p["support"] = draw(st.sampled_from(("supported", "weak"))) if p.get("pmid") \
            else "none"
        p["off_panel"] = draw(st.booleans())
        p["cell_type"] = draw(st.sampled_from(labels + [p["cell_type"]]))
        return p

    @HEAVY
    @given(st.lists(pair(), min_size=1, max_size=8))
    def prop(hits, pairs):
        hits["n"] += 1
        _seed(study_builder.root, pairs)
        clear_all_caches()
        built = records.build_all(Study.open("s"))
        assert _numbers(built) == base
        if any(b.record.claims_used for b in built.values()):
            hits["reached"] += 1

    try:
        _run(prop, {"reached": 0.5})
    finally:
        seed.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 2.4  verify comes from the rubric only
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-HR-05", "ARCH-8.2-01", "CLAUDE-RD-01")
def test_verify_and_band_ignore_the_label_text_rivals_and_reference_ids():
    """The model's words for the call, the rivals a reference lists and the label id are
    not rubric inputs: the assignment's band and verify are ``assess`` on the items."""
    words = st.text(max_size=30)

    @given(marker_items(), st.sets(st.sampled_from(ON_PANEL), min_size=1), words,
           st.lists(st.tuples(st.sampled_from(CELL_TYPES), st.lists(
               st.sampled_from(ON_PANEL + OFF_PANEL), max_size=4).map(tuple)), max_size=3),
           st.one_of(st.none(), words))
    def prop(hits, items, canon, label, rivals, label_id):
        hits["n"] += 1
        ref = wf_marker.MarkerReference(label=label, canonical=tuple(sorted(canon)),
                                        label_id=label_id, rivals=tuple(rivals))
        a = interpret(_ev(items), wf_marker.WORKFLOW, ref).value
        v = assess(items, frozenset(canon)).value
        assert (a.band, a.verify) == (v.band, v.verify)
        # Reference never supplies a value: every number on the assignment is an
        # input number on the same subject.
        given_ = {(i.subject.key, i.statistic) for i in items}
        assert {(i.subject.key, i.statistic) for i in a.drivers} <= given_
        assert all(i.statistic is None for i in a.unmeasured)
        hits["verify" if a.verify else "clear"] += 1

    _run(prop, {"verify": 0.1, "clear": 0.1})


@pytest.mark.req("CLAUDE-HR-05", "ARCH-8.2-01")
def test_an_override_note_with_any_text_or_literature_tension_leaves_verify(demo_id):
    """Composing the biologist's override (its claim text, the literature agree/dissent
    behind it) renames the call; verify, the band and every number stay the rubric's."""
    from panoscope.agent import compose
    from panoscope.agent.types import Citation, Note, ScopeRef, Tension
    from panoscope.pipeline.v1_verdicts import verdicts

    vs = verdicts()
    cite = st.builds(Citation, pmid=st.just(""), title=st.text(max_size=20),
                     authors=st.just(""), year=st.just(0), journal=st.just(""),
                     stance=st.sampled_from(("agree", "dissent", "context")))

    @given(st.sampled_from(sorted(vs)), st.text(max_size=40),
           st.sampled_from(("T cells", "Fibroblasts", "Tumor", "Unknown")),
           st.lists(cite, max_size=3), st.lists(cite, max_size=3), st.booleans())
    def prop(hits, c, claim, call, agree, dissent, thin):
        hits["n"] += 1
        v = vs[c]
        note = Note(id="n", claim=claim, scope="cluster", scope_ref=ScopeRef(demo_id, c),
                    basis="paper", status="tentative", subject_cell_type=call,
                    subject_markers=(), tension=Tension(tuple(agree), tuple(dissent),
                                                        thin, claim, ""),
                    author="me", created_at="", trigger="override", supersedes=None)
        out = compose.compose_verdict(v, note)
        assert out.cell_type == call
        assert (out.verify, out.confidence, out.confidence_score, out.demotions,
                out.evidence, out.source_trace) == (
            v.verify, v.confidence, v.confidence_score, v.demotions, v.evidence,
            v.source_trace)
        if dissent:
            hits["dissent"] += 1

    _run(prop, {"dissent": 0.3})


# --------------------------------------------------------------------------- #
# 2.6  viewing controls change the picture, never a value
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-IF-01", "UI-2.8-01", "ARCH-13.3-01")
def test_the_bin_size_changes_the_hexes_and_caption_but_no_count(demo_id):
    from panoscope.ui import conditionality as cnd
    from panoscope.ui import data_access as da
    from panoscope.ui.state import BIN_SIZES_UM

    @given(st.sampled_from(ON_PANEL), st.sampled_from(BIN_SIZES_UM),
           st.sampled_from(BIN_SIZES_UM))
    def prop(hits, g, b1, b2):
        hits["n"] += 1
        h1, h2 = da.hexbins(g, b1), da.hexbins(g, b2)
        c1 = cnd.gene_density_cond(gene=g, hb=h1, bin_um=b1)
        c2 = cnd.gene_density_cond(gene=g, hb=h2, bin_um=b2)
        # The values: total detections, the denominator, what it is not.
        assert int(h1["count"].sum()) == int(h2["count"].sum())
        assert (c1.denominator, c1.not_this, c1.exclusions, c1.drawn) == (
            c2.denominator, c2.not_this, c2.exclusions, c2.drawn)
        # The plot renders, it does not derive: density is the count over bin area.
        for h, b, c in ((h1, b1, c1), (h2, b2, c2)):
            assert (h["density"] - h["count"] / b ** 2).abs().max() < 1e-9
            assert f"{b} um" in c.transform
        if b1 != b2:     # non-vacuity: a bin control that does nothing fails here
            assert len(h1) != len(h2) and c1.transform != c2.transform
            hits["changed"] += 1

    _run(prop, {"changed": 0.4})


# --------------------------------------------------------------------------- #
# 2.7  ranking ignores everything but the stated lexical score
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-8.1-04")
def test_retrieval_order_ignores_record_order_and_bibliometric_metadata():
    """Never ranked by citation count -- or by year, publication type, open access or
    arrival order. Permuting all of those leaves the ranking unchanged."""
    from panoscope.corpus.retrieve import search
    from panoscope.corpus.store import Corpus, SourceRecord

    vocab = ON_PANEL[:6] + ("tumour", "macrophage", "breast", "spatial")
    text = st.lists(st.sampled_from(vocab), max_size=5).map(" ".join)

    @given(st.lists(st.tuples(text, text), min_size=1, max_size=8),
           st.lists(st.sampled_from(vocab), min_size=1, max_size=3), st.randoms(),
           st.data())
    def prop(hits, docs, terms, rnd, data):
        recs = [SourceRecord(uri=f"pmid:{900000 + k}", title=t or "x", abstract=a)
                for k, (t, a) in enumerate(docs)]
        shuffled = [dataclasses.replace(
            r, year=data.draw(st.integers(1990, 2026)),
            open_access=data.draw(st.booleans()),
            publication_type=tuple(data.draw(st.lists(
                st.sampled_from(("Review", "Journal Article")), max_size=2))),
            via=data.draw(st.sampled_from(("PubMed", "Europe PMC")))) for r in recs]
        rnd.shuffle(shuffled)
        hits["n"] += 1    # after the last draw: an aborted draw is not an example
        a = search(Corpus(records=tuple(recs)), terms, limit=50).value
        b = search(Corpus(records=tuple(shuffled)), terms, limit=50).value
        assert [(h.record.uri, h.score) for h in a.hits] == [
            (h.record.uri, h.score) for h in b.hits]
        if len(a.hits) >= 2:
            hits["ranked"] += 1

    _run(prop, {"ranked": 0.3})


# --------------------------------------------------------------------------- #
# 2.8  task kinds
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-TK-01")
def test_there_are_exactly_three_task_kinds():
    assert {t.value for t in TaskKind} == {"assignment", "characterisation", "comparison"}


@pytest.mark.req("CLAUDE-TK-02", "CLAUDE-TK-04")
def test_a_characterisation_never_carries_a_band_for_any_input():
    sets = ("HALLMARK_G2M_CHECKPOINT", "HALLMARK_EMT", "HALLMARK_APOPTOSIS",
            "HALLMARK_INTERFERON_GAMMA_RESPONSE")

    @st.composite
    def set_items(draw):
        out = []
        for s in draw(st.lists(st.sampled_from(sets), min_size=1, unique=True)):
            state = draw(st.sampled_from(MeasuredState))
            m = Measured(state, "collection:prop", coverage=(Coverage(
                wf_enrichment.PANEL_COVERAGE, draw(st.integers(0, 20)), 200, 3),))
            num = state is MeasuredState.OBSERVED
            out.append(EvidenceItem(
                subject("gene_set", set=gene_set(s)), draw(coefs) if num else None,
                "test_statistic", m,
                corroborating=(("p_adj_bh", draw(st.floats(0, 1))),) if num else ()))
        return tuple(out)

    @given(set_items(), st.sampled_from(("", "Tumor", "Macrophages")))
    def prop(hits, items, call):
        hits["n"] += 1
        f = interpret(_ev(items), wf_enrichment.WORKFLOW,
                      wf_enrichment.EnrichmentReference(cell_type=call)).value
        assert isinstance(f, Finding) and f.task_kind is TaskKind.CHARACTERISATION
        assert not {"band", "confidence", "confidence_score"} & {
            x.name for x in dataclasses.fields(f)}
        words = f.summary.casefold()
        assert not any(b.casefold() in words for b in BANDS if b != "Low")
        assert f"of {len(items)} set(s) scored" in f.summary   # per-item counts
        if any(i.evidence.measured.carries_number for i in f.items):
            hits["scored"] += 1

    _run(prop, {"scored": 0.3})


# --------------------------------------------------------------------------- #
# 2.9  the factual layer is deterministic, and reads its input
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-17-01")
def test_any_builder_study_builds_to_the_same_bytes_and_a_changed_coefficient_shows(
        study_builder):
    import pandas as pd

    from panoscope.core import record as core_record
    from panoscope.data.study import Study
    from panoscope.pipeline import records
    from tests.conftest import clear_all_caches

    def dump(sid):
        clear_all_caches()
        return {u: json.dumps(core_record.to_dict(b.record).value, sort_keys=True)
                for u, b in records.build_all(Study.open(sid)).items()}

    made = Counter()

    @HEAVY
    @given(st.lists(st.sampled_from([f"c{i}" for i in range(1, 10)]), min_size=1,
                    max_size=3, unique=True), st.integers(0, 10 ** 6),
           st.sampled_from((-0.5, 0.25, 1.0, 3.0)))
    def prop(hits, clusters, pick, delta):
        hits["n"] += 1
        made["k"] += 1
        sid = f"s{made['k']}"
        d = study_builder(sid, clusters=sorted(clusters))
        first = dump(sid)
        assert dump(sid) == first
        csv = d / "inputs" / "markers_top.csv"
        df = pd.read_csv(csv)
        rows = df.index[df["top_cluster"].isin(clusters)].tolist()
        r = rows[pick % len(rows)]
        df.loc[r, "glm_coef"] += delta
        df.to_csv(csv, index=False)
        u = df.loc[r, "top_cluster"]
        assert dump(sid)[u] != first[u]
        hits["reached"] += 1

    _run(prop, {"reached": 1.0})
