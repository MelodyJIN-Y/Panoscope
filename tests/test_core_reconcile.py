"""Invariants for the join, and the one distinction the whole project rests on.

Everything here is built from synthetic frames and synthetic readings, on purpose.
``test_core_rival.py`` is the module that must agree with the real study; this one is
about a rule that has to hold for inputs the demo study does not contain -- a second
method with a *different* panel, which is exactly the kidney study's situation and
exactly the case a demo-derived fixture cannot produce. Nothing is invented either
way: every item below goes through ``core.measure``, so its measurement status is
settled by the lattice and ``EvidenceItem`` raises if a number and a state ever
disagree.

**The test the module exists for** is ``test_the_three_cases_read_differently``.
Three subjects, one join: one both sides saw and agreed about, one both sides were
capable of seeing and read oppositely, and one the second side's panel never
contained. The third must NOT read as disagreement. If it ever does, the four-state
lattice has bought nothing and the confident-floor rule is broken in the one place it
is most expensive -- a reconciliation that says two methods conflict because one of
them could not look.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from panoscope.core.adjacency import Adjacency, AdjacencyRead, adjacency_check
from panoscope.core.concordance import (
    Concordance, IndependentClustering, Replicate, TooFewReplicates, concordance,
)
from panoscope.core.measure import Reading, measure
from panoscope.core.reconcile import (
    Agreement, Comparison, DifferentUnits, ReconcileResult, Reconciliation,
    Relationship, Side, reconcile,
)
from panoscope.types.claim import Authority
from panoscope.types.evidence import Direction, EvidenceItem
from panoscope.types.measured import Coverage, Frame, MeasuredState, MeasuredWorld
from panoscope.types.outputs import TaskKind
from panoscope.types.subject import EntityKind, gene, subject
from panoscope.types.trace import RULES

CORE = Path(__file__).resolve().parents[1] / "panoscope" / "core"

# Two methods with DIFFERENT panels. That difference is the entire fixture: with one
# shared panel every absence is a tested absence and the distinction under test
# cannot arise.
PANEL_A = ("ERBB2", "ESR1", "MKI67", "CD3D")
PANEL_B = ("ERBB2", "ESR1", "MKI67")          # no CD3D -- B could never see it


def _world(name: str, panel: tuple[str, ...]) -> MeasuredWorld:
    return MeasuredWorld((Frame(EntityKind.GENE, name, frozenset(panel)),))


def _items(world: MeasuredWorld, stat_name: str,
           rows: tuple[tuple[str, float | None, Direction | None], ...],
           shared_with: tuple[str, ...] = ()) -> tuple[EvidenceItem, ...]:
    """Readings adjudicated by the lattice. ``statistic is None`` -> tested absent.

    Nothing here hand-builds a ``Measured``: the state is whatever the frame and the
    reported set say it is, so a fixture cannot quietly assert a status the frame
    would have refused.
    """
    reported = frozenset(sym for sym, stat, _d in rows if stat is not None)
    out = []
    for sym, stat, direction in rows:
        r = Reading(subject("gene", g=gene(sym)), stat, stat_name,
                    direction=direction, shared_with=shared_with)
        out.append(measure(r, world, reported).value)
    return tuple(out)


@pytest.fixture
def marker_side() -> Side:
    """jazzPanda-shaped: a spatial marker call, ASSIGNMENT."""
    world = _world("panel:a", PANEL_A)
    return Side(
        module="jazzpanda", workflow="marker-interpretation",
        task_kind=TaskKind.ASSIGNMENT, unit="c1", authority=Authority.MEASUREMENT,
        items=_items(world, "glm_coef", (
            ("ERBB2", 21.4, Direction.UP),    # both saw it, both up      -> agree
            ("ESR1", 8.2, Direction.UP),      # B reads it down           -> disagree
            ("MKI67", 4.1, Direction.UP),     # B tested it, absent       -> disagree
            ("CD3D", 3.0, Direction.UP),      # B's panel never had it    -> cannot
        )),
        world=world,
    )


@pytest.fixture
def de_side() -> Side:
    """A second ASSIGNMENT method on a narrower panel. Same question, other apparatus."""
    world = _world("panel:b", PANEL_B)
    return Side(
        module="seurat_de", workflow="marker-interpretation",
        task_kind=TaskKind.ASSIGNMENT, unit="c1", authority=Authority.MEASUREMENT,
        items=_items(world, "log2fc", (
            ("ERBB2", 3.3, Direction.UP),
            ("ESR1", -1.8, Direction.DOWN),
            ("MKI67", None, None),            # in frame, tested, not reported
        )),
        world=world,
    )


# --------------------------------------------------------------------------- #
# The demonstration. This is the figure the module exists to produce.
# --------------------------------------------------------------------------- #
def demonstrate(left: Side, right: Side) -> tuple[ReconcileResult, str]:
    """Render the join as the review surface would read it."""
    out = reconcile(left, right, Relationship.PARTLY_INDEPENDENT)
    lines = [
        f"unit {out.value.unit}: {out.value.left} vs {out.value.right} "
        f"({out.value.relationship.value}, "
        f"same question: {out.value.same_question})",
        "",
    ]
    for p in out.value.pairs:
        lines.append(f"  {p.subject.display():<8} {p.outcome.value:<15} "
                     f"corroborates={str(p.corroborates):<5} "
                     f"blocked={','.join(p.blocked) or '-'}")
        for why in p.because:
            lines.append(f"      {why}")
    return out.value, "\n".join(lines)


def test_the_three_cases_read_differently(marker_side, de_side, capsys) -> None:
    """Agreement, conflict and could-not-be-compared are three outcomes, not two.

    ``CD3D`` is the case. Both the agreement and the conflict are unremarkable; what
    this project is for is the third row, where the second method's panel never
    contained the gene. A join that reported that as disagreement would be stating a
    negative observation no measurement produced -- the fabrication the four states
    make unrepresentable everywhere else, re-entering through the join.
    """
    result, figure = demonstrate(marker_side, de_side)
    with capsys.disabled():
        print("\n" + figure)

    by_key = {p.subject.one("g").id: p for p in result.pairs}

    # (a) genuine agreement
    assert by_key["ERBB2"].outcome is Agreement.AGREE
    assert by_key["ERBB2"].corroborates is True
    assert by_key["ERBB2"].blocked == ()

    # (b) genuine conflict, twice over: opposite directions, and a measured absence
    assert by_key["ESR1"].outcome is Agreement.DISAGREE
    assert by_key["MKI67"].outcome is Agreement.DISAGREE
    assert de_side.items[2].measured.state is MeasuredState.TESTED_ABSENT
    assert de_side.items[2].measured.absence_is_evidence is True

    # (c) NOT a conflict. B could not look, and the join says which side and why.
    cd3d = by_key["CD3D"]
    assert cd3d.outcome is Agreement.NOT_COMPARABLE
    assert cd3d.corroborates is False
    assert cd3d.blocked == ("seurat_de",)
    assert cd3d.right.measured.state is MeasuredState.NEVER_MEASURED
    assert "panel:b" in cd3d.because[0]
    assert cd3d.right.statistic is None


# --------------------------------------------------------------------------- #
# The join key -- the test that proves the Subject abstraction paid for itself
# --------------------------------------------------------------------------- #
def test_reconcile_joins_on_subject_key(marker_side, de_side) -> None:
    """Pairs meet on ``Subject.key`` and on nothing else.

    The two sides key their items through ``subject()``, so a pair is the same
    subject by construction rather than by two readers having spelled a gene the same
    way. The join is also over the UNION, so a subject one side never emitted appears
    as a row rather than vanishing -- the silent under-report ``Subject.key`` exists
    to prevent.
    """
    result = reconcile(marker_side, de_side, Relationship.PARTLY_INDEPENDENT).value
    for p in result.pairs:
        assert p.left.subject.key == p.subject.key == p.right.subject.key
    assert [p.subject.key for p in result.pairs] == [
        i.subject.key for i in marker_side.items
    ]


def test_a_subject_only_one_side_emitted_still_gets_a_row(marker_side) -> None:
    """"Nothing to say" is reported, and it is adjudicated against that side's frames.

    A missing row cannot tell "never in this panel" from "in the panel, tested, not
    reported". Resolving the absent side against its own world can, and it is the
    same move ``core/rival.py`` makes for a canonical marker a unit did not report.
    """
    world = _world("panel:b", PANEL_B)
    narrow = Side(module="seurat_de", workflow="marker-interpretation",
                  task_kind=TaskKind.ASSIGNMENT, unit="c1",
                  authority=Authority.MEASUREMENT,
                  items=_items(world, "log2fc", (("ERBB2", 3.3, Direction.UP),)),
                  world=world)
    result = reconcile(marker_side, narrow, Relationship.PARTLY_INDEPENDENT).value
    by_key = {p.subject.one("g").id: p for p in result.pairs}

    # ESR1 and MKI67 are in B's panel, so B's silence about them is TESTED_ABSENT:
    # a measured absence, which argues.
    assert by_key["ESR1"].right.measured.state is MeasuredState.TESTED_ABSENT
    assert by_key["ESR1"].outcome is Agreement.DISAGREE
    # CD3D is not, so B's silence argues nothing.
    assert by_key["CD3D"].right.measured.state is MeasuredState.NEVER_MEASURED
    assert by_key["CD3D"].outcome is Agreement.NOT_COMPARABLE


def test_a_side_with_no_world_cannot_argue_from_silence(marker_side) -> None:
    """The default ``MeasuredWorld()`` is the conservative one.

    With no frame declared, every subject the side did not emit resolves to
    NEVER_MEASURED -- "this side cannot speak to it" -- rather than to a tested
    absence it never earned. A caller who forgets to pass a world under-reports
    conflict; one who got the opposite default would manufacture it.
    """
    bare = Side(module="ihc", workflow="marker-interpretation",
                task_kind=TaskKind.ASSIGNMENT, unit="c1",
                authority=Authority.MEASUREMENT, items=())
    result = reconcile(marker_side, bare, Relationship.INDEPENDENT).value
    assert {p.outcome for p in result.pairs} == {Agreement.NOT_COMPARABLE}
    assert all(p.blocked == ("ihc",) for p in result.pairs)


def test_an_underpowered_side_is_not_a_disagreement_either(marker_side) -> None:
    """The lattice gate reads ``absence_is_evidence``, not a state list.

    UNDERPOWERED is the state ``IMPLEMENTATION.md`` §2.2 forgot. In frame, below the
    floor: the measurement could not have produced the positive, so the negative is
    not evidence, and the gate must catch it for the same reason it catches
    NEVER_MEASURED.
    """
    world = _world("panel:b", PANEL_B)
    r = Reading(subject("gene", g=gene("ERBB2")), 3.3, "log2fc",
                coverage=(Coverage("replicates", 1, 3, 2),), direction=Direction.UP)
    thin = Side(module="seurat_de", workflow="marker-interpretation",
                task_kind=TaskKind.ASSIGNMENT, unit="c1",
                authority=Authority.MEASUREMENT,
                items=(measure(r, world, frozenset({"ERBB2"})).value,), world=world)
    assert thin.items[0].measured.state is MeasuredState.UNDERPOWERED

    result = reconcile(marker_side, thin, Relationship.PARTLY_INDEPENDENT).value
    erbb2 = next(p for p in result.pairs if p.subject.one("g").id == "ERBB2")
    assert erbb2.outcome is Agreement.NOT_COMPARABLE
    assert erbb2.blocked == ("seurat_de",)


def test_both_tested_absent_is_agreement_not_silence() -> None:
    """Two measurements that both looked and both found nothing AGREE.

    This is the half a boolean cannot express, and it is the commonest kind of
    agreement there is. Filed as "not comparable" it would read to the biologist as a
    refusal, and the join would be unable to say that two panels concur about an
    absence.
    """
    world_a = _world("panel:a", PANEL_A)
    world_b = _world("panel:b", PANEL_B)
    left = Side(module="jazzpanda", workflow="marker-interpretation",
                task_kind=TaskKind.ASSIGNMENT, unit="c1",
                authority=Authority.MEASUREMENT,
                items=_items(world_a, "glm_coef",
                             (("ERBB2", 9.0, Direction.UP), ("MKI67", None, None))),
                world=world_a)
    right = Side(module="seurat_de", workflow="marker-interpretation",
                 task_kind=TaskKind.ASSIGNMENT, unit="c1",
                 authority=Authority.MEASUREMENT,
                 items=_items(world_b, "log2fc",
                              (("ERBB2", 3.3, Direction.UP), ("MKI67", None, None))),
                 world=world_b)
    result = reconcile(left, right, Relationship.PARTLY_INDEPENDENT).value
    mki67 = next(p for p in result.pairs if p.subject.one("g").id == "MKI67")
    assert mki67.outcome is Agreement.AGREE
    assert mki67.corroborates is True


# --------------------------------------------------------------------------- #
# Same question or different question (ARCHITECTURE §9.11)
# --------------------------------------------------------------------------- #
def test_different_question_never_agrees(marker_side, de_side) -> None:
    """An enrichment result does not corroborate a cell-type call.

    It is not answering the identity question, so "agree" would read as two witnesses
    where there is one. Coherent and tension are the only words available.
    """
    enrichment = dataclasses.replace(
        de_side, module="ora", workflow="enrichment-interpretation",
        task_kind=TaskKind.CHARACTERISATION,
    )
    result = reconcile(marker_side, enrichment,
                       Relationship.PARTLY_INDEPENDENT).value
    assert result.same_question is False
    outcomes = {p.outcome for p in result.pairs}
    assert Agreement.AGREE not in outcomes
    assert Agreement.DISAGREE not in outcomes
    assert outcomes <= {Agreement.COHERENT, Agreement.TENSION,
                        Agreement.NOT_COMPARABLE}
    assert result.with_outcome(Agreement.COHERENT)
    assert result.with_outcome(Agreement.TENSION)


def test_same_workflow_different_task_kind_is_a_different_question(
        marker_side, de_side) -> None:
    """Task kind belongs to the method and is half of what makes the question.

    Two modules under one workflow name that ask different task kinds are not asking
    one question, and treating the workflow string alone as the identity would let
    them "agree".
    """
    other = dataclasses.replace(de_side, task_kind=TaskKind.COMPARISON)
    result = reconcile(marker_side, other, Relationship.PARTLY_INDEPENDENT).value
    assert result.same_question is False


# --------------------------------------------------------------------------- #
# Reference data, and dependency structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("authority", [Authority.CURATED, Authority.LITERATURE,
                                       Authority.LAB])
def test_reference_data_never_counts_as_agreement(marker_side, de_side,
                                                  authority) -> None:
    """The rule about reference data, enforced before any number is read.

    A curated edge that happens to say ERBB2 is up in this unit is not a second
    measurement of ERBB2, and the join must not let it become one. The gate runs
    first, so the outcome does not depend on whether the numbers would have matched.
    """
    curated = dataclasses.replace(de_side, module="panglaodb", authority=authority)
    result = reconcile(marker_side, curated, Relationship.INDEPENDENT)
    assert {p.outcome for p in result.value.pairs} == {Agreement.NOT_COMPARABLE}
    assert result.value.corroborated == ()
    assert all("panglaodb" in p.blocked for p in result.value.pairs)
    assert result.decisions[0].rule == "reconcile:reference_cannot_corroborate"


def test_authority_has_no_default() -> None:
    """Structural, not conventional: a caller cannot forget to say where a side came
    from, because ``Side`` will not construct without it."""
    fields = {f.name: f for f in dataclasses.fields(Side)}
    assert fields["authority"].default is dataclasses.MISSING
    assert fields["authority"].default_factory is dataclasses.MISSING


def test_derived_is_not_corroboration(marker_side, de_side) -> None:
    """Enrichment computed from the marker result agreeing with its parent is the
    same result twice. The outcome stands -- it is true that they agree -- and the
    witness claim is what is withheld."""
    result = reconcile(marker_side, de_side, Relationship.DERIVED)
    assert any(p.outcome is Agreement.AGREE for p in result.value.pairs)
    assert result.value.corroborated == ()
    assert result.decisions[0].rule == "reconcile:derived_is_not_corroboration"


def test_corroboration_requires_an_independent_measurement(marker_side,
                                                           de_side) -> None:
    """The positive case, so the two tests above are not passing vacuously."""
    result = reconcile(marker_side, de_side, Relationship.PARTLY_INDEPENDENT).value
    assert result.corroborated
    assert all(p.outcome is Agreement.AGREE for p in result.corroborated)


# --------------------------------------------------------------------------- #
# No band, anywhere in this layer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [ReconcileResult, Reconciliation, Concordance,
                                 Adjacency])
def test_no_reconciliation_type_declares_a_confidence_field(cls) -> None:
    """A field guard, not a behavioural one.

    The band is a function of ONE method's output; every other layer can only
    subtract or flag. A behavioural test only catches a confidence value that is
    filled in; this fails the moment the field is declared, which is when the mistake
    costs nothing to undo. The first question anyone would ask of a reconciled band
    is whether agreement raises it, and §9.10 already says it must not.
    """
    banned = {"band", "confidence", "score", "weight", "verify", "strength"}
    names = {f.name for f in dataclasses.fields(cls)}
    assert not (names & banned), f"{cls.__name__} declares {sorted(names & banned)}"


def test_adjacency_is_not_a_band_input() -> None:
    """§9.9: display flag, until its firing rate is measured. Source scan, because a
    behavioural test cannot see an import that has not been used yet."""
    tree = ast.parse((CORE / "verdict.py").read_text())
    imported = [
        n.module for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
    ]
    assert not any("adjacency" in m or "concordance" in m or "reconcile" in m
                   for m in imported), imported


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_reconciling_across_units_raises(marker_side, de_side) -> None:
    """A join across units is well-formed and meaningless, and nothing downstream
    could tell -- every pair would look fine."""
    with pytest.raises(DifferentUnits):
        reconcile(marker_side, dataclasses.replace(de_side, unit="c7"),
                  Relationship.PARTLY_INDEPENDENT)


def test_every_decision_is_registered_and_in_code_order(marker_side,
                                                        de_side) -> None:
    """Decision order is code order and is hashed. The gates precede the pairs, and
    the pairs follow the join order."""
    out = reconcile(marker_side, de_side, Relationship.PARTLY_INDEPENDENT)
    for d in out.decisions:
        assert d.rule in RULES
    assert [d.rule for d in out.steps("reconcile")] == [
        "reconcile:agree",            # ERBB2
        "reconcile:disagree",         # ESR1, opposite directions
        "reconcile:disagree",         # MKI67, measured absence
        "reconcile:not_comparable",   # CD3D, never in B's panel
    ]


def test_the_same_input_produces_byte_identical_decisions(marker_side,
                                                          de_side) -> None:
    """``because`` strings are hashed into the golden record, and this module builds
    them from frozensets and dict lookups -- the two places iteration order leaks."""
    a = reconcile(marker_side, de_side, Relationship.PARTLY_INDEPENDENT)
    b = reconcile(marker_side, de_side, Relationship.PARTLY_INDEPENDENT)
    assert a.decisions == b.decisions
    assert a.value == b.value


def test_no_decision_ref_carries_a_number(marker_side, de_side) -> None:
    """A ref reading ``#glm_coef=...`` asserts a measurement. The join reads subjects,
    and most of its rows have no value to read."""
    out = reconcile(marker_side, de_side, Relationship.PARTLY_INDEPENDENT)
    refs = [r for d in out.decisions for r in d.refs]
    assert refs and all("#" not in r for r in refs), refs


# --------------------------------------------------------------------------- #
# adjacency (§9.9)
# --------------------------------------------------------------------------- #
def test_adjacency_reads_bleed_and_specificity_apart() -> None:
    """Same statistic, different readings, and only spatial data separates them.

    They are different facts with different fixes -- segmentation versus the marker
    -- so a single "not unique" flag would tell the biologist to do nothing in
    particular.
    """
    world = _world("panel:a", PANEL_A)
    item = _items(world, "glm_coef", (("ERBB2", 21.4, Direction.UP),),
                  shared_with=("c2", "c9"))[0]
    out = adjacency_check(item, "c1", frozenset({"c2"}))
    assert out.value.bleed_candidates == ("c2",)
    assert out.value.specificity_problems == ("c9",)
    assert [d.rule for d in out.decisions] == [
        "adjacency:bleed_candidate", "adjacency:specificity",
    ]
    assert [s.reads_as for s in out.value.shared] == [
        AdjacencyRead.BLEED_CANDIDATE, AdjacencyRead.SPECIFICITY,
    ]


def test_adjacency_without_a_neighbour_set_reads_conservatively() -> None:
    """An empty set is the conservative reading: the marker does mark something else
    either way, and only the *excuse* for it needs evidence this module lacks."""
    world = _world("panel:a", PANEL_A)
    item = _items(world, "glm_coef", (("ERBB2", 21.4, Direction.UP),),
                  shared_with=("c2",))[0]
    out = adjacency_check(item, "c1", frozenset())
    assert out.value.specificity_problems == ("c2",)
    assert out.value.bleed_candidates == ()


def test_an_unmeasured_item_shares_nothing() -> None:
    """A subject the frame refused has no localisation to share. A flag here would
    read as a measurement of something that was never measured."""
    world = _world("panel:b", PANEL_B)
    item = _items(world, "log2fc", (("CD3D", None, None),), shared_with=("c2",))[0]
    assert item.measured.state is MeasuredState.NEVER_MEASURED
    out = adjacency_check(item, "c1", frozenset({"c2"}))
    assert out.value.shared == ()
    assert out.decisions == ()


# --------------------------------------------------------------------------- #
# concordance (§9.10)
# --------------------------------------------------------------------------- #
def _replicate(sample: str, rows) -> Replicate:
    return Replicate(sample, _items(_world("panel:a", PANEL_A), "glm_coef", rows))


def test_concordance_requires_joint_clustering() -> None:
    """Cluster 3 in two independently clustered samples is two units sharing a label.
    There is no honest output, so it raises rather than returning one."""
    reps = (_replicate("s1", (("ERBB2", 9.0, Direction.UP),)),
            _replicate("s2", (("ERBB2", 8.0, Direction.UP),)))
    with pytest.raises(IndependentClustering):
        concordance("c1", reps, joint=False)
    assert concordance("c1", reps, joint=True).value.divergent is False


def test_concordance_needs_two_replicates() -> None:
    with pytest.raises(TooFewReplicates):
        concordance("c1", (_replicate("s1", (("ERBB2", 9.0, Direction.UP),)),),
                    joint=True)


def test_concordance_flags_divergence_without_claiming_a_cause() -> None:
    """Batch, section quality or real biology all look like this, and the check does
    not say which. It names the subject and the samples, and stops."""
    reps = (_replicate("s1", (("ERBB2", 9.0, Direction.UP),
                              ("ESR1", 5.0, Direction.UP))),
            _replicate("s2", (("ERBB2", 8.0, Direction.UP),
                              ("ESR1", -5.0, Direction.DOWN))))
    out = concordance("c1", reps, joint=True)
    assert out.value.divergent is True
    assert [s.one("g").id for s in out.value.diverging_subjects] == ["ESR1"]
    assert [d.rule for d in out.decisions] == [
        "concordance:consistent", "concordance:divergent",
    ]


def test_concordance_agreement_produces_no_number() -> None:
    """Agreement never raises confidence (§9.10). ``Concordance`` has nowhere to put
    one: the only thing agreement produces is the absence of a flag."""
    reps = (_replicate("s1", (("ERBB2", 9.0, Direction.UP),)),
            _replicate("s2", (("ERBB2", 8.0, Direction.UP),)))
    out = concordance("c1", reps, joint=True).value
    assert out.divergent is False
    assert out.coverage.k == 1 and out.coverage.n == 1 and out.coverage.ok


def test_a_sample_that_never_measured_it_is_not_divergence() -> None:
    """The lattice gate holds across samples too, because it is the same comparison.

    A subject outside one sample's frame cannot diverge from the other -- and the
    coverage says how much of the profile was comparable at all, so "one of three
    subjects could be checked" cannot read as "the replicates agree".
    """
    narrow = Replicate("s2", _items(_world("panel:b", PANEL_B), "glm_coef",
                                    (("ERBB2", 8.0, Direction.UP),)))
    wide = _replicate("s1", (("ERBB2", 9.0, Direction.UP),
                             ("CD3D", 3.0, Direction.UP)))
    out = concordance("c1", (wide, narrow), joint=True).value
    cd3d = next(s for s in out.per_subject if s.subject.one("g").id == "CD3D")
    assert cd3d.outcome is Comparison.NOT_COMPARABLE
    assert cd3d.blocked == ("s2",)
    assert out.divergent is False
    assert (out.coverage.k, out.coverage.n) == (1, 2)
