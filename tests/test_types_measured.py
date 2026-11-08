"""Invariants for the subject/measured contracts.

The spec is explicit that these tests, not the dataclasses, are the deliverable:
"anything called an invariant has a test that fails loudly". Each test below pins a
property that something downstream is entitled to assume, and several pin a bug that
was real in the pre-refactor code.
"""

from __future__ import annotations

import pytest

from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld, resolve,
)
from panoscope.types.subject import (
    SHAPES, Entity, EntityKind, Gate, Subject, UnknownShape, cell_type, gene,
    gene_set, subject, unit,
)

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({"ERBB2", "LUM", "POSTN", "ACTA2"}))
ANNOT = Frame(EntityKind.CELL_TYPE, "annotation:test", frozenset({"Tumor", "Stromal"}))
COLL = Frame(EntityKind.GENE_SET, "collection:test", frozenset({"HALLMARK_EMT"}))
WORLD = MeasuredWorld((PANEL, ANNOT, COLL))
REPORTED = frozenset({"ERBB2", "LUM", "HALLMARK_EMT"})


# --------------------------------------------------------------------------- #
# Subject: the join key
# --------------------------------------------------------------------------- #
def test_every_registered_shape_is_constructible():
    """A shape nobody can build is dead vocabulary. Catches a malformed registry."""
    made = []
    for name, shape in SHAPES.items():
        kw = {}
        for role in shape.roles:
            e = Entity(role.kind, f"X{len(kw)}")
            kw[role.name] = [e] if role.many else e
        made.append(subject(name, **kw))
    assert [s.shape for s in made] == list(SHAPES)


@pytest.mark.req("IMPL-2.1-01")
def test_symmetric_roles_are_canonical():
    """An unordered pair must be ONE subject however the reader emitted it.

    Without this, co-localisation (A,B) and (B,A) are two rows that never join, and
    reconciliation silently under-reports.
    """
    a = subject("cell_type_pair", a=cell_type("T_Cells"), b=cell_type("Mast"))
    b = subject("cell_type_pair", a=cell_type("Mast"), b=cell_type("T_Cells"))
    assert a == b and a.key == b.key


def test_ordered_roles_are_not_canonicalised():
    """The converse, and the reason symmetry is declared per shape rather than assumed.

    Neighbourhood is directional: 'T cells next to macrophages' is a different claim
    from its reverse.
    """
    a = subject("adjacency", centre=cell_type("T_Cells"), neighbour=cell_type("Mast"))
    b = subject("adjacency", centre=cell_type("Mast"), neighbour=cell_type("T_Cells"))
    assert a != b and a.key != b.key


@pytest.mark.req("IMPL-2.1-02")
def test_subject_key_is_injective_over_a_shape():
    keys = {
        subject("gene", g=gene(s)).key
        for s in ("ERBB2", "LUM", "POSTN", "ACTA2", "COL1A1")
    }
    assert len(keys) == 5


def test_label_is_never_load_bearing():
    """Two spellings of one thing must not become two subjects."""
    a = subject("gene", g=Entity(EntityKind.GENE, "ERBB2", "HER2"))
    b = subject("gene", g=Entity(EntityKind.GENE, "ERBB2", ""))
    assert a.key == b.key


def test_member_order_does_not_change_identity():
    a = subject("gene_set", set=gene_set("S"), member=[gene("B"), gene("A")])
    b = subject("gene_set", set=gene_set("S"), member=[gene("A"), gene("B")])
    assert a == b


def test_member_set_does_not_change_the_join_key():
    """Identity must not depend on the apparatus.

    The same gene set read off a 280-gene panel and a 5,000-gene panel has different
    measured members. If those produced different keys, reconciliation -- which is a
    join across task kinds on exactly this key -- would silently under-report, and two
    methods' readings of HALLMARK_EMT would never meet.

    Members are coverage, not identity. This is the same rule ``resolve()`` already
    follows by excluding MEMBER roles from the meet.
    """
    wide = subject("gene_set", set=gene_set("HALLMARK_EMT"),
                   member=[gene("A"), gene("B"), gene("C")])
    narrow = subject("gene_set", set=gene_set("HALLMARK_EMT"), member=[gene("A")])
    none_measured = subject("gene_set", set=gene_set("HALLMARK_EMT"))
    assert wide.key == narrow.key == none_measured.key
    # ...but a DIFFERENT set is still a different subject
    assert subject("gene_set", set=gene_set("HALLMARK_G2M")).key != wide.key


def test_required_roles_still_determine_identity():
    """The converse: excluding members must not collapse distinct subjects."""
    a = subject("interaction", ligand=gene("LUM"), receptor=gene("CXCR4"),
                source=cell_type("Stromal"), target=cell_type("Tumor"))
    b = subject("interaction", ligand=gene("LUM"), receptor=gene("CXCR4"),
                source=cell_type("Tumor"), target=cell_type("Stromal"))
    assert a.key != b.key


def test_unregistered_shape_raises():
    """Fail loudly; never guess. A silently malformed subject corrupts the join key."""
    with pytest.raises(UnknownShape):
        subject("neighbourhood_v2", centre=cell_type("T"))


def test_wrong_entity_kind_raises():
    with pytest.raises(ValueError, match="expects gene"):
        subject("gene", g=cell_type("Tumor"))


def test_missing_required_role_raises():
    with pytest.raises(ValueError, match="requires role"):
        subject("adjacency", centre=cell_type("T_Cells"))


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="no role"):
        subject("gene", g=gene("ERBB2"), bogus=gene("X"))


# --------------------------------------------------------------------------- #
# Measured: the panel-absence rule generalised
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-MS-01")
def test_the_four_states_are_distinguishable():
    obs = resolve(subject("gene", g=gene("ERBB2")), WORLD, REPORTED)
    absent = resolve(subject("gene", g=gene("ACTA2")), WORLD, REPORTED)
    never = resolve(subject("gene", g=gene("COL1A1")), WORLD, REPORTED)
    assert obs.state is MeasuredState.OBSERVED
    assert absent.state is MeasuredState.TESTED_ABSENT
    assert never.state is MeasuredState.NEVER_MEASURED


@pytest.mark.req("CLAUDE-MS-02")
def test_measured_absence_IS_evidence():
    """The productive half, and the reason a boolean will not do.

    The discriminator argues against a rival precisely because the rival's markers
    were measured and peaked elsewhere. Collapse the lattice and the panel-absence
    rule stops discriminating and becomes paralysis.
    """
    m = resolve(subject("gene", g=gene("ACTA2")), WORLD, REPORTED)
    assert m.absence_is_evidence is True


@pytest.mark.req("CLAUDE-MS-02")
def test_unmeasured_absence_is_NOT_evidence():
    """The project's headline claim, as a type-level property."""
    m = resolve(subject("gene", g=gene("COL1A1")), WORLD, REPORTED)
    assert m.absence_is_evidence is False
    assert m.testable is False


def test_only_observed_carries_a_number():
    """Quoting a statistic for an unmeasured thing must be unrepresentable."""
    for sym, expect in (("ERBB2", True), ("ACTA2", False), ("COL1A1", False)):
        assert resolve(subject("gene", g=gene(sym)), WORLD, REPORTED).carries_number is expect


@pytest.mark.req("ARCH-9.5-01")
def test_composite_state_is_the_meet_and_names_the_blocker():
    """A ligand-receptor pair with one off-panel partner is blocked, and says which.

    Bare names would not do: the UI has to tell the biologist WHICH partner was never
    measured, or the caveat is unactionable.
    """
    lr = subject("interaction", ligand=gene("LUM"), receptor=gene("CXCR4"),
                 source=cell_type("Stromal"), target=cell_type("Tumor"))
    m = resolve(lr, WORLD, REPORTED)
    assert m.state is MeasuredState.NEVER_MEASURED
    assert [e.id for e in m.blocked_by] == ["CXCR4"]


@pytest.mark.req("ARCH-9.5-01")
def test_missing_cell_type_blocks_a_pair():
    """The composition analogue of panel absence, for an analysis never built."""
    pair = subject("cell_type_pair", a=cell_type("Tumor"), b=cell_type("Podocyte"))
    m = resolve(pair, WORLD, REPORTED)
    assert m.state is MeasuredState.NEVER_MEASURED
    assert [e.id for e in m.blocked_by] == ["Podocyte"]


@pytest.mark.req("ARCH-9.5-02")
def test_coverage_demotes_but_never_promotes():
    gs = subject("gene_set", set=gene_set("HALLMARK_EMT"), member=[gene("ERBB2")])
    ok = resolve(gs, WORLD, REPORTED, coverage=(Coverage("panel_genes", 18, 200, 3),))
    bad = resolve(gs, WORLD, REPORTED, coverage=(Coverage("panel_genes", 2, 200, 3),))
    assert ok.state is MeasuredState.OBSERVED
    assert bad.state is MeasuredState.UNDERPOWERED
    # and it cannot lift an unmeasurable subject
    never = subject("gene_set", set=gene_set("NOT_A_SET"), member=[gene("ERBB2")])
    lifted = resolve(never, WORLD, REPORTED, coverage=(Coverage("panel_genes", 99, 99, 3),))
    assert lifted.state is MeasuredState.NEVER_MEASURED


def test_member_roles_do_not_block_the_subject():
    """A gene set with an off-panel member is still assessable.

    Members are coverage, not gates. Treating them as gates would make almost every
    gene set unmeasurable on a targeted panel, which is the opposite of the point.
    """
    gs = subject("gene_set", set=gene_set("HALLMARK_EMT"),
                 member=[gene("ERBB2"), gene("COL1A1")])
    assert resolve(gs, WORLD, REPORTED).state is MeasuredState.OBSERVED


@pytest.mark.req("ARCH-9.5-03")
def test_undeclared_frame_reads_as_never_measured():
    """Silence about capability is not permission to assert it.

    Mirrors the reference-data rule: a missing node reads as 'not in reference data',
    never as 'does not exist'.
    """
    empty = MeasuredWorld((PANEL,))
    m = resolve(subject("cell_type", ct=cell_type("Tumor")), empty, REPORTED)
    assert m.state is MeasuredState.NEVER_MEASURED


@pytest.mark.req("IMPL-2.2-01")
def test_frame_digest_is_derived_and_order_independent():
    a = Frame(EntityKind.GENE, "p", frozenset({"A", "B", "C"}))
    b = Frame(EntityKind.GENE, "p", frozenset({"C", "A", "B"}))
    assert a.digest == b.digest
    assert Frame(EntityKind.GENE, "p", frozenset({"A", "B"})).digest != a.digest


def test_frameref_carries_no_membership():
    """Claims are git-tracked and re-read per retrieval; inlining a panel is ~8 KB each."""
    ref = PANEL.ref()
    assert not any(
        isinstance(v, (set, frozenset, list, tuple))
        for v in (ref.kind, ref.name, ref.digest, ref.size)
    )
    assert ref.size == len(PANEL.members)


# --------------------------------------------------------------------------- #
# Contract hygiene
# --------------------------------------------------------------------------- #
def test_contracts_are_frozen_and_hashable():
    """Frozen is what makes these safe as cache keys and dict members."""
    for obj in (
        gene("ERBB2"),
        subject("gene", g=gene("ERBB2")),
        PANEL.ref(),
        Coverage("x", 1, 2, 3),
        resolve(subject("gene", g=gene("ERBB2")), WORLD, REPORTED),
    ):
        hash(obj)
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(obj, "__nonexistent__", 1)
            setattr(obj, next(iter(type(obj).__slots__)), "mutated")


def test_no_mutable_fields_in_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes. The record
    determinism invariant cannot hold if any contract carries one."""
    import dataclasses

    from panoscope.types import measured as M, subject as S

    bad = []
    for mod in (S, M):
        for name in dir(mod):
            obj = getattr(mod, name)
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            for f in dataclasses.fields(obj):
                t = str(f.type)
                if any(x in t for x in ("dict[", "list[", "set[")) and "frozenset" not in t:
                    bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable/unordered fields in frozen contracts: " + "; ".join(bad)


# --------------------------------------------------------------------------- #
# Identity is case- and whitespace-insensitive on BOTH gates, not just the frame
# --------------------------------------------------------------------------- #
def test_a_reported_gene_stays_observed_whatever_case_the_caller_spells_it() -> None:
    """The second half of a fix whose first half shipped alone, and the worse half.

    ``Frame.contains`` normalises, so the panel ADMITS a case variant. The reported
    set was compared exactly, so the same variant then fell through to
    TESTED_ABSENT -- whose contract is that absence IS evidence. A gene the method
    reported was being turned into a measured absence by a capitalisation.

    It is reachable wherever the entity and the reported set come from different
    files, which is every place a curated list meets a method's table:
    ``core/rival.py`` resolving a canonical marker and ``core/reconcile.py``
    resolving one method's subject against another's. The rival case is the sharp
    one -- see the test below, where it manufactures evidence against a rival.
    """
    panel = Frame(EntityKind.GENE, "panel:p", frozenset({"SYNG01"}))
    world = MeasuredWorld((panel,))
    reported = frozenset({"SYNG01"})          # the method's own spelling

    for spelling in ("SYNG01", "Syng01", "syng01", " SYNG01 "):
        m = resolve(subject("gene", g=gene(spelling)), world, reported, ())
        assert m.state is MeasuredState.OBSERVED, (
            f"{spelling!r}: the panel admitted it but the reported set did not, so "
            f"it resolved {m.state.value} -- a measured absence invented by case"
        )
        assert m.absence_is_evidence is False


def test_a_case_variant_cannot_manufacture_evidence_against_a_rival() -> None:
    """The consequence, at the layer a biologist reads.

    ``rival_test`` separates a call from an alternative on the rival's canonical
    markers that were MEASURED AND ABSENT. With the reported set compared exactly,
    a curated list spelling CD3E as ``Cd3e`` made an observed CD3E separate the
    call from T_Cells -- the productive half of the lattice, pointed the wrong way.
    """
    from panoscope.core.measure import Reading, measure
    from panoscope.core.rival import rival_test
    from panoscope.types.conditions import Condition, Conditions, Provenance
    from panoscope.types.evidence import NormalisedEvidence
    from panoscope.types.subject import unit as unit_entity

    panel = Frame(EntityKind.GENE, "panel:p", frozenset({"CD3E", "ERBB2"}))
    world = MeasuredWorld((panel,))
    seen = frozenset({"CD3E", "ERBB2"})
    items = tuple(
        measure(Reading(subject=subject("gene", g=gene(g)), statistic=s,
                        statistic_name="glm_coef"), world, seen).value
        for g, s in (("CD3E", 4.0), ("ERBB2", 5.0))
    )
    ev = NormalisedEvidence(
        unit="c1", unit_kind="cluster", unit_subject=subject("unit", u=unit_entity("c1")),
        unit_measured=items[0].measured, module="m", items=items,
        reported=(Frame(EntityKind.GENE, "reported:m:c1", seen),),
        conditions=Conditions(*[Condition("x", Provenance.STATED)] * 6,
                              frames=(panel.ref(),)),
    )
    for spelling in ("CD3E", "Cd3e"):
        r = rival_test(ev, call="Tumor", alternative="T_Cells",
                       canonical_call=("ERBB2",), canonical_rival=(spelling,),
                       world=world, floor=1).value
        assert not r.against_rival, (
            f"curated list spelling {spelling!r} separated the call from T_Cells on "
            f"{[i.subject.display() for i in r.against_rival]}, which was OBSERVED "
            "here -- evidence against a rival produced by capitalisation"
        )
