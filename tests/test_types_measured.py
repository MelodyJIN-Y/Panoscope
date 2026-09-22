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
    assert len(made) == len(SHAPES)


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
def test_the_four_states_are_distinguishable():
    obs = resolve(subject("gene", g=gene("ERBB2")), WORLD, REPORTED)
    absent = resolve(subject("gene", g=gene("ACTA2")), WORLD, REPORTED)
    never = resolve(subject("gene", g=gene("COL1A1")), WORLD, REPORTED)
    assert obs.state is MeasuredState.OBSERVED
    assert absent.state is MeasuredState.TESTED_ABSENT
    assert never.state is MeasuredState.NEVER_MEASURED


def test_measured_absence_IS_evidence():
    """The productive half, and the reason a boolean will not do.

    The discriminator argues against a rival precisely because the rival's markers
    were measured and peaked elsewhere. Collapse the lattice and the panel-absence
    rule stops discriminating and becomes paralysis.
    """
    m = resolve(subject("gene", g=gene("ACTA2")), WORLD, REPORTED)
    assert m.absence_is_evidence is True


def test_unmeasured_absence_is_NOT_evidence():
    """The project's headline claim, as a type-level property."""
    m = resolve(subject("gene", g=gene("COL1A1")), WORLD, REPORTED)
    assert m.absence_is_evidence is False
    assert m.testable is False


def test_only_observed_carries_a_number():
    """Quoting a statistic for an unmeasured thing must be unrepresentable."""
    for sym, expect in (("ERBB2", True), ("ACTA2", False), ("COL1A1", False)):
        assert resolve(subject("gene", g=gene(sym)), WORLD, REPORTED).carries_number is expect


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


def test_missing_cell_type_blocks_a_pair():
    """The composition analogue of panel absence, for an analysis never built."""
    pair = subject("cell_type_pair", a=cell_type("Tumor"), b=cell_type("Podocyte"))
    m = resolve(pair, WORLD, REPORTED)
    assert m.state is MeasuredState.NEVER_MEASURED
    assert [e.id for e in m.blocked_by] == ["Podocyte"]


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


def test_undeclared_frame_reads_as_never_measured():
    """Silence about capability is not permission to assert it.

    Mirrors the reference-data rule: a missing node reads as 'not in reference data',
    never as 'does not exist'.
    """
    empty = MeasuredWorld((PANEL,))
    m = resolve(subject("cell_type", ct=cell_type("Tumor")), empty, REPORTED)
    assert m.state is MeasuredState.NEVER_MEASURED


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
