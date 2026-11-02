"""Invariants for the four registry contracts and the record.

Two properties, and the first one is the reason the file exists: **a Protocol nobody
can implement is dead vocabulary.** Every stream in IMPLEMENTATION.md §11 declares
"contracts" as its dependency, so these are written against minimal conforming stubs
before any real adapter, method, source or workflow exists -- the same precedent as
``test_every_registered_shape_is_constructible``, which catches a malformed registry
by building one of everything in it.

Each contract also gets a *non*-conforming stub that drops exactly one declared
member, so the tests prove the check bites rather than that ``isinstance`` returns
True for anything with a pulse. The member dropped is in each case the one the
design is actually about: ``does_not_measure`` for a method, ``clustering_mode`` for
an adapter, ``uispec`` for a workflow, ``conditions`` for a source.

The second property is byte-determinism on the record, which is the reproducibility
claim. ``drop_reasons`` is the field the v2 spec got wrong -- a dict, whose insertion
order would have flowed into the serialised bytes -- so it is pinned in the strong
form: two records built from the same pairs in different orders must serialise
identically.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

from panoscope.types import contracts as C
from panoscope.types import record as R
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import (
    Check, MethodModule, PlatformAdapter, ReferenceSource, RivalResult, UnitSpec,
    Workflow,
)
from panoscope.types.evidence import NormalisedEvidence
from panoscope.types.measured import Frame, MeasuredWorld, resolve
from panoscope.types.outputs import Assignment, Finding, TaskKind
from panoscope.types.record import GENERATED_FIELDS, InterpretationRecord, ReadingRecord
from panoscope.types.subject import EntityKind, Subject, gene, subject
from panoscope.types.trace import Decision, Trace, Traced

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({"ERBB2", "LUM"}))
WORLD = MeasuredWorld((PANEL,))
ERBB2 = subject("gene", g=gene("ERBB2"))
CONDITIONS = Conditions(
    assay=Condition("xenium", Provenance.READ),
    modality=Condition("spatial_transcriptomics", Provenance.READ),
    tissue=Condition("UBERON:0000310", Provenance.STATED),
    method=Condition("jazzpanda", Provenance.STATED),
    segmentation=UNKNOWN, fixation=UNKNOWN,
    frames=(PANEL.ref(),),
)


# --------------------------------------------------------------------------- #
# Minimal conforming stubs. Deliberately the smallest thing that satisfies each
# contract: if one of these needed a helper to write, the contract is too big.
# --------------------------------------------------------------------------- #
class StubAdapter:
    name = "stub"

    def frames(self, store: Any) -> tuple[Frame, ...]:
        return (PANEL,)

    def unit_labels(self, store: Any) -> dict[str, str]:
        return {"c1": "Tumor"}

    def samples(self, store: Any) -> list[str]:
        return ["sample1"]

    def clustering_mode(self, store: Any) -> Literal["joint", "independent", "integrated"]:
        return "independent"

    def available_views(self, store: Any) -> frozenset[str]:
        return frozenset({"cell_map"})

    def coordinate_system(self, store: Any) -> tuple[str, float]:
        return ("micron", 1.0)

    def standing_conditions(self, store: Any) -> Conditions:
        return CONDITIONS

    def validate(self, store: Any) -> list[str]:
        return []


class StubMethod:
    name = "stub_method"
    workflow = "marker-interpretation"
    statistic_name = "glm_coef"
    measures = "spatial co-localisation of transcripts with a cluster"
    does_not_measure = "mean expression level, or any difference between conditions"
    assumptions = ("transcripts are assigned to the correct cell",)
    rival_space: Literal["reference", "data", "none"] = "reference"
    citation = "doi:10.0000/stub"

    def detect_unit(self, table: Any) -> UnitSpec:
        return UnitSpec("cluster", "cluster_id", ("c1",), ("named column",))

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> dict[str, NormalisedEvidence]:
        return {}

    def absence_reads_as(self, subject: Subject, measured: Any) -> str:
        return "on the panel, no spatial enrichment in this cluster"

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        t = Trace()
        t.note("band:from_coef", "High")
        return t.done("High")

    def traversals(self) -> tuple[tuple[str, str], ...]:
        return (("marks", "forward"),)


class StubWorkflow:
    name = "stub-interpretation"
    task_kind = TaskKind.CHARACTERISATION
    question = "what is running here?"
    skill_path = "skills/stub-interpretation/SKILL.md"

    def interpret(self, ev: NormalisedEvidence,
                  reference: Any) -> Traced[Assignment | Finding]:
        return Traced(value=Finding("c1", "cluster", self.task_kind, (), ""))

    def rival_test(self, ev: NormalisedEvidence, alternative: str,
                   reference: Any) -> Traced[RivalResult]:
        return Traced(value=RivalResult(alternative))

    def checks(self) -> tuple[Check, ...]:
        return (Check("panel_scoping", "how much of the process was on the panel?"),)

    def uispec(self) -> Any:     # UISpec lands in types/uispec.py, written in parallel
        return None


class StubSource:
    name = "stub_source"
    release = "2026-01"
    licence = "CC-BY-4.0"
    redistributable = True

    def conditions(self) -> Conditions:
        return CONDITIONS

    def nodes(self) -> Iterator[Any]:
        return iter(())

    def edges(self) -> Iterator[Any]:
        return iter(())


_CONFORMING = [
    (PlatformAdapter, StubAdapter),
    (MethodModule, StubMethod),
    (Workflow, StubWorkflow),
    (ReferenceSource, StubSource),
]


def _without(cls: type, member: str) -> type:
    """A stub identical to ``cls`` but missing exactly one declared member."""
    ns = {k: v for k, v in vars(cls).items() if k != member}
    return type(f"No_{member}_{cls.__name__}", (), ns)


def _name(x) -> str:
    return x if isinstance(x, str) else getattr(x, "__name__", "")


# --------------------------------------------------------------------------- #
# A Protocol nobody can implement is dead vocabulary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("proto,stub", _CONFORMING, ids=_name)
def test_every_contract_has_a_conforming_implementation(proto, stub):
    assert isinstance(stub(), proto), f"{stub.__name__} does not satisfy {proto.__name__}"


@pytest.mark.parametrize(
    "proto,stub,dropped",
    [
        # Each dropped member is the one that contract exists for.
        (MethodModule, StubMethod, "does_not_measure"),
        (MethodModule, StubMethod, "assumptions"),
        (PlatformAdapter, StubAdapter, "clustering_mode"),
        (PlatformAdapter, StubAdapter, "samples"),
        (Workflow, StubWorkflow, "uispec"),
        (Workflow, StubWorkflow, "task_kind"),
        (ReferenceSource, StubSource, "conditions"),
    ],
    ids=_name,
)
def test_dropping_a_declared_member_fails_the_contract(proto, stub, dropped):
    """The check has to bite, or the stubs above prove nothing.

    ``does_not_measure`` and ``assumptions`` are the point of ``MethodModule``: a
    method that does not declare its limits is the input that produces the failure
    this design exists to prevent, so it must be unregisterable rather than merely
    discouraged.
    """
    assert not isinstance(_without(stub, dropped)(), proto), (
        f"{proto.__name__} accepts an implementation with no {dropped!r}"
    )


def test_the_four_contracts_are_runtime_checkable():
    """Declared, so a registry can reject a malformed entry at registration time
    rather than at the first call into it."""
    for proto, _stub in _CONFORMING:
        assert getattr(proto, "_is_runtime_protocol", False), (
            f"{proto.__name__} is not @runtime_checkable"
        )


def test_method_declares_both_limit_fields():
    """The §3 invariant, in its field-presence half. ``test_every_method_declares_limits``
    will make them non-empty over the real registry; this one fails earlier, at the
    point where leaving them out costs nothing to fix."""
    for field in ("does_not_measure", "assumptions"):
        assert field in MethodModule.__annotations__


# --------------------------------------------------------------------------- #
# Compatibility with tests/test_contracts.py: Traced out, no Trace in
# --------------------------------------------------------------------------- #
_TRACE = re.compile(r"\bTrace\b")        # \b so it does not also match "Traced"


def _protocol_methods():
    for proto, _stub in _CONFORMING:
        for name, fn in vars(proto).items():
            if name.startswith("_") or not inspect.isfunction(fn):
                continue
            yield proto, name, fn


def test_no_protocol_method_accepts_a_trace():
    """Decisions flow out, never in. Enforced here *before* an implementation exists,
    because the seam is cheap to close in a declaration and expensive once four
    streams have written call sites through it.

    Annotations are read as strings (``from __future__ import annotations``) rather
    than resolved: ``UISpec``, ``Node`` and ``Edge`` are deliberate forward
    references to modules other streams own, so ``get_type_hints`` would raise
    NameError here for a reason that has nothing to do with the rule.
    """
    bad = [
        f"{proto.__name__}.{name}({param}: {t})"
        for proto, name, fn in _protocol_methods()
        for param, t in fn.__annotations__.items()
        if param != "return" and _TRACE.search(str(t))
    ]
    assert not bad, "contract methods accepting a Trace: " + "; ".join(bad)


def test_the_protocol_method_scan_is_wired():
    """Guard against silent vacuity: the test above passes trivially if the walk
    finds nothing, which is what a renamed Protocol would look like."""
    found = list(_protocol_methods())
    assert len(found) >= 20, f"the Protocol method scan found only {len(found)}"


def test_the_trace_detector_does_not_match_traced():
    """Pins the hole above: a plain substring check would flag every reasoning
    method, so the test would 'pass' by being impossible to satisfy and get deleted."""
    assert _TRACE.search("Trace | None")
    assert _TRACE.search("dict[str, Trace]")
    assert not _TRACE.search("Traced[str]")
    assert not _TRACE.search("Traced[Assignment | Finding]")


@pytest.mark.parametrize(
    "proto,name",
    [(MethodModule, "rubric"), (Workflow, "interpret"), (Workflow, "rival_test")],
    ids=_name,
)
def test_every_reasoning_contract_method_returns_traced(proto, name):
    """A contract method that reasons and returns a bare value has dropped its
    reasoning on the floor, and the record cannot be reconstructed afterwards --
    the rule ``test_every_core_function_returns_traced`` applies to ``core/``,
    declared here so the implementations land conforming instead of retrofitted."""
    ret = str(vars(proto)[name].__annotations__["return"])
    assert ret.startswith("Traced["), f"{proto.__name__}.{name} -> {ret}"


# --------------------------------------------------------------------------- #
# The record: sorted pairs, and the generated layer marked as such
# --------------------------------------------------------------------------- #
def _reading(drop_reasons=()) -> ReadingRecord:
    return ReadingRecord(
        path="results/markers.csv", sha256="0" * 64,
        statistic_column="glm_coef", unit_column="cluster_id",
        unit_kind="cluster", unit_detection_because=("named column",),
        rows_in=10, rows_dropped=sum(n for _r, n in drop_reasons),
        drop_reasons=drop_reasons,
        entities_outside_frame=("COL1A1",), entities_not_in_reference=("LUM",),
        frames=(PANEL.ref(),), clustering_mode="independent",
    )


def test_drop_reasons_are_sorted_at_construction():
    """The v2 bug, in its fixed form. A dict here would be unhashable and its
    insertion order would flow through ``asdict`` into the serialised bytes, so
    ``test_record_deterministic`` could never pass -- exactly why
    ``EvidenceItem.corroborating`` sorts in its own ``__post_init__``."""
    r = _reading((("off_panel", 3), ("malformed_row", 1), ("duplicate_unit", 2)))
    assert r.drop_reasons == (("duplicate_unit", 2), ("malformed_row", 1), ("off_panel", 3))


def test_the_record_serialises_identically_whatever_order_the_drops_arrived_in():
    """The property the sort exists for, stated end to end rather than on the field.

    Two readers that dropped the same rows for the same reasons must produce the
    same bytes, or the reproducibility claim is a claim about reader order.
    """
    pairs = (("off_panel", 3), ("malformed_row", 1), ("duplicate_unit", 2))
    a = json.dumps(dataclasses.asdict(_reading(pairs)))
    b = json.dumps(dataclasses.asdict(_reading(tuple(reversed(pairs)))))
    assert a == b


def test_prose_is_optional_and_is_the_only_generated_field():
    """``prose=None`` must be a complete, valid record: it is what the deterministic
    fallback writes when a request is slow or fails, and the demo may never break on
    that path. Everything else is deterministic output from ``core/``."""
    rec = InterpretationRecord(
        unit="c1", task_kind=TaskKind.ASSIGNMENT, reading=_reading(),
        decisions=(Decision("band:from_coef", "High"),),
        claims_used=(("claim:abc", "applied"), ("claim:def", "out_of_scope")),
        undecidable=("the deciding marker is off-panel",), spec_version="0.1",
    )
    assert rec.prose is None

    names = {f.name for f in dataclasses.fields(InterpretationRecord)}
    assert set(GENERATED_FIELDS) <= names, "GENERATED_FIELDS names a field that is gone"
    assert set(GENERATED_FIELDS) == {
        f.name for f in dataclasses.fields(InterpretationRecord)
        if str(f.type) == "str | None"
    }, "a second optional field appeared; decide whether grounding_check must gate it"


def test_records_are_frozen_and_hashable():
    """Hashable because the record is hashed into the golden output; frozen because
    a record edited after the fact is a record that no longer matches its inputs."""
    rec = _reading()
    hash(rec)
    with pytest.raises((AttributeError, TypeError)):
        rec.rows_in = 99        # type: ignore[misc]


# --------------------------------------------------------------------------- #
# The determinism scan, replicated over this stream's two modules
# --------------------------------------------------------------------------- #
def test_no_mutable_fields_in_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes. The record
    determinism invariant cannot hold if any contract carries one.

    The same scan ``test_types_measured.py`` runs over ``subject``/``measured``,
    pointed at this stream's modules. Replicated rather than imported: each stream
    owns its own guard, so deleting one cannot silently stop covering another.
    """
    bad = []
    for mod in (C, R):
        for name in dir(mod):
            obj = getattr(mod, name)
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            for f in dataclasses.fields(obj):
                t = str(f.type)
                if any(x in t for x in ("dict[", "list[", "set[")) and "frozenset" not in t:
                    bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable/unordered fields in frozen contracts: " + "; ".join(bad)


def test_the_mutable_field_scan_is_wired():
    """Guard against silent vacuity: the scan passes trivially if it finds no
    dataclasses at all, which is what a renamed module would look like."""
    found = [
        name for mod in (C, R) for name in dir(mod)
        if isinstance(getattr(mod, name), type)
        and dataclasses.is_dataclass(getattr(mod, name))
        and getattr(mod, name).__module__ in (C.__name__, R.__name__)
    ]
    assert len(found) >= 5, f"the dataclass scan found only {found}"


def test_contracts_reach_for_no_clock_and_no_io():
    """A contract module that imports ``datetime`` or ``pathlib`` has started doing
    something, and "frozen contracts, no logic" stops being checkable. A clock would
    also move the golden hash on every run."""
    banned = {"datetime", "time", "random", "pathlib", "os", "io", "requests"}
    offenders = []
    for mod in (C, R):
        tree = ast.parse(Path(mod.__file__).read_text(), filename=mod.__file__)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{mod.__name__}: import {a.name}" for a in node.names
                              if a.name.split(".")[0] in banned]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in banned:
                    offenders.append(f"{mod.__name__}: from {node.module}")
    assert not offenders, "clock or IO in types/: " + "; ".join(offenders)


# --------------------------------------------------------------------------- #
# The carriers §3 names
# --------------------------------------------------------------------------- #
def test_unit_spec_feeds_the_reading_record():
    """The fields line up so detection's reasoning survives into the record. If they
    stop lining up, ``unit_detection_because`` gets filled in by hand somewhere and
    stops being what the method actually decided."""
    spec = UnitSpec("cluster", "cluster_id", ("c1", "c2"), ("named column present",))
    rec = ReadingRecord(
        path="p", sha256="0" * 64, statistic_column="glm_coef",
        unit_column=spec.unit_column, unit_kind=spec.unit_kind,
        unit_detection_because=spec.because, rows_in=2, rows_dropped=0,
        drop_reasons=(), entities_outside_frame=(), entities_not_in_reference=(),
        frames=(PANEL.ref(),), clustering_mode="independent",
    )
    assert (rec.unit_kind, rec.unit_column, rec.unit_detection_because) == (
        spec.unit_kind, spec.unit_column, spec.because
    )


def test_a_rival_result_can_say_it_settled_nothing():
    """"The deciding marker is off-panel" is the productive output of the
    panel-absence rule, not a failure to report. A RivalResult that could only be
    constructed with discriminating evidence would force the rival test to either
    invent some or stay silent."""
    r = RivalResult("Myoepithelial", undecidable=("KRT14 is not on this panel",))
    assert r.separated_by == () and r.undecidable


def test_the_stubs_are_built_on_the_real_measurement_path():
    """Not a contract test -- a wiring check, so the fixtures above cannot drift into
    hand-made objects that no longer resemble what ``resolve`` produces."""
    assert resolve(ERBB2, WORLD, frozenset({"ERBB2"})).carries_number
