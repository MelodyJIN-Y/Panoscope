"""The four registry contracts: what a method, a platform, a source and a workflow must
*declare* before anything may use one.

They are ``Protocol`` classes, not base classes: an entry declares its contract by having
the attributes, so the registries share no ancestor and a plugin need not import
Panoscope's classes.

- :class:`MethodModule` is a machine-readable statement of what a method's paper says it
  measures and does not measure. ``does_not_measure`` and ``assumptions`` are mandatory;
  the Protocol makes them present, ``test_every_method_declares_limits`` non-empty.
- Clustering mode (§6.3's conditional blocker) belongs to the adapter:
  :meth:`PlatformAdapter.samples` holds the sample count and
  :meth:`PlatformAdapter.clustering_mode` raises :class:`UndeclaredClustering` rather
  than guessing.
- No protocol method accepts a ``Trace``; decisions flow out
  (``test_no_protocol_method_accepts_a_trace``).

``UISpec`` and ``reference.graph``'s ``Node``/``Edge`` are imported under
``TYPE_CHECKING``. ``UnitSpec``, ``Check`` and ``RivalResult`` appear in §3's signatures
with no declared home, so they are defined here as plain carriers; widening one is a
reviewed contract change (§11).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import (
    TYPE_CHECKING, Any, Iterator, Literal, Mapping, Protocol, runtime_checkable,
)

from panoscope.types.conditions import Condition, Conditions
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import Coverage, Frame, Measured
from panoscope.types.outputs import Assignment, Finding, TaskKind
from panoscope.types.subject import Subject
from panoscope.types.trace import Traced

if TYPE_CHECKING:  # pragma: no cover - annotations only; see the module docstring
    from panoscope.reference.graph import Edge, Node
    from panoscope.types.uispec import UISpec


class UndeclaredClustering(ValueError):
    """Raised when a multi-sample store does not say how it was clustered.

    Never defaulted. Joint and independent clustering give the same cluster *labels*
    two different meanings, and reading cluster 3 in sample A as the same population
    as cluster 3 in sample B is wrong under independent clustering and invisible
    under either -- the labels match whichever answer is true.
    """


class UnrecognisedResult(ValueError):
    """Raised when a method cannot read its unit kind and ids off the result schema.

    A method that guesses here assigns an interpretive question to a table that may
    not answer it. Refusing is the honest output: the biologist can say what the
    table is, and nothing downstream has to carry a guess.
    """


# --------------------------------------------------------------------------- #
# Carriers §3 names but §1 gives no home. No logic; see the module docstring.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class UnitSpec:
    """What :meth:`MethodModule.detect_unit` read off the result schema.

    Three of the four fields feed ``ReadingRecord`` straight through -- ``unit_kind``,
    ``unit_column`` and ``because`` are the record's ``unit_kind``, ``unit_column``
    and ``unit_detection_because``. Detection carries its reasoning because "which
    column is the unit" is the one read that silently reinterprets every row beneath
    it.
    """

    unit_kind: str                   # open vocabulary; only TaskKind is frozen
    unit_column: str
    units: tuple[str, ...]
    because: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Check:
    """One computable question a workflow asks of a result (ARCHITECTURE §9.7).

    Declared, not computed: :meth:`Workflow.checks` returns what the workflow *can*
    ask, so the review surface can show a check that returned *cannot be settled*
    next to the ones that fired. A check visible only when it produces an answer is
    a check the biologist cannot tell was skipped.
    """

    id: str          # REGISTERED id, never prose -- the UI and tests match on this
    question: str    # biologist-facing, one sentence


class CheckState(StrEnum):
    """Three states, and the third is the point.

    ``CANNOT_BE_SETTLED`` is not a soft ``FLAGGED``: a flagged check was asked and answered;
    an unsettled one could not be asked, and names what is missing. Here, not in a workflow,
    because every workflow and ``core/propagate.py`` need it.
    """

    OK = "ok"
    FLAGGED = "flagged"
    CANNOT_BE_SETTLED = "cannot_be_settled"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One :class:`Check`'s answer, with what it read and what it still needs.

    ``missing`` is the actionable half and is never prose about the tool: each entry
    names a thing that would settle the check -- "a second biological replicate",
    "cluster abundances per condition" -- so a biologist can go and get it.
    ``because`` is what the check actually read.
    """

    check_id: str
    state: CheckState
    headline: str
    because: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    coverage: tuple[Coverage, ...] = ()

    @property
    def settled(self) -> bool:
        return self.state is not CheckState.CANNOT_BE_SETTLED


@dataclass(frozen=True, slots=True)
class RivalResult:
    """The rival test's output (ARCHITECTURE §9.6), in the shape every task kind shares.

    Name a specific alternative account, name the measured evidence that separates
    it, then say what the data cannot settle. ``undecidable`` is not a failure field
    -- "the deciding marker is off-panel" is the productive output of the
    panel-absence rule and must be reported, not folded into a lower band.
    """

    alternative: str
    separated_by: tuple[EvidenceItem, ...] = ()
    undecidable: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# The four contracts
# --------------------------------------------------------------------------- #
@runtime_checkable
class PlatformAdapter(Protocol):
    """How one platform's store answers the questions interpretation depends on.

    Every member here is a *read* of the store. None of them computes a statistic:
    the adapter says what the apparatus was capable of and what it was told; the
    method says what it found.
    """

    name: str

    def frames(self, store: Any) -> tuple[Frame, ...]:
        """The capability frames this store declares, one per entity kind.

        Xenium: ``var_names`` minus negative controls and custom add-ons. **Derived
        from the file, NEVER a constant** -- panel size is dynamic across studies,
        and a hardcoded 280 turns the second study's immuno-oncology panel into 280
        genes that are not there.
        """

    def is_control(self, store: Any, entity: str) -> bool:
        """True if ``entity`` is a control probe of this platform, not a subject.

        Declared here rather than left to each reader because a control is not a
        state of the four-state lattice -- it is **excluded from every frame**, and
        ``MeasuredState`` has no value for "measured, but not about anything". Ask a
        frame about one and it answers ``NEVER_MEASURED``, byte-identical to a gene
        the assay never probed and false about a probe whose count is on disk. So
        the row has to be refused before it becomes a subject, and only the platform
        knows which rows those are.

        Xenium answers from the probe's NAME and from the panel file's own
        ``annotation`` column, because a vendor renames the first and not the second.
        """

    def unit_labels(self, store: Any) -> dict[str, str]:
        """Unit id -> display label. Display only; never load-bearing."""

    def samples(self, store: Any) -> list[str]:
        """The samples in this store, first-hand. The sample count lives here and
        nowhere else -- ``Conditions`` deliberately cannot express it."""

    def clustering_mode(self, store: Any) -> Literal["joint", "independent", "integrated"]:
        """MUST be explicit. Raises :class:`UndeclaredClustering`. Never guesses.

        §6.3's fourth blocker, and the only conditional one: it blocks the study when
        :meth:`samples` returns more than one. Enforcing it is the adapter's job,
        because the adapter is what holds the sample count.
        """

    def available_views(self, store: Any) -> frozenset[str]:
        """Which views the elements actually present can support. No points -> no
        density view. Never fake: an unavailable view drawn anyway is a plot that
        invented its data."""

    def coordinate_system(self, store: Any) -> tuple[str, float]:
        """``(name, scale)``, where scale converts one stored unit to micrometres.

        A spatial statistic read at the wrong scale is wrong by a constant nobody
        can see in the figure.
        """

    def reads(self, store: Any) -> Mapping[str, Condition]:
        """The partial read: ``Conditions`` field name -> what the files say.

        ``Provenance.READ`` for anything parsed out of the files, ``UNKNOWN`` where they are
        silent; never ``INFERRED``. Whether it is enough is intake's job. A mapping, unlike
        :meth:`standing_conditions`, survives a blocking unknown, so intake still gets what the
        store does declare. ``frames`` is not here; :meth:`frames` returns it.
        """

    def standing_conditions(self, store: Any) -> Conditions:
        """:meth:`reads`, finished into a ``Conditions``. Raises when it cannot be.

        The same values, assembled with ``frames`` into the object every
        applicability check consumes. Raises ``Blocked`` on a blocking unknown,
        which is what a *finished* object is for -- a caller that wants what the
        files said without that assertion calls :meth:`reads`.
        """

    def validate(self, store: Any) -> list[str]:
        """Problems that would make this store uninterpretable, as readable strings;
        empty means usable. Returns rather than raises because intake renders a
        checklist, and a traceback is not a checklist."""


@runtime_checkable
class MethodModule(Protocol):
    """A machine-readable declaration of what a method's paper says it measures and
    does not measure.

    Adding a method is a *declaration*, not interpretation logic. Everything that
    reads a number downstream reads it through the limits declared here, which is
    what makes the confidence assessment an audit of someone else's output rather
    than a self-assessment.
    """

    name: str
    workflow: str                    # which interpretive question this answers
    statistic_name: str
    measures: str
    does_not_measure: str            # MANDATORY, from the paper, rendered in the UI
    assumptions: tuple[str, ...]     # MANDATORY, from the paper
    rival_space: Literal["reference", "data", "none"]
    citation: str                    # the method's own paper. Real; looked up live.

    def detect_unit(self, table: Any) -> UnitSpec:
        """Read the unit kind and ids FROM THE RESULT SCHEMA.

        Raises :class:`UnrecognisedResult` on a shape it does not know. Never
        guesses -- a wrong unit kind routes the table to the wrong interpretive
        question, and every answer under it is confidently about the wrong thing.
        """

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> "Traced[dict[str, NormalisedEvidence]]":
        """Precomputed output in, normalised evidence out, keyed by unit.

        **Reads, never recomputes.** The method's output is a consumed input; that is the basis
        of the audit. Returns ``Traced`` because loading decides: it drops rows, applies frames
        and withholds statistics the frame refuses.
        """

    def absence_reads_as(self, subject: Subject, measured: Measured) -> str:
        """What this method's silence about a subject MEANS, in its own terms.

        The four states are generic; what a tested absence *argues* is not. A
        spatial method's ``TESTED_ABSENT`` says "on the panel, no spatial enrichment
        here"; a differential-expression method's says "no change between these
        conditions". One sentence, deterministic, no clock, no model.
        """

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        """The band, for ASSIGNMENT methods only, with the decisions that produced it.

        Declared per method because the band is a function of *one method's* output:
        a spatial-localisation rubric and a mean-expression rubric must disagree on
        a high-fold-change, low-localisation case, which is what
        ``test_module_rubrics_differ`` pins before the second method exists.
        """

    def traversals(self) -> tuple[tuple[str, str], ...]:
        """Which reference-graph edges this method's evidence licenses following, as
        ``(predicate, direction)`` pairs. Reference data selects and orders evidence;
        it never supplies a value."""


@runtime_checkable
class Workflow(Protocol):
    """Owns one interpretive question AND the review surface that answers it.

    The surface is part of the interpretation contract, not a UI concern: a volcano
    drawn on a 280-gene panel is indistinguishable from a whole-transcriptome one
    unless the thing that owns the question says so. :meth:`uispec` is therefore
    declared here and not in ``ui/``.
    """

    name: str                        # "marker-interpretation"
    task_kind: TaskKind
    question: str                    # the biologist-facing question, one sentence
    skill_path: str                  # skills/<name>/SKILL.md

    def interpret(self, ev: NormalisedEvidence,
                  reference: Any) -> Traced[Assignment | Finding]:
        """``Assignment`` for ASSIGNMENT, ``Finding`` for the other two. The union is
        the type-level form of the no-band rule: there is no shape in which a
        characterisation can carry one."""

    def rival_test(self, ev: NormalisedEvidence, alternative: str,
                   reference: Any) -> Traced[RivalResult]: ...

    def checks(self) -> tuple[Check, ...]:
        """What this workflow can ask of a result -- e.g. the DE workflow's four."""

    def uispec(self) -> UISpec:
        """The review surface this question requires (ARCHITECTURE §13)."""


@runtime_checkable
class ReferenceSource(Protocol):
    """A source of reference edges, and the conditions they were established under.

    ``conditions()`` is what keeps the rule about reference data enforceable: an edge
    carries the conditions of *its own* study, so a transfer check can refuse it here
    rather than let it quietly corroborate a measurement taken somewhere else.
    ``redistributable`` is separate from ``licence`` because the licence CI job needs
    a boolean it can act on, not a string it has to interpret.
    """

    name: str
    release: str
    licence: str
    redistributable: bool

    def conditions(self) -> Conditions:
        """The conditions under which THIS SOURCE's edges were established."""

    def nodes(self) -> Iterator[Node]: ...

    def edges(self) -> Iterator[Edge]: ...
