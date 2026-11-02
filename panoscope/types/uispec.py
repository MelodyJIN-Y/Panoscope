"""UISpec: the review surface as part of the interpretation contract.

A workflow owns an interpretive question AND the surface its answers are reviewed
on, because you cannot audit a cell-type call with the figure you would use to audit
an enrichment result (ARCHITECTURE §13). So the surface is *declared*, here, next to
the other contracts -- not hand-built in ``ui/``. A hand-built screen per method does
not generalise and cannot be tested; a declaration can be both.

Three rules are enforced structurally rather than reviewed, and they are the reason
this is a type and not a config file:

**R4 -- no conditionality record, no plot.** A volcano on a 280-gene panel is
pixel-identical to one on the whole transcriptome, and a reader infers genome-wide
coverage unless the figure says otherwise. A misleading figure travels further than a
misleading sentence, because it goes in a talk. So :class:`Conditionality` refuses to
exist without its four mandatory fields, and :class:`PlotSpec` refuses to exist
without a way to build one. Two gates, one principle: the declaration cannot promise
a figure it has no conditions for, and the figure cannot be drawn without them.

**R2 -- degradation goes DOWN, toward making no claim.** ``ui/enrichment_table.py``
degraded *upward*: it imported the assignment header and inherited a confidence band
it had no right to, putting a unit-level band on a task with no label space. The
guards in :meth:`UISpec.__post_init__` make that construction impossible -- a
non-assignment surface cannot hold a ``LabelBandHeader``, and an undeclared task kind
can hold nothing but ``ProvenanceOnlyHeader``, no plots and no pins.

**R8 -- every surface must be able to say "never measured".** A declared workflow
whose blocks cannot render ``NEVER_MEASURED`` has nowhere for the panel-absence rule
to appear, which is the project's headline claim with no pixel behind it.

**Plots render, they do not derive.** There is deliberately no slot on any type here
for an expression, a formula or a callable: ``PlotSpec`` names *fields and views*,
never a computation. The structural version of "no plot computes a statistic that is
not already in the normalised evidence" is that there is nowhere to write one down.
``MethodSpec.status_rule`` is the single callable in the module and it runs
server-side only, which ``test_only_methodspec_carries_code`` pins.

Three deviations from ``docs/interface-design.md`` §3, each because the design doc's
sketch predates a landed type and would have forked a vocabulary:

* ``task_kind`` is ``TaskKind | None``, not a four-member ``Literal`` with
  ``"unknown"``. ``TaskKind`` is frozen at three (CLAUDE.md), and "unknown" exists in
  the sketch only to spell the degraded case. ``None`` spells it without reopening a
  frozen enum that every exhaustive match would then have to handle. ``item_kind``
  is ``str | None`` for the same reason.
* Row status is :class:`MeasuredState`, not a sixth-member ``MeasurementStatus``
  ``Literal``. ``measured.py`` exists because the codebase already had three
  vocabularies for one idea; a fourth, living in the UI layer, is the same bug with
  better manners. The sketch's ``qc_failed`` is ``NEVER_MEASURED`` under the
  detectability frame -- that is what the second gene frame is for -- and its
  ``detected``/``not_supporting`` split is ``OBSERVED``/``TESTED_ABSENT`` exactly.
* ``MethodSpec`` declares ``does_not_measure``, not ``caveats``. CLAUDE.md makes
  ``does_not_measure`` and ``assumptions`` mandatory; two near-synonymous tuples
  would be filled arbitrarily, and the one that is mandatory is the one that must
  carry the weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Literal

from panoscope.types.measured import Measured, MeasuredState
from panoscope.types.outputs import TaskKind
from panoscope.types.subject import SHAPES

# --------------------------------------------------------------------------- #
# R4: no conditionality record, no plot. Enforced at construction.
# --------------------------------------------------------------------------- #
_REQUIRED_COND: tuple[str, ...] = ("drawn", "denominator", "transform", "not_this")


class UndrawableFigure(ValueError):
    """Raised for a figure that cannot state what it is conditioned on.

    The visual analogue of the grounding gate discarding an ungrounded sentence,
    and the cheapest defensible claim in the whole design: a panel that cannot say
    what it drew, out of what denominator, under what transform and what it is *not*
    simply does not render.
    """


@dataclass(frozen=True, slots=True)
class Conditionality:
    """What a figure is conditioned on. Built from the manifest plus the view model.

    The four required fields are the four a reader cannot reconstruct from the
    pixels. ``denominator`` is the one that matters most and the one always left
    out: *280 of ~20,000 transcriptome genes on panel* is the difference between a
    panel result and a genome-wide one, and nothing in the figure shows it.
    """

    drawn: str          # "transcript detections, pre-segmentation"
    denominator: str    # "280 of ~20,000 transcriptome genes on panel"
    transform: str      # "50 um hex bins, area-normalised (tx/um2), sqrt colour"
    not_this: str       # "not per-cell expression; not a statistic"
    unit_scope: str = ""    # "sample1/hb1 - 158,379 cells - c1 = 4,812"
    exclusions: str = ""    # "qv<20 and negative-control probes removed"
    provenance: str = ""    # "density/CDK1_50um.parquet sha256 9f3a... - run 2026-07-08"
    scale: str = ""         # "shared scale, max 0.084 tx/um2 from ERBB2"
    rendering: str = ""     # "background thinned to 45,000 of 153,567 for rendering"

    def __post_init__(self) -> None:
        missing = [f for f in _REQUIRED_COND if not getattr(self, f).strip()]
        if missing:
            raise UndrawableFigure(
                f"Conditionality is incomplete: {', '.join(missing)}. "
                "A panel that cannot state what it is conditioned on does not render."
            )

    def caption(self) -> tuple[str, str]:
        """The two-line mono caption drawn under the panel.

        Two lines rather than one because the first answers *what am I looking at*
        and the second answers *out of what* -- and the second is the one that gets
        truncated when they share a line.
        """
        line1 = " - ".join(p for p in (self.drawn, self.transform, self.scale) if p)
        line2 = " - ".join(
            p for p in (self.denominator, self.unit_scope, self.exclusions,
                        self.rendering, self.not_this) if p
        )
        return line1, line2


# --------------------------------------------------------------------------- #
# Closed vocabularies. Closed because a renderer needs a drawing routine per
# member: a kind it has never heard of cannot be drawn, and the graceful path for
# that is degraded_spec (which declares no plots at all), not a mystery kind.
# --------------------------------------------------------------------------- #
class HeaderKind(StrEnum):
    LABEL_BAND = "label_band"
    CENSUS = "census"
    CONFOUND_CHECKLIST = "confound_checklist"
    PROVENANCE_ONLY = "provenance_only"


class ColumnKind(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    GLYPH = "glyph"
    BAND_MICROBAR = "band_microbar"
    COVERAGE_BAR = "coverage_bar"
    SPECIFICITY_PAIR = "specificity_pair"
    SPARKSTRIP = "sparkstrip"
    DIRECTION = "direction"
    CITATION = "citation"
    NOTE_CHIP = "note_chip"


class PlotKind(StrEnum):
    CELL_MAP = "cell_map"
    DENSITY_HEX = "density_hex"
    UMAP = "umap"
    FEATURE_UMAP = "feature_umap"
    DOTPLOT = "dotplot"
    TESTABILITY_PLANE = "testability_plane"
    VOLCANO_GUTTER = "volcano_gutter"
    REPLICATE_STRIP = "replicate_strip"
    COMPOSITION_BAR = "composition_bar"
    PAIRED_ARM_MAPS = "paired_arm_maps"
    TIER_MATRIX = "tier_matrix"
    SHARED_EVIDENCE_GRAPH = "shared_evidence_graph"


# Derived, never re-listed: the states for which `evidence.py` forbids a statistic.
# Reading it off `Measured.carries_number` means widening that property moves the
# hatching with it, instead of leaving a second list behind for nobody to update.
_NO_NUMBER: tuple[MeasuredState, ...] = tuple(
    s for s in MeasuredState if not Measured(s, frame="").carries_number
)

# Column kinds whose cell IS a number. These are the cells that must never render
# blank or zero for a row that has no number -- "a never-measured item is not a
# blank row and not a zero" is the one claim the interface exists to demonstrate.
_NUMERIC_KINDS: frozenset[ColumnKind] = frozenset(
    {ColumnKind.NUMBER, ColumnKind.BAND_MICROBAR, ColumnKind.COVERAGE_BAR,
     ColumnKind.SPECIFICITY_PAIR, ColumnKind.SPARKSTRIP}
)


# --------------------------------------------------------------------------- #
# Headers. The variants ARE the argument: a census header has no band slot, so a
# band on a task with no label space is structurally undrawable rather than merely
# discouraged.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LabelBandHeader:
    """ASSIGNMENT only. The only header with a band, because it is the only task
    kind with a label space for the remaining probability mass to sit in."""

    kind: Literal[HeaderKind.LABEL_BAND] = HeaderKind.LABEL_BAND
    label_space: str = ""            # "cluster_key.json (9 cell types)"
    band_field: str = "confidence"
    band_score_field: str = "confidence_score"
    band_edges_ref: str = "agent.config.CONFIDENCE_BANDS"
    band_audit_fields: tuple[str, ...] = ("band_basis", "demotions")
    driver_field: str = "key_markers"
    verify_field: str = "verify"
    require_runner_up: bool = True   # the rejected alternative AND what separates it
    runner_up_source: str = "agent.discriminate.discriminate"


@dataclass(frozen=True, slots=True)
class CensusLine:
    label: str                            # "PROGRAMS"
    counts: tuple[tuple[str, str], ...]   # (("enriched","n_enriched"), ...)
    denominator_field: str                # "n_scored" -> "of 50 Hallmark sets scored"
    denominator_template: str = "of {value} scored"


@dataclass(frozen=True, slots=True)
class CensusHeader:
    """CHARACTERISATION. No unit-level band; every count carries its denominator.

    ``imported_context_fields`` is attributed and never re-derived: the cell-type
    call shown beside an enrichment result came from the marker workflow, and
    recomputing it here would give one number two homes and let them disagree.
    """

    kind: Literal[HeaderKind.CENSUS] = HeaderKind.CENSUS
    lines: tuple[CensusLine, ...] = ()
    top_item_template: str = ""
    scope_line_field: str = "panel_scope_caveat"
    gates_field: str = "gates"
    imported_context_fields: tuple[str, ...] = ("cell_type",)
    imported_context_note: str = (
        "call imported from the marker workflow; not re-derived here"
    )


@dataclass(frozen=True, slots=True)
class ConfoundCheck:
    key: str                         # "composition"
    label: str                       # "composition"
    # -> a CheckState value: "ok" | "flagged" | "cannot_be_settled". The third is
    # spelled as types/contracts.CheckState spells it, not as prose: this comment
    # said "not_assessable" while every check emitted "cannot_be_settled", so a
    # renderer written against the comment matched nothing and fell through to the
    # one reading a biologist must not be given -- a check that could not be asked,
    # rendered as one that passed.
    state_field: str
    detail_field: str
    claim_id_field: str = ""         # opens this claim card


@dataclass(frozen=True, slots=True)
class ConfoundChecklistHeader:
    """COMPARISON. Also no band: "does this result make sense?" has no label space
    either, so what the header carries is the checks and what blocked them."""

    kind: Literal[HeaderKind.CONFOUND_CHECKLIST] = HeaderKind.CONFOUND_CHECKLIST
    design_template: str = ""
    results_template: str = ""
    testable_denominator_template: str = ""
    scope_line_field: str = "scope_caveat"
    checks: tuple[ConfoundCheck, ...] = ()


@dataclass(frozen=True, slots=True)
class ProvenanceOnlyHeader:
    """The degradation header for an unregistered analysis type. Claims nothing.

    Not a band it could not compute, not a census it has no denominators for -- a
    banner, the file and its sha. An unregistered method has told us nothing about
    what it measured, so the surface must say nothing about what it found.
    """

    kind: Literal[HeaderKind.PROVENANCE_ONLY] = HeaderKind.PROVENANCE_ONLY
    banner: str = (
        "Unregistered analysis type - Panoscope is showing this artifact as a table "
        "with its provenance. No confidence band, no status assignment and no "
        "interpretation are available for it."
    )


HeaderSpec = (
    LabelBandHeader | CensusHeader | ConfoundChecklistHeader | ProvenanceOnlyHeader
)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
class BlankWhereNoNumber(ValueError):
    """Raised for a numeric column that would render blank where no number exists.

    The headline rule, at the one place it can be enforced before a pixel is drawn:
    a never-measured item is not a blank cell and not a zero. A blank reads as *we
    looked and found nothing*; a hatch reads as *we could not look*, and the
    difference is the difference between evidence and no evidence.
    """


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One table column. ``key`` names a field on the resolved row -- it never
    carries an expression, because a column that could compute would be a plot that
    derives, wearing a different hat."""

    key: str
    label: str
    kind: ColumnKind
    unit: str = ""                   # "glm_coef", "z", "log2FC", "tx/um2"
    precision: int = 2
    align: Literal["left", "right"] = "left"
    width: int = 0                   # 0 = the renderer decides
    hatch_when: tuple[MeasuredState, ...] = _NO_NUMBER
    ref_fields: tuple[str, ...] = () # extra fields carried for the claim card
    sortable: bool = True

    def __post_init__(self) -> None:
        if self.kind in _NUMERIC_KINDS:
            uncovered = [s.value for s in _NO_NUMBER if s not in self.hatch_when]
            if uncovered:
                raise BlankWhereNoNumber(
                    f"column {self.key!r} ({self.kind.value}) does not hatch "
                    f"{', '.join(uncovered)}; a row with no statistic would render "
                    "as blank or zero, which reads as a measured result"
                )


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """One absence block. Sorting happens WITHIN a block, never across one.

    Sorting across blocks is how a top-N quietly hides every never-measured item at
    the bottom of a list nobody scrolls -- which is why ``never_truncate`` and
    ``searchable`` default on: a never-measured item must stay findable.
    """

    key: str
    label_template: str              # "NEVER MEASURED - {count} off panel"
    statuses: tuple[MeasuredState, ...] = ()
    sort_by: str = ""
    sort_desc: bool = True
    collapsed_by_default: bool = False
    searchable: bool = True
    never_truncate: bool = True
    empty_text: str = ""             # shown at count == 0; the header still renders


@dataclass(frozen=True, slots=True)
class CoverageRibbonSpec:
    """The table's own conditionality: one 100%-width stacked bar, counts labelled.

    R4 applied to a table. The ribbon is what stops a screen of twelve supporting
    markers from reading as twelve out of twelve.
    """

    enabled: bool = True
    order: tuple[MeasuredState, ...] = (
        MeasuredState.OBSERVED, MeasuredState.TESTED_ABSENT,
        MeasuredState.UNDERPOWERED, MeasuredState.NEVER_MEASURED,
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PlotSpec:
    """One declared figure. Names fields, views and artifacts -- never a computation.

    ``conditionality_builder`` is a dotted path to a pure function returning a
    :class:`Conditionality`, and it is MANDATORY. IMPLEMENTATION §4 puts a literal
    ``Conditionality`` on the spec instead; that cannot work, because half of what a
    conditionality has to say -- the cell count for this unit, the sha of the
    artifact actually read, the shared colour maximum -- is only known once the view
    model exists. But a builder defaulting to ``""`` would have made R4
    unenforceable at declaration time, which is the one place it costs nothing. So:
    a builder, required.

    The two failure strings are deliberately distinct. "This view does not accept a
    gene set" and "this artifact was not precomputed for this dataset" are different
    facts about the world, and one shared "unavailable" message teaches a biologist
    that the tool is broken when it is in fact being honest.
    """

    key: str
    kind: PlotKind
    title: str
    conditionality_builder: str               # dotted path -> () -> Conditionality
    accepts_subject_kinds: tuple[str, ...] = ()   # registered SHAPES keys
    required_views: tuple[str, ...] = ()          # keys in manifest["views_available"]
    required_artifacts: tuple[str, ...] = ()      # paths hashed in manifest["artifacts"]
    scale_lockable: bool = False
    unsupported_kind_text: str = "{subject}: this view does not accept a {kind}"
    missing_artifact_text: str = (
        "{subject}: {artifact} not precomputed for this dataset"
    )

    def __post_init__(self) -> None:
        if not self.conditionality_builder.strip():
            raise UndrawableFigure(
                f"plot {self.key!r} declares no conditionality builder; a figure "
                "that cannot state its conditions does not render, so declaring it "
                "is a promise the surface cannot keep"
            )
        for k in self.accepts_subject_kinds:
            if k not in SHAPES:
                raise ValueError(
                    f"plot {self.key!r} accepts unregistered subject shape {k!r}; "
                    f"known: {sorted(SHAPES)}"
                )


# --------------------------------------------------------------------------- #
# The workflow declaration
# --------------------------------------------------------------------------- #
class NothingToClick(ValueError):
    """Raised for a rendered claim that resolves to nothing.

    *"Every sentence resolves to its claim, with authority and conditions visible.
    A sentence with nothing to click should not have been rendered."* An
    unanchored sentence is a bug in the caller, not a sentence to draw without
    its anchor, so this raises rather than degrading.

    It lives HERE, not in a renderer, because it is a property of the
    DECLARATION. It was defined in ``ui.surface`` first, so the second renderer
    could not reuse it without importing the first, and ``except NothingToClick``
    caught one medium's violation and not the other's. A second renderer is what
    made that visible.
    """


class SpecCarriesCode(ValueError):
    """Raised when a declaration handed to a renderer carries a callable.

    ``MethodSpec.status_rule`` is the single callable in this module and it runs
    server-side only; the wire carries resolved states. A callable reaching a
    renderer means the renderer is about to decide what counts as measured.

    Same ownership argument as :class:`NothingToClick`.
    """


class DegradedUpward(ValueError):
    """Raised for a surface that would claim more than its task kind permits.

    Named for the bug it prevents: ``ui/enrichment_table.py`` imported the
    assignment header into a characterisation surface and inherited a confidence
    band it had no right to. Degradation must go toward making NO claim, never
    toward reusing another task kind's claim.
    """


class NowhereToSayNeverMeasured(ValueError):
    """Raised for a declared surface whose blocks cannot render ``NEVER_MEASURED``.

    The panel-absence rule needs a place to appear. Without one, an off-panel marker
    is simply missing from the table, which is indistinguishable from one that was
    measured and found nothing -- the exact collapse the four states exist to stop.
    """


@dataclass(frozen=True, slots=True)
class UISpec:
    """A workflow's review surface, declared.

    ``task_kind is None`` means *not declared* -- the degraded surface. It is not a
    fourth task kind: ``TaskKind`` is frozen at three, and a fourth member would
    have to be handled by every exhaustive match over it forever, to spell a state
    that means "we know nothing".
    """

    workflow_id: str                 # "markers" | "pathways" | "contrasts"
    title: str
    task_kind: TaskKind | None       # None = undeclared (degraded surface only)
    unit_kind: str                   # open vocabulary; only TaskKind is frozen
    item_kind: str | None            # a registered SHAPES key, or None if undeclared
    interpretive_question: str       # the sentence the workflow exists to answer
    header: HeaderSpec
    columns: tuple[ColumnSpec, ...]
    blocks: tuple[BlockSpec, ...]
    coverage_ribbon: CoverageRibbonSpec = CoverageRibbonSpec()
    primary_plot: PlotSpec | None = None
    detail_plots: tuple[PlotSpec, ...] = ()
    overview_plots: tuple[PlotSpec, ...] = ()
    pin_kinds: tuple[str, ...] = ()  # registered SHAPES keys
    decision_states: tuple[str, ...] = ("open", "accepted", "overridden", "deferred")
    csv_columns: tuple[str, ...] = ()
    spec_version: str = "1.0"

    def __post_init__(self) -> None:
        declared = self.task_kind is not None
        is_band = self.header.kind is HeaderKind.LABEL_BAND

        # R2, both directions. Anything that is not an assignment has no label space
        # for a band to mean anything in; an assignment without one has no rubric.
        if is_band and self.task_kind is not TaskKind.ASSIGNMENT:
            raise DegradedUpward(
                f"{self.workflow_id}: {self.task_kind} has no label space, so a "
                "unit-level confidence band is a category error, not merely "
                "unhelpful. Use CensusHeader or ConfoundChecklistHeader."
            )
        if self.task_kind is TaskKind.ASSIGNMENT and not is_band:
            raise DegradedUpward(
                f"{self.workflow_id}: assignment must declare its label space and "
                "band rubric in a LabelBandHeader."
            )

        # The degradation principle, structurally: an undeclared task kind may make
        # no claim at all -- no header claim, no plot (it has no conditionality to
        # declare), no pin, and nothing to ground a chat answer on.
        if declared is (self.header.kind is HeaderKind.PROVENANCE_ONLY):
            raise DegradedUpward(
                f"{self.workflow_id}: ProvenanceOnlyHeader iff the task kind is "
                f"undeclared (task_kind={self.task_kind}, "
                f"header={self.header.kind.value}). An unregistered analysis type "
                "degrades toward making no claim, never toward borrowing one."
            )
        if not declared and (self.primary_plot or self.detail_plots
                             or self.overview_plots or self.pin_kinds):
            raise DegradedUpward(
                f"{self.workflow_id}: an undeclared task kind cannot declare plots "
                "or pins. Nobody declared a conditionality for them (R4), and a pin "
                "selects a subject this surface cannot name."
            )

        shapes = (*self.pin_kinds, *((self.item_kind,) if self.item_kind else ()))
        for k in shapes:
            if k not in SHAPES:
                raise ValueError(
                    f"{self.workflow_id}: unregistered subject shape {k!r}; "
                    f"known: {sorted(SHAPES)}"
                )

        # R8: somewhere to say "never measured", and no block that declares nothing.
        if declared:
            empty = [b.key for b in self.blocks if not b.statuses]
            if empty:
                raise NowhereToSayNeverMeasured(
                    f"{self.workflow_id}: block(s) {empty} declare no measurement "
                    "status, so their rows would render with no status at all"
                )
            covered = {s for b in self.blocks for s in b.statuses}
            if MeasuredState.NEVER_MEASURED not in covered:
                raise NowhereToSayNeverMeasured(
                    f"{self.workflow_id}: no block renders 'never_measured'. The "
                    "panel-absence rule must have somewhere to appear."
                )


# --------------------------------------------------------------------------- #
# The method declaration (registers INTO a workflow)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MethodSpec:
    """A machine-readable declaration of what a method's paper says it measures.

    ``assumptions`` and ``does_not_measure`` are mandatory and non-empty (CLAUDE.md).
    They are the point of the object: adding a method is a declaration, not
    interpretation logic, and a method that declares no limits is one whose output
    reads as having none.

    ``status_rule`` is the one callable in this module and it runs SERVER-SIDE only.
    The wire carries resolved statuses, so no renderer ever decides what counts as
    measured -- which is also why nothing else here may carry code.
    """

    method_id: str                   # "jazzpanda_spatial_glm"
    workflow_id: str
    display_name: str
    assumptions: tuple[str, ...]
    outputs: tuple[str, ...]         # "glm_coef (spatial GLM Estimate)", "pearson"
    does_not_measure: tuple[str, ...]
    status_rule: Callable[[object, object], MeasuredState] | None = field(
        repr=False, default=None
    )
    extra_columns: tuple[ColumnSpec, ...] = ()
    citation: str = ""               # the method's own paper

    def __post_init__(self) -> None:
        for name in ("assumptions", "does_not_measure"):
            if not getattr(self, name):
                raise ValueError(
                    f"{self.method_id}: {name} is mandatory. A method that declares "
                    "no limits is one whose output reads as having none."
                )


# --------------------------------------------------------------------------- #
# The registry. Data, like SHAPES / RULES / FLAGS -- adding a workflow is a
# registration, not a type change.
# --------------------------------------------------------------------------- #
_WORKFLOWS: dict[str, UISpec] = {}
_METHODS: dict[str, MethodSpec] = {}


class DuplicateRegistration(KeyError):
    """Raised when two registrations claim one id. Last-write-wins between two
    import-time registrations is decided by import order, which is decided by
    nothing -- and the loser fails at render time, far from the cause."""


def register_workflow(spec: UISpec) -> None:
    if spec.workflow_id in _WORKFLOWS:
        raise DuplicateRegistration(f"workflow {spec.workflow_id!r} already registered")
    _WORKFLOWS[spec.workflow_id] = spec


def register_method(spec: MethodSpec) -> None:
    if spec.method_id in _METHODS:
        raise DuplicateRegistration(f"method {spec.method_id!r} already registered")
    _METHODS[spec.method_id] = spec


def resolve(workflow_id: str, name: str, columns: tuple[str, ...]) -> UISpec:
    """The surface for ``workflow_id``, or a degraded one. NEVER raises.

    Takes the artifact's name and column names rather than an artifact handle: those
    are the only two things :func:`degraded_spec` reads, and depending on a handle
    type owned by another module would buy a forward reference and an import cycle
    for nothing.
    """
    spec = _WORKFLOWS.get(workflow_id)
    return spec if spec is not None else degraded_spec(name, columns)


def degraded_spec(name: str, columns: tuple[str, ...]) -> UISpec:
    """The fallback surface for an analysis type nobody registered.

    In order of importance:

    1. **No header claim.** A banner and the provenance. No band, no census, no
       checks.
    2. **No status assignment.** Nothing is called detected and -- critically --
       nothing is called absent, because an unregistered method has told us nothing
       about what it measured. The single block declares no statuses.
    3. **No plots.** A plot needs a conditionality; nobody declared one; R4 applies.
    4. **No pin, no sign-off, no CSV.** ``decision_states`` is ``("open",)`` and
       ``csv_columns`` is empty: an analysis type nobody registered cannot be signed
       off on.
    5. **Columns verbatim, as text.** The renderer must not reformat, round or
       right-align a number it does not understand -- that is interpretation, and
       interpretation is exactly what is missing here.
    """
    return UISpec(
        workflow_id=f"unregistered:{name}",
        title=name,
        task_kind=None,
        unit_kind="unknown",
        item_kind=None,
        interpretive_question="(not declared)",
        header=ProvenanceOnlyHeader(),
        columns=tuple(
            ColumnSpec(key=c, label=c, kind=ColumnKind.TEXT, hatch_when=())
            for c in columns
        ),
        blocks=(BlockSpec(key="all", label_template="ROWS - {count}"),),
        coverage_ribbon=CoverageRibbonSpec(enabled=False),
        decision_states=("open",),
        csv_columns=(),
    )
