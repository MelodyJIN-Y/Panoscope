"""The DE workflow: *does this result make sense?* -- and the surface that answers it.

**This workflow does not run differential expression.** It reads someone else's DE
result and audits it.

``TaskKind.COMPARISON``, therefore **no band** (§9.1): ``Finding`` has no field for
one and ``UISpec.__post_init__`` refuses a ``LabelBandHeader``. The header carries
the checks and what blocked them.

**The four checks (ARCHITECTURE §9.7)**, declared in :data:`CHECKS`:

1. :func:`panel_scoping` -- K of N tested, and which canonical genes for the affected
   process were never measured.
2. :func:`replicate_confound` -- is the effect driven by one replicate or section?
   From a result table alone this is *cannot be settled*, naming what is missing.
3. :func:`composition_vs_expression` -- did the gene change, or the mixture?
4. :func:`cross_method_coherence` -- does the DE result agree with the spatial result
   about the same subject? Delegates to ``core/reconcile.py``.

Three check states, not two (``types/contracts.CheckState``): every
``CANNOT_BE_SETTLED`` carries ``missing``, the named things that would settle it.

Not owned here: the lattice, the join, the effect-size ladder. Two methods register
into this workflow (``edger``, ``seurat_de``), picked by schema in
``pipeline/stages/de.py``; their difference is in their declarations and reaches no
branch here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from panoscope.core.propagate import requires
from panoscope.core.reconcile import Agreement, Relationship, Side, reconcile
from panoscope.types.claim import Authority
from panoscope.types.contracts import (
    Check, CheckResult, CheckState, RivalResult,
)
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import Coverage, MeasuredState, MeasuredWorld
from panoscope.types.outputs import Finding, FindingItem, TaskKind, Tier
from panoscope.types.subject import EntityKind
from panoscope.types.trace import Trace, Traced, fmt_num, ref
from panoscope.types.uispec import (
    BlockSpec, ColumnKind, ColumnSpec, ConfoundCheck, ConfoundChecklistHeader,
    CoverageRibbonSpec, UISpec,
)

WORKFLOW_ID: str = "de-interpretation"

#: ``UISpec.workflow_id`` is the short surface id the shell routes on; the workflow's
#: own ``name`` is the skill directory. Deliberately different strings, so renaming a
#: tab cannot silently re-point a skill.
SURFACE_ID: str = "contrasts"

# --------------------------------------------------------------------------- #
# The replicate frame, named once (§9.5): too few replicates makes the UNIT
# underpowered. A property of the QUESTION, so it lives here; the DE methods
# import both and write the Coverage at load time.
# --------------------------------------------------------------------------- #
REPLICATE_COVERAGE: str = "biological replicates"
REPLICATE_FLOOR: int = 2

# --------------------------------------------------------------------------- #
# The four checks, DECLARED. Ids, never prose: the UI and the tests match on
# these, and a check that returned *cannot be settled* must be visible next to
# the ones that fired. A check the biologist cannot see was skipped is a check
# they will assume passed.
# --------------------------------------------------------------------------- #
PANEL_SCOPING: str = "panel_scoping"
REPLICATE_CONFOUND: str = "replicate_confound"
COMPOSITION: str = "composition_vs_expression"
COHERENCE: str = "cross_method_coherence"

CHECKS: tuple[Check, ...] = (
    Check(PANEL_SCOPING,
          "Is this a transcriptome-wide result or a panel-scoped one -- and which "
          "canonical genes for the affected process were never measured?"),
    Check(REPLICATE_CONFOUND,
          "Is this effect driven by one replicate or section, or does it hold across "
          "biological replicates?"),
    Check(COMPOSITION,
          "Did the gene change, or did the cell-type mixture change?"),
    Check(COHERENCE,
          "Does this result say the same thing as the spatial result about the same "
          "subject?"),
)


@dataclass(frozen=True, slots=True)
class DEAudit:
    """Everything the four checks need that a DE result table cannot supply.

    Passed in (this layer does no IO); every field is *absent by default*, so the
    checks refuse rather than assume. ``world`` is here, as ``MarkerReference.world``
    is, because telling *never measured* from *tested absent* needs frame membership.
    """

    world: MeasuredWorld = MeasuredWorld()
    process: str = ""                                # the affected process, named
    canonical: tuple[str, ...] = ()                  # its canonical genes
    # (condition, this unit's share of cells under it). Sorted pairs, never a dict:
    # a dict field on a frozen dataclass is unhashable and its order would leak into
    # the serialised record.
    abundance: tuple[tuple[str, float], ...] = ()
    spatial: Side | None = None                      # the other method, SAME unit
    #: The method's declared ``tests_whole_frame``. False: a panel gene missing from
    #: the table may never have been tested, so its silence is not TESTED_ABSENT.
    tested_whole_frame: bool = False
    relationship: Relationship = Relationship.PARTLY_INDEPENDENT


# --------------------------------------------------------------------------- #
# Check 1: panel scoping
# --------------------------------------------------------------------------- #
def panel_scoping(ev: NormalisedEvidence, audit: DEAudit) -> Traced[CheckResult]:
    """A DE result on a targeted panel is not a transcriptome-wide result.

    States K of N against the panel the apparatus had, and names the affected
    process's canonical genes that were never on the panel. With no gene frame
    declared the check refuses rather than guessing a denominator.
    """
    t = Trace()
    anchor = (ref(ev.unit_subject),)
    frames = audit.world.for_kind(EntityKind.GENE)
    if not frames:
        r = CheckResult(
            PANEL_SCOPING, CheckState.CANNOT_BE_SETTLED,
            headline=f"{ev.unit}: the capability frame for genes was not declared, "
                     "so the scope of this result cannot be stated",
            missing=("the panel this result was produced on, as a capability frame; "
                     "panel size is dynamic across studies and must be read from the "
                     "file, never asserted",),
        )
        t.note("de:panel_frame_absent", r.state.value, (r.headline,), anchor)
        return t.done(r)

    frame = frames[0]
    observed = sum(1 for i in ev.items if i.measured.carries_number)
    cov = Coverage(f"{frame.name} genes with a result in {ev.unit}",
                   k=observed, n=len(frame.members), floor=1)
    head = (f"{ev.unit}: panel-scoped, not transcriptome-wide -- {cov}. "
            "Every absence below is an absence from this panel.")

    never = tuple(g for g in audit.canonical if g not in frame.members)
    if not audit.canonical:
        r = CheckResult(
            PANEL_SCOPING, CheckState.CANNOT_BE_SETTLED, headline=head,
            because=(str(cov),), coverage=(cov,),
            missing=("the canonical gene list for the affected process: without it, "
                     "which canonical genes were never measured cannot be stated, "
                     "and a reader will read this panel as the transcriptome",),
        )
    elif never:
        r = CheckResult(
            PANEL_SCOPING, CheckState.FLAGGED,
            headline=head + (f" {len(never)} canonical gene(s) for "
                             f"{audit.process or 'the affected process'} were never "
                             "measured: " + ", ".join(never)),
            because=(str(cov),
                     f"never on {frame.name}: {', '.join(never)} -- their absence "
                     "from this result argues nothing",),
            coverage=(cov,),
            missing=tuple(f"{g}: never on {frame.name}" for g in never),
        )
    else:
        r = CheckResult(
            PANEL_SCOPING, CheckState.OK,
            headline=head + (" Every canonical gene for "
                             f"{audit.process or 'the affected process'} was on the "
                             "panel."),
            because=(str(cov),
                     f"all {len(audit.canonical)} canonical gene(s) are in "
                     f"{frame.name}",),
            coverage=(cov,),
        )
    t.note("de:panel_scoped", r.state.value, r.because or (r.headline,), anchor)
    return t.done(r)


# --------------------------------------------------------------------------- #
# Check 2: replicate / sample confound -- §9.7's headline refusal
# --------------------------------------------------------------------------- #
def replicate_confound(ev: NormalisedEvidence, audit: DEAudit) -> Traced[CheckResult]:
    """Is the effect driven by one replicate or section? §9.7's headline behaviour.

    Reads the unit's own capability verdict (``unit_measured``, written by the method
    at load from the adapter's sample count). Never OK from a result table alone: it
    reports one number per gene per unit, so the per-sample results are named as
    missing (``core/concordance.py`` computes it once they exist).
    """
    t = Trace()
    anchor = (ref(ev.unit_subject),)
    cov = next((c for c in ev.unit_measured.coverage
                if c.what == REPLICATE_COVERAGE), None)

    if cov is None:
        r = CheckResult(
            REPLICATE_CONFOUND, CheckState.CANNOT_BE_SETTLED,
            headline=f"{ev.unit}: CANNOT BE SETTLED -- the design's replicate count "
                     "was never declared, so whether one replicate drives this "
                     "result cannot be asked",
            missing=("the number of biological replicates behind this result; the "
                     "platform adapter holds the sample count first-hand and nothing "
                     "else may guess it",),
        )
        t.note("de:replicate_confound", r.state.value, (r.headline,), anchor)
        return t.done(r)

    if not cov.ok:
        r = CheckResult(
            REPLICATE_CONFOUND, CheckState.CANNOT_BE_SETTLED,
            headline=f"{ev.unit}: CANNOT BE SETTLED -- {cov}. With {cov.k} "
                     "biological replicate(s) there is nothing to compare this "
                     "effect against, so condition, batch and section are "
                     "confounded by construction and no condition-level claim "
                     "follows from this result.",
            because=(str(cov),
                     f"the unit is {ev.unit_measured.state.value} in "
                     f"{ev.unit_measured.frame}: underpowered for a condition-level "
                     "claim as a UNIT, not as any individual gene in it",),
            missing=(f"at least {cov.floor - cov.k} further biological replicate(s) "
                     f"(have {cov.k}, floor {cov.floor})",
                     "or an orthogonal assay on independent material, which would be "
                     "an independent witness rather than a second look at this one",),
            coverage=(cov,),
        )
        t.note("de:replicate_confound", r.state.value, r.because, anchor)
        return t.done(r)

    r = CheckResult(
        REPLICATE_CONFOUND, CheckState.CANNOT_BE_SETTLED,
        headline=f"{ev.unit}: CANNOT BE SETTLED -- {cov} clears the floor, but this "
                 "result reports one number per gene for the whole unit, so which "
                 "replicate drives it is not in the table",
        because=(str(cov),),
        missing=("per-sample effect sizes -- the same contrast run within each "
                 "sample; core/concordance.py reads those and reports divergence as "
                 "a flag (agreement never raises confidence, ARCHITECTURE 9.10)",),
        coverage=(cov,),
    )
    t.note("de:replicate_not_in_result", r.state.value, r.because, anchor)
    return t.done(r)


# --------------------------------------------------------------------------- #
# Check 3: composition versus expression
# --------------------------------------------------------------------------- #
def composition_vs_expression(ev: NormalisedEvidence,
                              audit: DEAudit) -> Traced[CheckResult]:
    """Did the gene change, or did the cell-type mixture change?

    The lever is this unit's abundance across the arms, which must be supplied. **No
    threshold is invented**: any difference in share confounds the two claims;
    identical shares are the only case where the per-cell reading stands.
    """
    t = Trace()
    anchor = (ref(ev.unit_subject),)

    # THE CONDITION GATE, before the abundance one: a share of CELLS depends on
    # segmentation, so with segmentation unknown the question cannot be ASKED,
    # and that larger gap must not be hidden behind missing abundance.
    gate = t.absorb(requires(ev.conditions, COMPOSITION, "segmentation"))
    if not gate.settled:
        return t.done(gate)

    shares = tuple(sorted(audit.abundance))

    if len(shares) < 2:
        r = CheckResult(
            COMPOSITION, CheckState.CANNOT_BE_SETTLED,
            headline=f"{ev.unit}: CANNOT BE SETTLED -- whether these genes changed "
                     "or the cell-type mixture changed does not follow from this "
                     "result",
            because=(f"{len(shares)} arm(s) of cluster abundance supplied; two are "
                     "needed to compare",),
            missing=("this unit's share of cells under each arm being compared; a DE "
                     "table carries no abundance, and a gene reads as up either way",),
        )
        t.note("de:composition_not_supplied", r.state.value, r.because, anchor)
        return t.done(r)

    lo, hi = min(s for _c, s in shares), max(s for _c, s in shares)
    detail = ", ".join(f"{c} {fmt_num(s)}" for c, s in shares)
    if lo != hi:
        r = CheckResult(
            COMPOSITION, CheckState.FLAGGED,
            headline=f"{ev.unit}: CONFOUNDED -- this unit's share of cells differs "
                     f"across arms ({detail}), so a gene can read as up because the "
                     "cells expressing it became more numerous. This measurement "
                     "does not separate the two claims.",
            because=(f"cluster abundance: {detail}",
                     f"difference {fmt_num(hi - lo)} across arms",),
            missing=("a within-cell-type comparison, or a per-cell test conditioned "
                     "on abundance; either separates the claims and neither is in "
                     "this result",),
        )
        t.note("de:composition_confounded", r.state.value, r.because, anchor)
        return t.done(r)

    r = CheckResult(
        COMPOSITION, CheckState.OK,
        headline=f"{ev.unit}: this unit's share of cells is unchanged across arms "
                 f"({detail}), so a change here is not explained by the mixture",
        because=(f"cluster abundance: {detail}",),
    )
    t.note("de:composition_separated", r.state.value, r.because, anchor)
    return t.done(r)


# --------------------------------------------------------------------------- #
# Check 4: cross-method coherence -- delegated to core/reconcile.py
# --------------------------------------------------------------------------- #
def cross_method_coherence(ev: NormalisedEvidence,
                           audit: DEAudit) -> Traced[CheckResult]:
    """Does the DE result say the same thing as the spatial result about one subject?

    The join, the lattice gate and the §9.11 question-sameness mapping live in
    ``core/reconcile.py``; this supplies the DE side. Different questions, so only
    COHERENT, TENSION or NOT_COMPARABLE -- never AGREE.
    """
    t = Trace()
    anchor = (ref(ev.unit_subject),)
    if audit.spatial is None:
        r = CheckResult(
            COHERENCE, CheckState.CANNOT_BE_SETTLED,
            headline=f"{ev.unit}: CANNOT BE SETTLED -- no second method's result was "
                     "supplied, so cross-method coherence was not checked",
            missing=("a spatial result for this same unit; without one this is "
                     "unchecked, which is not the same as checked and coherent",),
        )
        t.note("de:coherence_not_comparable", r.state.value, (r.headline,), anchor)
        return t.done(r)

    mine = Side(module=ev.module, workflow=WORKFLOW_ID,
                task_kind=TaskKind.COMPARISON, unit=ev.unit,
                authority=Authority.MEASUREMENT, items=ev.items,
                # An empty world reads a silent gene as NEVER_MEASURED ("cannot speak
                # to it") when the method did not test every gene in the frame.
                world=audit.world if audit.tested_whole_frame else MeasuredWorld())
    joined = t.absorb(reconcile(audit.spatial, mine, audit.relationship))

    tension = joined.with_outcome(Agreement.TENSION)
    coherent = joined.with_outcome(Agreement.COHERENT)
    blocked = joined.with_outcome(Agreement.NOT_COMPARABLE)
    counts = (f"{len(coherent)} coherent, {len(tension)} in tension, "
              f"{len(blocked)} could not be compared, of {len(joined.pairs)} "
              "subject(s)")
    # Named, never counted away: "could not be compared" is a row with a reason.
    unblockable = tuple(f"{p.subject.display()}: {b}"
                        for p in blocked for b in p.because)

    if tension:
        names = ", ".join(p.subject.display() for p in tension)
        r = CheckResult(
            COHERENCE, CheckState.FLAGGED,
            headline=f"{ev.unit}: TENSION with {audit.spatial.module} on {names} "
                     f"({counts}). Different questions, so this is never a "
                     "disagreement about one fact -- it is two measurements that do "
                     "not sit together.",
            because=tuple(b for p in tension for b in p.because),
            missing=unblockable,
        )
        t.note("de:coherence_tension", r.state.value, r.because, anchor)
        return t.done(r)

    if not coherent:
        r = CheckResult(
            COHERENCE, CheckState.CANNOT_BE_SETTLED,
            headline=f"{ev.unit}: CANNOT BE SETTLED -- nothing could be compared "
                     f"with {audit.spatial.module} ({counts})",
            because=tuple(b for p in blocked for b in p.because),
            missing=unblockable or ("a subject both methods were capable of seeing",),
        )
        t.note("de:coherence_not_comparable", r.state.value, r.because, anchor)
        return t.done(r)

    r = CheckResult(
        COHERENCE, CheckState.OK,
        headline=f"{ev.unit}: coherent with {audit.spatial.module} ({counts}). "
                 "Coherent, never agreed: the two answer different questions, and "
                 "coherence raises no confidence here -- there is no band on either "
                 "side of this join.",
        because=tuple(b for p in coherent for b in p.because),
        missing=unblockable,
    )
    t.note("de:coherence_coherent", r.state.value, r.because, anchor)
    return t.done(r)


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DEInterpretation:
    """The registered workflow object. Declaration plus composition, no statistics."""

    name: str = WORKFLOW_ID
    task_kind: TaskKind = TaskKind.COMPARISON
    question: str = (
        "Does this differential-expression result make sense, and what about it "
        "cannot be settled?"
    )
    skill_path: str = "panoscope/skills/de-interpretation/SKILL.md"

    # ----------------------------------------------------------------------- #
    def interpret(self, ev: NormalisedEvidence, reference: Any) -> Traced[Finding]:
        """One unit's DE evidence, audited by the four checks. **No band, ever.**

        ``verify`` is deterministic: an item is exempt only when observed, no check
        flagged the unit and every check settled -- never true from a result table
        alone, since the replicate check cannot settle.
        """
        t = Trace()
        audit: DEAudit = reference if isinstance(reference, DEAudit) else DEAudit()

        results = tuple(t.absorb(fn(ev, audit)) for fn in (
            panel_scoping, replicate_confound, composition_vs_expression,
            cross_method_coherence,
        ))
        confounded = any(r.state is not CheckState.OK for r in results)

        items = tuple(_finding_item(i, confounded, results) for i in ev.items)
        return t.done(Finding(
            unit=ev.unit,
            unit_kind=ev.unit_kind,
            task_kind=TaskKind.COMPARISON,
            items=items,
            summary=_summary(ev, items, results),
        ))

    def rival_test(self, ev: NormalisedEvidence, alternative: str,
                   reference: Any) -> Traced[RivalResult]:
        """For a COMPARISON the rival is a CONFOUND, and it comes from the data (§9.6).

        ``alternative`` is matched against the registered check ids; an unrecognised
        one is reported as unasked. ``separated_by`` stays empty even on OK: a design
        confound is ruled out by the design, not an evidence item; the how is on the
        :class:`CheckResult`.
        """
        t = Trace()
        audit: DEAudit = reference if isinstance(reference, DEAudit) else DEAudit()
        fn = {
            PANEL_SCOPING: panel_scoping,
            REPLICATE_CONFOUND: replicate_confound,
            COMPOSITION: composition_vs_expression,
            COHERENCE: cross_method_coherence,
        }.get(alternative)

        if fn is None:
            return t.done(RivalResult(
                alternative=alternative,
                undecidable=("rivals for a comparison come from the data, not from "
                             "reference data (ARCHITECTURE 9.6); no confound check "
                             f"is registered as {alternative!r}. Registered: "
                             f"{', '.join(c.id for c in CHECKS)}",),
            ))

        r = t.absorb(fn(ev, audit))
        return t.done(RivalResult(
            alternative=alternative,
            undecidable=() if r.state is CheckState.OK
            else (r.headline,) + r.missing,
        ))

    def checks(self) -> tuple[Check, ...]:
        """§9.7's four, declared -- so one that could not be settled stays visible
        next to the ones that fired."""
        return CHECKS

    def uispec(self, method: Any = None) -> UISpec:
        """The review surface this question requires; the statistic column is
        labelled with the matched ``method``'s own statistic (logFC, avg_log2FC).

        A ``ConfoundChecklistHeader`` (no band slot). Four blocks, one per measurement
        state; a ``direction`` column because a down-regulated gene is a result.
        No plots: a ``PlotSpec`` needs a conditionality builder (R4), and the
        fold-change-vs-spatial plot belongs to reconciliation (§13.6).
        """
        stat = method.statistic_name if method is not None else "effect size"
        return UISpec(
            workflow_id=SURFACE_ID,
            title="Contrasts",
            task_kind=TaskKind.COMPARISON,
            unit_kind="cluster",
            item_kind="gene",
            interpretive_question=self.question,
            header=ConfoundChecklistHeader(
                design_template=("{n_samples} sample(s), replicate floor "
                                 f"{REPLICATE_FLOOR}"),
                results_template=("{n_items} gene(s): {n_supported} supported, "
                                  "{n_suggestive} suggestive, {n_not_supported} "
                                  "measured-absent, {n_untestable} untestable"),
                testable_denominator_template=("{n_tested} of {panel_size} panel "
                                               "genes have a result in this unit"),
                scope_line_field="scope_caveat",
                checks=tuple(
                    ConfoundCheck(key=c.id, label=c.id.replace("_", " "),
                                  state_field=f"{c.id}_state",
                                  detail_field=f"{c.id}_detail")
                    for c in CHECKS
                ),
            ),
            columns=(
                ColumnSpec("gene", "Gene", ColumnKind.TEXT),
                ColumnSpec("statistic", stat, ColumnKind.NUMBER,
                           unit=stat, align="right"),
                ColumnSpec("direction", "Direction", ColumnKind.DIRECTION),
                ColumnSpec("detection", "Detected in", ColumnKind.COVERAGE_BAR,
                           unit="pct", align="right"),
                ColumnSpec("status", "Status", ColumnKind.GLYPH),
                ColumnSpec("confounds", "Cannot be settled", ColumnKind.NOTE_CHIP),
                ColumnSpec("citation", "Biology", ColumnKind.CITATION, sortable=False),
            ),
            blocks=(
                BlockSpec("observed", "REPORTED - {count} with a measurement",
                          statuses=(MeasuredState.OBSERVED,), sort_by="statistic"),
                BlockSpec("tested_absent",
                          "TESTED, NOT DIFFERENT - {count} on panel, not reported here",
                          statuses=(MeasuredState.TESTED_ABSENT,),
                          empty_text="nothing was tested and found unchanged"),
                BlockSpec("underpowered", "TOO LITTLE SIGNAL - {count}",
                          statuses=(MeasuredState.UNDERPOWERED,),
                          empty_text="every on-panel gene cleared the floor"),
                BlockSpec("never_measured", "NEVER MEASURED - {count} off panel",
                          statuses=(MeasuredState.NEVER_MEASURED,),
                          empty_text="every gene asked about is on the panel"),
            ),
            coverage_ribbon=CoverageRibbonSpec(),
            pin_kinds=("gene",),
            csv_columns=("unit", "gene", "statistic", "direction", "tier", "verify",
                         "status", "check", "check_state"),
        )


def _finding_item(item: EvidenceItem, confounded: bool,
                  results: tuple[CheckResult, ...]) -> FindingItem:
    """One evidence item, tiered by the measurement and demoted by the unit's checks.

    ``permitted_tiers`` fixes the tier except SUPPORTED vs SUGGESTIVE, which the
    unit's checks decide, not the size of the number. ``verify`` flags anything
    observed-but-confounded or untestable; a tested absence is not flagged.
    """
    m = item.measured
    if m.carries_number:
        tier = Tier.SUGGESTIVE if confounded else Tier.SUPPORTED
    elif m.absence_is_evidence:
        tier = Tier.NOT_SUPPORTED
    else:
        tier = Tier.UNTESTABLE
    return FindingItem(
        evidence=item,
        tier=tier,
        verify=tier in (Tier.SUGGESTIVE, Tier.UNTESTABLE),
        because=tuple(f"{r.check_id}: {r.state.value}" for r in results
                      if r.state is not CheckState.OK),
    )


def _summary(ev: NormalisedEvidence, items: tuple[FindingItem, ...],
             results: tuple[CheckResult, ...]) -> str:
    """Deterministic prose, built from the items and the checks. NOT narration.

    Renders identically offline: per-item counts as the header (§9.1), and each
    unsettled check with what is missing.
    """
    n = {t: sum(1 for i in items if i.tier is t) for t in Tier}
    head = (f"{ev.unit}: {len(items)} item(s) -- {n[Tier.SUPPORTED]} supported, "
            f"{n[Tier.SUGGESTIVE]} suggestive, {n[Tier.NOT_SUPPORTED]} measured and "
            f"not different, {n[Tier.UNTESTABLE]} untestable. No band: 'does this "
            "result make sense?' has no label space, so a unit-level confidence is "
            "undefined")
    flagged = tuple(r for r in results if r.state is CheckState.FLAGGED)
    unsettled = tuple(r for r in results if not r.settled)
    if flagged:
        head += ". Flagged: " + "; ".join(f"{r.check_id} -- {r.headline}"
                                          for r in flagged)
    if unsettled:
        head += ". Cannot be settled: " + "; ".join(
            f"{r.check_id} -- {r.headline}" for r in unsettled)
        head += ". Missing: " + "; ".join(m for r in unsettled for m in r.missing)
    # A headline may already end in a full stop; one sentence terminator, not two.
    return head.rstrip(".") + "."


#: The registered instance. See the module docstring for why no ``register`` call
#: exists in ``panoscope/workflows/__init__.py`` yet.
WORKFLOW = DEInterpretation()
