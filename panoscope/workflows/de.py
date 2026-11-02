"""The DE workflow: *does this result make sense?* -- and the surface that answers it.

**This workflow does not run differential expression.** It reads someone else's DE
result and audits it. That is not a limitation to apologise for; it is the entire
basis on which the assessment is worth anything, because a module that ran the
analysis would be grading its own homework.

``TaskKind.COMPARISON``, and therefore **no band**. §9.1: there is no label space, so
there is no alternative for the remaining probability mass to sit on and a unit-level
band is *undefined*, not merely unhelpful. ``Finding`` has no field to put one in and
``UISpec.__post_init__`` refuses a ``LabelBandHeader`` on this task kind, so the rule
is structural rather than reviewed. What the header carries instead is the checks and
what blocked them.

**The four computable checks (ARCHITECTURE §9.7).** "Does this result make sense?" is
not rhetorical; it is four questions, declared in :data:`CHECKS` and computed by the
four functions below:

1. :func:`panel_scoping` -- a volcano on 280 genes is not a genome-wide volcano.
   K of N tested, and which canonical genes for the affected process were never
   measured. The panel-absence rule applied to a comparison.
2. :func:`replicate_confound` -- **the one that matters most.** Is the effect driven
   by one replicate or section? With a single biological replicate this returns
   *cannot be settled*, naming what is missing. §9.7 calls that the clearest
   demonstration of the refusal behaviour, and it is the honest output: a
   condition-level claim from one biological replicate is not weakly supported, it is
   unsupportable, and a number quoted beside it would be worse than no number.
3. :func:`composition_vs_expression` -- did the gene change, or did the cell-type
   mixture change? Two different claims that the measurement usually cannot separate.
4. :func:`cross_method_coherence` -- does the DE result agree with the spatial result
   about the same subject? **This delegates to** ``core/reconcile.py``, which already
   joins on ``Subject.key`` and distinguishes a genuine conflict from "could not be
   compared". Reimplementing that lattice gate here would be a second copy of the
   project's whole thesis, and the copy nobody updates is always the one that governs.

**Three states, not two.** :class:`CheckState` has ``CANNOT_BE_SETTLED`` beside
``OK`` and ``FLAGGED`` for the same reason ``measured.py`` has four states and not
two: "we checked and it is fine" and "we could not check" are opposite readings that
a boolean renders identically, and the second is the one the biologist must act on.
Every unsettled result carries ``missing`` -- the named things that would settle it --
because "insufficient information" is not actionable and "a second biological
replicate" is.

**What this module does NOT own.** Not the lattice (``types/measured.py``), not the
join (``core/reconcile.py``), not the effect-size ladder
(``methods/seurat_de.BAND_EDGES`` declares its own, and this module never reads it:
there is no unit band for it to feed). ``interpret`` is composition. If this file
ever computes a statistic, the layering has inverted.

**Not registered yet, deliberately.** ``panoscope/workflows/__init__.py`` has no
``register(de.WORKFLOW)`` line and ``panoscope/methods/__init__.py`` has no
``register(seurat_de.MODULE)``. Both are one line, and they land with the first real
DE artifact -- there is none in this repo, and a workflow registered against nothing
would show up in every listing as a question the study can be asked. Until then
``tests/test_workflow_de.py`` registers the pair inside an isolated-registry fixture
and exercises them against a declared synthetic fixture, which is also what keeps
``tests/test_registries.py``'s "a method cannot register into a workflow nobody
declared" supplied with ``de-interpretation`` as its unregistered example.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from panoscope.core.reconcile import Agreement, Relationship, Side, reconcile
from panoscope.types.claim import Authority
from panoscope.types.contracts import Check, RivalResult
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
# The replicate frame, named once. §9.5: "the design's replicate counts are the
# condition frame", and a count below the floor makes the UNIT underpowered, not
# the items. The floor and the Coverage label live HERE and not in the method,
# because "you cannot separate condition from batch with one replicate" is a
# property of the QUESTION -- every DE method registering here inherits it --
# and not of Seurat. ``methods/seurat_de.py`` imports both and writes the
# Coverage at load time, where the adapter's sample count is first-hand.
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


class CheckState(StrEnum):
    """Three states, and the third is the point.

    ``CANNOT_BE_SETTLED`` is not a soft ``FLAGGED``. A flagged check has been asked
    and answered against the result; an unsettled one could not be asked at all, and
    the honest output names what is missing rather than letting the reader infer that
    nothing was wrong.
    """

    OK = "ok"
    FLAGGED = "flagged"
    CANNOT_BE_SETTLED = "cannot_be_settled"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One check's answer, with what it read and what it still needs.

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
class DEAudit:
    """Everything the four checks need that a DE result table cannot supply.

    Passed in rather than looked up, because this layer does no IO. Every field is
    *absent by default*, which is why the checks refuse rather than assume: a default
    sample count, a default abundance or an assumed spatial side would each turn a
    refusal into a quiet answer.

    ``world`` is here for the reason ``MarkerReference.world`` is, and under the same
    protest: deciding whether a canonical gene was *never measured* or *tested and
    absent* needs frame membership, and ``NormalisedEvidence`` carries frames only as
    digests. Deriving the frames from what the result reported would be wrong in the
    one direction that matters -- every unreported gene would come back
    NEVER_MEASURED, collapsing the productive half of the lattice into "we don't
    know".
    """

    world: MeasuredWorld = MeasuredWorld()
    process: str = ""                                # the affected process, named
    canonical: tuple[str, ...] = ()                  # its canonical genes
    # (condition, this unit's share of cells under it). Sorted pairs, never a dict:
    # a dict field on a frozen dataclass is unhashable and its order would leak into
    # the serialised record.
    abundance: tuple[tuple[str, float], ...] = ()
    spatial: Side | None = None                      # the other method, SAME unit
    relationship: Relationship = Relationship.PARTLY_INDEPENDENT


# --------------------------------------------------------------------------- #
# Check 1: panel scoping
# --------------------------------------------------------------------------- #
def panel_scoping(ev: NormalisedEvidence, audit: DEAudit) -> Traced[CheckResult]:
    """A DE result on a targeted panel is not a transcriptome-wide result.

    States K of N -- how many panel genes this unit has a result for, against the
    panel the apparatus actually had -- and then the half that needs a curated list
    to even ask: which canonical genes for the affected process were never on the
    panel. Those are named, because "some markers were missing" is not something a
    biologist can act on and "CD3E and PTPRC were never on this panel" is.

    With no gene frame declared the check refuses. A denominator guessed here is a
    panel size asserted rather than read, and panel size is dynamic across studies.
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

    Reads the unit's own capability verdict, which is where the design already lives:
    ``NormalisedEvidence.unit_measured`` carries "too few replicates" because that is
    a property of the unit and not of any gene in it, and ``methods/seurat_de.py``
    writes the :class:`Coverage` at load time from the adapter's first-hand sample
    count. Nothing is re-derived here and nothing is assumed.

    **This check cannot return OK from a result table alone, and says so.** Even with
    enough replicates, a FindAllMarkers table reports one number per gene for the
    whole unit; which replicate drives it is not in the file. The honest output is
    *cannot be settled*, with the per-sample results named as what is missing --
    ``core/concordance.py`` is what computes it once they exist. An OK here would be
    a check that passed because it never ran.
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

    Two different claims, and §9.7 says the measurement usually cannot separate them.
    The lever is this unit's abundance across the arms being compared, which is not
    in a DE table and must be supplied.

    **No threshold is invented.** Any difference in this unit's share of cells
    between the arms means the two claims are confounded -- an expression change and
    a composition change produce the same number, and nothing in this result says
    which. Identical shares are the only case where the per-cell reading stands on
    its own. A tolerance here would be a number that looked like it came from an
    analysis nobody ran.
    """
    t = Trace()
    anchor = (ref(ev.unit_subject),)
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

    The join, the lattice gate and the §9.11 question-sameness mapping all live in
    ``core/reconcile.py``; this function supplies the DE side and reads the outcomes.
    A second implementation of that gate here would be a second place where "the
    other method never measured it" could be mistaken for "the other method
    disagrees", which is the one mistake the four states exist to make impossible.

    The two workflows ask different questions, so ``reconcile`` can only return
    COHERENT, TENSION or NOT_COMPARABLE -- never AGREE. A spatial coefficient does
    not corroborate a fold change; it can sit with it or pull against it.
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
                authority=Authority.MEASUREMENT, items=ev.items, world=audit.world)
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

        ``Finding`` rather than ``Assignment`` is the type-level form of the no-band
        rule: a comparison has no label space, so there is no shape in which this
        could carry one.

        ``verify`` is deterministic and is never offered to a model. An item is
        exempt only when it was observed *and* no check flagged this unit *and* every
        check was settled -- which, from a result table alone, never happens, because
        the replicate check cannot be settled from one. That is the intended shape: a
        condition-level reading of a single-replicate DE result is re-check-this by
        construction, and the summary says so rather than leaving the biologist to
        infer it from a small number.
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

        No reference list is consulted and none would help: the alternative account
        of a fold change is "one replicate drove it" or "the mixture changed", and
        only the design and the data separate those. ``alternative`` is therefore
        matched against the registered check ids rather than against a curated rival
        space, and an unrecognised one is reported as unasked rather than settled.

        ``separated_by`` stays empty even when a check returns OK, and that gap is
        worth naming rather than papering over: ``RivalResult.separated_by`` holds
        ``EvidenceItem``\\s, but what rules out a design confound is the *design* -- a
        second replicate, an unchanged abundance -- and there is no evidence item to
        point at. An empty ``undecidable`` is the signal that the check settled it;
        the sentence saying how is on the :class:`CheckResult`.
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

    def uispec(self) -> UISpec:
        """The review surface this question requires.

        A ``ConfoundChecklistHeader``, which has no band slot: on a COMPARISON a
        unit-level band is a category error, and ``UISpec.__post_init__`` raises
        ``DegradedUpward`` on the other combination, so the rule is enforced before a
        pixel is drawn rather than reviewed afterwards.

        Four blocks, one per measurement state, so a never-measured gene is a row
        that says so. The ``direction`` column exists because a comparison's answer
        is a ranked set *with direction*, and a down-regulated gene is a result, not
        a weak one.

        **No plots are declared, for two separate reasons.** The general one is
        marker's: a ``PlotSpec`` is a promise to draw, and R4 makes it conditional on
        a conditionality builder that can state the cell count and the artifact sha;
        those belong to the view layer and have not landed. The specific one is
        §13.6 -- the fold-change-against-spatial-coefficient plot "belongs to
        reconciliation, not to either workflow", so declaring it here would take a
        figure away from the layer that owns it. A volcano, when it lands, refuses to
        draw without the panel denominator the scope line carries.
        """
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
                ColumnSpec("statistic", "avg_log2FC", ColumnKind.NUMBER,
                           unit="avg_log2FC", align="right"),
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

    The tier is almost entirely a function of the measurement -- ``permitted_tiers``
    allows only SUPPORTED/SUGGESTIVE where a number exists, only NOT_SUPPORTED on a
    tested absence, only UNTESTABLE otherwise -- and the one judgement left is
    SUPPORTED versus SUGGESTIVE. Here that judgement is made by the unit's checks and
    not by the size of the number, which is the right shape for a COMPARISON: a large
    fold change under a single-replicate design is not better supported than a small
    one, it is unsupportable in the same way.

    ``verify`` follows: re-check anything observed-but-confounded and anything that
    could not be tested. A tested absence is NOT flagged -- it was asked and it was
    answered, and flagging it would make a real refusal indistinguishable from an
    answer the biologist already has.
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

    It must render identically with the model offline, so it carries no adjective the
    counts do not license. Per-item counts are the header for a task kind with no
    band (§9.1), and the unsettled checks are spelled out with what is missing --
    because a refusal that does not name what would settle it is indistinguishable
    from the tool being broken.
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
