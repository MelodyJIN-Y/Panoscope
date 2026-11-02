"""Why a value came out the way it did: ``Traced[T]`` and the local-only ``Trace``.

Eight signatures in the pre-refactor code returned ``tuple[R, tuple[Decision, ...]]``
by hand. This is that, typed -- and the reason it is worth a module is the *direction*
rule it enforces:

**Decisions flow out, never in.** A ``Trace`` is constructed INSIDE a function and
returned as frozen data. It is never a parameter. That is what keeps ``core/``
un-injectable: there is no seam a caller could reach through to supply a recorder, so
a function is pure in its declared arguments and ``test_record_not_generated`` does
not have to rest on nobody exploiting one.

Three alternatives, each rejected for a stated reason:

- a bare ``tuple[R, tuple[Decision, ...]]`` return -- untyped, and unreadable once
  calls nest;
- a recorder passed **in** -- invites exactly the caller-injected recorder above;
- a ``ContextVar`` -- an *ambient* seam, which is the same injection point wearing a
  disguise, and the one hardest to see in a review.

``Decision.rule`` is a REGISTERED id rather than prose. The code this replaces decided
whether to draw a UI badge by string-matching the exact English phrase "localizes
better with another cluster" (``ui/evidence_table.py:53``), so a copy edit in the
analysis layer silently stopped a badge rendering -- and the test asserted the same
literal, so it stayed green. Ids are what tests and the UI match on; English lives in
``RULES`` and in ``ui``'s copy tables, where editing it is harmless.

Determinism: decision order is **code order** -- append, splice on ``absorb``, never
sort. ``because`` strings are built with ``fmt_num`` at fixed precision and carry no
clock, because they are hashed into the golden record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from panoscope.types.subject import Subject

T = TypeVar("T")
X = TypeVar("X")


@dataclass(frozen=True, slots=True)
class Rule:
    """A decision the system is allowed to make, declared once.

    ``step`` lives here rather than on ``Decision`` because it is a property of the
    rule, not of the occasion: "coverage_floor" is always part of measuring. Putting
    it on each Decision let two emitters of one rule disagree about which step they
    were in, which made grouping the record by step unreliable.
    """

    id: str
    step: str   # "measure" | "confidence" | ...
    what: str   # human-readable; NEVER matched on


def _rules(*rules: Rule) -> dict[str, Rule]:
    return {r.id: r for r in rules}


# --------------------------------------------------------------------------- #
# The registry. Each entry is a decision some core function actually emits --
# this is a transcription of today's behaviour, not a wish list.
# --------------------------------------------------------------------------- #
RULES: dict[str, Rule] = _rules(
    # -- step: measure (panoscope.types.measured.resolve) --
    Rule("measure:never_measured", "measure",
         "a constituent lies outside the capability frame for its kind"),
    Rule("measure:tested_absent", "measure",
         "in frame and tested, but not reported -- absence IS evidence here"),
    Rule("measure:observed", "measure", "reported with a statistic"),
    Rule("measure:coverage_floor", "measure",
         "K of N fell below a declared floor; demoted to underpowered"),
    # -- step: confidence (agent.verdict._compute_band) --
    Rule("band:no_canonical_driver", "confidence",
         "no canonical marker supports the call"),
    Rule("band:driver_nosig", "confidence", "the driving marker is NoSig"),
    Rule("band:from_coef", "confidence", "base band read off the driver's glm_coef"),
    Rule("band:promote_pearson_high", "confidence",
         "strong spatial specificity on the driver; +1 band"),
    Rule("band:demote_pearson_low", "confidence",
         "weak spatial specificity on the driver; -1 band"),
    Rule("band:demote_localises_elsewhere", "confidence",
         "driver localises better with another cluster; -1 band"),
    Rule("band:demote_not_unique", "confidence",
         "driver's spatial pattern is not unique; -1 band"),
    Rule("band:fragile_cap", "confidence",
         "fragile cluster (<=2 markers) cannot read above Medium-High"),
    Rule("band:fragile_floor", "confidence",
         "fragile cluster with a canonical driver stays at or above Medium"),
    # -- step: rival (panoscope.core.rival.rival_test) --
    # Three ids rather than one with three outcomes, for the reason measure's four
    # states have four: the UI and the grounding tests match on the id, and a single
    # "rival:verdict" would make "the markers rule it out" indistinguishable from
    # "the markers cannot speak to it" to everything downstream.
    Rule("rival:argued_against", "rival",
         "the rival's canonical markers were in frame, tested, and not reported here "
         "-- a measured absence, which is the half of the lattice that argues"),
    Rule("rival:ambiguous", "rival",
         "the rival's canonical markers were also observed here; genuine rival signal"),
    Rule("rival:undecidable", "rival",
         "too few of the rival's canonical markers were in frame; the measurement "
         "cannot separate the two accounts"),
    # -- step: claims (panoscope.claims.evaluability, panoscope.claims.store) --
    # The capability half of ARCHITECTURE §7.5's split. These say what the
    # INSTRUMENT could do about a claim; they are never a judgement of the claim.
    Rule("claims:out_of_scope", "claims",
         "the claim does not provably apply in this context; the scope gate is "
         "fail-closed, so it is withheld WITH this reason rather than dropped"),
    Rule("claims:evaluable", "claims",
         "every entity the claim rests on is admitted by every frame for its kind"),
    Rule("claims:never_measured", "claims",
         "an entity the claim rests on lies outside a capability frame -- the "
         "panel-absence rule; this is a fact about the instrument, not the claim"),
    Rule("claims:undetectable", "claims",
         "an entity the claim rests on is in frame but yields no usable signal; "
         "the detectability frame refused it (§7.5, the one gate that varies)"),
    Rule("claims:no_frame", "claims",
         "no capability frame was declared for an entity kind the claim rests on"),
    # -- step: transfer (panoscope.claims.transfer) --
    Rule("transfer:no_verdict", "transfer",
         "database-curated and literature claims get a PROVENANCE STAMP, never a "
         "verdict: §7.4 measured that verdict at 0.000 bits over 280 edges"),
    Rule("transfer:transferable", "transfer",
         "every condition the claim was formed under matches this context"),
    Rule("transfer:delta", "transfer",
         "a named condition differs between formation and use -- reported as the "
         "delta itself, never as a verdict distribution"),
    Rule("transfer:cannot_be_settled", "transfer",
         "a condition is UNKNOWN on one side, so it can neither match nor differ; "
         "an unknown never silently applies"),
    # -- step: reference (panoscope.reference.graph.lookup / .sources.build) --
    # Three lookup ids rather than one, for the reason measure has four: "read it and
    # it is not there" and "could not read it at all" render as the same sentence to
    # a reviewer unless the id keeps them apart, and the second is a failed download
    # reported as knowledge. None of the three ever argues -- reference silence is
    # coverage, not a negative result (Lookup.absence_argues is constantly False).
    Rule("reference:in_reference", "reference",
         "the consulted sources hold a node for this entity"),
    Rule("reference:not_in_reference", "reference",
         "every source was read and none mentions it -- NOT in reference data, which "
         "says nothing about whether it exists or occurs here"),
    Rule("reference:no_source", "reference",
         "no source could be read; reference data was not consulted, so its silence "
         "carries no information at all"),
    Rule("reference:source_read", "reference",
         "a registered source was read, with its release and licence"),
    Rule("reference:source_unavailable", "reference",
         "a registered source could not be read -- never fetched, or offline; the "
         "graph records it rather than returning an empty one that reads as "
         "'nothing exists'"),
    # -- step: reconcile (panoscope.core.reconcile.reconcile) --
    # Five outcome ids rather than one "reconcile:verdict" with five outcomes, for
    # the reason rival has three: the §9.11 table is the point, and a single id would
    # make "two methods agree" indistinguishable from "they are coherent but are not
    # answering the same question" to every consumer that branches on the record.
    Rule("reconcile:agree", "reconcile",
         "two methods asking the SAME question read one subject the same way"),
    Rule("reconcile:disagree", "reconcile",
         "same question, incompatible readings -- both sides were capable of seeing "
         "it, so this is a real conflict and not a silence"),
    Rule("reconcile:coherent", "reconcile",
         "different questions whose answers sit together; never 'agree', because "
         "neither is a second witness to the other's question (§9.11)"),
    Rule("reconcile:tension", "reconcile",
         "different questions whose answers pull against each other"),
    Rule("reconcile:not_comparable", "reconcile",
         "one side never measured the subject or was underpowered for it, so its "
         "silence is not a negative observation -- the join reports WHICH side and "
         "WHICH frame rather than calling it a disagreement"),
    Rule("reconcile:reference_cannot_corroborate", "reconcile",
         "a side carries reference authority, not MEASUREMENT; reference data may "
         "select and order evidence but may never supply a value or count as "
         "agreement, so every pair is refused before any number is read"),
    Rule("reconcile:derived_is_not_corroboration", "reconcile",
         "one result was computed from the other, so their agreeing is the same "
         "result twice; the outcome stands but no pair counts as a second witness"),
    # -- step: adjacency (panoscope.core.adjacency.adjacency_check) --
    Rule("adjacency:bleed_candidate", "adjacency",
         "the marker also localises in a unit interdigitated with this one -- a "
         "segmentation-bleed candidate. Display flag, never a band input (§9.9)"),
    Rule("adjacency:specificity", "adjacency",
         "the marker also localises in a unit that sits elsewhere -- a genuine "
         "specificity problem. Display flag, never a band input (§9.9)"),
    # -- step: concordance (panoscope.core.concordance.concordance) --
    Rule("concordance:consistent", "concordance",
         "the replicates read this subject the same way. Agreement NEVER raises "
         "confidence: replicates are not independent witnesses (§9.10)"),
    Rule("concordance:divergent", "concordance",
         "the replicates read this subject differently -- batch effect, section "
         "quality or real biology, and the check does not claim which"),
    Rule("concordance:not_comparable", "concordance",
         "a sample never measured the subject, was underpowered for it, or emitted "
         "no item at all; replicates cannot diverge about something one could not see"),
    # -- step: enrichment (panoscope.workflows.enrichment.interpret) --
    # Four ids rather than one "enrichment:tier", for the reason measure has four
    # states: "we measured it and it is not enriched here" and "the panel carried too
    # little of it to ask" are different answers, and a single id would let the UI and
    # the grounding tests collapse them back into "unknown". None of these is a band:
    # a characterisation has no label space, so the unit-level number does not exist.
    Rule("enrichment:enriched", "enrichment",
         "the set's q clears the workflow's enriched bar; verify is set only where "
         "the panel footprint is thin, which conditions the call and never tiers it"),
    Rule("enrichment:suggestive", "enrichment",
         "in frame and reported, but q sits between the enriched bar and the "
         "method's own reporting bar -- real but weak, and it carries verify"),
    Rule("enrichment:not_enriched", "enrichment",
         "the set was scored against this unit and did not clear the method's "
         "reporting bar -- a measured absence, which is the half that argues"),
    Rule("enrichment:untestable", "enrichment",
         "too little of the set was inside the capability frames, or it was never "
         "scored at all; its absence argues nothing and its statistic is withheld"),
    # -- step: confidence (panoscope.methods.seurat_de.SeuratDE.rubric) --
    # A second method needs its OWN ladder ids, not jazzPanda's. "band:from_coef"
    # says "read off the driver's glm_coef" and would be a lie over an avg_log2FC;
    # worse, one shared id would make the two methods' decisions indistinguishable
    # in the record, which is exactly the method-blindness ARCHITECTURE 9.4 forbids.
    # Neither is ever a unit band: a COMPARISON has no label space (9.1), so these
    # label a per-item effect size and nothing reads them for a unit verdict.
    Rule("band:from_effect_size", "confidence",
         "per-item effect size read off a method's own effect-size statistic, "
         "through that method's own declared edges -- never a unit-level band"),
    Rule("band:no_effect_reported", "confidence",
         "no item carried an effect-size statistic, so the floor label is recorded "
         "rather than omitted: a rubric whose answer simply did not appear is "
         "indistinguishable from one that never ran"),
    # -- step: de (panoscope.workflows.de, ARCHITECTURE 9.7's four checks) --
    # Three outcomes per check rather than one id with a state field, for the reason
    # measure has four states and rival has three: the UI and the grounding tests
    # branch on the id, and "we checked and it is fine" must never render the same
    # as "we could not check". CANNOT_BE_SETTLED is the productive outcome here --
    # it is the refusal 9.7 names as the clearest demonstration of the behaviour --
    # so it gets its own id everywhere it can occur, and every one of them carries
    # what is missing rather than a bare refusal.
    Rule("de:panel_scoped", "de",
         "a DE result on a targeted panel is not a transcriptome-wide result: K of N "
         "stated, and the canonical genes for the affected process that were never "
         "measured named, because their absence from it argues nothing"),
    Rule("de:panel_frame_absent", "de",
         "no capability frame for genes was declared, so the scope of the result "
         "cannot be stated; a denominator guessed here would be a panel size "
         "asserted rather than read"),
    Rule("de:replicate_confound", "de",
         "the design cannot separate condition from batch or section -- one "
         "biological replicate, or none declared -- so the condition-level claim "
         "CANNOT BE SETTLED and the refusal names the replicates that are missing"),
    Rule("de:replicate_not_in_result", "de",
         "enough replicates exist, but the result reports one number per gene for "
         "the whole unit, so which replicate drives it is not in the table; "
         "core/concordance.py settles it once per-sample results exist"),
    Rule("de:composition_confounded", "de",
         "this unit's share of cells differs across the arms, so a gene can read as "
         "up because the cells expressing it became more numerous; the measurement "
         "does not separate a composition change from an expression change"),
    Rule("de:composition_separated", "de",
         "this unit's share of cells is unchanged across the arms, so a change here "
         "is not explained by the mixture"),
    Rule("de:composition_not_supplied", "de",
         "no cluster abundance was supplied for the arms being compared, and a DE "
         "table carries none; the question was not asked rather than answered"),
    Rule("de:coherence_tension", "de",
         "the DE result and the spatial result pull against each other about one "
         "subject. Different questions, so never a disagreement about one fact "
         "(9.11) -- two measurements that do not sit together"),
    Rule("de:coherence_coherent", "de",
         "the two results sit together. Coherent, never AGREED: they answer "
         "different questions (9.11), and coherence raises no confidence -- there "
         "is no band on either side of this join"),
    Rule("de:coherence_not_comparable", "de",
         "no second method's result was supplied, or one side never measured the "
         "subjects the other raised; silence that is not a negative observation "
         "cannot be a disagreement"),
)


class UnregisteredRule(KeyError):
    """Raised for a decision whose rule id is not in ``RULES``. Never guessed at."""


@dataclass(frozen=True, slots=True)
class Decision:
    """One recorded step. Frozen, hashable, clock-free, and hashed into the record."""

    rule: str
    outcome: str
    because: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()   # ref() strings: what this decision read

    @property
    def step(self) -> str:
        return RULES[self.rule].step

    @property
    def what(self) -> str:
        return RULES[self.rule].what

    def __str__(self) -> str:
        return f"{self.rule} -> {self.outcome}"


@dataclass(frozen=True, slots=True)
class Traced(Generic[T]):
    """A value with the decisions that produced it. The return type of every core fn."""

    value: T
    decisions: tuple[Decision, ...] = ()

    def steps(self, step: str) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.step == step)


class Trace:
    """A LOCAL, mutable builder. Construct inside a function; return ``done()``.

    Never accept one as a parameter -- ``test_no_core_function_accepts_a_trace``
    enforces that, and it is the whole point of the design.
    """

    __slots__ = ("_decisions",)

    def __init__(self) -> None:
        self._decisions: list[Decision] = []

    def note(self, rule: str, outcome: str,
             because: tuple[str, ...] = (), refs: tuple[str, ...] = ()) -> None:
        """Record one decision. Fails loudly on an unregistered rule id.

        Unregistered means the record would carry a decision nothing can explain,
        render or test against -- which is how prose crept back in last time.
        """
        if rule not in RULES:
            raise UnregisteredRule(
                f"unregistered decision rule {rule!r}; register it in RULES first. "
                f"Known: {sorted(RULES)}"
            )
        self._decisions.append(
            Decision(rule=rule, outcome=outcome, because=tuple(because), refs=tuple(refs))
        )

    def absorb(self, t: Traced[X]) -> X:
        """Splice a callee's decisions in call order and unwrap its value."""
        self._decisions.extend(t.decisions)
        return t.value

    def done(self, value: T) -> Traced[T]:
        return Traced(value=value, decisions=tuple(self._decisions))


# --------------------------------------------------------------------------- #
# Deterministic formatting. ONE home, because these strings are hashed.
# --------------------------------------------------------------------------- #
_PRECISION = 4   # matches the provenance strings already emitted (tools.py:193)


def fmt_num(x: float | int | None) -> str:
    """Fixed-precision, locale-free, platform-stable. The only way to put a number
    into a ``because`` or a ``ref``.

    ``None`` renders as "na" rather than "None" so a missing statistic reads the same
    whether it was never measured or tested-absent -- the distinction belongs in the
    Measured state, not in a formatting accident.
    """
    if x is None:
        return "na"
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    f = float(x)
    if f != f:
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf" if f > 0 else "-inf"
    s = f"{f:.{_PRECISION}f}"
    return "0." + "0" * _PRECISION if s == "-" + "0." + "0" * _PRECISION else s


def ref(anchor: Subject | str, stat: str = "", value: float | int | None = None) -> str:
    """A click-back anchor: what a decision read, in a form that resolves.

    Anchored on ``Subject.key`` because that is the one identity the rest of the
    system joins on. A prettier anchor (``gene:ERBB2``) would not resolve back to the
    subject for any shape with more than one role, so the interface could not click
    through from a ligand-receptor sentence to its evidence.
    """
    head = anchor.key if isinstance(anchor, Subject) else anchor
    if not stat:
        return head
    return f"{head}#{stat}={fmt_num(value)}"
