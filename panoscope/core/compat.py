"""The v1 on-disk view, projected from v2 outputs. One adapter, nothing rewritten.

``IMPLEMENTATION.md`` §10, *On-disk compatibility*: the committed per-unit JSONs are
never mutated; v2 writes its own records and the v1 files become a **generated
compatibility view** through one adapter, so ``ui/`` and ``pipeline/store.py`` are
untouched during the migration. This is that adapter, for ``TaskKind.ASSIGNMENT``.

**The gate is the reason this module is small.** §10: "rebuild each record from the
same hashed inputs, project it back, and compare field by field." The projection is
therefore written as a transcription of ``agent/verdict.py``'s composition, not as an
improvement on it -- every template string, every ``:.2f``, every join character here
is pinned by ``tests/test_core_compat.py`` against the nine committed JSONs.

**Why prose lives here and nowhere else.** ``evidence.py`` and ``trace.py`` moved the
analysis layer off English onto registered ids, because the interface used to decide
whether to draw a badge by matching the exact sentence "localizes better with another
cluster" (``ui/evidence_table.py:74``), so a copy edit silently stopped the badge and
the test, asserting the same literal, stayed green. The v1 bytes still contain those
sentences and the untouched UI still matches them, so *something* has to render ids
back into them. That something is a compat layer -- one reviewable translation table,
at the boundary, feeding a file format that is on its way out. Putting it anywhere
inside the analysis layer is what the ids exist to prevent.

**A dict out, not a ``ClusterVerdict``.** ``core/`` does not import ``agent/`` (the
same layering ``export.py`` keeps by duck-typing its rows), and the thing §10 compares
is the *on-disk shape*, which is the dict. ``pipeline/serialize.verdict_from_dict``
rebuilds the dataclass from exactly this shape if a caller wants one.

**No decisions, for ``export.py``'s reason.** Every public function returns
``Traced[...]`` with an empty decision tuple. A projection makes no judgement -- it
re-spells an interpretation that has already been made, and a ``note`` emitted here
would put a rule in the record that explains a format choice as if it were an
inference.

**What this does NOT do.** It does not project the enrichment workflow. There is no
v2 ``Finding`` producer for enrichment yet (that is the method refactor), and
inventing one here to project back would mean writing the thing the projection is
supposed to be checked against. When it lands, note that
``ClusterEnrichment.confidence`` stays: §10 deletes it *with* the method refactor,
deliberately sequenced, because it is persisted in a committed CSV and rendered at
``ui/enrichment_table.py:174``. ``export.py``'s docstring records the same decision
for that table's last two columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from panoscope.core import export
from panoscope.core.verdict import (
    FLAG_LOCALISES_ELSEWHERE,
    FLAG_LOW_PEARSON,
    FLAG_NOT_UNIQUE,
    Verdict,
)
from panoscope.core.verdict import _drivers as _eligible_drivers
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import MeasuredState
from panoscope.types.outputs import TaskKind
from panoscope.types.trace import Traced

# --------------------------------------------------------------------------- #
# The allowlist. §10: "adding to it must be a deliberate diff, not a green test."
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Dropped:
    """One v1 field the projection deliberately does not emit, and why.

    ``reason`` has no default on purpose: an entry cannot be added without writing
    one, and both halves show up in the diff of this file. ``path`` is the literal
    path the comparison walker produces -- dotted, with ``[]`` standing for "every
    element of this list" -- so there is no pattern syntax a widening entry could
    hide behind. An entry here is only half the change: the projection must also stop
    emitting the field, which is a second deliberate edit in this module.
    """

    path: str
    reason: str


V1_DROPPED: tuple[Dropped, ...] = (
    Dropped(
        "evidence[].within_cluster_pctile",
        "Vestigial. agent/verdict.py computes it from the glm_coef RANK "
        "((n-1-rank)/(n-1)) and its own docstring says the band does not use it: "
        "'confidence is glm_coef DIRECT ... There is NO within-cluster percentile.' "
        "v2 carries no rank and no percentile anywhere, so emitting one would mean "
        "re-implementing a deleted rule inside the compat layer. Its only reader, "
        "ui/format.pct_fmt, has zero call sites (it appears in __all__ and nowhere "
        "else), so nothing on screen changes. §10 names it: 'a vestigial percentile'.",
    ),
    Dropped(
        "opening.driving_markers[].within_cluster_pctile",
        "The same field at a second path. OpeningInterpretation.driving_markers "
        "holds copies of the canonical supporters from `evidence`, so one deletion "
        "surfaces twice; the two are listed separately rather than wildcarded "
        "because a pattern is exactly the shape a later widening would take.",
    ),
)

DROPPED_PATHS: frozenset[str] = frozenset(d.path for d in V1_DROPPED)


# --------------------------------------------------------------------------- #
# The second allowlist: whole ITEMS v1 has no shape for. Same discipline as
# V1_DROPPED -- a reason written longhand, an entry that only takes effect if the
# projection also stops emitting the item, and a two-sided test.
# --------------------------------------------------------------------------- #
PANEL_FRAME_PREFIX: str = "panel:"
"""The naming convention a platform adapter uses for its measured-set frame.

``platforms/xenium.py`` declares two gene frames, ``panel:<study>`` and
``detect:<study>``, and ``_state_for`` names the one that REFUSED an entity. Both
refusals are ``NEVER_MEASURED``, and v1's ``OffPanelNote`` can only say one of the
two things: its message is literally "is off-panel (never measured)". A gene that is
*on* the panel and produced no usable spatial signal is not off-panel, and writing it
into that list would be the false sentence -- so the prefix is matched here, at the
translation boundary, rather than a second status field being invented in ``types/``.
"""

V1_EVIDENCE: str = "evidence"
V1_OFFPANEL: str = "offpanel_notes"
V1_OMIT_TESTED_ABSENT: str = "tested_absent"
V1_OMIT_UNDETECTABLE: str = "never_measured_outside_the_panel_frame"


@dataclass(frozen=True, slots=True)
class Unrepresentable:
    """One class of v2 item the v1 record cannot hold, and why it is omitted.

    ``bucket`` is the id :func:`_v1_bucket` returns, so an entry cannot be written for
    a class the classifier does not produce. ``reason`` has no default, for
    ``Dropped``'s reason: adding an entry costs a sentence in this file's diff.
    """

    bucket: str
    reason: str


V1_UNREPRESENTABLE: tuple[Unrepresentable, ...] = (
    Unrepresentable(
        V1_OMIT_TESTED_ABSENT,
        "v1 has exactly two buckets -- an `evidence` row, which requires a float "
        "glm_coef and three float specificity columns, or an `offpanel_notes` entry, "
        "whose message asserts the gene is off-panel. A tested-absent gene is neither: "
        "it was on the panel, it was tested, and it peaked somewhere else. Writing it "
        "as an evidence row needs a number that does not exist; writing it as an "
        "off-panel note states the opposite of what was measured. v1 predates the "
        "lattice and genuinely cannot express the state, so the v1 view says NOTHING "
        "about these genes -- which is the only option that cannot mislead, because "
        "silence is not a claim while an OffPanelNote is. The state itself is not "
        "lost: every one of these items is carried, with its measure:tested_absent "
        "decision and the frame that admitted it, in the v2 InterpretationRecord "
        "written beside the view at records/assignment/<unit>.json (§10). The count "
        "is reported by pipeline/stages/verdicts.py at the moment the view is "
        "generated, so the omission is announced rather than silent.",
    ),
    Unrepresentable(
        V1_OMIT_UNDETECTABLE,
        "NEVER_MEASURED, but refused by a frame that is not the panel -- in the demo, "
        "the 49 of 280 panel genes that yield no usable spatial signal, refused by "
        "detect:<study>. v1's OffPanelNote is the only shape available and its message "
        "says 'is off-panel (never measured)', which is false of a gene that IS on the "
        "panel: 'order a different panel' and 'this section gave no signal' are "
        "different facts to a biologist and v1 has one word for both. Omitted for the "
        "same reason as above, and recorded in the same place -- the v2 record names "
        "detect:<study> as the refusing frame on every one of them.",
    ),
)

UNREPRESENTABLE_BUCKETS: frozenset[str] = frozenset(
    u.bucket for u in V1_UNREPRESENTABLE
)


class NoV1Shape(ValueError):
    """Raised for a v2 item the v1 record has no shape for.

    Fails closed rather than skipping the item. v1 ``MarkerEvidence`` requires a
    ``float`` glm_coef and three ``float`` specificity columns, so a tested-absent or
    underpowered item cannot be written as one. Dropping it quietly would leave the
    biologist a record that reads as though the measurement never happened -- the
    exact confusion ``measured.py`` exists to prevent, one layer down.

    Two item classes are now omitted rather than raised, each with an entry in
    :data:`V1_UNREPRESENTABLE` and a counted announcement at generation time. Every
    other class -- ``UNDERPOWERED`` today -- still raises, because an omission nobody
    has written a reason for is exactly the silent skip this failure mode is about.
    """


# --------------------------------------------------------------------------- #
# v1 constants and prose, transcribed. Pinned by tests/test_core_compat.py.
# --------------------------------------------------------------------------- #
SCORE_MAP: dict[str, float] = {        # agent/config.SCORE_MAP; drift test in the suite
    "Very High": 0.95,
    "High": 0.85,
    "Medium-High": 0.70,
    "Medium": 0.55,
    "Low": 0.30,
}

EVIDENCE_SOURCE: str = "jazzpanda:top_result"
OFFPANEL_SOURCE: str = "panel:absence"

# Registered flag id -> the v1 caveat sentence, in agent/verdict._caveats_for's
# APPEND order. Emitting in this fixed order rather than in the item's flag order
# makes the bytes independent of how a method happened to assemble its flags.
CAVEAT_FOR: tuple[tuple[str, str], ...] = (
    (FLAG_LOCALISES_ELSEWHERE, "localizes better with another cluster"),
    (FLAG_NOT_UNIQUE, "spatial pattern not unique"),
    (FLAG_LOW_PEARSON, "low spatial specificity"),
)

_PEARSON: str = "pearson"


def _corroborator(item: EvidenceItem, name: str) -> float | None:
    return next((v for k, v in item.corroborating if k == name), None)


def _demotion_sentence(rule: str, driver: EvidenceItem | None) -> str:
    """One registered band rule id, in the English ``agent/verdict.py`` wrote to disk.

    The two pearson rules interpolate the driver's own pearson, which is why this is
    a function and not a dict.
    """
    plain = {
        "band:no_canonical_driver": "no canonical marker supports the call",
        "band:driver_nosig": "driving marker is NoSig",
        "band:demote_localises_elsewhere":
            "driver localizes better with another cluster -> -1 band",
        "band:demote_not_unique": "driver spatial pattern not unique -> -1 band",
        "band:fragile_cap": "fragile cluster (<=2 markers): capped at Medium-High",
        "band:fragile_floor": "fragile cluster with canonical driver: floored at Medium",
    }
    if rule in plain:
        return plain[rule]
    if rule not in ("band:promote_pearson_high", "band:demote_pearson_low"):
        raise NoV1Shape(f"no v1 sentence for band rule {rule!r}")
    pearson = _corroborator(driver, _PEARSON) if driver is not None else None
    if pearson is None:
        raise NoV1Shape(
            f"{rule} renders the driver's pearson into its v1 sentence, and the "
            "driver carries no pearson corroborator"
        )
    sign = "+" if rule == "band:promote_pearson_high" else "-"
    word = "high" if sign == "+" else "low"
    return f"{word} pearson on driver ({pearson:.2f}) -> {sign}1 band"


@dataclass(frozen=True, slots=True)
class UnitLabel:
    """The four cluster-key strings v1 persists beside the band.

    v2 models none of them: ``Assignment`` carries ``label``/``label_id`` and nothing
    about category or lineage, which are annotation metadata rather than measurement.
    They are passed in rather than looked up because ``core/`` does no IO.
    """

    cell_type: str
    cell_type_short: str
    category: str
    lineage: str


# --------------------------------------------------------------------------- #
# Row projections
# --------------------------------------------------------------------------- #
def _gene(item: EvidenceItem) -> str:
    g = item.subject.one("g")
    if item.subject.shape != "gene" or g is None:
        raise NoV1Shape(
            f"v1 MarkerEvidence is one gene per row; {item.subject.key!r} is a "
            f"{item.subject.shape!r} subject"
        )
    return g.id


def _number(item: EvidenceItem, name: str) -> float:
    v = _corroborator(item, name)
    if v is None:
        raise NoV1Shape(
            f"{_gene(item)}: v1 MarkerEvidence requires a float {name}; the item "
            "carries no such corroborator"
        )
    return v


def _marker_row(item: EvidenceItem, unit: str, canon: frozenset[str]) -> dict[str, Any]:
    """One ``EvidenceItem`` as a v1 ``MarkerEvidence`` dict.

    ``top_cluster`` is the unit itself: ``_build_evidence`` reads a cluster's assigned
    rows, which are by construction the rows whose top cluster is that cluster, and
    all nine committed JSONs agree (zero mismatches). ``p_value`` is ``None`` because
    v1 hardcodes it so -- the markers_full ``c\\d+`` term it was meant to carry was
    never wired up.
    """
    if not item.measured.carries_number:
        raise NoV1Shape(
            f"{_gene(item)} is {item.measured.state.value} in {item.measured.frame!r}; "
            "v1 MarkerEvidence has no shape for an item without a statistic, and "
            "silently dropping it would read as a measurement that never happened"
        )
    gene = _gene(item)
    low = FLAG_LOW_PEARSON in item.flags
    canonical = gene.upper() in canon
    return {
        "gene": gene,
        "top_cluster": unit,
        "glm_coef": item.statistic,
        "pearson": _number(item, _PEARSON),
        "max_gg_corr": _number(item, "max_gg_corr"),
        "max_gc_corr": _number(item, "max_gc_corr"),
        "p_value": None,
        "is_canonical": canonical,
        "is_on_panel": item.measured.state is not MeasuredState.NEVER_MEASURED,
        "role": "expected_absent" if (canonical and low) else "supports",
        "caveats": [prose for flag, prose in CAVEAT_FOR if flag in item.flags],
        "source": EVIDENCE_SOURCE,
    }


def _offpanel_note(item: EvidenceItem, cell_type: str) -> dict[str, Any]:
    """A ``NEVER_MEASURED`` item as a v1 ``OffPanelNote`` dict.

    §10: ``OffPanelNote`` is deleted and "becomes an ``EvidenceItem`` with
    ``NEVER_MEASURED``". This is that identification, read backwards.
    """
    gene = _gene(item)
    return {
        "gene": gene,
        "cell_type": cell_type,
        "message": (
            f"{gene} is off-panel (never measured); its absence is not "
            f"evidence against {cell_type}."
        ),
        "source": OFFPANEL_SOURCE,
    }


def _v1_bucket(item: EvidenceItem) -> str:
    """Which v1 shape this v2 item becomes, or which omission class it falls in.

    One classifier, used by the projection and by :func:`v1_omissions`, so the thing
    that is counted and the thing that is omitted cannot drift. Raises for any state
    no one has written an entry for.

    Private, and so returns a bare ``str``: ``test_every_core_function_returns_traced``
    requires every PUBLIC function in ``core/`` to hand its reasoning up, and a
    per-item classifier that returned ``Traced`` would emit one empty trace per gene
    -- 280 of them per unit -- to say nothing. The reasoning that matters is the
    count, which :func:`v1_omissions` carries.
    """
    if item.measured.carries_number:
        return V1_EVIDENCE
    if item.measured.state is MeasuredState.NEVER_MEASURED:
        return (V1_OFFPANEL if item.measured.frame.startswith(PANEL_FRAME_PREFIX)
                else V1_OMIT_UNDETECTABLE)
    if item.measured.state is MeasuredState.TESTED_ABSENT:
        return V1_OMIT_TESTED_ABSENT
    raise NoV1Shape(
        f"{_gene(item)} is {item.measured.state.value} in {item.measured.frame!r}; "
        "v1 has only two buckets (an evidence row carrying a number, or an off-panel "
        "note) and no entry in V1_UNREPRESENTABLE covers this state. Add one with a "
        "reason, or give v1 a shape for it -- do not let it be skipped."
    )


def v1_omissions(evidence: NormalisedEvidence) -> Traced[tuple[tuple[str, int], ...]]:
    """How many items of each unrepresentable class this unit's v1 view leaves out.

    Sorted pairs, not a dict, for every other frozen carrier's reason: the caller
    announces this at the moment the lossy view is written, so "omitted because v1
    cannot hold it" arrives as a counted statement rather than as a gap a reader has
    to notice.
    """
    counts = {b: 0 for b in sorted(UNREPRESENTABLE_BUCKETS)}
    for item in evidence.items:
        bucket = _v1_bucket(item)
        if bucket in counts:
            counts[bucket] += 1
    return Traced(tuple(sorted(counts.items())))


def _headline(cell_type: str, band: str, drivers: tuple[dict[str, Any], ...]) -> str:
    if drivers:
        d = drivers[0]
        return (
            f"{cell_type}, {band} confidence, driven by {d['gene']} "
            f"(glm_coef {d['glm_coef']:.2f}, pearson {d['pearson']:.2f})."
        )
    return f"{cell_type}, {band} confidence; no canonical marker drives the call."


def _notes(cell_type: str, band: str, drivers: tuple[dict[str, Any], ...],
           demotions: tuple[str, ...], fragile: bool, verify: bool) -> str:
    parts: list[str] = []
    if drivers:
        bits = ", ".join(
            f"{d['gene']} glm_coef={d['glm_coef']:.2f}/pearson={d['pearson']:.2f}"
            for d in drivers[:3]
        )
        parts.append(f"{cell_type} call driven by {bits}.")
    else:
        parts.append(f"{cell_type} call has no canonical marker support.")
    parts.append(f"Confidence {band}.")
    if demotions:
        parts.append("Adjustments: " + "; ".join(demotions) + ".")
    if fragile:
        parts.append("Fragile cluster (<=2 assigned markers): re-check this.")
    if verify:
        parts.append("verify=TRUE, re-check this call.")
    return " ".join(parts)


def _source_trace(rows: tuple[dict[str, Any], ...],
                  offpanel: tuple[dict[str, Any], ...]) -> list[str]:
    trace: list[str] = []
    for r in rows:
        trace.append(f"jz:{r['gene']}:glm_coef={r['glm_coef']:.6f}")
        trace.append(f"jz:{r['gene']}:pearson={r['pearson']:.6f}")
    trace += [f"panel:{n['gene']}:off_panel=True" for n in offpanel]
    return trace


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def to_v1_assignment(evidence: NormalisedEvidence, verdict: Verdict,
                     label: UnitLabel,
                     canonical: frozenset[str]) -> Traced[dict[str, Any]]:
    """One unit's v2 output as the v1 per-unit JSON dict (``ClusterVerdict`` shape).

    ``canonical`` is passed in for the reason ``core.verdict.assess`` takes it:
    ``core/`` does no IO, and a curated marker list is reference data, which may
    select and order evidence but never supply a measured value. It supplies none
    here -- it decides which rows count as canonical supporters, and the measurement
    alone fills every number.

    The driver set comes from ``core.verdict._drivers`` rather than being re-derived,
    because "who is eligible to carry the call" is the band's own rule. A second copy
    of it here could put a different gene in ``opening.driving_markers`` than the one
    the band was read off, and the record would then disagree with itself.
    """
    canon = frozenset(g.upper() for g in canonical)
    # One pass through the one classifier. Anything v1 has no shape for and no
    # reviewed entry for raises in there; the two classes that DO have entries are
    # omitted here and counted by v1_omissions for the caller to announce.
    buckets = tuple(_v1_bucket(i) for i in evidence.items)
    observed = tuple(i for i, b in zip(evidence.items, buckets) if b == V1_EVIDENCE)
    absent = tuple(i for i, b in zip(evidence.items, buckets) if b == V1_OFFPANEL)

    rows = tuple(_marker_row(i, evidence.unit, canon) for i in observed)
    offpanel = tuple(_offpanel_note(i, label.cell_type) for i in absent)
    driver_genes = {_gene(i) for i in _eligible_drivers(observed, canon)}
    drivers = tuple(r for r in rows if r["gene"] in driver_genes)

    demotions = [_demotion_sentence(r, verdict.driver) for r in verdict.demotions]
    hooks = [
        {
            "claim": f"{d['gene']} marks {label.cell_type}",
            "marker": d["gene"],
            "cell_type": label.cell_type,
            "query_terms": [d["gene"], label.cell_type.replace("_", " "), "marker"],
            "status": "unfilled",
        }
        for d in drivers
    ]

    return Traced(value={
        "cluster": evidence.unit,
        "cell_type": label.cell_type,
        "cell_type_short": label.cell_type_short,
        "confidence": verdict.band,
        "confidence_score": SCORE_MAP[verdict.band],
        "key_markers": [r["gene"] for r in rows[:5]],
        "notes": _notes(label.cell_type, verdict.band, drivers, tuple(demotions),
                        verdict.fragile, verdict.verify),
        "category": label.category,
        "lineage": label.lineage,
        # v1's engine never excludes: exclusion is a lab note applied at report
        # composition, never a property of the deterministic verdict.
        "exclude": False,
        "verify": verdict.verify,
        "small_n": verdict.fragile,
        "evidence": list(rows),
        "offpanel_notes": list(offpanel),
        "opening": {
            "cluster": evidence.unit,
            "cell_type": label.cell_type,
            "confidence": verdict.band,
            "headline": _headline(label.cell_type, verdict.band, drivers),
            "driving_markers": list(drivers),
            "offpanel_notes": list(offpanel),
            "literature_hooks": hooks,
            "verify": verdict.verify,
        },
        "band_basis": verdict.basis,
        "demotions": demotions,
        "source_trace": _source_trace(rows, offpanel),
    })


def v1_verdict_csv(units: Iterable[Mapping[str, Any]],
                   header: bool = True) -> Traced[str]:
    """The frozen 11-column verdict CSV, from projected v1 dicts.

    Thin on purpose. ``core/export.py`` already reproduces that table
    byte-identically, and its columns are the ones pinned by the golden test and read
    by downstream R; a second column list here is the drift ``export.py`` exists to
    make unrepresentable. The only work left is handing its duck-typed getters
    something with attributes.
    """
    return export.to_csv(TaskKind.ASSIGNMENT,
                         [SimpleNamespace(**dict(u)) for u in units], header=header)
