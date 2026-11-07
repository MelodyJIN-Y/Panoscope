"""Deterministic discriminator: "what would settle it".

Tier A (deterministic, no network, no LLM). Given a cluster whose call is **A**
and a competing hypothesis **B** (named by the biologist or derived from the
cluster's own off-type markers), this module names the markers that separate A
from B, each classified honestly against what was actually measured.

The governing data fact: jazzPanda output is **one row per gene** — each gene is
assigned to a single winning cluster with a single glm_coef/pearson. There is no
(cluster x gene) matrix, so a marker's number exists only for its own top
cluster. The hard rule enforced here: **a jazzPanda number is only ever attached
to a gene that is a top marker of the cluster in question**; everything else is
classified without a number. That keeps every quoted value groundable and lets
the agent answer clear the confident floor.

**The bucketing engine is ``core/rival.py``.** This module keeps the published
surface its ten call sites read -- ``Discrimination``, ``DiscriminatorMarker``,
``discriminate``, ``settle_summary``, ``settle_line``, ``_canon_key`` -- and does the
four jobs ``core/rival.py:61-67`` explicitly disowns: deriving the rival from the
unit's own off-type markers, resolving aliases and within-lineage refinements,
re-attaching *where else* a marker went, and prose. The buckets themselves are filters
on ``Measured.state``, computed once in core:

- ``supporting_A``    <- OBSERVED, canonical for A (with numbers).
- ``b_here``          <- ``supports_rival``: OBSERVED, canonical for B. Genuine B
                        signal here, which is ambiguity and not a refutation.
- ``b_elsewhere``     <- ``against_rival``: TESTED_ABSENT -- in frame, measured, not
                        reported here. The productive half; no number attached here.
- ``offpanel_absent`` <- ``undecidable``: NEVER_MEASURED or UNDERPOWERED. **Not only
                        off-panel.** A gene the panel carried and the measurement
                        could not localize anywhere lands here too; the old engine
                        asked only ``panel_contains`` and filed those under
                        "argues against B", which is the panel-absence rule inverted
                        into a confident false negative. The prose names WHICH frame
                        refused each one. We never recommend running an experiment.

The governing data fact behind the numbers: jazzPanda output is **one row per gene**
-- each gene is assigned to a single winning cluster with a single glm_coef/pearson.
There is no (cluster x gene) matrix, so a marker's number exists only for its own top
cluster. The rule that **a number is only ever attached to a gene that is a top marker
of the cluster in question** is no longer a convention held by three call sites:
``EvidenceItem`` raises if a statistic and a measurement status disagree.

``settleable_on_panel`` is ``Rivalry.settleable`` -- K of N of B's canonical set in
frame, against a floor of 1, which is what the old ``bool(b_here or b_elsewhere)``
meant with the denominator thrown away.

Reuses the grounded primitives: ``core.rival.rival_test``, ``pipeline.records``,
``annotation.canonical_markers``, ``data.get_cluster_markers`` / ``get_marker`` /
``cell_type_for``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.types.measured import MeasuredState


@data._per_study
def _world_and_builds():
    """The v2 evidence for every unit of the current study, plus its capability frames.

    Cached per study by the same decorator every other zero-arg reader uses, so the
    autouse fixture in ``tests/conftest.py`` clears it with the rest and a study
    switch cannot serve stale evidence. ~0.1 s cold.

    The imports are function-local because ``agent -> pipeline`` is a cycle risk --
    the same reason ``loop.py`` takes its ``pipeline`` imports that way. There is
    deliberately no ``try``/``except``: a discriminator that silently answers from no
    evidence is the failure shape that cost three bugs tonight. Let it raise.
    """
    from panoscope.pipeline import records
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.measured import MeasuredWorld

    study = cfg.current_study()
    return records.build_all(study), MeasuredWorld(XeniumAdapter().frames(study))


def _off_panel(m) -> bool:
    """True iff it was the PANEL frame that refused this subject.

    ``Measured.frame`` names the ONE frame that refused a ``NEVER_MEASURED`` subject
    (``types/measured._state_for``), so this separates *not on the panel* from *on
    the panel but produced no usable spatial signal*. They are different facts to a
    biologist -- the first is fixed by ordering a different panel, the second is not
    -- and the ``off_panel: bool`` this replaces collapsed them.
    """
    return m.state is MeasuredState.NEVER_MEASURED and m.frame.startswith("panel:")


def _corr(item, name: str) -> Optional[float]:
    """A corroborating statistic off an evidence item. Read, never computed here."""
    return dict(item.corroborating).get(name)


def _marker(item, role: "Role", top_cluster: Optional[str]) -> "DiscriminatorMarker":
    """One bucket entry, read off a v2 evidence item.

    No number can be invented: ``EvidenceItem`` enforces ``statistic is None`` iff
    the subject was not OBSERVED, and the corroborator is dropped on the same test
    rather than on a separate list of states that could drift from it.
    """
    observed = item.measured.carries_number
    return DiscriminatorMarker(
        gene=item.subject.display(),
        role=role,
        on_panel=not _off_panel(item.measured),
        top_cluster=top_cluster,
        glm_coef=item.statistic,
        pearson=_corr(item, "pearson") if observed else None,
    )


def _locates(gene: str) -> Optional[str]:
    """Which unit this gene peaked in. A locating fact, never a number."""
    row = data.get_marker(gene)
    return str(row["top_cluster"]) if row is not None else None


def _is_canonical(gene: str, cell_type: str) -> bool:
    """True iff ``gene`` is a canonical marker for ``cell_type`` (per the dataset's
    annotation, else the bundled fallback)."""
    return gene.upper() in {g.upper() for g in annotation.canonical_markers(cell_type)}

Role = Literal["supports_A", "supports_B_here", "b_localizes_elsewhere", "offpanel_absent"]

# jazzPanda "not significant in any cluster" sentinel (a gene on the panel that
# jazzPanda did not assign to a real cluster).
_NOSIG = "NoSig"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiscriminatorMarker:
    """One marker in the A-vs-B comparison.

    ``glm_coef``/``pearson`` are set ONLY when the gene is a top marker of the
    cluster under discussion (``role`` in {supports_A, supports_B_here}); for
    markers that localize elsewhere or are off-panel they are None — no number is
    ever attributed to a cluster that did not produce it.
    """

    gene: str
    role: Role
    on_panel: bool
    top_cluster: Optional[str]  # where the gene localizes; None if off-panel
    glm_coef: Optional[float]
    pearson: Optional[float]


@dataclass(frozen=True)
class Discrimination:
    cluster: str
    call_A: str
    alt_B: Optional[str]  # normalized canonical-type key, or None if no rival
    supporting_A: tuple[DiscriminatorMarker, ...]
    b_here: tuple[DiscriminatorMarker, ...]
    b_elsewhere: tuple[DiscriminatorMarker, ...]
    offpanel_absent: tuple[DiscriminatorMarker, ...]
    settleable_on_panel: bool
    reason: str
    source_trace: tuple[str, ...]
    # The biologist named a SUBTYPE / synonym of the call's own lineage (e.g. CAF for
    # Stromal): a within-lineage refinement, not a rivalry the panel can discriminate.
    refinement: bool = False


# --------------------------------------------------------------------------- #
# Cell-type normalization
# --------------------------------------------------------------------------- #
def _canon_key(s: str) -> str:
    """Collapse a cell-type string to a comparison key: upper, no spaces/_/-."""
    return s.strip().upper().replace(" ", "").replace("_", "").replace("-", "")


# Common wet-lab synonyms -> canonical map key. Kept small and unambiguous.
#
# BREAST-DEMO LABELS. Every target here is a key of this study's annotation and of
# ``annotation.CANONICAL_MARKERS_FALLBACK``, which ``annotation._is_bundled_demo``
# already gates for exactly that reason. This table was not gated, so on any other
# study the biologist's "CAF" was silently translated into a label that study does
# not use -- the demo's label space reaching into a dataset it knows nothing about,
# which is the failure the annotation gate exists to prevent one file over.
_ALIASES: dict[str, str] = {
    "CAF": "Stromal",
    "FIBROBLAST": "Stromal",
    "FIBROBLASTS": "Stromal",
    "MACROPHAGE": "Macrophages",
    "TAM": "Macrophages",
    "TAMS": "Macrophages",
    "TCELL": "T_Cells",
    "BCELL": "B_Cells",
    "DC": "Dendritic",
    "DENDRITICCELL": "Dendritic",
    "MAST": "Mast_Cells",
    "ENDOTHELIUM": "Endothelial",
}

def _key_by_canon() -> dict[str, str]:
    """Comparison-key -> canonical cell-type name, over the dataset's annotated types."""
    return {_canon_key(k): k for k in annotation.all_canonical()}


def _normalize_cell_type(s: Optional[str]) -> Optional[str]:
    """Map a free-text cell type onto a canonical-map key, or None if unknown."""
    if not s:
        return None
    key = _canon_key(s)
    kmap = _key_by_canon()
    if key in kmap:
        return kmap[key]
    # The study's own annotated types are always tried first and are never gated.
    # The synonym table is demo content: off the demo an unrecognised word stays
    # unrecognised, and ``discriminate`` declines it BY NAME -- which is a true
    # answer, where a translated label is a quiet substitution.
    if annotation._is_bundled_demo():
        return _ALIASES.get(key)
    return None


def _canonical_types_for(gene: str) -> tuple[str, ...]:
    """All canonical cell types a gene marks (the map is disjoint, so usually <=1)."""
    return tuple(t for t in annotation.all_canonical() if _is_canonical(gene, t))


def _lineage_keys_for(call_A: str) -> tuple[str, ...]:
    """Comparison-keys that name the CALL's own lineage: its type name plus any alias
    that maps to it (for Stromal: STROMAL, CAF, FIBROBLAST, FIBROBLASTS)."""
    keys = {_canon_key(call_A)}
    keys.update(k for k, t in _ALIASES.items() if t == call_A)
    return tuple(k for k in keys if k)


def _refines_call(alt_raw: str, call_A: str) -> bool:
    """True if the biologist's alternative is a subtype/synonym of the CALL's lineage
    (e.g. 'cancer-associated fibroblast', 'CAF', 'myofibroblast' when the call is
    Stromal). That is a within-lineage refinement, not a distinct rival to discriminate.
    Matched by substring so multi-word subtype names ('matrix-remodelling CAF') resolve."""
    key = _canon_key(alt_raw)
    if not key:
        return False
    return any(kw in key or key in kw for kw in _lineage_keys_for(call_A))


def _refinement_reason(alt_raw: str, call_A: str) -> str:
    """Honest prose for a within-lineage refinement: the panel supports the lineage, the
    markers that would resolve a subtype are off-panel, so it is a tissue-context call."""
    A = _label(call_A)
    alt = alt_raw.strip()
    off = tuple(annotation.offpanel_canonical(call_A))
    if off:
        tail = (
            f" The canonical markers that would resolve a subtype ({', '.join(off)}) are "
            f"off-panel and were never measured, so the panel cannot separate {alt} from generic {A}."
        )
    else:
        tail = f" The panel cannot separate {alt} from generic {A} on the measured markers."
    return (
        f"{alt} is a subtype of the {A} lineage, not a distinct cell type this panel can discriminate."
        f"{tail} Refining to {alt} is a tissue-context call to record, not one the markers settle."
    )


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _derive_alt(cluster: str, call_A: str) -> Optional[str]:
    """Strongest competing type evident in the cluster's OWN markers.

    Walk the cluster's markers by glm_coef and return the first non-A canonical type
    any of them marks. None if the cluster's markers all point at A (a clean call with
    no rival in the measured data).

    Rival derivation is one of the four jobs ``core/rival.py`` disowns -- it needs the
    whole canonical map rather than one rival's list -- so it still reads the marker
    table, for gene NAMES in the method's own order. No number is taken from it: every
    number in the result comes up through ``core.measure``.
    """
    for g in data.get_cluster_markers(cluster)["gene"]:  # glm_coef descending
        for t in _canonical_types_for(str(g)):
            if t != call_A:
                return t
    return None


def discriminate(cluster: str, alt_cell_type: Optional[str] = None) -> Discrimination:
    """Bucket the markers that separate this cluster's call from an alternative.

    Deterministic and network-free. ``KeyError`` if the cluster is unknown.
    """
    call_A = data.cell_type_for(cluster)  # raises KeyError on unknown cluster

    # Resolve the competing hypothesis B.
    refinement = False
    if alt_cell_type is not None:
        alt_B = _normalize_cell_type(alt_cell_type)
        if _canon_key(alt_cell_type) == _canon_key(call_A):
            # the biologist named the SAME type as the call: a no-op, nothing to discriminate.
            alt_B, reason = None, f"the alternative equals the call ({call_A}); nothing to discriminate."
        elif alt_B == call_A or (alt_B is None and _refines_call(alt_cell_type, call_A)):
            # a DIFFERENT term that maps to the call's lineage (e.g. CAF for Stromal): a
            # within-lineage refinement, not a rivalry the panel can discriminate.
            refinement, alt_B = True, None
            reason = _refinement_reason(alt_cell_type, call_A)
        elif alt_B is None:
            reason = f"'{alt_cell_type}' is not a cell type with a canonical marker set; cannot discriminate."
        else:
            reason = ""
    else:
        alt_B = _derive_alt(cluster, call_A)
        reason = (
            "" if alt_B is not None
            else f"no competing cell-type hypothesis is evident in {cluster}'s measured markers; "
                 f"the call rests on its {call_A} markers."
        )

    # The buckets come from core.rival, against the four-state lattice. supporting_A
    # is the same filter core.rival applies to ``supports_call``, applied here because
    # it is wanted even when there is no rival to test.
    builds, world = _world_and_builds()
    ev = builds[cluster].evidence
    canon_A = {g.upper() for g in annotation.canonical_markers(call_A)}
    supporting_A = tuple(
        _marker(i, "supports_A", cluster)
        for i in ev.items  # statistic-descending: the strongest support leads
        if i.measured.carries_number and i.subject.display().upper() in canon_A
    )

    b_here: tuple[DiscriminatorMarker, ...] = ()
    b_elsewhere: tuple[DiscriminatorMarker, ...] = ()
    offpanel: tuple[DiscriminatorMarker, ...] = ()
    settleable = False

    if alt_B is not None:
        from panoscope.core.rival import rival_test

        r = rival_test(
            ev, call_A, alt_B,
            tuple(annotation.canonical_markers(call_A)),
            tuple(annotation.canonical_markers(alt_B)),
            world,
            floor=1,  # v1's settleable_on_panel was bool(b_here or b_elsewhere)
        ).value
        b_here = tuple(_marker(i, "supports_B_here", cluster) for i in r.supports_rival)
        # top_cluster is re-attached here, not in core: an item carries one unit's
        # evidence and has no idea where else the gene went. "localizes to c5, not c3"
        # is a locating fact with no number, and the UI and prose both read it.
        b_elsewhere = tuple(
            _marker(i, "b_localizes_elsewhere", _locates(i.subject.display()))
            for i in r.against_rival
        )
        # NOT only off-panel any more: a gene on the panel that produced no usable
        # spatial signal lands here too, where v1 filed it under "argues against B".
        offpanel = tuple(_marker(i, "offpanel_absent", None) for i in r.undecidable)
        settleable = r.settleable

    return Discrimination(
        cluster=cluster,
        call_A=call_A,
        alt_B=alt_B,
        supporting_A=supporting_A,
        b_here=tuple(b_here),
        b_elsewhere=tuple(b_elsewhere),
        offpanel_absent=tuple(offpanel),
        settleable_on_panel=settleable,
        reason=reason,
        source_trace=_trace(supporting_A, tuple(b_here), tuple(b_elsewhere), tuple(offpanel)),
        refinement=refinement,
    )


def _trace(
    supporting_A: tuple[DiscriminatorMarker, ...],
    b_here: tuple[DiscriminatorMarker, ...],
    b_elsewhere: tuple[DiscriminatorMarker, ...],
    offpanel: tuple[DiscriminatorMarker, ...],
) -> tuple[str, ...]:
    """Every fact used. Numbers appear ONLY for this cluster's own top markers."""
    trace: list[str] = []
    for m in (*supporting_A, *b_here):  # both are top markers of this cluster
        trace.append(f"jz:{m.gene}:glm_coef={m.glm_coef:.6f}")
        trace.append(f"jz:{m.gene}:pearson={m.pearson:.6f}")
    for m in b_elsewhere:  # a locating fact, not a number attributed here
        trace.append(f"jz:{m.gene}:top_cluster={m.top_cluster}")
    for m in offpanel:
        # WHICH frame refused it. "off_panel=True" for a gene the panel carried and
        # the measurement could not localize would be a fabricated fact about the panel.
        trace.append(f"panel:{m.gene}:off_panel=True" if not m.on_panel
                     else f"detect:{m.gene}:no_usable_signal=True")
    return tuple(trace)


# --------------------------------------------------------------------------- #
# Deterministic prose (used by the fallback and the proactive opening line)
# --------------------------------------------------------------------------- #
def _fmt(markers: tuple[DiscriminatorMarker, ...], n: int = 3) -> str:
    """Gene (glm_coef X.XX) list for markers that carry a number."""
    return ", ".join(
        f"{m.gene} (glm_coef {m.glm_coef:.2f})" for m in markers[:n] if m.glm_coef is not None
    )


def _label(cell_type: str) -> str:
    return cell_type.replace("_", " ")


def _by_cluster(markers) -> dict[str, list[str]]:
    """Genes grouped by the cluster they localize to (``NoSig`` when none), in order."""
    out: dict[str, list[str]] = {}
    for m in markers:
        out.setdefault(m.top_cluster or _NOSIG, []).append(m.gene)
    return out


def _undecidable_bits(d: Discrimination) -> list[tuple[list[str], str, str]]:
    """``(genes, sentence phrase, compact phrase)`` per refusing frame.

    Two different facts, never collapsed into one sentence: *off the panel* is fixed
    by ordering a different panel, *on the panel but with no usable spatial signal*
    is not. v1 said "off-panel" unconditionally because its only capability question
    was ``panel_contains``; the bucket now also holds genes the detectability frame
    refused (49 of 280 on the demo panel), and calling those off-panel would be a
    fabricated fact about the panel.
    """
    off = [m.gene for m in d.offpanel_absent if not m.on_panel]
    nosig = [m.gene for m in d.offpanel_absent if m.on_panel]
    out: list[tuple[list[str], str, str]] = []
    if off:
        out.append((off, "is off-panel and was never measured" if len(off) == 1
                    else "are off-panel and were never measured",
                    "off-panel (never measured)"))
    if nosig:
        out.append((nosig, f"{'is' if len(nosig) == 1 else 'are'} on the panel but "
                           "produced no usable spatial signal",
                    "on the panel but with no usable spatial signal"))
    return out


def settle_summary(d: Discrimination) -> str:
    """Grounded, panel-absence-safe prose. Quotes numbers only for own top markers,
    flags off-panel genes by name, and never recommends an experiment."""
    A = _label(d.call_A)

    if d.alt_B is None:
        lead = f"{d.cluster} reads {A}"
        if d.supporting_A:
            lead += f", supported by {_fmt(d.supporting_A)}"
        lead += "."
        if d.reason:
            lead += " " + d.reason[0].upper() + d.reason[1:]
        return lead

    B = _label(d.alt_B)
    parts: list[str] = [f"To tell {A} from {B} in {d.cluster}:"]

    if d.supporting_A:
        parts.append(f"{A} is supported here by {_fmt(d.supporting_A)}.")

    if d.b_here:
        n = sum(1 for m in d.b_here if m.glm_coef is not None)
        noun = "marker" if n == 1 else "markers"
        verb = "is also a top marker" if n == 1 else "are also top markers"
        parts.append(
            f"{B} {noun} {_fmt(d.b_here)} {verb} here, so there is genuine {B} signal — ambiguous."
        )

    if d.b_elsewhere:
        bits = []
        for cl, genes in _by_cluster(d.b_elsewhere).items():
            where = "not significant in any cluster" if cl == _NOSIG else f"localize to {cl}, not {d.cluster}"
            bits.append(f"{', '.join(genes)} ({where})")
        parts.append(f"{B} markers are on the panel but {'; '.join(bits)} — that argues against {B}.")

    for genes, why, _short in _undecidable_bits(d):
        one = len(genes) == 1
        noun = f"a canonical {B} marker" if one else f"canonical {B} markers"
        tail = "it cannot settle it" if one else "they cannot settle it"
        parts.append(f"{', '.join(genes)} — {noun} — {why}, so {tail}.")

    if d.b_here:
        parts.append("The panel shows mixed signal here; re-check this.")
    elif d.settleable_on_panel:
        parts.append(f"On the measured markers this leans {A}.")
    else:
        # "off-panel" would be a claim about the panel; the bucket now also holds genes
        # the panel carried and the measurement could not localize. Frame-neutral.
        parts.append(f"The markers that would decide it were never measured, so the panel cannot settle {A} vs {B}.")

    return " ".join(parts)


def _elsewhere_bits(d: Discrimination) -> str:
    """'GENE, GENE localize to cX' phrases grouped by target cluster."""
    out = []
    for cl, genes in _by_cluster(d.b_elsewhere).items():
        where = "not significant anywhere" if cl == _NOSIG else f"localize to {cl}"
        out.append(f"{', '.join(genes)} {where}")
    return "; ".join(out)


def settle_line(d: Discrimination) -> str:
    """One compact grounded sentence for the proactive opening. Empty if there is no
    rival hypothesis to settle. Quotes numbers only for markers that peak in this
    cluster; flags off-panel genes without recommending any experiment."""
    if d.alt_B is None:
        return ""
    A, B = _label(d.call_A), _label(d.alt_B)
    bits: list[str] = []
    if d.b_here:
        bits.append(f"{B} markers {_fmt(d.b_here)} also peak here")
    if d.b_elsewhere:
        bits.append(f"{B} markers {_elsewhere_bits(d)}")
    for genes, _why, short in _undecidable_bits(d):
        bits.append(f"{', '.join(genes)} {short}")
    tail = (
        "mixed signal, re-check this"
        if d.b_here
        else f"leans {A}" if d.settleable_on_panel
        else f"the panel cannot settle it"
    )
    return f" To settle {A} vs {B}: {'; '.join(bits)} — {tail}."
