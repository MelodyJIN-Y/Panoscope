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

Buckets for B's canonical markers:
- ``supporting_A``  - this cluster's own top markers canonical for A (with numbers).
- ``b_here``        - canonical-B markers that ARE top markers of this cluster
                      (genuine B signal here; with numbers).
- ``b_elsewhere``   - canonical-B markers on the panel but localizing to another
                      cluster (measured, but the B program lives there, not here;
                      no number attached here).
- ``offpanel_absent`` - canonical-B markers not on the panel: never measured, so
                      only FLAGGED (panel-absence). We never recommend running an
                      experiment.

``settleable_on_panel`` is True when at least one canonical-B marker was on the
panel (so the measured data speaks to A-vs-B); False when B's markers are all
off-panel (the panel cannot settle it).

Reuses the grounded primitives: ``verdict.CANONICAL_MARKERS`` / ``_is_canonical``,
``data.get_cluster_markers`` / ``get_marker`` / ``panel_contains`` /
``cell_type_for``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from agent import annotation
from agent import data


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
    return _ALIASES.get(key)


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
def _own_markers(cluster: str) -> dict[str, dict]:
    """This cluster's own top markers keyed by UPPER gene -> {glm_coef, pearson, top_cluster}."""
    df = data.get_cluster_markers(cluster)
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[str(r["gene"]).upper()] = {
            "gene": str(r["gene"]),
            "glm_coef": float(r["glm_coef"]),
            "pearson": float(r["pearson"]),
            "top_cluster": str(r["top_cluster"]),
        }
    return out


def _derive_alt(call_A: str, own: dict[str, dict]) -> Optional[str]:
    """Strongest competing type evident in the cluster's OWN markers.

    Walk the cluster's markers by glm_coef (``own`` preserves that order) and
    return the first non-A canonical type any of them marks. None if the cluster's
    markers all point at A (a clean call with no rival in the measured data).
    """
    for info in own.values():  # insertion order = glm_coef desc from get_cluster_markers
        for t in _canonical_types_for(info["gene"]):
            if t != call_A:
                return t
    return None


def discriminate(cluster: str, alt_cell_type: Optional[str] = None) -> Discrimination:
    """Bucket the markers that separate this cluster's call from an alternative.

    Deterministic and network-free. ``KeyError`` if the cluster is unknown.
    """
    call_A = data.cell_type_for(cluster)  # raises KeyError on unknown cluster
    own = _own_markers(cluster)

    # supporting_A: the cluster's own top markers canonical for A (with numbers).
    supporting_A = tuple(
        DiscriminatorMarker(
            gene=info["gene"],
            role="supports_A",
            on_panel=data.panel_contains(info["gene"]),
            top_cluster=cluster,
            glm_coef=info["glm_coef"],
            pearson=info["pearson"],
        )
        for info in own.values()
        if _is_canonical(info["gene"], call_A)
    )

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
        alt_B = _derive_alt(call_A, own)
        reason = (
            "" if alt_B is not None
            else f"no competing cell-type hypothesis is evident in {cluster}'s measured markers; "
                 f"the call rests on its {call_A} markers."
        )

    b_here: list[DiscriminatorMarker] = []
    b_elsewhere: list[DiscriminatorMarker] = []
    offpanel: list[DiscriminatorMarker] = []

    if alt_B is not None:
        for gene in annotation.canonical_markers(alt_B):
            up = gene.upper()
            if up in own:  # a top marker of THIS cluster -> genuine B signal, with number
                info = own[up]
                b_here.append(
                    DiscriminatorMarker(
                        gene=info["gene"],
                        role="supports_B_here",
                        on_panel=data.panel_contains(gene),
                        top_cluster=cluster,
                        glm_coef=info["glm_coef"],
                        pearson=info["pearson"],
                    )
                )
            elif data.panel_contains(gene):  # measured, localizes elsewhere -> no number here
                row = data.get_marker(gene)
                top = str(row["top_cluster"]) if row is not None else None
                b_elsewhere.append(
                    DiscriminatorMarker(
                        gene=gene,
                        role="b_localizes_elsewhere",
                        on_panel=True,
                        top_cluster=top,
                        glm_coef=None,
                        pearson=None,
                    )
                )
            else:  # off-panel -> never measured, only flagged
                offpanel.append(
                    DiscriminatorMarker(
                        gene=gene,
                        role="offpanel_absent",
                        on_panel=False,
                        top_cluster=None,
                        glm_coef=None,
                        pearson=None,
                    )
                )

    settleable = bool(b_here or b_elsewhere)

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
        trace.append(f"panel:{m.gene}:off_panel=True")
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
        by_cluster: dict[str, list[str]] = {}
        for m in d.b_elsewhere:
            by_cluster.setdefault(m.top_cluster or _NOSIG, []).append(m.gene)
        bits = []
        for cl, genes in by_cluster.items():
            where = "not significant in any cluster" if cl == _NOSIG else f"localize to {cl}, not {d.cluster}"
            bits.append(f"{', '.join(genes)} ({where})")
        parts.append(f"{B} markers are on the panel but {'; '.join(bits)} — that argues against {B}.")

    if d.offpanel_absent:
        genes = ", ".join(m.gene for m in d.offpanel_absent)
        if len(d.offpanel_absent) == 1:
            parts.append(
                f"{genes} — a canonical {B} marker — is off-panel and was never measured, so it cannot settle it."
            )
        else:
            parts.append(
                f"{genes} — canonical {B} markers — are off-panel and were never measured, so they cannot settle it."
            )

    if d.b_here:
        parts.append("The panel shows mixed signal here; re-check this.")
    elif d.settleable_on_panel:
        parts.append(f"On the measured markers this leans {A}.")
    else:
        parts.append(f"The markers that would decide it are off-panel, so the panel cannot settle {A} vs {B}.")

    return " ".join(parts)


def _elsewhere_bits(d: Discrimination) -> str:
    """'GENE, GENE localize to cX' phrases grouped by target cluster."""
    by_cluster: dict[str, list[str]] = {}
    for m in d.b_elsewhere:
        by_cluster.setdefault(m.top_cluster or _NOSIG, []).append(m.gene)
    out = []
    for cl, genes in by_cluster.items():
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
    if d.offpanel_absent:
        genes = ", ".join(m.gene for m in d.offpanel_absent)
        bits.append(f"{genes} off-panel (never measured)")
    tail = (
        "mixed signal, re-check this"
        if d.b_here
        else f"leans {A}" if d.settleable_on_panel
        else f"the panel cannot settle it"
    )
    return f" To settle {A} vs {B}: {'; '.join(bits)} — {tail}."
