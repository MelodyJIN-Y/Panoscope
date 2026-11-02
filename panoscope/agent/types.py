"""Shared frozen dataclasses — the contracts that cross every module boundary.

Transcribed from docs/archive/BLUEPRINT.md section 2 (superseded by IMPLEMENTATION.md). All dataclasses are frozen (the
immutability rule): construct new objects, never mutate. Modules import these
types rather than passing raw dicts, so the grounding tests and the UI agree on
exactly what a marker, a verdict, a note, and an agent response are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# --------------------------------------------------------------------------- #
# Closed vocabularies
# --------------------------------------------------------------------------- #
Confidence = Literal["Very High", "High", "Medium-High", "Medium", "Low"]
MarkerRole = Literal["supports", "expected_absent", "off_panel"]
Scope = Literal["cluster", "dataset", "lab"]
Basis = Literal["paper", "own_validation", "convention"]
Status = Literal["firm", "tentative"]
# The kind of lab-knowledge a note captures (see docs/note-capture-design.md). The
# type routes a note to render next to its anchor and never mutates a grounded number.
NoteType = Literal[
    "celltype_override",        # the cell-type call is rejected/replaced
    "marker_reinterpretation",  # what ONE marker means here (call unchanged); anchor = gene
    "program_reinterpretation", # an enriched program re-read as co-infiltration; anchor = gene_set
    "marker_convention",        # a panel/tissue trust convention about a marker; anchor = gene
    "validation",               # an own-validation (IHC/flow) reinforcing the call
    "confidence_adjustment",    # the biologist's confidence stance (overlay ONLY, never the number)
    "exclude",                  # exclude a cluster (doublet/artifact); applied at report composition
    "cross_cluster",            # two+ clusters are one population; anchor = a set of clusters
]


# --------------------------------------------------------------------------- #
# Evidence row the verdict/UI use (role column = panel-absence rule made visible)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarkerEvidence:
    gene: str
    top_cluster: str
    glm_coef: float
    pearson: float
    max_gg_corr: float
    max_gc_corr: float
    p_value: Optional[float]               # from markers_full c\d+ term only, else None
    within_cluster_pctile: float           # 0..1 (1.0 = strongest in cluster)
    is_canonical: bool
    is_on_panel: bool
    role: MarkerRole
    caveats: tuple[str, ...] = ()
    source: str = "jazzpanda:top_result"


@dataclass(frozen=True)
class OffPanelNote:
    gene: str                              # e.g. "COL1A1"
    cell_type: str                         # "Stromal"
    message: str                           # "<gene> is off-panel (never measured); ..."
    source: str = "panel:absence"


# --------------------------------------------------------------------------- #
# Literature / citations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Citation:
    pmid: str
    title: str
    authors: str
    year: int
    journal: str
    abstract: str = ""
    url: str = ""                          # https://pubmed.ncbi.nlm.nih.gov/{pmid}/
    stance: str = "context"                # agree | dissent | context | unclassified
    is_real: bool = True                   # True only if resolved via live MCP or frozen-real cache
    fetched_at: str = ""                   # iso; honest snapshot stamp for cached citations


@dataclass(frozen=True)
class LiteratureHook:
    """Engine emits WHAT to look up; the loop fills it live. Engine writes ZERO citations."""
    claim: str
    marker: str
    cell_type: str
    query_terms: tuple[str, ...]
    status: Literal["unfilled"] = "unfilled"


# --------------------------------------------------------------------------- #
# Opening interpretation (posted before any question)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OpeningInterpretation:
    cluster: str
    cell_type: str
    confidence: Confidence
    headline: str
    driving_markers: tuple[MarkerEvidence, ...]
    offpanel_notes: tuple[OffPanelNote, ...]
    literature_hooks: tuple[LiteratureHook, ...]
    verify: bool


# --------------------------------------------------------------------------- #
# Cluster verdict == CSV output contract + UI affordances
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterVerdict:
    cluster: str
    cell_type: str
    cell_type_short: str
    confidence: Confidence
    confidence_score: float                # fixed band anchor {0.95,0.85,0.70,0.55,0.30}
    key_markers: tuple[str, ...]           # top 3-5 by glm_coef
    notes: str                             # grounded rationale, cites glm_coef/pearson
    category: str
    lineage: str
    exclude: bool
    verify: bool
    # UI / audit extras (not in CSV):
    small_n: bool
    evidence: tuple[MarkerEvidence, ...]
    offpanel_notes: tuple[OffPanelNote, ...]
    opening: OpeningInterpretation
    band_basis: str                        # "percentile" | "small-n absolute"
    demotions: tuple[str, ...]             # audit trail of band changes
    source_trace: tuple[str, ...]          # every (gene,stat,value) used — grounding tests read this


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScopeRef:
    dataset: str
    cluster: Optional[str]                 # set iff scope == "cluster"


@dataclass(frozen=True)
class Tension:
    agree: tuple[Citation, ...]
    dissent: tuple[Citation, ...]
    thin: bool
    query: str
    looked_up_at: str


@dataclass(frozen=True)
class Note:
    id: str
    claim: str
    scope: Scope
    scope_ref: ScopeRef
    basis: Basis
    status: Status
    subject_cell_type: Optional[str]
    subject_markers: tuple[str, ...]
    tension: Tension
    author: str
    created_at: str
    trigger: Literal["override", "manual_add", "holistic_review"]
    supersedes: Optional[str]
    # Typed/anchored capture (docs/note-capture-design.md). All optional-with-defaults,
    # so every existing note JSON still parses. NONE may carry a grounded number.
    type: NoteType = "celltype_override"
    subject_gene_sets: tuple[str, ...] = ()   # enrichment anchor (HALLMARK_*), the marker analog
    subject_clusters: tuple[str, ...] = ()    # anchor SET for cross_cluster (fires on each)
    # For a celltype_override: the biologist's NEW call, applied at composition (never
    # mutating the deterministic verdict). subject_cell_type holds the new cell type.
    subject_lineage: str = ""                 # new lineage for the override (agent-inferred)
    subject_category: str = ""                # new category for the override (agent-inferred)


@dataclass(frozen=True)
class NoteDraft:
    """A proposed lab note, reconciled against the literature but NOT yet saved.

    The agent produces this at an override (via ``memory_draft``); the biologist
    confirms scope/basis/status in the chat and only then is it persisted
    (``memory.save_draft``). Nothing hits disk until the biologist saves. The
    ``tension`` is already computed from the claim, so adjusting scope/basis/status
    at confirm time never needs another literature lookup.
    """
    claim: str
    scope: Scope
    basis: Basis
    status: Status
    cluster: Optional[str]                  # active cluster; used iff scope=="cluster"
    subject_cell_type: Optional[str]
    subject_markers: tuple[str, ...]
    tension: Tension
    dataset: str
    # Typed/anchored capture — mirrors Note; carried through save_draft (defaults keep
    # the existing override path unchanged).
    type: NoteType = "celltype_override"
    subject_gene_sets: tuple[str, ...] = ()
    subject_clusters: tuple[str, ...] = ()
    subject_lineage: str = ""
    subject_category: str = ""


# --------------------------------------------------------------------------- #
# Agent I/O
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    kind: Literal["jz", "panel", "lit", "mem"]
    ref: str                               # gene | pmid | note_id
    value: Optional[str]                   # glm_coef value | title | claim
    detail: str = ""


@dataclass(frozen=True)
class GroundingSidecar:
    """Machine-readable claim manifest the grounding checker prefers over prose."""
    numbers: tuple[tuple[str, str, float], ...]   # (gene, stat, value)
    markers: tuple[str, ...]
    pmids: tuple[str, ...]
    notes_used: tuple[str, ...]


@dataclass(frozen=True)
class AgentResponse:
    text: str                              # markdown, PMID:xxx + [note:id] inline
    sources: tuple[Source, ...]
    verify: bool
    grounding: GroundingSidecar
    pin_marker: Optional[str] = None
    citations: tuple[Citation, ...] = ()
    note_written: Optional[Note] = None
    note_draft: Optional[NoteDraft] = None   # proposed note awaiting biologist confirm
    used_fallback: bool = False
    opening: bool = False
    tools_used: tuple[str, ...] = ()         # tool names the turn actually ran (UI badges)


# --------------------------------------------------------------------------- #
# Gene-set enrichment (second interpretation workflow) — the enrichment mirror
# of MarkerEvidence / ClusterVerdict. ``score`` + ``score_kind`` keep the record
# self-describing (the jazzPanda competitive gene-set test). The panel-coverage
# fields are the confident floor's spine (enrichment analog of
# MarkerEvidence.is_on_panel).
# --------------------------------------------------------------------------- #
ScoreKind = Literal["jazzpanda_enrichment"]
EnrichmentTier = Literal["enriched", "suggestive", "untestable"]


@dataclass(frozen=True)
class PathwayEvidence:                      # mirrors MarkerEvidence (per gene set)
    gene_set: str                          # "HALLMARK_G2M_CHECKPOINT"
    gene_set_collection: str               # "MSigDB_Hallmark"
    score: float                           # jazzPanda test_statistic (bigger = more enriched)
    score_kind: ScoreKind
    p_value: Optional[float]
    q_value: Optional[float]               # BH-adjusted (jazzPanda p_adj_bh)
    # panel-coverage grounding spine
    set_size_full: int                     # |set| in the source collection (~200)
    # Set genes EVERY gene capability frame admits -- on the panel AND producing
    # usable spatial signal. NOT the result's own ``n_overlap``: see
    # ``agent/enrichment._load_jazzpanda_rows`` for why the method's word about its
    # own capability is not taken.
    panel_hits: int
    panel_coverage: float                  # panel_hits / set_size_full (0..1)
    leading_edge: tuple[str, ...]          # the driving genes (all on-panel, real)
    n_leading_edge: int                    # honest driving-gene count (gate uses this)
    gc_corr: Optional[float]               # jazzPanda spatial specificity of the leading edge
    tier: EnrichmentTier
    panel_scope_caveat: str                # deterministic "panel-scoped, not genome-wide"
    caveats: tuple[str, ...] = ()
    source: str = "jazzpanda:enrichment"


@dataclass(frozen=True)
class ClusterEnrichment:                   # mirrors ClusterVerdict (per cluster)
    """One cluster's enrichment call. **No band, and nowhere to put one.**

    ``confidence``, ``confidence_score`` and ``demotions`` were deleted here
    (IMPLEMENTATION.md §10, *The one behaviour change with a visible surface*).
    CHARACTERISATION has no label space, so a unit-level band over it is undefined
    rather than merely unhelpful -- there is no alternative for the remaining
    probability mass to sit on. ``demotions`` went with them because it only ever
    held band-change sentences ("-> -1 band"), and an always-empty audit trail for a
    rule that no longer exists is an invitation to refill it.

    :attr:`census_line` is the replacement, and it is strictly more informative than
    the band it replaces: it names how many programs cleared each bar and how many
    the panel could not test at all.

    ``verify`` stays. "A human should look" is well-defined without a label space,
    and it comes from the deterministic rubric in ``agent/enrichment.py``, never from
    a model.
    """

    cluster: str
    cell_type: str                         # from CLUSTER_KEY (not re-derived)
    method: ScoreKind
    enriched: tuple[PathwayEvidence, ...]      # tier=="enriched", score desc
    suggestive: tuple[PathwayEvidence, ...]    # tier=="suggestive" (verify=TRUE)
    all_tested: tuple[PathwayEvidence, ...]    # full audit incl. untestable
    top_theme: Optional[str]               # leading enriched set name, or None
    verify: bool
    source_trace: tuple[str, ...]          # every (set, score, q, K) used

    # --- the census: per-item counts, the header a no-label-space task permits -- #
    # Properties, not fields, because each is a count of a list this object already
    # carries. A stored copy could disagree with the rows under it, and
    # ``dataclasses.asdict`` would then persist the disagreement.
    @property
    def n_untestable(self) -> int:
        """Programs the panel could not test: below the panel-hit or leading-edge floor."""
        return sum(1 for p in self.all_tested if p.tier == "untestable")

    @property
    def census(self) -> tuple[int, int, int]:
        """``(supported, suggestive, untestable)`` -- the three counts, in header order."""
        return (len(self.enriched), len(self.suggestive), self.n_untestable)

    @property
    def census_line(self) -> str:
        """``7 supported \u00b7 3 suggestive \u00b7 12 untestable``: the band's replacement.

        One spelling, here, because several call sites render it (the chat context,
        the report section, the conversation opener) and several spellings of a
        header is how the numbers in them drift apart.
        """
        supported, suggestive, untestable = self.census
        return (f"{supported} supported \u00b7 {suggestive} suggestive "
                f"\u00b7 {untestable} untestable")
