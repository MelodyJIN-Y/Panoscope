"""Tier-A gene-set enrichment engine (deterministic) — the enrichment mirror of
``agent.verdict``.

Turns a per-cluster gene-set enrichment RESULT into a grounded ``ClusterEnrichment``
per cluster: the enriched pathways, a two-tier split (``enriched`` vs
``suggestive · verify``), a per-item census, and a ``verify`` flag — every value
tracing to the enrichment result or to the study's capability frames. No network,
no LLM.

The method wired here is the biologist's own jazzPanda competitive gene-set test
(``_load_jazzpanda_rows`` parses its result CSV). The schema keeps a ``score_kind``
so the record stays self-describing, but this workflow interprets one method.

**No band.** This workflow answers *what is RUNNING here*, which has no label space,
so a unit-level confidence over a cluster is undefined rather than merely unhelpful.
``ClusterEnrichment`` carries no ``confidence``/``confidence_score``/``demotions``
and this module computes none. The header is ``ClusterEnrichment.census_line`` —
*7 supported · 3 suggestive · 12 untestable*. ``verify`` stays, because "a human
should look" is well-defined without a label space, and the deterministic rubric in
:func:`_cluster_enrichment` is what decides it.

Confident floor (enrichment edition):
- Never invent a pathway, a score, a q-value, or a leading-edge gene — all come
  from the result.
- **Panel-coverage rule:** a Hallmark set has ~200 genes but only ``panel_hits``
  were usably measured here; enrichment is only measured over those. Every row
  carries ``panel_hits``/``panel_coverage`` + a deterministic "panel-scoped, not
  genome-wide" caveat. A set with too few measured genes, or a tiny leading edge,
  is ``untestable`` and never surfaced.
- **Two tiers:** ``enriched`` (q < ENRICH_Q_MAX, leading edge >= MIN_LEADING_EDGE,
  panel_hits >= MIN_PANEL_HITS); ``suggestive`` (same gate but q in
  [ENRICH_Q_MAX, SUGGESTIVE_Q_MAX]) carries ``verify = TRUE``; everything else is
  ``untestable``.
"""

from __future__ import annotations

import csv

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent.types import ClusterEnrichment, PathwayEvidence
from panoscope.methods.jazzpanda_cgst import _in_frame, _split, _to_float
from panoscope.types.subject import EntityKind

# The dataset's jazzPanda enrichment result: prefer the per-dataset tree
# (inputs/enrichment.csv), else the bundled legacy demo file. Absent -> no
# Pathways slice (fail-soft in the pipeline).
def _jz_enrichment_csv():
    """The active study's jazzPanda enrichment result, resolved at CALL time.

    Was a module constant, which froze the study at import AND referenced a private
    config attribute removed in the Study refactor. Because the ternary
    short-circuited under the bundled demo, the breakage was invisible there and
    made the whole Pathways slice unimportable for every other study.

    The bundled demo file is a fallback ONLY for the demo; any other study lacking
    inputs/enrichment.csv simply has no Pathways slice, never the demo's.
    """
    legacy = (
        cfg.DATA_DIR_PATH / "jazzpanda" / "hbc_sp1_top10_hallmark_test_statistic.csv"
        if cfg.DATASET_ID == cfg.BUNDLED_DEMO_ID
        else cfg.current_study().inputs / "enrichment.csv"
    )
    return cfg._active_input("enrichment.csv", legacy)
_JZ_SCORE_KIND = "jazzpanda_enrichment"

#: How a collection names its sets -> the collection. Derivation, not assertion:
#: MSigDB's hallmark release names every one of its sets ``HALLMARK_<program>``, so a
#: result whose set carries that prefix was scored against it. Matched per ROW rather
#: than once per study because the row IS the enrichment input -- which also means a
#: ``PathwayEvidence`` rebuilt from a committed record by ``tests/golden_projection``
#: re-derives the same name with no study loaded and no file read.
_COLLECTION_BY_SET_PREFIX: tuple[tuple[str, str], ...] = (
    ("HALLMARK_", "MSigDB_Hallmark"),
)

#: What a set whose naming matches no entry above is called. ``unknown`` is
#: first-class here and propagates; a wrong collection name does not -- it states
#: false provenance for the row (the confident floor forbids it on its own terms) and,
#: where the sets are invented, it names a real restrictively-licensed collection as
#: the source of rows nobody licensed. Add a prefix entry, never a default.
UNKNOWN_COLLECTION: str = "unknown"


def _collection_for(gene_set: str) -> str:
    """The collection a set was drawn from, read off the set's own name.

    Was ``_JZ_COLLECTION = "MSigDB_Hallmark"``, asserted over every study. On the
    demo that constant happened to be true; on the synthetic fixture it wrote
    "MSigDB_Hallmark" as the provenance of four invented sets (SYNSET_ALPHA/BETA/
    GAMMA/DELTA) into a committed artifact -- naming a real, restrictively-licensed
    collection as the source of rows that came from nowhere, in the study whose
    reason for existing is that nothing licensed is bundled.
    """
    name = gene_set.strip().upper()
    for prefix, collection in _COLLECTION_BY_SET_PREFIX:
        if name.startswith(prefix):
            return collection
    return UNKNOWN_COLLECTION


# --------------------------------------------------------------------------- #
# Panel size (the enrichment universe) — read from the panel file, per study.
# --------------------------------------------------------------------------- #
@data._per_study
def _panel_size() -> int:
    """The analysed panel's gene count, derived and never asserted.

    ``_per_study`` rather than a bare ``lru_cache(maxsize=1)``: a zero-argument
    reader keyed on the empty tuple has ONE slot shared across studies, so opening
    the kidney study after the demo served the demo's 280 into every caveat sentence
    the kidney study wrote. A wrong panel size is not a cosmetic staleness — it is
    the denominator of the panel-coverage rule.
    """
    return int(len(data.load_panel()))


def _panel_scope_caveat(gene_set: str, panel_hits: int, set_size_full: int) -> str:
    """The mandatory, deterministic panel-scoped caveat (not model-generated).

    Says *usably measured*, not *on the panel*. ``panel_hits`` is now the count
    every gene capability frame admits, so a set gene that is on the panel but
    produced no usable spatial signal is not in it; calling that number "on this
    panel" would describe the panel design where the sentence is reporting what this
    section actually measured.
    """
    return (
        f"Panel-scoped: only {panel_hits} of {set_size_full} genes in {gene_set} "
        f"were usably measured on this {_panel_size()}-gene panel (on-panel and "
        f"producing spatial signal); over-representation among measured genes, not "
        f"genome-wide."
    )


# --------------------------------------------------------------------------- #
# jazzPanda enrichment result adapter (the FIRST method).
# --------------------------------------------------------------------------- #
#: The key :func:`_load_jazzpanda_rows` installs on every row, and the ONLY panel-hit
#: count anything downstream is allowed to read. Deliberately not ``n_overlap``: a
#: distinct name means the result's own count can never be mistaken for the
#: frame-checked one by a reader skimming for a familiar column.
PANEL_HITS_IN_FRAME: str = "panel_hits_in_frame"


@data._per_study
def _gene_world():
    """The study's capability frames, as a ``MeasuredWorld``. Declared, not counted here.

    Imported inside the function for the reason ``agent/discriminate.py`` gives: the
    adapter pulls pandas and the ``agent -> platforms`` edge is kept narrow. There is
    no platform registry yet, so the Xenium adapter is named directly — the same gap
    ``pipeline/stages/enrichment.py`` flags.
    """
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.measured import MeasuredWorld

    study = cfg.current_study()
    world = MeasuredWorld(tuple(XeniumAdapter().frames(study)))
    if not world.for_kind(EntityKind.GENE):
        raise ValueError(
            f"[enrichment] study {study.id!r} declares no gene capability frame, so "
            "no set's panel footprint can be counted. Every panel hit would come "
            "back 0 and every program would read as untestable — a study-wide "
            "refusal with a false reason behind it. Fix inputs/panel.parquet (and "
            "inputs/markers_top.csv for detectability) before interpreting."
        )
    return world


@data._per_study
def _load_jazzpanda_rows() -> tuple[dict, ...]:
    """Parse the jazzPanda enrichment result CSV into per-(cluster, set) rows.

    Columns consumed: gene_set, geneset_size, test_statistic, p_value, p_adj_bh,
    gc_corr, genes_tested, genes_selected, cluster. The honest driving genes are
    ``genes_selected`` (the lasso sig_features).

    **Why ``n_overlap`` is not read.** The result carries the method's own count of a
    set's genes on the panel. Taking it would credit a set with genes the measurement
    never usably saw — on the demo study, the 49 of 280 panel genes that yield no
    usable spatial signal — which is the panel-absence rule violated in the
    arithmetic rather than in a sentence. Capability is declared by the apparatus
    before the result is read, so every row's footprint is recounted here against
    EVERY gene frame the study declares, and the count is installed under
    :data:`PANEL_HITS_IN_FRAME`. This is the one place raw rows enter, so there is
    one place the frame is applied.

    ``_in_frame`` and ``_split`` are imported from ``methods/jazzpanda_cgst`` rather
    than re-implemented. The two engines are compared against each other on every
    build (``pipeline/stages/enrichment._report_divergence``), and two copies of the
    counting rule is exactly how that comparison would come to prove nothing.

    Per study, not ``lru_cache(maxsize=1)``: the rows now depend on the study's
    frames as well as on its result file, and one shared slot served the first
    study's counts to the second.
    """
    path = _jz_enrichment_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"[enrichment] jazzPanda enrichment result missing: {path}. "
            f"Place the per-cluster top-set result there."
        )
    world = _gene_world()
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for row in rows:
        if row.get("genes_tested") is None:
            raise KeyError(
                f"[enrichment] {path} has no 'genes_tested' column. Without the "
                "set's tested symbols the panel footprint cannot be checked against "
                "the capability frames, and falling back to 'n_overlap' would take "
                "the method's word for its own capability."
            )
        row[PANEL_HITS_IN_FRAME] = _in_frame(_split(row["genes_tested"]), world)
    return tuple(rows)


def _pathway_from_jz_row(row: dict) -> PathwayEvidence:
    """Build a grounded PathwayEvidence from one jazzPanda result row.

    **The leading edge goes through the same frames ``panel_hits`` does.** The
    result's ``genes_selected`` is the lasso's pick out of the whole panel, taken
    before anything asked whether those genes produced usable spatial signal. Left
    unfiltered, one record said both "only 3 of 36 genes in HALLMARK_ANGIOGENESIS
    were usably measured" and "LPL, LUM, S100A4, STC1 are driving it", with LPL and
    STC1 rejected by the detectability frame -- a count and a gene list disagreeing
    about the same measurement, on 33 of 70 demo rows. A gene the frames reject is
    not a driver: it is not counted, not named, and (``ui/enrichment_spatial``
    reads :attr:`PathwayEvidence.leading_edge` directly) not plotted as one.

    Filtered HERE rather than in :func:`_load_jazzpanda_rows`, where the panel-hit
    recount lives, for one reason: this is the derivation, and the filter is
    idempotent. ``genes_tested`` is a raw column that had to be recounted before a
    stale ``n_overlap`` could be read by mistake, so the frame check is installed on
    the row. The leading edge survives into the record as a field, which means
    ``tests/golden_projection`` reconstructs raw rows out of committed records and
    feeds them straight back through this function; filtering an already-filtered
    list changes nothing, so the projection keeps round-tripping without a seam in
    product code. ``_in_frame`` is still the only frame rule anyone runs -- asked
    one gene at a time, so "every gene frame admits it" cannot drift from the rule
    that produced ``panel_hits``.

    The rejected genes are not dropped silently, but the count does NOT live on
    this object, and that is forced rather than chosen. How many genes the frames
    refused is a fact about the RAW result, and a ``PathwayEvidence`` rebuilt from a
    committed record cannot recover it -- the record carries the admitted genes, so
    recomputing any caveat about the refused ones yields zero. A number that only
    the first build can produce has no business in a field the golden reconstructs
    and re-derives; it would read as "nothing was rejected" on every record that
    came back off disk. It is reported where the raw rows still exist, out loud on
    every build, by ``pipeline/stages/enrichment._report_divergence``. Putting it on
    the record properly needs a persisted ``n_leading_edge_rejected`` field; see the
    report accompanying this change.

    Dropping the rejected genes also pushes rows under ``cfg.MIN_LEADING_EDGE``,
    which is the floor doing its job rather than a regression.
    """
    gene_set = str(row["gene_set"]).strip()
    set_size_full = int(float(row["geneset_size"]))
    if PANEL_HITS_IN_FRAME not in row:
        raise KeyError(
            f"[enrichment] row for {gene_set!r} carries no {PANEL_HITS_IN_FRAME!r}. "
            "That key is installed by _load_jazzpanda_rows after checking the set's "
            "tested genes against the study's capability frames; a row that skipped "
            "it has an unaudited footprint, and reading 'n_overlap' instead would "
            "silently restore the count this change exists to replace."
        )
    panel_hits = int(row[PANEL_HITS_IN_FRAME])   # set genes EVERY gene frame admits
    leading_edge = tuple(
        g for g in _split(row.get("genes_selected", ""))
        if _in_frame((g,), _gene_world())
    )
    n_leading_edge = len(leading_edge)
    score = _to_float(row.get("test_statistic")) or 0.0
    p_value = _to_float(row.get("p_value"))
    q_value = _to_float(row.get("p_adj_bh"))
    gc_corr = _to_float(row.get("gc_corr"))
    coverage = (panel_hits / set_size_full) if set_size_full else 0.0

    # Tier: the report gate + two tiers (untestable never surfaced as a theme).
    if panel_hits < cfg.MIN_PANEL_HITS or n_leading_edge < cfg.MIN_LEADING_EDGE:
        tier = "untestable"
    elif q_value is not None and q_value < cfg.ENRICH_Q_MAX:
        tier = "enriched"
    elif q_value is not None and q_value <= cfg.SUGGESTIVE_Q_MAX:
        tier = "suggestive"
    else:
        tier = "untestable"

    caveats: list[str] = []
    if coverage < 0.05:
        caveats.append("very low panel coverage")
    if n_leading_edge < 5:
        caveats.append("tiny leading edge")
    if gc_corr is not None and gc_corr < 0.30:
        caveats.append("low spatial specificity")

    return PathwayEvidence(
        gene_set=gene_set,
        gene_set_collection=_collection_for(gene_set),
        score=score,
        score_kind=_JZ_SCORE_KIND,
        p_value=p_value,
        q_value=q_value,
        set_size_full=set_size_full,
        panel_hits=panel_hits,
        panel_coverage=coverage,
        leading_edge=leading_edge,
        n_leading_edge=n_leading_edge,
        gc_corr=gc_corr,
        tier=tier,
        panel_scope_caveat=_panel_scope_caveat(gene_set, panel_hits, set_size_full),
        caveats=tuple(caveats),
        source="jazzpanda:enrichment",
    )


# --------------------------------------------------------------------------- #
# The per-cluster call (two-tier + census + verify), method-agnostic.
#
# There is deliberately no band ladder here. The one that stood in this place
# ("_band_cluster": band_for_enrichment on the top set's score, demoted for low
# coverage or a tiny leading edge) was a unit-level confidence on a CHARACTERISATION,
# which has no label space — the category error IMPLEMENTATION.md §10 sequenced for
# deletion and ``methods/jazzpanda_cgst.rubric`` already refuses to produce. The
# per-item counts on ``ClusterEnrichment.census_line`` replace it.
# --------------------------------------------------------------------------- #
def _cluster_enrichment(
    cluster: str, pathways: tuple[PathwayEvidence, ...], method: str
) -> ClusterEnrichment:
    """Assemble a ClusterEnrichment from that cluster's PathwayEvidence rows."""
    cell_type = annotation.cell_type_for(cluster)
    enriched = tuple(
        sorted((p for p in pathways if p.tier == "enriched"), key=lambda p: p.score, reverse=True)
    )
    suggestive = tuple(
        sorted((p for p in pathways if p.tier == "suggestive"), key=lambda p: p.score, reverse=True)
    )
    top = enriched[0] if enriched else None
    # The deterministic verify rubric. Unchanged by the band's removal, because it
    # never depended on one: verify when nothing clears the enriched bar, or when the
    # top call rests on too little of its set. The model is never offered the choice.
    verify = (
        not enriched
        or (top is not None and (top.panel_coverage < 0.05 or top.n_leading_edge < cfg.MIN_LEADING_EDGE))
    )
    source_trace = tuple(
        f"{p.gene_set}: score={p.score:.2f} q={p.q_value} K={p.panel_hits}/{p.set_size_full} LE={p.n_leading_edge}"
        for p in pathways
    )
    return ClusterEnrichment(
        cluster=cluster,
        cell_type=cell_type,
        method=method,
        enriched=enriched,
        suggestive=suggestive,
        all_tested=pathways,
        top_theme=top.gene_set if top else None,
        verify=bool(verify),
        source_trace=source_trace,
    )


# --------------------------------------------------------------------------- #
# Public API (mirrors agent.verdict)
# --------------------------------------------------------------------------- #
def enrichment_for_cluster(cluster: str) -> ClusterEnrichment:
    """Return the jazzPanda-method enrichment verdict for one cluster.

    Raises KeyError for an unknown cluster id. Reads the biologist's enrichment
    result; computes nothing new — tiers the reported sets and counts them.
    """
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(f"[enrichment] unknown cluster {cluster!r}")
    rows = [r for r in _load_jazzpanda_rows() if str(r.get("cluster")) == cluster]
    pathways = tuple(_pathway_from_jz_row(r) for r in rows)
    return _cluster_enrichment(cluster, pathways, _JZ_SCORE_KIND)


def load_or_compute(cluster: str) -> ClusterEnrichment:
    """The persisted enrichment record for ``cluster`` if the tree has one, else
    computed live by :func:`enrichment_for_cluster`."""
    from panoscope.pipeline import store  # lazy: pipeline imports agent

    persisted = store.load_enrichment(cluster)
    return persisted if persisted is not None else enrichment_for_cluster(cluster)


def all_enrichments() -> list[ClusterEnrichment]:
    """Enrichment verdicts for all nine clusters (c1..c9), in order."""
    return [enrichment_for_cluster(c) for c in cfg.CLUSTER_ORDER]
