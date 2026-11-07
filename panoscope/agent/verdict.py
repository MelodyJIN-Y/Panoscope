"""Deterministic cell-type verdict engine.

Pure function of jazzPanda's precomputed marker output (via ``agent.data``) and
the authoritative cluster key + confidence bands (via ``agent.config``). No
network, no LLM, no invented numbers: every value in a ``ClusterVerdict`` traces
back to a jazzPanda column, the panel list, or the cluster key.

Confidence rubric (user override, supersedes the percentile talk in BLUEPRINT):
    confidence is ``glm_coef`` DIRECT. The driving canonical marker's ``glm_coef``
    fed to ``config.band_for_coef`` gives the base band (bigger glm_coef -> more
    confident). There is NO within-cluster percentile.

    - ``pearson`` corroborates: a high pearson on the driver can PROMOTE one band.
    - ``max_gc_corr > pearson`` (localizes better elsewhere), a low pearson, or
      ``max_gg_corr ~= 1`` (spatial pattern not unique) each DEMOTE one band.
    - Canonical COVERAGE demotes once: K of N of the label's canonical markers
      that a capability frame admitted (on panel AND detectable) against a
      declared floor of one in three. This is the only modifier that is about the
      APPARATUS rather than about the driver, and it also sets ``verify``.
    - ``NoSig`` / near-cutoff / no-canonical-support -> Low.
    - Fragile clusters (<= 2 assigned markers, e.g. c9 Mast) are capped and get
      ``verify = True``; a fragile cluster with a real canonical driver floors at
      Medium rather than being driven to Low by a single demote.
    - glm_coef partly tracks cluster size, so small clusters read lower by design.

Panel-absence notes (the headline catch) are attached for context and NEVER
change the band, the score, or the verify flag. The OFF_PANEL_CANONICAL map is
asserted against ``data.panel_contains`` at import so a mislabelled "off-panel"
gene fails loud instead of silently poisoning a note.

**Notes are per-GENE; coverage is per-LABEL, and only the second reaches the band.**
The notes above say "ERBB2 was never measured, so its absence argues nothing", which
is a statement about one gene and must not move a number. Canonical coverage is a
different quantity: how much of the label's curated support the measurement was
capable of speaking to AT ALL. Without it the rubric had a cliff at one. A unit with
one canonical marker on panel and five never measured read Very High with
``verify=False``, and the off-panel notes sat beside that number saying exactly how
little had been measurable while the number ignored them. On a panel designed for a
different question than the tissue -- an immuno-oncology panel against kidney -- most
resident types land on zero or one, and zero was already handled honestly. One was
not. Coverage still supplies no value: the band is the driver's own ``glm_coef``, and
coverage can only take a rung away and ask for a second look.
"""

from __future__ import annotations

import csv
import io

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent.types import (
    ClusterVerdict,
    LiteratureHook,
    MarkerEvidence,
    OffPanelNote,
    OpeningInterpretation,
)
from panoscope.platforms.xenium import is_no_signal
from panoscope.types.measured import Coverage

# --------------------------------------------------------------------------- #
# Band ladder + demote/promote thresholds (grounded in SKILL rubric)
# --------------------------------------------------------------------------- #
BANDS: tuple[str, ...] = ("Very High", "High", "Medium-High", "Medium", "Low")

# pearson corroboration thresholds (SKILL Step 3d modifiers)
PROMOTE_PEARSON: float = 0.85   # high spatial specificity on the driver -> +1 band
LOW_PEARSON: float = 0.30       # low spatial specificity on the driver -> -1 band
GG_NOT_UNIQUE: float = 0.98     # spatial pattern ~= another gene's -> -1 band
EPS: float = 1e-9

# Fragile-cluster rule: a cluster with this many assigned markers or fewer is
# capped and always carries verify=True (SKILL: "one or two assigned markers").
FRAGILE_MARKER_COUNT: int = 2

# Canonical-COVERAGE floor: K of N of the label's canonical markers that a
# capability frame admitted (on panel AND detectable), as an exact integer ratio so
# no float epsilon decides a band. Duplicated in core/verdict.py under the same
# name; tests/test_core_verdict.py fails the moment the two diverge.
#
# One in three. Two thirds of a curated marker list may be off-panel or undetectable
# and the call still stands on its own number; below that the panel was not asking
# this question, and a band read off one surviving marker is a confident answer to
# a question the measurement could not pose. The band is still the driver's number
# -- coverage only subtracts.
CANONICAL_COVERAGE_FLOOR: tuple[int, int] = (1, 3)
CANONICAL_COVERAGE_WHAT: str = "canonical markers in frame"

# NoSig / no-marker sentinel band.
LOW_BAND: str = "Low"
FRAGILE_FLOOR_BAND: str = "Medium"  # a fragile cluster with a canonical driver floors here


# --------------------------------------------------------------------------- #
# The canonical / off-panel marker maps used to live here, duplicated.
#
# They were dead at runtime: ``_is_canonical`` reads
# ``annotation.canonical_markers()``, and ``offpanel_notes`` reads
# ``annotation.offpanel_canonical()``. Worse, the import-time assertion that
# guarded them validated the UNUSED copy, and did filesystem I/O at import to do
# it. Both are gone. The live maps are in agent/annotation.py, and the invariant
# is now a test (tests/test_verdict.py) asserted against the map actually used.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Evidence construction
# --------------------------------------------------------------------------- #
def _is_canonical(gene: str, cell_type: str) -> bool:
    """True iff ``gene`` is a canonical marker for ``cell_type`` (case-insensitive)."""
    canon = annotation.canonical_markers(cell_type)
    return gene.upper() in {g.upper() for g in canon}


def _detectable_genes() -> frozenset[str]:
    """Panel genes that produced a usable spatial signal, upper-cased.

    jazzPanda's ``NoSig`` sentinel is the detectability frame in v1 vocabulary: a
    gene it names but assigns nowhere was looked at and yielded nothing usable, and
    49 of the demo panel's 280 are in that state. Derived from the table every time,
    never a constant -- the count is a property of the study.
    """
    m = data.load_markers()
    return frozenset(
        m.loc[~m["top_cluster"].map(is_no_signal), "gene"]
         .astype(str).str.strip().str.upper()
    )


def coverage_floor(n: int) -> int:
    """Smallest K that clears :data:`CANONICAL_COVERAGE_FLOOR` out of ``n``.

    ``ceil(n * num / den)`` in integer arithmetic, so no float comparison decides a
    band. Transcribed from ``core/verdict._coverage_floor``.
    """
    num, den = CANONICAL_COVERAGE_FLOOR
    return -(-n * num // den)


def canonical_coverage(cell_type: str) -> Coverage:
    """K of N canonical markers for ``cell_type`` that were CAPABLE of being seen.

    N is the label's whole curated list; K is the subset a capability frame admits
    -- on the panel AND detectable. Both frames must admit it: an off-panel gene was
    never measured, and an on-panel gene with no usable signal was equally never
    measured, and neither absence argues anything about this unit.

    This is reference data selecting which question the measurement is asked, which
    it may do. It supplies no value: the band is still read off the driver's own
    ``glm_coef``, and coverage can only take a rung away.
    """
    canon = annotation.canonical_markers(cell_type)
    detectable = _detectable_genes()
    k = sum(1 for g in canon
            if data.panel_contains(g) and g.upper() in detectable)
    n = len(canon)
    return Coverage(CANONICAL_COVERAGE_WHAT, k, n, coverage_floor(n))


def _coverage_sentence(cov: Coverage) -> str:
    """The v1 English for the coverage demote.

    The other copy is ``core/verdict._v1_coverage_sentence``; ``core/`` may not be
    imported from under ``agent/``, and the two are pinned equal by a test rather
    than by hope.
    """
    return (f"canonical coverage {cov.k}/{cov.n} below floor {cov.floor} "
            f"-> -1 band")


def _caveats_for(pearson: float, max_gc_corr: float, max_gg_corr: float) -> tuple[str, ...]:
    """Grounded caveats from the jazzPanda specificity columns (SKILL Step 3a)."""
    caveats: list[str] = []
    if max_gc_corr > pearson + EPS:
        caveats.append("localizes better with another cluster")
    if max_gg_corr >= GG_NOT_UNIQUE:
        caveats.append("spatial pattern not unique")
    if pearson < LOW_PEARSON:
        caveats.append("low spatial specificity")
    return tuple(caveats)


def _in_the_gene_frame(gene: str) -> bool:
    """The capability frame for a gene, in the one form this engine can express.

    Transcribed from the guard in ``core/verdict.py``, which is a frame refusal
    rather than a function: ``core.measure`` resolves every reading against the
    platform's gene frames, a gene the panel frame does not admit comes back
    ``NEVER_MEASURED`` with its statistic withheld, and
    ``EvidenceItem.__post_init__`` then makes a row that both carries a number and
    names an unmeasured subject structurally impossible. ``agent/`` may not import
    ``core/`` -- the reason is written out at ``core/verdict._v1_coverage_sentence``
    -- so the guard is duplicated here exactly as the band constants are, and
    ``tests/test_core_verdict.py`` fails the moment the two copies refuse different
    genes.

    Detectability, core's second gene frame, needs no clause here: a row that
    reaches this function was assigned to a cluster, which is precisely what
    :func:`_detectable_genes` counts, and ``get_cluster_markers`` has already
    dropped the ``NoSig`` rows that fail it.
    """
    return data.panel_contains(gene)


def _build_evidence(cluster: str, cell_type: str) -> tuple[MarkerEvidence, ...]:
    """Turn a cluster's assigned (on-panel, glm_coef-desc) rows into evidence.

    Assigned rows come from ``data.get_cluster_markers`` (NoSig already excluded,
    sorted glm_coef descending), and :func:`_in_the_gene_frame` then drops the ones
    the panel frame does not admit.

    **That filter is not a tidy-up, it is the headline rule.** A row naming
    something the panel does not hold -- a control probe, a gene carried over from a
    table built against another panel -- is a subject this measurement never
    measured, so quoting its ``glm_coef`` attaches a real number to a
    ``NEVER_MEASURED`` thing, and ``is_on_panel=False`` beside that number says so
    in the same row. The method table cannot catch it: jazzPanda reports what it was
    handed, and what the panel could speak to is the adapter's answer rather than
    the method's.

    Every surviving row is therefore on-panel and ``is_on_panel`` below is always
    True. It is still read from the absence primitive rather than written as a
    constant, because the column's job is to record what the frame said.
    """
    rows = data.get_cluster_markers(cluster)
    rows = rows[rows["gene"].astype(str).map(_in_the_gene_frame)]
    n = len(rows)
    evidence: list[MarkerEvidence] = []
    for rank, (_, r) in enumerate(rows.iterrows()):
        gene = str(r["gene"])
        pearson = float(r["pearson"])
        max_gc = float(r["max_gc_corr"])
        max_gg = float(r["max_gg_corr"])
        canonical = _is_canonical(gene, cell_type)
        on_panel = data.panel_contains(gene)
        caveats = _caveats_for(pearson, max_gc, max_gg)
        # role: canonical + spatially specific -> supports; canonical but weak ->
        # a real down-weight (expected_absent); non-canonical on-panel -> supports.
        if canonical and pearson >= LOW_PEARSON:
            role = "supports"
        elif canonical:
            role = "expected_absent"
        else:
            role = "supports"
        # within-cluster percentile kept for UI/audit only (1.0 = strongest by
        # glm_coef rank); it is NOT used to set the band under the direct rubric.
        pctile = 1.0 if n <= 1 else (n - 1 - rank) / (n - 1)
        evidence.append(
            MarkerEvidence(
                gene=gene,
                top_cluster=str(r["top_cluster"]),
                glm_coef=float(r["glm_coef"]),
                pearson=pearson,
                max_gg_corr=max_gg,
                max_gc_corr=max_gc,
                p_value=None,
                within_cluster_pctile=pctile,
                is_canonical=canonical,
                is_on_panel=on_panel,
                role=role,
                caveats=caveats,
            )
        )
    return tuple(evidence)


# --------------------------------------------------------------------------- #
# Band computation (glm_coef DIRECT + pearson/gc/gg modifiers)
# --------------------------------------------------------------------------- #
def _demote(band: str) -> str:
    return BANDS[min(BANDS.index(band) + 1, len(BANDS) - 1)]


def _promote(band: str) -> str:
    return BANDS[max(BANDS.index(band) - 1, 0)]


def _compute_band(
    driver: MarkerEvidence | None,
    *,
    fragile: bool,
    coverage: Coverage,
) -> tuple[str, str, tuple[str, ...]]:
    """Return (band, basis, demotions) for the driving canonical marker.

    Base band = ``config.band_for_coef(driver.glm_coef)``. Then apply at most one
    promote (high pearson) and one demote per failing specificity condition, then
    one demote if too little of the label's canonical list was ever in frame. For
    a fragile cluster that has a canonical driver, floor the band at Medium so a
    single specificity demote does not push a real call down to Low (the
    fragility is expressed through ``verify=True`` and the cap, not a hard Low).

    ``coverage`` is a REQUIRED keyword and has no default. A default of ``None``
    would mean every caller that forgot it silently got the uncapped band back --
    which is the cliff this parameter exists to remove, reintroduced as a soft
    default. A caller that cannot produce a Coverage should fail here, loudly.
    """
    if driver is None:
        # No canonical support -> Low regardless of any non-canonical strength.
        # No coverage demote: the band is already at the floor, and a demote that
        # moved nothing would read as a rung the call lost.
        return LOW_BAND, "no canonical driver", ("no canonical marker supports the call",)

    if is_no_signal(driver.top_cluster) and driver.top_cluster is not None:
        return LOW_BAND, "driver NoSig", ("driving marker is NoSig",)

    band = cfg.band_for_coef(driver.glm_coef)
    basis = "glm_coef direct"
    changes: list[str] = []

    # Promote once on strong spatial specificity.
    if driver.pearson >= PROMOTE_PEARSON:
        promoted = _promote(band)
        if promoted != band:
            band = promoted
            changes.append(f"high pearson on driver ({driver.pearson:.2f}) -> +1 band")

    # Demote once per failing specificity condition.
    if driver.pearson < LOW_PEARSON:
        band = _demote(band)
        changes.append(f"low pearson on driver ({driver.pearson:.2f}) -> -1 band")
    if driver.max_gc_corr > driver.pearson + EPS:
        band = _demote(band)
        changes.append("driver localizes better with another cluster -> -1 band")
    if driver.max_gg_corr >= GG_NOT_UNIQUE:
        band = _demote(band)
        changes.append("driver spatial pattern not unique -> -1 band")

    # The condition that is about the APPARATUS rather than about the driver.
    if coverage.n and not coverage.ok:
        band = _demote(band)
        changes.append(_coverage_sentence(coverage))

    if fragile:
        basis = "glm_coef direct (fragile cap)"
        # Cap: a fragile cluster cannot read above Medium-High.
        if BANDS.index(band) < BANDS.index("Medium-High"):
            band = "Medium-High"
            changes.append("fragile cluster (<=2 markers): capped at Medium-High")
        # Floor: a fragile cluster with a real canonical driver stays >= Medium.
        if BANDS.index(band) > BANDS.index(FRAGILE_FLOOR_BAND):
            band = FRAGILE_FLOOR_BAND
            changes.append("fragile cluster with canonical driver: floored at Medium")

    return band, basis, tuple(changes)


def _driving_markers(evidence: tuple[MarkerEvidence, ...]) -> tuple[MarkerEvidence, ...]:
    """All canonical supporters, glm_coef-desc (the driver is the first)."""
    return tuple(e for e in evidence if e.is_canonical and e.role == "supports")


# --------------------------------------------------------------------------- #
# Panel-absence notes
# --------------------------------------------------------------------------- #
def offpanel_notes(cell_type: str) -> tuple[OffPanelNote, ...]:
    """Panel-absence notes for a cell type: canonical markers never measured.

    Context only, the caller must NOT let these change band/score/verify. Each
    gene is re-asserted off-panel here so a bad entry fails loud at call time too.
    """
    notes: list[OffPanelNote] = []
    for gene in annotation.offpanel_canonical(cell_type):
        assert not data.panel_contains(gene), (
            f"[verdict] {gene} is on the panel; cannot be an off-panel note"
        )
        notes.append(
            OffPanelNote(
                gene=gene,
                cell_type=cell_type,
                message=(
                    f"{gene} is off-panel (never measured); its absence is not "
                    f"evidence against {cell_type}."
                ),
            )
        )
    return tuple(notes)


# --------------------------------------------------------------------------- #
# Literature hooks + opening interpretation
# --------------------------------------------------------------------------- #
def _literature_hooks(
    cell_type: str, drivers: tuple[MarkerEvidence, ...]
) -> tuple[LiteratureHook, ...]:
    """One unfilled hook per driving marker: WHAT to look up, no citations here."""
    hooks: list[LiteratureHook] = []
    for d in drivers:
        hooks.append(
            LiteratureHook(
                claim=f"{d.gene} marks {cell_type}",
                marker=d.gene,
                cell_type=cell_type,
                query_terms=(d.gene, cell_type.replace("_", " "), "marker"),
            )
        )
    return tuple(hooks)


def _headline(cell_type: str, band: str, drivers: tuple[MarkerEvidence, ...]) -> str:
    if drivers:
        d = drivers[0]
        return (
            f"{cell_type}, {band} confidence, driven by {d.gene} "
            f"(glm_coef {d.glm_coef:.2f}, pearson {d.pearson:.2f})."
        )
    return f"{cell_type}, {band} confidence; no canonical marker drives the call."


def opening_interpretation(cluster: str) -> OpeningInterpretation:
    """Build the opening interpretation posted before any question.

    Carries the call, the confidence, the driving canonical markers with their
    real glm_coef/pearson numbers, the panel-absence notes, and unfilled
    literature hooks (the loop fills citations live). No number is invented.
    """
    return verdict_for_cluster(cluster).opening


# --------------------------------------------------------------------------- #
# Notes / trace composition
# --------------------------------------------------------------------------- #
def _compose_notes(
    cell_type: str,
    band: str,
    drivers: tuple[MarkerEvidence, ...],
    demotions: tuple[str, ...],
    fragile: bool,
    verify: bool,
) -> str:
    parts: list[str] = []
    if drivers:
        driver_bits = ", ".join(
            f"{d.gene} glm_coef={d.glm_coef:.2f}/pearson={d.pearson:.2f}"
            for d in drivers[:3]
        )
        parts.append(f"{cell_type} call driven by {driver_bits}.")
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


def _source_trace(
    evidence: tuple[MarkerEvidence, ...], offpanel: tuple[OffPanelNote, ...]
) -> tuple[str, ...]:
    """Every (gene,stat,value) actually used, grounding tests read this."""
    trace: list[str] = []
    for e in evidence:
        trace.append(f"jz:{e.gene}:glm_coef={e.glm_coef:.6f}")
        trace.append(f"jz:{e.gene}:pearson={e.pearson:.6f}")
    for note in offpanel:
        trace.append(f"panel:{note.gene}:off_panel=True")
    return tuple(trace)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def verdict_for_cluster(cluster: str) -> ClusterVerdict:
    """Deterministic cell-type verdict for one cluster.

    Pure function of ``agent.data`` + ``agent.config``. Cell type comes straight
    from the cluster key. Confidence is the driving canonical marker's glm_coef
    band with pearson/max_gc/max_gg modifiers. Panel-absence notes are attached
    but never alter band/score/verify. KeyError if ``cluster`` is unknown.
    """
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(
            f"[verdict] unknown cluster {cluster!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}"
        )
    meta = annotation.meta_for(cluster)
    cell_type = meta["cell_type"]

    evidence = _build_evidence(cluster, cell_type)
    n_assigned = len(evidence)
    fragile = n_assigned <= FRAGILE_MARKER_COUNT

    drivers = _driving_markers(evidence)
    driver = drivers[0] if drivers else None

    coverage = canonical_coverage(cell_type)
    band, basis, demotions = _compute_band(driver, fragile=fragile,
                                           coverage=coverage)
    score = cfg.SCORE_MAP[band]

    # verify: Low band OR NoSig driver OR fragile (<=2 markers) OR no canonical
    # support OR too little of the label's canonical list ever in frame. The last
    # is why the coverage demote is not the whole fix: one rung off Very High is
    # still High, which reads as settled, and a call the panel could barely ask for
    # has to say re-check this.
    driver_is_nosig = (driver is not None and driver.top_cluster is not None
                       and is_no_signal(driver.top_cluster))
    verify = ((band == LOW_BAND) or driver_is_nosig or fragile or (driver is None)
              or (bool(coverage.n) and not coverage.ok))

    offpanel = offpanel_notes(cell_type)

    key_markers = tuple(e.gene for e in evidence[:5])

    hooks = _literature_hooks(cell_type, drivers)
    opening = OpeningInterpretation(
        cluster=cluster,
        cell_type=cell_type,
        confidence=band,
        headline=_headline(cell_type, band, drivers),
        driving_markers=drivers,
        offpanel_notes=offpanel,
        literature_hooks=hooks,
        verify=verify,
    )

    return ClusterVerdict(
        cluster=cluster,
        cell_type=cell_type,
        cell_type_short=meta["cell_type_short"],
        confidence=band,
        confidence_score=score,
        key_markers=key_markers,
        notes=_compose_notes(cell_type, band, drivers, demotions, fragile, verify),
        category=meta["category"],
        lineage=meta["lineage"],
        exclude=False,
        verify=verify,
        small_n=fragile,
        evidence=evidence,
        offpanel_notes=offpanel,
        opening=opening,
        band_basis=basis,
        demotions=demotions,
        source_trace=_source_trace(evidence, offpanel),
    )


def all_verdicts() -> list[ClusterVerdict]:
    """Verdicts for c1..c9 in cluster order."""
    return [verdict_for_cluster(c) for c in cfg.CLUSTER_ORDER]


# --------------------------------------------------------------------------- #
# CSV export (the 11-column contract, in exact order)
# --------------------------------------------------------------------------- #
CSV_COLUMNS: tuple[str, ...] = (
    "cluster",
    "cell_type",
    "cell_type_short",
    "confidence",
    "confidence_score",
    "key_markers",
    "notes",
    "category",
    "lineage",
    "exclude",
    "verify",
)


def to_csv_row(v: ClusterVerdict) -> dict[str, object]:
    """Return one verdict as an ordered dict of the 11 CSV columns.

    ``key_markers`` is joined with ';'. Booleans render as TRUE/FALSE to match
    the R-importable CSV convention in the SKILL.
    """
    return {
        "cluster": v.cluster,
        "cell_type": v.cell_type,
        "cell_type_short": v.cell_type_short,
        "confidence": v.confidence,
        "confidence_score": v.confidence_score,
        "key_markers": ";".join(v.key_markers),
        "notes": v.notes,
        "category": v.category,
        "lineage": v.lineage,
        "exclude": "TRUE" if v.exclude else "FALSE",
        "verify": "TRUE" if v.verify else "FALSE",
    }


def to_csv(verdicts: list[ClusterVerdict], header: bool = True) -> str:
    """Render verdicts to the 11-column CSV string (exact column order)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS))
    if header:
        writer.writeheader()
    for v in verdicts:
        writer.writerow(to_csv_row(v))
    return buf.getvalue()
