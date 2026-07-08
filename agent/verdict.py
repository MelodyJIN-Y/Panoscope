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
    - ``NoSig`` / near-cutoff / no-canonical-support -> Low.
    - Fragile clusters (<= 2 assigned markers, e.g. c9 Mast) are capped and get
      ``verify = True``; a fragile cluster with a real canonical driver floors at
      Medium rather than being driven to Low by a single demote.
    - glm_coef partly tracks cluster size, so small clusters read lower by design.

Panel-absence notes (the headline catch) are attached for context and NEVER
change the band, the score, or the verify flag. The OFF_PANEL_CANONICAL map is
asserted against ``data.panel_contains`` at import so a mislabelled "off-panel"
gene fails loud instead of silently poisoning a note.
"""

from __future__ import annotations

import csv
import io

from agent import config as cfg
from agent import data
from agent.types import (
    ClusterVerdict,
    LiteratureHook,
    MarkerEvidence,
    OffPanelNote,
    OpeningInterpretation,
)

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

# NoSig / no-marker sentinel band.
LOW_BAND: str = "Low"
FRAGILE_FLOOR_BAND: str = "Medium"  # a fragile cluster with a canonical driver floors here


# --------------------------------------------------------------------------- #
# Canonical marker map — the on-panel canonical markers per cell type.
# Grounded: these are established lineage markers; the driver is whichever of
# these ranks highest by glm_coef among the cluster's assigned (on-panel) rows.
# Only genes actually present in the cluster's marker table can ever drive a
# verdict, so an entry that is off-panel or absent simply never fires.
# --------------------------------------------------------------------------- #
CANONICAL_MARKERS: dict[str, tuple[str, ...]] = {
    "Tumor": ("ERBB2", "EPCAM", "KRT8", "KRT7", "FOXA1", "GATA3", "KRT18"),
    "Stromal": ("LUM", "POSTN", "PDGFRA", "PDGFRB", "DCN", "COL1A1"),
    "Macrophages": ("LYZ", "CD68", "CD163", "ITGAX", "FCGR3A", "FCER1G", "CSF1R"),
    "Myoepithelial": ("MYLK", "ACTA2", "KRT14", "KRT5", "MYH11", "OXTR", "TP63"),
    "T_Cells": ("IL7R", "PTPRC", "TRAC", "CD3E", "CD3D", "CD8A", "CD4"),
    "B_Cells": ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "MZB1"),
    "Endothelial": ("PECAM1", "VWF", "CD93", "AQP1", "CLDN5", "CDH5", "FLT1"),
    "Dendritic": ("LILRA4", "TCL1A", "SPIB", "PLD4", "IL3RA", "CLEC4C", "IRF7"),
    "Mast_Cells": ("CPA3", "TPSAB1", "KIT", "CTSG", "MS4A2", "TPSB2"),
}

# --------------------------------------------------------------------------- #
# Panel-absence notes (P8). P0 populates only c2/Stromal. These are canonical
# markers that were NEVER on the panel: their absence is not evidence against
# the type. Notes are context only — they never touch band/score/verify.
# --------------------------------------------------------------------------- #
OFF_PANEL_CANONICAL: dict[str, tuple[str, ...]] = {
    "Stromal": ("COL1A1", "COL1A2", "DCN", "VIM", "FAP"),
}


def _assert_off_panel_map() -> None:
    """Fail loud if any listed off-panel canonical gene is actually on the panel.

    The whole point of a panel-absence note is that the gene was never measured;
    listing an on-panel gene here would be a grounding error, so we refuse to
    import rather than emit a misleading note.
    """
    for cell_type, genes in OFF_PANEL_CANONICAL.items():
        for gene in genes:
            if data.panel_contains(gene):
                raise AssertionError(
                    f"[verdict] OFF_PANEL_CANONICAL[{cell_type!r}] lists {gene!r} "
                    f"but data.panel_contains({gene!r}) is True — it IS on the panel. "
                    f"An off-panel-absence note must reference a gene that was never "
                    f"measured. Fix the map."
                )


_assert_off_panel_map()


# --------------------------------------------------------------------------- #
# Evidence construction
# --------------------------------------------------------------------------- #
def _is_canonical(gene: str, cell_type: str) -> bool:
    """True iff ``gene`` is a canonical marker for ``cell_type`` (case-insensitive)."""
    canon = CANONICAL_MARKERS.get(cell_type, ())
    return gene.upper() in {g.upper() for g in canon}


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


def _build_evidence(cluster: str, cell_type: str) -> tuple[MarkerEvidence, ...]:
    """Turn a cluster's assigned (on-panel, glm_coef-desc) rows into evidence.

    Assigned rows come from ``data.get_cluster_markers`` (NoSig already excluded,
    sorted glm_coef descending). Every assigned marker is on-panel by
    construction of the jazzPanda top table + our panel-filtered demo pipeline;
    we still record ``is_on_panel`` from the absence primitive to keep the role
    column honest.
    """
    rows = data.get_cluster_markers(cluster)
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
) -> tuple[str, str, tuple[str, ...]]:
    """Return (band, basis, demotions) for the driving canonical marker.

    Base band = ``config.band_for_coef(driver.glm_coef)``. Then apply at most one
    promote (high pearson) and one demote per failing specificity condition. For
    a fragile cluster that has a canonical driver, floor the band at Medium so a
    single specificity demote does not push a real call down to Low (the
    fragility is expressed through ``verify=True`` and the cap, not a hard Low).
    """
    if driver is None:
        # No canonical support -> Low regardless of any non-canonical strength.
        return LOW_BAND, "no canonical driver", ("no canonical marker supports the call",)

    if driver.top_cluster == "NoSig":
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

    Context only — the caller must NOT let these change band/score/verify. Each
    gene is re-asserted off-panel here so a bad entry fails loud at call time too.
    """
    notes: list[OffPanelNote] = []
    for gene in OFF_PANEL_CANONICAL.get(cell_type, ()):
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
            f"{cell_type} — {band} confidence, driven by {d.gene} "
            f"(glm_coef {d.glm_coef:.2f}, pearson {d.pearson:.2f})."
        )
    return f"{cell_type} — {band} confidence; no canonical marker drives the call."


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
        parts.append("verify=TRUE — re-check this call.")
    return " ".join(parts)


def _source_trace(
    evidence: tuple[MarkerEvidence, ...], offpanel: tuple[OffPanelNote, ...]
) -> tuple[str, ...]:
    """Every (gene,stat,value) actually used — grounding tests read this."""
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
    meta = cfg.CLUSTER_KEY[cluster]
    cell_type = meta["cell_type"]

    evidence = _build_evidence(cluster, cell_type)
    n_assigned = len(evidence)
    fragile = n_assigned <= FRAGILE_MARKER_COUNT

    drivers = _driving_markers(evidence)
    driver = drivers[0] if drivers else None

    band, basis, demotions = _compute_band(driver, fragile=fragile)
    score = cfg.SCORE_MAP[band]

    # verify: Low band OR NoSig driver OR fragile (<=2 markers) OR no canonical support.
    driver_is_nosig = driver is not None and driver.top_cluster == "NoSig"
    verify = (band == LOW_BAND) or driver_is_nosig or fragile or (driver is None)

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


def assess(cluster: str) -> ClusterVerdict:
    """Alias for :func:`verdict_for_cluster` (agent-core naming)."""
    return verdict_for_cluster(cluster)


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


if __name__ == "__main__":
    for v in all_verdicts():
        print(f"{v.cluster} {v.cell_type:14s} {v.confidence:12s} verify={v.verify}")
