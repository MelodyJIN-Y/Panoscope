"""The dedicated interpretation summary — assembled, reviewable, downloadable.

A per-dataset report the biologist reviews on the Summary page and downloads as
.docx and PDF. Assembled entirely from durable, grounded artifacts (verdicts,
cited notes, the settle line, the biologist's notes, the holistic review); no
network and no recomputation at build time. ``build_report`` is pure; Streamlit
and ``ui.data_access`` are imported lazily.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass
from typing import Any, Optional

from panoscope.agent import config as cfg
from panoscope.agent import discriminate
from panoscope.agent.enrichment_themes import pretty_gene_set
from panoscope.agent.types import ClusterVerdict, MarkerEvidence, Note

# --------------------------------------------------------------------------- #
# Human labels for the closed note vocabularies (relabel only; never change one).
# --------------------------------------------------------------------------- #
_SCOPE_LABEL = {"cluster": "cluster", "dataset": "this dataset", "lab": "all datasets"}
_BASIS_LABEL = {"paper": "a paper", "own_validation": "our own data", "convention": "convention"}
_STATUS_LABEL = {"firm": "firm rule", "tentative": "tentative"}


# --------------------------------------------------------------------------- #
# Report model — plain, serializable, no I/O
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoteLine:
    claim: str
    scope: str
    basis: str
    status: str
    tension: str  # "" when none
    author: str
    date: str


@dataclass(frozen=True)
class ClusterSection:
    cluster: str
    cell_type: str
    confidence: str
    verify: bool
    driver_line: str  # "ERBB2 (glm_coef 21.44, pearson 0.59), ..." or key markers
    biology: str
    biology_pmid: str  # "" when none
    settle: str        # "" unless there is a genuine rival to settle
    notes: tuple[NoteLine, ...]
    enrichment: str = ""             # plain-text enriched-programs block (with PMIDs), "" if none
    enrichment_census: str = ""      # "N supported · N suggestive · N untestable", "" if none
    #: The record's own undecidable sentences, verbatim.
    undecidable: tuple[str, ...] = ()
    #: Withdrawn by the biologist: rendered with the withdrawal stated, never dropped.
    excluded: bool = False
    lineage: str = ""                # for the overview table
    key_markers: str = ""            # top gene names (no numbers), for the overview table
    #: Panel-absence rule: ``(gene, message)`` for every canonical marker of this
    #: call the panel never measured. Plain pairs, so renderers see only this.
    off_panel: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReportModel:
    dataset: str
    generated_at: str  # passed in; "" for a deterministic (test) build
    n_clusters: int
    n_flagged: int
    panel_size: int
    sections: tuple[ClusterSection, ...]
    coherence_notes: tuple[str, ...]
    refinements: tuple[str, ...]
    dataset_notes: tuple[NoteLine, ...]  # dataset/lab-scoped notes, shown once
    tissue: str = ""                     # header context (from the manifest)
    platform: str = ""                   # header context (from the manifest)
    method: str = "jazzPanda spatial markers"
    # ARCHITECTURE §14 provenance. Each field states what it is or that it was not
    # consulted; a blank would read as "checked and fine".
    #: ``(frame name, digest)`` for every capability frame the reading stood on.
    frames: tuple[tuple[str, str], ...] = ()
    #: One line per registered reference source, or empty -- rendered as "not
    #: consulted" rather than omitted.
    reference_releases: tuple[str, ...] = ()
    #: Corpus version + staleness, or why there is none.
    corpus_status: str = ""
    #: Generator model + how it was reached. Deterministic builds say so.
    generator: str = ""
    #: ``core.record.SPEC_VERSION`` -- which contract wrote this document.
    spec_version: str = ""

    #: How many panel genes produced no usable spatial signal (detectability frame).
    panel_no_signal: int = 0


# --------------------------------------------------------------------------- #
# Assembly (pure)
# --------------------------------------------------------------------------- #
def _pmids(cites) -> str:
    return ", ".join(f"PMID:{c.pmid}" for c in cites if getattr(c, "pmid", ""))


def _tension_text(note: Note) -> str:
    """Literature-tension summary off the stored note, or '' when there is none."""
    t = note.tension
    if t.agree or t.dissent:
        bits = []
        if t.agree:
            bits.append(f"agrees ({len(t.agree)}): {_pmids(t.agree)}")
        if t.dissent:
            bits.append(f"dissents ({len(t.dissent)}): {_pmids(t.dissent)}")
        return "literature tension: " + " · ".join(bits)
    # ``lookup_state`` separates "searched, found nothing" from "never looked up".
    from panoscope.claims.notes import Resolution, lookup_state

    state = lookup_state(t)
    if state is Resolution.NOT_FOUND:
        return "literature thin: searched, no supporting reference found"
    if state is Resolution.NOT_LOOKED_UP:
        return "literature not checked: no lookup completed for this note"
    return ""


def note_scope_label(note: Note) -> str:
    """The note's scope as one readable span; a lab-scoped note names the study that
    formed it, so its conditions travel with it. Shared by the report and the board.
    """
    if note.scope == "cluster" and note.scope_ref.cluster:
        return f"cluster {note.scope_ref.cluster}"
    label = _SCOPE_LABEL.get(note.scope, note.scope)
    if note.scope != "lab":
        return label
    if not note.scope_ref.dataset:
        return f"{label} · formation study not recorded"
    return f"{label} · formed on {note.scope_ref.dataset}"


def _note_line(note: Note) -> NoteLine:
    return NoteLine(
        claim=note.claim,
        scope=note_scope_label(note),
        basis=_BASIS_LABEL.get(note.basis, note.basis),
        status=_STATUS_LABEL.get(note.status, note.status),
        tension=_tension_text(note),
        author=note.author or "you",
        date=(note.created_at.split("T", 1)[0] if note.created_at else "n/a"),
    )


def _driver_line(v: ClusterVerdict) -> str:
    """Driving markers with real numbers, or the key-marker names if none drive."""
    drivers: tuple[MarkerEvidence, ...] = v.opening.driving_markers
    if drivers:
        return ", ".join(
            f"{m.gene} (glm_coef {m.glm_coef:.2f}, pearson {m.pearson:.2f})" for m in drivers[:4]
        )
    return ", ".join(v.key_markers) if v.key_markers else "no canonical driver"


def _settle_for(cluster: str, verify: bool, cell_type: str = "") -> str:
    """The 'what would settle it' summary, only when there is a genuine rival.

    Withheld (not recomputed) when an override replaced the call the discrimination
    was computed for.
    """
    d = discriminate.discriminate(cluster)
    computed_for = str(getattr(d, "call_A", "") or "").strip().casefold()
    if cell_type and computed_for and computed_for != str(cell_type).strip().casefold():
        return ""
    if d.alt_B is not None and (verify or d.b_here):
        return discriminate.settle_summary(d)
    return ""


def _n_no_signal() -> int:
    """How many panel genes produced no usable spatial signal, counted off the method
    table per study (never asserted); 0 when unreadable."""
    try:
        from panoscope.agent import data as _data
        from panoscope.platforms.xenium import is_no_signal

        return sum(1 for t in _data.load_markers()["top_cluster"]
                   if is_no_signal(str(t)))
    except Exception:  # noqa: BLE001 - unreadable method table: say nothing
        return 0


def _celltype_note_for(celltype_notes: dict, v) -> dict:
    """This section's cell-type note, or ``{}`` when it describes another call.

    A citation must never be attributed to a call it was not written about, so an
    overridden call gets no biology rather than the old call's.
    """
    note = celltype_notes.get(getattr(v, "cluster", "")) or {}
    if not note:
        return {}
    described = str(note.get("cell_type") or "").strip().casefold()
    showing = str(getattr(v, "cell_type", "") or "").strip().casefold()
    return note if (not described or described == showing) else {}


def _enrichment_block(ce, pathway_notes_for_cluster: dict,
                      cell_type: str = "") -> tuple[str, str]:
    """A plain-text block of the cluster's enriched programs + cited biology, and the
    per-item census (never a band). ('', '') when there is no enrichment."""
    if ce is None:
        return "", ""
    # The enrichment rubric's re-check flag, carried into the export.
    flag = " — re-check this." if getattr(ce, "verify", False) else ""
    if not getattr(ce, "enriched", ()):  # no program clears the gate
        return "", ce.census_line + flag
    bits: list[str] = []
    for p in ce.enriched[:4]:
        short = pretty_gene_set(p.gene_set)
        note = (pathway_notes_for_cluster or {}).get(p.gene_set) or {}
        # Same guard as the cell-type note: no citation under another call.
        described = str(note.get("cell_type") or "").strip().casefold()
        if described and cell_type and described != str(cell_type).strip().casefold():
            note = {}
        # The join ends the line, so a bit must not end with its own period.
        bio = str(note.get("summary") or "").strip().removesuffix(".").rstrip()
        pmid = note.get("pmid")
        cite = f" (PMID:{pmid})" if pmid else ""
        # The panel-coverage denominator travels with the program (R4).
        hits = getattr(p, "panel_hits", None)
        full = getattr(p, "set_size_full", None)
        scope = f" [{hits}/{full} set genes on panel]" if hits and full else ""
        bits.append(f"{short}{scope} — {bio}{cite}" if bio else f"{short}{scope}")
    return "; ".join(bits) + ".", ce.census_line + flag


def build_report(
    *,
    verdicts: list[ClusterVerdict],
    celltype_notes: dict,
    notes: list[Note],
    holistic,
    panel_size: int,
    panel_no_signal: int = 0,
    dataset: str | None = None,
    generated_at: str = "",
    enrichments: dict | None = None,
    pathway_notes: dict | None = None,
    tissue: str = "",
    platform: str = "",
    undecidable: dict | None = None,
    # §14 provenance, filled by `build_report_from_sources`.
    frames: tuple[tuple[str, str], ...] = (),
    reference_releases: tuple[str, ...] = (),
    corpus_status: str = "",
    generator: str = "",
    spec_version: str = "",
) -> ReportModel:
    """Assemble the interpretation report from durable, grounded inputs. Pure.

    ``enrichments`` maps cluster -> ClusterEnrichment and ``pathway_notes`` maps
    cluster -> {gene_set: note}. ``undecidable`` maps cluster -> the v2 record's
    undecidable sentences, carried verbatim so the export is never quieter than
    the screen.
    """
    dataset = dataset or cfg.DATASET_ID
    enrichments = enrichments or {}
    pathway_notes = pathway_notes or {}
    undecidable = undecidable or {}
    sections: list[ClusterSection] = []
    for v in verdicts:
        # Guarded against the composed call this section is headed with.
        ct_note = _celltype_note_for(celltype_notes, v)
        cluster_notes = tuple(
            _note_line(n)
            for n in notes
            if n.scope == "cluster" and n.scope_ref.cluster == v.cluster
        )
        enr_text, enr_census = _enrichment_block(
            enrichments.get(v.cluster), pathway_notes.get(v.cluster), v.cell_type
        )
        sections.append(
            ClusterSection(
                cluster=v.cluster,
                cell_type=v.cell_type,
                confidence=v.confidence,
                verify=v.verify,
                excluded=bool(getattr(v, "exclude", False)),
                driver_line=_driver_line(v),
                biology=str(ct_note.get("summary") or ""),
                biology_pmid=str(ct_note.get("pmid") or ""),
                settle=_settle_for(v.cluster, v.verify, v.cell_type),
                notes=cluster_notes,
                enrichment=enr_text,
                enrichment_census=enr_census,
                undecidable=tuple(undecidable.get(v.cluster) or ()),
                lineage=getattr(v, "lineage", "") or "",
                key_markers=", ".join(v.key_markers[:4]) if v.key_markers else "",
                # Read off the verdict, never re-derived.
                off_panel=tuple((o.gene, o.message)
                                for o in getattr(v, "offpanel_notes", ()) or ()),
            )
        )

    dataset_notes = tuple(_note_line(n) for n in notes if n.scope in ("dataset", "lab"))

    coherence_notes: tuple[str, ...] = ()
    refinements: tuple[str, ...] = ()
    if holistic is not None:
        coherence_notes = tuple(holistic.coherence_notes)
        refinements = tuple(
            f"{r.cluster}: {r.from_call} → {r.to_call} — {r.rationale}"
            for r in holistic.refinements
        )

    return ReportModel(
        dataset=dataset,
        generated_at=generated_at,
        n_clusters=len(sections),
        n_flagged=sum(1 for s in sections if s.verify),
        panel_size=panel_size,
        panel_no_signal=panel_no_signal,
        sections=tuple(sections),
        coherence_notes=coherence_notes,
        refinements=refinements,
        dataset_notes=dataset_notes,
        tissue=tissue,
        platform=platform,
        frames=frames,
        reference_releases=reference_releases,
        corpus_status=corpus_status,
        generator=generator,
        spec_version=spec_version,
    )


def _dataset_undecidables(dataset: str) -> dict:
    """``{cluster: (sentence, ...)}`` off the v2 records (``interp/records/``).

    ``{}`` when the records are unreadable: a thinner report, never a wrong one.
    """
    try:
        import json

        from panoscope.core.record import RECORDS_DIR
        from panoscope.pipeline import paths
        from panoscope.types.outputs import TaskKind

        got: dict = {}
        base = paths.interp_dir(dataset) / RECORDS_DIR / TaskKind.ASSIGNMENT.value
        for path in sorted(base.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            said = tuple(str(s) for s in rec.get("undecidable") or ())
            if said:
                got[str(rec.get("unit") or path.stem)] = said
        return got
    except Exception:  # noqa: BLE001 - no records yet just means no undecidable line
        return {}


def _provenance(dataset: str) -> dict:
    """ARCHITECTURE §14's provenance header, read off what actually ran.

    Fail-soft per field and never silent. No network.
    """
    frames: tuple[tuple[str, str], ...] = ()
    study = dataset
    try:
        from panoscope import platforms
        from panoscope.data.study import Study

        store = Study.open(dataset)
        study = store.id
        frames = tuple((f.name, f.digest)
                       for f in platforms.for_study(store).frames(store))
    except Exception:  # noqa: BLE001 - a thinner header, never a wrong one
        frames = ()

    releases: tuple[str, ...] = ()
    try:
        from panoscope.reference import sources as ref_sources
        releases = tuple(f"{s.name} {s.release}" for s in ref_sources.SOURCES.values())
    except Exception:  # noqa: BLE001
        releases = ()

    corpus = "not built; no literature was retrieved for this run"
    try:
        from panoscope.corpus import store as corpus_store

        # Keyed by the study (corpus_path's key), never by a dataset name.
        c = corpus_store.load(corpus_store.corpus_path(study))
        v = c.version
        built = (v.built_at if v else "") or c.built or "no build stamp"
        corpus = (f"{built} · {len(c.records)} record(s), {len(c.misses)} miss(es), "
                  f"{len(c.statements)} statement(s)"
                  f" · spec {c.spec_version}")
        if v is not None:
            corpus += f" · version {v.content_hash} · {v.generator} · {v.cost}"
    except Exception:  # noqa: BLE001 - CorpusUnavailable is the ordinary case
        pass

    try:
        from panoscope.core.record import SPEC_VERSION
        spec = SPEC_VERSION
    except Exception:  # noqa: BLE001
        spec = ""

    # Generator: numbers, bands and absences are deterministic; the only model-written
    # content is the cited note prose, whose generator is not recorded. The access
    # date is the study build's timestamp. Never a configured default model name.
    built = version = ""
    try:
        manifest = cfg.current_study().manifest
        built = str(manifest.get("generated_at") or "")
        version = str(manifest.get("pipeline_version") or "")
    except Exception:  # noqa: BLE001 - no manifest: say nothing rather than guess
        pass

    generator = "deterministic engine" + (f" (pipeline {version})" if version else "")
    try:
        from panoscope.ui import data_access as _da

        if _da.celltype_notes() or _da.pathway_notes():
            generator += ("; cited note prose written by a live model pass whose "
                          "generator is not recorded in the study tree")
    except Exception:  # noqa: BLE001 - no notes stage: the deterministic half stands
        pass
    if built:
        generator += f"; study built {built}"

    return {"frames": frames, "reference_releases": releases,
            "corpus_status": corpus, "generator": generator, "spec_version": spec}


def build_report_from_sources(generated_at: str = "") -> ReportModel:
    """Pull the durable artifacts from ``ui.data_access`` and assemble the report."""
    from panoscope.ui import data_access as da

    try:
        enrichments = {ce.cluster: ce for ce in da.all_enrichments()}
        pathway_notes = da.pathway_notes()
    except Exception:  # noqa: BLE001 - no enrichment slice -> marker-only report
        enrichments, pathway_notes = {}, {}

    tissue, platform = da.tissue_platform()
    prov = _provenance(cfg.DATASET_ID)
    return build_report(
        verdicts=da.composed_verdicts(),  # reflect confirmed overrides/excludes in the report
        celltype_notes=da.celltype_notes(),
        notes=da.read_notes(),
        holistic=da.holistic(),
        panel_size=len(da.panel_names()),
        panel_no_signal=_n_no_signal(),
        dataset=cfg.DATASET_ID,
        generated_at=generated_at,
        enrichments=enrichments,
        pathway_notes=pathway_notes,
        tissue=tissue,
        platform=platform,
        undecidable=_dataset_undecidables(cfg.DATASET_ID),
        **prov,
    )


# --------------------------------------------------------------------------- #
# Header + overview table (the confirmed call per cluster), shared by the documents.
# --------------------------------------------------------------------------- #
_OVERVIEW_COLS = ("Cluster", "Cell type", "Lineage", "Confidence", "Top markers")


def _header_meta(report: ReportModel) -> list[str]:
    """Grounded header lines (ARCHITECTURE §14). Every provenance line is emitted,
    even when the answer is "not consulted"."""
    ctx = " · ".join(p for p in (report.tissue, report.platform) if p)
    lines = [report.dataset + (f" · {ctx}" if ctx else "")]
    lines.append(
        f"{report.n_clusters} clusters · {report.n_flagged} flagged for re-check · "
        f"{report.panel_size}-gene panel"
        + (f" ({report.panel_no_signal} with no usable signal)"
           if report.panel_no_signal else "")
        + f" · marker method: {report.method}"
    )
    if report.frames:
        lines.append("frames: " + " · ".join(
            f"{n} {d[:12]}" for n, d in report.frames))
    lines.append("reference: " + (" · ".join(report.reference_releases)
                                  if report.reference_releases
                                  else "no source registered; not consulted"))
    lines.append("corpus: " + (report.corpus_status or "not built; no literature "
                                                       "was retrieved for this run"))
    lines.append("generator: " + (report.generator or "deterministic; no model was "
                                                      "called"))
    if report.spec_version:
        lines.append(f"spec version {report.spec_version}")
    if report.generated_at:
        lines.append(f"generated {report.generated_at}")
    return lines


def overview_rows(report: ReportModel) -> list[tuple[str, str, str, str, str]]:
    """One row per cluster for the overview table: the confirmed calls at a glance.

    Columns match ``_OVERVIEW_COLS``: cluster, cell type (re-check marked), lineage,
    confidence, top markers."""
    rows: list[tuple[str, str, str, str, str]] = []
    for s in report.sections:
        ct = s.cell_type.replace("_", " ") + ("  (re-check)" if s.verify else "")
        rows.append((s.cluster, ct, s.lineage, s.confidence, s.key_markers))
    return rows


# --------------------------------------------------------------------------- #
# Downloadable documents (.docx + PDF) — deterministic, no network
# --------------------------------------------------------------------------- #
_PDF_MAP = {
    "—": "-", "–": "-", "→": "->", "←": "<-", "↳": "->", "≈": "~", "×": "x",
    "…": "...", "’": "'", "‘": "'", "“": '"', "”": '"', "•": "*", "·": "-", "⚑": "!",
}


def _pdf_safe(s: str) -> str:
    """fpdf2 core fonts are latin-1; map the unicode punctuation we emit, then
    replace anything still outside latin-1 so a stray glyph can never crash export."""
    for k, v in _PDF_MAP.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _note_text(nl: NoteLine) -> str:
    base = f'"{nl.claim}" — {nl.scope}, basis: {nl.basis}, {nl.status} ({nl.author}, {nl.date})'
    return base + (f" — {nl.tension}" if nl.tension else "")


# --------------------------------------------------------------------------- #
# ``PMID:xxxx`` tokens become PubMed hyperlinks in both the .docx and the PDF.
# --------------------------------------------------------------------------- #
_PMID_LINK_RE = _re.compile(r"PMID:\s*(\d+)", _re.IGNORECASE)
_LINK_HEX = "0F7B87"          # teal, matches the app accent
_LINK_RGB = (15, 123, 135)


def _pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _docx_hyperlink(paragraph: Any, url: str, text: str) -> None:
    """Append a clickable hyperlink run (teal, underlined) to a python-docx paragraph."""
    from docx.oxml.shared import OxmlElement, qn

    r_id = paragraph.part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), _LINK_HEX); rpr.append(color)
    underline = OxmlElement("w:u"); underline.set(qn("w:val"), "single"); rpr.append(underline)
    run.append(rpr)
    node = OxmlElement("w:t"); node.text = text; run.append(node)
    link.append(run)
    paragraph._p.append(link)


def _docx_linked_paragraph(doc: Any, text: str, style: Optional[str] = None) -> Any:
    """Add a paragraph whose ``PMID:xxxx`` tokens are clickable PubMed links."""
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    last = 0
    for m in _PMID_LINK_RE.finditer(text):
        if m.start() > last:
            p.add_run(text[last:m.start()])
        pmid = m.group(1)
        _docx_hyperlink(p, _pubmed_url(pmid), f"PMID:{pmid}")
        last = m.end()
    if last < len(text):
        p.add_run(text[last:])
    return p


def _pdf_write_linked(pdf: Any, text: str, h: float) -> None:
    """Flow ``text`` (wrapping), rendering ``PMID:xxxx`` tokens as clickable PubMed links."""
    last = 0
    for m in _PMID_LINK_RE.finditer(text):
        if m.start() > last:
            pdf.write(h, _pdf_safe(text[last:m.start()]))
        pmid = m.group(1)
        token = _pdf_safe(f"PMID:{pmid}")
        # Keep the citation whole: break the line first if the token will not fit.
        try:
            room = pdf.w - pdf.r_margin - pdf.get_x()
            if pdf.get_string_width(token) > room:
                pdf.ln(h)
        except Exception:  # noqa: BLE001 - a geometry query must never lose the citation
            pass
        pdf.set_text_color(*_LINK_RGB)
        pdf.write(h, token, link=_pubmed_url(pmid))
        pdf.set_text_color(0, 0, 0)
        last = m.end()
    if last < len(text):
        pdf.write(h, _pdf_safe(text[last:]))
    pdf.ln(h)


def _section_lines(report: ReportModel) -> list[tuple[str, str]]:
    """Flatten the report into (style, text) lines shared by both exporters.

    style in {title, meta, h2, body, note}. Keeps docx and pdf in sync.
    """
    lines: list[tuple[str, str]] = [("title", "Panoscope: interpretation summary")]
    for h in _header_meta(report):
        lines.append(("meta", h))
    lines.append(("overview", ""))  # rendered as a table by the doc builders

    for s in report.sections:
        head = (f"{s.cluster} · {s.cell_type}" + (f" ({s.confidence})" if s.confidence else "")
                + ("  [re-check]" if s.verify else ""))
        if s.excluded:
            head += "  [EXCLUDED BY THE BIOLOGIST]"
        lines.append(("h2", head))
        if s.excluded:
            lines.append(("body",
                          "Excluded by the biologist. The measurement below is "
                          "reported as computed; it is not a call this study "
                          "stands behind."))
        lines.append(("body", f"Drivers: {s.driver_line}"))
        if s.biology:
            bio = s.biology + (f" (PMID:{s.biology_pmid})" if s.biology_pmid else "")
            lines.append(("body", f"Biology: {bio}"))
        if s.enrichment:
            conf = f" ({s.enrichment_census})" if s.enrichment_census else ""
            lines.append(("body", f"Programs{conf}, panel-scoped: {s.enrichment}"))
        if s.settle:
            lines.append(("body", f"What would settle it: {s.settle}"))
        # One line per sentence, never joined.
        if s.off_panel:
            names = ", ".join(g for g, _m in s.off_panel)
            lines.append(("body",
                          f"Never measured ({len(s.off_panel)} off panel): {names}"
                          " - absent from the capability frame, so their absence is"
                          " not evidence against this call."))
        for u in s.undecidable:
            lines.append(("body", f"Not settled by this measurement: {u}"))
        for nl in s.notes:
            lines.append(("note", _note_text(nl)))

    lines.append(("h2", "Cross-cluster review"))
    for c in report.coherence_notes:
        lines.append(("body", c))
    for r in report.refinements:
        lines.append(("body", f"- {r}"))

    if report.dataset_notes:
        lines.append(("h2", "My notes (dataset / all datasets)"))
        for nl in report.dataset_notes:
            lines.append(("note", _note_text(nl)))
    return lines


def _docx_overview_table(doc, overview: list[tuple]) -> None:
    """Render the confirmed-call overview as a real Word table."""
    table = doc.add_table(rows=1, cols=len(_OVERVIEW_COLS))
    try:
        table.style = "Table Grid"
    except Exception:  # noqa: BLE001 - a missing style just means an unbordered table
        pass
    for i, col in enumerate(_OVERVIEW_COLS):
        table.rows[0].cells[i].text = col
    for row in overview:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)


def _lines_to_docx(lines: list[tuple[str, str]], *, overview: list[tuple] | None = None) -> bytes:
    """Render (style, text) lines as a Word .docx (editable). Requires python-docx.

    An ``("overview", "")`` line renders the confirmed-call table from ``overview``."""
    from io import BytesIO

    from docx import Document

    doc = Document()
    for style, text in lines:
        if style == "title":
            doc.add_heading(text, level=0)
        elif style == "meta":
            _docx_linked_paragraph(doc, text)
        elif style == "overview":
            if overview:
                _docx_overview_table(doc, overview)
        elif style == "h2":
            doc.add_heading(text, level=2)
        elif style == "note":
            _docx_linked_paragraph(doc, text, style="List Bullet")
        else:
            _docx_linked_paragraph(doc, text)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# PDF overview-table column widths (mm), summing under the ~190mm usable A4 width.
_PDF_COL_W = (16, 40, 32, 26, 68)


def _fit_cell(pdf, text: str, w: float) -> str:
    """Trim ``text`` to what fits in a ``w``-mm cell (measured; the font is
    proportional), with an ellipsis."""
    limit = w - 2  # the cell's own left/right padding
    if pdf.get_string_width(text) <= limit:
        return text
    ell = "..."
    room = limit - pdf.get_string_width(ell)
    cut = text
    while cut and pdf.get_string_width(cut) > room:
        cut = cut[:-1]
    return (cut + ell) if cut else ""


def _pdf_overview_table(pdf, overview: list[tuple]) -> None:
    """Render the confirmed-call overview as a bordered grid in the PDF."""
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 8)
    for w, col in zip(_PDF_COL_W, _OVERVIEW_COLS):
        pdf.cell(w, 6, _pdf_safe(col), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7.5)
    for row in overview:
        for w, val in zip(_PDF_COL_W, row):
            pdf.cell(w, 6, _fit_cell(pdf, _pdf_safe(str(val)), w), border=1)
        pdf.ln()
    pdf.ln(2)


def _lines_to_pdf(lines: list[tuple[str, str]], *, overview: list[tuple] | None = None) -> bytes:
    """Render (style, text) lines as a PDF (locked). Requires fpdf2."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def cell(h: float, text: str) -> None:
        # new_x=LMARGIN resets the cursor to the left margin so the next full-width
        # multi_cell has room (fpdf2 otherwise leaves x at the right margin).
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for style, raw in lines:
        if style == "overview":
            if overview:
                _pdf_overview_table(pdf, overview)
            continue
        if style == "title":
            pdf.set_font("Helvetica", "B", 16)
            cell(8, _pdf_safe(raw))
            pdf.ln(1)
        elif style == "meta":
            pdf.set_font("Helvetica", "I", 9)
            _pdf_write_linked(pdf, raw, 5)  # meta rarely has a PMID, but stays consistent
            pdf.ln(2)
        elif style == "h2":
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            cell(6, _pdf_safe(raw))
        elif style == "note":
            pdf.set_font("Helvetica", "I", 10)
            _pdf_write_linked(pdf, f"  {raw}", 5)
        else:
            pdf.set_font("Helvetica", "", 10)
            _pdf_write_linked(pdf, raw, 5)
    return bytes(pdf.output())


def report_to_docx(report: ReportModel) -> bytes:
    """Render the structured report as a Word .docx (editable). Requires python-docx."""
    return _lines_to_docx(_section_lines(report), overview=overview_rows(report))


def report_to_pdf(report: ReportModel) -> bytes:
    """Render the structured report as a PDF (locked). Requires fpdf2."""
    return _lines_to_pdf(_section_lines(report), overview=overview_rows(report))


# --------------------------------------------------------------------------- #
# Editable working space — the auto-seed per-cluster summary + its export.
# --------------------------------------------------------------------------- #
def default_cluster_summary(s: ClusterSection) -> str:
    """The auto-seed per-cluster summary for the editable working space.

    Identity (call, drivers, cited biology), then programs (panel-scoped), then the
    settle line, never-measured markers, undecidables and notes. PMIDs inline.
    """
    ident = (
        f"Identity — {s.cluster} is {s.cell_type}"
        + (f" at {s.confidence} confidence" if s.confidence else ", with no band")
        + (" (flagged to re-check)" if s.verify else "")
        + f", driven by {s.driver_line}."
    )
    if s.biology:
        ident += " " + s.biology + (f" (PMID:{s.biology_pmid})" if s.biology_pmid else "")
    paras = [ident]

    if s.enrichment:
        conf = f" ({s.enrichment_census})" if s.enrichment_census else ""
        paras.append(
            f"Programs — enriched gene sets{conf}, panel-scoped (measured only over the "
            f"set genes on the panel, not genome-wide): {s.enrichment}"
        )
    elif s.enrichment_census:
        paras.append("Programs — no gene-set program clears the enrichment gate for this cluster.")

    if s.settle:
        paras.append(f"What would settle it — {s.settle}")
    # Also seeded here: ``working_docx`` exports this text, not ``_section_lines``.
    if s.off_panel:
        names = ", ".join(g for g, _m in s.off_panel)
        paras.append(
            f"Never measured ({len(s.off_panel)} off panel) — {names}. Absent from "
            "the capability frame, so their absence is not evidence against this "
            "call.")
    for u in s.undecidable:
        paras.append(f"Not settled by this measurement — {u}")
    for nl in s.notes:
        t = f" — {nl.tension}" if nl.tension else ""
        paras.append(f'Note — "{nl.claim}" ({nl.scope} · {nl.basis} · {nl.status}){t}.')
    return "\n\n".join(paras)


def _blocks(text: str) -> list[tuple[str, str]]:
    return [("body", p.strip()) for p in text.split("\n\n") if p.strip()]


def _working_lines(
    sections: list[tuple[str, str, str, bool, str]],
    *,
    dataset: str,
    generated_at: str,
    header: list[str] | None = None,
    has_overview: bool = False,
    global_check: str = "",
    caveats: str = "",
    lab_notes: tuple[NoteLine, ...] = (),
) -> list[tuple[str, str]]:
    """(style, text) lines for the EDITED working-space export: an overview table, the
    per-cluster summaries, the cross-cluster global check, then the caveats (the
    biologist's report shape). ``header`` replaces the bare meta line when given."""
    lines: list[tuple[str, str]] = [("title", "Panoscope: interpretation summary")]
    if header:
        for h in header:
            lines.append(("meta", h))
    else:
        meta = f"{dataset} · {len(sections)} clusters"
        if generated_at:
            meta += f" · {generated_at}"
        lines.append(("meta", meta))
    if has_overview:
        lines.append(("overview", ""))
    for cluster, cell_type, confidence, verify, text in sections:
        head = (f"{cluster} · {cell_type}" + (f" ({confidence})" if confidence else "")
                + ("  [re-check]" if verify else ""))
        lines.append(("h2", head))
        lines.extend(_blocks(text))
    if global_check.strip():
        lines.append(("h2", "Cross-cluster global check"))
        lines.extend(_blocks(global_check))
    if caveats.strip():
        lines.append(("h2", "Caveats"))
        lines.extend(_blocks(caveats))
    if lab_notes:
        lines.append(("h2", "My notes (dataset / all datasets)"))
        for nl in lab_notes:
            lines.append(("note", _note_text(nl)))
    return lines


def global_check_text(holistic, themes) -> str:
    """Cross-cluster global check: does the whole set cohere? Marker coherence +
    the one refinement + the enrichment programs that recur across clusters. Every
    statement is read off the holistic review / themes (no new claim)."""
    paras: list[str] = []
    if holistic is not None:
        paras.extend(holistic.coherence_notes)
        for r in holistic.refinements:
            paras.append(
                f"Refinement to consider — {r.cluster}: {r.from_call} -> {r.to_call} ({r.rationale})."
            )
    if themes is not None and getattr(themes, "recurring", ()):
        bits = [
            f"{pretty_gene_set(t.gene_set)} (in {t.n_clusters} clusters: {', '.join(t.clusters)})"
            for t in themes.recurring[:4]
        ]
        paras.append(
            "Programs shared across clusters — " + "; ".join(bits) + ". A program enriched in "
            "many clusters partly reflects the panel's design and shared genes; weight it less "
            "as cluster-specific evidence."
        )
    return "\n\n".join(paras) if paras else "The set is coherent across clusters."


def caveats_text(verdicts, enrichments: dict, panel_size: int) -> str:
    """The interpretation's caveats: panel scope, panel-absence, flagged/small calls,
    and the cross-lineage-enrichment reminder. Derived deterministically from the
    verdicts + enrichment records; nothing invented."""
    paras = [
        f"Panel scope — this is a targeted {panel_size}-gene panel. Enrichment is measured only "
        "over each set's genes that are on the panel, so no program is a genome-wide claim, and a "
        "program can be missed simply because its genes were not on the panel.",
        "Panel absence — the absence of a canonical marker is not evidence against a cell type "
        "when that gene was never on the panel; missing canonical markers were checked against "
        "the panel before any down-weight.",
    ]
    flagged = [f"{v.cluster} {v.cell_type}" for v in verdicts if v.verify]
    if flagged:
        paras.append(
            "Flagged for re-check — " + ", ".join(flagged) + ". These calls rest on thin or "
            "non-specific evidence; confirm before relying on them."
        )
    small = [f"{v.cluster} {v.cell_type}" for v in verdicts if getattr(v, "small_n", False)]
    if small:
        paras.append(
            "Small clusters — " + ", ".join(small) + " have few assigned markers, so both the "
            "call and its enrichment are lower-powered."
        )
    paras.append(
        "Cross-lineage programs — a program enriched in a cluster whose leading-edge genes mark "
        "another lineage usually reflects co-infiltration or shared panel genes, not a re-typing "
        "of the cluster; these are flagged in the per-cluster programs above."
    )
    return "\n\n".join(paras)


def working_docx(sections, *, report_model: ReportModel | None = None, **kw) -> bytes:
    """Export the (edited) working space as a Word .docx.

    ``report_model`` (the assembled report) adds the grounded header and the
    confirmed-call overview table at the top."""
    header = _header_meta(report_model) if report_model else None
    overview = overview_rows(report_model) if report_model else None
    lines = _working_lines(sections, header=header, has_overview=overview is not None, **kw)
    return _lines_to_docx(lines, overview=overview)


def working_pdf(sections, *, report_model: ReportModel | None = None, **kw) -> bytes:
    """Export the (edited) working space as a PDF (with the header + overview table)."""
    header = _header_meta(report_model) if report_model else None
    overview = overview_rows(report_model) if report_model else None
    lines = _working_lines(sections, header=header, has_overview=overview is not None, **kw)
    return _lines_to_pdf(lines, overview=overview)
