"""The dedicated interpretation summary — assembled, reviewable, downloadable.

The Summary page keeps its marker table, pathways result, and cross-cluster
review, and gains this fourth region: a per-dataset interpretation report the
biologist reviews on-page and downloads as .docx and PDF.

It is assembled ENTIRELY from durable, already-grounded artifacts — the per-cluster
verdicts (jazzPanda numbers), the cited cell-type biology notes, the deterministic
"what would settle it" line for shaky calls, your own notes with their
agree/dissent citations, and the cross-cluster holistic review. There is **no live
network at build/download time**: nothing is fetched, nothing is recomputed, so the
report is reproducible and unit-testable, yet it still reads as "what came out of
your conversations with the agent."

Design boundaries:
- ``build_report`` is a pure function of its inputs (no I/O) → fully testable.
- Streamlit / ``ui.data_access`` are imported lazily, so ``build_report`` /
  ``report_to_docx`` / ``report_to_pdf`` work with no server running.
- No new ``ui.data_access`` accessor is added (the concurrent enrichment session
  owns that file's tail); this module calls only committed accessors.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass
from typing import Any, Optional

from panoscope.agent import config as cfg
from panoscope.agent import discriminate
from panoscope.agent.types import ClusterVerdict, MarkerEvidence, Note

# --------------------------------------------------------------------------- #
# Human labels for the closed note vocabularies (relabel only; never change one).
# Inlined here (rather than importing ui.lab_knowledge) so this module's pure
# functions stay free of Streamlit / data_access.
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
    #: The record's own undecidable sentences, verbatim — what this measurement did
    #: NOT settle. Today they are the §9.10 replicate read; the field is the record's
    #: field, not concordance's, so a later producer's undecidable travels too.
    undecidable: tuple[str, ...] = ()
    #: The biologist withdrew this unit. NOT a reason to drop the section: the
    #: document is the record, and a unit that silently vanishes cannot be
    #: argued with. It is rendered WITH the withdrawal stated.
    excluded: bool = False
    lineage: str = ""                # for the overview table
    key_markers: str = ""            # top gene names (no numbers), for the overview table
    #: The panel-absence rule, carried into the document: ``(gene, message)`` for
    #: every canonical marker of this call the panel never measured.
    #:
    #: It was missing, and it was the worst omission in the export. On screen c2
    #: names COL1A1, COL1A2, DCN, VIM and FAP as never measured; the document a
    #: biologist hands on named none of them, so a reader could not tell a call
    #: resting on a complete canonical set from one where five of its markers were
    #: never on the panel. ``ClusterVerdict.offpanel_notes`` had carried them the
    #: whole time -- ``build_report`` simply never read the field.
    #:
    #: Pairs rather than the ``OffPanelNote`` object: this model is the boundary
    #: between the v1 verdict and three renderers, and it holds plain data so a
    #: renderer cannot reach past it for something it was not meant to expose.
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
    # ----------------------------------------------------------------------- #
    # ARCHITECTURE §14's REPORT header: "study, dataset, platform, frames +
    # hashes, reference releases, corpus version + staleness, generator model +
    # version + access date, spec version". Everything after `platform` was
    # missing, so the document carried no way to tell which panel produced it,
    # whether reference data was consulted, or which spec version wrote it.
    #
    # Each field says what it IS or says plainly that it was not consulted. "No
    # reference source registered" and "corpus not built" are answers; a blank
    # line is not, and silence here reads as "checked and fine".
    # ----------------------------------------------------------------------- #
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

    #: How many of those panel genes produced no usable spatial signal. The
    #: detectability frame as a NUMBER, not a digest: the header said "280-gene
    #: panel" and the frame line carried only a truncated hash, so a reader
    #: believed 280 genes informed every call when 231 did.
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
    # Asked of ``lookup_state``, not of the stored ``thin`` boolean. The export was
    # the one reader still trusting that flag, and it collapsed two different facts
    # into one sentence: a search that RAN and found nothing (thin literature -- a
    # finding, and sayable) and a search that never completed (which says nothing
    # about the literature at all). Both printed "literature thin: no supporting
    # reference on file" in the document that leaves the lab. ``claims/notes.py``
    # derives this from ``looked_up_at`` for exactly that reason.
    from panoscope.claims.notes import Resolution, lookup_state

    state = lookup_state(t)
    if state is Resolution.NOT_FOUND:
        return "literature thin: searched, no supporting reference found"
    if state is Resolution.NOT_LOOKED_UP:
        return "literature not checked: no lookup completed for this note"
    return ""


def note_scope_label(note: Note) -> str:
    """The note's scope as one readable span — the LAB rung carrying its formation study.

    The lab rung is deliberately cross-study: a standing judgement is MEANT to travel,
    and ``memory.note_in_study`` lets it. What it must not do is travel with its
    conditions stripped. Rendered as a bare "all datasets", a call formed while looking
    at one 280-gene FFPE breast panel arrives on an off-target kidney IO panel under the
    biologist's own authority with nothing on screen naming the study that formed it --
    so the reader cannot see the tension, and "the value is in the disagreement" requires
    that they can. ``scope_ref.dataset`` was stored and correct all along; this names it.

    One spelling for both surfaces (the exported report and the Summary note board), so
    the two cannot drift into disagreeing about what a note's scope is.

    Only the lab rung gets the suffix: a dataset- or cluster-scoped note fires nowhere
    but the study it names, so "this dataset" already says which study formed it.
    """
    if note.scope == "cluster" and note.scope_ref.cluster:
        return f"cluster {note.scope_ref.cluster}"
    label = _SCOPE_LABEL.get(note.scope, note.scope)
    if note.scope != "lab":
        return label
    # A pre-schema note can name no dataset. Saying that out loud is the honest
    # answer; a bare "all datasets" reads as though the formation study were known.
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

    Withheld when the section's call is not the one the discrimination was computed
    for. ``discriminate`` reads ``data.cell_type_for(cluster)`` -- the deterministic
    table, never an override -- so after the biologist overrode c3 to Dendritic the
    document still printed "To tell Macrophages from T Cells in c3: Macrophages is
    supported here by LYZ (glm_coef 11.84)...": a discrimination argued for a call
    they had rejected, carrying real statistics.

    Silence rather than a recomputation. Re-running the rival test against the new
    call is a real answer and belongs in ``discriminate``, which would have to take
    the call as an argument; inventing one here from the old result would not be.
    """
    d = discriminate.discriminate(cluster)
    computed_for = str(getattr(d, "call_A", "") or "").strip().casefold()
    if cell_type and computed_for and computed_for != str(cell_type).strip().casefold():
        return ""
    if d.alt_B is not None and (verify or d.b_here):
        return discriminate.settle_summary(d)
    return ""


def _n_no_signal() -> int:
    """How many panel genes produced no usable spatial signal. Derived, per study.

    CLAUDE.md calls detectability a capability frame rather than a footnote, and on
    this demo it is 49 of 280 -- 17.5% of the panel that could not inform any call.
    Counted off the method table through the adapter's own predicate, so the demo's
    49 is read and never asserted; a study with full detectability answers 0 and the
    header simply does not mention it.
    """
    try:
        from panoscope.agent import data as _data
        from panoscope.platforms.xenium import is_no_signal

        return sum(1 for t in _data.load_markers()["top_cluster"]
                   if is_no_signal(str(t)))
    except Exception:  # noqa: BLE001 - unreadable method table: say nothing
        return 0


def _celltype_note_for(celltype_notes: dict, v) -> dict:
    """This section's cell-type note, or ``{}`` when it describes another call.

    Every note records the ``cell_type`` it was written about, and nothing read it.
    Override c3 from Macrophages to Dendritic and the exported document kept the
    macrophage paragraph and its PMID under a heading that now read Dendritic -- a
    real citation attributed to a call it was never about, which is worse than
    carrying no citation at all. The guard data was in the note the whole time.

    Empty rather than relabelled: after an override this study has no cited biology
    for the new call yet, and writing one would be inventing it. ``ui.data_access``
    carries the same guard for the on-screen table; the check is duplicated rather
    than imported because this module builds documents from plain dicts and must
    stay usable without the cached UI layer.
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
    enrichment CENSUS. ('' , '') when there is no enrichment for the cluster.

    The second element used to be ``ce.confidence`` -- a unit-level band on a task
    with no label space, which is the one shape the no-band rule exists to catch.
    It is now ``ce.census_line`` ("4 supported - 0 suggestive - 2 untestable"),
    which says the thing the band could not: how many programs the panel could not
    test at all.
    """
    if ce is None:
        return "", ""
    # The ENRICHMENT rubric's re-check flag, which reached no export at all.
    # It is deterministic and per workflow, and it is True on 7 of the demo's 9
    # clusters -- so the Pathways header said "re-check this" while the .docx
    # and the PDF carried the same programs with nothing beside them. A
    # document that leaves the lab must not be quieter than the screen it came
    # from.
    flag = " — re-check this." if getattr(ce, "verify", False) else ""
    if not getattr(ce, "enriched", ()):  # no program clears the gate
        return "", ce.census_line + flag
    bits: list[str] = []
    for p in ce.enriched[:4]:
        short = p.gene_set.replace("HALLMARK_", "").replace("_", " ").title()
        note = (pathway_notes_for_cluster or {}).get(p.gene_set) or {}
        # The SAME guard the celltype note gets. A pathway note records the
        # cell_type it was written about, and this block did not read it -- so
        # overriding c3 from Macrophages to Dendritic kept four real PMIDs,
        # argued for macrophage biology, under a Dendritic heading. Guarding one
        # of a section's two cited-prose blocks fixes half the defect.
        described = str(note.get("cell_type") or "").strip().casefold()
        if described and cell_type and described != str(cell_type).strip().casefold():
            note = {}
        # The join below ends the whole line, so a bit must not end one too:
        # "driving genes CCND1, CENPF, HMGA1.." is what the document printed for
        # every program whose note carries no PMID to close the clause.
        bio = str(note.get("summary") or "").strip().removesuffix(".").rstrip()
        pmid = note.get("pmid")
        cite = f" (PMID:{pmid})" if pmid else ""
        # The denominator travels with the program. The Pathways surface shows it
        # per row ("5/100") and the stored artifact carries it, but the document
        # printed the program name and its biology alone -- so a set scored on 5 of
        # its 100 genes read exactly like one scored on 90. Every plot carries its
        # conditionality (§R4); a paragraph is not exempt because it is prose.
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
    # §14 provenance. Defaulted so every existing caller and test keeps working;
    # `build_report_from_sources` fills them from what actually ran.
    frames: tuple[tuple[str, str], ...] = (),
    reference_releases: tuple[str, ...] = (),
    corpus_status: str = "",
    generator: str = "",
    spec_version: str = "",
) -> ReportModel:
    """Assemble the interpretation report from durable, grounded inputs. Pure.

    ``enrichments`` maps cluster -> ClusterEnrichment and ``pathway_notes`` maps
    cluster -> {gene_set: note}; when given, each section integrates the cluster's
    enriched programs + their cited biology alongside the marker call. ``tissue`` /
    ``platform`` populate the report header (grounded context, from the manifest).

    ``undecidable`` maps cluster -> the v2 record's own undecidable sentences, and
    is the half of the interpretation that had no way out of the building. The
    exported document is what goes into a slide, a thesis chapter or a supplementary
    file; a report that is honest on screen and silent in the file that leaves the
    lab is not honest, and until this parameter existed the §9.10 replicate read --
    *these two sections read this gene differently* -- reached ``panoscope explain``
    and the markers page and stopped there, while the document named the diverging
    gene as a driver and called the unit High confidence. The sentences are carried
    verbatim, never summarised: they are the one place the three §9.10 outcomes stay
    apart, and a document that renders "divergent" and "could not be compared" with
    one word is the four-states-into-two collapse arriving in the export.
    """
    dataset = dataset or cfg.DATASET_ID
    enrichments = enrichments or {}
    pathway_notes = pathway_notes or {}
    undecidable = undecidable or {}
    sections: list[ClusterSection] = []
    for v in verdicts:
        # Through the guard: an overridden call must not inherit the previous
        # call's cited biology. ``verdicts`` here are the COMPOSED ones, so
        # ``v.cell_type`` is the call this section is actually headed with.
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
                # Read off the verdict, never re-derived: the same objects the
                # evidence table renders, so the document cannot disagree with the
                # screen about which markers were never measured.
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


def _dataset_tissue_platform(dataset: str) -> tuple[str, str]:
    """Read (tissue, platform) from the dataset manifest for the report header.

    Grounded context, straight from the data; ('', '') if the manifest is unreadable."""
    try:
        import json

        from panoscope.pipeline import paths

        m = json.loads((paths.dataset_dir(dataset) / "manifest.json").read_text(encoding="utf-8"))
        return str(m.get("tissue", "")).strip(), str(m.get("platform", "")).strip()
    except Exception:  # noqa: BLE001 - a missing manifest just means a plainer header
        return "", ""


def _dataset_undecidables(dataset: str) -> dict:
    """``{cluster: (sentence, ...)}`` off the v2 records, for the report's caveats.

    Read from ``interp/records/<task_kind>/<unit>.json`` rather than from the v1
    ``ClusterVerdict``, because the v1 shape has no field for an undecidable and
    inventing one would mean re-deriving in the UI what ``core/record`` already
    wrote. ``core/compat.with_v1_replicate_read`` does put the §9.10 sentences into
    the v1 ``notes`` string for the CSV's sake; parsing them back out of that
    sentence here would be prose-matching, which is the failure
    ``ui/evidence_table``'s ``FLAG_COPY`` was rebuilt to stop.

    ``{}`` when the records are unreadable -- an unbuilt study, or one built before
    the records existed. That is a thinner report, never a wrong one: the sections
    then carry no undecidable line at all, which is the same thing the record says.
    Same ``except Exception`` as the manifest reader directly above, for the same
    reason: the Summary page must render on a half-built tree.
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

    Fail-soft per field and never silent: a field that cannot be established says
    so, because an omitted reference line reads as though reference data agreed.
    Nothing here consults the network -- §14's REPORT block is explicit that the
    document is assembled "from durable artifacts, no live network".
    """
    frames: tuple[tuple[str, str], ...] = ()
    try:
        from panoscope import platforms
        from panoscope.data.study import Study

        store = Study.open(dataset)
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

        c = corpus_store.load(corpus_store.corpus_path(dataset))
        # The fields Corpus ACTUALLY has: records, misses, built, spec_version.
        # This read ``version`` and ``sources``, neither of which exists, so both
        # getattrs always took their defaults and a corpus holding two records and
        # a build stamp rendered as "? · 0 source(s)" -- a §14 provenance line
        # whose stated purpose is that "corpus not built" is an answer and a blank
        # is not. It was neither: it was a built corpus reported as empty.
        #
        # ``misses`` is counted alongside ``records`` because it is first class
        # here: it is what separates "asked and found nothing" from "never asked".
        built = getattr(c, "built", "") or "no build stamp"
        corpus = (f"{built} · {len(getattr(c, 'records', ()))} record(s), "
                  f"{len(getattr(c, 'misses', ()))} miss(es) · spec "
                  f"{getattr(c, 'spec_version', '?')}")
    except Exception:  # noqa: BLE001 - CorpusUnavailable is the ordinary case
        pass

    try:
        from panoscope.core.record import SPEC_VERSION
        spec = SPEC_VERSION
    except Exception:  # noqa: BLE001
        spec = ""

    # §14 asks for generator model + version + access date. What this said instead
    # was ``cfg.PRIMARY_MODEL`` -- which is ``os.getenv("PANOSCOPE_MODEL",
    # "claude-sonnet-4-6")``, a DEFAULT and not a record of anything -- plus
    # "reachable", inferred from an env var being set in whatever shell ran the
    # export. Nothing is called here: this module assembles from durable artifacts
    # with no network. So the header named a model that never ran and a
    # reachability nobody tested, and emitted no date at all; exporting with a
    # deliberately invalid key still printed "reachable".
    #
    # What is true: every number, band and absence in this document is
    # deterministic, and the only model-written content is the cited note prose,
    # whose generator the notes stage does not record. Both halves are now stated,
    # and the access date is the study build's own timestamp.
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
    """Convenience: pull the durable artifacts from ``ui.data_access`` and assemble.

    Used by the Summary page. Lazily imports data_access so the pure functions above
    (and the doc exporters) never require a Streamlit runtime.
    """
    from panoscope.ui import data_access as da

    try:
        enrichments = {ce.cluster: ce for ce in da.all_enrichments()}
        pathway_notes = da.pathway_notes()
    except Exception:  # noqa: BLE001 - no enrichment slice -> marker-only report
        enrichments, pathway_notes = {}, {}

    tissue, platform = _dataset_tissue_platform(cfg.DATASET_ID)
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
# Header + overview table — shared by the documents. The
# overview table is the skill's Output 1 lead: the confirmed call per cluster.
# --------------------------------------------------------------------------- #
_OVERVIEW_COLS = ("Cluster", "Cell type", "Lineage", "Confidence", "Top markers")


def _header_meta(report: ReportModel) -> list[str]:
    """Grounded header lines for the report -- ARCHITECTURE §14's REPORT header.

    Every provenance line is emitted even when the answer is "not consulted". A
    document that simply omits the reference line reads as though reference data
    agreed; one that says *no reference source was registered* cannot be misread.
    """
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
# Clickable PMID links in the exported documents. A ``PMID:xxxx`` token becomes a hyperlink to its PubMed record in both the
# .docx and the PDF, so every cited claim in the report is one click from the paper.
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
        # Keep the citation whole. ``write`` wraps at the margin wherever it
        # happens to be, so a PMID that starts near the right edge is split
        # across two lines -- the reader sees "PMID:374797" ending one line and
        # the remaining digits beginning the next, which reads as a different,
        # wrong identifier, and only the first half carries the link. Break the
        # line first when the whole token will not fit.
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
        head = f"{s.cluster} · {s.cell_type} ({s.confidence})" + ("  [re-check]" if s.verify else "")
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
        # One line per sentence, never joined: each is a different outcome of the
        # same check, and a reader skimming a paragraph reads the first clause.
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
    """Trim ``text`` to what actually fits in a ``w``-mm cell, with an ellipsis.

    The truncation was a flat 44 CHARACTERS for every column, while the columns
    are 16 to 68 mm wide. A 16mm cell holds about a dozen characters at this
    size, so 44 of them ran straight through the border and over the neighbour
    -- and the cell that overflowed was still cut off, so the reader lost the
    end of one value and the start of the next. Measuring is the only way to
    know, because the font is proportional.
    """
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
    """The auto-seed integrated per-cluster summary (the editable working space).

    Reads like a biologist's synthesis, in two grounded movements: IDENTITY (the
    marker cell-type call + its drivers + cited biology — *what the cell is*) and
    PROGRAMS (the enriched gene sets + their cited biology — *what the cell is
    doing*, panel-scoped), then any what-would-settle-it line and this cluster's
    your notes. Plain text with PMID:xxx inline (linkified on-page, kept verbatim in
    the exported document). The concordance read (do the programs fit the call?) is
    carried by the per-program biology, which flags cross-lineage signal explicitly.
    """
    ident = (
        f"Identity — {s.cluster} is {s.cell_type} at {s.confidence} confidence"
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
    # Seeded into the EDITABLE space as well as the assembled report, because
    # ``working_docx`` exports this text and not ``_section_lines``. A caveat that
    # only reaches one of the two documents is the same defect one export out.
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
        head = f"{cluster} · {cell_type} ({confidence})" + ("  [re-check]" if verify else "")
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


def _short_set(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


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
            f"{_short_set(t.gene_set)} (in {t.n_clusters} clusters: {', '.join(t.clusters)})"
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
