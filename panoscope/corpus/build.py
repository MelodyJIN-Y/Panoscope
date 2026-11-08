"""The corpus build: vocabulary, retrieval and reading for the D2 scope, then a version.

Built once per study after the reference view; resumable, because every stage skips
what the store already holds and retries only FAILED rows. A top-up runs the same
stages for named genes and writes a NEW version. Never part of the factual layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from typing import Any, Callable, Final

from panoscope.corpus import vocab
from panoscope.corpus.channels import _Run, retrieve
from panoscope.corpus.read import Reader, Receipt, model_reader, read
from panoscope.corpus.store import (
    Corpus, CorpusUnavailable, CorpusVersion, corpus_path, load, log_hash, read_version,
)
from panoscope.corpus.window import StudyContext
from panoscope.corpus.window import context_for as _context
from panoscope.reference.sources import SourceUnavailable

STALE_DAYS: Final = 180     # ponytail: one policy for every study; per-tissue when asked
TOP_N: Final = 10           # each cluster's top method markers (D2)
NOT_BUILT: Final = "corpus not built — cannot settle"

Progress = Callable[[int, int, str], None]


def scope(study: Any, cell_types: tuple[str, ...]) -> tuple[str, ...]:
    """D2: the frame types' reference canonical markers, then each cluster's top method markers."""
    import pandas as pd

    from panoscope import methods
    from panoscope.reference import view as ref_view
    from panoscope.workflows.marker import WORKFLOW_ID

    view = ref_view.for_study(study)
    genes = [g for t in cell_types for g in view.canonical(t)]
    path = study.inputs / "markers_top.csv"   # ponytail: single-sample table only
    if path.is_file():
        method = methods.for_workflow(WORKFLOW_ID)[0]
        t = pd.read_csv(path)
        unit = method.detect_unit(t).unit_column
        t = t[t[unit].astype(str).isin(study.units)]
        top = t.sort_values(method.statistic_name, ascending=False).groupby(unit).head(TOP_N)
        genes += top["gene"].astype(str).tolist()
    return tuple(dict.fromkeys(genes))


def context_for(study: Any) -> StudyContext:
    """The study's context plus the disease block's MeSH entry terms (§3), so a span saying
    "breast carcinoma" matches a "breast cancer" study; a failed lookup keeps the literal terms."""
    ctx = _context(study)
    mesh: list[str] = []
    for t in ctx.disease_terms:
        try:
            mesh += vocab.mesh_terms(t)
        except SourceUnavailable:
            pass        # ponytail: fewer disease matches (wider rung), never more; not logged
    return replace(ctx, disease_mesh=tuple(dict.fromkeys(mesh)))


def tissue_terms(study: Any, ctx: StudyContext) -> tuple[str, ...]:
    """T block (§3): the intake's UBERON label too, since the stated prose ("human breast
    cancer") as one quoted phrase matches almost no title or abstract."""
    return tuple(dict.fromkeys(t for t in (study.manifest.get("tissue_label"), ctx.tissue) if t))


def brief_hash(study: Any) -> str:
    """What a brief for this study asks: conditions, frame and D2 genes. An import checks it."""
    ctx = context_for(study)
    labels = tuple(e.label for e in ctx.cell_types)
    asked = [study.id, ctx.tissue, ctx.disease_state, ctx.platform, labels, scope(study, labels)]
    return hashlib.sha256(json.dumps(asked).encode()).hexdigest()[:16]


def version_for(study: Any, ctx: StudyContext, corpus: Corpus, generator: str, cost: str,
                day: date) -> CorpusVersion:
    """The version a build or an import writes: conditions, reference releases, log hash."""
    from panoscope.reference import view as ref_view

    bg = study.manifest.get("background") or {}
    return CorpusVersion(
        study=study.id,
        conditions=(("tissue", ctx.tissue), ("tissue_id", ctx.tissue_id),
                    ("platform", ctx.platform), ("modality_class", ctx.modality_class),
                    ("disease_state", ctx.disease_state), *ctx.disease_qualifiers),
        reference=ref_view.for_study(study).releases,
        generator=generator,
        query_log_hash=log_hash(corpus.queries),
        background_hash=hashlib.sha256(json.dumps(bg, sort_keys=True).encode()).hexdigest()[:16],
        cost=cost, built_at=day.isoformat(),
        stale_after=(day + timedelta(days=STALE_DAYS)).isoformat())


def build(study: Any, *, today: date, receipt: Receipt, reader: Reader | None = None,
          genes: tuple[str, ...] = (), resume: bool = False,
          progress: Progress = lambda i, n, line: None) -> Corpus:
    """Build the study's corpus (``resume`` continues one), or top up ``genes`` as a new version.

    Refuses with no model key before any network call; a fresh build starts from empty.
    """
    reader = reader or model_reader(receipt)
    ctx = context_for(study)
    p = corpus_path(study.id)
    if genes and not p.is_file():
        raise CorpusUnavailable(f"top-up refused: no corpus built for {study.id}; build it first")
    if not (genes or resume):
        p.unlink(missing_ok=True)                 # rebuilt wholesale (§8.3)
    todo = genes or scope(study, tuple(e.label for e in ctx.cell_types))
    tissue = tissue_terms(study, ctx)
    bg = study.manifest.get("background") or {}
    papers = tuple(str(c.get("value", ""))[5:] for c in bg.get("papers_of_interest", ())
                   if str(c.get("value", "")).startswith("pmid:"))   # ponytail: DOIs unread
    stamp = today.isoformat()
    for i, gene in enumerate(todo, 1):
        progress(i - 1, len(todo), f"{gene}: vocabulary")
        terms, runs = vocab.vocabulary(gene, tissue)
        try:
            run = _Run(load(p), p, 0, stamp)
        except CorpusUnavailable:
            run = _Run(Corpus(), p, 0, stamp)
        for r in runs:                            # replaces this gene's earlier vocab rows
            run.log(r)
        progress(i - 1, len(todo), f"{gene}: retrieval")
        retrieve(ctx, gene, terms, today=today, tissue_terms=tissue, path=p,
                 papers_of_interest=papers)
        progress(i - 1, len(todo), f"{gene}: reading")
        read(ctx, gene, terms, reader=reader, receipt=receipt, tissue_terms=tissue, path=p,
             retrieved=stamp)
    corpus = load(p)
    version = version_for(study, ctx, corpus, f"{receipt.model} via the Anthropic API (user key)",
                          receipt.text, today)
    run = _Run(corpus, p, 0, stamp)
    run.commit(version=version, built=stamp)
    left = len(run.corpus.unclosed)
    progress(len(todo), len(todo), f"version {version.content_hash}"
             + (f"; {left} id(s) hold no disposition" if left else ""))
    return run.corpus


def staleness(v: CorpusVersion, today: date) -> str:
    """The version's age against its policy date, as the header says it."""
    if not v.stale_after:
        return "staleness unknown"
    return (f"STALE since {v.stale_after}" if today.isoformat() >= v.stale_after
            else f"current until {v.stale_after}")


def header(study_id: str, today: date | None = None) -> str:
    """The corpus line for the study header and doctor: version and staleness, or not built."""
    try:
        v = read_version(corpus_path(study_id))
    except Exception:  # noqa: BLE001 - CorpusUnavailable, or a cache inside the repo
        return NOT_BUILT
    if v is None:
        return "corpus build incomplete (resume it) — cannot settle"
    return (f"corpus {v.content_hash} built {v.built_at}, "
            f"{staleness(v, today or date.today())}")


def refresh_notes(study: Any) -> None:
    """Rewrite the study's cell-type and gene notes from the corpus cards (no model call)."""
    from panoscope.agent import config as cfg
    from panoscope.pipeline.stages import notes

    if not (study.inputs / "markers_top.csv").is_file():
        return                                    # no marker workflow, no marker notes
    with cfg.use_study(study):
        notes.run_celltype_notes(study.id, study.dir.parent)
        notes.run_gene_notes(study.id, study.dir.parent)
