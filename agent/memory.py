"""File-based lab memory — a reconciliation layer, not a memory of the user.

Memory is where the biologist's judgment and the literature get reconciled into
something the lab owns. This module is deliberately dumb about "learning": it
writes structured :class:`~agent.types.Note` objects to JSON files under
``context/corrections/`` and re-reads them. Nothing is trained.

What it enforces (the load-bearing invariants):

- **Scope is a choke point.** A cluster-scoped note fires ONLY for its own
  cluster. Dataset- and lab-scoped notes fire for every cluster in the dataset.
  :func:`apply_notes` is the single gate every caller goes through, and it is
  fail-closed (an out-of-scope or wrong-dataset note never fires).
- **Cite on use.** A note may only be applied if it is also cited. :func:`cite_note`
  returns a :class:`~agent.types.Source` (``kind="mem"``) the agent must surface,
  and :func:`render_citation` renders it with any attached tension visible.
- **The value is in the disagreement.** :func:`reconcile` runs an *injected*
  ``literature_search`` callable (testable without MCP), splits the results into
  agreement and dissent, and attaches that :class:`~agent.types.Tension` to the
  note. Citations are only kept if the callable marks them real; a fabricated
  citation is the worst possible failure.

A note is a versioned, git-tracked file. The base directory is configurable
(default ``context/``) so tests can point it at a tmp dir.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from agent import config as cfg
from agent.types import (
    Basis,
    Citation,
    Note,
    NoteDraft,
    NoteType,
    Scope,
    ScopeRef,
    Source,
    Status,
    Tension,
)

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
# An injected literature search: given a free-text query, return zero or more
# Citations. Each Citation carries its own ``stance`` (agree | dissent | ...)
# and ``is_real`` flag. Injecting this keeps reconcile() testable with a stub
# and honest in production (real PMIDs only).
LiteratureSearch = Callable[[str], Iterable[Citation]]

# --------------------------------------------------------------------------- #
# Paths (configurable base dir; default context/)
# --------------------------------------------------------------------------- #
_DEFAULT_BASE: Path = cfg.PROJECT_ROOT / "context"
_CORRECTIONS_SUBDIR = "corrections"
_DECISIONS_SUBDIR = "decisions"
_DECISION_LOG_NAME = "decision_log.jsonl"

_STANCE_DISSENT = "dissent"


def _now_iso() -> str:
    """UTC timestamp, second precision, honest snapshot stamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def corrections_dir(base_dir: Path | str | None = None) -> Path:
    """Directory holding note JSON files. Created on demand."""
    base = Path(base_dir) if base_dir is not None else _DEFAULT_BASE
    d = base / _CORRECTIONS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def decisions_dir(base_dir: Path | str | None = None) -> Path:
    """Directory holding the append-only decision log. Created on demand."""
    base = Path(base_dir) if base_dir is not None else _DEFAULT_BASE
    d = base / _DECISIONS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# (De)serialization — Note <-> plain JSON dict
# --------------------------------------------------------------------------- #
def _citation_from_dict(d: dict) -> Citation:
    return Citation(
        pmid=str(d.get("pmid", "")),
        title=str(d.get("title", "")),
        authors=str(d.get("authors", "")),
        year=int(d.get("year", 0)),
        journal=str(d.get("journal", "")),
        abstract=str(d.get("abstract", "")),
        url=str(d.get("url", "")),
        stance=str(d.get("stance", "context")),
        is_real=bool(d.get("is_real", True)),
        fetched_at=str(d.get("fetched_at", "")),
    )


def _tension_from_dict(d: dict) -> Tension:
    return Tension(
        agree=tuple(_citation_from_dict(c) for c in d.get("agree", ())),
        dissent=tuple(_citation_from_dict(c) for c in d.get("dissent", ())),
        thin=bool(d.get("thin", True)),
        query=str(d.get("query", "")),
        looked_up_at=str(d.get("looked_up_at", "")),
    )


def _note_to_dict(note: Note) -> dict:
    """Serialize a Note to a plain JSON-safe dict (dataclasses.asdict handles nesting)."""
    return asdict(note)


def _note_from_dict(d: dict) -> Note:
    """Rebuild a frozen Note (and its nested frozen objects) from a JSON dict."""
    scope_ref_d = d.get("scope_ref") or {}
    scope_ref = ScopeRef(
        dataset=str(scope_ref_d.get("dataset", cfg.DATASET_ID)),
        cluster=scope_ref_d.get("cluster"),
    )
    return Note(
        id=str(d["id"]),
        claim=str(d["claim"]),
        scope=d["scope"],
        scope_ref=scope_ref,
        basis=d["basis"],
        status=d["status"],
        subject_cell_type=d.get("subject_cell_type"),
        subject_markers=tuple(d.get("subject_markers", ())),
        tension=_tension_from_dict(d.get("tension") or {}),
        author=str(d.get("author", "")),
        created_at=str(d.get("created_at", "")),
        trigger=d.get("trigger", "manual_add"),
        supersedes=d.get("supersedes"),
        # Typed/anchored fields default so pre-existing note files still parse.
        type=d.get("type", "celltype_override"),
        subject_gene_sets=tuple(d.get("subject_gene_sets", ())),
        subject_clusters=tuple(d.get("subject_clusters", ())),
        subject_lineage=str(d.get("subject_lineage", "")),
        subject_category=str(d.get("subject_category", "")),
    )


def _note_path(note_id: str, base_dir: Path | str | None) -> Path:
    return corrections_dir(base_dir) / f"{note_id}.json"


def _write_note(note: Note, base_dir: Path | str | None) -> Path:
    """Write a note to its JSON file (one file per note, git-diffable)."""
    path = _note_path(note.id, base_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_note_to_dict(note), fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return path


# --------------------------------------------------------------------------- #
# Reconciliation — the value is in the disagreement
# --------------------------------------------------------------------------- #
def _reconcile_query(
    claim: str,
    subject_cell_type: Optional[str],
    subject_markers: tuple[str, ...],
    subject_gene_sets: tuple[str, ...] = (),
) -> str:
    """Build the literature query for a note's claim (markers + gene sets + cell type + claim)."""
    parts: list[str] = []
    if subject_markers:
        parts.append(" ".join(subject_markers))
    if subject_gene_sets:
        # HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION -> "Epithelial Mesenchymal Transition"
        parts.append(" ".join(gs.replace("HALLMARK_", "").replace("_", " ") for gs in subject_gene_sets))
    if subject_cell_type:
        parts.append(subject_cell_type)
    parts.append(claim)
    return " ".join(p for p in parts if p).strip()


def reconcile(note: Note, literature_search: Optional[LiteratureSearch] = None) -> Tension:
    """Cross-check a note's claim against the literature and split agree/dissent.

    ``literature_search`` is an INJECTED callable ``(query) -> Iterable[Citation]``
    so this is testable without MCP. Only citations the callable marks
    ``is_real=True`` are kept — a fabricated citation is worse than none. Each
    surviving citation is bucketed by its ``stance``: ``"dissent"`` goes to
    dissent, everything else (agree / context / unclassified) to agree, so the
    biologist's call is kept WITH the tension visible, never smoothed over.

    Returns a fresh :class:`Tension`. When no search is provided or nothing real
    resolves, ``thin=True`` (say the literature is thin; never invent a ref).
    """
    query = _reconcile_query(
        note.claim, note.subject_cell_type, note.subject_markers, note.subject_gene_sets
    )
    looked_up_at = _now_iso()

    if literature_search is None:
        return Tension(agree=(), dissent=(), thin=True, query=query, looked_up_at=looked_up_at)

    try:
        results = list(literature_search(query))
    except Exception:
        # A failed lookup is thin literature, never a fabricated fallback.
        return Tension(agree=(), dissent=(), thin=True, query=query, looked_up_at=looked_up_at)

    real = [c for c in results if isinstance(c, Citation) and c.is_real]
    dissent = tuple(c for c in real if c.stance == _STANCE_DISSENT)
    agree = tuple(c for c in real if c.stance != _STANCE_DISSENT)
    thin = len(real) == 0
    return Tension(
        agree=agree,
        dissent=dissent,
        thin=thin,
        query=query,
        looked_up_at=looked_up_at,
    )


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #
def _validate_note_fields(claim: str, scope: str, basis: str, status: str) -> None:
    """Validate the closed-vocab fields shared by draft + create (raises ValueError)."""
    if not claim or not claim.strip():
        raise ValueError("[memory] note claim must be non-empty")
    if scope not in ("cluster", "dataset", "lab"):
        raise ValueError(f"[memory] invalid scope {scope!r}")
    if basis not in ("paper", "own_validation", "convention"):
        raise ValueError(f"[memory] invalid basis {basis!r}")
    if status not in ("firm", "tentative"):
        raise ValueError(f"[memory] invalid status {status!r}")


def draft_note(
    *,
    claim: str,
    scope: Scope,
    basis: Basis,
    status: Status = "firm",
    cluster: Optional[str] = None,
    subject_cell_type: Optional[str] = None,
    subject_markers: Optional[Iterable[str]] = None,
    note_type: NoteType = "celltype_override",
    subject_gene_sets: Optional[Iterable[str]] = None,
    subject_clusters: Optional[Iterable[str]] = None,
    subject_lineage: str = "",
    subject_category: str = "",
    dataset: str = cfg.DATASET_ID,
    literature_search: Optional[LiteratureSearch] = None,
) -> NoteDraft:
    """Reconcile a proposed note against the literature WITHOUT persisting it.

    This is the first half of capture-at-override: the agent proposes a note (of one
    of the eight :data:`~agent.types.NoteType`\\ s, anchored to a gene / gene set /
    cluster set), we cross-check the claim against the literature (real PMIDs only),
    and return a :class:`~agent.types.NoteDraft` carrying that tension. Nothing is
    written — the biologist confirms scope/basis/status and only then
    :func:`save_draft` persists it. Validates the closed-vocab fields (and a cluster
    for cluster scope) so a malformed draft fails fast.
    """
    _validate_note_fields(claim, scope, basis, status)
    if scope == "cluster":
        if cluster is None:
            raise ValueError("[memory] a cluster-scoped note must name a cluster")
        if cluster not in cfg.KNOWN_CLUSTERS:
            raise ValueError(
                f"[memory] unknown cluster {cluster!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}"
            )

    markers = tuple(subject_markers) if subject_markers else ()
    gene_sets = tuple(subject_gene_sets) if subject_gene_sets else ()
    clusters = tuple(subject_clusters) if subject_clusters else ()
    stub = Note(
        id="__draft__",
        claim=claim.strip(),
        scope=scope,
        scope_ref=ScopeRef(dataset=dataset, cluster=cluster if scope == "cluster" else None),
        basis=basis,
        status=status,
        subject_cell_type=subject_cell_type,
        subject_markers=markers,
        tension=Tension(agree=(), dissent=(), thin=True, query="", looked_up_at=""),
        author="",
        created_at=_now_iso(),
        trigger="override",
        supersedes=None,
        type=note_type,
        subject_gene_sets=gene_sets,
        subject_clusters=clusters,
        subject_lineage=subject_lineage,
        subject_category=subject_category,
    )
    tension = reconcile(stub, literature_search)
    return NoteDraft(
        claim=claim.strip(),
        scope=scope,
        basis=basis,
        status=status,
        cluster=cluster if scope == "cluster" else None,
        subject_cell_type=subject_cell_type,
        subject_markers=markers,
        tension=tension,
        dataset=dataset,
        type=note_type,
        subject_gene_sets=gene_sets,
        subject_clusters=clusters,
        subject_lineage=subject_lineage,
        subject_category=subject_category,
    )


def save_draft(
    draft: NoteDraft,
    *,
    attributed_to: str = "melody.xyjin@gmail.com",
    trigger: str = "override",
    supersedes: Optional[str] = None,
    base_dir: Path | str | None = None,
) -> Note:
    """Persist a (possibly biologist-edited) :class:`~agent.types.NoteDraft`.

    The second half of capture-at-override: the biologist confirmed scope/basis/
    status, so we write the note WITH the tension the draft already carries (no
    second literature lookup). Scope is enforced at save on the FINAL scope — a
    cluster-scoped note must name a cluster; dataset/lab notes drop any cluster so
    they can never masquerade as cluster-scoped. Returns the written Note.
    """
    _validate_note_fields(draft.claim, draft.scope, draft.basis, draft.status)
    if draft.scope == "cluster":
        if not draft.cluster:
            raise ValueError("[memory] a cluster-scoped note must name a cluster")
        if draft.cluster not in cfg.KNOWN_CLUSTERS:
            raise ValueError(
                f"[memory] unknown cluster {draft.cluster!r}; "
                f"known: {sorted(cfg.KNOWN_CLUSTERS)}"
            )
        scope_cluster: Optional[str] = draft.cluster
    else:
        scope_cluster = None

    note_id = uuid.uuid4().hex[:12]
    note = Note(
        id=note_id,
        claim=draft.claim.strip(),
        scope=draft.scope,
        scope_ref=ScopeRef(dataset=draft.dataset, cluster=scope_cluster),
        basis=draft.basis,
        status=draft.status,
        subject_cell_type=draft.subject_cell_type,
        subject_markers=tuple(draft.subject_markers),
        tension=draft.tension,
        author=attributed_to,
        created_at=_now_iso(),
        trigger=trigger,  # type: ignore[arg-type]
        supersedes=supersedes,
        type=draft.type,
        subject_gene_sets=tuple(draft.subject_gene_sets),
        subject_clusters=tuple(draft.subject_clusters),
        subject_lineage=draft.subject_lineage,
        subject_category=draft.subject_category,
    )
    _write_note(note, base_dir)
    log_decision(
        kind="note_created",
        cluster=scope_cluster,
        note_id=note_id,
        actor=attributed_to,
        detail=f"scope={draft.scope} basis={draft.basis} status={draft.status} (confirmed)",
        base_dir=base_dir,
    )
    return note


def create_note(
    *,
    claim: str,
    scope: Scope,
    basis: Basis,
    status: Status = "firm",
    cluster: Optional[str] = None,
    subject_cell_type: Optional[str] = None,
    subject_markers: Optional[Iterable[str]] = None,
    note_type: NoteType = "celltype_override",
    subject_gene_sets: Optional[Iterable[str]] = None,
    subject_clusters: Optional[Iterable[str]] = None,
    subject_lineage: str = "",
    subject_category: str = "",
    dataset: str = cfg.DATASET_ID,
    attributed_to: str = "melody.xyjin@gmail.com",
    trigger: str = "override",
    supersedes: Optional[str] = None,
    literature_search: Optional[LiteratureSearch] = None,
    base_dir: Path | str | None = None,
) -> Note:
    """Create, reconcile, and persist a lab note. Returns the written Note.

    Scope enforcement is baked in at birth: a ``scope="cluster"`` note MUST name
    a ``cluster`` (else ValueError) and stores it on ``scope_ref``; dataset/lab
    notes never carry a cluster, so they cannot masquerade as cluster-scoped.

    ``literature_search`` (injected, testable) drives :func:`reconcile` so the
    note carries its tension from the moment it is captured — at the override,
    where the lab's knowledge diverges from the default.
    """
    if not claim or not claim.strip():
        raise ValueError("[memory] note claim must be non-empty")
    if scope not in ("cluster", "dataset", "lab"):
        raise ValueError(f"[memory] invalid scope {scope!r}")
    if basis not in ("paper", "own_validation", "convention"):
        raise ValueError(f"[memory] invalid basis {basis!r}")
    if status not in ("firm", "tentative"):
        raise ValueError(f"[memory] invalid status {status!r}")

    if scope == "cluster":
        if cluster is None:
            raise ValueError("[memory] a cluster-scoped note must name a cluster")
        if cluster not in cfg.KNOWN_CLUSTERS:
            raise ValueError(
                f"[memory] unknown cluster {cluster!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}"
            )
        scope_cluster: Optional[str] = cluster
    else:
        # dataset/lab notes are dataset-wide by construction; drop any stray cluster.
        scope_cluster = None

    markers = tuple(subject_markers) if subject_markers else ()
    gene_sets = tuple(subject_gene_sets) if subject_gene_sets else ()
    clusters = tuple(subject_clusters) if subject_clusters else ()

    # Reconcile against the literature FIRST, so the note is born with its tension.
    stub_note = Note(
        id="__pending__",
        claim=claim,
        scope=scope,
        scope_ref=ScopeRef(dataset=dataset, cluster=scope_cluster),
        basis=basis,
        status=status,
        subject_cell_type=subject_cell_type,
        subject_markers=markers,
        tension=Tension(agree=(), dissent=(), thin=True, query="", looked_up_at=""),
        author=attributed_to,
        created_at=_now_iso(),
        trigger=trigger,  # type: ignore[arg-type]
        supersedes=supersedes,
        type=note_type,
        subject_gene_sets=gene_sets,
        subject_clusters=clusters,
        subject_lineage=subject_lineage,
        subject_category=subject_category,
    )
    tension = reconcile(stub_note, literature_search)

    note_id = uuid.uuid4().hex[:12]
    note = Note(
        id=note_id,
        claim=claim,
        scope=scope,
        scope_ref=ScopeRef(dataset=dataset, cluster=scope_cluster),
        basis=basis,
        status=status,
        subject_cell_type=subject_cell_type,
        subject_markers=markers,
        tension=tension,
        author=attributed_to,
        created_at=stub_note.created_at,
        trigger=trigger,  # type: ignore[arg-type]
        supersedes=supersedes,
        type=note_type,
        subject_gene_sets=gene_sets,
        subject_clusters=clusters,
        subject_lineage=subject_lineage,
        subject_category=subject_category,
    )
    _write_note(note, base_dir)
    log_decision(
        kind="note_created",
        cluster=scope_cluster,
        note_id=note_id,
        actor=attributed_to,
        detail=f"scope={scope} basis={basis} status={status}",
        base_dir=base_dir,
    )
    return note


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def read_notes(base_dir: Path | str | None = None) -> list[Note]:
    """Read every note JSON under ``corrections/``, sorted by created_at.

    Malformed files are skipped rather than crashing the app; a note the loader
    cannot parse simply does not fire (fail-closed).
    """
    out: list[Note] = []
    for path in sorted(corrections_dir(base_dir).glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                out.append(_note_from_dict(json.load(fh)))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda n: (n.created_at, n.id))
    return out


# Alias matching the BLUEPRINT signature (`list_notes`).
def list_notes(dataset: Optional[str] = None, base_dir: Path | str | None = None) -> list[Note]:
    """Read notes, optionally filtered to one dataset."""
    notes = read_notes(base_dir)
    if dataset is None:
        return notes
    return [n for n in notes if n.scope_ref.dataset == dataset]


def get_note(note_id: str, base_dir: Path | str | None = None) -> Note | None:
    """Load a single note by id, or None if it does not exist / cannot parse."""
    path = _note_path(note_id, base_dir)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return _note_from_dict(json.load(fh))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Scope enforcement — THE choke point
# --------------------------------------------------------------------------- #
def note_in_scope(note: Note, *, cluster: Optional[str], dataset: str = cfg.DATASET_ID) -> bool:
    """True iff ``note`` is allowed to fire in the given (cluster, dataset) context.

    Fail-closed rule (lab ⊇ dataset ⊇ cluster):

    - ``lab``     — fires for every dataset and every cluster.
    - ``dataset`` — fires only within its own dataset, for any cluster in it.
    - ``cluster`` — fires ONLY for its own cluster within its own dataset.

    A cluster-scoped note therefore NEVER fires for a different cluster, and a
    dataset/cluster note NEVER fires for a different dataset.
    """
    if note.scope == "lab":
        return True

    # dataset and cluster scopes are pinned to a dataset.
    if note.scope_ref.dataset != dataset:
        return False

    # An anchored note (cross_cluster) belongs to a SET of clusters and fires on each
    # of them only — even though it is dataset-scoped, it never fires dataset-wide.
    if note.subject_clusters:
        return cluster is not None and cluster in note.subject_clusters

    if note.scope == "dataset":
        return True

    # cluster scope: must match the requested cluster exactly.
    if note.scope == "cluster":
        return cluster is not None and note.scope_ref.cluster == cluster

    return False  # unknown scope -> fail closed


def apply_notes(
    cluster: Optional[str],
    dataset: str = cfg.DATASET_ID,
    base_dir: Path | str | None = None,
) -> list[Note]:
    """Return only the notes whose scope FIRES for ``(cluster, dataset)``.

    This is the single gate every caller goes through. Cluster-scoped notes fire
    only for their own cluster; dataset and lab notes fire for all. Everything
    else is filtered out (fail-closed).
    """
    return [
        n for n in read_notes(base_dir) if note_in_scope(n, cluster=cluster, dataset=dataset)
    ]


# --------------------------------------------------------------------------- #
# Cite on use
# --------------------------------------------------------------------------- #
def cite_note(note: Note) -> Source:
    """Return the citation Source for a note (kind="mem"). Cite-on-use primitive.

    The agent must surface this whenever it applies the note — a note may not be
    used silently. ``ref`` is the note id (renders inline as ``[note:<id>]``);
    ``value`` is the claim; ``detail`` records scope + basis + status so the
    provenance travels with the citation.
    """
    scope_txt = note.scope
    if note.scope == "cluster" and note.scope_ref.cluster:
        scope_txt = f"cluster:{note.scope_ref.cluster}"
    elif note.scope == "dataset":
        scope_txt = f"dataset:{note.scope_ref.dataset}"
    detail = f"scope={scope_txt} basis={note.basis} status={note.status}"
    return Source(kind="mem", ref=note.id, value=note.claim, detail=detail)


def render_citation(note: Note, *, refresh: Optional[LiteratureSearch] = None) -> str:
    """Render a cite-on-use markdown line for a note, tension made visible.

    Shows the claim, its ``[note:<id>]`` handle, scope/basis/status, and — when
    the note carries tension — the count of agreeing vs dissenting citations
    with their PMIDs. When ``refresh`` is given, the tension is re-reconciled
    live before rendering (real PMIDs only). Never smooths disagreement over.
    """
    tension = reconcile(note, refresh) if refresh is not None else note.tension
    src = cite_note(note)
    line = f'"{note.claim}" [note:{note.id}] ({src.detail})'

    if tension.thin and not tension.agree and not tension.dissent:
        return line + " — literature is thin; no supporting reference found."

    def _pmids(cites: tuple[Citation, ...]) -> str:
        return ", ".join(f"PMID:{c.pmid}" for c in cites if c.pmid)

    bits: list[str] = []
    if tension.agree:
        bits.append(f"agree ({len(tension.agree)}): {_pmids(tension.agree)}")
    if tension.dissent:
        bits.append(f"dissent ({len(tension.dissent)}): {_pmids(tension.dissent)}")
    if bits:
        line += " — tension: " + "; ".join(bits)
    return line


# --------------------------------------------------------------------------- #
# Supersede (immutable "edit")
# --------------------------------------------------------------------------- #
def supersede_note(
    old_id: str,
    *,
    base_dir: Path | str | None = None,
    literature_search: Optional[LiteratureSearch] = None,
    **new_fields,
) -> Note:
    """Create a new note that supersedes ``old_id`` (notes are never mutated).

    Carries forward the old note's fields unless overridden in ``new_fields``.
    Records the ``supersedes`` link and logs the decision.
    """
    old = get_note(old_id, base_dir)
    if old is None:
        raise KeyError(f"[memory] cannot supersede unknown note {old_id!r}")

    fields = dict(
        claim=old.claim,
        scope=old.scope,
        basis=old.basis,
        status=old.status,
        cluster=old.scope_ref.cluster,
        subject_cell_type=old.subject_cell_type,
        subject_markers=old.subject_markers,
        note_type=old.type,
        subject_gene_sets=old.subject_gene_sets,
        subject_clusters=old.subject_clusters,
        subject_lineage=old.subject_lineage,
        subject_category=old.subject_category,
        dataset=old.scope_ref.dataset,
        attributed_to=old.author,
        trigger="manual_add",
    )
    fields.update(new_fields)
    fields["supersedes"] = old_id
    return create_note(
        base_dir=base_dir,
        literature_search=literature_search,
        **fields,
    )


# --------------------------------------------------------------------------- #
# Decision log (append-only ledger under context/decisions/)
# --------------------------------------------------------------------------- #
def log_decision(
    *,
    kind: str,
    cluster: Optional[str] = None,
    note_id: Optional[str] = None,
    actor: str = "melody.xyjin@gmail.com",
    detail: Optional[str] = None,
    base_dir: Path | str | None = None,
) -> dict:
    """Append one event to the decision log (JSONL). Returns the written event.

    Every memory mutation and every applied override should leave a dated,
    attributed trace here. Append-only: the ledger is the audit trail.
    """
    event = {
        "ts": _now_iso(),
        "kind": kind,
        "cluster": cluster,
        "note_id": note_id,
        "actor": actor,
        "detail": detail,
    }
    log_path = decisions_dir(base_dir) / _DECISION_LOG_NAME
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event


def read_decisions(base_dir: Path | str | None = None) -> list[dict]:
    """Read the decision log as a list of events (empty if none)."""
    log_path = decisions_dir(base_dir) / _DECISION_LOG_NAME
    if not log_path.exists():
        return []
    out: list[dict] = []
    with open(log_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out
