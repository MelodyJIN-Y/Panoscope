"""The bridge from the note schema to ``Claim`` (IMPLEMENTATION §10, the memory row).

A **reader, not a rewriter**: maps each :class:`~panoscope.agent.types.Note` under
``context/corrections/`` onto one or more :class:`~panoscope.types.claim.Claim` objects
and never touches the note's bytes. ``memory._note_from_dict`` reads both the pre-typed
and current note shapes.

**Scope.** The claim lattice has no cluster rung, so the cluster is carried by the
**subject**, and the scope string does the dataset gate. Together they reproduce
``memory.note_in_scope`` exactly:

===============================  ===============  ================================
note                             fires for        claim surfaces for
===============================  ===============  ================================
``scope=cluster`` (c2)           c2 only          subject ``unit:c2``
``subject_clusters={c5,c7}``     c5 and c7        subjects ``unit:c5``, ``unit:c7``
``scope=dataset`` + anchor       every cluster    subject = the gene / gene set
``scope=dataset``, no anchor     every cluster    subject = the cell type
``scope=lab``                    everywhere       ``scope="lab"``
===============================  ===============  ================================

A note the anchoring rules cannot place raises :class:`UnanchoredNote` rather than being
dropped.

- The gene a cluster note is about rides in ``depends_on`` (the claim rests on it), so
  ``evaluability`` applies the panel-absence rule to it.
- ``Note.tension``: each real citation becomes a LITERATURE claim about the same subject,
  and the LAB claim's ``tension`` names the dissenting ones. ``is_real=False`` citations
  become nothing.
- :func:`lookup_state` derives the three-state literature signal from fields the note
  already persists (``looked_up_at``), reusing ``corpus.store.Resolution``; it ignores
  the stored ``thin`` boolean.

Clock-free: every timestamp here is one the note already carried.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

from panoscope.agent.types import Citation, Note, Tension
from panoscope.claims.store import ClaimStore
from panoscope.corpus.store import Resolution
from panoscope.types.claim import Authority, Claim
from panoscope.types.conditions import Conditions
from panoscope.types.subject import (
    Entity, Subject, cell_type, gene, gene_set, subject, unit,
)

__all__ = ["UnanchoredNote", "load_notes_as_claims", "lookup_state", "note_to_claims"]

_DISSENT: Final = "dissent"
_STANCES: Final[frozenset[str]] = frozenset({"agree", _DISSENT})


class UnanchoredNote(ValueError):
    """Raised for a note that names nothing a claim can be about.

    A dataset- or lab-scoped note with no cluster, no marker, no gene set and no cell
    type is free text with a scope attached. It has no subject, so it would key to
    nothing in the store and never be retrieved -- the silent-disappearance failure
    the fail-closed gates exist to move to the write.
    """


# --------------------------------------------------------------------------- #
# The thin-literature signal, as three states (the gap the corpus closed already).
# Derived from fields the note already stores; nothing is added to the JSON.
# --------------------------------------------------------------------------- #
def lookup_state(tension: Tension) -> Resolution:
    """Did the literature pass resolve, come back empty, or never complete?

    * :attr:`Resolution.RESOLVED` -- real citations came back.
    * :attr:`Resolution.NOT_FOUND` -- a lookup COMPLETED and nothing real cleared it.
      This is the productive half: the literature here is genuinely thin, and saying
      so is a finding.
    * :attr:`Resolution.NOT_LOOKED_UP` -- no lookup completed (none was injected, or
      the connector raised). It says nothing at all about the literature, and must
      not be rendered in the same voice as the case above.

    ``looked_up_at`` is the discriminator and it is already persisted:
    ``memory.reconcile`` stamps it only when a search returned, and leaves it empty on
    both no-lookup paths. The stored ``thin`` boolean is deliberately NOT read -- same
    reason ``claims.store.thin_literature`` derives rather than trusts.
    """
    if tension.agree or tension.dissent:
        return Resolution.RESOLVED
    if tension.looked_up_at:
        return Resolution.NOT_FOUND
    return Resolution.NOT_LOOKED_UP


# --------------------------------------------------------------------------- #
# Anchoring: what the claim is ABOUT, and what it RESTS ON
# --------------------------------------------------------------------------- #
def _rests_on(note: Note) -> tuple[Entity, ...]:
    """The entities a note's reading rests on: its genes, gene sets and cell type.

    These go in ``depends_on`` rather than into the subject whenever the note is
    pinned to a cluster, because the note fires for that cluster and not wherever the
    gene is mentioned. ``evaluability`` then gates them against the panel and
    annotation frames, so a note anchored to an off-panel gene says so by itself.
    """
    out = [gene(g) for g in note.subject_markers]
    out += [gene_set(s) for s in note.subject_gene_sets]
    if note.subject_cell_type:
        out.append(cell_type(note.subject_cell_type))
    return tuple(out)


def _subjects(note: Note) -> tuple[Subject, ...]:
    """Every subject this note becomes a claim about. Narrowest placement wins.

    A cluster pin (``scope="cluster"`` or an explicit ``subject_clusters`` set) always
    wins over a gene or gene-set anchor: it is the narrower reading, and widening is
    the one direction a scope migration must never go.
    """
    clusters = note.subject_clusters or (
        (note.scope_ref.cluster,)
        if note.scope == "cluster" and note.scope_ref.cluster
        else ()
    )
    if clusters:
        return tuple(subject("unit", u=unit(c)) for c in clusters)

    # Dataset- or lab-wide: the note genuinely fires everywhere in its scope, so the
    # anchor it names IS the subject.
    out = [subject("gene", g=gene(g)) for g in note.subject_markers]
    out += [subject("gene_set", set=gene_set(s)) for s in note.subject_gene_sets]
    if out:
        return tuple(out)
    if note.subject_cell_type:
        return (subject("cell_type", ct=cell_type(note.subject_cell_type)),)
    raise UnanchoredNote(
        f"note {note.id}: scope {note.scope!r} names no cluster, gene, gene set or "
        "cell type, so there is nothing for a claim to be about. It would key to "
        "nothing and never be retrieved."
    )


def _scope(note: Note) -> str:
    """Note scope -> claim scope. The cluster rung lives in the subject, not here.

    Refuses anything outside the closed vocabulary rather than falling through to
    ``dataset:``. The else-branch was a fail-OPEN on a scope nobody recognised: a
    note reading ``"cluster "`` lost its cluster pin in ``_subjects`` AND got a
    dataset-wide scope here, so a one-cluster judgement became a claim about the
    whole study. ``agent.memory._note_from_dict`` now refuses such a note on read;
    this is the second lock, because ``note_to_claims`` is also called with notes
    built in memory.
    """
    if note.scope not in ("cluster", "dataset", "lab"):
        raise UnanchoredNote(
            f"note {note.id}: scope {note.scope!r} is not one of cluster/dataset/"
            "lab, so there is no rung to place this claim on. Refusing rather than "
            "widening it to the dataset."
        )
    return "lab" if note.scope == "lab" else f"dataset:{note.scope_ref.dataset}"


def _predicate(note: Note, subj: Subject) -> tuple[str, str]:
    """``(predicate, object)``.

    ``is_cell_type`` only for a note that replaces the call on a UNIT, which is what
    the registered predicate says it means. On any other subject the same note is a
    local reading, and ``means_here`` -- "what the subject means UNDER THESE
    CONDITIONS" -- is the honest predicate for all seven remaining note types. A
    ``cell_type``-subject override would otherwise assert that a cell type is itself.
    """
    if (note.type == "celltype_override" and note.subject_cell_type
            and subj.shape == "unit"):
        return "is_cell_type", note.subject_cell_type
    return "means_here", note.claim


# --------------------------------------------------------------------------- #
# Note -> Claim
# --------------------------------------------------------------------------- #
def _lit_claims(cites: Iterable[Citation], subj: Subject, scope: str,
                lab_id: str, created: str) -> tuple[Claim, ...]:
    """One LITERATURE claim per REAL citation. A fabricated one becomes nothing.

    ``conditions`` stays None by design: a paper read off a PMID has no study intake
    behind it, so its assay and tissue are genuinely unknown (``types/claim.py``).
    """
    out: list[Claim] = []
    for c in cites:
        if not c.is_real or not c.pmid:
            continue
        out.append(Claim(
            id=f"{lab_id}:lit:{c.pmid}",
            subject=subj,
            predicate="means_here",
            object=c.title.strip() or f"pmid {c.pmid}",
            authority=Authority.LITERATURE,
            source=c.pmid if ":" in c.pmid else f"pmid:{c.pmid}",   # pmcid:/doi: kept whole
            scope=scope,
            # A stance the reconciliation pass could not classify is not a firm
            # reading of the paper; saying so is cheaper than pretending.
            status="firm" if c.stance in _STANCES else "tentative",
            created=created,
        ))
    return tuple(out)


def note_to_claims(note: Note, conditions: Conditions) -> tuple[Claim, ...]:
    """One note -> its LAB claim(s) and the LITERATURE claims they are in tension with.

    ``conditions`` is the study intake the note was formed under and is supplied by
    the caller, never invented here: ``Claim`` refuses a LAB claim without one
    (``UnformedLabClaim``), because a lab judgement stored without the frames it was
    formed under either applies everywhere or nowhere.

    Nothing is written, nothing is merged, nothing is deduped, and the note file is
    not touched.
    """
    scope = _scope(note)
    subjects = _subjects(note)
    rests_on = _rests_on(note)
    many = len(subjects) > 1
    out: list[Claim] = []
    for subj in subjects:
        # A multi-subject note (a cross-cluster call) needs one id per subject, and
        # the anchor is the readable, stable spelling -- an index would renumber if
        # the note's cluster set were ever re-read in another order.
        anchor = subj.parts[0][1]
        lab_id = f"lab:{note.id}:{anchor.id}" if many else f"lab:{note.id}"
        predicate, obj = _predicate(note, subj)
        agree = _lit_claims(note.tension.agree, subj, scope, lab_id,
                            note.tension.looked_up_at)
        dissent = _lit_claims(note.tension.dissent, subj, scope, lab_id,
                              note.tension.looked_up_at)
        out.append(Claim(
            id=lab_id,
            subject=subj,
            predicate=predicate,
            object=obj,
            authority=Authority.LAB,
            source=f"note:{note.id}",
            conditions=conditions,
            # The subject's own entity is not a dependency of itself.
            depends_on=tuple(e for e in rests_on if e != anchor),
            scope=scope,
            status=note.status,          # firm | tentative, kept 1:1 (§10)
            created=note.created_at,
            author=note.author or None,
            # The disagreement is the product: the call keeps a pointer to every
            # paper that argues against it, by id, so neither can be rendered
            # without the other being one click away.
            tension=tuple(c.id for c in dissent),
        ))
        out.extend(agree)
        out.extend(dissent)
    return tuple(out)


def load_notes_as_claims(base_dir: Path | str | None,
                         conditions: Conditions,
                         *, dataset: str | None = None) -> ClaimStore:
    """Every note under ``base_dir`` as claims, in ``memory.read_notes`` order.

    Reads through ``agent.memory`` rather than re-parsing the JSON: that loader is the
    one that already accepts both note shapes, and a second opinion about the file
    format is how the two drift apart. ``memory`` is imported lazily because it pulls
    ``agent.config``, and importing this module must not read a dataset.

    An unanchorable note raises. During a migration, loud beats quiet.
    """
    from panoscope.agent import memory

    notes = (memory.list_notes(dataset, base_dir) if dataset
             else memory.read_notes(base_dir))
    return ClaimStore(tuple(
        c for n in notes for c in note_to_claims(n, conditions)
    ))
