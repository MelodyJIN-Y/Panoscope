"""Per-cluster annotation decisions: every cluster is OPEN until the biologist calls it.

The proposal (skill or shipped label) is never changed here; the call sits beside it.
Stored in ``review_state.json``; every transition is also logged to the decision ledger.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from panoscope.data.study import NOT_YET_CALLED
from panoscope.pipeline import store

OPEN, ACCEPTED, OVERRIDDEN, DEFERRED = "OPEN", "ACCEPTED", "OVERRIDDEN", "DEFERRED"
TERMINAL = frozenset({ACCEPTED, OVERRIDDEN, DEFERRED})


def state(decisions: dict, cluster: str) -> str:
    """The cluster's state; no record reads OPEN."""
    return (decisions.get(cluster) or {}).get("state", OPEN)


def call(decisions: dict, cluster: str) -> Optional[str]:
    """The biologist's label (the effective label downstream); None while OPEN or DEFERRED."""
    return (decisions.get(cluster) or {}).get("label")


def _record(cluster: str, entry: dict, dataset_id: Optional[str], root: Optional[Path],
            note_id: Optional[str] = None, base_dir: Any = None) -> dict:
    from panoscope.agent import memory

    decisions = dict(store.load_decisions(dataset_id, root))
    decisions[cluster] = entry
    store.save_decisions(decisions, dataset_id, root, saved_at=entry["at"])
    memory.log_decision(kind=f"decision_{entry['state'].lower()}", cluster=cluster,
                        note_id=note_id, actor=entry["by"],
                        detail=entry.get("label"), base_dir=base_dir)
    return entry


def accept(cluster: str, proposal: str, *, by: str, at: str,
           dataset_id: Optional[str] = None, root: Optional[Path] = None,
           base_dir: Any = None) -> dict:
    """The biologist accepts the proposal as the call. Nothing to accept when uncalled."""
    if not proposal or proposal == NOT_YET_CALLED:
        raise ValueError(f"{cluster} has no proposal to accept; override or defer it")
    return _record(cluster, {"state": ACCEPTED, "label": proposal, "by": by, "at": at},
                   dataset_id, root, base_dir=base_dir)


def defer(cluster: str, *, by: str, at: str, dataset_id: Optional[str] = None,
          root: Optional[Path] = None, base_dir: Any = None) -> dict:
    """A terminal 'not settled here'; carries no label."""
    return _record(cluster, {"state": DEFERRED, "by": by, "at": at},
                   dataset_id, root, base_dir=base_dir)


def override(cluster: str, draft: Any, *, by: str, at: str,
             dataset_id: Optional[str] = None, root: Optional[Path] = None,
             base_dir: Any = None) -> dict:
    """Commit the agent's ``NoteDraft`` (scope + basis tapped) as a lab claim, then the call.

    The note goes through ``memory.save_draft``, so it never merges with the measurement claim.
    """
    from panoscope.agent import memory

    label = (draft.subject_cell_type or "").strip()
    if draft.type != "celltype_override" or not label:
        raise ValueError("an override needs a celltype_override draft naming the new label")
    if draft.scope == "cluster" and draft.cluster != cluster:
        raise ValueError(f"the draft is scoped to {draft.cluster!r}, not {cluster!r}")
    note = memory.save_draft(draft, attributed_to=by, base_dir=base_dir)
    return _record(cluster, {"state": OVERRIDDEN, "label": label, "by": by, "at": at,
                             "note_id": note.id, "scope": draft.scope, "basis": draft.basis},
                   dataset_id, root, note_id=note.id, base_dir=base_dir)


def export_refusal(clusters: Iterable[str], decisions: dict) -> Optional[str]:
    """None when every cluster is terminal, else the refusal naming the OPEN ones."""
    open_ = [c for c in clusters if state(decisions, c) not in TERMINAL]
    if not open_:
        return None
    return ("Export needs a call (accepted, overridden or deferred) on every cluster; "
            f"still OPEN: {', '.join(open_)}")


def label_space(view: Any, expected: Iterable[str] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(candidates from the reference view, expected types shown beside them, never alone)."""
    cands = view.candidates()
    if not cands:
        return (), ()
    keys = {view.key(c) for c in cands}
    return cands, tuple(e for e in expected if view.key(e) not in keys)


class Assessments(NamedTuple):
    """The proposal's band and the same band function on the biologist's label."""

    proposal: Any               # core.verdict.Verdict, as built
    biologist: Any              # core.verdict.Verdict, or None while uncalled
    reads_as: str               # the reference lookup sentence for the biologist's label


def assessments(build: Any, label: Optional[str], view: Any) -> Assessments:
    """Second deterministic assessment: ``label``'s canonical set against the SAME items.

    A label with no reference node gets an empty canonical set, never another tissue's.
    """
    from panoscope.core import verdict as core_verdict

    if label is None:
        return Assessments(build.verdict, None, "")
    # ponytail: reuses the build's items, whose appended absences follow the PROPOSAL's
    # marker list; an on-panel gene of this label that the method table never named has
    # no item, so it is uncounted in K (errs toward verify). Rebuild absences per label
    # if a method stops enumerating its whole table.
    canon = frozenset(str(g).upper() for g in view.canonical(label))
    v = core_verdict.assess(build.evidence.items, canon,
                            build.verdict.replicate_coverage).value
    return Assessments(build.verdict, v, view.reads_as(label))
