"""The biologist's confirmed overrides, applied to what is SHOWN.

Panoscope keeps two things apart on purpose. ``pipeline.v1_verdicts`` computes the
call from one method's output and is never mutated: it is what ``panoscope
build`` writes into the tree, what the golden projection hashes, and what the
CSV exports. This module is the other half — the biologist's own call, laid
over the computed one at the moment something is rendered or written into
prose, producing NEW objects and leaving the computed ones intact.

It lives in ``agent/`` rather than ``ui/`` because that is where the problem
was. The composition was implemented in ``ui/data_access`` while the prose was
generated underneath it in ``agent/fallback``, so after the biologist
overrode c2 Stromal -> CAF the board, the marker header and the exported
overview all said CAF while five of the seven chat answers for that same
cluster went on saying Stromal -- the opening interpretation, the rival
answer, the panel-absence answer, the what-would-settle-it answer and the
confidence answer. That is the silent overrule the memory rule forbids,
committed by the layering rather than by any one line.

What it does NOT do: invent. An override renames a call; it never supplies a
number, moves a band or adds a marker. The statistics in a composed object are
the computed ones, unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from panoscope.agent.types import ClusterVerdict, OffPanelNote


# --------------------------------------------------------------------------- #
# The biologist's notes
# --------------------------------------------------------------------------- #
def override_notes() -> dict:
    """The most-recent firing ``celltype_override`` note per cluster (one that
    carries a new call), keyed by cluster.

    Study-gated by ``memory.notes_in_study``: an override is the biologist's own
    call, so applying it to a study they never made it in attributes their
    judgement somewhere they never gave it.
    """
    from panoscope.agent import memory

    out: dict = {}
    try:
        for n in memory.notes_in_study():  # sorted by created_at -> last write wins
            if (
                getattr(n, "type", "") == "celltype_override"
                and n.scope == "cluster"
                and n.scope_ref.cluster
                and (getattr(n, "subject_cell_type", "") or "").strip()
            ):
                out[n.scope_ref.cluster] = n
    except Exception:  # noqa: BLE001 - a malformed note never breaks the composition
        return {}
    return out


def override_call(cluster: str) -> Optional[str]:
    """The confirmed NEW cell-type call for ``cluster``, or None."""
    ov = override_notes().get(cluster)
    if ov is None:
        return None
    return ov.subject_cell_type.strip()  # override_notes keeps only non-empty calls


# --------------------------------------------------------------------------- #
# Panel-absence notes follow the call
# --------------------------------------------------------------------------- #
def offpanel_notes_for(cell_type: str, existing: tuple = ()) -> tuple:
    """The off-panel canonical markers for ONE call, as ``OffPanelNote``s.

    **Falls back to the genes already on the verdict, re-worded.** Returning
    empty whenever this study's annotation does not name the new call deleted a
    true statement: overriding c2 Stromal -> CAF emptied COL1A1, COL1A2, DCN,
    VIM and FAP out of the exported document while the markers table beside it
    still showed all five, because the table builds rows from the record's own
    evidence. Screen and document then disagreed about which markers were never
    measured -- the one thing ``ui/report`` says must not happen -- on exactly
    the cluster the biologist had curated.

    Being off-panel is a fact about the PANEL, not about the call, so the genes
    stay and only the sentence changes: "COL1A1 is off-panel (never measured);
    its absence is not evidence against CAF" is true whatever the call is
    renamed to. An annotation that DOES name the new call still wins, because
    then there is a real canonical set for it.
    """
    from panoscope.agent import annotation

    ct = (cell_type or "").strip()
    if not ct:
        return tuple(existing)

    def _note(gene: str) -> OffPanelNote:
        return OffPanelNote(
            gene=gene,
            cell_type=ct,
            message=(f"{gene} is off-panel (never measured); its absence is not "
                     f"evidence against {ct}."),
            source="panel:absence",
        )

    derived = annotation.offpanel_canonical(ct)
    if derived:
        return tuple(_note(g) for g in derived)
    return tuple(_note(n.gene) for n in existing)


# --------------------------------------------------------------------------- #
# Composed objects
# --------------------------------------------------------------------------- #
def compose_verdict(v: ClusterVerdict, ov) -> ClusterVerdict:
    """``v`` with ``ov``'s call, lineage and category applied, notes recomputed.

    ``verify`` is raised only when the LITERATURE dissents from the biologist's
    call -- a deterministic property of the stored note, not a judgement made
    here, and not the model's to make.
    """
    new_call = ov.subject_cell_type.strip()  # non-empty: see override_notes
    changes: dict = {"cell_type": new_call, "cell_type_short": new_call}
    if getattr(ov, "subject_lineage", "").strip():
        changes["lineage"] = ov.subject_lineage.strip()
    if getattr(ov, "subject_category", "").strip():
        changes["category"] = ov.subject_category.strip()
    if getattr(getattr(ov, "tension", None), "dissent", False):
        changes["verify"] = True
    changes["offpanel_notes"] = offpanel_notes_for(new_call, v.offpanel_notes)
    # The opening is renamed and carries the new notes; its numbers are untouched.
    changes["opening"] = dataclasses.replace(
        v.opening, cell_type=new_call, offpanel_notes=changes["offpanel_notes"])
    return dataclasses.replace(v, **changes)


def composed_verdict_for(cluster: str) -> ClusterVerdict:
    """This cluster's verdict as it should be SHOWN and spoken about.

    The computed verdict when there is no override, so the default path costs
    one dictionary lookup and changes nothing.
    """
    from panoscope.pipeline.v1_verdicts import verdict_for

    v = verdict_for(cluster)
    ov = override_notes().get(cluster)
    return compose_verdict(v, ov) if ov is not None else v


def computed_call(cluster: str) -> str:
    """The call the METHOD produced, whatever the biologist has since decided.

    Kept so prose can show the tension rather than smooth it over: an answer
    that quietly swapped one label for the other would be the same silent
    overrule in the opposite direction.
    """
    from panoscope.pipeline.v1_verdicts import verdict_for

    return verdict_for(cluster).cell_type


__all__ = [
    "compose_verdict",
    "composed_verdict_for",
    "computed_call",
    "offpanel_notes_for",
    "override_call",
    "override_notes",
]
