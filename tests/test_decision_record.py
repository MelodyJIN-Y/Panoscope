"""The decision layer (interface-design §5): what a commit records, what it must never
touch, and the parts of the DecisionRecord that are not built yet (strict gaps).

Expected values come from the demo's committed inputs (verdicts.csv) or the spec's
own lists, never from the module under test.
"""
from __future__ import annotations

import csv
import inspect
import json
import os
from pathlib import Path

import pytest

from panoscope.pipeline import decisions as D
from tests.conftest import DEMO_ID, clear_all_caches

AT = "2026-11-09T00:00:00"


def _computed(app_tree) -> dict[str, dict]:
    with (app_tree / DEMO_ID / "interp" / "verdicts.csv").open() as f:
        return {r["cluster"]: r for r in csv.DictReader(f)}


def _override(cluster: str, label: str, dissent: tuple = ()) -> dict:
    """The review panel's 'Call it yourself' commit, through the real decisions API."""
    from panoscope.agent import config as cfg
    from panoscope.agent.types import NoteDraft, Tension

    draft = NoteDraft(claim=f"{cluster} is {label} by our IHC", scope="cluster",
                      basis="own_validation", status="firm", cluster=cluster,
                      subject_cell_type=label, subject_markers=(), dataset=cfg.DATASET_ID,
                      tension=Tension((), dissent, not dissent, "", ""))
    return D.override(cluster, draft, by="tester", at=AT)


def _frozen(app_tree) -> dict[str, bytes]:
    interp = app_tree / DEMO_ID / "interp"
    return {str(p): p.read_bytes()
            for p in [interp / "verdicts.csv", *(interp / "records").rglob("*.json")]}


# --------------------------------------------------------------------------- #
# §5.8.7  Decision grounding: a decision never moves the computed reading
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-5.8.7-01")
def test_no_commit_moves_the_computed_band_score_or_record(app_tree) -> None:
    """Accept c1, override c3, defer c5: the built band, confidence_score and every
    record byte stay as the build wrote them, and only a terminal state per exported
    unit lets the export through."""
    from panoscope.ui import data_access as da

    want = _computed(app_tree)
    before = _frozen(app_tree)
    D.accept("c1", want["c1"]["cell_type"], by="tester", at=AT)
    _override("c3", "T_Cells")
    D.defer("c5", by="tester", at=AT)
    assert _frozen(app_tree) == before
    got = {v.cluster: v for v in da.all_verdicts()}
    for c in ("c1", "c3", "c5"):
        assert (got[c].confidence, got[c].confidence_score) == (
            want[c]["confidence"], float(want[c]["confidence_score"])), c
    decs = da.decisions()
    assert D.export_refusal(["c1", "c3", "c5"], decs) is None
    assert D.export_refusal(list(want), decs).endswith("c2, c4, c6, c7, c8, c9")


@pytest.mark.req("UI-5.2-07")
def test_a_changed_record_never_auto_revokes_the_call(app_tree) -> None:
    """The claim under a decision changes (a rebuild rewrote c1's record): the call
    stays ACCEPTED with its label. Flagging it STALE is the gap below; revoking it
    silently is what must never happen."""
    from panoscope.ui import data_access as da

    label = _computed(app_tree)["c1"]["cell_type"]
    D.accept("c1", label, by="tester", at=AT)
    rec = app_tree / DEMO_ID / "interp" / "records" / "assignment" / "c1.json"
    rec.write_text(rec.read_text().replace('"prose": ', '"prose_rebuilt": 1, "prose": ', 1))
    clear_all_caches()
    assert da.decision("c1")["state"] == D.ACCEPTED and da.effective_call("c1") == label


@pytest.mark.req("UI-5.2-07", "UI-5.8.7-01", gap=True)
def test_a_decision_pins_the_claim_it_was_made_on_so_a_change_reads_stale(app_tree) -> None:
    """§5.2: STALE overlays a committed decision whose claim_sha changed; §5.8.7: the
    claim_sha resolves. Today a decision records no claim_sha, so nothing can go stale."""
    entry = D.accept("c1", _computed(app_tree)["c1"]["cell_type"], by="tester", at=AT)
    assert entry.get("claim_sha") or entry.get("claim_ref"), sorted(entry)
    assert getattr(D, "STALE", None) == "STALE"


# --------------------------------------------------------------------------- #
# §5.4  The DecisionRecord and its closed reason-code vocabulary
# --------------------------------------------------------------------------- #
_RECORD_FIELDS = {"id", "ts", "actor", "dataset", "unit", "workflow_id", "claim_ref",
                  "decision", "final_call", "reason_codes", "note_ids", "exhibits",
                  "argument_digest", "supersedes"}
_REASON_CODES = {"OFFPANEL_CANON", "TISSUE_CONTEXT", "OWN_VALIDATION", "LAB_CONVENTION",
                 "SPILLOVER", "LIT_CONTRADICTS", "THIN_EVIDENCE", "NEEDS_ASSAY"}


@pytest.mark.req("UI-5.4-01", gap=True)
def test_a_commit_writes_every_decision_record_field(app_tree) -> None:
    entry = D.accept("c1", _computed(app_tree)["c1"]["cell_type"], by="tester", at=AT)
    assert _RECORD_FIELDS - set(entry) == set(), sorted(entry)


@pytest.mark.req("UI-5.4-02", gap=True)
def test_reason_codes_are_the_closed_vocabulary() -> None:
    assert set(getattr(D, "REASON_CODES", ())) == _REASON_CODES
    assert "reason_codes" in inspect.signature(D.defer).parameters


@pytest.mark.req("UI-5.3-03", gap=True)
def test_a_commit_can_attach_an_exhibit_receipt() -> None:
    """§5.3 step 4: {view, subject, bin_um, artifact_sha} attached at commit."""
    assert "exhibits" in inspect.signature(D.accept).parameters


@pytest.mark.req("UI-5.5-01", gap=True)
def test_the_argument_digest_is_frozen_into_the_commit(app_tree) -> None:
    entry = _override("c3", "T_Cells")
    assert "argument_digest" in entry, sorted(entry)


# --------------------------------------------------------------------------- #
# §5.1  Layer 2 (chat) never writes layer 3 (decisions) implicitly
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-5.1-02")
def test_a_chat_note_never_writes_the_decision_ledger_and_no_model_parses_a_commit(
        app_tree) -> None:
    """A confirmed chat override note leaves review_state's decisions untouched; the
    state machine imports no model client. Pinned as built: the note still PROJECTS to
    OVERRIDDEN on the rail (test plan decision §6.7, awaiting the user)."""
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.pipeline import store
    from panoscope.ui import data_access as da

    memory.create_note(claim="c3 is Fooblast", scope="cluster", basis="own_validation",
                       cluster="c3", subject_cell_type="Fooblast",
                       note_type="celltype_override", dataset=cfg.DATASET_ID,
                       literature_search=lambda _q: ())
    assert store.load_decisions() == {}
    assert da.decision("c3")["state"] == D.OVERRIDDEN          # §6.7: pinned, not endorsed
    src = Path(D.__file__).read_text()
    assert not any(w in src for w in ("anthropic", "agent.loop", "agent import loop"))


# --------------------------------------------------------------------------- #
# §5.3 step 10  The holistic re-check recomputes after each commit
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-5.3-08")
def test_an_override_commit_recomputes_the_cross_cluster_check(app_tree) -> None:
    """verdicts.csv calls c5 T_Cells. Calling c3 T_Cells too makes two clusters share a
    call, which the whole-set check must now report."""
    from panoscope.ui import data_access as da

    computed = _computed(app_tree)
    assert computed["c5"]["cell_type"] == "T_Cells"
    assert len({r["cell_type"] for r in computed.values()}) == len(computed)
    assert da.holistic().set_is_coherent
    _override("c3", "T_Cells")
    after = da.holistic()
    assert not after.set_is_coherent
    assert any("T_Cells" in n for n in after.coherence_notes), after.coherence_notes


@pytest.mark.req("UI-5.3-08", gap=True)
def test_a_deferral_shows_in_the_cross_cluster_check(app_tree) -> None:
    from panoscope.ui import data_access as da

    D.defer("c1", by="tester", at=AT)
    notes = da.holistic().coherence_notes
    assert any("c1" in n and "defer" in n.lower() for n in notes), notes


# --------------------------------------------------------------------------- #
# Logs and propagation (ARCH §9.12, §10.6; corpus A11)
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-10.6-02", "CORP-A11", gap=True)
def test_an_override_against_the_literature_appends_a_disagreement_event(app_tree) -> None:
    _override("c3", "T_Cells")
    names = [p.name for p in Path(os.environ["PANOSCOPE_CONTEXT_DIR"]).rglob("*")]
    assert names, "the override wrote nothing at all"
    assert any("disagree" in n for n in names), names


@pytest.mark.req("ARCH-9.12-01", gap=True)
def test_a_flagged_call_carries_its_flag_into_the_clusters_pathways_record(app_tree) -> None:
    """verdicts.csv sets verify=TRUE on c9 only; its enrichment record must say the call
    it depends on is flagged ('N downstream results depend on a call I flagged')."""
    flagged = [c for c, r in _computed(app_tree).items() if r.get("verify") == "TRUE"]
    assert flagged == ["c9"]
    rec = json.loads((app_tree / DEMO_ID / "interp" / "records" / "characterisation"
                      / "c9.json").read_text())
    assert any("upstream" in d["rule"] or "flag" in d["rule"] for d in rec["decisions"]), \
        sorted({d["rule"] for d in rec["decisions"]})
