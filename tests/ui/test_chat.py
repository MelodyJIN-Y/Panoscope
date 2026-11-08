"""The chat panes, driven through the REAL agent loop with a scripted client
(plan Stage 5: 5.13-5.16, 5.27, 5.31).

Only ``messages.create`` is fake (tests/fixtures/fake_model.py): the tool loop, the
grounding gate, the repair round, the fallback and the draft card all run. Expected
values come from the study's files or the spec, never from the module that renders them.
"""
from __future__ import annotations

import csv
import re

import pytest

from panoscope.agent.types import Citation
from tests.conftest import DEMO_ID
from tests.fixtures.fake_model import forget_agent, reply, tool_use
from tests.ui.screen import open_app, plain, raw_markdown, run, screen_text, widget_keys

_FALLBACK = "fallback: the computed reading, not a model answer"
# The verify flag line (resp.verify), distinct from fallback prose that says "re-check".
_VERIFY_FLAG = "\u2691 re-check this"


def _csv_verify(root, name: str) -> dict[str, bool]:
    with (root / DEMO_ID / "interp" / name).open() as f:
        col = "verify" if name == "verdicts.csv" else "cluster_verify"
        return {r["cluster"]: r[col] == "TRUE" for r in csv.DictReader(f)}


def _notes() -> list:
    from panoscope.agent import memory, tools

    return memory.read_notes(tools.memory_base_dir())


def _decisions() -> dict:
    from panoscope.pipeline import store

    return store.load_decisions(DEMO_ID)


def _open_chat(cluster: str):
    at = open_app("markers", cluster=cluster)
    return run(at.toggle(key=f"rv_reveal_{cluster}").set_value(True))


def _ask(at, cluster: str, text: str, *, form: str = "ask_form", box: str | None = None):
    at.text_input(key=box or f"conv_ask_input_{cluster}").input(text)
    return run(at.button(key=f"FormSubmitter:{form}_{cluster}-Ask").click())


def _last_agent_turn(at) -> str:
    """The newest agent bubble's raw html (the thread's last ``turn a``)."""
    turns = [t for t in raw_markdown(at).split('<div class="turn ') if t.startswith('a"')]
    assert turns, "no agent turn rendered"
    return turns[-1]


def _rail(at) -> dict[str, str]:
    return {b.key: b.label for b in at.button if (b.key or "").startswith("rail_")}


@pytest.fixture
def offline_lit(monkeypatch):
    """The literature reconciler, answered from a fixture list (no network)."""
    from panoscope.agent import tools

    found: list[Citation] = []
    monkeypatch.setattr(tools, "_LITERATURE_SEARCH_FN", lambda q: list(found))
    return found


# --------------------------------------------------------------------------- #
# 5.13  Every real id in an answer is a link, PMCIDs included
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.9-05")
def test_a_pmid_and_a_pmcid_in_an_answer_are_both_links(app_tree, fake_model, offline_lit):
    fake_model.real_ids |= {"12345678", "PMC7654321"}
    fake_model.script("c1 reads as tumour epithelium; see PMID:12345678 and PMC7654321.")
    at = _ask(_open_chat("c1"), "c1", "what is c1?")
    turn = _last_agent_turn(at)
    assert _FALLBACK not in plain(turn), turn                 # the model's answer, gated
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/12345678/"' in turn, turn
    assert re.search(r'<a [^>]*href="[^"]*PMC7654321[^"]*"[^>]*>[^<]*PMC7654321', turn), turn


# --------------------------------------------------------------------------- #
# 5.14  Override -> draft card -> basis -> Save writes one note; Discard writes none
# --------------------------------------------------------------------------- #
_DRAFT = {"claim": "c2 is CAF here", "scope": "cluster", "basis": "own_validation",
          "cluster": "c2", "subject_cell_type": "CAF"}
_AGREE = Citation(pmid="11111111", title="CAF in breast", authors="A", year=2020,
                  journal="J", stance="agree")
_DISSENT = Citation(pmid="22222222", title="Not CAF", authors="B", year=2021,
                    journal="J", stance="dissent")


def _override_to_draft(fake_model, draft=_DRAFT, say="override: it's a CAF"):
    fake_model.script(reply(tool_use("memory_draft", draft)),
                      "Keeping your call. Confirm the basis below to save it.")
    at = _ask(_open_chat("c2"), "c2", say)
    save = next((k for k in widget_keys(at) if k.startswith("dsave_c2_")), None)
    assert save, sorted(widget_keys(at))
    return at, save, save.replace("dsave_", "ddisc_"), save.replace("dsave_", "dbasis_")


@pytest.mark.req("CLAUDE-MEM-01", "CLAUDE-MEM-02", "SKM-S6-01")
def test_an_override_drafts_a_card_and_save_writes_one_note_with_the_chosen_basis(
        app_tree, fake_model, offline_lit):
    """The biologist's call is kept WITH the tension visible: one agree, one dissent."""
    offline_lit += [_AGREE, _DISSENT]
    at, save, _, basis = _override_to_draft(fake_model)
    card = plain(raw_markdown(at))
    assert "c2 is CAF here" in card and "this cluster (c2)" in card
    assert "1 agree" in card and "1 dissent" in card, card     # tension shown before saving
    assert _notes() == []                                     # nothing until the biologist saves

    at.button_group(key=basis).set_value("a paper")           # the biologist's choice, not the draft's
    at = run(at.button(key=save).click())

    [note] = _notes()
    assert (note.claim, note.scope, note.scope_ref.cluster, note.basis) == (
        "c2 is CAF here", "cluster", "c2", "paper")
    assert note.subject_cell_type == "CAF" and note.type == "celltype_override"
    assert [c.pmid for c in note.tension.agree] == ["11111111"]
    assert [c.pmid for c in note.tension.dissent] == ["22222222"]
    text = screen_text(at)
    assert "Saved to My notes · cluster c2 · a paper" in text
    assert "1 agree / 1 dissent kept visible" in text
    assert not any(k.startswith("dsave_") for k in widget_keys(at))   # the card closed


@pytest.mark.req("CLAUDE-MEM-02")
def test_discarding_the_draft_writes_no_note_and_says_so(app_tree, fake_model, offline_lit):
    at, _, discard, _ = _override_to_draft(fake_model)
    at = run(at.button(key=discard).click())
    assert _notes() == []
    assert "Draft discarded, nothing was saved." in screen_text(at)
    assert not any(k.startswith("dsave_") for k in widget_keys(at))
    assert _rail(at)["rail_c2"].startswith("○")


@pytest.mark.req("UI-5.3-07")
def test_a_confirmed_chat_override_projects_the_cluster_to_overridden(app_tree, fake_model,
                                                                       offline_lit):
    """Pinned as built (decision §6.7, default: keep): the confirm tap on a chat
    override note counts as the biologist's commit, so the rail marks c2 overridden
    under the new call. If you rule that only the review panel commits, flip this."""
    at, save, _, _ = _override_to_draft(fake_model)
    assert _rail(at)["rail_c2"].startswith("○")
    at = run(at.button(key=save).click())
    assert _rail(at)["rail_c2"].startswith("✎") and "CAF" in _rail(at)["rail_c2"]


# --------------------------------------------------------------------------- #
# 5.15  No model key: never promise a save that will not happen
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_key(app_tree, monkeypatch):
    from panoscope.agent import loop

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(loop, "load_dotenv", lambda *a, **k: False)   # the repo .env too
    monkeypatch.setattr(loop, "_DEFAULT_AGENT", None)
    forget_agent()


@pytest.mark.req("CLAUDE-MEM-01")
def test_with_no_model_key_an_override_is_not_promised_a_save(no_key, offline_lit):
    at = _ask(_open_chat("c2"), "c2", "override: it's a CAF")
    text = screen_text(at)
    assert _notes() == []                                     # offline, nothing is saved...
    assert "saves it to My notes" not in text, text           # ...so nothing promises it
    assert "Saved to My notes" not in text
    assert not any(k.startswith("dsave_") for k in widget_keys(at))
    assert _FALLBACK in plain(_last_agent_turn(at))
    assert "Call it yourself" in text                         # where the call CAN be made


@pytest.mark.req("CLAUDE-MEM-01", "CORP-A25")
def test_with_no_model_key_the_pathway_chat_promises_no_draft_and_no_live_literature(
        no_key, offline_lit):
    run(_open_chat("c2").button(key="rv_accept_c2").click())    # Pathways open after a call
    text = screen_text(open_app("pathways", cluster="c2"))
    assert "No model key is set" in text, text[-800:]
    assert "draft a note to save" not in text and "live literature" not in text


# --------------------------------------------------------------------------- #
# 5.16  Enforcement split: a falsified number rejects the turn; a fabricated PMID
#       drops only its sentence, with a visible stub
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.9-04", "CLAUDE-CP-01")
def test_a_falsified_number_rejects_the_whole_turn_for_the_fallback(app_tree, fake_model,
                                                                     offline_lit):
    """ERBB2's glm_coef in markers_top.csv is 21.44; 99.9 is invented. The repair
    round repeats it, so the turn falls back."""
    lie = "c1 is driven by ERBB2 (glm_coef 99.9)."
    fake_model.script(lie, lie)
    at = _ask(_open_chat("c1"), "c1", "what drives c1?")
    turn = plain(_last_agent_turn(at))
    assert "99.9" not in turn and _FALLBACK in turn, turn
    assert len(fake_model.calls) == 2                         # one answer, one repair
    assert _notes() == []


@pytest.mark.req("UI-2.9-04", gap=True)
def test_a_fabricated_pmid_drops_only_its_sentence_and_shows_a_stub(app_tree, fake_model,
                                                                     offline_lit):
    fake_model.script("c1 reads as tumour epithelium. A 2019 study agrees (PMID:99999999).")
    at = _ask(_open_chat("c1"), "c1", "what is c1?")
    turn = plain(_last_agent_turn(at))
    assert "99999999" not in turn
    assert "c1 reads as tumour epithelium." in turn, turn     # the grounded sentence stays
    assert "1 sentence removed" in turn


# --------------------------------------------------------------------------- #
# 5.27  Deterministic fallback for every agent call, labelled, writing nothing
# --------------------------------------------------------------------------- #
_FAILURES = {
    "timeout": TimeoutError("Request timed out."),
    "api_error": RuntimeError("529 overloaded_error"),
    "empty": reply(),
}


@pytest.mark.req("ARCH-11.1-01", "CLAUDE-HR-05")
@pytest.mark.parametrize("failure", sorted(_FAILURES))
def test_a_failed_model_call_shows_a_labelled_fallback_and_writes_nothing(
        app_tree, fake_model, offline_lit, failure):
    """c9 is the rubric's verify=TRUE cluster in verdicts.csv: the fallback keeps it."""
    want = _csv_verify(app_tree, "verdicts.csv")["c9"]
    assert want is True                                       # else the check below is vacuous
    fake_model.script(_FAILURES[failure])
    at = _open_chat("c9")
    before = _decisions()
    at = _ask(at, "c9", "what is c9?")
    turn = plain(_last_agent_turn(at))
    assert _FALLBACK in turn and turn.strip() != _FALLBACK, turn
    assert _VERIFY_FLAG in turn, turn                         # the flag, not just prose
    assert _notes() == [] and _decisions() == before
    assert not any(k.startswith("dsave_") for k in widget_keys(at))


@pytest.mark.req("ARCH-11.1-01", "CLAUDE-HR-05")
def test_the_pathway_chat_fallback_is_labelled_and_keeps_the_enrichment_rubric(
        app_tree, fake_model, offline_lit):
    """c2: enrichment verify TRUE, marker verify FALSE (enrichment.csv vs verdicts.csv).
    The fallback must carry the ENRICHMENT rubric's flag, not the marker one."""
    assert _csv_verify(app_tree, "enrichment.csv")["c2"] is True
    assert _csv_verify(app_tree, "verdicts.csv")["c2"] is False
    fake_model.script(TimeoutError("Request timed out."))
    run(_open_chat("c2").button(key="rv_accept_c2").click())    # Pathways open after a call
    at = open_app("pathways", cluster="c2")
    before, n_notes = _decisions(), len(_notes())
    at = _ask(at, "c2", "what does EMT mean here?", form="pwask_form", box="pwconv_ask_input_c2")
    turn = plain(_last_agent_turn(at))
    assert _FALLBACK in turn and _VERIFY_FLAG in turn, turn
    assert _decisions() == before and len(_notes()) == n_notes


# --------------------------------------------------------------------------- #
# 5.31  A gene named in chat outside the corpus: NOT_BUILT, "cannot settle"
# --------------------------------------------------------------------------- #
_GENE_NOTE = {**_DRAFT, "claim": "ACTA2 marks CAFs here", "subject_markers": ["ACTA2"],
              "note_type": "marker_reinterpretation"}


@pytest.mark.req("CORP-A16", "PLAN-P3.4-02", gap=True)
def test_a_note_on_a_gene_the_corpus_never_covered_reads_cannot_settle_not_thin(
        app_tree, fake_model, offline_lit):
    """The demo has no corpus built, so every key is NOT_BUILT: the draft's literature
    line must say it cannot settle, never that the literature is thin."""
    at, *_ = _override_to_draft(fake_model, _GENE_NOTE, "note this: ACTA2 marks CAFs here")
    card = plain(raw_markdown(at)).split("Draft to save", 1)[1]
    assert "thin" not in card, card
    assert "cannot settle" in card


@pytest.mark.req("CORP-A16", "ARCH-8.1-02", gap=True)
def test_a_note_on_an_uncovered_gene_offers_a_dated_top_up(app_tree, fake_model, offline_lit):
    at, *_ = _override_to_draft(fake_model, _GENE_NOTE, "note this: ACTA2 marks CAFs here")
    labels = [str(b.label).lower() for b in at.button]
    assert any("top up" in lb or "top-up" in lb for lb in labels), labels
