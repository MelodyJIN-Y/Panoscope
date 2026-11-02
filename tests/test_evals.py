"""Offline tests of the eval SCORERS, against the committed raw results.

The split this file enforces is the one that makes ``evals/panel_absence`` worth
anything: a HARNESS costs money and is run by hand; a SCORER is pure and is run by
the suite. Nothing here opens a socket or reads an API key, and nothing here invokes
a harness. If an assertion below fails, either a scorer changed or a committed
results file did -- and in both cases a README number is now wrong.

The figures asserted here are the figures the two READMEs quote. That is the point:
the READMEs are checkable rather than quotable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"


def _load(name: str, path: Path):
    """Import a scorer by path; ``evals/`` is deliberately not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def seeded():
    return _load("seeded_score", EVALS / "seeded_errors" / "score.py")


@pytest.fixture(scope="module")
def refusal():
    return _load("refusal_score", EVALS / "refusal" / "score.py")


def _doc(*parts: str) -> dict:
    return json.loads((EVALS.joinpath(*parts)).read_text())


# --------------------------------------------------------------------------- #
# seeded_errors -- the deterministic arm, which is what the README reports
# --------------------------------------------------------------------------- #
def test_seeded_deterministic_reproduces_readme(seeded):
    s = seeded.summarise(_doc("seeded_errors", "results.json"))
    a = s["by_arm"]["deterministic"]
    assert s["arms"] == ["deterministic"]
    assert (a["recall_k"], a["recall_n"]) == (55, 55)
    assert (a["boundary_caught_k"], a["boundary_n"]) == (0, 27)
    assert (a["false_positive_k"], a["false_positive_n"]) == (0, 75)


def test_seeded_misses_are_exactly_the_three_preregistered_classes(seeded):
    """The honest half. If this set shrinks, the README's boundary claim moved."""
    s = seeded.summarise(_doc("seeded_errors", "results.json"))
    missed = {m.split(":")[0] for m in s["by_arm"]["deterministic"]["misses"]}
    assert missed == {"number_without_stat_keyword",
                      "off_panel_absence_as_evidence",
                      "real_pmid_wrong_claim"}


def test_seeded_every_class_has_a_clean_control(seeded):
    """A detection rate with no clean control is a recall number with no gate."""
    doc = _doc("seeded_errors", "results.json")
    poisoned = {r["cls"] for r in doc["rows"] if not r["clean"]}
    clean = {r["cls"] for r in doc["rows"] if r["clean"]}
    assert not poisoned - clean, poisoned - clean


def test_seeded_model_smoke_scores(seeded):
    """One Haiku replicate on one cluster. A smoke, not a result -- but committed,
    so the README's description of it has to keep matching the bytes."""
    s = seeded.summarise(_doc("seeded_errors", "results_model.json"))
    a = s["by_arm"]["model"]
    assert (a["recall_k"], a["recall_n"]) == (4, 4)
    assert (a["boundary_caught_k"], a["boundary_n"]) == (3, 3)
    assert (a["false_positive_k"], a["false_positive_n"]) == (0, 7)


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #
def test_refusal_gated_reproduces_readme(refusal):
    s = refusal.summarise(_doc("refusal", "results.json"))
    a = s["by_arm"]["gated"]
    assert (a["answered_k"], a["answered_n"]) == (8, 23)
    assert (a["acc_given_answered_k"], a["acc_given_answered_n"]) == (8, 8)
    assert (a["acc_refusal_wrong_k"], a["acc_refusal_wrong_n"]) == (8, 8)
    assert (a["false_refusal_k"], a["false_refusal_n"]) == (0, 8)
    assert (a["over_answer_k"], a["over_answer_n"]) == (0, 15)


def test_refusal_model_smoke_scores(refusal):
    s = refusal.summarise(_doc("refusal", "results_model.json"))
    a = s["by_arm"]["model"]
    assert (a["answered_k"], a["answered_n"]) == (5, 15)
    assert (a["false_refusal_k"], a["false_refusal_n"]) == (0, 5)
    assert (a["over_answer_k"], a["over_answer_n"]) == (0, 10)


# --------------------------------------------------------------------------- #
# The arithmetic itself, on rows written here -- the committed fixtures are all
# zeros on the two honesty numbers, so they cannot show that a non-zero one is
# counted correctly. These rows can.
# --------------------------------------------------------------------------- #
def _row(cls, expect, answered, answer, truth, arm="gated"):
    return {"id": f"{cls}:{answer}", "cls": cls, "route": "marker_stat",
            "question": "", "expect": expect, "truth": truth, "payload": None,
            "arm": arm, "model": "", "ok": True, "error": "",
            "answered": answered, "answer": answer, "reason": "", "raw": ""}


def test_false_refusal_and_over_answer_are_counted(refusal):
    doc = {"rows": [
        _row("a", "answer", True, 1.0, 1.0),      # answered, correct
        _row("a", "answer", True, 9.0, 1.0),      # answered, WRONG
        _row("a", "answer", False, None, 1.0),    # FALSE REFUSAL
        _row("b", "refuse", True, 3.0, None),     # OVER-ANSWER
        _row("b", "refuse", False, None, None),   # correct refusal
    ]}
    a = refusal.summarise(doc)["by_arm"]["gated"]
    assert (a["false_refusal_k"], a["false_refusal_n"]) == (1, 3)
    assert (a["over_answer_k"], a["over_answer_n"]) == (1, 2)
    assert (a["answered_k"], a["answered_n"]) == (3, 5)
    # accuracy | answered counts only answerable turns that were answered: 1 of 2.
    assert (a["acc_given_answered_k"], a["acc_given_answered_n"]) == (1, 2)
    # with refusal counted as wrong, the denominator is every answerable turn.
    assert (a["acc_refusal_wrong_k"], a["acc_refusal_wrong_n"]) == (1, 3)


def test_a_refusal_is_never_scored_correct(refusal):
    """The whole point: folding a refusal into accuracy is the failure mode."""
    assert not refusal.correct(_row("a", "answer", False, None, 1.0))


def test_string_truth_matches_by_substring(refusal):
    assert refusal.correct(_row("a", "answer", True, "c1: Tumor, panel-scoped",
                                "Tumor"))
    assert not refusal.correct(_row("a", "answer", True, "c1: Stromal", "Tumor"))


def test_seeded_scorer_flags_a_regression_in_an_expected_class(seeded):
    """A miss in a class pre-registered as catchable must stay visible."""
    doc = {"rows": [
        {"id": "wrong_number:x:poisoned", "cls": "wrong_number", "kind": "prose",
         "clean": False, "arm": "deterministic", "ok": True, "detected": False,
         "expected_caught": True, "violations": []},
        {"id": "wrong_number:x:clean", "cls": "wrong_number", "kind": "prose",
         "clean": True, "arm": "deterministic", "ok": True, "detected": True,
         "expected_caught": True, "violations": ["spurious"]},
    ]}
    a = seeded.summarise(doc)["by_arm"]["deterministic"]
    assert a["misses"] == ["wrong_number:x:poisoned"]
    assert (a["recall_k"], a["recall_n"]) == (0, 1)
    assert (a["false_positive_k"], a["false_positive_n"]) == (1, 1)


def test_scorers_import_nothing_that_needs_the_network(seeded, refusal):
    """A scorer that imported ``anthropic`` would make the suite need a key."""
    import sys
    assert "anthropic" not in sys.modules
