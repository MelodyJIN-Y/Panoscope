"""Portable user memory: distill, round-trip, and prompt-context injection.

Writes only to a pytest tmp file (``_PATH`` monkeypatched), never the real
``context/user_memory.jsonl``. Open-ceiling context only: nothing here can carry a
number that would override jazzPanda.
"""
from __future__ import annotations

from agent import user_memory


def test_distill_is_deterministic_and_tissue_tagged():
    line = user_memory.distill(
        claim="c2 is a CAF, matrix-remodelling/POSTN+ subtype.",
        cell_type="Stromal",
        markers=["LUM", "POSTN", "MMP2"],
        dataset_label="human breast · Xenium",
    )
    assert line == "Stromal in human breast · Xenium, driven by LUM, POSTN, MMP2"
    # underscores in a cell type read as spaces; no markers -> headline only
    assert user_memory.distill(
        claim="", cell_type="Mast_Cells", markers=[], dataset_label="human breast"
    ) == "Mast Cells in human breast"


def test_as_prompt_context_empty_when_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(user_memory, "_PATH", tmp_path / "user_memory.jsonl")
    assert user_memory.load() == []
    assert user_memory.as_prompt_context() == ""


def test_record_then_load_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(user_memory, "_PATH", tmp_path / "user_memory.jsonl")
    entry = {
        "id": "um_test1", "date": "2026-07-12", "from": "xenium_hbreast_sample1",
        "cluster": "c2", "cell_type": "Stromal", "markers": ["LUM", "POSTN"],
        "basis": "paper", "status": "firm", "claim": "c2 is a CAF.",
        "summary": "Stromal in human breast · Xenium, driven by LUM, POSTN",
    }
    user_memory.record(entry)
    loaded = user_memory.load()
    assert loaded == [entry]


def test_prompt_context_labels_and_attributes(monkeypatch, tmp_path):
    monkeypatch.setattr(user_memory, "_PATH", tmp_path / "user_memory.jsonl")
    user_memory.record({
        "summary": "Stromal in human breast · Xenium, driven by LUM, POSTN",
        "basis": "paper", "status": "firm", "from": "xenium_hbreast_sample1",
    })
    ctx = user_memory.as_prompt_context()
    assert "PRIOR LAB KNOWLEDGE" in ctx
    assert "[firm, paper]" in ctx
    assert "from xenium_hbreast_sample1" in ctx
    # it must announce its open-ceiling boundary
    assert "never" in ctx.lower() and "override" in ctx.lower()


def test_malformed_lines_are_skipped(monkeypatch, tmp_path):
    path = tmp_path / "user_memory.jsonl"
    path.write_text('{"summary": "ok"}\nnot json\n\n', encoding="utf-8")
    monkeypatch.setattr(user_memory, "_PATH", path)
    loaded = user_memory.load()
    assert loaded == [{"summary": "ok"}]
