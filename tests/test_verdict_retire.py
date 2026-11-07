"""One band engine behind every surface, not two.

``pipeline/stages/verdicts.py`` was cut over to ``core/verdict.py``. Five call
sites outside it were not, and they are the ones a biologist actually reads: the
band injected into the chat system prompt on every turn, the ``verify`` flag on
every chat answer, the second opinion, the cited cell-type and per-gene notes,
and the Summary table when a study has no built tree.

``agent/verdict.py``, the second engine, has been deleted, so a surface can no longer
route around ``core/``. What stays pinned is the VALUE each surface produces, checked
against ``pipeline.v1_verdicts`` -- several of them swallow exceptions, so each
assertion checks the value, never merely that it did not raise. On
``synthetic_multisample_fixture`` c3 the old engine stated a different reason for the
same Low band; the fallback must state ``core/``'s.
"""

from __future__ import annotations

import dataclasses

import pytest

from panoscope.agent import config as cfg

DIVERGENT_STUDY = "synthetic_multisample_fixture"
DIVERGENT_UNIT = "c3"


@pytest.fixture
def core_verdicts():
    """The bands as ``core/`` produces them -- the table's own numbers."""
    from panoscope.pipeline import v1_verdicts

    return v1_verdicts.verdicts()


# --------------------------------------------------------------------------- #
# agent/loop.py -- the band in the system prompt, every chat turn
# --------------------------------------------------------------------------- #
def test_prompt_bands_are_the_tables_bands(core_verdicts):
    """All nine units, band and verify, as the chat is told them.

    ``_global_interpretation_lines`` catches every exception, so a revert to the
    v1 engine would not raise -- it would print the cluster key with no band.
    Asserting the band text is what makes that visible.
    """
    from panoscope.agent import loop

    text = loop._global_interpretation_lines()
    lines = text.splitlines()
    assert len(lines) == len(cfg.CLUSTER_ORDER) == 9

    for unit, line in zip(cfg.CLUSTER_ORDER, lines):
        v = core_verdicts[unit]
        assert line.startswith(f"  {unit}: {v.cell_type} — {v.confidence} confidence")
        assert ("[VERIFY — re-check]" in line) is v.verify
        for marker in v.key_markers[:3]:
            assert marker in line
    assert "NO VERDICT" not in text


def test_prompt_never_shows_a_cluster_without_saying_it_has_no_band(monkeypatch):
    """A build failure must not read as a confident call.

    Every other line in the block ends "— <band> confidence". A line carrying a
    cell type and nothing else sits in that list as an invitation to infer one,
    which is the fabrication the confident floor forbids.
    """
    from panoscope.agent import loop
    from panoscope.pipeline import v1_verdicts

    def _broken():
        raise RuntimeError("no method output")

    monkeypatch.setattr(v1_verdicts, "verdicts", _broken)
    text = loop._global_interpretation_lines()

    assert len(text.splitlines()) == 9
    for line in text.splitlines():
        assert "NO VERDICT" in line
        assert "do not state a confidence" in line
        assert "confidence (drivers:" not in line
        assert " confidence" not in line.split("—", 1)[1].split("(the", 1)[0]


def test_chat_verify_flag_is_the_tables_verify_flag(core_verdicts):
    """The deterministic rubric reaches every chat answer, on all nine units."""
    from panoscope.agent import loop

    produced = {u: loop._cluster_verify_default(u) for u in cfg.CLUSTER_ORDER}
    assert produced == {u: core_verdicts[u].verify for u in cfg.CLUSTER_ORDER}
    # The demo is not all-True; a rubric that answered True for everything would
    # pass a weaker version of this test.
    assert set(produced.values()) == {True, False}


def test_verify_fails_closed_when_the_rubric_cannot_be_computed(monkeypatch):
    """No rubric is not "no re-check needed"; it is "a human should look"."""
    from panoscope.agent import loop
    from panoscope.pipeline import v1_verdicts

    def _broken(_unit):
        raise RuntimeError("no method output")

    monkeypatch.setattr(v1_verdicts, "verdict_for", _broken)
    assert loop._cluster_verify_default("c1") is True


# --------------------------------------------------------------------------- #
# agent/skeptic.py + loop's deterministic skeptic response
# --------------------------------------------------------------------------- #
def test_second_opinion_argues_against_the_tables_call(core_verdicts):
    """All nine. The call challenged, and the band it is held at, are the table's."""
    from panoscope.agent import skeptic

    for unit in cfg.CLUSTER_ORDER:
        v = core_verdicts[unit]
        report = skeptic.second_opinion(unit)
        assert report.call == v.cell_type
        assert report.cluster == unit
        if report.survives:
            assert f"Holds at {v.confidence}." in report.verdict_line
        else:
            assert v.confidence in report.verdict_line or report.challenges


def test_deterministic_skeptic_response_quotes_the_tables_numbers(core_verdicts):
    """``loop._deterministic_skeptic_response`` swallows a verdict failure into an
    EMPTY grounding sidecar, so a revert would quietly produce a second opinion
    citing no numbers at all. Assert the numbers, on every unit that has any."""
    from panoscope.agent import loop

    agent = loop.PanoscopeAgent()
    for unit in cfg.CLUSTER_ORDER:
        expected = [
            e.gene for e in core_verdicts[unit].evidence if e.role == "supports"
        ][:5]
        resp = agent._deterministic_skeptic_response(unit)
        assert list(resp.grounding.markers) == expected, unit
        for gene, stat, value in resp.grounding.numbers:
            match = next(e for e in core_verdicts[unit].evidence if e.gene == gene)
            assert value == pytest.approx(getattr(match, stat))
    # Not vacuous: the demo has units with supporting markers.
    assert any(agent._deterministic_skeptic_response(u).grounding.markers
               for u in cfg.CLUSTER_ORDER)


# --------------------------------------------------------------------------- #
# pipeline/stages/notes.py -- the cited notes
# --------------------------------------------------------------------------- #
def test_celltype_notes_are_written_from_the_tables_call(core_verdicts,
                                                         tmp_tree, monkeypatch):
    """The cell type a note describes, and the markers it is asked about, come
    from ``core/``. The stage prints and continues on an exception, so a revert
    would write an EMPTY notes file rather than fail -- assert the contents."""
    from panoscope.agent import loop as agent_loop
    from panoscope.pipeline.stages import notes as notes_stage

    asked: list[str] = []

    class _Resp:
        text = "A grounded clause about the cell type."
        citations = ()
        verify = False

    def _chat(query, cluster=None, **_kw):
        asked.append(query)
        return _Resp()

    monkeypatch.setattr(agent_loop, "chat", _chat)
    written = notes_stage.run_celltype_notes(cfg.DATASET_ID, root=tmp_tree)

    assert sorted(written) == sorted(cfg.CLUSTER_ORDER)
    for unit in cfg.CLUSTER_ORDER:
        assert written[unit]["cell_type"] == core_verdicts[unit].cell_type
    assert len(asked) == 9
    for unit, query in zip(cfg.CLUSTER_ORDER, asked):
        for marker in core_verdicts[unit].key_markers[:3]:
            assert marker in query


# --------------------------------------------------------------------------- #
# ui/data_access.py -- the Summary table and its export
# --------------------------------------------------------------------------- #
def test_summary_table_falls_back_to_the_same_engine_as_the_table(
    core_verdicts, monkeypatch
):
    """The unbuilt-study fallback. All nine units, object for object.

    ``store.load_verdict`` is forced to miss: the demo HAS a built tree, so
    without this the persisted arm answers every call and the fallback -- the arm
    that was on the v1 engine -- is never reached.
    """
    from panoscope.pipeline import store
    from panoscope.ui import data_access as da

    monkeypatch.setattr(store, "load_verdict", lambda *_a, **_k: None)
    _clear(da.verdict_for)
    for unit in cfg.CLUSTER_ORDER:
        assert da.verdict_for(unit) == core_verdicts[unit]
    _clear(da.verdict_for)


def test_the_persisted_arm_and_the_fallback_arm_agree(core_verdicts):
    """Both arms of ``verdict_for`` are now one engine.

    The persisted file is written by ``pipeline/stages/verdicts.py`` through
    ``core/``; the fallback builds the same projection. If they ever disagree,
    the band a biologist sees depends on whether the tree happens to be built.
    """
    from panoscope.pipeline import store

    for unit in cfg.CLUSTER_ORDER:
        persisted = store.load_verdict(unit)
        if persisted is None:
            pytest.skip("the demo tree is not built in this working copy")
        assert persisted == core_verdicts[unit], unit


def test_the_fallback_states_the_cores_reason_not_the_v1_engines(monkeypatch):
    """The one study on disk where the two engines tell different stories.

    ``synthetic_multisample_fixture`` c3: both engines band it Low and they disagree
    about WHY, and the sentence is printed in the Summary table's notes column and
    exported in the CSV. This is the assertion that fails if the fallback goes back
    to ``agent/verdict.py``.

    **Why ``load_verdict`` is forced to miss.** This used to assert that the fixture
    had NO built tree, which made the one gate pinning the verdict retirement -- on
    the one study where the two band engines differ at all -- depend on nobody ever
    running the documented ``panoscope build --study synthetic_multisample_fixture``.
    The fixture exists to be built against, so the gate punished the thing it is
    for. Forcing the miss is the sibling test's move one function up, for the same
    reason: the fallback arm is reachable whatever is on disk, and the arm that was
    left on the v1 engine is the one being pinned. The persisted arm is pinned
    below, so this module now covers whichever arm a given working copy takes.
    """
    from panoscope.data.study import Study

    if not Study.open(DIVERGENT_STUDY).exists:
        pytest.skip(f"study {DIVERGENT_STUDY!r} is not on disk")

    from panoscope.pipeline import store
    from panoscope.ui import data_access as da

    monkeypatch.setattr(store, "load_verdict", lambda *_a, **_k: None)
    with cfg.use_study(DIVERGENT_STUDY):
        _clear(da.verdict_for)
        shown = da.verdict_for(DIVERGENT_UNIT)
        _clear(da.verdict_for)

    assert shown.band_basis == "canonical driver tested-absent"
    assert "leading canonical marker was measured and did not mark this unit" \
        in shown.notes
    assert "no canonical marker supports the call" not in shown.notes


def test_the_fixtures_persisted_arm_states_the_cores_reason_too():
    """And if this working copy HAS built the fixture, the arm it takes is pinned too.

    ``panoscope build --study synthetic_multisample_fixture`` is a documented
    command and the fixture is there to be built against, so whether
    ``interp/clusters/`` exists is a property of the checkout and not of the code.
    Skipping on absence is honest here in a way it was not above: there is nothing
    to read, rather than a code path going unchecked. The build writes through
    ``core/``, so a persisted verdict stating the v1 engine's reason would mean the
    stage had been reverted.
    """
    from panoscope.data.study import Study
    from panoscope.pipeline import store

    if not Study.open(DIVERGENT_STUDY).exists:
        pytest.skip(f"study {DIVERGENT_STUDY!r} is not on disk")

    with cfg.use_study(DIVERGENT_STUDY):
        persisted = store.load_verdict(DIVERGENT_UNIT)
        if persisted is None:
            pytest.skip(f"{DIVERGENT_STUDY} is not built in this working copy")
        from panoscope.pipeline import v1_verdicts

        assert persisted == v1_verdicts.verdict_for(DIVERGENT_UNIT)


def test_exported_csv_carries_the_11_frozen_columns(core_verdicts):
    """The download the biologist hands a collaborator.

    Byte-identity between ``core/export``'s ASSIGNMENT profile and the exporter
    it replaces is proven by ``tests/test_core_export.py``, so this cannot and
    does not try to tell the two apart. What it pins is the committed contract:
    11 columns in order, one row per unit, cells that match the composed verdict.
    """
    import csv
    import io

    from panoscope.core.export import PROFILES
    from panoscope.types.outputs import TaskKind
    from panoscope.ui import data_access as da

    CSV_COLUMNS = PROFILES[TaskKind.ASSIGNMENT].names

    rows = list(csv.reader(io.StringIO(da.verdict_csv())))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 1 + len(cfg.CLUSTER_ORDER)
    composed = {v.cluster: v for v in da.composed_verdicts()}
    for row in rows[1:]:
        cells = dict(zip(CSV_COLUMNS, row))
        v = composed[cells["cluster"]]
        assert cells["confidence"] == v.confidence
        assert cells["verify"] == ("TRUE" if v.verify else "FALSE")
        assert cells["key_markers"] == ";".join(v.key_markers)


def test_the_export_reads_the_profile_registry_not_a_private_copy(monkeypatch):
    """A column added to the registered profile reaches the UI's download.

    The byte-identity test above passes whichever exporter is wired in, so it is
    not evidence that the cutover happened. This is: it adds a column to
    ``core/export``'s ASSIGNMENT profile and fails unless the download carries
    it. ``IMPLEMENTATION.md`` §10 adds and removes columns here (``per-unit
    counts``, dropping ``cluster_confidence``); a second hand-written copy of the
    column list in the UI would silently keep shipping the old table.
    """
    import csv
    import io

    from panoscope.core import export
    from panoscope.types.outputs import TaskKind
    from panoscope.ui import data_access as da

    profile = export.PROFILES[TaskKind.ASSIGNMENT]
    extra = export.Column("unit_kind", lambda v: v.cluster + ":probe")
    monkeypatch.setitem(
        export.PROFILES, TaskKind.ASSIGNMENT,
        dataclasses.replace(profile, columns=profile.columns + (extra,)),
    )

    rows = list(csv.reader(io.StringIO(da.verdict_csv())))
    assert rows[0][-1] == "unit_kind"
    assert rows[1][-1] == rows[1][0] + ":probe"


def _clear(fn) -> None:
    """Drop a Streamlit ``cache_data`` entry. ``tests/conftest.py`` only finds
    ``cache_clear``, and ``st.cache_data`` spells it ``clear``."""
    clear = getattr(fn, "clear", None) or getattr(fn, "cache_clear", None)
    if clear is not None:
        clear()
