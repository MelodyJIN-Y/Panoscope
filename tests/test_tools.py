"""Tests for agent/tools.py — the 7 agent tools and the uniform envelope.

Grounding-floor coverage:

* ``panel_lookup`` reports on/off panel truthfully (ERBB2 on, COL1A1 off) — the
  panel-absence primitive surfaced as a tool.
* ``marker_lookup`` returns LUM's REAL jazzPanda glm_coef for c2 (numbers come
  only from agent.data, never fabricated).
* ``get_spatial('density','LUM')`` reports the precomputed frame as available and
  never recomputes.
* A note written via ``memory.create_note`` round-trips through ``memory_read``
  in a tmp context dir, with an injected (stubbed) literature_search so no network
  is used and the note is born with real-citation tension.
* Every tool goes through ``dispatch`` and returns ``{ok, data, sources, error}``.
* ``literature_search`` returns ``ok=True`` with >=1 real PMID when the connector
  is up, OR ``ok=False`` gracefully when offline (never a fabricated PMID).

Memory tests point tools at a tmp dir via ``set_memory_base_dir`` and never touch
the real ``context/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from panoscope.agent import data as _data
from panoscope.agent import memory
from panoscope.agent import tools
from panoscope.agent.types import Citation

# --------------------------------------------------------------------------- #
# LUM's real jazzPanda numbers for c2 (read straight from agent.data at collect
# time so the assertion never hardcodes a value that could drift from source).
# --------------------------------------------------------------------------- #
_LUM_ROW = _data.get_marker("LUM")
LUM_GLM_COEF = float(_LUM_ROW["glm_coef"])
LUM_TOP_CLUSTER = str(_LUM_ROW["top_cluster"])


# --------------------------------------------------------------------------- #
# Fixtures: isolate memory writes + reset injected literature search per test
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_memory(tmp_path: Path):
    """Point tools' memory at a tmp context dir; restore afterwards."""
    prev = tools._MEMORY_BASE_DIR
    tools.set_memory_base_dir(str(tmp_path / "context"))
    try:
        yield tmp_path / "context"
    finally:
        tools.set_memory_base_dir(prev)


@pytest.fixture(autouse=True)
def clear_injected_search():
    """Ensure no leaked injected stub between tests (default = live path)."""
    prev = tools._LITERATURE_SEARCH_FN
    tools.set_literature_search(None)
    try:
        yield
    finally:
        tools.set_literature_search(prev)


def _stub_search(query: str):
    """Deterministic literature stub: one agreeing + one dissenting real citation."""
    return [
        Citation(
            pmid="40000001",
            title="LUM lumican is a fibroblast/stromal marker",
            authors="Doe J, Roe R",
            year=2022,
            journal="Nat Spatial",
            stance="agree",
            is_real=True,
        ),
        Citation(
            pmid="40000002",
            title="LUM also reported in tumor epithelium",
            authors="Smith A",
            year=2020,
            journal="Cancer Cell",
            stance="dissent",
            is_real=True,
        ),
    ]


# --------------------------------------------------------------------------- #
# Envelope shape
# --------------------------------------------------------------------------- #
def _assert_envelope(env: dict) -> None:
    assert set(env.keys()) == {"ok", "data", "sources", "error"}
    assert isinstance(env["ok"], bool)
    assert isinstance(env["sources"], list)
    for s in env["sources"]:
        assert set(s.keys()) == {"kind", "ref", "value", "detail"}


def test_dispatch_unknown_tool_fails_cleanly():
    env = tools.dispatch("no_such_tool", {})
    _assert_envelope(env)
    assert env["ok"] is False
    assert "unknown tool" in env["error"]


def test_tool_schemas_match_dispatch():
    names = {s["name"] for s in tools.TOOL_SCHEMAS}
    # Every model-facing schema has a dispatch entry. No model-facing tool
    # persists a note the biologist has not confirmed — the model drafts
    # (memory_draft) and the UI saves.
    assert names <= set(tools._DISPATCH)
    assert "memory_write" not in names
    assert "memory_draft" in names
    assert "discriminate_call" in names
    assert len(tools.TOOL_SCHEMAS) == 8
    for schema in tools.TOOL_SCHEMAS:
        assert "name" in schema and "description" in schema and "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"


# --------------------------------------------------------------------------- #
# 1. panel_lookup
# --------------------------------------------------------------------------- #
def test_panel_lookup_erbb2_on_panel():
    env = tools.dispatch("panel_lookup", {"gene": "ERBB2"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["on_panel"] is True
    assert env["data"]["gene"] == "ERBB2"
    # annotation comes straight from the panel file (non-null on-panel).
    assert env["data"]["annotation"] is not None
    assert env["sources"][0]["kind"] == "panel"


def test_panel_lookup_col1a1_off_panel():
    env = tools.dispatch("panel_lookup", {"gene": "COL1A1"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["on_panel"] is False
    assert env["data"]["annotation"] is None
    # absence must be flagged as "never measured", not "not expressed".
    assert "never measured" in env["sources"][0]["detail"]


def test_panel_lookup_empty_gene_fails():
    env = tools.dispatch("panel_lookup", {"gene": ""})
    assert env["ok"] is False


# --------------------------------------------------------------------------- #
# 2. marker_lookup
# --------------------------------------------------------------------------- #
def test_marker_lookup_cluster_c2_returns_lum_with_real_glm_coef():
    env = tools.dispatch("marker_lookup", {"cluster": "c2"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["cluster"] == "c2"
    markers = env["data"]["markers"]
    assert len(markers) >= 1
    lum = next((m for m in markers if m["gene"].upper() == "LUM"), None)
    assert lum is not None, "LUM should be a c2 marker"
    # the number must be the REAL jazzPanda value from agent.data (no fabrication).
    assert lum["glm_coef"] == pytest.approx(LUM_GLM_COEF, abs=1e-6)
    assert lum["top_cluster"] == LUM_TOP_CLUSTER
    # markers are glm_coef descending -> LUM (the strongest c2 marker) is first.
    assert markers[0]["gene"].upper() == "LUM"


def test_marker_lookup_gene_only():
    env = tools.dispatch("marker_lookup", {"gene": "LUM"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["found"] is True
    assert env["data"]["marker"]["glm_coef"] == pytest.approx(LUM_GLM_COEF, abs=1e-6)


def test_marker_lookup_unknown_cluster_fails():
    env = tools.dispatch("marker_lookup", {"cluster": "c99"})
    assert env["ok"] is False
    assert "unknown cluster" in env["error"]


def test_marker_lookup_no_args_fails():
    env = tools.dispatch("marker_lookup", {})
    assert env["ok"] is False


def test_marker_lookup_never_fabricates_missing_gene():
    env = tools.dispatch("marker_lookup", {"gene": "NOTAREALGENE123"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["found"] is False
    assert env["data"]["marker"] is None


# --------------------------------------------------------------------------- #
# 3. get_spatial
# --------------------------------------------------------------------------- #
def test_get_spatial_density_lum_available():
    env = tools.dispatch("get_spatial", {"view": "density", "marker": "LUM"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["available"] is True
    assert env["data"]["marker"] == "LUM"
    assert env["data"]["bin_um"] == 50
    assert env["data"]["n_bins"] > 0


def test_get_spatial_density_different_bin_reads_different_frame():
    e50 = tools.dispatch("get_spatial", {"view": "density", "marker": "LUM", "bin_um": 50})
    e25 = tools.dispatch("get_spatial", {"view": "density", "marker": "LUM", "bin_um": 25})
    assert e50["ok"] and e25["ok"]
    assert e50["data"]["bin_um"] == 50 and e25["data"]["bin_um"] == 25
    # finer bins -> a different precomputed frame (more, smaller bins), never recomputed.
    assert e25["data"]["n_bins"] != e50["data"]["n_bins"]


def test_get_spatial_density_missing_marker_not_available():
    env = tools.dispatch("get_spatial", {"view": "density", "marker": "NOTAGENE"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["available"] is False


def test_get_spatial_density_requires_marker():
    env = tools.dispatch("get_spatial", {"view": "density"})
    assert env["ok"] is False


def test_get_spatial_unknown_view_fails():
    env = tools.dispatch("get_spatial", {"view": "heatmap"})
    assert env["ok"] is False


def test_get_spatial_cell_map():
    env = tools.dispatch("get_spatial", {"view": "cell_map", "marker": "LUM"})
    _assert_envelope(env)
    assert env["ok"] is True
    # cell_map availability depends on the tidy file; either way the envelope is valid.
    assert "available" in env["data"]


# --------------------------------------------------------------------------- #
# 6. create_note -> memory_read round-trip (c2-scoped)
# --------------------------------------------------------------------------- #
def _write_note(**kw):
    """Persist a note the way the UI save path does (scope enforced at birth)."""
    return memory.create_note(
        literature_search=tools._literature_search_fn(),
        base_dir=tools._MEMORY_BASE_DIR,
        **kw,
    )


def test_memory_write_then_read_roundtrip(isolated_memory):
    tools.set_literature_search(_stub_search)

    note = _write_note(
        claim="In our breast TME, LUM marks CAFs in this cluster",
        scope="cluster",
        basis="own_validation",
        cluster="c2",
        subject_cell_type="Stromal",
        subject_markers=["LUM"],
    )
    # reconciled against the stub -> tension carries the real agree/dissent PMIDs.
    assert note.tension.thin is False
    assert "40000001" in [c.pmid for c in note.tension.agree]
    assert "40000002" in [c.pmid for c in note.tension.dissent]

    # read back for c2 -> the note fires, cited as kind="mem".
    read_env = tools.dispatch("memory_read", {"cluster": "c2"})
    _assert_envelope(read_env)
    assert read_env["ok"] is True
    ids = [n["id"] for n in read_env["data"]["notes"]]
    assert note.id in ids
    assert any(
        s["kind"] == "mem" and s["ref"] == note.id for s in read_env["sources"]
    )


def test_memory_read_scope_enforced_other_cluster(isolated_memory):
    tools.set_literature_search(_stub_search)
    _write_note(
        claim="c2-only stromal convention",
        scope="cluster",
        basis="convention",
        cluster="c2",
    )
    # a cluster-scoped note must NOT fire for a different cluster.
    other = tools.dispatch("memory_read", {"cluster": "c5"})
    assert other["ok"] is True
    assert other["data"]["n_notes"] == 0


def test_memory_write_cluster_scope_requires_cluster(isolated_memory):
    tools.set_literature_search(_stub_search)
    with pytest.raises(ValueError, match="(?i)cluster"):
        _write_note(claim="no cluster given", scope="cluster", basis="convention")


def test_memory_write_invalid_scope_fails(isolated_memory):
    with pytest.raises(ValueError):
        _write_note(claim="x", scope="galaxy", basis="convention")


# --------------------------------------------------------------------------- #
# 4 + 5. literature tools — real PMIDs live, or graceful ok=False offline
# --------------------------------------------------------------------------- #
def test_literature_search_stub_returns_real_pmids():
    """With an injected stub, the tool returns real PMIDs and ok=True (no network)."""
    tools.set_literature_search(_stub_search)
    env = tools.dispatch("literature_search", {"query": "LUM fibroblast marker"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["n_results"] >= 1
    pmids = [r["pmid"] for r in env["data"]["results"]]
    assert "40000001" in pmids
    assert env["sources"][0]["kind"] == "lit"


def test_literature_search_empty_query_fails():
    env = tools.dispatch("literature_search", {"query": ""})
    assert env["ok"] is False


def test_literature_search_live_or_graceful_offline():
    """One real MCP lookup: ok=True with >=1 PMID if up, else ok=False (never fake).

    This exercises the live path (no injected stub). It must NEVER fabricate a
    PMID: either the connector resolves >=1 real PMID, or the tool reports the
    connector is unavailable via ok=False.
    """
    tools.set_literature_search(None)  # force the live path
    env = tools.dispatch(
        "literature_search",
        {"query": "lumican LUM fibroblast marker", "max_results": 3},
    )
    _assert_envelope(env)
    if env["ok"]:
        # connector up: every returned PMID is a real, numeric id from the server.
        assert env["data"]["n_results"] >= 1
        for r in env["data"]["results"]:
            assert r["pmid"].isdigit()
    else:
        # connector down/offline: graceful failure, no fabricated citation.
        assert env["error"]
        assert env["data"] in (None, {}, [])


def test_literature_fetch_bad_pmids_fails():
    env = tools.dispatch("literature_fetch", {"pmids": ["notanid", "xx"]})
    assert env["ok"] is False


def test_memory_draft_reconciles_but_persists_nothing(isolated_memory):
    """memory_draft proposes a reconciled note WITHOUT writing it (UI-gated save)."""
    tools.set_literature_search(_stub_search)

    env = tools.dispatch(
        "memory_draft",
        {
            "claim": "In our breast TME, LUM marks CAFs in this cluster",
            "scope": "cluster",
            "basis": "own_validation",
            "cluster": "c2",
            "subject_markers": ["LUM"],
        },
    )
    _assert_envelope(env)
    assert env["ok"] is True
    # It reconciled (real agree/dissent PMIDs on the payload) ...
    tension = env["data"]["tension"]
    assert tension["agree"] or tension["dissent"]
    assert "id" not in env["data"]  # a draft has no persisted id
    # ... but nothing was written: an in-scope read finds no notes.
    read_env = tools.dispatch("memory_read", {"cluster": "c2"})
    assert read_env["data"]["n_notes"] == 0


def test_memory_write_not_exposed_to_model():
    """The model cannot call memory_write (only memory_draft); persistence is UI-gated."""
    schema_names = {s["name"] for s in tools.TOOL_SCHEMAS}
    assert "memory_write" not in schema_names
    assert "memory_draft" in schema_names


# --------------------------------------------------------------------------- #
# The envelope must tell a FAILED lookup apart from a genuinely empty one.
# "ok=True, n_results=0" reads downstream as "the literature is thin" — a claim
# about the world. A broken connector has not earned that claim.
# --------------------------------------------------------------------------- #
class _StubClient:
    """A started connector whose call returns nothing, with or without an error."""

    def __init__(self, last_error=None):
        self.available = True
        self.last_error = last_error

    def search_articles(self, query, max_results=5):
        return []

    def fetch_articles(self, pmids):
        return []


def _use_stub_client(monkeypatch, client):
    from panoscope.agent import mcp_client

    monkeypatch.setattr(mcp_client, "get_mcp_client", lambda: client)


def test_literature_search_broken_connector_is_not_thin_literature(monkeypatch):
    """An unparseable/failed lookup must be ok=False, never ok=True with 0 results."""
    _use_stub_client(monkeypatch, _StubClient("pubmed_search_articles: unrecognized response shape"))
    env = tools.dispatch("literature_search", {"query": "LUM fibroblast marker"})
    _assert_envelope(env)
    assert env["ok"] is False
    assert "unrecognized response shape" in env["error"]


def test_literature_search_genuine_zero_hits_stays_ok(monkeypatch):
    """A search the server ran and answered with nothing IS thin literature."""
    _use_stub_client(monkeypatch, _StubClient(None))
    env = tools.dispatch("literature_search", {"query": "no such thing anywhere"})
    _assert_envelope(env)
    assert env["ok"] is True
    assert env["data"]["n_results"] == 0


def test_literature_fetch_broken_connector_is_not_unresolved(monkeypatch):
    """A failed fetch says nothing about the ids — it must not report them unresolved."""
    _use_stub_client(monkeypatch, _StubClient("pubmed_fetch_articles timed out after 8.0s"))
    env = tools.dispatch("literature_fetch", {"pmids": ["37692065"]})
    _assert_envelope(env)
    assert env["ok"] is False
    assert "timed out" in env["error"]


# --------------------------------------------------------------------------- #
# The tool envelope must carry looked_up_at.
#
# memory.reconcile stamps looked_up_at ONLY when a search completed, and
# claims.notes.lookup_state reads it (never the `thin` flag) to tell
# "searched, found nothing" from "never searched". The draft the UI PERSISTS is
# the one loop._draft_from_payload rebuilds off this envelope, so a field dropped
# here is a field lost to disk — and it is lost exactly in the thin case, which
# is the half that carries a finding.
# --------------------------------------------------------------------------- #
def _empty_search(query: str):
    """A lookup that COMPLETED and found nothing. Not a failed lookup."""
    return []


def _roundtrip_draft(env):
    from panoscope.agent import loop as agent_loop

    return agent_loop._draft_from_payload(env["data"])


def test_empty_literature_survives_the_envelope_as_not_found(isolated_memory):
    from panoscope.claims import notes as notes_claims
    from panoscope.corpus.store import Resolution

    tools.set_literature_search(_empty_search)
    env = tools.dispatch(
        "memory_draft",
        {
            "claim": "In our breast TME, LUM marks CAFs in this cluster",
            "scope": "cluster",
            "basis": "own_validation",
            "cluster": "c2",
            "subject_markers": ["LUM"],
        },
    )
    assert env["ok"] is True
    assert env["data"]["tension"]["looked_up_at"], "envelope dropped looked_up_at"

    draft = _roundtrip_draft(env)
    assert draft is not None
    assert notes_claims.lookup_state(draft.tension) is Resolution.NOT_FOUND, (
        "a completed-but-empty search rebuilt as NOT_LOOKED_UP — the three states "
        "collapsed back to two across the tool envelope"
    )


def test_real_citations_still_round_trip_as_resolved(isolated_memory):
    from panoscope.claims import notes as notes_claims
    from panoscope.corpus.store import Resolution

    tools.set_literature_search(_stub_search)
    env = tools.dispatch(
        "memory_draft",
        {
            "claim": "In our breast TME, LUM marks CAFs in this cluster",
            "scope": "cluster",
            "basis": "own_validation",
            "cluster": "c2",
            "subject_markers": ["LUM"],
        },
    )
    draft = _roundtrip_draft(env)
    assert notes_claims.lookup_state(draft.tension) is Resolution.RESOLVED


def test_old_envelope_without_the_field_means_not_looked_up():
    """Fail-closed: a payload written before the field still reads as no lookup."""
    from panoscope.agent import loop as agent_loop
    from panoscope.claims import notes as notes_claims
    from panoscope.corpus.store import Resolution

    draft = agent_loop._draft_from_payload(
        {"claim": "x", "cluster": "c2", "tension": {"thin": True, "query": "q"}}
    )
    assert notes_claims.lookup_state(draft.tension) is Resolution.NOT_LOOKED_UP


def test_note_payload_carries_looked_up_at(isolated_memory):
    """memory_read's model-facing payload carries it too (same collapse, same field)."""
    tools.set_literature_search(_empty_search)
    _write_note(
        claim="LUM marks CAFs here",
        scope="cluster",
        basis="own_validation",
        cluster="c2",
        subject_markers=["LUM"],
    )
    env = tools.dispatch("memory_read", {"cluster": "c2"})
    assert env["data"]["notes"], "no note came back"
    assert "looked_up_at" in env["data"]["notes"][0]["tension"]
