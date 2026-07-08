"""Tests for agent/tools.py — the 7 agent tools and the uniform envelope.

Grounding-floor coverage:

* ``panel_lookup`` reports on/off panel truthfully (ERBB2 on, COL1A1 off) — the
  panel-absence primitive surfaced as a tool.
* ``marker_lookup`` returns LUM's REAL jazzPanda glm_coef for c2 (numbers come
  only from agent.data, never fabricated).
* ``get_spatial('density','LUM')`` reports the precomputed frame as available and
  never recomputes.
* ``memory_write`` -> ``memory_read`` round-trips a c2-scoped note through a tmp
  context dir, with an injected (stubbed) literature_search so no network is used
  and the note is born with real-citation tension.
* Every tool goes through ``dispatch`` and returns ``{ok, data, sources, error}``.
* ``literature_search`` returns ``ok=True`` with >=1 real PMID when the connector
  is up, OR ``ok=False`` gracefully when offline (never a fabricated PMID).

Memory tests point tools at a tmp dir via ``set_memory_base_dir`` and never touch
the real ``context/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import data as _data
from agent import tools
from agent.types import Citation

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
    assert names == set(tools._DISPATCH)
    assert len(tools.TOOL_SCHEMAS) == 7
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
# 6 + 7. memory_write -> memory_read round-trip (c2-scoped)
# --------------------------------------------------------------------------- #
def test_memory_write_then_read_roundtrip(isolated_memory):
    tools.set_literature_search(_stub_search)

    write_env = tools.dispatch(
        "memory_write",
        {
            "claim": "In our breast TME, LUM marks CAFs in this cluster",
            "scope": "cluster",
            "basis": "own_validation",
            "cluster": "c2",
            "subject_cell_type": "Stromal",
            "subject_markers": ["LUM"],
        },
    )
    _assert_envelope(write_env)
    assert write_env["ok"] is True
    note_id = write_env["data"]["id"]
    assert write_env["data"]["cluster"] == "c2"
    assert write_env["data"]["scope"] == "cluster"
    # reconciled against the stub -> tension carries the real agree/dissent PMIDs.
    tension = write_env["data"]["tension"]
    assert tension["thin"] is False
    assert "40000001" in tension["agree"]
    assert "40000002" in tension["dissent"]
    # cited on write (kind="mem").
    assert write_env["sources"][0]["kind"] == "mem"
    assert write_env["sources"][0]["ref"] == note_id

    # read back for c2 -> the note fires.
    read_env = tools.dispatch("memory_read", {"cluster": "c2"})
    _assert_envelope(read_env)
    assert read_env["ok"] is True
    ids = [n["id"] for n in read_env["data"]["notes"]]
    assert note_id in ids


def test_memory_read_scope_enforced_other_cluster(isolated_memory):
    tools.set_literature_search(_stub_search)
    tools.dispatch(
        "memory_write",
        {
            "claim": "c2-only stromal convention",
            "scope": "cluster",
            "basis": "convention",
            "cluster": "c2",
        },
    )
    # a cluster-scoped note must NOT fire for a different cluster.
    other = tools.dispatch("memory_read", {"cluster": "c5"})
    assert other["ok"] is True
    assert other["data"]["n_notes"] == 0


def test_memory_write_cluster_scope_requires_cluster(isolated_memory):
    tools.set_literature_search(_stub_search)
    env = tools.dispatch(
        "memory_write",
        {"claim": "no cluster given", "scope": "cluster", "basis": "convention"},
    )
    assert env["ok"] is False
    assert "cluster" in env["error"].lower()


def test_memory_write_invalid_scope_fails(isolated_memory):
    env = tools.dispatch(
        "memory_write", {"claim": "x", "scope": "galaxy", "basis": "convention"}
    )
    assert env["ok"] is False


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
