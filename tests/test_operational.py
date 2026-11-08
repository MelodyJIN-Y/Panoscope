"""Operational and platform rows (ARCH §5-§7, §9.5, §11.2, §12, §18; llm-access §5;
IMPLEMENTATION §8): what holds today, and strict gaps for what is not built.

Each gap fails on an assert about the spec's observable outcome, so it turns red the
day the feature lands.
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

from tests.conftest import DATASETS_ROOT, DEMO_ID

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "panoscope"


# --------------------------------------------------------------------------- #
# ARCH §7: reference data resolves, it never retrieves at question time
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-7-01")
def test_a_chat_turn_that_reads_reference_data_opens_no_connection(
        app_tree, fake_model, no_network) -> None:
    """The model asks for the panel, the markers and a rival test (all of which read the
    reference view); the turn completes with zero internet connects and nothing new in
    the reference cache."""
    from panoscope.agent import loop
    from tests.fixtures.fake_model import reply, tool_use

    fake_model.script(
        reply(tool_use("panel_lookup", {"gene": "CD4"}),
              tool_use("marker_lookup", {"cluster": "c3"}),
              tool_use("discriminate_call", {"cluster": "c3", "alt_cell_type": "T cells"})),
        "c3 reads as Macrophages.")
    r = loop.chat("is c3 really macrophages?", cluster="c3")
    assert len(fake_model.calls) == 2, "the tools ran and the model was asked again"
    tool_results = fake_model.calls[1]["messages"][-1]["content"]
    assert len(tool_results) == 3 and r.text
    assert no_network == []
    assert list(Path(os.environ["PANOSCOPE_REFERENCE_DIR"]).rglob("*.*")) == []


# --------------------------------------------------------------------------- #
# ARCH §11.2 / llm-access §5: the client, caching, credentials, dependencies
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-11.2-03")
def test_anthropic_base_url_is_honoured(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from panoscope.agent import loop

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/gateway")
    client = loop.PanoscopeAgent(api_key="sk-test")._get_client()
    assert str(client.base_url).startswith("http://127.0.0.1:9/gateway")


@pytest.mark.req("ARCH-11.2-03", gap=True)
def test_panoscope_provider_selects_an_openai_compatible_client(monkeypatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    from panoscope.agent import loop

    monkeypatch.setenv("PANOSCOPE_PROVIDER", "openai_compatible")
    client = loop.PanoscopeAgent(api_key="sk-test")._get_client()
    assert not isinstance(client, anthropic.Anthropic), type(client)


@pytest.mark.req("ARCH-11.2-02", gap=True)
def test_every_model_request_asks_for_prompt_caching(app_tree, fake_model) -> None:
    from panoscope.agent import loop

    fake_model.script("c1 is the tumour cluster.")
    loop.chat("what is c1?", cluster="c1")
    assert fake_model.calls
    assert "cache_control" in json.dumps(fake_model.calls[0], default=str)


@pytest.mark.req("ARCH-18-05", gap=True)
def test_a_chat_turn_logs_with_one_correlation_id(app_tree, fake_model, caplog) -> None:
    from panoscope.agent import loop

    fake_model.script("c1 is the tumour cluster.")
    with caplog.at_level("DEBUG"):
        loop.chat("what is c1?", cluster="c1")
    ids = {getattr(r, "correlation_id", None) for r in caplog.records} - {None}
    assert len(ids) == 1, [r.getMessage()[:80] for r in caplog.records][:10]


@pytest.mark.req("ARCH-10.6-02", gap=True)
def test_a_chat_turn_appends_to_a_usage_log(app_tree, fake_model) -> None:
    from panoscope.agent import loop

    fake_model.script("c1 is the tumour cluster.")
    loop.chat("what is c1?", cluster="c1")
    names = [p.name for p in Path(os.environ["PANOSCOPE_CONTEXT_DIR"]).rglob("*")]
    assert any("usage" in n for n in names), names


_CLOUD_CREDENTIALS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                      "GOOGLE_APPLICATION_CREDENTIALS", "AZURE_CLIENT_SECRET",
                      "AZURE_OPENAI_API_KEY", "VERTEX_API_KEY")


@pytest.mark.req("LLM-5-01")
def test_nothing_asks_the_biologist_to_paste_a_cloud_credential() -> None:
    """The SDK's own credential chain owns cloud credentials: no shipped string, and no
    line of .env.example, names one."""
    shipped = [p for p in PKG.rglob("*") if p.suffix in (".py", ".md")]
    offenders = [f"{p.relative_to(REPO)}: {c}" for p in [*shipped, REPO / ".env.example"]
                 for c in _CLOUD_CREDENTIALS if c in p.read_text(errors="ignore")]
    assert offenders == []
    assert "ANTHROPIC_API_KEY" in (REPO / ".env.example").read_text()   # the file is read


def _pyproject() -> dict:
    with (REPO / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


@pytest.mark.req("LLM-5-02")
def test_no_gateway_library_is_a_dependency() -> None:
    proj = _pyproject()["project"]
    deps = [*proj["dependencies"], *(d for ds in proj["optional-dependencies"].values()
                                     for d in ds)]
    assert any(d.startswith("anthropic") for d in deps)               # the list is real
    assert [d for d in deps if d.lower().startswith(("litellm", "openrouter"))] == []


# --------------------------------------------------------------------------- #
# IMPLEMENTATION §8: the study store never ships; the wheel stays small
# --------------------------------------------------------------------------- #
@pytest.mark.req("IMPL-8-02")
def test_the_wheel_excludes_the_study_store_and_its_package_is_under_5_mb() -> None:
    """The CI wheel job builds and measures the real wheel; this is its offline floor:
    only panoscope/ is packaged, the exclude list names the store, and everything under
    panoscope/ adds up to under 5 MB."""
    build = _pyproject()["tool"]["hatch"]["build"]
    assert build["targets"]["wheel"]["packages"] == ["panoscope"]
    assert {"studies", "tests"} <= set(build["exclude"])
    assert not (PKG / "studies").exists()
    size = sum(p.stat().st_size for p in PKG.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts)
    assert size < 5 * 1024 * 1024, size


# --------------------------------------------------------------------------- #
# ARCH §12: source, version and licence per claim; --official-sources-only
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-12-02")
def test_every_reference_edge_names_its_source_release_and_licence() -> None:
    from panoscope.agent import config as cfg
    from panoscope.reference import view

    v = view.for_study(cfg.current_study())
    releases = dict(v.releases)
    assert releases and all(releases.values())                     # a version per source
    edges = v.graph.edges
    assert edges
    for e in edges:
        assert e.source and e.licence, e


@pytest.mark.req("ARCH-12-02", "CORP-A10", gap=True)
def test_build_takes_an_official_sources_only_flag() -> None:
    import argparse

    from panoscope.cli import build_parser

    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert "--official-sources-only" in sub.choices["build"].format_help()


# --------------------------------------------------------------------------- #
# Not built: one strict gap each, asserting the spec's observable outcome
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-6.5-01", gap=True)
def test_the_marker_skill_ships_its_grounding_checker_as_an_executable() -> None:
    folder = PKG / "skills" / "marker-interpretation"
    assert (folder / "SKILL.md").is_file()
    assert [p for p in folder.rglob("*.py")], sorted(p.name for p in folder.rglob("*"))


@pytest.mark.req("ARCH-18-07", gap=True)
def test_the_manifest_carries_an_intake_cost_receipt() -> None:
    manifest = json.loads((DATASETS_ROOT / DEMO_ID / "manifest.json").read_text())
    assert any("cost" in k or "usage" in k or "receipt" in k for k in manifest), sorted(manifest)


@pytest.mark.req("ARCH-6.4-03", gap=True)
def test_a_lab_convention_given_at_intake_is_stored_as_a_lab_scoped_claim(demo) -> None:
    from panoscope.agent.intake import derive
    from panoscope.platforms.xenium import XeniumAdapter

    draft = derive(XeniumAdapter(), demo).stated("lab_conventions", "we call FAP+ cells CAFs")
    (entry,) = draft.background.to_manifest()["lab_conventions"]
    assert entry["value"] == "we call FAP+ cells CAFs"
    assert entry.get("scope") == "lab" and entry.get("author") and entry.get("created"), entry


@pytest.mark.req("ARCH-9.5-06", gap=True)
def test_a_differential_abundance_workflow_is_registered() -> None:
    """n=2 replicates must read UNDERPOWERED at the UNIT level; there is no DA
    workflow to carry that unit yet."""
    from panoscope import workflows

    assert any("abundance" in n for n in workflows.names()), workflows.names()


@pytest.mark.req("ARCH-5.2-02", gap=True)
def test_confirmed_calls_export_as_an_obs_column() -> None:
    from panoscope.core import export

    assert [n for n in dir(export) if "obs" in n.lower()], "no obs-column export"


@pytest.mark.req("ARCH-5.3-02", gap=True)
def test_a_two_sample_study_builds_per_sample_records() -> None:
    from panoscope.data.study import Study
    from panoscope.pipeline import records

    study = Study.open("synthetic_multisample_fixture")
    samples = sorted(p.name for p in (study.dir / "inputs" / "samples").iterdir())
    assert samples == ["sampleA", "sampleB"]
    built = records.build_all(study)
    assert any(s in str(k) for k in built for s in samples), sorted(built)


@pytest.mark.req("UI-4.5-02", gap=True)
def test_the_record_says_whether_a_replicate_divergence_touches_the_call() -> None:
    """The fixture's c1 has one marker that reads differently across its two samples;
    verify rises only when the call depends on that marker (affects_call)."""
    from panoscope.data.study import Study
    from panoscope.pipeline import records

    rec = records.build_all(Study.open("synthetic_multisample_fixture"))["c1"].record
    assert any(d.rule == "concordance:divergent" for d in rec.decisions)
    assert any("affects_call" in d.rule for d in rec.decisions), \
        sorted({d.rule for d in rec.decisions})


# --------------------------------------------------------------------------- #
# Implementation plan, minor: a cluster's own measurement is OBSERVED
# --------------------------------------------------------------------------- #
@pytest.mark.req("PLAN-MINOR-01")
def test_every_cluster_reads_its_own_measurement_as_observed() -> None:
    """Every cluster that jazzPanda reported rows for (markers_top.csv top_cluster) is
    OBSERVED in its own frame in both of its records, never TESTED_ABSENT."""
    import csv

    from panoscope.types.subject import subject, unit

    study = DATASETS_ROOT / DEMO_ID
    with (study / "inputs" / "markers_top.csv").open() as f:
        clusters = {r["top_cluster"] for r in csv.DictReader(f)} - {"NoSig", ""}
    assert len(clusters) == 9
    for c in sorted(clusters):
        key = subject("unit", u=unit(c)).key
        for kind in ("assignment", "characterisation"):
            rec = json.loads((study / "interp" / "records" / kind / f"{c}.json").read_text())
            own = [d["outcome"] for d in rec["decisions"] if d.get("refs") == [key]]
            assert own == ["observed"], (c, kind, own)
