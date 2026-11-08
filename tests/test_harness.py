"""The test harness itself: per-test study trees, the requirement registry and its
markers, the reverse-order switch, the network block, the study builders and the
scripted model. Each check fails if its piece of the harness stops doing its job."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

HARNESS = "IMPL-7-01"  # test isolation infrastructure (IMPLEMENTATION §7 Phase A)


# --------------------------------------------------------------------------- #
# Per-test study tree. These two run in file order: under the old session-shared
# tree the first one's decision was still there when the second one opened.
# --------------------------------------------------------------------------- #
@pytest.mark.req(HARNESS)
def test_a_decision_written_in_one_tests_tree(app_tree) -> None:
    from panoscope.pipeline import decisions as D
    from panoscope.ui import data_access as da

    D.accept("c1", "Tumor", by="tester", at="2026-11-09T00:00:00+00:00")
    assert da.decisions()["c1"]["state"] == "ACCEPTED"


@pytest.mark.req(HARNESS)
def test_is_not_seen_by_the_next_test(app_tree) -> None:
    from panoscope.ui import data_access as da

    assert da.decisions() == {}
    assert (app_tree / "xenium_hbreast_sample1" / "viz").is_symlink()
    assert not (app_tree / "xenium_hbreast_sample1" / "interp").is_symlink()


# --------------------------------------------------------------------------- #
# The registry and the req marker, run in a child pytest on a throwaway suite.
# --------------------------------------------------------------------------- #
_REG = "".join(f'[[req]]\nid = "R-{i}"\nsource = "s"\ntext = "t"\naliases = []\n\n'
               for i in (1, 2, 3))
_SUITE = '''
import pytest

@pytest.mark.req("R-1")
def test_built():
    assert True

@pytest.mark.req("R-2", gap=True)
def test_gap_missing():
    assert False, "not built"

@pytest.mark.req("R-2", gap=True)
def test_gap_wrong_reason():
    {}["missing"]

@pytest.mark.req("R-2", gap=True)
def test_gap_closed():
    assert True
'''


def _child(tmp_path: Path, suite: str, *args: str, env: dict | None = None):
    (tmp_path / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (tmp_path / "requirements.toml").write_text(_REG)
    (tmp_path / "test_child.py").write_text(suite)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-v",
         "--rootdir", str(tmp_path), str(tmp_path), *args],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
        env={**{k: v for k, v in os.environ.items() if k != "PANOSCOPE_TEST_ORDER"},
             **(env or {})})


@pytest.mark.req(HARNESS)
def test_a_gap_is_a_strict_xfail_that_only_an_assert_satisfies(tmp_path) -> None:
    r = _child(tmp_path, _SUITE, "--trace-report", str(tmp_path / "t.md"))
    out = r.stdout
    assert "test_built PASSED" in out, out
    assert "test_gap_missing XFAIL" in out
    assert "test_gap_wrong_reason FAILED" in out      # a KeyError is not the gap
    assert "test_gap_closed FAILED" in out            # XPASS(strict): the gap closed
    report = (tmp_path / "t.md").read_text()
    assert "| R-1 | built |" in report and "| R-2 | gap |" in report
    assert "| R-3 | none |" in report


@pytest.mark.req(HARNESS)
def test_an_unknown_id_is_a_usage_error(tmp_path) -> None:
    r = _child(tmp_path, 'import pytest\n\n@pytest.mark.req("R-9")\ndef test_x():\n    pass\n')
    assert r.returncode == 4 and "R-9" in r.stdout + r.stderr


@pytest.mark.req(HARNESS)
def test_trace_strict_fails_on_an_untested_requirement(tmp_path) -> None:
    r = _child(tmp_path, _SUITE, "--collect-only", "--trace-strict")
    msg = (r.stdout + r.stderr).split("no test:")[-1]
    assert r.returncode == 4 and "R-3" in msg and "R-1" not in msg and "R-2" not in msg


@pytest.mark.req(HARNESS)
def test_a_deselected_live_or_budget_test_still_covers_its_row(tmp_path) -> None:
    """R-3's only test runs in the nightly budget job; -m "not live" must not report it untested."""
    suite = _SUITE + '\n@pytest.mark.budget\n@pytest.mark.req("R-3")\ndef test_fast():\n    pass\n'
    r = _child(tmp_path, suite, "--collect-only", "-m", "not live", "--trace-strict",
               "--trace-report", str(tmp_path / "t.md"))
    assert r.returncode == 0, r.stdout + r.stderr     # strict: R-3 is covered
    assert "| R-3 | built |" in (tmp_path / "t.md").read_text()


@pytest.mark.req(HARNESS)
def test_reverse_order_reverses_the_run(tmp_path) -> None:
    def order(env):
        out = _child(tmp_path, _SUITE, "--collect-only", "-q", "-q", env=env).stdout
        return [ln.split("::")[1] for ln in out.splitlines() if "::" in ln]

    forward = order({})
    assert len(forward) == 4 and order({"PANOSCOPE_TEST_ORDER": "reverse"}) == forward[::-1]


@pytest.mark.req(HARNESS)
def test_browser_and_budget_run_only_when_the_marker_expression_names_them(tmp_path) -> None:
    suite = ("import pytest\n\ndef test_plain():\n    pass\n\n"
             "@pytest.mark.browser\ndef test_click():\n    pass\n\n"
             "@pytest.mark.budget\ndef test_fast():\n    pass\n")

    def ran(*args):
        out = _child(tmp_path, suite, "--collect-only", "-q", "-q", *args).stdout
        return sorted(ln.split("::")[1] for ln in out.splitlines() if "::" in ln)

    assert ran("-m", "not live") == ["test_plain"]
    assert ran("-m", "browser") == ["test_click"]
    assert ran("-m", "budget") == ["test_fast"]


@pytest.mark.req(HARNESS)
def test_every_registry_id_is_unique_and_has_a_source() -> None:
    rows = tomllib.loads((Path(__file__).parent / "requirements.toml").read_text())["req"]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all(r["source"] and r["text"] for r in rows)
    assert all(r.get("status") in (None, "decision", "deferred", "refused") for r in rows)


# --------------------------------------------------------------------------- #
# Network block
# --------------------------------------------------------------------------- #
@pytest.mark.req(HARNESS)
@pytest.mark.parametrize("host", ["192.0.2.1", "2001:db8::1"])  # IPv4 and IPv6 test-nets
def test_an_internet_connect_is_refused_and_recorded(host, no_network) -> None:
    with pytest.raises(OSError, match="network blocked"):
        socket.create_connection((host, 443), timeout=1)
    assert no_network and no_network[-1][1] == 443


# --------------------------------------------------------------------------- #
# Study builders: every shape builds, and the build reads what the builder wrote.
# --------------------------------------------------------------------------- #
_SHAPES = {
    "labelled_subset": dict(clusters=["c1", "c5"]),
    "labelless": dict(labels=False),
    "partially_labelled": dict(clusters=["c1", "c2", "c3"], labels={"c1": "Tumor"}),
    "two_sample_joint": dict(clusters=["c1", "c2"], samples=["s1", "s2"], clustering="joint"),
    "enrichment_only": dict(source="synthetic", methods=("jazzpanda_cgst",), labels=False),
    "de_only": dict(source="synthetic", methods=("edger",), labels=False),
}


@pytest.mark.req(HARNESS)
@pytest.mark.parametrize("shape", sorted(_SHAPES) + ["kidney_standin"])
def test_every_builder_shape_builds(shape, study_builder, monkeypatch) -> None:
    from panoscope.agent import loop
    from panoscope.pipeline import paths
    from panoscope.pipeline import run as pipeline_run
    from tests.fixtures.studies import KIDNEY_LAB_MARKERS, make_kidney_standin

    monkeypatch.setattr(loop, "has_model_key", lambda: False)
    if shape == "kidney_standin":
        d = make_kidney_standin(study_builder.root)
    else:
        d = study_builder(shape, **_SHAPES[shape])
    inputs = d / "inputs"
    table = next(inputs / f for f in ("markers_top.csv", "enrichment.csv", "de_edger.csv")
                 if (inputs / f).exists())
    df = pd.read_csv(table)
    want = sorted(set(df["top_cluster" if "top_cluster" in df else "cluster"]) - {"NoSig"})

    pipeline_run.run(d.name)
    man = json.loads(paths.manifest_json(d.name).read_text())
    assert man["clusters"] == want
    if shape == "kidney_standin":
        panel = set(pd.read_parquet(inputs / "panel.parquet")["gene"])
        assert man["tissue"] == "human kidney"
        assert len([g for _, g in KIDNEY_LAB_MARKERS if g in panel]) <= 2
    if shape == "two_sample_joint":
        assert man["samples"] == ["s1", "s2"] and man["clustering_mode"] == "joint"
    if "clusters" in _SHAPES.get(shape, {}):
        assert want == sorted(_SHAPES[shape]["clusters"])
    if table.name == "markers_top.csv":  # the builder honoured ``labels``
        labels = _SHAPES.get(shape, {}).get("labels", shape != "kidney_standin")
        labelled = set(df.dropna(subset=["cell_type"])["top_cluster"]) if "cell_type" in df else set()
        assert (inputs / "cluster_key.json").exists() is (labels is True)
        if labels is False:
            assert labelled == set()
        elif isinstance(labels, dict):
            assert labelled == set(labels)


# --------------------------------------------------------------------------- #
# The scripted model drives the REAL agent loop.
# --------------------------------------------------------------------------- #
@pytest.mark.req(HARNESS)
def test_the_fake_model_is_called_by_the_real_loop(app_tree, fake_model) -> None:
    from panoscope.agent import loop, tools

    fake_model.script("c1 is the tumour cluster.")
    loop.chat("what is c1?", cluster="c1")
    first = fake_model.calls[0]
    assert first["tools"] == tools.TOOL_SCHEMAS
    assert first["messages"][-1] == {"role": "user", "content": "what is c1?"}


@pytest.mark.req(HARNESS)
def test_an_exhausted_script_degrades_to_the_fallback_without_raising(app_tree, fake_model) -> None:
    from panoscope.agent import loop

    r = loop.chat("what is c1?", cluster="c1")
    assert fake_model.calls and r.text and r.used_fallback
    with pytest.raises(RuntimeError, match="exhausted"):
        fake_model.create()


@pytest.mark.req(HARNESS)
def test_a_citation_resolves_only_when_listed_as_real(fake_model) -> None:
    from panoscope.agent import loop

    assert not loop._default_literature_verifier("PMID:12345")
    fake_model.real_ids.add("PMID:12345")
    assert loop._default_literature_verifier("PMID:12345")
