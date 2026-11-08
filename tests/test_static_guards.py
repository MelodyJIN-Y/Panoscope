"""Architectural rules that are about structure, checked on the source with ``ast``.

Each guard reads the code, so it bites the day someone writes the forbidden line,
before any behaviour test happens to exercise it:

* ``core/`` never imports a generation client, and importing it loads none
  (IMPLEMENTATION §9 ``test_record_not_generated``, both forms);
* ``ui/`` writes the study store only through ``ui/data_access.py`` (interface §1.2);
* ``ui/`` does no arithmetic on a method statistic (interface §1.2), and the dot plot,
  which aggregates per-cell values in the renderer, is a named open violation;
* no live-citation path is left: rendering a note looks nothing up, and nothing at
  question time reaches PubMed or Europe PMC (corpus A26).
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "panoscope"
CLIENTS = ("anthropic", "mcp", "openai", "litellm")


def _modules(sub: str) -> list[Path]:
    return sorted((PKG / sub).rglob("*.py"))


def _imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


@pytest.mark.req("IMPL-9-01")
@pytest.mark.parametrize("path", _modules("core"), ids=lambda p: p.name)
def test_no_core_module_imports_a_generation_client(path):
    names = _imports(ast.parse(path.read_text()))
    bad = {n for n in names if n.split(".")[0] in CLIENTS or n.startswith("panoscope.agent")}
    assert not bad, f"{path.name} imports {sorted(bad)}"


@pytest.mark.req("IMPL-9-01")
def test_importing_core_loads_no_generation_client():
    """The transitive form: an indirect import through another package is still the
    record being written next to a model client."""
    mods = [f"panoscope.core.{p.stem}" for p in _modules("core") if p.stem != "__init__"]
    code = ("import importlib, sys\n"
            f"for m in {mods!r}: importlib.import_module(m)\n"
            f"print(sorted(m for m in sys.modules if m.split('.')[0] in {CLIENTS!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[]", out


# --------------------------------------------------------------------------- #
# ui/ writes only through the data seam
# --------------------------------------------------------------------------- #
def _store_writes(tree: ast.AST) -> list[str]:
    """``save_*`` calls on ``panoscope.pipeline.store``, however it was imported."""
    aliases, direct = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "panoscope.pipeline":
            aliases |= {a.asname or a.name for a in n.names if a.name == "store"}
        elif isinstance(n, ast.ImportFrom) and n.module == "panoscope.pipeline.store":
            direct |= {a.asname or a.name for a in n.names if a.name.startswith("save_")}
        elif isinstance(n, ast.Import):
            aliases |= {a.asname for a in n.names if a.name == "panoscope.pipeline.store" and a.asname}
    hits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr.startswith("save_") and (
                (isinstance(f.value, ast.Name) and f.value.id in aliases)
                or ast.unparse(f.value).endswith("pipeline.store")):
            hits.append(f"line {n.lineno}: {ast.unparse(f)}")
        elif isinstance(f, ast.Name) and f.id in direct:
            hits.append(f"line {n.lineno}: {f.id}")
    return hits


def test_the_store_write_detector_sees_each_import_form():
    for src in ("from panoscope.pipeline import store\nstore.save_x(1)",
                "from panoscope.pipeline import store as s\ns.save_x(1)",
                "from panoscope.pipeline.store import save_x\nsave_x(1)",
                "import panoscope.pipeline.store\npanoscope.pipeline.store.save_x(1)",
                "from panoscope import pipeline\npipeline.store.save_x(1)"):
        assert _store_writes(ast.parse(src)), src
    assert not _store_writes(ast.parse("from panoscope.pipeline import store\nstore.load_x()"))


@pytest.mark.req("UI-1.2-02")
@pytest.mark.parametrize("path", [p for p in _modules("ui") if p.name != "data_access.py"],
                         ids=lambda p: p.name)
def test_ui_writes_the_store_only_through_data_access(path):
    assert not _store_writes(ast.parse(path.read_text())), path.name


# --------------------------------------------------------------------------- #
# ui/ does no arithmetic on a statistic
# --------------------------------------------------------------------------- #
#: The statistics the renderer shows. Names from the methods' papers and the spec.
STATS = {"glm_coef", "pearson", "z", "cgst_z", "log2FC", "logFC", "avg_log2FC",
         "test_statistic", "statistic", "max_gc_corr", "max_gg_corr"}
_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)


def _stat_arithmetic(tree: ast.AST) -> list[str]:
    def names(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in STATS:
                yield n.id
            elif isinstance(n, ast.Attribute) and n.attr in STATS:
                yield n.attr
            elif isinstance(n, ast.Constant) and n.value in STATS:
                yield n.value

    hits = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.BinOp, ast.AugAssign)) and isinstance(n.op, _ARITH):
            if isinstance(n.op, ast.Mod) and isinstance(getattr(n, "left", None), ast.Constant):
                continue                      # "%s" % x is formatting, not arithmetic
            if found := sorted(set(names(n))):
                hits.append(f"line {n.lineno}: {found}: {ast.unparse(n)[:80]}")
    return hits


def test_the_arithmetic_detector_sees_a_statistic_being_scaled():
    assert _stat_arithmetic(ast.parse("x = item.statistic * 2"))
    assert _stat_arithmetic(ast.parse("x = row['glm_coef'] - 1"))
    assert not _stat_arithmetic(ast.parse("x = f'{item.statistic:.2f}'"))
    assert not _stat_arithmetic(ast.parse("v |= {'statistic': 1}"))


@pytest.mark.req("UI-1.2-01", "ARCH-13.3-01")
@pytest.mark.parametrize("path", _modules("ui"), ids=lambda p: p.name)
def test_ui_does_no_arithmetic_on_a_statistic(path):
    assert not _stat_arithmetic(ast.parse(path.read_text())), path.name


@pytest.mark.req("ARCH-13.3-01")
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BUG (decision §6.6 pending): the dot plot computes mean and "
                          "% expressing per cluster in the renderer")
def test_the_dot_plot_renders_values_it_was_given_and_derives_none():
    """Plots render, they do not derive. The dot plot aggregates exported per-cell
    values (``.mean()``) inside ``ui/spatial_stage._render_dotplot`` and admits it in
    its caption. Either the pipeline precomputes those values as evidence, or the spec
    records an exception; until then this stays red under xfail."""
    from panoscope.ui import spatial_stage

    tree = ast.parse(inspect.getsource(spatial_stage._render_dotplot))
    aggregations = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("mean", "median", "sum", "std", "groupby", "agg")]
    assert not aggregations, aggregations


# --------------------------------------------------------------------------- #
# No live-citation path left (corpus A26)
# --------------------------------------------------------------------------- #
@pytest.mark.req("CORP-A26")
def test_rendering_a_note_citation_looks_nothing_up():
    from panoscope.agent import memory

    assert list(inspect.signature(memory.render_citation).parameters) == ["note"]


#: Where a PubMed / Europe PMC call may live: the corpus build and top-up, the MCP
#: client itself, and the CLI that runs ``corpus build``.
_LOOKUP_ALLOWED = {"corpus", "agent/mcp_client.py", "cli.py"}


def _reaches_the_literature(path: Path) -> bool:
    text = path.read_text()
    return ("panoscope.agent.mcp_client" in _imports(ast.parse(text))
            or "mcp_client" in text or "europepmc" in text.lower()
            or "eutils.ncbi" in text)


@pytest.mark.req("CORP-A26")
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BUG (needs a product decision): the chat's literature_search "
                          "tool and the override reconcile call PubMed at question time; "
                          "CLAUDE.md Memory asks for that cross-check at override")
def test_nothing_at_question_time_reaches_pubmed_or_europe_pmc():
    rel = {str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if _reaches_the_literature(p)}
    outside = {r for r in rel if r.split("/")[0] not in _LOOKUP_ALLOWED
               and r not in _LOOKUP_ALLOWED}
    assert not outside, sorted(outside)
