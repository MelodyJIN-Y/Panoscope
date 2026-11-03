"""The COMPARISON root runs, and the registry -- not the root -- picks the method.

``tests/test_workflow_de.py`` drives the four §9.7 checks against hand-written
evidence and a stub adapter. Nothing drove them through a composition root, because
there was none: ``workflows/de.py`` had no production caller at all, so its checks
and ``core/reconcile.py``'s join existed only under test.

What is pinned here is the wiring and only the wiring. No check's content is
re-asserted -- that file owns it, and a second set of expectations would be the
duplicate that drifts. The three things this adds are:

* the root writes a record per unit under ``records/comparison/``;
* the METHOD is resolved by schema through the registry, so adding a second one is
  a registration rather than an edit to the root;
* the cross-method join actually joins -- the DE result against the spatial result
  for the same unit, which is the one check that cannot be demonstrated without two
  methods and a study carrying both.

**The fixture is declared synthetic and says so in its own manifest.** Every number
in ``de_edger.csv`` was written by hand; nothing was fitted, by edgeR or anything
else. A plausible-looking DE table in ``data/`` would be a fabricated result, which
is the worst failure this project can produce. A registered workflow demonstrated
against a *declared* synthetic artifact is an honest deliverable; a fake dataset is
not.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from panoscope import methods, workflows
from panoscope.agent import config as cfg
from panoscope.data import layout
from panoscope.data.study import Study
from panoscope.pipeline.stages import de as de_stage
from panoscope.types.outputs import TaskKind

FIXTURE = "synthetic_multisample_fixture"
DEMO = "xenium_hbreast_sample1"


@pytest.fixture
def built(tmp_path: Path):
    """The fixture study built into a copy, with its COMPARISON records written.

    A copy, because ``run_de`` writes: the committed fixture is inputs-only and the
    tree guard would report any record left in it as residue.
    """
    from panoscope.pipeline import records

    root = tmp_path / "root"
    shutil.copytree(layout.studies_root(), root / "studies")
    monkey = pytest.MonkeyPatch()
    monkey.setenv("PANOSCOPE_DATA_ROOT", str(root))
    layout.clear_cache()
    try:
        study = Study.open(FIXTURE)
        with cfg.use_study(study):
            marker = {u: b.evidence for u, b in records.build_all(study).items()}
            written = de_stage.run_de(marker, FIXTURE)
        yield study, written
    finally:
        monkey.undo()
        layout.clear_cache()


def test_the_de_workflow_and_both_methods_are_registered() -> None:
    """If this is false, every assertion below passes for the wrong reason."""
    assert "de-interpretation" in workflows.names()
    names = {m.name for m in methods.for_workflow("de-interpretation")}
    assert {"edger", "seurat_de"} <= names, names


def test_the_root_writes_one_comparison_record_per_unit(built) -> None:
    """The slice exists on disk, under the task kind the workflow declares."""
    _study, written = built
    assert set(written) == {"c1", "c2", "c3"}, written
    for unit, path in written.items():
        assert path.parent.name == "comparison", path
        assert path.is_file()
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["unit"] == unit
        assert rec["task_kind"] == TaskKind.COMPARISON.value


def test_no_comparison_record_carries_a_band(built) -> None:
    """Checked as a FIELD NAME, not as a value.

    A behavioural test only catches a band that is filled in; the field guard fails
    the moment one is declared, which is when it costs nothing to undo. CLAUDE.md:
    no unit-level band on CHARACTERISATION or COMPARISON.
    """
    _study, written = built
    for unit, path in written.items():
        flat = json.dumps(json.loads(path.read_text(encoding="utf-8")))
        assert '"band"' not in flat, f"{unit}: a comparison record declares a band"
        assert '"confidence"' not in flat, f"{unit}: comparison record has confidence"


def test_the_method_is_chosen_by_schema_not_by_filename(tmp_path: Path) -> None:
    """The registry decides, so a second method is a registration not a root edit.

    Two tables, two schemas, two methods -- and the root never names either. A
    filename convention would have meant the root deciding that ``de_edger.csv`` is
    an edgeR table, which is the root knowing about a method.
    """
    import pandas as pd

    edger_csv = tmp_path / "de_a.csv"
    edger_csv.write_text(
        "gene,cluster,logFC,logCPM,FDR\nSYNG01,c1,2.0,7.0,1.0e-04\n",
        encoding="utf-8")
    seurat_csv = tmp_path / "de_b.csv"
    seurat_csv.write_text(
        "gene,cluster,avg_log2FC,p_val_adj\nSYNG01,c1,2.0,1.0e-04\n",
        encoding="utf-8")

    assert de_stage._method_for(pd.read_csv(edger_csv)).name == "edger"
    assert de_stage._method_for(pd.read_csv(seurat_csv)).name == "seurat_de"


def test_a_table_no_method_can_read_refuses_and_names_the_schemas(
        tmp_path: Path) -> None:
    """"This is not a DE table" is only useful beside what one would have looked
    like, so the refusal quotes every method's own reason."""
    import pandas as pd

    bad = tmp_path / "de_bad.csv"
    bad.write_text("gene,top_cluster,glm_coef\nSYNG01,c1,19.0\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        de_stage._method_for(pd.read_csv(bad))
    assert "edger" in str(e.value) and "seurat_de" in str(e.value)


def test_a_study_with_no_de_result_builds_no_comparison_slice() -> None:
    """Absence is not an error. Most studies have no DE result, and the slice is
    simply not built -- the same shape as the enrichment stage."""
    study = Study.open(DEMO)
    assert de_stage.de_result_path(study) is None
    with cfg.use_study(study):
        assert de_stage.run_de({}, DEMO) == {}


def test_the_cross_method_join_actually_joins(built) -> None:
    """§9.7 check 4, on real composed evidence rather than a hand-built ``Side``.

    This is the check that cannot run without two methods and a study carrying
    both, and the one ARCHITECTURE calls "the figure that proves the framework is
    not method-blind". The fixture is built so one subject pulls both ways: SYNG03
    carries a high spatial coefficient and a slightly negative fold change.

    The assertion is that the join RAN and reached a verdict naming the other
    method -- not what the verdict is, which ``tests/test_workflow_de.py`` owns.
    """
    _study, written = built
    rec = json.loads(written["c1"].read_text(encoding="utf-8"))
    coherence = [d for d in rec["decisions"] if d["rule"].startswith("de:coherence")]
    assert coherence, "the cross-method check left no decision in the record"
    assert any("jazzpanda" in json.dumps(d) for d in coherence), (
        "the coherence decision does not name the other method, so nothing was "
        "actually joined"
    )


def test_every_de_decision_in_a_record_is_a_registered_rule(built) -> None:
    """``Trace.note`` refuses an unregistered id, so this would already have raised.

    That is the point: it pins that the guarantee is structural rather than
    reviewed, and it is how a free-string rule gets caught before it ships.
    """
    from panoscope.types.trace import RULES

    _study, written = built
    for unit, path in written.items():
        for d in json.loads(path.read_text(encoding="utf-8"))["decisions"]:
            assert d["rule"] in RULES, f"{unit}: unregistered rule {d['rule']!r}"
