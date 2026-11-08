"""The method contract, over EVERY registered method (ARCHITECTURE §4, IMPLEMENTATION §9).

``test_registries.py`` proves the registry's refusals with jazzPanda and a stub. This
file runs each real entry through the same five questions, and asserts what comes
out, not that it ran:

(a) it joins a registered workflow; (b) its limits reach the biologist, word for word,
in that workflow's "What produced these numbers" footer; (c) its declared statistic is
read off the result table, never recomputed; (d) silence means what the four states
say it means; (e) only an ASSIGNMENT method's evidence ever ends in a band.

Inputs are the committed ``synthetic_multisample_fixture`` (made-up genes, says so in
its manifest), read through the real Xenium adapter, so a method is loaded the way the
pipeline loads it. Seurat has no committed table, so one is written from the edgeR
rows in Seurat's own column names. A method added without a row in ``_INPUTS`` fails
here by name: "every registered method" is the point.
"""

from __future__ import annotations

import dataclasses
import html

import pandas as pd
import pytest

from panoscope import methods, platforms, workflows
from panoscope.core.interpret import interpret
from panoscope.data.study import Study
from panoscope.types.contracts import UnrecognisedResult
from panoscope.types.measured import Measured, MeasuredState
from panoscope.types.outputs import Assignment, TaskKind
from panoscope.types.subject import gene, gene_set, subject
from panoscope.ui import surface
from panoscope.workflows.enrichment import EnrichmentReference
from panoscope.workflows.marker import MarkerReference

STUDY = "synthetic_multisample_fixture"
OFF = "OFFPANELGENE1"

#: method -> (committed input, the column its paper reports the statistic in, the
#: unit column, a row that names only genes the panel does not carry). Literal on
#: purpose: the expected column comes from the method's output format, not its module.
_INPUTS = {
    "jazzpanda_spatial_glm": ("markers_top.csv", "glm_coef", "top_cluster",
                              {"gene": OFF, "top_cluster": "c1", "glm_coef": 30.0,
                               "pearson": 0.9, "max_gg_corr": 0.9, "max_gc_corr": 0.9}),
    "jazzpanda_cgst": ("enrichment.csv", "test_statistic", "cluster",
                       {"gene_set": "SYNSET_OFFPANEL", "cluster": "c1", "test_statistic": 9.0,
                        "p_adj_bh": 1e-6, "geneset_size": 50, "n_overlap": 3,
                        "genes_tested": f"{OFF},OFFPANELGENE2,OFFPANELGENE3",
                        "genes_selected": f"{OFF},OFFPANELGENE2,OFFPANELGENE3"}),
    "edger": ("de_edger.csv", "logFC", "cluster",
              {"gene": OFF, "cluster": "c1", "logFC": 3.0, "FDR": 1e-6}),
    "seurat_de": (None, "avg_log2FC", "cluster",
                  {"gene": OFF, "cluster": "c1", "avg_log2FC": 3.0, "p_val_adj": 1e-6}),
}

#: What each task kind's workflow is handed as its reference side.
_REFERENCE = {
    TaskKind.ASSIGNMENT: MarkerReference(label="Synthetic Type A", canonical=()),
    TaskKind.CHARACTERISATION: EnrichmentReference(),
    TaskKind.COMPARISON: None,
}

ALL = sorted(methods.METHODS)


def _table(name: str) -> pd.DataFrame:
    src, *_ = _INPUTS[name]
    inputs = Study.open(STUDY).inputs
    if src is not None:
        return pd.read_csv(inputs / src)
    edger = pd.read_csv(inputs / "de_edger.csv")      # Seurat's FindAllMarkers names
    return pd.DataFrame({"gene": edger["gene"], "cluster": edger["cluster"],
                         "avg_log2FC": edger["logFC"], "p_val_adj": edger["FDR"],
                         "p_val": edger["PValue"]})


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """name -> (input table with the off-panel row, Traced evidence by unit)."""
    study = Study.open(STUDY)
    adapter = platforms.for_study(study)
    out = {}
    for name in ALL:
        assert name in _INPUTS, f"{name} is registered but has no contract input here"
        df = pd.concat([_table(name), pd.DataFrame([_INPUTS[name][3]])], ignore_index=True)
        path = tmp_path_factory.mktemp(name) / "input.csv"
        df.to_csv(path, index=False)
        out[name] = (df, methods.get(name).load(str(path), adapter, study))
    return out


@pytest.mark.req("ARCH-4-01", "ARCH-4-02")
@pytest.mark.parametrize("name", ALL)
def test_every_method_joins_a_registered_workflow(name):
    m = methods.get(name)
    assert m.workflow in workflows.WORKFLOWS
    assert m in methods.for_workflow(m.workflow)


@pytest.mark.req("ARCH-4-01", "CLAUDE-ST-01")
@pytest.mark.parametrize("name", ALL)
def test_every_limit_and_assumption_reaches_the_method_footer(name):
    """The whole sentence, escaped as rendered: a truncated or paraphrased limit is a
    second copy of the paper that can drift from the first."""
    m = methods.get(name)
    assert str(m.does_not_measure).strip() and m.assumptions
    footer = surface.method_limits_html(workflows.get(m.workflow).uispec())
    assert "What produced these numbers" in footer
    for text in (m.does_not_measure, *m.assumptions):
        assert html.escape(str(text).strip(), quote=True) in footer, text[:60]


@pytest.mark.req("ARCH-4-01")
@pytest.mark.parametrize("name", ALL)
def test_the_declared_statistic_is_read_off_the_table_not_recomputed(name, loaded):
    m = methods.get(name)
    _src, column, unit_col, _off = _INPUTS[name]
    df, traced = loaded[name]
    observed = [(u, i) for u, ev in traced.value.items() for i in ev.items
                if i.measured.state is MeasuredState.OBSERVED]
    assert observed, f"{name}: nothing observed on the fixture, so nothing was checked"
    for unit, item in observed:
        assert item.statistic_name == m.statistic_name
        reported = set(pd.to_numeric(df.loc[df[unit_col].astype(str) == unit, column]))
        assert item.statistic in reported, (unit, item.subject.display(), item.statistic)
    with pytest.raises(UnrecognisedResult):      # the column is required, not optional
        m.detect_unit(df.drop(columns=[column]))


@pytest.mark.req("ARCH-4-01", "CLAUDE-HR-01")
@pytest.mark.parametrize("name", ALL)
def test_a_row_naming_only_off_panel_genes_comes_back_without_a_number(name, loaded):
    """The panel-absence rule, through each method's own loader: a number in the file
    for a gene the panel never carried is not a measurement."""
    _df, traced = loaded[name]
    hits = [i for ev in traced.value.values() for i in ev.items
            if OFF in i.subject.display() or "SYNSET_OFFPANEL" in i.subject.display()]
    assert hits, f"{name}: the off-panel row vanished instead of being reported"
    for i in hits:
        assert i.statistic is None and not i.measured.carries_number
        if i.subject.display() == OFF:
            assert i.measured.state is MeasuredState.NEVER_MEASURED


@pytest.mark.req("ARCH-4-01", "CLAUDE-MS-02", "UI-2.3-02")
@pytest.mark.parametrize("name", ALL)
def test_only_a_tested_absence_argues_against_anything(name):
    m = methods.get(name)
    subj = (subject("gene_set", set=gene_set("SYNSET_ALPHA"), member=[gene("SYNG01")])
            if m.workflow == "enrichment-interpretation" else subject("gene", g=gene("SYNG01")))
    said = {s: m.absence_reads_as(subj, Measured(s, frame="panel:test"))
            for s in (MeasuredState.NEVER_MEASURED, MeasuredState.UNDERPOWERED,
                      MeasuredState.TESTED_ABSENT)}
    assert "argues against" in said[MeasuredState.TESTED_ABSENT]
    assert "argues nothing" in said[MeasuredState.NEVER_MEASURED]
    for s in (MeasuredState.NEVER_MEASURED, MeasuredState.UNDERPOWERED):
        assert "argues against" not in said[s], s
    assert len(set(said.values())) == 3, "two states read the same"


@pytest.mark.req("ARCH-4-01", "CLAUDE-TK-01")
@pytest.mark.parametrize("name", ALL)
def test_only_an_assignment_method_ends_in_a_band(name, loaded):
    m = methods.get(name)
    wf = workflows.get(m.workflow)
    _df, traced = loaded[name]
    for ev in traced.value.values():
        out = interpret(ev, wf, _REFERENCE[wf.task_kind]).value
        fields = {f.name for f in dataclasses.fields(out)}
        if wf.task_kind is TaskKind.ASSIGNMENT:
            assert isinstance(out, Assignment) and out.band
        else:
            assert not isinstance(out, Assignment)
            assert not {f for f in fields if "band" in f or "confidence" in f}, fields
