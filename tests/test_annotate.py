"""The annotate stage: parse the skill's JSON, ground off-panel, generate offline.

The live agent call is mocked, so these stay deterministic and network-free; the
run writes only under a pytest tmp dir (never the bundled demo tree).

**The invariant these exist for.** This stage writes the canonical-marker slot (always
empty: canonical lists are reference data, never model-written), and
``pipeline/records.py`` hands that set to ``core/verdict.assess`` as the list deciding
which marker may drive the confidence band. So the tests below are not about parsing:
they are about what reaches that slot when the model fails. The failure path used to
write the cluster's own top five OBSERVED markers there, which made every driver
canonical by construction and turned the band into the measurement agreeing with
itself. ``test_the_old_promotion_would_have_inflated_the_band_on_the_demo`` measures
what that was worth in bands, over all nine demo clusters, so the fix is a recorded
correction rather than a claim.
"""
from __future__ import annotations

import inspect
import json

import pytest

from panoscope.agent import config as cfg
from panoscope.core import verdict as core_verdict
from panoscope.pipeline.stages import annotate
from tests.test_core_verdict import _inputs

_GOOD = ('{"cell_type":"Tumor","cell_type_short":"Tum_Epi","category":"Epithelial",'
         '"lineage":"Epithelial","canonical_markers":["ERBB2","EPCAM"]}')


@pytest.fixture(autouse=True)
def _a_model_key(monkeypatch):
    """The stage asks only with a key; these mock the call, so they need one too."""
    monkeypatch.setattr(annotate.agent_loop, "has_model_key", lambda: True)


class _Resp:
    def __init__(self, text: str):
        self.text = text


# --------------------------------------------------------------------------- #
# Parsing: a label slot takes a label, or nothing
# --------------------------------------------------------------------------- #
def test_parse_accepts_valid_and_rejects_bad():
    obj = annotate._parse(_GOOD)
    assert obj and obj["cell_type"] == "Tumor"
    assert annotate._parse("no json here") is None
    assert annotate._parse('{"cell_type":"Tumor"}') is None  # missing required fields


@pytest.mark.parametrize("payload", [
    '{"cell_type":null,"cell_type_short":"T","category":"E","lineage":"E"}',
    '{"cell_type":"","cell_type_short":"T","category":"E","lineage":"E"}',
    '{"cell_type":"   ","cell_type_short":"T","category":"E","lineage":"E"}',
    '{"cell_type":123,"cell_type_short":"T","category":"E","lineage":"E"}',
])
def test_parse_rejects_a_present_but_unusable_label(payload):
    """``str(None)`` is the string "None", and it used to be written as a cell type.

    The key was present, so the membership check passed, and the biologist was shown a
    cluster labelled "None" -- a label nothing said. No label is a better record than
    a fabricated one, so these leave the cluster not yet called.
    """
    assert annotate._parse(payload) is None


# --------------------------------------------------------------------------- #
# The canonical slot: reference data only
# --------------------------------------------------------------------------- #
def test_the_models_canonical_markers_never_reach_the_record():
    """§15: canonical lists are reference data. A proposal names a label, nothing more."""
    rec = annotate._record("c1", annotate._parse(_GOOD))
    assert rec["cell_type"] == "Tumor"
    assert rec["canonical_markers"] == [] and rec["offpanel_canonical"] == []


def test_record_cannot_see_the_measurement():
    """Structural, not behavioural: the markers are not a parameter of this function.

    A later edit can re-introduce the substitution only by first re-introducing the
    argument, which this fails on -- the same shape as ``verify`` being a field rather
    than a parameter in ``core/verdict``.
    """
    params = set(inspect.signature(annotate._record).parameters)
    assert params == {"cluster", "obj"}, params


# --------------------------------------------------------------------------- #
# What it is worth in bands, on all nine demo clusters
# --------------------------------------------------------------------------- #
def _band_with(cluster: str, canonical: frozenset[str]):
    return core_verdict.assess(_inputs(cluster)[0], canonical).value


@pytest.mark.parametrize("cluster", list(cfg.CLUSTER_ORDER))
def test_an_unannotated_cluster_lands_on_the_no_driver_rule(cluster):
    """Nine units, not one: empty canonical -> Low, verify, and the named rule."""
    v = _band_with(cluster, frozenset())
    assert v.band == core_verdict.LOW_BAND
    assert v.verify is True
    assert "band:no_canonical_driver" in v.demotions, (v.basis, v.demotions)
    assert v.driver is None


@pytest.mark.parametrize("cluster", list(cfg.CLUSTER_ORDER))
def test_the_old_promotion_would_have_inflated_the_band_on_the_demo(cluster):
    """The correction, measured. Not one cluster gets to keep the inflated band."""
    top5 = frozenset(g.upper() for g, _, _ in annotate._cluster_markers(cluster)[:5])
    inflated = _band_with(cluster, top5)
    honest = _band_with(cluster, frozenset())
    assert honest.band == core_verdict.LOW_BAND
    assert core_verdict.BANDS.index(inflated.band) <= core_verdict.BANDS.index(honest.band)
    # and it is not a vacuous comparison: on this study the promotion really did lift
    # the band off the floor, or clear verify, or both.
    assert (inflated.band, inflated.verify) != (honest.band, honest.verify)


def test_the_promotion_was_self_corroborating_on_every_demo_cluster():
    """Why it inflated: the promoted set IS the top of the measurement, so a driver
    exists by construction on all nine clusters, whatever the biology says."""
    for cluster in cfg.CLUSTER_ORDER:
        top5 = frozenset(g.upper() for g, _, _ in annotate._cluster_markers(cluster)[:5])
        v = _band_with(cluster, top5)
        assert v.driver is not None, cluster
        assert v.driver.subject.entities()[0].id.upper() in top5, cluster
    assert len(cfg.CLUSTER_ORDER) == 9


# --------------------------------------------------------------------------- #
# Resume: a stored record is never silently replaced
# --------------------------------------------------------------------------- #
def test_load_existing_refuses_an_unreadable_annotation(tmp_path):
    out = tmp_path / "annotation.json"
    out.write_text('{"clusters": {"c1": ', encoding="utf-8")  # truncated write
    with pytest.raises(json.JSONDecodeError):
        annotate._load_existing(out)
    out.write_text('["not", "an", "annotation"]', encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        annotate._load_existing(out)
    assert out.read_text(encoding="utf-8") == '["not", "an", "annotation"]'


def test_a_damaged_annotation_stops_the_stage_instead_of_overwriting_it(monkeypatch, tmp_path):
    from panoscope.pipeline import paths

    out = paths.interp_dir(cfg.DATASET_ID, tmp_path) / "annotation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{oops", encoding="utf-8")
    monkeypatch.setattr(annotate.agent_loop, "chat", lambda *a, **k: _Resp(_GOOD))
    with pytest.raises(json.JSONDecodeError):
        annotate.run_annotate(root=tmp_path)
    assert out.read_text(encoding="utf-8") == "{oops"  # untouched


# --------------------------------------------------------------------------- #
# End to end, offline
# --------------------------------------------------------------------------- #
def test_run_annotate_generates_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(annotate.agent_loop, "chat", lambda *a, **k: _Resp(_GOOD))
    result = annotate.run_annotate(root=tmp_path, force=True)
    assert result and all(v["cell_type"] == "Tumor" for v in result.values())
    assert len(result) == len(cfg.CLUSTER_ORDER)


@pytest.mark.parametrize("behaviour", ["raises", "unparseable"])
def test_a_failing_model_annotates_nothing_rather_than_something(
    behaviour, monkeypatch, tmp_path, capsys
):
    """All nine clusters, both failure modes: no record, so each stays not yet called."""
    def _chat(*a, **k):
        if behaviour == "raises":
            raise RuntimeError("provider down")
        return _Resp("I think this is a tumour cluster.")

    monkeypatch.setattr(annotate.agent_loop, "chat", _chat)
    result = annotate.run_annotate(root=tmp_path, force=True)
    assert result == {}
    assert "NOT YET CALLED" in capsys.readouterr().out  # loud, not silent
