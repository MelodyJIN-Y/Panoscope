"""The enrichment half of the cutover: v2 records beside an untouched v1 view.

``stages/enrichment.py`` now runs BOTH engines on every build. The v1 engine still
writes ``interp/enrichment/`` and ``interp/enrichment.csv``; the v2 stack
(``platforms/tidy`` -> ``methods/jazzpanda_cgst`` -> ``workflows/enrichment`` ->
``core/record``) writes ``interp/records/characterisation/``. The two disagree about
``panel_hits``, deliberately and measurably, and that disagreement is why the engine
was not swapped -- so this file pins three things:

* the v1 bytes did not move (a swap would move a committed CSV and the golden hash);
* a v2 record exists for **every** unit and carries no band (no label space);
* the divergence report states the measured counts, so the cutover's go/no-go is a
  number on stdout rather than something a reader has to go and compute.

**Every unit, never one.** The checks below are over all nine demo clusters for the
reason ``test_pipeline_records`` gives: a one-unit version of a check like this
passed while five of nine were wrong.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E
from panoscope.pipeline.stages.enrichment import run_enrichment
from tests.conftest import DEMO_ID

# Same gate as tests/test_enrichment.py: the jazzPanda enrichment result is
# unpublished and gitignored, so it may be absent on a fresh clone / CI.
pytestmark = pytest.mark.skipif(
    not E._jz_enrichment_csv().exists(),
    reason=(
        f"jazzPanda enrichment result not present (gitignored): "
        f"{E._jz_enrichment_csv()}."
    ),
)

#: The divergence on the committed demo result, measured. v1 reads ``n_overlap``;
#: ``jazzpanda_cgst._in_frame`` counts ``genes_tested`` against the panel frame AND
#: the detectability frame, so the 49 of 280 genes with no usable spatial signal stop
#: counting toward a set's footprint. One undetectable gene in
#: HALLMARK_INTERFERON_GAMMA_RESPONSE (10/200 -> 9/200) crosses the 5% demotion
#: threshold and moves the band on three units.
EXPECTED_ROWS = "panel_hits differs on 35 of 70 (unit, set) rows"
EXPECTED_BANDS = (
    "the v1 band would move on 3 of 9 units "
    "(c3 Medium-High->Medium; c6 Very High->High; c7 High->Medium-High)"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, str]:
    """One build of the demo enrichment stage into a scratch tree, plus its stdout."""
    if cfg.DATASET_ID != DEMO_ID:
        pytest.skip(f"this file pins {DEMO_ID}; this process opened {cfg.DATASET_ID}")
    root = tmp_path_factory.mktemp("enrichment")
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_enrichment(root=root)
    return root / cfg.DATASET_ID / "interp", buf.getvalue()


def test_the_v1_view_did_not_move_a_byte(built) -> None:
    """The reason the engine was not swapped, asserted rather than asserted-about.

    ``enrichment.csv`` is committed and ``pathway_themes.json`` is golden-pinned
    through ``ClusterEnrichment.confidence``. Byte equality against the committed
    files is what makes "the v2 half is additive" a checkable claim.
    """
    interp, _ = built
    committed = cfg.current_study().interp
    assert (interp / "enrichment.csv").read_bytes() == \
        (committed / "enrichment.csv").read_bytes()
    for path in sorted((committed / "enrichment").glob("c*.json")):
        assert (interp / "enrichment" / path.name).read_bytes() == path.read_bytes(), \
            f"the v1 view moved for {path.stem}"


def test_a_record_lands_for_every_unit_and_carries_no_band(built) -> None:
    """Nine records, stamped characterisation, with nowhere for a band to be.

    ``Finding`` has no band field and ``jazzpanda_cgst.rubric`` raises, so the
    absence is structural upstream -- this asserts it survived serialisation, which
    is the layer where a stray ``confidence`` key would actually reach a reader.
    """
    interp, _ = built
    for unit in cfg.CLUSTER_ORDER:
        path = interp / "records" / "characterisation" / f"{unit}.json"
        assert path.is_file(), f"no v2 record for {unit}"
        record = json.loads(path.read_text())
        assert record["unit"] == unit
        assert record["task_kind"] == "characterisation"
        assert record["decisions"], f"{unit} carries a record with no reasoning"
        flat = json.dumps(record)
        assert '"band"' not in flat and '"confidence"' not in flat


def test_the_record_does_not_claim_the_programs_were_never_measured(built) -> None:
    """The panel-absence rule, at the one place this stage could have broken it.

    ``ReadingRecord.entities_outside_frame`` means *never measured, so not evidence*.
    The GENE_SET capability frame comes from ``jazzpanda_cgst.scored_frame``, which
    is not on the ``MethodModule`` protocol and so is easy to leave out; without it
    ``read_tidy`` has no frame for the kind and lists **every** Hallmark set as
    outside -- a record asserting that no program in the file was ever scored.
    """
    interp, _ = built
    for unit in cfg.CLUSTER_ORDER:
        record = json.loads(
            (interp / "records" / "characterisation" / f"{unit}.json").read_text())
        assert record["reading"]["entities_outside_frame"] == [], \
            f"{unit}: the record says its own scored sets were never measured"
        assert any(f["kind"] == "gene_set" for f in record["reading"]["frames"]), \
            f"{unit}: no gene-set capability frame in the reading"


def test_the_divergence_between_the_engines_is_printed_with_its_counts(built) -> None:
    """The go/no-go evidence, on stdout, every build.

    Pinned to exact counts on purpose. If these move, either the capability frames
    changed or the committed result did, and both are interpretation changes that
    must not pass quietly -- the same failure shape as a band moving under a silent
    row re-ordering.
    """
    _, printed = built
    assert EXPECTED_ROWS in printed, printed
    assert EXPECTED_BANDS in printed, printed
    assert "0 of those cross the panel-hit floor" in printed, printed
    assert "The v1 view on disk is unchanged" in printed
