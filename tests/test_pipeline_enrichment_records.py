"""The enrichment half of the cutover: v2 records beside an untouched v1 view.

``stages/enrichment.py`` now runs BOTH engines on every build. The v1 engine still
writes ``interp/enrichment/`` and ``interp/enrichment.csv``; the v2 stack
(``platforms/tidy`` -> ``methods/jazzpanda_cgst`` -> ``workflows/enrichment`` ->
``core/record``) writes ``interp/records/characterisation/``. The two used to disagree
about ``panel_hits`` -- v1 read the result's own ``n_overlap``, v2 counted the
capability frames -- and they no longer do, because v1 was moved onto the frame count.
So this file pins three things:

* the v1 view on disk matches what the engine produces today, byte for byte;
* a v2 record exists for **every** unit and carries no band (no label space);
* the divergence report reads **zero**, which is what keeps two code paths over one
  file from drifting apart again without anyone noticing.

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

#: The divergence on the committed demo result, now CLOSED. v1 used to read
#: ``n_overlap``; it now counts ``genes_tested`` against every gene frame through the
#: same ``jazzpanda_cgst._in_frame`` the v2 method uses, so the 49 of 280 genes with
#: no usable spatial signal no longer count toward a set's footprint in either engine.
#: Zero is therefore the only acceptable reading, and it is pinned as a string so a
#: reintroduced disagreement cannot pass as a smaller number.
EXPECTED_ROWS = "panel_hits differs on 0 of 70 (unit, set) rows"


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


def test_the_two_engines_agree_about_the_measurement(built) -> None:
    """Zero divergence, on stdout, every build.

    This used to measure a real gap (35 of 70 rows, the band moving on c3/c6/c7) and
    was the go/no-go evidence for a cutover. The cutover happened in the arithmetic:
    ``agent/enrichment`` counts the capability frames now. So the check inverted --
    it is no longer measuring how far apart two engines are, it is what stops them
    drifting apart again, which two code paths reading one file silently will.

    Pinned to the exact sentence on purpose: a reintroduced disagreement must not be
    able to pass as a smaller number nobody read.
    """
    _, printed = built
    assert EXPECTED_ROWS in printed, printed
    assert "0 of those cross the panel-hit floor" in printed, printed
    assert "Zero is the expected reading" in printed
