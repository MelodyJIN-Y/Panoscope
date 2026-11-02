"""The enrichment gate, on a study whose input is actually in the repository.

``tests/test_pipeline_enrichment_records.py`` pins the enrichment half of the
cutover on the demo study, and it cannot run anywhere but this laptop: its input is
``data/jazzpanda/hbc_sp1_top10_hallmark_test_statistic.csv``, excluded by
``.gitignore:53``. That file stays out -- its rows are MSigDB Hallmark sets scored on
10x's Xenium breast demo, and CLAUDE.md is explicit that restrictively-licensed data
is fetched and never bundled. So on a fresh clone, which is to say on CI and for
every reader of this open-source repo, that whole module skips.

That skip was not a normal one. ``tests/golden_projection.py`` feeds ``panel_hits``
from the committed enrichment records straight back in under
``E.PANEL_HITS_IN_FRAME``, so the golden hash is blind to a change in the panel or
the detectability file -- the comment there names ``test_pipeline_enrichment_records``
as the backstop for exactly that, and the backstop did not run. The
divergence-is-zero check between the two enrichment engines went with it.

This module is that backstop, on the synthetic fixture, which exists so a gate can
run without licensed data. It runs unconditionally: every byte it reads is committed
and made up. Three options were weighed --

* **make the skip a hard failure on CI.** Fails the build without checking anything;
  the coverage is still missing, only louder.
* **derive the input instead of reading it.** The gate is about an engine reading a
  file; a derived input means the two engines are compared on something neither of
  them will ever be handed.
* **a committed fixture exercising the same path.** Chosen. Same
  ``run_enrichment`` -> ``agent/enrichment`` + ``platforms/tidy`` ->
  ``methods/jazzpanda_cgst`` -> ``core/record`` call, same resolver
  (``_jz_enrichment_csv`` returns ``<study>/inputs/enrichment.csv`` for any study
  that is not the bundled demo), one study over.

What this cannot pin, and the demo module still can: byte-equality of the committed
v1 view (``interp/enrichment.csv``, ``interp/enrichment/c*.json``), which the fixture
does not ship. This covers the frame arithmetic, the record shape and the engine
agreement; those are what a panel or detectability change moves.
"""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E
from panoscope.data.study import Study
from panoscope.methods.jazzpanda_cgst import _in_frame, _split
from panoscope.pipeline.stages.enrichment import run_enrichment

FIXTURE = "synthetic_multisample_fixture"

#: Units and (unit, set) rows in the committed fixture result. Spelled out so a
#: fixture quietly shrunk to nothing cannot make "0 of 0 rows differ" read as a pass.
FIXTURE_UNITS = ("c1", "c2", "c3")
FIXTURE_ROWS = 12


def _input_path() -> Path:
    return Study.open(FIXTURE).inputs / "enrichment.csv"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, str]:
    """One enrichment build of the fixture study into a scratch tree, plus its stdout.

    Never the repo's own tree: a built fixture is what broke
    ``test_verdict_retire``, and this module is not going to reintroduce that
    coupling one directory over.
    """
    root = tmp_path_factory.mktemp("fixture_enrichment")
    buf = io.StringIO()
    with cfg.use_study(FIXTURE):
        with redirect_stdout(buf):
            run_enrichment(root=root)
        interp = root / cfg.DATASET_ID / "interp"
    return interp, buf.getvalue()


def test_the_input_this_gate_reads_is_committed() -> None:
    """The guard for the whole module.

    Everything below is evidence only if its input is in the repository. If this
    file ever becomes gitignored, generated or fetched, the module silently becomes
    the thing it was written to replace.
    """
    path = _input_path()
    assert path.is_file(), f"the fixture's enrichment result is missing: {path}"
    assert path.resolve().is_relative_to(Study.open(FIXTURE).dir.resolve())
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == FIXTURE_ROWS
    assert sorted({r["cluster"] for r in rows}) == list(FIXTURE_UNITS)


def test_the_fixtures_own_numbers_make_the_frame_check_bite() -> None:
    """The fixture's ``n_overlap`` disagrees with the frame count -- on purpose.

    This is what stops the zero in :func:`test_the_two_engines_agree_about_the_measurement`
    from being free. The demo result's ``n_overlap`` differs from the frame count on
    35 of 70 rows because 49 of its 280 panel genes yield no usable spatial signal;
    the fixture reproduces that shape with ``SYNG10``/``SYNG11``/``SYNG12``, which
    are on the panel and carry ``top_cluster = NoSig``. If both counts agreed, an
    engine that went back to reading ``n_overlap`` would still report zero
    divergence and this module would pass while checking nothing.
    """
    with cfg.use_study(FIXTURE):
        world = E._gene_world()
        with _input_path().open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        naive = [int(r["n_overlap"]) for r in rows]
        framed = [_in_frame(_split(r["genes_tested"]), world) for r in rows]

    differing = [i for i, (n, k) in enumerate(zip(naive, framed)) if n != k]
    assert differing, (
        "every row's n_overlap equals its frame count, so this fixture cannot tell "
        "the two counting rules apart"
    )
    crossing = [
        i for i in differing
        if (naive[i] >= cfg.MIN_PANEL_HITS) != (framed[i] >= cfg.MIN_PANEL_HITS)
    ]
    assert crossing, (
        "no row changes tier between the two counting rules, so a revert to "
        "n_overlap would not move a single reported program here"
    )


def test_a_record_lands_for_every_unit_and_carries_no_band(built) -> None:
    """Three records, stamped characterisation, with nowhere for a band to be.

    The demo module's wording, on the study that can run it. There is no label space
    over a program, so a ``band``/``confidence`` key reaching a reader is the
    category error ``jazzpanda_cgst.rubric`` refuses upstream -- asserted at the
    serialisation layer, which is where a stray key would actually surface.
    """
    interp, _ = built
    for unit in FIXTURE_UNITS:
        path = interp / "records" / "characterisation" / f"{unit}.json"
        assert path.is_file(), f"no v2 record for {unit}"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["unit"] == unit
        assert record["task_kind"] == "characterisation"
        assert record["decisions"], f"{unit} carries a record with no reasoning"
        flat = json.dumps(record)
        assert '"band"' not in flat and '"confidence"' not in flat


def test_the_record_does_not_claim_the_programs_were_never_measured(built) -> None:
    """The panel-absence rule, at the one place this stage could have broken it.

    ``entities_outside_frame`` means *never measured, so not evidence*. The GENE_SET
    frame comes from ``jazzpanda_cgst.scored_frame``, which is not on the
    ``MethodModule`` protocol and so is easy to leave out; without it ``read_tidy``
    has no frame for the kind and lists every scored set as outside -- a record
    asserting that nothing in the file was ever measured.
    """
    interp, _ = built
    for unit in FIXTURE_UNITS:
        record = json.loads(
            (interp / "records" / "characterisation" / f"{unit}.json").read_text(
                encoding="utf-8"))
        assert record["reading"]["entities_outside_frame"] == [], \
            f"{unit}: the record says its own scored sets were never measured"
        assert any(f["kind"] == "gene_set" for f in record["reading"]["frames"]), \
            f"{unit}: no gene-set capability frame in the reading"


def test_the_two_engines_agree_about_the_measurement(built) -> None:
    """Zero divergence, on stdout, every build -- the assertion the fresh clone lost.

    ``agent/enrichment`` and ``methods/jazzpanda_cgst`` read one file through two
    code paths, which is how two counts of the same thing drift apart without anyone
    noticing. Pinned to the exact row count as well as the zero: a disagreement that
    reappeared on a fixture silently shrunk to one row would otherwise pass.

    Only the panel-hit sentence is pinned. The leading-edge line the stage prints
    beside it is a measurement of the panel against the tissue and is not expected
    to be zero.
    """
    _, printed = built
    assert f"panel_hits differs on 0 of {FIXTURE_ROWS} (unit, set) rows" in printed, \
        printed
    assert "0 of those cross the panel-hit floor" in printed, printed
    assert "do not agree on the unit set" not in printed, printed
