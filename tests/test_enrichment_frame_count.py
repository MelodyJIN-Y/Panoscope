"""``panel_hits`` is a capability-frame count, not the result's word for itself.

The enrichment engine used to read ``n_overlap`` — jazzPanda's own count of a set's
genes on the panel. That credits a set with genes the measurement never usably saw:
49 of the demo's 280 panel genes (17.5%) yield no usable spatial signal, and a set
whose footprint includes them was never measurable to that depth however the panel
was designed. It is the panel-absence rule broken in the arithmetic rather than in a
sentence, which is worse, because every number downstream still looks well formed.

``agent/enrichment._load_jazzpanda_rows`` now recounts every row's ``genes_tested``
against EVERY gene frame the study declares, through the same
``methods/jazzpanda_cgst._in_frame`` the v2 method uses. This file pins that:

* on all nine demo units, ``panel_hits`` equals the frame count and **differs** from
  ``n_overlap`` on 35 of 70 rows — the second half matters, because a test that only
  checked "panel_hits == the frame count" would also pass if the frame count happened
  to equal ``n_overlap`` everywhere, and would then be proving nothing;
* the LEADING EDGE goes through the same frames, which the first version of this
  change forgot: ``panel_hits`` was recounted and ``genes_selected`` was not, so a
  record could say "3 of 36 genes usably measured" and then name four drivers, two
  of them rejected by the detectability frame. Those genes are also what
  ``ui/enrichment_spatial`` plots;
* the loud failures: a result with no ``genes_tested``, and a study with no gene
  frame, both raise rather than quietly counting zero;
* the committed artifacts carry no band — on every unit, in the JSON and in the CSV.

**Every unit, never one.** The checks are over all nine clusters: the values that
moved under this change moved on three of them, so a one-unit version of any of these
would have had a two-in-three chance of seeing nothing.
"""

from __future__ import annotations

import csv
import json

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E
from panoscope.core.export import PROFILES
from panoscope.methods.jazzpanda_cgst import _in_frame, _split
from panoscope.types.outputs import TaskKind

# Same gate as tests/test_enrichment.py: the jazzPanda enrichment result is
# unpublished and gitignored, so it may be absent on a fresh clone / CI.
_needs_result = pytest.mark.skipif(
    not E._jz_enrichment_csv().exists(),
    reason=f"jazzPanda enrichment result not present (gitignored): {E._jz_enrichment_csv()}.",
)

#: Measured on the committed demo result. Half the rows move, which is what makes the
#: equality check below evidence rather than a tautology.
EXPECTED_DIFFERING_ROWS = 35
EXPECTED_TOTAL_ROWS = 70

#: Rows whose ``genes_selected`` holds at least one gene the frames reject. Same
#: reasoning as above: without this number the leading-edge check below would also
#: pass on an engine that filtered nothing, in a study where every selected gene
#: happened to be detectable.
EXPECTED_ROWS_WITH_REJECTED_DRIVERS = 33


def _raw_rows() -> list[dict]:
    """The result CSV as the file has it, with ``n_overlap`` untouched."""
    with open(E._jz_enrichment_csv(), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@_needs_result
def test_panel_hits_is_the_frame_count_and_not_n_overlap_on_every_unit():
    """Both halves, on all nine units: it equals the frame count AND it moved.

    The second assertion is the one that bites. Without it this test would pass on an
    engine that still read ``n_overlap``, in any study where the panel happened to be
    fully detectable -- which is every study with no detectability file, and was the
    state of the code this change replaced.
    """
    world = E._gene_world()
    rows = E._load_jazzpanda_rows()
    raw = {(r["cluster"], r["gene_set"]): r for r in _raw_rows()}
    assert len(rows) == EXPECTED_TOTAL_ROWS

    differing = 0
    seen_units = set()
    for row in rows:
        key = (row["cluster"], row["gene_set"])
        expected = _in_frame(_split(row["genes_tested"]), world)
        assert row[E.PANEL_HITS_IN_FRAME] == expected, key
        # Never more than the method's own count: the frames can only subtract.
        n_overlap = int(float(raw[key]["n_overlap"]))
        assert expected <= n_overlap, f"{key}: frame count exceeds n_overlap"
        differing += expected != n_overlap
        seen_units.add(row["cluster"])

    assert seen_units == set(cfg.CLUSTER_ORDER), seen_units
    assert differing == EXPECTED_DIFFERING_ROWS, (
        f"{differing} of {len(rows)} rows differ from n_overlap; expected "
        f"{EXPECTED_DIFFERING_ROWS}. If the panel or detectability file changed, that "
        "is a real interpretation change, not a number to retune here."
    )


@_needs_result
def test_the_built_evidence_carries_the_frame_count_through_to_every_pathway():
    """End to end on nine units: the number on the record is the frame count.

    Checking the row dict is not enough -- ``_pathway_from_jz_row`` could read the
    wrong key, and ``panel_coverage`` and the caveat sentence are derived from
    whatever it read.
    """
    world = E._gene_world()
    by_key = {(r["cluster"], r["gene_set"]): r for r in E._load_jazzpanda_rows()}
    for ce in E.all_enrichments():
        assert ce.all_tested, ce.cluster
        for p in ce.all_tested:
            row = by_key[(ce.cluster, p.gene_set)]
            assert p.panel_hits == _in_frame(_split(row["genes_tested"]), world)
            assert p.panel_coverage == p.panel_hits / p.set_size_full
            assert f"{p.panel_hits} of {p.set_size_full}" in p.panel_scope_caveat


@_needs_result
def test_a_result_without_genes_tested_raises_rather_than_counting_zero(tmp_path, monkeypatch):
    """No tested-gene column -> refuse. The alternative is a study-wide false zero.

    ``_in_frame`` over an empty symbol list returns 0, so a silent path here would
    mark every program in the study untestable -- a plausible-looking total refusal
    with a false reason behind it, which is the shape of every serious bug this work
    has turned up.
    """
    src = _raw_rows()
    stripped = tmp_path / "enrichment.csv"
    fields = [k for k in src[0] if k != "genes_tested"]
    with open(stripped, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in src:
            w.writerow({k: r[k] for k in fields})

    monkeypatch.setattr(E, "_jz_enrichment_csv", lambda: stripped)
    E._load_jazzpanda_rows.cache_clear()
    with pytest.raises(KeyError, match="genes_tested"):
        E._load_jazzpanda_rows()
    E._load_jazzpanda_rows.cache_clear()


def test_a_study_with_no_gene_frame_raises_rather_than_counting_zero(monkeypatch):
    """No panel file -> refuse, naming the files to fix.

    ``MeasuredWorld.for_kind(GENE)`` empty means "nothing declares what could have
    been measured". ``_in_frame`` answers 0 for every set, which reads as "the panel
    covers none of this collection" -- a measurement claim nobody made.
    """
    from panoscope.platforms.xenium import XeniumAdapter

    monkeypatch.setattr(XeniumAdapter, "frames", lambda self, store: ())
    E._gene_world.cache_clear()
    with pytest.raises(ValueError, match="no gene capability frame"):
        E._gene_world()
    E._gene_world.cache_clear()


def test_the_panel_size_in_the_caveat_follows_the_open_study():
    """The denominator of the panel-coverage sentence is per study, not per process.

    ``_panel_size`` was an ``lru_cache(maxsize=1)`` over a zero-argument reader, so
    one slot was shared by every study: the first one opened supplied its panel size
    to every caveat sentence any later study wrote. CLAUDE.md names this directly --
    panel size is derived from the file, never asserted -- and the kidney study's
    panel is a different size from the demo's.
    """
    demo = E._panel_size()
    assert demo == 280
    with cfg.use_study("demo_mini"):
        assert E._panel_size() != demo
    assert E._panel_size() == demo, "switching back must restore the demo's panel"


# --------------------------------------------------------------------------- #
# The committed artifacts carry no band. Checked against the FILES, because the file
# is where the deleted band survived after the page stopped drawing it.
# --------------------------------------------------------------------------- #
_BAND_KEYS = ("confidence", "confidence_score", "demotions")


def test_no_committed_enrichment_record_carries_a_band():
    study = cfg.current_study()
    records = sorted((study.interp / "enrichment").glob("c*.json"))
    assert len(records) == len(cfg.CLUSTER_ORDER), records
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in _BAND_KEYS:
            assert key not in record, f"{path.name} still carries {key!r}"
        # ...and the replacement is reconstructible from what IS there.
        assert record["all_tested"], path.name


def test_the_committed_enrichment_csv_has_counts_where_the_band_was():
    study = cfg.current_study()
    with open(study.interp / "enrichment.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    assert header == PROFILES[TaskKind.CHARACTERISATION].names
    assert "cluster_confidence" not in header
    assert rows, "the committed enrichment CSV has no rows"
    for row in rows:
        counts = (int(row["cluster_n_supported"]), int(row["cluster_n_suggestive"]),
                  int(row["cluster_n_untestable"]))
        assert sum(counts) > 0, row["cluster"]
        assert row["cluster_verify"] in ("TRUE", "FALSE")


@_needs_result
def test_the_leading_edge_names_only_genes_the_frames_admit_on_every_unit():
    """The drivers are frame-filtered, and filtering them MOVED 33 of 70 rows.

    Three things, because the bug was one record telling two stories: the named
    genes are the admitted ones, the count matches the list, and the count can no
    longer exceed the set's measured footprint -- ``n_leading_edge <= panel_hits``
    is the invariant ``tests/test_panel_coverage`` already assumed in prose and
    that an unfiltered leading edge quietly broke on 12 rows.
    """
    world = E._gene_world()
    by_key = {(r["cluster"], r["gene_set"]): r for r in E._load_jazzpanda_rows()}
    rejecting = checked = 0
    for ce in E.all_enrichments():
        for p in ce.all_tested:
            selected = _split(by_key[(ce.cluster, p.gene_set)]["genes_selected"])
            admitted = tuple(g for g in selected if _in_frame((g,), world))
            assert p.leading_edge == admitted, (ce.cluster, p.gene_set)
            assert p.n_leading_edge == len(p.leading_edge)
            assert p.n_leading_edge <= p.panel_hits, (ce.cluster, p.gene_set)
            checked += 1
            rejecting += len(selected) != len(admitted)

    assert checked == EXPECTED_TOTAL_ROWS
    assert rejecting == EXPECTED_ROWS_WITH_REJECTED_DRIVERS, (
        f"{rejecting} of {checked} rows select a gene the frames reject; expected "
        f"{EXPECTED_ROWS_WITH_REJECTED_DRIVERS}. If the panel or detectability file "
        "changed, that is a real interpretation change, not a number to retune here."
    )


@_needs_result
def test_the_build_says_out_loud_how_many_drivers_the_frames_rejected(tmp_path, capsys):
    """The rejected count exists in exactly one place, and the build prints it.

    It cannot live on ``PathwayEvidence``: a record rebuilt from disk carries the
    admitted genes only, so a field derived from the raw selection would come back
    as zero and read as "nothing was rejected". The build still has the raw rows, so
    this is where the number is said -- and silence here would mean "nobody looked",
    which is the same failure the panel-hit line exists to prevent.
    """
    from panoscope.pipeline.stages.enrichment import run_enrichment

    world = E._gene_world()
    expected_genes = expected_rows = 0
    for row in E._load_jazzpanda_rows():
        selected = _split(row["genes_selected"])
        rejected = sum(1 for g in selected if not _in_frame((g,), world))
        expected_genes += rejected
        expected_rows += bool(rejected)

    run_enrichment(cfg.DATASET_ID, tmp_path)      # a throwaway tree, not the repo's
    out = capsys.readouterr().out
    assert (f"leading edge, capability frames: {expected_genes} of ") in out, out
    assert f"rejected across {expected_rows} of " in out, out
    assert expected_genes and expected_rows, "nothing rejected: the check is vacuous"


# --------------------------------------------------------------------------- #
# The collection a set came from is read off the set, never asserted.
# --------------------------------------------------------------------------- #
def test_the_collection_is_derived_from_the_set_and_unknown_when_it_cannot_be():
    """``gene_set_collection`` was the constant ``"MSigDB_Hallmark"``, every study.

    On the demo that constant is true. On ``synthetic_multisample_fixture`` it wrote
    a real, restrictively-licensed collection's name as the provenance of four
    invented sets -- false provenance under the confident floor, and the licensing
    rule ("fetch restrictively-licensed data, never bundle it") broken in the one
    study that exists because nothing licensed is bundled.

    Both halves are pinned. Checking only that HALLMARK_ still resolves would pass
    on the asserted constant; checking only that SYNSET_ does not would pass on a
    function that returned ``unknown`` for everything.
    """
    assert E._collection_for("HALLMARK_G2M_CHECKPOINT") == "MSigDB_Hallmark"
    assert E._collection_for("hallmark_g2m_checkpoint") == "MSigDB_Hallmark"
    assert E._collection_for("SYNSET_ALPHA") == E.UNKNOWN_COLLECTION
    assert E._collection_for("") == E.UNKNOWN_COLLECTION
    assert E.UNKNOWN_COLLECTION == "unknown"
    # and nothing anywhere still hands out the old constant
    assert not hasattr(E, "_JZ_COLLECTION")


def test_the_synthetic_fixtures_pathways_name_no_real_collection():
    """End to end on the study the defect was committed in. Ungated: the fixture's
    ``inputs/enrichment.csv`` is committed, unlike the demo's."""
    from panoscope.data.study import Study

    fixture = "synthetic_multisample_fixture"
    if not Study.open(fixture).exists:
        pytest.skip(f"study {fixture!r} is not on disk")
    with cfg.use_study(fixture):
        built = [E._pathway_from_jz_row(r) for r in E._load_jazzpanda_rows()]
    assert built, "the fixture has enrichment rows; this must not pass vacuously"
    assert {p.gene_set_collection for p in built} == {E.UNKNOWN_COLLECTION}
    assert all(p.gene_set.startswith("SYNSET_") for p in built)


@_needs_result
def test_the_demos_pathways_still_name_the_collection_they_really_came_from():
    """The other half: `unknown` must not become the answer for everything."""
    built = [E._pathway_from_jz_row(r) for r in E._load_jazzpanda_rows()]
    assert {p.gene_set_collection for p in built} == {"MSigDB_Hallmark"}
