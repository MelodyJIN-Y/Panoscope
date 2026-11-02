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
from panoscope.methods.jazzpanda_cgst import _in_frame, _split

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
    assert header == E.ENRICHMENT_CSV_COLUMNS
    assert "cluster_confidence" not in header
    assert rows, "the committed enrichment CSV has no rows"
    for row in rows:
        counts = (int(row["cluster_n_supported"]), int(row["cluster_n_suggestive"]),
                  int(row["cluster_n_untestable"]))
        assert sum(counts) > 0, row["cluster"]
        assert row["cluster_verify"] in ("TRUE", "FALSE")
