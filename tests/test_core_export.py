"""Invariants for the CSV export profiles.

The single thing this module can get wrong is a byte. The 11-column verdict CSV is
pinned by ``tests/test_golden.py``, restated in ``skills/marker-interpretation/
SKILL.md`` Output 2 and read by downstream R; the 14-column enrichment CSV is a
committed artifact. So the central tests here do not inspect the output and judge it
-- they feed the same rows through ``agent.verdict.to_csv`` / ``agent.enrichment.to_csv``
and through the profile and assert the two strings are ``==``. "Looks the same" is
what lets a trailing ``\\r`` or a ``0.70`` for a ``0.7`` through.

Both halves are tested: the real demo study (which is what the golden hash covers)
and synthetic rows chosen for the cases the real data happens not to contain -- a
``None`` q_value, a band anchor whose ``str()`` and ``.2f`` differ, and a notes field
holding a comma, a quote and a newline, which is where two csv spellings would
diverge if they ever did.

Rows are duck-typed stubs rather than real ``ClusterVerdict``s. ``core/`` does not
import ``agent/``, and the profile getters only ever do attribute access; building
the real frozen dataclass would drag in the loaders and test the fixtures instead.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from panoscope.agent import enrichment as E
from panoscope.agent import verdict as V
from panoscope.core.export import (
    ASSIGNMENT_COLUMNS, CHARACTERISATION_COLUMNS, PROFILES, NoExportProfile, to_csv,
)
from panoscope.types.outputs import TaskKind
from panoscope.types.trace import Traced


def _pairs(enrichments) -> list[tuple[object, object]]:
    """The characterisation row shape: one row per surfaced pathway, carrying its
    unit. Enriched then suggestive, which is the order the committed file is in."""
    return [(ce, p) for ce in enrichments for p in (*ce.enriched, *ce.suggestive)]


# --------------------------------------------------------------------------- #
# Byte-identity against the two hand-written exporters -- the whole contract
# --------------------------------------------------------------------------- #
def test_assignment_profile_is_byte_identical_on_the_demo_study():
    verdicts = V.all_verdicts()
    assert to_csv(TaskKind.ASSIGNMENT, verdicts).value == V.to_csv(verdicts)


#: The CHARACTERISATION half of this module reads the jazzPanda enrichment input,
#: which is gitignored on purpose -- its rows are MSigDB Hallmark sets scored on
#: 10x's Xenium demo, and CLAUDE.md is explicit that restrictively-licensed data is
#: fetched and never bundled. So on a fresh clone, which is to say on CI and for
#: every reader of this repo, that input is absent.
#:
#: Guarded per-test rather than with a module-level ``pytestmark``: the ASSIGNMENT
#: half and the synthetic-row tests below need no such file, and skipping those too
#: would retire most of this module's coverage on exactly the machine the README
#: asks a reviewer to verify on. Four sibling modules already guard this way
#: (test_enrichment.py:36, test_enrichment_frame_count.py:45,
#: test_pipeline_enrichment_records.py:48, test_pipeline.py:36); this module was the
#: one that did not, which is what made ``pytest -m "not live"`` red on a fresh
#: clone while staying green on a machine that happens to hold the file.
_needs_enrichment_input = pytest.mark.skipif(
    not E._jz_enrichment_csv().exists(),
    reason=(
        f"jazzPanda enrichment result not present (gitignored): "
        f"{E._jz_enrichment_csv()}. The ASSIGNMENT half of the byte-identity "
        f"contract still runs here, and the enrichment CSV itself stays pinned by "
        f"tests/test_golden.py against the committed records."
    ),
)


@_needs_enrichment_input
def test_characterisation_profile_is_byte_identical_on_the_demo_study():
    enrichments = E.all_enrichments()
    assert to_csv(TaskKind.CHARACTERISATION, _pairs(enrichments)).value == E.to_csv(
        enrichments
    )


def test_byte_identical_without_a_header_too():
    """``header=False`` is the append path; a profile that got the header right and
    the rows wrong would still pass the two tests above on their first line."""
    verdicts = V.all_verdicts()
    assert to_csv(TaskKind.ASSIGNMENT, verdicts, header=False).value == V.to_csv(
        verdicts, header=False
    )


@_needs_enrichment_input
def test_byte_identical_without_a_header_on_characterisation_too():
    """The CHARACTERISATION half of the append path, split out so the ASSIGNMENT
    half above still runs where the gitignored enrichment input is absent."""
    enrichments = E.all_enrichments()
    assert to_csv(
        TaskKind.CHARACTERISATION, _pairs(enrichments), header=False
    ).value == E.to_csv(enrichments, header=False)


# --------------------------------------------------------------------------- #
# The cases the demo study does not contain
# --------------------------------------------------------------------------- #
_NASTY = 'driven by ERBB2, "strongly"\nsecond line; ;'

_VERDICT_STUBS = (
    NS(cluster="c1", cell_type="Tumor", cell_type_short="Tum", confidence="Very High",
       confidence_score=0.95, key_markers=("ERBB2", "KRT7"), notes=_NASTY,
       category="Epithelial", lineage="Epithelial", exclude=False, verify=False),
    # 0.70 is a real band anchor and the reason confidence_score must stay unformatted:
    # str(0.70) is "0.7" and f"{0.70:.2f}" is "0.70".
    NS(cluster="c9", cell_type="Unknown", cell_type_short="Unk", confidence="Medium",
       confidence_score=0.70, key_markers=(), notes="", category="", lineage="",
       exclude=True, verify=True),
)


def _pathway(**kw):
    base = dict(gene_set="HALLMARK_EMT", tier="enriched", score=6.5,
                score_kind="jazzpanda_enrichment", q_value=1.2e-9, panel_hits=6,
                set_size_full=200, panel_coverage=0.03, n_leading_edge=2,
                leading_edge=("CCND1", "SCD"))
    return NS(**(base | kw))


# ``census``, not ``confidence``: the band on a CHARACTERISATION is deleted, and the
# profile's last four columns are the three per-item counts plus cluster_verify.
# A zero untestable count and a non-zero one are both here, because untestable rows
# are the ones NOT in the file -- an error in that column is invisible in every
# other cell.
_ENRICHMENT_STUBS = (
    (NS(cluster="c1", cell_type="Tumor", census=(7, 3, 12), verify=False), _pathway()),
    # q_value None renders as an empty cell, and an empty leading edge as an empty
    # one too -- two blanks a tuple writer and a dict writer could disagree about.
    (NS(cluster="c2", cell_type='Stromal, "fibro"', census=(0, 1, 0), verify=True),
     _pathway(q_value=None, leading_edge=(), n_leading_edge=0, score=0.0,
              panel_coverage=0.0, tier="suggestive")),
)


def test_assignment_profile_matches_the_exporter_on_awkward_values():
    rows = [V.to_csv_row(v) for v in _VERDICT_STUBS]
    import csv
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(V.CSV_COLUMNS))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    assert to_csv(TaskKind.ASSIGNMENT, _VERDICT_STUBS).value == buf.getvalue()


def test_characterisation_profile_matches_the_exporter_on_awkward_values():
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(E.ENRICHMENT_CSV_COLUMNS)
    for ce, p in _ENRICHMENT_STUBS:
        w.writerow(E._csv_row(ce, p))
    assert to_csv(TaskKind.CHARACTERISATION, _ENRICHMENT_STUBS).value == buf.getvalue()


def test_the_band_anchor_is_not_reformatted():
    """Pins the one cell where a tidy-up would move the golden hash."""
    out = to_csv(TaskKind.ASSIGNMENT, _VERDICT_STUBS, header=False).value
    assert ",0.7," in out and ",0.70," not in out


# --------------------------------------------------------------------------- #
# The column contracts, frozen in count and in order
# --------------------------------------------------------------------------- #
def test_assignment_columns_are_the_eleven_in_exact_order():
    assert PROFILES[TaskKind.ASSIGNMENT].names == V.CSV_COLUMNS
    assert len(ASSIGNMENT_COLUMNS) == 11


def test_characterisation_columns_are_the_sixteen_in_exact_order():
    assert PROFILES[TaskKind.CHARACTERISATION].names == E.ENRICHMENT_CSV_COLUMNS
    assert len(CHARACTERISATION_COLUMNS) == 16


def test_the_characterisation_table_carries_no_band_column():
    """The deleted ``cluster_confidence``, pinned by name.

    A band on a characterisation row is the shape the no-band rule exists to catch,
    and a committed CSV is where it survived longest: the Pathways page stopped
    drawing it and downstream R went on reading it. The three count columns that
    replace it are asserted present in the same breath, because deleting a column
    without replacing it would leave a reader unable to tell "nothing else was
    running" from "nothing else could be asked".
    """
    names = PROFILES[TaskKind.CHARACTERISATION].names
    assert not {n for n in names if "confidence" in n or n.endswith("band")}, names
    assert names[-4:] == ("cluster_n_supported", "cluster_n_suggestive",
                          "cluster_n_untestable", "cluster_verify")


def test_a_profile_is_keyed_on_its_own_task_kind():
    for kind, profile in PROFILES.items():
        assert profile.task_kind is kind


# --------------------------------------------------------------------------- #
# Failing closed, and the empty record
# --------------------------------------------------------------------------- #
def test_a_task_kind_with_no_profile_raises_rather_than_exporting_nothing():
    """COMPARISON has no committed CSV. A header-only file would read to the
    biologist as a run that found nothing."""
    with pytest.raises(NoExportProfile):
        to_csv(TaskKind.COMPARISON, ())


def test_export_records_no_decisions():
    """An export projects an interpretation, it does not make one. A decision here
    would put a rule in the record that explains a formatting choice as an
    inference."""
    out = to_csv(TaskKind.ASSIGNMENT, V.all_verdicts())
    assert isinstance(out, Traced)
    assert out.decisions == ()
