"""Invariants for the CSV export profiles.

The single thing this module can get wrong is a byte. The 11-column verdict CSV is
pinned by ``tests/test_golden.py``, restated in ``skills/marker-interpretation/
SKILL.md`` Output 2 and read by downstream R; the 14-column enrichment CSV is a
committed artifact. So the central tests here do not inspect the output and judge it
-- they assert the profile's output is ``==`` to the committed bytes (both
hand-written exporters are retired; the golden hash pins them too). "Looks the
same" is what lets a trailing ``\\r`` or a ``0.70`` for a ``0.7`` through.

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

from panoscope.agent import config as cfg
from panoscope.pipeline import paths, v1_verdicts
from panoscope.core.export import (
    ASSIGNMENT_COLUMNS, CHARACTERISATION_COLUMNS, PROFILES, NoExportProfile, to_csv,
)
from panoscope.types.outputs import TaskKind
from panoscope.types.trace import Traced


# --------------------------------------------------------------------------- #
# Byte-identity against the committed verdict CSV -- the whole contract
# --------------------------------------------------------------------------- #
def _committed_verdict_csv() -> str:
    return paths.verdicts_csv(cfg.DATASET_ID).read_bytes().decode("utf-8")


def test_assignment_profile_is_byte_identical_on_the_demo_study():
    verdicts = list(v1_verdicts.verdicts().values())
    assert to_csv(TaskKind.ASSIGNMENT, verdicts).value == _committed_verdict_csv()


def test_byte_identical_without_a_header_too():
    """``header=False`` is the append path; a profile that got the header right and
    the rows wrong would still pass the test above on its first line."""
    verdicts = list(v1_verdicts.verdicts().values())
    committed = _committed_verdict_csv()
    assert to_csv(TaskKind.ASSIGNMENT, verdicts, header=False).value == \
        committed[committed.index("\r\n") + 2:]


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


_ASSIGNMENT_NAMES = (
    "cluster", "cell_type", "cell_type_short", "confidence", "confidence_score",
    "key_markers", "notes", "category", "lineage", "exclude", "verify",
)


def test_assignment_profile_matches_the_exporter_on_awkward_values():
    # The exact bytes the retired agent/verdict.to_csv wrote for these rows.
    assert to_csv(TaskKind.ASSIGNMENT, _VERDICT_STUBS).value == (
        ",".join(_ASSIGNMENT_NAMES) + "\r\n"
        'c1,Tumor,Tum,Very High,0.95,ERBB2;KRT7,"driven by ERBB2, ""strongly""\n'
        'second line; ;",Epithelial,Epithelial,FALSE,FALSE\r\n'
        "c9,Unknown,Unk,Medium,0.7,,,,,TRUE,TRUE\r\n"
    )


_CHARACTERISATION_NAMES = (
    "cluster", "cell_type", "gene_set", "tier", "score", "score_kind", "q_value",
    "panel_hits", "set_size_full", "panel_coverage", "n_leading_edge", "leading_edge",
    "cluster_n_supported", "cluster_n_suggestive", "cluster_n_untestable",
    "cluster_verify",
)


def test_characterisation_profile_matches_the_exporter_on_awkward_values():
    # The exact bytes the retired agent/enrichment.to_csv wrote for these rows.
    assert to_csv(TaskKind.CHARACTERISATION, _ENRICHMENT_STUBS).value == (
        ",".join(_CHARACTERISATION_NAMES) + "\r\n"
        "c1,Tumor,HALLMARK_EMT,enriched,6.5000,jazzpanda_enrichment,1.200e-09,6,200,"
        "0.0300,2,CCND1;SCD,7,3,12,FALSE\r\n"
        'c2,"Stromal, ""fibro""",HALLMARK_EMT,suggestive,0.0000,jazzpanda_enrichment,,'
        "6,200,0.0000,0,,0,1,0,TRUE\r\n"
    )


def test_the_band_anchor_is_not_reformatted():
    """Pins the one cell where a tidy-up would move the golden hash."""
    out = to_csv(TaskKind.ASSIGNMENT, _VERDICT_STUBS, header=False).value
    assert ",0.7," in out and ",0.70," not in out


# --------------------------------------------------------------------------- #
# The column contracts, frozen in count and in order
# --------------------------------------------------------------------------- #
def test_assignment_columns_are_the_eleven_in_exact_order():
    assert PROFILES[TaskKind.ASSIGNMENT].names == _ASSIGNMENT_NAMES
    assert len(ASSIGNMENT_COLUMNS) == 11


def test_characterisation_columns_are_the_sixteen_in_exact_order():
    assert PROFILES[TaskKind.CHARACTERISATION].names == _CHARACTERISATION_NAMES
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
    out = to_csv(TaskKind.ASSIGNMENT, list(v1_verdicts.verdicts().values()))
    assert isinstance(out, Traced)
    assert out.decisions == ()


#: SKILL.md Output 2, copied from the skill, not from the code.
_SKILL_OUTPUT_2 = ("cluster,cell_type,cell_type_short,confidence,confidence_score,"
                   "key_markers,notes,category,lineage,exclude,verify")


@pytest.mark.req("IMPL-10-03", "UI-3.2-01", "SKM-O2-01")
def test_the_verdict_csv_header_is_the_skills_eleven_columns_in_order():
    """Equality on the header line, never membership: a reordered or extra column is
    what breaks the downstream R reader, and ``in`` passes both."""
    from panoscope.workflows import marker

    out = to_csv(TaskKind.ASSIGNMENT, _VERDICT_STUBS).value
    assert out.splitlines()[0] == _SKILL_OUTPUT_2
    assert _committed_verdict_csv().splitlines()[0] == _SKILL_OUTPUT_2
    assert ",".join(marker.WORKFLOW.uispec().csv_columns) == _SKILL_OUTPUT_2
