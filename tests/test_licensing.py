"""Restricted reference data is fetched, never bundled, and never re-emitted
(ARCHITECTURE §12, corpus A9).

Two halves. The static half: no marker-database file is tracked under the repo or
ships inside the package. The integration half: a study built against a lab list that
is not redistributable exports the edge's id and citation, never the row's own text.
The lab row here carries a free-text ``note``; that sentence is the restricted content
a careless export would copy, so it is the thing searched for in every output.
"""

from __future__ import annotations

import io
import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
#: CellMarker 2.0's and PanglaoDB's release files, by the names their sites serve.
MARKER_DB = re.compile(r"(?i)(panglao.*\.(tsv|csv|txt|gz)$|cell_?marker.*\.(xlsx|csv|tsv|txt|gz)$)")


@pytest.mark.req("ARCH-12-01", "CORP-A9")
def test_no_marker_database_file_is_tracked_or_packaged():
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    packaged = [str(p.relative_to(REPO)) for p in (REPO / "panoscope").rglob("*") if p.is_file()]
    hits = [p for p in tracked + packaged if MARKER_DB.search(Path(p).name)]
    assert not hits, hits


def test_the_filename_rule_recognises_the_real_release_names():
    for name in ("PanglaoDB_markers_27_Mar_2020.tsv.gz", "Cell_marker_All.xlsx",
                 "cellmarker_human.csv"):
        assert MARKER_DB.search(name), name
    assert not MARKER_DB.search("sources.py")


RESTRICTED = "Lab-only row text 7f3a: do not redistribute this sentence"
CITE = "file:lab-list-row-0007"


@pytest.mark.req("CORP-A9")
def test_an_export_carries_the_edge_id_and_citation_never_the_restricted_row(
        study_builder, monkeypatch):
    from panoscope.agent import config as cfg
    from panoscope.agent import loop
    from panoscope.pipeline import run as pipeline_run
    from panoscope.reference import sources
    from panoscope.ui import report as R

    monkeypatch.setattr(loop, "has_model_key", lambda: False)
    d = study_builder("lic", clusters=["c1", "c2"])
    lab = d / "inputs" / "reference" / "lab_markers.csv"
    pd.read_csv(lab).assign(note=RESTRICTED, citation=CITE).to_csv(lab, index=False)
    assert sources.LabCurated.redistributable is False

    pipeline_run.run(d.name)
    with cfg.use_study(d.name):
        docx = R.report_to_docx(R.build_report_from_sources())
    outputs = {str(p.relative_to(d)): p.read_text(errors="replace")
               for p in d.rglob("*") if p.is_file() and "inputs" not in p.parts
               and p.suffix in (".json", ".csv", ".md", ".txt")}
    outputs["report.docx"] = (zipfile.ZipFile(io.BytesIO(docx))
                              .read("word/document.xml").decode("utf-8"))
    leaked = sorted(k for k, v in outputs.items() if "Lab-only row text 7f3a" in v)
    assert not leaked, leaked
    # Not vacuous: the lab list was read, and the build names it by source id.
    assert any("lab:" in v for v in outputs.values()), "the lab list never reached the build"
