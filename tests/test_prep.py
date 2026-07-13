"""The prep stage: read-if-present, else the documented contract error."""
from __future__ import annotations

import pytest

from pipeline.stages import prep


def test_noop_when_tidy_inputs_present():
    # the bundled demo's tidy inputs exist -> no-op, no R invoked, no error
    prep.run_prep()


def test_contract_error_when_inputs_absent_and_no_r(monkeypatch):
    # a dataset with no tidy inputs and no Rscript on PATH -> explicit contract error
    monkeypatch.setattr(prep.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError):
        prep.run_prep("an_unprepared_dataset_123")
