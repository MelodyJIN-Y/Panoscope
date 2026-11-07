"""Stage 4 -- build the per-unit records through ``core/``, and write both views.

``IMPLEMENTATION.md`` §10: v3 records are new files and the v1 files become a
generated compatibility view. So this stage writes twice:

  * ``interp/records/assignment/<unit>.json`` -- the ``InterpretationRecord``, via
    ``core/record.write``.
  * ``interp/clusters/<unit>.json`` + ``interp/verdicts.csv`` -- the v1 view,
    projected through ``core/compat.py`` (plus ``pipeline/records._with_pctile``),
    pinned by the golden hash.

The UI's files come from the same functions ``panoscope explain`` renders from.
Computes nothing; every number arrives from ``core/``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from panoscope.agent import config as cfg
from panoscope.agent.types import ClusterVerdict

from panoscope.pipeline import paths, records
from panoscope.pipeline.serialize import verdict_from_dict


def run_verdicts(
    dataset_id: str | None = None, root: Optional[Path] = None
) -> list[ClusterVerdict]:
    """Build every unit through ``core/``; write both views; return the v1 objects.

    The returned ``ClusterVerdict``s are reloaded from the v1 dicts just written, so
    the calibration table and manifest describe exactly the bytes on disk.

    The study is the ACTIVE one (``cfg.current_study()``), not ``dataset_id``: a
    ``dataset_id`` that is not the active study writes one study's output under
    another's name (known, preserved behaviour).
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    study = cfg.current_study()
    built = records.build_all(study)

    missing = [u for u in cfg.CLUSTER_ORDER if u not in built]
    if missing:
        raise KeyError(
            f"[verdicts] the method produced no evidence for unit(s) {missing}; "
            f"it found {sorted(built)}. A unit the study declares and the method "
            "never saw must not be written as a verdict with nothing behind it."
        )

    cdir = paths.clusters_dir(dataset_id, root)
    cdir.mkdir(parents=True, exist_ok=True)
    interp = paths.interp_dir(dataset_id, root)

    from panoscope.core import compat

    verdicts: list[ClusterVerdict] = []
    rows: list[dict] = []
    omitted: Counter[str] = Counter()
    no_replicate_shape = 0
    for unit in cfg.CLUSTER_ORDER:
        write_record(built[unit], interp)
        omitted.update(dict(compat.v1_omissions(built[unit].evidence).value))
        # The §9.10 replicate read, into the v1 view's one free-text field. Spliced
        # here rather than inside ``records.to_v1_dict`` because that projection is
        # handed an evidence/verdict pair and never the record, and the record is
        # where the concordance decisions and their sentences live. ``core/compat``
        # still owns every word of it; this stage hands it the two pieces and writes
        # what comes back.
        v1 = compat.with_v1_replicate_read(
            records.to_v1_dict(built[unit]),
            built[unit].record.undecidable,
            built[unit].verdict.replicate_coverage,
        ).value
        if built[unit].verdict.replicate_coverage is None:
            no_replicate_shape += 1
        paths.cluster_json(dataset_id, unit, root).write_text(
            json.dumps(v1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        verdicts.append(verdict_from_dict(v1))
        rows.append(v1)

    # The v1 view is lossy BY DESIGN (§10), and the loss is announced rather than
    # left for a reader to notice: these items exist, in full, in the v2 records
    # written above, and the v1 files say nothing at all about them -- which is the
    # only representation that cannot be mistaken for "never measured". The reasons
    # are written longhand in core/compat.V1_UNREPRESENTABLE.
    if sum(omitted.values()):
        print("[verdicts] v1 view omits "
              + ", ".join(f"{n} {what}" for what, n in sorted(omitted.items()) if n)
              + f" across {len(cfg.CLUSTER_ORDER)} units; v1 has no shape for them "
              "(core/compat.V1_UNREPRESENTABLE). They are in the v2 records under "
              f"{interp / 'records'}.")

    # The same discipline for the one WHOLE-RECORD omission: a unit whose §9.10
    # replicate read was never askable keeps that reason in the v2 record and the
    # export, and the v1 row says nothing. Announced at the moment the lossy view is
    # written, for the reason above -- an omission a reader has to notice is the one
    # that gets read as "nothing to report".
    if no_replicate_shape:
        print(f"[verdicts] v1 view omits the replicate read on "
              f"{no_replicate_shape} of {len(cfg.CLUSTER_ORDER)} units; the §9.10 "
              "comparison was never askable there and v1 has no field for why "
              f"(core/compat.V1_NO_REPLICATE_SHAPE). The reason is in the v2 "
              f"records under {interp / 'records'} and in the exported document.")

    # The CSV comes off the SAME projections just written, never a second pass:
    # one row in the table and one file on disk cannot then disagree.
    csv_path = paths.verdicts_csv(dataset_id, root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(compat.v1_verdict_csv(rows).value, encoding="utf-8")

    return verdicts


def write_record(built: records.UnitBuild, interp: Path) -> Path:
    """The v2 record, under ``interp/records/<task_kind>/<unit>.json``.

    ``core/record.write`` owns the path construction and is the only writer: §10's
    "never mutate the committed per-unit JSONs" is checkable rather than
    conventional precisely because the v1 tree is not a path that function can
    produce.
    """
    from panoscope.core import record as core_record

    return core_record.write(built.record, interp).value
