"""Stage 4 — build the per-unit records through ``core/``, and write both views.

``IMPLEMENTATION.md`` §10: "Never mutate the committed per-unit JSONs. v3 records
write to ``interp/records/<task_kind>/<unit>.json``; the v1 files become a
**generated compatibility view** through one adapter, so the UI and the store are
untouched during migration." This stage is where that happens, so it writes twice:

  * ``interp/records/assignment/<unit>.json`` — the ``InterpretationRecord``, via
    ``core/record.write``. The reading (path, sha256, rows dropped and why, the
    entities no frame admits), every decision with its refs, the claims, the
    undecidables. This is the artifact; it is new, and nothing reads it yet.
  * ``interp/clusters/<unit>.json`` + ``interp/verdicts.csv`` — the v1 view,
    projected through ``core/compat.py`` (plus the one field it drops, see
    ``pipeline/records._with_pctile``). Byte-identical to what ``agent/verdict.py``
    wrote, which is the §10 gate's claim and ``tests/test_core_compat.py``'s
    assertion over all nine demo units.

It used to run ``agent.verdict.all_verdicts()``. That is the cutover: the files the
UI reads are now produced by the same functions ``panoscope explain`` renders from,
so the two surfaces cannot disagree about a band. ``agent/verdict.py`` stays exactly
where it is -- it is the independent reference implementation ``test_core_compat``
compares against, and a shim over ``core/`` would make that gate compare ``core/``
to itself.

Computes nothing. Every number arrives from ``core/``; this stage arranges and
serialises.
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

    The returned ``ClusterVerdict``s are rebuilt from the v1 dicts this stage just
    wrote, rather than being a second in-memory product: the calibration table and
    the manifest downstream then describe exactly the bytes on disk. ``run.py`` and
    ``calibration.py`` are unchanged by the cutover for that reason.

    The study is the ACTIVE one (``cfg.current_study()``), not ``dataset_id``. That
    is what ``agent.data`` resolved before the cutover, so the behaviour is
    preserved exactly -- including the pre-existing mismatch when a caller passes a
    ``dataset_id`` that is not the active study, which writes one study's output
    under another's name. Named here rather than silently fixed; see the report.
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
    for unit in cfg.CLUSTER_ORDER:
        write_record(built[unit], interp)
        omitted.update(dict(compat.v1_omissions(built[unit].evidence).value))
        v1 = records.to_v1_dict(built[unit])
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
