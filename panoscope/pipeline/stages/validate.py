"""Stage 0 -- validate a dataset before any compute (fail closed).

1. **The store's §6.3 blockers**, from the declared platform's adapter: measured
   set, detectability frame, declared units, tissue/platform, per-sample tables,
   clustering mode. A missing label set is the starting state, not a blocker.
2. **The four v1 input checks**, through ``agent.data``: the cluster key covers
   every cluster, ``top_cluster`` labels are known, every cluster has an assigned
   marker, and the panel is non-empty.

The adapter *returns* its problems (§6.4); this stage *raises*, naming all of them.
"""

from __future__ import annotations

import pandas as pd

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.data.study import Study
from panoscope import platforms
from panoscope.platforms.xenium import is_no_signal


def validate(dataset_id: str | None = None) -> None:
    """Assert the dataset is interpretable. Raises on any gap, naming all of them."""
    dataset_id = dataset_id or cfg.DATASET_ID

    # Scope everything below to the study that was ASKED about. The v1 checks read
    # through ``agent.data``, which resolves the active study and has never looked at
    # ``dataset_id`` -- so ``panoscope build --study kidney``, which does not scope
    # anything (``cli.py``), validated the demo and reported the kidney study clean.
    # An already-active study with the right id is used as-is, because that is a
    # caller who has scoped a tree we could not resolve ourselves.
    store = cfg.current_study()
    if store.id != dataset_id:
        store = Study.open(dataset_id)
    with cfg.use_study(store):
        _check(store)


def _check(store: Study) -> None:
    """The two check groups, with the study already scoped."""
    try:
        problems = platforms.for_study(store).validate(store)
    except platforms.UnregisteredPlatform as exc:
        problems = [str(exc)]
    if problems:
        raise ValueError(
            f"[pipeline] study {store.id!r} is blocked (ARCHITECTURE §6.3); "
            f"{len(problems)} problem(s) must be fixed before it can be built:\n  - "
            + "\n  - ".join(problems)
        )

    markers = data.load_markers()
    panel = data.load_panel()

    # cluster_key.json is optional: cell types are the marker-skill annotation
    # (interp/annotation.json). If a key IS provided it must cover every cluster.
    try:
        key = data.load_cluster_key()
    except Exception:  # noqa: BLE001 - absent key is fine; annotate assigns the types
        key = None
    if key is not None:
        missing_key = [c for c in cfg.CLUSTER_ORDER if c not in key]
        if missing_key:
            raise ValueError(f"[pipeline] cluster_key is missing clusters: {missing_key}")

    # Read the file, not ``data.load_markers()``: that reader drops the control
    # rows, and a control row carrying a bogus unit label is exactly the incoherence
    # this check exists to refuse. Filtering first made the guard quieter the moment
    # the reader got stricter.
    raw = pd.read_csv(data._markers_top())
    allowed = set(cfg.KNOWN_CLUSTERS)
    seen = {str(x).strip() for x in raw["top_cluster"] if not is_no_signal(x)}
    unknown = sorted(seen - allowed)
    if unknown:
        raise ValueError(
            f"[pipeline] markers_top has unknown top_cluster labels: {unknown} "
            f"(allowed: c1..cN + the no-signal sentinel in any case)"
        )

    empty = [c for c in cfg.CLUSTER_ORDER if data.get_cluster_markers(c).empty]
    if empty:
        raise ValueError(f"[pipeline] clusters with no assigned markers: {empty}")

    if panel.empty:
        raise ValueError("[pipeline] panel is empty; the absence primitive needs it")
