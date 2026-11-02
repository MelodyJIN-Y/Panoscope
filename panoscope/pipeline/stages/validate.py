"""Stage 0 — validate a dataset before any compute (fail closed).

Two checks, in that order, because they fail at different depths.

**The store's §6.3 blockers come first**, from the platform adapter. The adapter is
the only object that reads the manifest and the capability frames first-hand, and
``XeniumAdapter.validate`` has always checked all four blockers — no measured set, no
detectability frame, no annotation label set, no declared units — plus the
tissue/platform declaration, the missing per-sample tables and the undeclared
clustering mode. It had **no caller anywhere**: a study could fail every one of them
and reach the end of a build reporting success. Wiring it in here is the whole of the
fix; the checks themselves were already written and already right.

**Then the four v1 input checks**, through ``agent.data`` (exactly what the app will
see): the cluster key covers every cluster, the marker table's ``top_cluster`` labels
are known, every cluster has at least one assigned marker, and the panel is non-empty.
They are kept rather than replaced — they catch table-level incoherence the adapter
does not look at — but they miss every blocker above, which is why they are second.

The adapter *returns* its problems (§6.4: a checklist, not a traceback) and this
stage *raises*, naming all of them at once. A build is not an intake interview: by the
time ``panoscope build`` runs, a blocked study must stop, not be summarised.
"""

from __future__ import annotations

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.data.study import Study
from panoscope.platforms.xenium import XeniumAdapter


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
    problems = XeniumAdapter().validate(store)
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

    allowed = set(cfg.KNOWN_CLUSTERS) | {"NoSig"}
    seen = set(markers["top_cluster"].astype(str))
    unknown = sorted(seen - allowed)
    if unknown:
        raise ValueError(
            f"[pipeline] markers_top has unknown top_cluster labels: {unknown} "
            f"(allowed: c1..cN + NoSig)"
        )

    empty = [c for c in cfg.CLUSTER_ORDER if data.get_cluster_markers(c).empty]
    if empty:
        raise ValueError(f"[pipeline] clusters with no assigned markers: {empty}")

    if panel.empty:
        raise ValueError("[pipeline] panel is empty; the absence primitive needs it")
