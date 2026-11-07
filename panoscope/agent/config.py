"""Panoscope constants and configuration.

Single source of truth for paths, the authoritative cluster->cell-type key,
model name, confidence bands, and the demo marker set.

Grounding discipline: DEMO_MARKERS is DERIVED at import time by reading the two
source files (the thresholded jazzPanda top-marker table and the Xenium panel
list). It is never a hand-typed literal, so it cannot drift from source. If a
required source file is missing, import fails loudly with a named error rather
than silently substituting a stale list.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING
from pathlib import Path

import pandas as pd

from panoscope.data import layout
from panoscope.platforms.xenium import is_no_signal
from panoscope.data.study import Study

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repo root = two levels above this package; ``PANOSCOPE_DATA_ROOT`` overrides.
# Transitional: ``data/layout.resolve_data_root()`` is the real resolver.
def _resolve_repo_root() -> Path:
    env = os.getenv("PANOSCOPE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _resolve_repo_root()

# Raw Xenium input (read by scripts/precompute_density.py).
RAW_XEN: str = "Raw_data_Xenium_hbreast_sample1"
RAW_XEN_DIR: Path = PROJECT_ROOT / RAW_XEN
TRANSCRIPTS_CSV_GZ: Path = RAW_XEN_DIR / "transcripts.csv.gz"


def _data_dir_path() -> Path:
    return layout.resolve_data_root()


def _tidy_markers_csv() -> Path:
    return _data_dir_path() / "jazzpanda" / "markers_top.csv"


def _tidy_panel_parquet() -> Path:
    return _data_dir_path() / "panels" / "panel.parquet"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
PRIMARY_MODEL: str = os.getenv("PANOSCOPE_MODEL", "claude-sonnet-4-6")

# --------------------------------------------------------------------------- #
# Active dataset (selectable via PANOSCOPE_DATASET; one dataset per process).
# Everything below derives from THIS dataset's files, never a hardcoded literal.
# --------------------------------------------------------------------------- #
# The bundled demo dataset. Its identity is fixed (the bundled fallback maps /
# legacy files belong to it); any OTHER dataset derives everything from its own tree.
BUNDLED_DEMO_ID: str = "xenium_hbreast_sample1"


# --------------------------------------------------------------------------- #
# The current study: a ContextVar, so it scopes per Streamlit session/thread and
# per ``use_study()`` block.
# --------------------------------------------------------------------------- #
_CURRENT_STUDY: ContextVar[Optional[Study]] = ContextVar("panoscope_study", default=None)


def current_study() -> Study:
    """The active study. Environment is read at CALL time, never at import time."""
    s = _CURRENT_STUDY.get()
    if s is not None:
        return s
    s = Study.open(layout.default_study_id() or BUNDLED_DEMO_ID)
    _CURRENT_STUDY.set(s)
    return s


def set_study(study: Study | str) -> Study:
    """Set the active study for this context. Accepts a Study or a study id."""
    s = study if isinstance(study, Study) else Study.open(study)
    _CURRENT_STUDY.set(s)
    return s


@contextmanager
def use_study(study: Study | str):
    """Scope the active study to a block. The idiom tests should use."""
    token = _CURRENT_STUDY.set(study if isinstance(study, Study) else Study.open(study))
    try:
        yield _CURRENT_STUDY.get()
    finally:
        _CURRENT_STUDY.reset(token)


def _active_input(name: str, legacy: Path) -> Path:
    """The study's own file; the bundled legacy flat file ONLY for the bundled demo.

    A non-demo study without the file gets the missing path, so readers raise naming
    it rather than silently reading another tissue's markers or panel.
    """
    cand = current_study().inputs / name
    if cand.exists():
        return cand
    return legacy if current_study().id == BUNDLED_DEMO_ID else cand


def _active_markers_csv() -> Path:
    return _active_input("markers_top.csv", _tidy_markers_csv())


def _active_panel_parquet() -> Path:
    return _active_input("panel.parquet", _tidy_panel_parquet())


def _derive_cluster_order() -> tuple[str, ...]:
    """Cluster ids in the active dataset's markers (top_cluster != NoSig), sorted
    naturally (c1, c2, ... c10); without a marker table, the adapter's unit frame.
    Derived from the data, never a hardcoded literal."""
    import re

    if _active_markers_csv().exists():
        m = pd.read_csv(_active_markers_csv())
        # ``is_no_signal``, not ``!= "NoSig"``: this set IS the cluster label space, so
        # any spelling of the sentinel must stay out of it.
        labels = {str(x).strip() for x in m["top_cluster"].unique()
                  if not is_no_signal(x)}
    else:
        from panoscope import platforms
        from panoscope.types.subject import EntityKind

        study = current_study()
        labels = {u for f in platforms.for_study(study).frames(study)
                  if f.kind is EntityKind.UNIT for u in f.members}
        if not labels:
            raise FileNotFoundError(
                f"[config] markers missing: {_active_markers_csv()} (run the data prep)"
            )

    def _key(c: str):
        mm = re.match(r"c(\d+)$", c)
        return (0, int(mm.group(1))) if mm else (1, c)

    return tuple(sorted(labels, key=_key))

# --------------------------------------------------------------------------- #
# Authoritative cluster -> cell type key — READ from the active dataset's committed
# inputs/cluster_key.json (the dataset's OWN annotation: jazzPanda / Seurat `anno`),
# never a hardcoded literal. Every label traces to that provenance-hashed input file;
# Panoscope assesses confidence FOR these labels, it does not invent them.
# Prefix convention (marker-interpretation SKILL naming): Tum_ tumor, Str_ stroma,
# Imm_Mac/Imm_T/Imm_B/Imm_DC/Imm_Mast immune subsets, Myoepi_ myoepithelial, Endo_ endothelial.
# --------------------------------------------------------------------------- #
def _load_cluster_key() -> dict[str, dict[str, str]]:
    """Unit -> {cell_type, cell_type_short, category, lineage}, via ``Study.unit_labels``.

    Does not raise on absence; the caller reports missing labels at the point of use.
    """
    return {c: {k: str(v) for k, v in meta.items() if isinstance(v, (str, int, float))}
            for c, meta in current_study().unit_labels.items()}


# --------------------------------------------------------------------------- #
# Gene-set enrichment rubric (second workflow; TUNABLE, calibrate on real data).
# Bands are keyed by score_kind (the schema stays method-agnostic); the jazzPanda
# competitive test_statistic (z-like, ~2–30) is NOT a glm_coef, so it has its own
# thresholds.
# --------------------------------------------------------------------------- #
ENRICHMENT_BANDS: dict[str, dict[str, float]] = {
    "jazzpanda_enrichment": {"Very High": 12.0, "High": 8.0, "Medium-High": 5.0, "Medium": 3.0, "Low": 0.0},
}
ENRICH_Q_MAX: float = 0.05      # q below this + gate -> "enriched"
SUGGESTIVE_Q_MAX: float = 0.25  # q in [ENRICH_Q_MAX, this] + gate -> "suggestive · verify"
MIN_PANEL_HITS: int = 3         # set genes on panel; below -> untestable
MIN_LEADING_EDGE: int = 3       # driving genes; below -> not defensible


def band_for_enrichment(score: float | None, score_kind: str) -> str:
    """Map an enrichment score to a confidence label, per method. Bigger -> higher.

    Unknown score_kind or None score -> "Low". Pure; thresholds from
    ENRICHMENT_BANDS (tunable). Never conflates the two methods' score scales.
    """
    bands = ENRICHMENT_BANDS.get(score_kind)
    if score is None or bands is None:
        return "Low"
    s = float(score)
    for label in ("Very High", "High", "Medium-High", "Medium"):
        if s >= bands[label]:
            return label
    return "Low"


# --------------------------------------------------------------------------- #
# DEMO_MARKERS — top 3 ON-PANEL markers per cluster by glm_coef.
# DERIVED from the source files (never a literal). ~26 genes (c9 Mast has only
# 2 on-panel assigned markers, so it contributes 2 rather than 3).
# --------------------------------------------------------------------------- #
DEMO_MARKERS_PER_CLUSTER: int = 3


def _load_panel_gene_set() -> frozenset[str]:
    """Return the analyzed-panel gene set from the committed tidy panel file.

    Reads data/panels/panel.parquet (committed) rather than the raw, gitignored
    TSV, so the app / DEMO_MARKERS derivation works from a fresh clone or in CI.
    The 33 "Custom" add-on genes are already excluded when the tidy panel is built,
    and panel size is dataset-dependent (dynamic), so no fixed count is asserted.
    """
    if not _active_panel_parquet().exists():
        raise FileNotFoundError(
            f"[config] panel missing: {_active_panel_parquet()} — produced by the data "
            f"prep (inputs/panel.parquet). Run the prep before importing."
        )
    panel = pd.read_parquet(_active_panel_parquet())
    if "gene" not in panel.columns:
        raise ValueError(
            f"[config] panel {_active_panel_parquet()} has no 'gene' column; got {list(panel.columns)}"
        )
    return frozenset(panel["gene"].astype(str))


def _derive_demo_markers() -> list[str]:
    """Top-3 on-panel markers per cluster (c1..c9) by glm_coef, in cluster order.

    Reads the committed tidy top-marker table (data/jazzpanda/markers_top.csv,
    already thresholded, proper columns) so DEMO_MARKERS derives from committed
    data on a fresh clone / CI. NoSig rows and off-panel genes are excluded.
    Order within a cluster is glm_coef descending; clusters are emitted c1..c9.
    """
    if not _active_markers_csv().exists():
        raise FileNotFoundError(
            f"[config] top-marker file missing: {_active_markers_csv()} — produced by "
            f"the data prep (inputs/markers_top.csv). Run the prep before importing."
        )
    markers = pd.read_csv(_active_markers_csv())
    expected_cols = {"gene", "top_cluster", "glm_coef"}
    missing = expected_cols - set(markers.columns)
    if missing:
        raise ValueError(
            f"[config] markers CSV {_active_markers_csv()} missing columns {missing}; "
            f"got {list(markers.columns)}"
        )

    panel_genes = _load_panel_gene_set()

    on_panel = markers[
        (~markers["top_cluster"].map(is_no_signal))
        & (markers["gene"].isin(panel_genes))
        & (markers["glm_coef"] > 0)
    ]

    demo: list[str] = []
    for cluster in _derive_cluster_order():
        sub = on_panel[on_panel["top_cluster"] == cluster].sort_values(
            "glm_coef", ascending=False
        )
        demo.extend(sub["gene"].head(DEMO_MARKERS_PER_CLUSTER).tolist())
    return demo


# --------------------------------------------------------------------------- #
# Lazy module attributes (PEP 562): ``cfg.X`` resolves against the CURRENT study at
# call time. ``from config import X`` freezes it, so don't. No import-time I/O.
# --------------------------------------------------------------------------- #
_LAZY = {
    "DATASET_ID":          lambda: current_study().id,
    "CLUSTER_KEY":         _load_cluster_key,
    "CLUSTER_ORDER":       _derive_cluster_order,
    "KNOWN_CLUSTERS":      lambda: frozenset(_derive_cluster_order()),
    "DEMO_MARKERS":        _derive_demo_markers,
    "DATA_DIR_PATH":       _data_dir_path,
    "ACTIVE_MARKERS_CSV":  _active_markers_csv,
    "ACTIVE_PANEL_PARQUET": _active_panel_parquet,
}


def __getattr__(name: str):
    fn = _LAZY.get(name)
    if fn is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return fn()


def __dir__():
    return sorted({*globals(), *_LAZY})


if TYPE_CHECKING:  # so type checkers and IDEs still see the names
    DATASET_ID: str
    CLUSTER_KEY: dict[str, dict[str, str]]
    CLUSTER_ORDER: tuple[str, ...]
    KNOWN_CLUSTERS: frozenset[str]
    DEMO_MARKERS: list[str]
    DATA_DIR_PATH: Path
    ACTIVE_MARKERS_CSV: Path
    ACTIVE_PANEL_PARQUET: Path
