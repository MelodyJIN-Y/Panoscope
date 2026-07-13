"""Config derives from the active dataset (nothing is a fixed literal)."""
from __future__ import annotations

import re

import pandas as pd

from agent import config as cfg


def _natural(c: str):
    m = re.match(r"c(\d+)$", c)
    return (0, int(m.group(1))) if m else (1, c)


def test_cluster_order_is_derived_from_the_markers():
    markers = pd.read_csv(cfg.ACTIVE_MARKERS_CSV)
    expected = tuple(
        sorted({str(x) for x in markers["top_cluster"] if str(x) != "NoSig"}, key=_natural)
    )
    assert cfg.CLUSTER_ORDER == expected
    assert cfg.KNOWN_CLUSTERS == frozenset(cfg.CLUSTER_ORDER)


def test_active_paths_resolve_to_the_dataset_tree():
    # the demo ships its inputs under the tree, so the active paths point there
    assert cfg.ACTIVE_MARKERS_CSV.name == "markers_top.csv"
    assert cfg.ACTIVE_MARKERS_CSV.exists()
    assert cfg.ACTIVE_PANEL_PARQUET.exists()


def test_demo_markers_derived_and_nonempty():
    assert cfg.DEMO_MARKERS and all(isinstance(g, str) for g in cfg.DEMO_MARKERS)
