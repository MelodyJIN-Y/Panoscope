"""Panoscope per-dataset pipeline.

One command per dataset (`python -m pipeline.run --dataset <id>`) that turns a
dataset's raw inputs (jazzPanda markers, panel list, cluster key, + viz sources)
into a self-contained ``data/datasets/<id>/`` tree the UI reads with no live
recomputation. Deterministic where it can be, live-cited only where it must be;
jazzPanda is never run — its output is a consumed input.

This package is built in slices. Slice 1 (here): the scaffold, input validation,
and the deterministic per-cluster VERDICT persistence (verdicts.csv + per-cluster
JSON) plus the dataset manifest. Later slices add the viz precompute (density,
umap, expr) and the live-cited notes.

Every persisted value traces to jazzPanda, the panel, or the cluster key — the
confident floor holds on disk exactly as it does in the engine.

Entry point: ``python -m pipeline.run --dataset <id>`` (see :mod:`pipeline.run`).
"""

from __future__ import annotations
