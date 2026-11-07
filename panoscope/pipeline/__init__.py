"""Panoscope per-dataset pipeline.

``panoscope build --study <id>`` turns a dataset's raw inputs (method output, panel,
cluster key, viz sources) into a self-contained ``data/studies/<id>/`` tree the UI
reads with no live recomputation. The method is never run; its output is a consumed
input, and every persisted value traces to it, the panel, or the cluster key.

Entry point: :mod:`panoscope.pipeline.run`.
"""

from __future__ import annotations
