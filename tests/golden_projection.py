"""The golden projection — one normalised, hashable view of the deterministic
interpretation artifacts for the demo study.

This module is the single implementation shared by the golden test
(``tests/test_golden.py``) and the regeneration script
(``scripts/regenerate_golden.py``), so the thing that is checked and the thing
that is written can never drift.

What it covers (IMPLEMENTATION.md §7 Phase A.4):

  * the 9 per-cluster verdicts      ``agent.verdict.all_verdicts()``
  * the verdict CSV                 ``agent.verdict.to_csv`` (11 frozen columns)
  * the holistic review             ``agent.holistic.holistic_review()``
  * the calibration table           ``pipeline.calibration.calibration_markdown``

...and the second workflow, which the hash was blind to until SCHEMA_VERSION 2
(mutating ``MIN_PANEL_HITS`` or ``ENRICHMENT_BANDS`` left the hash byte-identical):

  * the 9 per-cluster enrichments   ``agent.enrichment`` (tiers, census, caveats)
  * the enrichment CSV              ``agent.enrichment.to_csv`` (14 columns)
  * the cross-cluster themes        ``agent.enrichment_themes.pathway_themes()``

See :func:`pinned_enrichment` for why the enrichment half reads its raw rows from
the committed ``interp/enrichment/`` records rather than the gitignored source CSV.

**Why not ``pipeline.run.run()``.** ``pipeline/stages/viz.py`` collects the viz
frames with ``shutil.move`` — running the full pipeline from a test would MOVE
real files out of the repo working tree, and ``run()`` additionally reaches for
the live annotate stage on an incomplete dataset. The four artifacts above are
Tier A (pure functions of the committed ``inputs/``), so this module computes
them straight from the ``agent.*`` / ``pipeline.calibration`` entry points that
``run()`` itself calls. Nothing here writes to the repo, opens a socket, or
calls an LLM.

**Normalisation, so the hash tracks behaviour and not noise.**

  * every float is rendered at fixed precision (``{:.6f}``, matching
    ``agent.verdict._source_trace``), and ``-0.0`` collapses to ``0.0``; float
    ``repr`` drift can therefore never move the hash
  * mapping keys are sorted — dict iteration order is not semantically meaningful
  * sequence order is PRESERVED: evidence rows are glm_coef-ranked, ``key_markers``
    is a ranking, ``demotions`` and ``source_trace`` are ordered audit trails, and
    the coherence notes are ordered. Sorting those would hide real regressions
  * the per-cluster verdicts are keyed by cluster id (their order is pinned
    separately by ``cluster_order``) so a diff can name the cluster that moved
  * an unexpected value type raises rather than being coerced — a refactor that
    smuggles a non-JSON object into an artifact should fail loudly here
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from contextlib import contextmanager

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E
from panoscope.agent import enrichment_themes as agent_themes
from panoscope.agent import holistic as agent_holistic
from panoscope.agent import verdict as V
from panoscope.pipeline import calibration as calibration_mod
from panoscope.pipeline import serialize, store

# --------------------------------------------------------------------------- #
# Fixture location + the documented regeneration path
# --------------------------------------------------------------------------- #
GOLDEN_DIR: Path = Path(__file__).resolve().parent / "golden"
FIXTURE_PATH: Path = GOLDEN_DIR / "demo_interpretation.json"
HASH_PATH: Path = GOLDEN_DIR / "demo_interpretation.sha256"

#: The study this projection pins. The fixture is study-specific, so the test
#: skips rather than lies when the process was pointed at a different dataset.
DEMO_ID: str = "xenium_hbreast_sample1"

#: Bump when the SHAPE of the projection changes (a new artifact, a renamed key),
#: so a reviewer can tell a projection-format change from a behaviour change.
SCHEMA_VERSION: int = 2

#: Fixed float precision. Same format string as ``agent.verdict._source_trace``.
FLOAT_FORMAT: str = "{:.6f}"

#: Named in the failure message. Regeneration is always a deliberate act.
REGEN_COMMAND: str = ".venv/bin/python scripts/regenerate_golden.py --write"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalise(value: Any) -> Any:
    """Return ``value`` as JSON-safe, order-stable, fixed-precision data.

    Floats become fixed-precision strings, mappings are key-sorted, sequences
    keep their order (see the module docstring), and anything else raises.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value  # bool/int before float: bool is an int subclass, not a float
    if isinstance(value, float):
        if value == 0.0:
            value = 0.0  # collapse -0.0, which would render as "-0.000000"
        return FLOAT_FORMAT.format(value)
    if isinstance(value, dict):
        return {str(k): normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar -> python scalar
    if callable(item):
        return normalise(item())
    raise TypeError(
        f"[golden] refusing to normalise {type(value).__name__} ({value!r}); the "
        f"projection must contain only JSON-safe values. If an artifact gained a "
        f"new field type, teach normalise() about it deliberately."
    )


# --------------------------------------------------------------------------- #
# The enrichment half — recomputed from a COMMITTED input snapshot
# --------------------------------------------------------------------------- #
# ``agent.enrichment`` reads the raw jazzPanda result CSV, and for the demo study
# that file resolves to ``data/jazzpanda/hbc_sp1_top10_hallmark_test_statistic.csv``
# — gitignored, so it does not exist on a fresh clone or in CI. Projecting
# straight off it would make the golden hash machine-dependent, which is worse
# than the blindness it is meant to fix.
#
# The raw numbers ARE committed though: ``interp/enrichment/c*.json`` persists
# every ``PathwayEvidence`` in ``all_tested``, including the untestable ones, and
# carries each row's untouched source fields (score, p, q, panel_hits,
# set_size_full, leading_edge, gc_corr). Those fields are copied verbatim from the
# CSV and never derived, so reversing them back into raw-row shape yields exactly
# the rows the adapter would have read.
#
# Everything DERIVED is then recomputed live: tier, caveats, panel_coverage, the
# panel-scoped caveat, the census, verify, the source trace and the cross-cluster
# themes. That is what makes the hash sensitive to MIN_PANEL_HITS, MIN_LEADING_EDGE,
# ENRICH_Q_MAX and SUGGESTIVE_Q_MAX.
#
# Second known ceiling, now that panel_hits is a capability-frame count: the frames
# are NOT re-derived here, because `panel_hits` in a committed record already is the
# frame count and is fed straight back. A change to the study's panel or
# detectability file therefore reaches this projection only after the enrichment
# records are rebuilt. `tests/test_pipeline_enrichment_records.py` builds from the
# real inputs and is where a frame change is caught.
#
# Known ceiling: a change to the RAW numbers in those committed JSONs would be
# absorbed rather than flagged, because they are the input here. Committing
# ``data/datasets/<id>/inputs/enrichment.csv`` would remove that ceiling and let
# this read the real input directly — see the report accompanying this change.
def _raw_rows_from_committed() -> dict[str, list[dict[str, Any]]]:
    """Committed per-cluster enrichment records -> raw jazzPanda rows, by cluster.

    Only the columns ``_pathway_from_jz_row`` consumes are reconstructed; the
    others (AIC, variance_*, status, ...) are not read by any interpretation code.
    """
    persisted = store.load_all_enrichments()
    if persisted is None:
        raise FileNotFoundError(
            "[golden] the committed enrichment records "
            "(data/datasets/<id>/interp/enrichment/c*.json) are missing or "
            "incomplete; the enrichment half of the projection cannot be built."
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    for ce in persisted:
        rows[ce.cluster] = [
            {
                "gene_set": p.gene_set,
                "geneset_size": p.set_size_full,
                # The FRAME-checked footprint, under the key the engine reads. Not
                # "n_overlap": ``agent/enrichment`` stopped reading the result's own
                # count, and offering it here would offer a number the engine
                # deliberately refuses. ``panel_hits`` in a committed record is
                # already the frame count, so this round-trips exactly.
                E.PANEL_HITS_IN_FRAME: p.panel_hits,
                "genes_selected": ",".join(p.leading_edge),
                "test_statistic": p.score,
                "p_value": p.p_value,
                "p_adj_bh": p.q_value,
                "gc_corr": p.gc_corr,
            }
            for p in ce.all_tested
        ]
    return rows


@contextmanager
def pinned_enrichment():
    """Point ``agent.enrichment.all_enrichments`` at the committed rows.

    A module-attribute swap rather than a new seam in the product code:
    ``agent.enrichment_themes`` resolves ``all_enrichments`` at call time, so one
    swap covers both the per-cluster records and the cross-cluster themes, and
    nothing outside ``tests/`` changes. Restored in ``finally``.
    """
    rows = _raw_rows_from_committed()
    original = E.all_enrichments

    def _from_committed() -> list:
        return [
            E._cluster_enrichment(
                cluster,
                tuple(E._pathway_from_jz_row(r) for r in rows.get(cluster, ())),
                E._JZ_SCORE_KIND,
            )
            for cluster in cfg.CLUSTER_ORDER
        ]

    E.all_enrichments = _from_committed
    try:
        yield _from_committed()
    finally:
        E.all_enrichments = original


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def build_projection() -> dict[str, Any]:
    """Compute the normalised projection from the committed study inputs.

    Pure and side-effect free: reads ``data/datasets/<id>/inputs/`` (via
    ``agent.data`` / ``agent.config``), writes nothing, touches no network.
    """
    verdicts = V.all_verdicts()
    review = agent_holistic.holistic_review()
    with pinned_enrichment() as enrichments:
        themes = agent_themes.pathway_themes()
        enrichment_csv = E.to_csv(enrichments).splitlines()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": cfg.DATASET_ID,
        # Order is pinned here so the verdicts map can be keyed (and diffed) by cluster.
        "cluster_order": list(cfg.CLUSTER_ORDER),
        # The 11-column CSV contract, in exact order.
        "csv_columns": list(V.CSV_COLUMNS),
        "verdicts": normalise({v.cluster: serialize.verdict_to_dict(v) for v in verdicts}),
        # Line-split so a diff points at the row that moved, not the whole blob.
        "verdicts_csv": V.to_csv(verdicts).splitlines(),
        "holistic": normalise(serialize.holistic_to_dict(review)),
        "calibration_table": calibration_mod.calibration_markdown(verdicts).splitlines(),
        # --- the enrichment workflow (CHARACTERISATION) ------------------- #
        "enrichment_csv_columns": list(E.ENRICHMENT_CSV_COLUMNS),
        "enrichment": normalise(
            {e.cluster: serialize.enrichment_to_dict(e) for e in enrichments}
        ),
        "enrichment_csv": enrichment_csv,
        "pathway_themes": normalise(serialize.pathway_themes_to_dict(themes)),
    }


def canonical_json(projection: dict[str, Any]) -> str:
    """Render a projection as the exact bytes that get hashed and committed."""
    return json.dumps(projection, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def projection_hash(projection: dict[str, Any]) -> str:
    """sha256 of the canonical JSON — the one value the golden test compares."""
    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Fixture I/O
# --------------------------------------------------------------------------- #
def load_fixture() -> dict[str, Any]:
    """Load the committed expected projection. FileNotFoundError names the fix."""
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"[golden] missing fixture {FIXTURE_PATH}. Create it deliberately with:\n"
            f"    {REGEN_COMMAND}"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def load_expected_hash() -> str:
    """Load the committed expected sha256 (first whitespace-delimited token)."""
    if not HASH_PATH.exists():
        raise FileNotFoundError(
            f"[golden] missing hash file {HASH_PATH}. Create it deliberately with:\n"
            f"    {REGEN_COMMAND}"
        )
    return HASH_PATH.read_text(encoding="utf-8").split()[0]


def write_fixture(projection: dict[str, Any]) -> str:
    """Write the fixture + its hash file. Only ever called by the regen script."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(canonical_json(projection), encoding="utf-8")
    digest = projection_hash(projection)
    HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    return digest


# --------------------------------------------------------------------------- #
# Readable diff — "which cluster and which field moved", not "hash mismatch"
# --------------------------------------------------------------------------- #
def _label_for(index: int, item: Any) -> str:
    """Index label for a sequence element, enriched with its identity when it has one."""
    if isinstance(item, dict):
        for key in ("gene", "cluster", "marker", "gene_set"):
            name = item.get(key)
            if isinstance(name, str) and name:
                return f"{index}:{name}"
    return str(index)


def flatten(value: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten the projection to ``dotted.path -> leaf`` so two runs can be compared.

    Paths read like ``verdicts.c1.evidence[0:ERBB2].glm_coef`` — enough for a
    developer mid-refactor to see exactly which cluster and which field moved.
    """
    out = {} if out is None else out
    if isinstance(value, dict):
        if not value:
            out[prefix] = "{}"
        for k, v in value.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(value, list):
        if not value:
            out[prefix] = "[]"
        for i, v in enumerate(value):
            flatten(v, f"{prefix}[{_label_for(i, v)}]", out)
    else:
        out[prefix] = value
    return out


def _render(value: Any, width: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= width else text[: width - 3] + "..."


def diff_report(
    expected: dict[str, Any], actual: dict[str, Any], max_per_section: int = 25
) -> str:
    """A human-readable account of what changed between two projections."""
    exp_flat = flatten(expected)
    act_flat = flatten(actual)

    changed = sorted(k for k in exp_flat.keys() & act_flat.keys() if exp_flat[k] != act_flat[k])
    removed = sorted(exp_flat.keys() - act_flat.keys())
    added = sorted(act_flat.keys() - exp_flat.keys())

    lines: list[str] = []

    def section(title: str, keys: list[str], render) -> None:
        if not keys:
            return
        lines.append(f"{title} ({len(keys)}):")
        for k in keys[:max_per_section]:
            lines.append(f"  {render(k)}")
        if len(keys) > max_per_section:
            lines.append(f"  ... and {len(keys) - max_per_section} more")
        lines.append("")

    section(
        "CHANGED",
        changed,
        lambda k: f"{k}\n      expected: {_render(exp_flat[k])}\n      actual:   {_render(act_flat[k])}",
    )
    section("MISSING (in the fixture, gone from this run)", removed, lambda k: f"{k} = {_render(exp_flat[k])}")
    section("NEW (not in the fixture)", added, lambda k: f"{k} = {_render(act_flat[k])}")

    if not lines:
        return (
            "No field-level difference found, yet the hashes differ. That means the "
            "canonical JSON serialisation itself changed (key order, indentation, or "
            "SCHEMA_VERSION) rather than any artifact value."
        )
    return "\n".join(lines).rstrip()
