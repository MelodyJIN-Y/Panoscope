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

from panoscope.agent import config as cfg
from panoscope.agent import holistic as agent_holistic
from panoscope.agent import verdict as V
from panoscope.pipeline import calibration as calibration_mod
from panoscope.pipeline import serialize

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
SCHEMA_VERSION: int = 1

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
# Projection
# --------------------------------------------------------------------------- #
def build_projection() -> dict[str, Any]:
    """Compute the normalised projection from the committed study inputs.

    Pure and side-effect free: reads ``data/datasets/<id>/inputs/`` (via
    ``agent.data`` / ``agent.config``), writes nothing, touches no network.
    """
    verdicts = V.all_verdicts()
    review = agent_holistic.holistic_review()
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
