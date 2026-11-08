"""The golden-output test — the safety net the refactor lands behind.

IMPLEMENTATION.md §7 Phase A.4 calls this "the single most valuable artifact in
the plan". It hashes a normalised projection of the deterministic interpretation
artifacts for the demo study and fails if any byte of that projection moves:

  * the 9 per-cluster verdicts      ``pipeline.v1_verdicts.verdicts()``
  * the verdict CSV                 ``core.export.to_csv`` (11 frozen columns)
  * the holistic review             ``agent.holistic.holistic_review()``
  * the calibration table           ``pipeline.calibration.calibration_markdown``
  * the 9 per-cluster enrichments   ``agent.enrichment`` + its 14-column CSV
  * the cross-cluster themes        ``agent.enrichment_themes.pathway_themes()``

It is two guarantees at once. For the refactor: 23k lines can be moved, renamed
and de-singleton-ed, and if a single confidence band, driving marker, CSV cell
or coherence note changes, this goes red with the field named. For the paper: the
committed hash IS the reproducibility claim — a fresh clone recomputing these
artifacts from the committed inputs must land on the same bytes.

Design notes:

  * the projection is built by ``tests/golden_projection.py``, the same module
    ``scripts/regenerate_golden.py`` writes from, so checked and generated
    values cannot drift
  * it never calls ``pipeline.run.run()``, which writes into the study tree and
    reaches for the live annotate stage. The four artifacts are
    Tier A pure functions of the committed ``inputs/``, so they are computed
    directly. No network, no LLM, no writes into the repo
  * floats are pinned at ``{:.6f}`` and mappings are key-sorted, so the hash
    moves on a BEHAVIOUR change and not on float ``repr`` or dict-order drift
  * process-global ``lru_cache``s are cleared by the autouse fixture in
    ``tests/conftest.py`` — this module relies on it and does not re-implement it
  * regeneration is never automatic. A failure names
    ``scripts/regenerate_golden.py --write``; accepting a change is a reviewable
    diff in ``tests/golden/``
"""

from __future__ import annotations

import json

import pytest

from panoscope.agent import config as cfg
from panoscope.core.export import PROFILES
from panoscope.core.verdict import BANDS
from panoscope.pipeline import serialize, store
from panoscope.types.outputs import TaskKind
from tests import golden_projection as G


@pytest.fixture(scope="module", autouse=True)
def _demo_study_only() -> None:
    """The fixture pins one study; skip loudly rather than fail elsewhere."""
    if cfg.DATASET_ID != G.DEMO_ID:
        pytest.skip(
            f"golden fixture pins {G.DEMO_ID}; this process opened {cfg.DATASET_ID} "
            f"(PANOSCOPE_DATASET)"
        )


@pytest.fixture
def projection() -> dict:
    return G.build_projection()


# --------------------------------------------------------------------------- #
# The golden check
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-18-02")
def test_interpretation_projection_is_unchanged(projection, tmp_path) -> None:
    """Every byte of the normalised interpretation projection is as committed."""
    expected = G.load_fixture()
    expected_hash = G.projection_hash(expected)
    actual_hash = G.projection_hash(projection)

    if actual_hash == expected_hash:
        return

    # Hand the developer the artifact and the diff, not a bare hash mismatch.
    actual_path = tmp_path / "demo_interpretation.actual.json"
    actual_path.write_text(G.canonical_json(projection), encoding="utf-8")

    pytest.fail(
        "\n".join(
            [
                "",
                "The deterministic interpretation output changed.",
                "",
                f"  expected sha256 : {expected_hash}",
                f"  actual   sha256 : {actual_hash}",
                f"  fixture         : {G.FIXTURE_PATH}",
                f"  this run        : {actual_path}",
                "",
                G.diff_report(expected, projection),
                "",
                "-" * 72,
                "If this change is WRONG, it is a regression — fix the code.",
                "If this change is INTENDED, accept it deliberately:",
                f"    {G.REGEN_COMMAND}",
                "and commit the resulting tests/golden/ diff as the record of the change.",
                "",
                f"To inspect the whole projection: diff {G.FIXTURE_PATH} {actual_path}",
            ]
        ),
        pytrace=False,
    )


@pytest.mark.req("PLAN-P2.5-01")
def test_a_new_reference_release_moves_the_hash_and_a_corpus_does_not(
        projection, monkeypatch) -> None:
    """P2.5: the reference release enters the factual-layer hash; the literature
    corpus never does. Hashes are compared in memory; nothing is regenerated."""
    import dataclasses

    from panoscope.corpus.store import Corpus, SourceRecord, corpus_path, save

    base = G.projection_hash(projection)
    assert projection["reference"], "no release recorded: the check would be vacuous"

    # A corpus on disk for this study changes nothing in the hash.
    save(Corpus(records=(SourceRecord(uri="pmid:10000001", title="Synthetic, not a paper"),),
                built="2026-11-09"), corpus_path(cfg.DATASET_ID))
    assert G.projection_hash(G.build_projection()) == base

    # A different release of the same reference sources moves it, by name.
    real = G.reference_view.for_study
    monkeypatch.setattr(G.reference_view, "for_study", lambda study: dataclasses.replace(
        real(study), releases=tuple((n, r + "-next") for n, r in real(study).releases)))
    moved = G.build_projection()
    assert G.projection_hash(moved) != base
    assert {k for k in moved if moved[k] != projection[k]} == {"reference"}


def test_committed_hash_matches_committed_fixture() -> None:
    """The ``.sha256`` file agrees with the ``.json`` beside it.

    Guards the one way the golden check could be defeated: hand-editing the
    fixture (or the hash) instead of regenerating both together.
    """
    fixture = G.load_fixture()
    assert G.projection_hash(fixture) == G.load_expected_hash(), (
        f"{G.HASH_PATH.name} does not match {G.FIXTURE_PATH.name}. Both are written "
        f"together by `{G.REGEN_COMMAND}` — do not hand-edit either."
    )


@pytest.mark.req("ARCH-18-02")
def test_fixture_bytes_are_canonical() -> None:
    """The committed fixture is byte-for-byte the canonical rendering.

    So a reviewer's diff shows only real value changes, never reformatting.
    """
    fixture = G.load_fixture()
    assert G.FIXTURE_PATH.read_text(encoding="utf-8") == G.canonical_json(fixture), (
        f"{G.FIXTURE_PATH.name} is not in canonical form; rewrite it with "
        f"`{G.REGEN_COMMAND}`."
    )


# --------------------------------------------------------------------------- #
# Guards on the projection itself — a hash is only as good as what it covers
# --------------------------------------------------------------------------- #
def test_projection_is_deterministic_within_a_process(projection) -> None:
    """Two builds in one process agree — no set iteration, no clock, no rng."""
    again = G.build_projection()
    assert G.canonical_json(again) == G.canonical_json(projection), (
        "build_projection() is not deterministic: two calls in the same process "
        "disagreed. Something in the artifacts depends on iteration order, time, "
        "or randomness — the golden hash cannot protect anything until that is fixed."
    )


def test_projection_covers_all_artifacts(projection) -> None:
    """Every artifact is actually present, so the hash cannot pass vacuously."""
    assert projection["dataset_id"] == G.DEMO_ID
    assert projection["cluster_order"] == list(cfg.CLUSTER_ORDER)
    # The release the canonical lists came from (§17.2); empty would pin nothing.
    assert projection["reference"], "no reference release in the projection"

    verdicts = projection["verdicts"]
    assert sorted(verdicts) == sorted(cfg.CLUSTER_ORDER)
    for cluster, v in verdicts.items():
        assert v["cluster"] == cluster
        assert v["cell_type"], f"{cluster} has no cell type"
        assert v["confidence"] in set(BANDS)
        assert v["evidence"], f"{cluster} carries no marker evidence"
        assert v["source_trace"], f"{cluster} carries no source trace"

    # CSV: header + one row per cluster.
    assert len(projection["verdicts_csv"]) == len(cfg.CLUSTER_ORDER) + 1
    # Calibration: header + separator + one row per cluster.
    assert len(projection["calibration_table"]) == len(cfg.CLUSTER_ORDER) + 2

    holistic = projection["holistic"]
    assert holistic["coherence_notes"], "holistic review carries no coherence notes"
    assert "refinements" in holistic and "set_is_coherent" in holistic


def test_projection_covers_the_enrichment_workflow(projection) -> None:
    """The second workflow is in the hash — the blindness this schema version fixed.

    Until SCHEMA_VERSION 2 the projection never touched ``agent.enrichment``, so
    gutting the panel-coverage rule (``MIN_PANEL_HITS`` 3 -> 1) or flattening
    ``ENRICHMENT_BANDS`` left the golden hash byte-identical. These assertions are
    what stops that regressing: the enrichment half must be present, non-empty,
    and must carry the panel-coverage fields the gate reads.
    """
    enrichment = projection["enrichment"]
    assert sorted(enrichment) == sorted(cfg.CLUSTER_ORDER)
    assert projection["enrichment_csv_columns"] == list(PROFILES[TaskKind.CHARACTERISATION].names)

    surfaced = 0
    for cluster, e in enrichment.items():
        assert e["cluster"] == cluster
        assert e["method"] == "jazzpanda_enrichment"
        assert e["all_tested"], f"{cluster} carries no tested gene sets"
        assert e["source_trace"], f"{cluster} carries no source trace"
        # No band, and the hash is where that has to be pinned: a band reintroduced
        # as a default-valued field would draw nothing on screen and still be written
        # into every committed record, which is exactly how the deleted one survived
        # being taken off the Pathways page.
        assert "confidence" not in e and "confidence_score" not in e, cluster
        assert "demotions" not in e, cluster
        surfaced += len(e["enriched"]) + len(e["suggestive"])
        for p in e["all_tested"]:
            # The gate's inputs must be in the hash, or a threshold change is invisible.
            assert p["tier"] in {"enriched", "suggestive", "untestable"}
            assert "panel_hits" in p and "n_leading_edge" in p and "q_value" in p
            assert p["panel_scope_caveat"].startswith("Panel-scoped:")

    assert surfaced, "no gene set is surfaced anywhere — the hash would be vacuous"
    # CSV: header + one row per surfaced (enriched|suggestive) pathway.
    assert len(projection["enrichment_csv"]) == surfaced + 1

    themes = projection["pathway_themes"]
    assert themes["per_cluster"], "pathway themes carry no per-cluster lines"
    assert "recurring" in themes and "coherence_notes" in themes
    # The census is what the band became, so the hash has to see it: a per-cluster
    # line that lost its counts would otherwise regress silently to a bare name.
    for line in themes["per_cluster"]:
        assert "confidence" not in line, line["cluster"]
        assert "untestable" in line["census_line"], line["cluster"]


def test_enrichment_half_reproduces_the_committed_records(projection) -> None:
    """The recomputed enrichment equals the records the pipeline shipped.

    ``pinned_enrichment()`` feeds the adapter raw rows reversed out of
    ``interp/enrichment/c*.json`` and recomputes every DERIVED field. If that
    recomputation agrees with the committed records, the reversal is faithful and
    the golden's enrichment half is the demo study's real output — not an
    artifact of the reconstruction.
    """
    committed = store.load_all_enrichments()
    assert committed is not None, "the committed enrichment records are missing"
    expected = G.normalise({e.cluster: serialize.enrichment_to_dict(e) for e in committed})
    assert projection["enrichment"] == expected, (
        "the recomputed enrichment no longer matches data/datasets/<id>/interp/"
        "enrichment/*.json. Either interpretation behaviour changed (regenerate the "
        "study tree AND the golden), or the raw-row reversal in pinned_enrichment() "
        "lost a field."
    )


def test_csv_column_contract_is_frozen(projection) -> None:
    """The 11 CSV columns, in exact order, are a contract (CLAUDE.md Output format)."""
    expected_columns = G.load_fixture()["csv_columns"]
    assert projection["csv_columns"] == expected_columns
    assert list(PROFILES[TaskKind.ASSIGNMENT].names) == expected_columns
    assert len(expected_columns) == 11
    # The rendered CSV header must match the declared contract, not just the constant.
    assert projection["verdicts_csv"][0] == ",".join(expected_columns)


def test_projection_carries_no_floats_or_absolute_paths(projection) -> None:
    """Normalisation actually happened: no raw floats, no machine-specific paths.

    A raw float would make the hash sensitive to ``repr`` drift; an absolute path
    would make the fixture unreproducible on another machine.
    """
    flat = G.flatten(projection)
    floats = sorted(k for k, v in flat.items() if isinstance(v, float))
    assert not floats, f"un-normalised floats in the projection: {floats[:10]}"

    blob = json.dumps(projection, ensure_ascii=False)
    assert str(cfg.PROJECT_ROOT) not in blob, "the projection embeds an absolute path"
