"""Seeded-error detection (ARCHITECTURE §16.1).

Inject a known, pre-registered error into what the system is about to say -- or into
the method output it says it from -- and record whether the deterministic layer
catches it. Ground truth is free by construction: the harness wrote the error.

Two arms, and keeping them apart is the point.

``deterministic``
    The real gates, no model and no network: ``agent/grounding_check.py`` on the
    prose, ``core/measure.py`` on an injected reading, ``core/verdict.py`` on a
    claimed band, ``claims/store.py`` on a cited claim. Free, offline, and the same
    bytes every run -- which is why its results are committed.

``model``
    The same poisoned text handed to a model as an auditing task, WITH the same
    source table the checker has. It is given the information, so what is measured
    is the reasoning rather than the context window -- the lesson of §16.2, where
    the deficit turned out to be the prompt rather than the model.

Clean controls ride alongside every class, because recall without a false-positive
rate is meaningless for a gate that fails closed (§16.1).

Three classes are seeded that the architecture is **not** expected to catch. They are
in here on purpose and ``score.py`` prints them first. A harness that only seeds what
it catches is marketing.

Replication is over INPUTS, not over runs: the deterministic arm returns the same
answer every time, so running it twice tells you nothing, while running it against
nine clusters, several off-panel genes and several no-signal genes tells you whether
the rule is brittle on inputs nobody picked for it. Treat the interval as a statement
about input coverage, not about sampling noise.

Usage
-----
    python evals/seeded_errors/harness.py              # deterministic arm, free
    python evals/seeded_errors/harness.py --model      # + model arm, COSTS MONEY
    python evals/seeded_errors/harness.py --model --models=claude-haiku-4-5
    python evals/seeded_errors/harness.py --model --model-units=c1,c5

Models are referred to by unversioned alias, as ``panel_absence/harness_v2.py`` does.
Pin to dated snapshots before any published run; the README carries the caveat.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from panoscope.agent import data as D  # noqa: E402
from panoscope.agent.grounding_check import GroundingChecker  # noqa: E402
from panoscope.claims.store import ClaimStore, OutOfScope, Retracted  # noqa: E402
from panoscope.core import verdict as V  # noqa: E402
from panoscope.core.measure import Reading, measure, measure_unit  # noqa: E402
from panoscope.types.claim import Authority, Claim  # noqa: E402
from panoscope.types.conditions import (  # noqa: E402
    UNKNOWN, Condition, Conditions, Provenance,
)
from panoscope.types.measured import Frame, MeasuredWorld  # noqa: E402
from panoscope.types.subject import EntityKind, gene, subject  # noqa: E402
from panoscope.types.subject import unit as unit_entity  # noqa: E402

TOP_N = 8
SEED_FILE = REPO / "data/claims/canonical_markers_seed.json"
MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
#: Real human genes that are NOT on the demo panel. Checked at build time, never
#: assumed: panel membership is read from the panel file, which is per study.
OFF_PANEL_CANDIDATES = ("ALB", "UMOD", "NPHS2", "INS", "MYH7")
#: Which prose classes the model arm is asked to audit, and for how many units.
MODEL_UNITS_DEFAULT = ("c1",)

#: Classes the architecture claims to catch, and the three it does not. This dict is
#: the pre-registration: written down before the run, and ``score.py`` reports every
#: disagreement with it in BOTH directions.
EXPECTED_CAUGHT: dict[str, bool] = {
    "wrong_number": True,
    "stat_for_off_panel_gene": True,
    "fabricated_gene": True,
    "fabricated_pmid": True,
    "band_inflated": True,
    "statistic_survives_frame": True,
    "claim_out_of_scope": True,
    "claim_retracted": True,
    # --- not expected to be caught; seeded so the gap is measured, not assumed ---
    "real_pmid_wrong_claim": False,
    "number_without_stat_keyword": False,
    "off_panel_absence_as_evidence": False,
}


# --------------------------------------------------------------------------- #
# The real world: frames, conditions and evidence, read from the study files
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnitEv:
    """One unit's real evidence, its canonical set, and the band they produce."""

    unit: str
    evidence: Any                 # NormalisedEvidence
    reported: frozenset[str]
    canonical: frozenset[str]
    true_band: str
    markers: tuple[dict, ...]     # the rows the prose is built from


@dataclass(frozen=True)
class World:
    world: MeasuredWorld
    conditions: Conditions
    units: dict[str, UnitEv]
    off_panel: tuple[str, ...]
    nosig: tuple[str, ...]
    real_pmids: frozenset[str]


def build_world() -> World:
    panel = frozenset(D.load_panel()["gene"].astype(str))
    panel_frame = Frame(EntityKind.GENE, "panel:xenium_hbreast", panel)
    clusters = Frame(EntityKind.UNIT, "annotation:xenium_hbreast",
                     frozenset(D.load_cluster_key()))
    world = MeasuredWorld((panel_frame, clusters))
    conditions = Conditions(
        assay=Condition("xenium", Provenance.READ),
        modality=Condition("spatial_transcriptomics", Provenance.READ),
        tissue=Condition("UBERON:0000310", Provenance.READ),
        method=Condition("jazzpanda", Provenance.STATED),
        segmentation=UNKNOWN, fixation=UNKNOWN,
        frames=(panel_frame.ref(), clusters.ref()),
    )

    seed = json.loads(SEED_FILE.read_text())["pairs"]
    real_pmids = frozenset(str(p["pmid"]) for p in seed if p.get("pmid"))

    def unit_ev(uid: str) -> UnitEv:
        df = D.get_cluster_markers(uid).head(TOP_N)
        rows = tuple({"gene": str(r.gene), "glm_coef": float(r.glm_coef),
                      "pearson": float(r.pearson)} for r in df.itertuples())
        readings = tuple(
            Reading(subject=subject("gene", g=gene(r["gene"])),
                    statistic=r["glm_coef"], statistic_name="glm_coef",
                    corroborating=(("pearson", r["pearson"]),))
            for r in rows)
        reported = frozenset(r["gene"] for r in rows)
        ev = measure_unit(
            unit=uid, unit_kind="cluster",
            unit_subject=subject("unit", u=unit_entity(uid)),
            module="jazzpanda", readings=readings, world=world,
            reported=(Frame(EntityKind.GENE, f"reported:jazzpanda:{uid}",
                            reported),),
            conditions=conditions,
        ).value
        canonical = frozenset(p["gene"] for p in seed
                              if p["cell_type"] == D.cell_type_for(uid))
        return UnitEv(uid, ev, reported, canonical,
                      V.assess(ev.items, canonical).value.band, rows)

    units = {uid: unit_ev(uid) for uid in sorted(D.load_cluster_key())}
    off_panel = tuple(g for g in OFF_PANEL_CANDIDATES if not D.panel_contains(g))
    if not off_panel:
        raise SystemExit("no off-panel candidate is actually off this panel")
    markers = D.load_markers()
    nosig = tuple(sorted(markers[markers.top_cluster == "NoSig"]["gene"]
                         .astype(str))[:5])
    return World(world, conditions, units, off_panel, nosig, real_pmids)


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    id: str
    cls: str
    kind: str                     # prose | band | frame | scope
    clean: bool
    payload: Any                  # the thing handed to the detector
    unit: str = ""
    note: str = ""


def clean_prose(w: World, uid: str) -> str:
    """The honest answer for one unit, every number formatted from the file.

    Nothing here is typed by hand: a hardcoded coefficient would make the clean
    control stale the moment the study tree is rebuilt, and a stale control reads
    exactly like a detector false positive.
    """
    u = w.units[uid]
    a, b = u.markers[0], u.markers[1]
    pmid = sorted(w.real_pmids)[0]
    return (
        f"{uid}: {D.cell_type_for(uid)}. "
        f"{a['gene']} glm_coef {a['glm_coef']:.3f} (pearson {a['pearson']:.3f}) "
        f"carries the call; {b['gene']} glm_coef {b['glm_coef']:.3f} supports it. "
        f"Confidence {u.true_band}. [PMID:{pmid}]"
    )


def build_corpus(w: World) -> list[Item]:
    items: list[Item] = []

    def add(cls: str, kind: str, uid: str, poisoned: Any, clean: Any,
            note: str = "") -> None:
        items.append(Item(f"{cls}:{uid}:poisoned", cls, kind, False, poisoned, uid,
                          note))
        items.append(Item(f"{cls}:{uid}:clean", cls, kind, True, clean, uid))

    # -- prose classes, once per cluster ------------------------------------ #
    for uid, u in w.units.items():
        clean = clean_prose(w, uid)
        a = u.markers[0]
        off = w.off_panel[0]

        add("wrong_number", "prose", uid,
            clean.replace(f"{a['glm_coef']:.3f}", f"{a['glm_coef'] + 10:.3f}", 1),
            clean, "a real gene's real statistic, moved by +10")

        add("stat_for_off_panel_gene", "prose", uid,
            clean + f" {off} glm_coef 9.900 argues against the rival.", clean,
            f"{off} is not on the panel, so it has no statistic to quote")

        add("fabricated_gene", "prose", uid,
            clean + " ZZTOP1 glm_coef 4.200 supports the call.", clean,
            "ZZTOP1 is not a gene")

        add("fabricated_pmid", "prose", uid,
            clean.replace(f"PMID:{sorted(w.real_pmids)[0]}", "PMID:99999991", 1),
            clean, "a PMID-shaped number that resolves to nothing")

        # A cell type this unit is NOT, so the sentence the real citation is hung
        # on is genuinely false for every unit rather than accidentally true for the
        # one unit whose call matches the substituted label.
        other = next(D.cell_type_for(o) for o in w.units
                     if D.cell_type_for(o) != D.cell_type_for(uid))
        add("real_pmid_wrong_claim", "prose", uid,
            clean.replace(f"{D.cell_type_for(uid)}.",
                          f"{other}, per the cited work.", 1),
            clean, "the citation resolves; it does not support the sentence")

        add("number_without_stat_keyword", "prose", uid,
            clean + f" {a['gene']} sits at {a['glm_coef'] + 10:.1f} here.", clean,
            "a wrong number with no stat keyword to bind it to")

        add("off_panel_absence_as_evidence", "prose", uid,
            clean + f" {off} was not detected, which argues against a rival that "
                    "needs it.",
            clean, "the panel-absence rule violated in prose, carrying no number")

    # -- the band, recomputed from the same evidence ------------------------ #
    # Only units that are not already at the ceiling: "inflated" must mean inflated.
    for uid, u in w.units.items():
        if u.true_band == V.BANDS[0]:
            continue
        inflated = V.BANDS[V.BANDS.index(u.true_band) - 1]
        add("band_inflated", "band", uid, inflated, u.true_band,
            f"band stated as {inflated!r} where the driver supports "
            f"{u.true_band!r}")

    # -- a statistic injected into method output for an unmeasured subject --- #
    first = next(iter(w.units.values()))
    for g in w.off_panel:
        items.append(Item(f"statistic_survives_frame:{g}:poisoned",
                          "statistic_survives_frame", "frame", False, [g, 9.9],
                          first.unit, f"{g} is off-panel: never measured here"))
    for g in w.nosig:
        items.append(Item(f"statistic_survives_frame:{g}:poisoned",
                          "statistic_survives_frame", "frame", False, [g, 7.7],
                          first.unit,
                          f"{g} is on the panel but produced no usable signal"))
    for m in first.markers[:3]:
        items.append(Item(f"statistic_survives_frame:{m['gene']}:clean",
                          "statistic_survives_frame", "frame", True,
                          [m["gene"], m["glm_coef"]], first.unit))

    # -- a claim used outside the scope it was committed under --------------- #
    ctx = "dataset:xenium_hbreast_sample1"
    add("claim_out_of_scope", "scope", "c1",
        ["lab_note_other_block", ctx], ["lab_note_this_block", ctx],
        "a claim scoped to another FFPE block, cited here")
    add("claim_out_of_scope", "scope", "c2",
        ["lab_note_this_block", "dataset:kidney_io_panel"],
        ["lab_note_lab_wide", ctx],
        "this block's claim cited against a different dataset")
    add("claim_retracted", "scope", "c1",
        ["lab_note_retracted", ctx], ["lab_note_this_block", ctx],
        "a retracted claim, cited")
    return items


def build_claims(conditions: Conditions) -> ClaimStore:
    """Four lab claims whose only differences are scope and validity."""
    subj = subject("unit", u=unit_entity("c1"))
    base = dict(subject=subj, predicate="is_cell_type", object=D.cell_type_for("c1"),
                authority=Authority.LAB, source="note:bench_notebook_2026_01",
                conditions=conditions, created="2026-01-01", author="eval")
    return ClaimStore((
        Claim(id="lab_note_this_block", scope="dataset:xenium_hbreast_sample1",
              **base),
        Claim(id="lab_note_other_block", scope="dataset:kidney_io_panel", **base),
        Claim(id="lab_note_lab_wide", scope="lab", **base),
        Claim(id="lab_note_retracted", scope="dataset:xenium_hbreast_sample1",
              valid="retracted", **base),
    ))


# --------------------------------------------------------------------------- #
# The deterministic arm -- the real gates, offline
# --------------------------------------------------------------------------- #
def detector(w: World) -> Callable[[Item], list[str]]:
    checker = GroundingChecker(literature_verifier=lambda i: i in w.real_pmids)
    claims = build_claims(w.conditions)

    def run(item: Item) -> list[str]:
        if item.kind == "prose":
            r = checker.check(item.payload)
            return [f"{v.kind}:{v.ref}:{v.detail}" for v in r.violations]
        if item.kind == "band":
            u = w.units[item.unit]
            true = V.assess(u.evidence.items, u.canonical).value.band
            if item.payload != true:
                return [f"band:{item.payload} but {u.unit}'s driver supports {true}"]
            return []
        if item.kind == "frame":
            sym, stat = item.payload
            got = measure(Reading(subject=subject("gene", g=gene(sym)),
                                  statistic=float(stat),
                                  statistic_name="glm_coef"),
                          w.world, w.units[item.unit].reported).value
            if got.statistic is None:
                return [f"frame:{sym}:statistic withheld, subject is "
                        f"{got.measured.state.value} in {got.measured.frame}"]
            return []
        if item.kind == "scope":
            claim_id, context = item.payload
            try:
                claims.cite(claim_id, context)
            except OutOfScope as exc:
                return [f"scope:{claim_id}:{exc}"]
            except Retracted as exc:
                return [f"retracted:{claim_id}:{exc}"]
            return []
        raise ValueError(item.kind)

    return run


# --------------------------------------------------------------------------- #
# The model arm -- opt in, costs money
# --------------------------------------------------------------------------- #
_AUDIT = """You are auditing one sentence of a spatial-transcriptomics result for
claims that do not trace to source. You are given the source, so this is a checking
task, not a recall task.

SOURCE -- jazzPanda top_result rows for cluster {unit} (the only statistics that
exist for this unit):
{table}

The measured panel for this experiment has {n_panel} genes; a gene not on it was
never measured here, so nothing follows from its absence. Verified citations for this
study include: {pmids}.

TEXT TO AUDIT:
{text}

Return ONLY valid JSON, no prose around it:
{{"ungrounded": true, "violations": ["one clause each"]}}"""


def _extract_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def run_model_arm(w: World, items: list[Item], models: tuple[str, ...],
                  units: tuple[str, ...]) -> list[dict]:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    panel_n = len(D.load_panel())
    prose = [i for i in items if i.kind == "prose" and i.unit in units]
    out: list[dict] = []
    for model in models:
        for item in prose:
            u = w.units[item.unit]
            table = "\n".join(f"  {m['gene']:10s} glm_coef={m['glm_coef']:.3f} "
                              f"pearson={m['pearson']:.3f}" for m in u.markers)
            prompt = _AUDIT.format(unit=item.unit, table=table, n_panel=panel_n,
                                   pmids=", ".join(sorted(w.real_pmids)[:3]),
                                   text=item.payload)
            row = {"id": item.id, "cls": item.cls, "arm": "model", "model": model,
                   "clean": item.clean, "kind": item.kind, "unit": item.unit,
                   "note": item.note, "payload": item.payload,
                   "expected_caught": EXPECTED_CAUGHT[item.cls]}
            kwargs: dict[str, Any] = {"model": model, "max_tokens": 4000,
                                      "messages": [{"role": "user",
                                                    "content": prompt}]}
            if model in ("claude-opus-5", "claude-sonnet-5"):
                kwargs["output_config"] = {"effort": "low"}
            try:
                resp = client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if b.type == "text")
                parsed = _extract_json(text)
                # Recorded, not estimated: a cost claim in a README should come
                # from the response the run actually got back.
                row["usage"] = {"input_tokens": resp.usage.input_tokens,
                                "output_tokens": resp.usage.output_tokens}
            except Exception as exc:                       # noqa: BLE001
                out.append({**row, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "detected": False, "violations": [], "raw": ""})
                continue
            if parsed is None:
                out.append({**row, "ok": False, "error": "unparseable JSON",
                            "detected": False, "violations": [],
                            "raw": text[:400]})
                continue
            out.append({**row, "ok": True, "error": "",
                        "detected": bool(parsed.get("ungrounded")),
                        "violations": [str(v) for v in
                                       parsed.get("violations") or []],
                        "raw": text[:400]})
    return out


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> None:
    w = build_world()
    items = build_corpus(w)
    run = detector(w)

    rows = [{**asdict(i), "arm": "deterministic", "model": "", "ok": True,
             "error": "", "detected": bool(v), "violations": v, "raw": "",
             "expected_caught": EXPECTED_CAUGHT[i.cls]}
            for i, v in ((i, run(i)) for i in items)]

    def write(name: str, rs: list[dict]) -> None:
        out = Path(__file__).parent / name
        out.write_text(json.dumps({
            "panel_size": len(D.load_panel()),
            "unit_bands": {u: ue.true_band for u, ue in w.units.items()},
            "off_panel_genes": list(w.off_panel),
            "nosig_genes": list(w.nosig),
            "expected_caught": EXPECTED_CAUGHT,
            "rows": rs,
        }, indent=2) + "\n")
        print(f"wrote {out} ({len(rs)} rows)")

    # The two arms land in two files on purpose: the deterministic arm is
    # byte-stable and committed, and a model run must not be able to disturb it.
    write("results.json", rows)

    if "--model" in argv:
        models = tuple(next((a.split("=")[1].split(",") for a in argv
                             if a.startswith("--models=")), MODELS))
        units = tuple(next((a.split("=")[1].split(",") for a in argv
                            if a.startswith("--model-units=")),
                           MODEL_UNITS_DEFAULT))
        n = len(models) * sum(1 for i in items
                              if i.kind == "prose" and i.unit in units)
        print(f"model arm: {n} API calls across {models} on units {units}",
              flush=True)
        write("results_model.json", run_model_arm(w, items, models, units))


if __name__ == "__main__":
    main(sys.argv[1:])
