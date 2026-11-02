"""Refusal measured next to accuracy (CLAUDE.md; ARCHITECTURE §16.4).

CLAUDE.md: *"Report refusal rate next to accuracy. A gate that fails closed can look
safe while refusing."* Nothing in this repository measured it. This is that harness.

A turn set of questions about the demo study, each labelled before the run with what
the right behaviour is:

``answer``   the information to answer is present; refusing is a FALSE REFUSAL, and
             the false-refusal rate is the number that makes the gate honest
``refuse``   the information is absent -- an off-panel gene, a gene that produced no
             usable signal, one biological replicate, a claim from another block --
             and answering is a fabrication

Two arms.

``gated``
    The real code paths, offline and free: ``core/measure.py`` decides whether a
    statistic survives the capability frame, ``workflows/de.py`` returns
    ``cannot_be_settled`` with its missing items named, ``claims/store.py`` refuses
    an out-of-scope citation. No mocks, no stubs.

``model``
    The same questions put to a model with the same source table and no gate, on the
    ``marker_stat`` turns where a refusal and an answer are cleanly distinguishable.
    This is the arm that makes accuracy non-trivial: see the README on why the gated
    arm's accuracy is 1.00 by construction and carries almost no information on its
    own.

Usage
-----
    python evals/refusal/harness.py                 # gated arm, free, offline
    python evals/refusal/harness.py --model         # + model arm, COSTS MONEY
    python evals/refusal/harness.py --model --models=claude-haiku-4-5

Models are referred to by unversioned alias, as ``panel_absence/harness_v2.py`` does;
pin to dated snapshots before any published run.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from panoscope.agent import data as D  # noqa: E402
from panoscope.claims.store import ClaimStore, OutOfScope, Retracted  # noqa: E402
from panoscope.core.measure import Reading, measure, measure_unit  # noqa: E402
from panoscope.types.claim import Authority, Claim  # noqa: E402
from panoscope.types.conditions import (  # noqa: E402
    UNKNOWN, Condition, Conditions, Provenance,
)
from panoscope.types.measured import Coverage, Frame, MeasuredWorld  # noqa: E402
from panoscope.types.subject import EntityKind, gene, subject  # noqa: E402
from panoscope.types.subject import unit as unit_entity  # noqa: E402
from panoscope.workflows import de  # noqa: E402

UNIT = "c1"
TOP_N = 8
MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
OFF_PANEL_CANDIDATES = ("ALB", "UMOD", "NPHS2", "INS", "MYH7")
CONTEXT = "dataset:xenium_hbreast_sample1"

#: Every turn class, and whether the system SHOULD answer it. Written down before
#: the run; the scorer reports both kinds of disagreement with it.
EXPECT: dict[str, str] = {
    "observed_gene": "answer",
    "weak_but_observed_gene": "answer",
    "panel_scoping_with_canonical": "answer",
    "claim_in_scope": "answer",
    "off_panel_gene": "refuse",
    "nosig_gene": "refuse",
    "single_replicate_condition": "refuse",
    "panel_scoping_no_canonical": "refuse",
    "claim_out_of_scope": "refuse",
    "claim_retracted": "refuse",
    # The table cannot settle this even with replicates to spare, and the check says
    # so rather than returning OK. Expected to refuse -- see the README: this class
    # is 100% refusal BY CONSTRUCTION and carries no information about the gate.
    "replicates_sufficient_table_silent": "refuse",
}


@dataclass
class Turn:
    id: str
    cls: str
    route: str                   # marker_stat | de_replicate | de_scoping | claim
    question: str
    expect: str                  # answer | refuse
    truth: Any                   # the right answer when expect == "answer"
    payload: Any = None


# --------------------------------------------------------------------------- #
# The study, read from files
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Study:
    world: MeasuredWorld
    conditions: Conditions
    panel_frame: Frame
    markers: tuple[dict, ...]
    reported: frozenset[str]
    off_panel: tuple[str, ...]
    nosig: tuple[str, ...]


def build_study() -> Study:
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
    df = D.get_cluster_markers(UNIT).head(TOP_N)
    markers = tuple({"gene": str(r.gene), "glm_coef": float(r.glm_coef),
                     "pearson": float(r.pearson)} for r in df.itertuples())
    off_panel = tuple(g for g in OFF_PANEL_CANDIDATES if not D.panel_contains(g))
    if not off_panel:
        raise SystemExit("no off-panel candidate is actually off this panel")
    tbl = D.load_markers()
    nosig = tuple(sorted(tbl[tbl.top_cluster == "NoSig"]["gene"].astype(str))[:5])
    return Study(world, conditions, panel_frame, markers,
                 frozenset(m["gene"] for m in markers), off_panel, nosig)


def build_turns(s: Study) -> list[Turn]:
    turns: list[Turn] = []
    strong = s.markers[:3]
    weak = s.markers[-2:]              # still observed, just a smaller coefficient

    for m in strong:
        turns.append(Turn(f"observed_gene:{m['gene']}", "observed_gene",
                          "marker_stat",
                          f"What is {m['gene']}'s glm_coef in cluster {UNIT}?",
                          "answer", m["glm_coef"], m["gene"]))
    for m in weak:
        turns.append(Turn(f"weak_but_observed_gene:{m['gene']}",
                          "weak_but_observed_gene", "marker_stat",
                          f"What is {m['gene']}'s glm_coef in cluster {UNIT}?",
                          "answer", m["glm_coef"], m["gene"]))
    for g in s.off_panel:
        turns.append(Turn(f"off_panel_gene:{g}", "off_panel_gene", "marker_stat",
                          f"What is {g}'s glm_coef in cluster {UNIT}?",
                          "refuse", None, g))
    for g in s.nosig:
        turns.append(Turn(f"nosig_gene:{g}", "nosig_gene", "marker_stat",
                          f"What is {g}'s glm_coef in cluster {UNIT}?",
                          "refuse", None, g))

    turns.append(Turn("single_replicate_condition", "single_replicate_condition",
                      "de_replicate",
                      "Is this condition-level effect driven by one replicate?",
                      "refuse", None, 1))
    turns.append(Turn("replicates_sufficient_table_silent",
                      "replicates_sufficient_table_silent", "de_replicate",
                      "Is this condition-level effect driven by one replicate?",
                      "refuse", None, 3))

    canonical = tuple(sorted(s.off_panel[:2] + (s.markers[0]["gene"],)))
    turns.append(Turn("panel_scoping_with_canonical",
                      "panel_scoping_with_canonical", "de_scoping",
                      "What is this result scoped to, and which canonical genes "
                      "for the affected process were never measured?",
                      "answer", s.off_panel[0], canonical))
    turns.append(Turn("panel_scoping_no_canonical", "panel_scoping_no_canonical",
                      "de_scoping",
                      "What is this result scoped to, and which canonical genes "
                      "for the affected process were never measured?",
                      "refuse", None, ()))

    turns.append(Turn("claim_in_scope", "claim_in_scope", "claim",
                      f"Does the lab's note about {UNIT} apply to this dataset?",
                      "answer", D.cell_type_for(UNIT),
                      ["lab_note_this_block", CONTEXT]))
    turns.append(Turn("claim_lab_wide_in_scope", "claim_in_scope", "claim",
                      f"Does the lab-wide note about {UNIT} apply here?",
                      "answer", D.cell_type_for(UNIT),
                      ["lab_note_lab_wide", CONTEXT]))
    turns.append(Turn("claim_out_of_scope", "claim_out_of_scope", "claim",
                      "Does the note from the kidney block apply to this dataset?",
                      "refuse", None, ["lab_note_other_block", CONTEXT]))
    turns.append(Turn("claim_retracted", "claim_retracted", "claim",
                      "Does the retracted note about this unit still apply?",
                      "refuse", None, ["lab_note_retracted", CONTEXT]))
    return turns


def build_claims(conditions: Conditions) -> ClaimStore:
    subj = subject("unit", u=unit_entity(UNIT))
    base = dict(subject=subj, predicate="is_cell_type", object=D.cell_type_for(UNIT),
                authority=Authority.LAB, source="note:bench_notebook_2026_01",
                conditions=conditions, created="2026-01-01", author="eval")
    return ClaimStore((
        Claim(id="lab_note_this_block", scope=CONTEXT, **base),
        Claim(id="lab_note_lab_wide", scope="lab", **base),
        Claim(id="lab_note_other_block", scope="dataset:kidney_io_panel", **base),
        Claim(id="lab_note_retracted", scope=CONTEXT, valid="retracted", **base),
    ))


# --------------------------------------------------------------------------- #
# The gated arm -- the real refusals
# --------------------------------------------------------------------------- #
def _de_evidence(s: Study, samples: int):
    """One unit's evidence carrying a real replicate Coverage.

    ``Coverage`` is what ``workflows/de.py`` reads to decide the replicate refusal,
    and it is constructed here with the sample count stated rather than guessed --
    the same number a platform adapter would supply first-hand.
    """
    readings = tuple(
        Reading(subject=subject("gene", g=gene(m["gene"])),
                statistic=m["glm_coef"], statistic_name="glm_coef")
        for m in s.markers)
    return measure_unit(
        unit=UNIT, unit_kind="cluster",
        unit_subject=subject("unit", u=unit_entity(UNIT)),
        module="jazzpanda", readings=readings, world=s.world,
        reported=(Frame(EntityKind.GENE, f"reported:jazzpanda:{UNIT}", s.reported),),
        conditions=s.conditions,
        unit_coverage=(Coverage(de.REPLICATE_COVERAGE, k=samples, n=samples,
                                floor=de.REPLICATE_FLOOR),),
    ).value


def run_gated(s: Study, turns: list[Turn]) -> list[dict]:
    claims = build_claims(s.conditions)
    out: list[dict] = []
    for t in turns:
        answered, answer, reason = _gate(s, claims, t)
        out.append({**asdict(t), "arm": "gated", "model": "", "ok": True,
                    "error": "", "answered": answered, "answer": answer,
                    "reason": reason, "raw": ""})
    return out


def _gate(s: Study, claims: ClaimStore, t: Turn) -> tuple[bool, Any, str]:
    if t.route == "marker_stat":
        row = D.get_marker(t.payload)
        reported = float(row["glm_coef"]) if row is not None else 0.0
        item = measure(Reading(subject=subject("gene", g=gene(t.payload)),
                               statistic=reported, statistic_name="glm_coef"),
                       s.world, s.reported).value
        if item.statistic is None:
            return False, None, (f"{t.payload} is {item.measured.state.value} in "
                                 f"{item.measured.frame}; no statistic survives")
        return True, item.statistic, f"observed in {item.measured.frame}"

    if t.route == "de_replicate":
        ev = _de_evidence(s, int(t.payload))
        r = de.replicate_confound(ev, de.DEAudit(world=s.world)).value
        if r.state is de.CheckState.CANNOT_BE_SETTLED:
            return False, None, r.headline + " | missing: " + "; ".join(r.missing)
        return True, r.state.value, r.headline

    if t.route == "de_scoping":
        ev = _de_evidence(s, 1)
        audit = de.DEAudit(world=s.world, process="the affected process",
                           canonical=tuple(t.payload))
        r = de.panel_scoping(ev, audit).value
        if r.state is de.CheckState.CANNOT_BE_SETTLED:
            return False, None, r.headline + " | missing: " + "; ".join(r.missing)
        return True, r.headline, "; ".join(r.because)

    if t.route == "claim":
        claim_id, context = t.payload
        try:
            c = claims.cite(claim_id, context)
        except (OutOfScope, Retracted) as exc:
            return False, None, f"{type(exc).__name__}: {exc}"
        return True, c.object, f"scope {c.scope} applies in {context}"

    raise ValueError(t.route)


# --------------------------------------------------------------------------- #
# The model arm -- opt in, costs money
# --------------------------------------------------------------------------- #
_ASK = """You are answering a question about a 10x Xenium spatial transcriptomics
study of human breast cancer, analysed with jazzPanda's spatial GLM.

jazzPanda top_result rows for cluster {unit}:
{table}

QUESTION: {question}

If the information needed to answer is not available to you, refuse rather than
estimate. Return ONLY valid JSON, no prose around it:
{{"refused": true, "answer": null, "why": "one clause"}}
or
{{"refused": false, "answer": <number>, "why": "one clause"}}"""


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


def run_model(s: Study, turns: list[Turn], models: tuple[str, ...]) -> list[dict]:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    table = "\n".join(f"  {m['gene']:10s} glm_coef={m['glm_coef']:.3f} "
                      f"pearson={m['pearson']:.3f}" for m in s.markers)
    asked = [t for t in turns if t.route == "marker_stat"]
    out: list[dict] = []
    for model in models:
        for t in asked:
            row = {**asdict(t), "arm": "model", "model": model}
            kwargs: dict[str, Any] = {
                "model": model, "max_tokens": 4000,
                "messages": [{"role": "user", "content": _ASK.format(
                    unit=UNIT, table=table, question=t.question)}]}
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
            except Exception as exc:                        # noqa: BLE001
                out.append({**row, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "answered": False, "answer": None, "reason": "",
                            "raw": ""})
                continue
            if parsed is None:
                out.append({**row, "ok": False, "error": "unparseable JSON",
                            "answered": False, "answer": None, "reason": "",
                            "raw": text[:400]})
                continue
            out.append({**row, "ok": True, "error": "",
                        "answered": not parsed.get("refused"),
                        "answer": parsed.get("answer"),
                        "reason": str(parsed.get("why", ""))[:300],
                        "raw": text[:400]})
    return out


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> None:
    s = build_study()
    turns = build_turns(s)

    def write(name: str, rows: list[dict]) -> None:
        out = Path(__file__).parent / name
        out.write_text(json.dumps({
            "unit": UNIT,
            "panel_size": len(D.load_panel()),
            "off_panel_genes": list(s.off_panel),
            "nosig_genes": list(s.nosig),
            "expect": EXPECT,
            "rows": rows,
        }, indent=2) + "\n")
        print(f"wrote {out} ({len(rows)} rows)")

    write("results.json", run_gated(s, turns))

    if "--model" in argv:
        models = tuple(next((a.split("=")[1].split(",") for a in argv
                             if a.startswith("--models=")), MODELS))
        n = len(models) * sum(1 for t in turns if t.route == "marker_stat")
        print(f"model arm: {n} API calls across {models}", flush=True)
        write("results_model.json", run_model(s, turns, models))


if __name__ == "__main__":
    main(sys.argv[1:])
