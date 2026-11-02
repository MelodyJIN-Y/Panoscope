"""Generator swap and deterministic invariance (ARCHITECTURE §16.3).

The architecture's claim is narrow and it is the only one worth testing here:

    swap the model that writes the prose, and the band, the basis, ``verify``, the
    demotion ids, the driver and the measurement states do not move.

It is **not** a claim that the model does not matter. §16.2 already retracted the
"gate out-reasons the model" framing on our own data, and ``evals/seeded_errors``
found a model catching three error classes the gate misses. What is measured here is
one property: whether a deterministic field has a seam through which generated text
can reach it.

Two kinds of field, and the split is the whole result.

``pure``
    Computed by a function whose arguments are a cluster id and the study files.
    ``agent/verdict.py::verdict_for_cluster`` takes no prose, no model and no
    message history, so a swap cannot move these -- and the harness says so rather
    than quoting the invariance as if it had been hard won. They are still computed
    under every generator, because "no seam" is a claim about today's code and the
    cheapest way to keep it true is to re-measure it.

``seam``
    A field on the object the biologist actually sees that IS a function of
    generated text. There is exactly one in the live code:
    ``loop._assemble`` sets ``AgentResponse.verify`` to
    ``_text_asks_verify(text) or _cluster_verify_default(cluster)``, and
    ``_text_asks_verify`` is a substring test over the model's prose. CLAUDE.md:
    *"``verify`` comes from the deterministic rubric -- never offer the model the
    choice."* This is the choice, and it is measured rather than asserted.

The seam was found by reading, not by the harness; the harness is what makes the
finding a number and keeps it from quietly closing or quietly widening.

Arms
----
``swap``    free, offline, byte-stable. Five hand-written prose styles standing in
            for five generators, over all nine clusters. The prose differs in style
            only -- the facts in it come from the same deterministic record -- so any
            field that moves, moved on style alone.
``model``   opt-in, costs money. Real generators write the prose from the same
            record, and the same extraction runs over what they wrote. The pure
            fields are trivially invariant here too; the number that is worth an API
            call is how often a real generator's prose trips the seam.

Usage
-----
    python evals/generator_swap/harness.py            # swap arm, free
    python evals/generator_swap/harness.py --model    # + model arm, COSTS MONEY
    python evals/generator_swap/harness.py --model --models=claude-haiku-4-5

Models are referred to by unversioned alias, as the other harnesses do. Pin to dated
snapshots before any published run; the README carries the caveat.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from panoscope.agent import data as D  # noqa: E402
from panoscope.agent import loop  # noqa: E402
from panoscope.agent.verdict import verdict_for_cluster  # noqa: E402
from panoscope.core.measure import Reading, measure  # noqa: E402
from panoscope.types.measured import Frame, MeasuredWorld  # noqa: E402
from panoscope.types.subject import EntityKind, gene, subject  # noqa: E402

TOP_N = 5
MODELS = ("claude-haiku-4-5", "claude-sonnet-5")
OFF_PANEL_CANDIDATES = ("ALB", "UMOD", "NPHS2", "INS", "MYH7")

#: Pre-registration. True = the architecture claims a generator cannot move this.
#: ``verify_response`` is registered False because the seam was found by reading the
#: code before the run; registering it True and "discovering" it would be theatre.
EXPECTED_INVARIANT: dict[str, bool] = {
    "cell_type": True,
    "band": True,
    "band_basis": True,
    "demotions": True,
    "driver": True,
    "verify_record": True,
    "measured_states": True,
    # --- the one live seam: AgentResponse.verify is a function of model prose ---
    "verify_response": False,
}

#: Five prose styles standing in for five generators. Every fact in them is read
#: from the record at build time; nothing is typed by hand, because a hardcoded
#: coefficient goes stale the moment the study tree is rebuilt and a stale control
#: reads exactly like a real field move.
#:
#: ``hedged`` and ``flagging`` are the interesting two. ``flagging`` writes the
#: literal token the seam looks for. ``hedged`` writes an ordinary English sentence
#: that happens to contain "re-check this" -- a generator being polite, not a
#: generator making a rubric decision.
STYLES: tuple[str, ...] = ("terse", "verbose", "hedged", "flagging", "clinical")


def _prose(style: str, cluster: str, cell_type: str, band: str,
           driver: str, coef: float, pearson: float) -> str:
    """One generator's prose for one cluster. Same facts, five voices."""
    if style == "terse":
        return (f"{cluster}: {cell_type}. {driver} glm_coef {coef:.3f} "
                f"(pearson {pearson:.3f}). Confidence {band}.")
    if style == "verbose":
        return (f"Cluster {cluster} reads as {cell_type}. The call rests on "
                f"{driver}, whose glm_coef is {coef:.3f} with a pearson of "
                f"{pearson:.3f} in this unit, and the confidence band that follows "
                f"is {band}.")
    if style == "hedged":
        return (f"{cluster} looks like {cell_type} on {driver} "
                f"(glm_coef {coef:.3f}, pearson {pearson:.3f}), band {band}, "
                f"though I would re-check this before acting on it.")
    if style == "flagging":
        return (f"{cluster}: {cell_type}, band {band}, driver {driver} "
                f"glm_coef {coef:.3f} pearson {pearson:.3f}. verify=TRUE.")
    if style == "clinical":
        return (f"{cell_type} ({cluster}). Driver {driver}: glm_coef {coef:.3f}, "
                f"pearson {pearson:.3f}. Band: {band}.")
    raise ValueError(style)


# --------------------------------------------------------------------------- #
# The deterministic record, and the fields a swap is tested against
# --------------------------------------------------------------------------- #
def _world() -> MeasuredWorld:
    panel = frozenset(D.load_panel()["gene"].astype(str))
    return MeasuredWorld((
        Frame(EntityKind.GENE, "panel:xenium_hbreast", panel),
        Frame(EntityKind.UNIT, "annotation:xenium_hbreast",
              frozenset(D.load_cluster_key())),
    ))


def _measured_states(world: MeasuredWorld, cluster: str,
                     off_panel: tuple[str, ...]) -> dict[str, str]:
    """The four-state lattice for this unit's top markers plus off-panel controls.

    Included because §16.3's claim names the measurement states explicitly, and
    because they are the field a generator would most plausibly be allowed to
    influence in a looser design -- an answer that says "not detected" about an
    off-panel gene is one edit away from a state flip if anything downstream read
    the prose. Nothing does; this records that.
    """
    df = D.get_cluster_markers(cluster).head(TOP_N)
    reported = frozenset(str(r.gene) for r in df.itertuples())
    rframe = (Frame(EntityKind.GENE, f"reported:jazzpanda:{cluster}", reported),)
    out: dict[str, str] = {}
    for sym in sorted(reported) + list(off_panel):
        row = D.get_marker(sym)
        stat = float(row["glm_coef"]) if row is not None else 0.0
        item = measure(Reading(subject=subject("gene", g=gene(sym)),
                               statistic=stat, statistic_name="glm_coef"),
                       world, rframe).value
        out[sym] = item.measured.state.value
    return out


def fields(world: MeasuredWorld, cluster: str, off_panel: tuple[str, ...],
           prose: str) -> dict[str, Any]:
    """Every field under test, for one cluster under one generator's prose.

    The pure fields ignore ``prose`` entirely -- that is the point, and passing it
    in anyway is what makes the invariance a measurement rather than a diagram.
    """
    v = verdict_for_cluster(cluster)
    support = [e for e in v.evidence if e.role == "supports"]
    return {
        "cell_type": v.cell_type,
        "band": v.confidence,
        "band_basis": v.band_basis,
        "demotions": list(v.demotions),
        "driver": support[0].gene if support else None,
        "verify_record": v.verify,
        "measured_states": _measured_states(world, cluster, off_panel),
        # The live seam, computed exactly as agent/loop.py::_assemble computes it.
        "verify_response": bool(loop._text_asks_verify(prose)
                                or loop._cluster_verify_default(cluster)),
    }


def rows_for(world: MeasuredWorld, cluster: str, off_panel: tuple[str, ...],
             arm: str, model: str, generator: str, prose: str,
             baseline: dict[str, Any]) -> list[dict]:
    """One row per field: did this generator move it off the baseline?"""
    got = fields(world, cluster, off_panel, prose)
    return [{
        "id": f"{cluster}:{generator}:{name}",
        "cluster": cluster,
        "generator": generator,
        "field": name,
        "kind": "pure" if EXPECTED_INVARIANT[name] else "seam",
        "arm": arm,
        "model": model,
        "ok": True,
        "error": "",
        "expected_invariant": EXPECTED_INVARIANT[name],
        "baseline": baseline[name],
        "value": got[name],
        "moved": got[name] != baseline[name],
        "prose": prose,
    } for name in EXPECTED_INVARIANT]


def baseline_for(world: MeasuredWorld, cluster: str,
                 off_panel: tuple[str, ...]) -> dict[str, Any]:
    """The record with NO generator at all: empty prose, the rubric's own answer.

    Comparing every generator against this and not against each other is deliberate.
    Generators agreeing with each other while all of them disagree with the rubric
    is the failure that a pairwise comparison cannot see.
    """
    return fields(world, cluster, off_panel, "")


# --------------------------------------------------------------------------- #
# The model arm -- opt in, costs money
# --------------------------------------------------------------------------- #
_WRITE = """Write the one-paragraph interpretation a biologist reads first for this
cluster of a 10x Xenium spatial transcriptomics study of human breast cancer,
analysed with jazzPanda's spatial GLM.

Everything you may state is here. Do not add a statistic, a gene or a citation that
is not in this record.

  cluster        {cluster}
  call           {cell_type}
  confidence     {band}
  driver         {driver}  glm_coef={coef:.3f}  pearson={pearson:.3f}
  other markers
{table}

Write 2-4 sentences of prose. No JSON, no bullet list, no preamble."""


def run_model_arm(world: MeasuredWorld, off_panel: tuple[str, ...],
                  models: tuple[str, ...], clusters: tuple[str, ...]) -> list[dict]:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out: list[dict] = []
    for model in models:
        for cluster in clusters:
            v = verdict_for_cluster(cluster)
            support = [e for e in v.evidence if e.role == "supports"]
            if not support:
                continue
            drv = support[0]
            table = "\n".join(f"    {e.gene:10s} glm_coef={e.glm_coef:.3f} "
                              f"pearson={e.pearson:.3f}" for e in support[1:TOP_N])
            prompt = _WRITE.format(cluster=cluster, cell_type=v.cell_type,
                                   band=v.confidence, driver=drv.gene,
                                   coef=drv.glm_coef, pearson=drv.pearson,
                                   table=table or "    (none)")
            kwargs: dict[str, Any] = {"model": model, "max_tokens": 1000,
                                      "messages": [{"role": "user",
                                                    "content": prompt}]}
            if model in ("claude-opus-5", "claude-sonnet-5"):
                kwargs["output_config"] = {"effort": "low"}
            base = baseline_for(world, cluster, off_panel)
            try:
                resp = client.messages.create(**kwargs)
                prose = "".join(b.text for b in resp.content
                                if b.type == "text").strip()
            except Exception as exc:                        # noqa: BLE001
                out.append({"id": f"{cluster}:{model}:call", "cluster": cluster,
                            "generator": model, "field": "", "kind": "",
                            "arm": "model", "model": model, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "expected_invariant": True, "baseline": None,
                            "value": None, "moved": False, "prose": ""})
                continue
            rows = rows_for(world, cluster, off_panel, "model", model, model,
                            prose, base)
            # Recorded, not estimated: a cost claim in a README should come from
            # the response the run actually got back.
            rows[0]["usage"] = {"input_tokens": resp.usage.input_tokens,
                                "output_tokens": resp.usage.output_tokens}
            out += rows
    return out


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> None:
    world = _world()
    off_panel = tuple(g for g in OFF_PANEL_CANDIDATES if not D.panel_contains(g))
    if not off_panel:
        raise SystemExit("no off-panel candidate is actually off this panel")
    clusters = tuple(sorted(D.load_cluster_key()))

    rows: list[dict] = []
    for cluster in clusters:
        base = baseline_for(world, cluster, off_panel)
        v = verdict_for_cluster(cluster)
        support = [e for e in v.evidence if e.role == "supports"]
        if not support:
            continue
        drv = support[0]
        for style in STYLES:
            prose = _prose(style, cluster, v.cell_type, v.confidence, drv.gene,
                           drv.glm_coef, drv.pearson)
            rows += rows_for(world, cluster, off_panel, "swap", "", style, prose,
                             base)

    def write(name: str, rs: list[dict]) -> None:
        out = Path(__file__).parent / name
        out.write_text(json.dumps({
            "panel_size": len(D.load_panel()),
            "clusters": list(clusters),
            "styles": list(STYLES),
            "off_panel_genes": list(off_panel),
            "expected_invariant": EXPECTED_INVARIANT,
            "rows": rs,
        }, indent=2) + "\n")
        print(f"wrote {out} ({len(rs)} rows)")

    # Two files on purpose: the swap arm is byte-stable and committed, and a model
    # run must not be able to disturb it.
    write("results.json", rows)

    if "--model" in argv:
        models = tuple(next((a.split("=")[1].split(",") for a in argv
                             if a.startswith("--models=")), MODELS))
        print(f"model arm: {len(models) * len(clusters)} API calls across {models}",
              flush=True)
        write("results_model.json",
              run_model_arm(world, off_panel, models, clusters))


if __name__ == "__main__":
    main(sys.argv[1:])
