"""Panel-absence ablation: does telling a model the panel stop it misreading absence?

The question this answers, from ARCHITECTURE.md §16.2:

    A canonical marker that was never on the panel is not absent -- it was never
    measured. Does a frontier model still treat its absence as evidence against the
    correct cell type WHEN TOLD what the panel contains?

If stating the panel fixes the error, the contribution is a prompting insight and a
deterministic gate is over-engineering. If the error survives being told, that is a
result about model behaviour and it is what justifies the architecture.

Four arms, increasing in how much help the model is given:

    bare       the marker table only -- no panel, no rule
    panel      the marker table AND the full measured gene list
    panel+rule the same, plus an explicit statement that an off-panel gene was never
               measured and its absence is not evidence
    panoscope  the deterministic layer (no model in the loop for this decision)

SCORING is structural, not textual. The model returns JSON with an explicit
``evidence_against`` list. An error is an off-panel canonical marker for the TRUE
cell type appearing in that list: the model naming a gene that was never measured as
a reason to doubt. That is machine-checkable from the panel file, with no NLP and no
judgement call.

A second, softer signal is also recorded: whether the model *mentions* an off-panel
gene at all, and whether it correctly labels it as unmeasured. Mentioning it while
saying "not measured" is the CORRECT behaviour and must not be scored as an error --
conflating the two would make the benchmark reward silence rather than accuracy.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from panoscope.agent import annotation as A, data as D  # noqa: E402

MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
ARMS = ("bare", "panel", "panel_rule", "panoscope")
TOP_MARKERS = 12


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Item:
    cluster: str
    true_cell_type: str
    markers: tuple[dict, ...]           # this cluster's own top markers, with stats
    offpanel_canonical: tuple[str, ...] # canonical for the true type, NEVER measured
    onpanel_canonical: tuple[str, ...]


def build_items() -> list[Item]:
    key = json.loads((REPO / "data/studies/xenium_hbreast_sample1/inputs/cluster_key.json").read_text())
    items: list[Item] = []
    for cluster, meta in key.items():
        ct = meta["cell_type"]
        canon = set(A.CANONICAL_MARKERS_FALLBACK.get(ct, ())) | set(
            A.OFF_PANEL_CANONICAL_FALLBACK.get(ct, ())
        )
        off = tuple(sorted(g for g in canon if not D.panel_contains(g)))
        if not off:
            continue  # nothing to misread; the item cannot exhibit the failure
        on = tuple(sorted(g for g in canon if D.panel_contains(g)))
        df = D.get_cluster_markers(cluster).head(TOP_MARKERS)
        markers = tuple(
            {
                "gene": r.gene,
                "glm_coef": round(float(r.glm_coef), 3),
                "pearson": round(float(r.pearson), 3),
            }
            for r in df.itertuples()
        )
        items.append(Item(cluster, ct, markers, off, on))
    return items


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_SCHEMA_NOTE = """Return ONLY valid JSON, no prose around it:
{
  "cell_type": "<your call>",
  "confidence": "<Very High|High|Medium-High|Medium|Low>",
  "evidence_for":     ["GENE - one clause why it supports the call", ...],
  "evidence_against": ["GENE - one clause why it argues against the call", ...],
  "unmeasured":       ["GENE", ...]
}
Put a gene in "evidence_against" ONLY if you are using it as a reason to doubt the
call. Put a gene in "unmeasured" if you believe it was not measured at all."""

_TASK = """You are annotating a cluster from a 10x Xenium spatial transcriptomics
experiment of human breast cancer. Below are the cluster's top markers from a spatial
GLM: glm_coef is the spatial coefficient (larger = the gene's transcripts localise
more strongly where this cluster's cells are), pearson is spatial specificity.

Cluster {cluster} top markers:
{markers}

Call the cell type for this cluster and justify it."""

_PANEL_BLOCK = """
This is a TARGETED panel assay. The complete list of {n} genes measured in this
experiment is:
{panel}
No gene outside that list was measured."""

_RULE_BLOCK = """
IMPORTANT: a gene that is not on the panel above was never measured. Its absence from
the results is not evidence for or against any cell type -- you cannot conclude
anything from it. Only reason from genes that were actually measured."""


def build_prompt(item: Item, arm: str, panel: list[str]) -> str:
    table = "\n".join(
        f"  {m['gene']:10s} glm_coef={m['glm_coef']:<8} pearson={m['pearson']}"
        for m in item.markers
    )
    p = _TASK.format(cluster=item.cluster, markers=table)
    if arm in ("panel", "panel_rule"):
        p += _PANEL_BLOCK.format(n=len(panel), panel=", ".join(panel))
    if arm == "panel_rule":
        p += _RULE_BLOCK
    return p + "\n\n" + _SCHEMA_NOTE


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    cluster: str
    arm: str
    model: str
    replicate: int
    ok: bool
    raw: str = ""
    parsed: Optional[dict] = None
    error: str = ""
    # scoring
    offpanel_in_against: tuple[str, ...] = ()
    offpanel_mentioned: tuple[str, ...] = ()
    offpanel_called_unmeasured: tuple[str, ...] = ()
    called: str = ""
    confidence: str = ""


def _extract_json(text: str) -> Optional[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i : j + 1])
    except json.JSONDecodeError:
        return None


def score(item: Item, parsed: dict) -> dict:
    """Structural scoring. See the module docstring for why mention != error."""
    def names(field_val) -> set[str]:
        out = set()
        for e in field_val or []:
            token = str(e).replace("-", " ").split()
            if token:
                out.add(token[0].strip(" :,").upper())
        return out

    against = names(parsed.get("evidence_against"))
    unmeasured = names(parsed.get("unmeasured"))
    blob = json.dumps(parsed).upper()
    off = set(item.offpanel_canonical)
    return {
        "offpanel_in_against": tuple(sorted(off & against)),
        "offpanel_called_unmeasured": tuple(sorted(off & unmeasured)),
        "offpanel_mentioned": tuple(sorted(g for g in off if g in blob)),
        "called": str(parsed.get("cell_type", "")),
        "confidence": str(parsed.get("confidence", "")),
    }


def run_one(client, item: Item, arm: str, model: str, rep: int, panel: list[str]) -> Result:
    if arm == "panoscope":
        from panoscope.pipeline import v1_verdicts

        v = v1_verdicts.verdict_for(item.cluster)
        # The deterministic layer names off-panel genes ONLY as never-measured notes;
        # by construction they cannot enter an against-list.
        return Result(
            item.cluster, arm, "deterministic", rep, True,
            parsed={"cell_type": v.cell_type, "confidence": v.confidence},
            offpanel_called_unmeasured=tuple(sorted(n.gene for n in v.offpanel_notes)),
            called=v.cell_type, confidence=v.confidence,
        )
    prompt = build_prompt(item, arm, panel)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = _extract_json(text)
            if parsed is None:
                return Result(item.cluster, arm, model, rep, False, raw=text[:500],
                              error="unparseable JSON")
            return Result(item.cluster, arm, model, rep, True, raw=text[:500],
                          parsed=parsed, **score(item, parsed))
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return Result(item.cluster, arm, model, rep, False,
                              error=f"{type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt + random.random())
    return Result(item.cluster, arm, model, rep, False, error="unreachable")


def main(reps: int = 3, models: tuple[str, ...] = MODELS, smoke: bool = False) -> None:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    items = build_items()
    panel = sorted(D.load_panel()["gene"].tolist())
    if smoke:
        items, models, reps = items[:1], models[:1], 1

    jobs = [
        (it, arm, m, r)
        for it in items
        for arm in ARMS
        for m in (["deterministic"] if arm == "panoscope" else list(models))
        for r in range(reps if arm != "panoscope" else 1)
    ]
    print(f"{len(items)} items x {len(ARMS)} arms -> {len(jobs)} calls", flush=True)

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(run_one, client, it, arm, m, r, panel) for it, arm, m, r in jobs]
        for n, f in enumerate(futs, 1):
            results.append(f.result())
            if n % 10 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    out = Path(__file__).parent / ("smoke.json" if smoke else "results.json")
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv,
         reps=int(next((a.split("=")[1] for a in sys.argv if a.startswith("--reps=")), 3)))
