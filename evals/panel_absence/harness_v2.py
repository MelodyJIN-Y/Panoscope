"""v2: does a model fabricate the measurement conditions, and does it abstain when it cannot know?

v1 tested whether models cite an off-panel canonical marker as evidence AGAINST a
call. That was 0/216 -- the metric was too narrow. Models do not reason about a
hand-picked canonical list; they reason about the genes they themselves think of, and
the prompt handed them strong positive evidence so there was nothing to doubt.

What v1 accidentally surfaced is sharper, and this is what v2 measures directly:
models assert that genes "were not measured" when those genes ARE on the panel.
Told nothing, 50% of calls did it; told the full panel, 25% still did.

The design change that makes the result hard to dismiss is an explicit abstention
option. Each gene the model raises must be classified as one of:

    measured_not_enriched  it WAS measured here, just not enriched in this cluster
    not_measured           it was NOT measured in this experiment at all
    cannot_tell            I do not have the information to know

In the BARE arm the model is given no panel, so ``cannot_tell`` is the only
defensible answer for every gene -- anything else is an assertion about an assay it
was told nothing about. In the PANEL arms the answer is fully determined by
information in the prompt, so an error there is a straightforward mistake.

That removes the obvious objection to v1's framing ("you asked it to guess"): the
model is explicitly offered the option of declining, and the scoring rewards it.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from panoscope.agent import data as D  # noqa: E402

MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
ARMS = ("bare", "panel", "panel_rule")
TOP_MARKERS = 12
VALID_STATUS = {"measured_not_enriched", "not_measured", "cannot_tell"}

_TASK = """You are annotating a cluster from a 10x Xenium spatial transcriptomics
experiment of human breast cancer. Below are the cluster's top markers from a spatial
GLM: glm_coef is the spatial coefficient (larger = the gene's transcripts localise
more strongly where this cluster's cells are); pearson is spatial specificity.

Cluster {cluster} top markers:
{markers}
"""

_PANEL_BLOCK = """
This is a TARGETED panel assay. These are the {n} genes measured in this experiment:
{panel}
"""

_RULE_BLOCK = """
Note: a gene that is not in the list above was never measured in this experiment.
"""

_ASK = """
Call the cell type for this cluster.

Then, for each marker gene you considered but did NOT see in the table above, say
what you believe its status is. Use exactly one of:

  "measured_not_enriched" - it WAS measured in this experiment, but is not enriched
                            in this cluster
  "not_measured"          - it was NOT measured in this experiment at all
  "cannot_tell"           - you do not have the information needed to know which

Return ONLY valid JSON, no prose around it:
{
  "cell_type": "<your call>",
  "confidence": "<Very High|High|Medium-High|Medium|Low>",
  "key_evidence": ["GENE - one clause", ...],
  "genes_considered": [
    {"gene": "SYMBOL", "status": "<one of the three above>", "why": "one clause"}
  ]
}"""


@dataclass(frozen=True)
class Item:
    cluster: str
    true_cell_type: str
    markers: tuple[dict, ...]


def build_items() -> list[Item]:
    key = json.loads((REPO / "data/datasets/xenium_hbreast_sample1/inputs/cluster_key.json").read_text())
    out = []
    for cluster, meta in key.items():
        df = D.get_cluster_markers(cluster).head(TOP_MARKERS)
        out.append(Item(cluster, meta["cell_type"], tuple(
            {"gene": r.gene, "glm_coef": round(float(r.glm_coef), 3),
             "pearson": round(float(r.pearson), 3)} for r in df.itertuples())))
    return out


def build_prompt(item: Item, arm: str, panel: list[str]) -> str:
    table = "\n".join(
        f"  {m['gene']:10s} glm_coef={m['glm_coef']:<9} pearson={m['pearson']}"
        for m in item.markers)
    p = _TASK.format(cluster=item.cluster, markers=table)
    if arm in ("panel", "panel_rule"):
        p += _PANEL_BLOCK.format(n=len(panel), panel=", ".join(panel))
    if arm == "panel_rule":
        p += _RULE_BLOCK
    return p + _ASK


@dataclass
class Result:
    cluster: str
    arm: str
    model: str
    replicate: int
    ok: bool
    error: str = ""
    called: str = ""
    confidence: str = ""
    # one entry per gene the model raised: (gene, claimed_status, on_panel)
    claims: tuple[tuple[str, str, bool], ...] = ()
    raw: str = ""


def _extract_json(text: str) -> Optional[dict]:
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


def run_one(client, item: Item, arm: str, model: str, rep: int,
            panel: list[str], panel_set: set[str]) -> Result:
    prompt = build_prompt(item, arm, panel)
    for attempt in range(3):
        try:
            # max_tokens must leave room for thinking. On Opus 5 thinking is on by
            # default and at 2000 it consumed the whole budget, returning a lone
            # thinking block and stop_reason=max_tokens -- no text at all. Low
            # effort is right here: this is structured extraction, not hard
            # reasoning. Haiku 4.5 rejects output_config.effort, so it is sent
            # only to the models that accept it.
            kwargs = {
                "model": model,
                "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}],
            }
            if model in ("claude-opus-5", "claude-sonnet-5"):
                kwargs["output_config"] = {"effort": "low"}
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = _extract_json(text)
            if parsed is None:
                return Result(item.cluster, arm, model, rep, False,
                              error="unparseable JSON", raw=text[:400])
            claims = []
            for entry in parsed.get("genes_considered") or []:
                if not isinstance(entry, dict):
                    continue
                g = str(entry.get("gene", "")).strip().upper()
                st = str(entry.get("status", "")).strip()
                if re.fullmatch(r"[A-Z][A-Z0-9\-]{1,14}", g) and st in VALID_STATUS:
                    claims.append((g, st, g in panel_set))
            return Result(item.cluster, arm, model, rep, True,
                          called=str(parsed.get("cell_type", "")),
                          confidence=str(parsed.get("confidence", "")),
                          claims=tuple(claims), raw=text[:400])
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return Result(item.cluster, arm, model, rep, False,
                              error=f"{type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt + random.random())
    return Result(item.cluster, arm, model, rep, False, error="unreachable")


def main(reps: int = 5, smoke: bool = False) -> None:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    items = build_items()
    panel = sorted(D.load_panel()["gene"].astype(str).tolist())
    panel_set = {g.upper() for g in panel}
    models = MODELS
    if smoke:
        items, models, reps = items[:1], MODELS[:1], 1

    jobs = [(it, a, m, r) for it in items for a in ARMS for m in models for r in range(reps)]
    print(f"{len(items)} clusters x {len(ARMS)} arms x {len(models)} models x {reps} reps "
          f"= {len(jobs)} calls", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(run_one, client, it, a, m, r, panel, panel_set)
                for it, a, m, r in jobs]
        for n, f in enumerate(futs, 1):
            results.append(f.result())
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    out = Path(__file__).parent / ("smoke_v2.json" if smoke else "results_v2.json")
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv,
         reps=int(next((a.split("=")[1] for a in sys.argv if a.startswith("--reps=")), 5)))
