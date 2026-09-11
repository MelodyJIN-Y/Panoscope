"""Second-opinion skeptic (Tier A, deterministic, network-free).

A specialist whose ONLY job is to try to REFUTE a cluster's cell-type call, using
nothing but grounded facts. It is the adversarial complement to the verdict
engine: the engine argues the call, the skeptic argues against it, and the
disagreement is made visible. It never fabricates — every challenge quotes a real
jazzPanda number (from this cluster's own markers) or a panel fact.

It composes three grounded signals that already exist:

1. **Evidence thinness** — from :func:`agent.verdict.assess`: a call flagged
   ``verify``/``small_n`` or resting on very few supporting markers is fragile.
2. **Localization tension** — the top driver's ``max_gc_corr`` vs ``pearson``:
   if the gene localizes at least as well with some *other* cluster, that weakens
   its claim as a driver here.
3. **A competing hypothesis** — from :func:`agent.discriminate.discriminate`:
   canonical markers of the strongest rival type that ALSO peak in this cluster
   (genuine competing signal) weaken the call; rival markers that localize
   elsewhere argue *for* the call; off-panel rival markers cannot settle it.

The output is a :class:`SkepticReport`: a list of grounded challenges and a single
honest verdict — the call either *withstands* the challenge or the skeptic finds
grounds to ``re-check``. This module is deterministic and demo-safe; a live LLM
refutation, if ever wired in, must pass the same grounding gate and fall back to
this report. It is NOT wired into the app UI — it is a standalone specialist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from agent import discriminate as disc
from agent import verdict as vd
from agent.types import ClusterVerdict, MarkerEvidence

# A gene "localizes at least as well elsewhere" when its correlation to some other
# cluster meets or exceeds its correlation to its own assigned cluster, beyond a
# small margin (rounding noise). Kept tiny so only real tension trips it.
_LOCALIZATION_MARGIN: float = 1e-3

# "Few supporting markers" threshold — a call resting on <= this many on-panel
# canonical drivers is fragile enough for the skeptic to flag on its own.
_FEW_SUPPORT_MARKERS: int = 2

Effect = Literal["weakens", "supports", "neutral"]


@dataclass(frozen=True)
class SkepticChallenge:
    """One grounded point for or against the call."""

    kind: str        # "thinness" | "localization" | "competing_signal" | "checked"
    effect: Effect   # weakens | supports | neutral
    detail: str      # grounded prose; every number traces to source


@dataclass(frozen=True)
class SkepticReport:
    cluster: str
    call: str
    alternative: Optional[str]           # strongest rival type, or None
    challenges: tuple[SkepticChallenge, ...]
    survives: bool                       # True iff no challenge weakens the call
    verdict_line: str
    source_trace: tuple[str, ...]


def _label(cell_type: str) -> str:
    return cell_type.replace("_", " ")


def _supporting(verdict: ClusterVerdict) -> tuple[MarkerEvidence, ...]:
    """This cluster's supporting drivers, strongest first.

    Prefer the explicit ``supports`` role; fall back to on-panel canonical
    positives so the skeptic still works if the role vocabulary shifts.
    """
    support = tuple(e for e in verdict.evidence if e.role == "supports")
    if not support:
        support = tuple(
            e for e in verdict.evidence
            if e.is_canonical and e.is_on_panel and e.glm_coef > 0
        )
    return tuple(sorted(support, key=lambda e: e.glm_coef, reverse=True))


def second_opinion(cluster: str) -> SkepticReport:
    """Try to refute ``cluster``'s call from grounded facts. ``KeyError`` if unknown."""
    verdict = vd.assess(cluster)
    d = disc.discriminate(cluster)
    call = verdict.cell_type
    support = _supporting(verdict)

    challenges: list[SkepticChallenge] = []
    trace: list[str] = []

    # 1) Evidence thinness -------------------------------------------------- #
    n = len(support)
    if verdict.verify or verdict.small_n or n <= _FEW_SUPPORT_MARKERS:
        flag = "flagged verify" if verdict.verify else "few supporting markers"
        challenges.append(
            SkepticChallenge(
                kind="thinness",
                effect="weakens",
                detail=(
                    f"The call rests on {n} supporting marker(s) and is {flag} "
                    f"(confidence {verdict.confidence}) — fragile."
                ),
            )
        )
        trace.append(f"verdict:{cluster}:verify={verdict.verify}:small_n={verdict.small_n}:n_support={n}")

    # 2) Top-driver localization tension ------------------------------------ #
    if support:
        top = support[0]
        trace.append(f"jz:{top.gene}:pearson={top.pearson:.6f}")
        trace.append(f"jz:{top.gene}:max_gc_corr={top.max_gc_corr:.6f}")
        if top.max_gc_corr >= top.pearson + _LOCALIZATION_MARGIN:
            challenges.append(
                SkepticChallenge(
                    kind="localization",
                    effect="weakens",
                    detail=(
                        f"{top.gene} (glm_coef {top.glm_coef:.2f}) localizes at least as "
                        f"well with another cluster (max_gc_corr {top.max_gc_corr:.2f} "
                        f">= pearson {top.pearson:.2f}) — weaker as a driver here."
                    ),
                )
            )

    # 3) The strongest competing hypothesis --------------------------------- #
    if d.alt_B is not None:
        B = _label(d.alt_B)
        if d.b_here:
            genes = ", ".join(
                f"{m.gene} (glm_coef {m.glm_coef:.2f})" for m in d.b_here if m.glm_coef is not None
            )
            challenges.append(
                SkepticChallenge(
                    kind="competing_signal",
                    effect="weakens",
                    detail=f"{B} markers {genes} also peak in {cluster} — genuine competing signal.",
                )
            )
            for m in d.b_here:
                trace.append(f"jz:{m.gene}:glm_coef={m.glm_coef:.6f}")
        elif d.b_elsewhere:
            by_cluster: dict[str, list[str]] = {}
            for m in d.b_elsewhere:
                by_cluster.setdefault(m.top_cluster or "NoSig", []).append(m.gene)
            where = "; ".join(f"{', '.join(g)}->{cl}" for cl, g in by_cluster.items())
            challenges.append(
                SkepticChallenge(
                    kind="checked",
                    effect="supports",
                    detail=(
                        f"Checked {B}: its panel markers localize elsewhere ({where}), "
                        f"not {cluster} — that argues against {B}."
                    ),
                )
            )
            for m in d.b_elsewhere:
                trace.append(f"jz:{m.gene}:top_cluster={m.top_cluster}")
        elif d.offpanel_absent:
            genes = ", ".join(m.gene for m in d.offpanel_absent)
            challenges.append(
                SkepticChallenge(
                    kind="checked",
                    effect="neutral",
                    detail=(
                        f"{B}'s canonical markers ({genes}) are off-panel and were never "
                        f"measured, so the panel cannot weigh {B} here."
                    ),
                )
            )
            for m in d.offpanel_absent:
                trace.append(f"panel:{m.gene}:off_panel=True")

    survives = not any(c.effect == "weakens" for c in challenges)
    verdict_line = _verdict_line(cluster, call, verdict.confidence, support, challenges, survives)

    return SkepticReport(
        cluster=cluster,
        call=call,
        alternative=d.alt_B,
        challenges=tuple(challenges),
        survives=survives,
        verdict_line=verdict_line,
        source_trace=tuple(trace),
    )


def _verdict_line(
    cluster: str,
    call: str,
    confidence: str,
    support: tuple[MarkerEvidence, ...],
    challenges: list[SkepticChallenge],
    survives: bool,
) -> str:
    A = _label(call)
    if survives:
        if support:
            top = support[0]
            spine = (
                f"{len(support)} supporting markers, led by {top.gene} "
                f"(glm_coef {top.glm_coef:.2f}, pearson {top.pearson:.2f})"
            )
        else:
            spine = "its supporting markers"
        return (
            f"The {A} call withstands challenge — {spine}, no competing signal on the "
            f"panel. Holds at {confidence}."
        )
    reasons = "; ".join(c.detail for c in challenges if c.effect == "weakens")
    return f"The skeptic finds grounds to re-check the {A} call: {reasons} Recommend verify = TRUE."


def skeptic_summary(report: SkepticReport) -> str:
    """Grounded prose block for a CLI/demo: the challenges, then the verdict."""
    marks = {"weakens": "-", "supports": "+", "neutral": "o"}
    lines = [f"Second opinion on {report.cluster} ({_label(report.call)}):"]
    for c in report.challenges:
        lines.append(f"  [{marks.get(c.effect, '?')}] {c.detail}")
    lines.append("")
    lines.append(report.verdict_line)
    return "\n".join(lines)
