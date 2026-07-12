"""The Anthropic tool-use loop the biologist talks to.

This is the conversational spine of Panoscope. A biologist opens a cluster and
asks about it in plain language; :class:`PanoscopeAgent` answers with a
cell-type call, a confidence, and the evidence behind it — every number, marker,
and citation traced to source.

The confident-floor contract (never violated here):

* **Grounding gate on every answer.** :meth:`PanoscopeAgent.chat` and
  :meth:`PanoscopeAgent.opening_interpretation` both run every candidate answer
  through :class:`agent.grounding_check.GroundingChecker` in :meth:`_finalize`
  before it is returned. An answer that fails the floor is NEVER emitted: the
  loop repairs it once (feeding the violations back to the model), and if the
  repair still fails it discards the model answer and returns a deterministic,
  fully-grounded fallback from :class:`agent.fallback.FallbackStore`.
* **Deterministic fallback for every network call.** Every Anthropic API call
  and every MCP literature call is wrapped so a failure, timeout, or missing key
  degrades to a grounded fallback — an exception NEVER reaches the UI.
* **Numbers only from tools.** The model may not state a jazzPanda statistic it
  did not read from :mod:`agent.tools`; the grounding gate enforces this by
  re-checking every prose number against :mod:`agent.data`.
* **Citations real or absent.** PMIDs are resolved through the live MCP
  connector; the grounding gate's literature verifier is wired to
  :meth:`agent.mcp_client.PubMedMCP.verify_pmid`, so a fabricated PMID fails
  closed and the answer is rejected.

The opening interpretation is posted BEFORE any question: it is built from the
deterministic verdict engine (:func:`agent.fallback.fallback_opening`) and, when
the connector is warm, enriched with ONE real literature citation for the driving
marker. Even with the network down, the opening still posts (grounded, no PMID).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from agent import config as cfg
from agent import fallback as fb
from agent import profile as agent_profile
from agent import tools as agent_tools
from agent.grounding_check import GroundingChecker
from agent.types import (
    AgentResponse,
    Citation,
    GroundingSidecar,
    NoteDraft,
    Source,
    Tension,
)

# python-dotenv: load ANTHROPIC_API_KEY (+ NCBI creds for the MCP client) from
# .env so the loop authenticates without the caller wiring env vars by hand.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv
except Exception:  # pragma: no cover

    def load_dotenv(*_a: Any, **_k: Any) -> bool:  # type: ignore[misc]
        return False


# The Anthropic SDK. Guarded so importing this module never hard-crashes on a
# machine without the SDK — the loop simply always uses the fallback path then.
try:
    import anthropic

    _ANTHROPIC_IMPORTED = True
except Exception:  # pragma: no cover - environment without anthropic SDK
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_IMPORTED = False


# --------------------------------------------------------------------------- #
# Loop constants (mirrors BLUEPRINT §5 guards)
# --------------------------------------------------------------------------- #
MAX_TOOL_ROUNDS: int = getattr(cfg, "MAX_TOOL_ROUNDS", 6)
AGENT_TIMEOUT_S: float = float(getattr(cfg, "AGENT_TIMEOUT_S", 25))
_MAX_TOKENS: int = 2048
_TEMPERATURE: float = 0.0
# One extra model round is allowed to REPAIR an answer that failed the floor.
_MAX_REPAIRS: int = 1
# Cap the live-literature enrichment on an opening so it never blocks the demo.
_OPENING_LIT_MAX: int = 3

_SKILLS_DIR: Path = cfg.PROJECT_ROOT / "skills"
_DEFAULT_SKILL: str = "jazzpanda-markers"


# --------------------------------------------------------------------------- #
# System prompt assembly (skill + confident-floor contract + cluster context)
# --------------------------------------------------------------------------- #
_GROUNDING_CONTRACT = """
# CONFIDENT-FLOOR CONTRACT (you MUST obey this)

You are Panoscope's interpretation agent for a wet-lab biologist. jazzPanda is
the engine; you are the interpretation layer. You never run jazzPanda live.

1. NEVER fabricate. Every marker, number, and confidence value you state must
   come from a tool result in THIS conversation:
   - jazzPanda numbers (glm_coef, pearson, max_gg_corr, max_gc_corr) come ONLY
     from `marker_lookup`. Never state a statistic you did not read from a tool.
   - Panel membership comes ONLY from `panel_lookup`.
2. PANEL-ABSENCE RULE. Before down-weighting a missing canonical marker, call
   `panel_lookup`. If a gene is off-panel it was NEVER measured — say "not
   measured", never "not expressed", and never treat its absence as evidence
   against a cell type.
3. CITE EVERYTHING, REAL ONLY. Every interpretive (literature) claim must carry a
   real PMID fetched live via `literature_search` / `literature_fetch`. NEVER
   write a PMID from memory. If a lookup returns nothing, say the literature is
   thin — do not invent a reference. A fabricated citation is the worst possible
   failure.
4. MEMORY IS SCOPED, TYPED, AND CITED. In-scope lab notes are given to you above;
   cite any note you use as [note:<id>] and show its tension (never use one
   silently). When the biologist ASSERTS a judgment that diverges from, sharpens, or
   scopes the grounded default, PROPOSE a note with `memory_draft` (never persist on
   your own): classify it into ONE note_type and infer its anchor — a cell-type
   override (set subject_cell_type to the biologist's NEW call, and infer
   subject_lineage + subject_category for that new call, e.g. CAF → Stromal); a
   marker_reinterpretation (what one marker means here, call unchanged →
   subject_markers=[gene]); a program_reinterpretation (an enriched program re-read
   as co-infiltration → subject_gene_sets=[HALLMARK set]); a marker_convention (a
   panel/tissue trust rule about a marker, scope dataset|lab → subject_markers=[gene]);
   a validation (own IHC/flow → basis=own_validation); a confidence_adjustment (their
   confidence stance — NEVER change a number, it is an overlay); an exclude (a
   doublet/artifact cluster); or a cross_cluster note (two+ clusters are one →
   scope=dataset, subject_clusters=[ids]). It cross-checks the claim against the
   literature and the biologist confirms scope/basis before it is saved. Do NOT draft
   for questions, acknowledgements, view commands, or mid-thought hedges. You MUST
   actually CALL the `memory_draft` tool — that is what shows the confirm card; never
   just say "drafting the note" without calling it. Then, in ONE short sentence, tell
   them to confirm scope/basis below; keep the disagreement visible, never a bare
   acknowledgement.
5. WHEN UNSURE, say "re-check this" and set the verify flag — do not guess to
   seem helpful.
6. PLAIN CHAT PROSE. This is a conversation, not a document: reply in a few short
   sentences. NEVER use markdown headings (#), tables, or bullet lists. Every point
   names a gene and its number; state the confidence and the caveat. No hype, short
   over long — two or three sentences beats a formatted write-up.

Inline conventions: cite literature as `PMID:xxxxxxx` and notes as `[note:id]`.
Only state a jazzPanda number in the exact form the tool returned it (you may
round to two decimals). Do not state a statistic for an off-panel gene — its
absence is not a number.
""".strip()


@lru_cache(maxsize=4)
def _skill_text(skill: str = _DEFAULT_SKILL) -> str:
    """Load a skill's SKILL.md (cached per skill). Empty string if missing.

    ``skill`` names a directory under ``skills/`` (e.g. "jazzpanda-markers" for the
    marker workflow, "geneset-enrichment" for the pathway workflow), so the agent
    carries the right interpretation contract for each workflow.
    """
    try:
        return (_SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    except Exception:  # pragma: no cover - skill file should exist
        return ""


@lru_cache(maxsize=1)
def _global_interpretation_lines() -> str:
    """One grounded line per cluster: cell type + confidence + verify + drivers.

    This is the whole annotation's interpretation result, read from the
    deterministic verdicts (never fabricated). It is injected on EVERY turn so the
    agent stays aware of the full picture even though each conversation thread is
    scoped to a single cluster — the biologist can ask relational questions ("how
    does this sit next to c4?") and the agent keeps the annotation coherent.
    Cached (verdicts are static within a run). Falls back to the key on any error.
    """
    from agent import verdict as _verdict

    lines: list[str] = []
    for c in cfg.CLUSTER_ORDER:
        meta = cfg.CLUSTER_KEY.get(c, {})
        try:
            v = _verdict.verdict_for_cluster(c)
            flag = "  [VERIFY — re-check]" if v.verify else ""
            drivers = ", ".join(v.key_markers[:3])
            lines.append(
                f"  {c}: {v.cell_type} — {v.confidence} confidence{flag}"
                f" (drivers: {drivers})"
            )
        except Exception:  # pragma: no cover - degrade to the bare key, never raise
            lines.append(f"  {c}: {meta.get('cell_type', '?')}")
    return "\n".join(lines)


def _inscope_notes_block(cluster: Optional[str]) -> str:
    """The lab notes IN SCOPE for ``cluster``, injected on EVERY turn.

    Recall must not depend on the model remembering to call ``memory_read``: an
    override the lab saved is surfaced here every turn so it is never silently
    forgotten. The agent MUST still honor and cite each note as ``[note:id]`` and
    show its tension (a note may not be used silently). Read FRESH (notes mutate
    on save) from the same base dir the memory tools write to. Never raises; empty
    string when there are none.
    """
    if not cluster or cluster not in cfg.KNOWN_CLUSTERS:
        return ""
    try:
        from agent import memory

        notes = memory.apply_notes(cluster, base_dir=agent_tools.memory_base_dir())
    except Exception:  # pragma: no cover - notes are best-effort context
        return ""
    if not notes:
        return ""

    lines: list[str] = []
    for n in notes:
        t = n.tension
        if t.agree or t.dissent:
            tension = f"agree {len(t.agree)} / dissent {len(t.dissent)}"
        else:
            tension = "literature thin" if t.thin else "no tension recorded"
        scope_txt = (
            f"cluster {n.scope_ref.cluster}" if n.scope == "cluster" else n.scope
        )
        lines.append(
            f"  [note:{n.id}] ({scope_txt}, basis={n.basis}, {n.status}) "
            f"{n.claim}  <{tension}>"
        )
    return (
        "\n\n# IN-SCOPE LAB NOTES (the lab's own calls — you MUST honor these and "
        "cite each as [note:id] when you use it, showing its tension; a note may "
        "not be used silently)\n" + "\n".join(lines)
    )


def _cluster_context(cluster: Optional[str]) -> str:
    """Grounded context block: GLOBAL interpretation + active cluster + lab notes.

    The global block (every cluster's call, confidence, verify, drivers) and the
    in-scope lab notes are always present, so the agent's knowledge of the whole
    annotation AND of the lab's own overrides is independent of the per-cluster
    conversation history it is given.
    """
    header = (
        "# GLOBAL INTERPRETATION (authoritative — every cluster's call; do not "
        "restate a number you have not read from a tool, but you MAY use these "
        "calls and confidence bands to reason relationally and stay consistent)\n"
        + _global_interpretation_lines()
    )
    if cluster and cluster in cfg.KNOWN_CLUSTERS:
        meta = cfg.CLUSTER_KEY[cluster]
        active = (
            f"\n\n# ACTIVE CLUSTER: {cluster} — {meta['cell_type']} "
            f"({meta['category']} / {meta['lineage']}). "
            f"Answer about {cluster} unless the biologist names another."
        )
    else:
        active = ""
    return header + active + _inscope_notes_block(cluster)


def _enrichment_context(cluster: Optional[str]) -> str:
    """Grounded per-cluster ENRICHMENT context injected for the geneset-enrichment
    skill: the cluster's enriched (and suggestive) programs with their scores,
    panel coverage, and leading-edge genes — every number the source of truth. So
    a pathway chat is grounded in the real enrichment records, not fetched via a
    marker tool. Empty string if the cluster has no enrichment (chat still works).
    """
    if not cluster or cluster not in cfg.KNOWN_CLUSTERS:
        return ""
    try:
        from agent import enrichment as agent_enrichment

        ce = agent_enrichment.enrichment_for_cluster(cluster)
    except Exception:  # noqa: BLE001 - no enrichment -> no context, never crash
        return ""

    lines = [
        f"# ENRICHMENT CONTEXT — cluster {cluster} ({ce.cell_type.replace('_', ' ')}), "
        "jazzPanda competitive gene-set test (MSigDB Hallmark, panel-scoped)",
        f"Enrichment confidence: {ce.confidence}{' — verify (re-check)' if ce.verify else ''}.",
    ]
    if ce.enriched:
        lines.append("Enriched programs (panel-scoped — K of N set genes on the 280-gene panel):")
        for p in ce.enriched:
            q = "n/a" if p.q_value is None else f"{p.q_value:.1e}"
            lines.append(
                f"- {p.gene_set}: test_statistic {p.score:.2f}, q {q}, "
                f"panel {p.panel_hits}/{p.set_size_full}, leading edge {', '.join(p.leading_edge)}."
            )
    if ce.suggestive:
        lines.append("Suggestive programs (below the strict bar — re-check):")
        for p in ce.suggestive:
            q = "n/a" if p.q_value is None else f"{p.q_value:.1e}"
            lines.append(f"- {p.gene_set}: q {q}, leading edge {', '.join(p.leading_edge)}.")
    if not ce.enriched and not ce.suggestive:
        lines.append("No program clears the enrichment gate for this cluster.")
    lines.append(
        "Discuss ONLY these programs and their leading-edge genes; every number above is "
        "the source of truth (never invent a program, score, or gene). Always state the "
        "panel-scoped caveat (K of N genes measured, not genome-wide). Flag a cross-lineage "
        "program as a tension to check, never a re-typing of the cluster. Cite real, live "
        "PubMed papers; never a PMID from memory."
    )
    return "\n".join(lines)


def _research_context() -> str:
    """Optional personalization block: the biologist's research interest, used ONLY
    to sharpen literature search. Empty string when no profile is set.

    It biases WHICH real paper the agent searches for and cites so citations are
    tissue/field-appropriate; it can never change a jazzPanda number, a marker, a
    confidence band, or a cell-type call, and never excuses a PMID from memory.
    """
    interest = agent_profile.load()
    if not interest:
        return ""
    return (
        "# BIOLOGIST RESEARCH CONTEXT (personalization for literature search ONLY)\n"
        f"The biologist studies: {interest}\n"
        "When you SEARCH or CITE the literature, prefer real papers relevant to this "
        "context so the citation is more precise and tissue-appropriate. This may "
        "influence WHICH real paper you cite; it must NEVER invent, change, or override "
        "any jazzPanda number, marker, confidence band, or cell-type call, and never "
        "justifies writing a PMID from memory."
    )


def build_system_prompt(cluster: Optional[str], skill: str = _DEFAULT_SKILL) -> str:
    """Assemble the full system prompt: skill + contract + per-cluster context.

    ``skill`` selects both the SKILL.md contract and the context: the enrichment
    skill carries the cluster's enrichment records (``_enrichment_context``); the
    marker skill carries the marker global interpretation (``_cluster_context``).
    """
    context = _enrichment_context(cluster) if skill == "geneset-enrichment" else _cluster_context(cluster)
    parts = [
        _skill_text(skill),
        _GROUNDING_CONTRACT,
        _research_context(),
        context,
    ]
    return "\n\n---\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Literature verifier for the grounding gate (wraps the MCP client, never raises)
# --------------------------------------------------------------------------- #
def _default_literature_verifier(ident: str) -> bool:
    """Resolve a PMID through the live MCP connector. False on any failure.

    Wired into :class:`GroundingChecker` so a fabricated PMID fails closed. Only
    numeric PMIDs are checkable here; a DOI (rare in this app) cannot be resolved
    by the PubMed fetch tool, so it fails closed too — the agent is instructed to
    cite PMIDs, not DOIs.
    """
    pid = str(ident or "").strip()
    if not pid.isdigit():
        return False
    try:
        from agent import mcp_client

        client = mcp_client.get_mcp_client()
        if not client.available:
            return False
        return bool(client.verify_pmid(pid))
    except Exception:  # pragma: no cover - never raise into the gate
        return False


# --------------------------------------------------------------------------- #
# Rebuild a NoteDraft from a memory_draft tool payload
# --------------------------------------------------------------------------- #
def _cite_from_dict(d: dict[str, Any]) -> Citation:
    """Rebuild a real Citation from a draft-payload tension entry."""
    year = d.get("year")
    return Citation(
        pmid=str(d.get("pmid", "")),
        title=str(d.get("title", "")),
        authors=str(d.get("authors", "")),
        year=int(year) if str(year).isdigit() else 0,
        journal=str(d.get("journal", "")),
        stance=str(d.get("stance", "agree")),
        is_real=True,
    )


def _draft_from_payload(data: Any) -> Optional[NoteDraft]:
    """Reconstruct a :class:`NoteDraft` from a ``memory_draft`` envelope's data.

    Returns None if the payload is malformed (the turn simply carries no draft).
    """
    if not isinstance(data, dict) or not data.get("claim"):
        return None
    t = data.get("tension") or {}
    tension = Tension(
        agree=tuple(_cite_from_dict(c) for c in t.get("agree", ()) if isinstance(c, dict)),
        dissent=tuple(_cite_from_dict(c) for c in t.get("dissent", ()) if isinstance(c, dict)),
        thin=bool(t.get("thin", True)),
        query=str(t.get("query", "")),
        looked_up_at="",
    )
    return NoteDraft(
        claim=str(data["claim"]),
        scope=data.get("scope", "cluster"),
        basis=data.get("basis", "own_validation"),
        status=data.get("status", "firm"),
        cluster=data.get("cluster"),
        subject_cell_type=data.get("subject_cell_type"),
        subject_markers=tuple(data.get("subject_markers") or ()),
        tension=tension,
        dataset=str(data.get("dataset", cfg.DATASET_ID)),
        type=data.get("type", "celltype_override"),
        subject_gene_sets=tuple(data.get("subject_gene_sets") or ()),
        subject_clusters=tuple(data.get("subject_clusters") or ()),
        subject_lineage=str(data.get("subject_lineage", "")),
        subject_category=str(data.get("subject_category", "")),
    )


# --------------------------------------------------------------------------- #
# Sidecar / source accumulation from tool results
# --------------------------------------------------------------------------- #
class _Ledger:
    """Accumulates the grounded facts a turn actually used, from tool results.

    Every tool the model calls returns the uniform envelope; the ledger reads the
    ``data``/``sources`` off each success envelope and records exactly the
    numbers, markers, PMIDs, notes, and Source chips that flowed back. The
    resulting :class:`GroundingSidecar` is a localization aid for the checker (the
    checker still re-derives every prose number from source), and the Source chips
    are surfaced to the UI.
    """

    def __init__(self) -> None:
        self.numbers: list[tuple[str, str, float]] = []
        self.markers: list[str] = []
        self.pmids: list[str] = []
        self.notes: list[str] = []
        self.sources: list[Source] = []
        self.citations: list[Citation] = []
        self.pin_marker: Optional[str] = None
        self.note_draft: Optional[NoteDraft] = None

    # -- record one tool call's envelope ----------------------------------- #
    def record(self, tool_name: str, envelope: dict[str, Any]) -> None:
        if not isinstance(envelope, dict) or not envelope.get("ok"):
            return
        data = envelope.get("data")
        for s in envelope.get("sources", ()):  # Source-shaped dicts
            self._record_source(s)
        if tool_name == "marker_lookup":
            self._record_markers(data)
        elif tool_name == "panel_lookup":
            self._record_panel(data)
        elif tool_name in ("literature_search", "literature_fetch"):
            self._record_literature(data)
        elif tool_name == "memory_read":
            self._record_notes(data)
        elif tool_name == "memory_write":
            self._record_note_write(data)
        elif tool_name == "memory_draft":
            self._record_draft(data)

    def _record_source(self, s: Any) -> None:
        if not isinstance(s, dict):
            return
        kind = s.get("kind")
        if kind not in ("jz", "panel", "lit", "mem"):
            return
        self.sources.append(
            Source(
                kind=kind,
                ref=str(s.get("ref", "")),
                value=s.get("value"),
                detail=str(s.get("detail", "")),
            )
        )

    def _record_one_marker(self, m: dict[str, Any]) -> None:
        gene = str(m.get("gene", "")).upper()
        if not gene:
            return
        self.markers.append(gene)
        for stat in ("glm_coef", "pearson", "max_gg_corr", "max_gc_corr"):
            if stat in m and m[stat] is not None:
                try:
                    self.numbers.append((gene, stat, float(m[stat])))
                except (TypeError, ValueError):
                    continue
        if self.pin_marker is None:
            self.pin_marker = gene

    def _record_markers(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if isinstance(data.get("markers"), list):
            for m in data["markers"]:
                if isinstance(m, dict):
                    self._record_one_marker(m)
        elif isinstance(data.get("marker"), dict):
            self._record_one_marker(data["marker"])

    def _record_panel(self, data: Any) -> None:
        if isinstance(data, dict) and data.get("gene"):
            # A panel gene is a legitimate token to name (absence rule); it carries
            # no jazzPanda number, so it goes in markers-mentioned only.
            self.markers.append(str(data["gene"]).upper())

    def _record_literature(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        recs = data.get("results") or data.get("articles") or ()
        for r in recs:
            if not isinstance(r, dict):
                continue
            pmid = str(r.get("pmid", "")).strip()
            if not pmid:
                continue
            self.pmids.append(pmid)
            self.citations.append(
                Citation(
                    pmid=pmid,
                    title=str(r.get("title", "")),
                    authors=str(r.get("authors", "")),
                    year=int(r["year"]) if str(r.get("year", "")).isdigit() else 0,
                    journal=str(r.get("journal", "")),
                    abstract=str(r.get("abstract", "")),
                    url=str(r.get("url", ""))
                    or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""),
                    is_real=True,
                )
            )

    def _record_notes(self, data: Any) -> None:
        if isinstance(data, dict) and isinstance(data.get("notes"), list):
            for n in data["notes"]:
                if isinstance(n, dict) and n.get("id"):
                    self.notes.append(str(n["id"]))

    def _record_note_write(self, data: Any) -> None:
        if isinstance(data, dict) and data.get("id"):
            self.notes.append(str(data["id"]))

    def _record_draft(self, data: Any) -> None:
        """Capture a proposed (unsaved) note + its tension citations.

        The draft becomes ``AgentResponse.note_draft`` (the UI renders the confirm
        card). Its real agree/dissent PMIDs are recorded like any literature cite
        so the grounding gate accepts the override prose and the UI can link them.
        """
        draft = _draft_from_payload(data)
        if draft is None:
            return
        self.note_draft = draft
        for c in draft.tension.agree + draft.tension.dissent:
            if c.pmid:
                self.pmids.append(c.pmid)
                self.citations.append(c)

    # -- finalize ---------------------------------------------------------- #
    def sidecar(self) -> GroundingSidecar:
        return GroundingSidecar(
            numbers=tuple(dict.fromkeys(self.numbers)),
            markers=tuple(dict.fromkeys(self.markers)),
            pmids=tuple(dict.fromkeys(self.pmids)),
            notes_used=tuple(dict.fromkeys(self.notes)),
        )

    def source_chips(self) -> tuple[Source, ...]:
        # De-dup on the (kind, ref, value) identity, preserving order.
        seen: set[tuple[str, str, Optional[str]]] = set()
        out: list[Source] = []
        for s in self.sources:
            key = (s.kind, s.ref, s.value)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return tuple(out)

    def citation_tuple(self) -> tuple[Citation, ...]:
        seen: set[str] = set()
        out: list[Citation] = []
        for c in self.citations:
            if c.pmid in seen:
                continue
            seen.add(c.pmid)
            out.append(c)
        return tuple(out)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class PanoscopeAgent:
    """The tool-use loop a biologist talks to. Never lets an ungrounded answer out.

    Parameters
    ----------
    cluster:
        The active cluster (c1..c9), or None until :meth:`set_cluster`.
    model:
        Anthropic model id; defaults to :data:`agent.config.PRIMARY_MODEL`.
    api_key:
        Explicit key; defaults to ``ANTHROPIC_API_KEY`` from the environment/.env.
    literature_verifier:
        Callable ``(pmid)->bool`` the grounding gate uses to resolve citations.
        Defaults to the live MCP connector (fabricated PMIDs fail closed).
    fallback_store:
        The deterministic fallback source; defaults to a fresh
        :class:`agent.fallback.FallbackStore`.
    """

    def __init__(
        self,
        cluster: Optional[str] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        literature_verifier: Optional[Callable[[str], bool]] = None,
        fallback_store: Optional[fb.FallbackStore] = None,
    ) -> None:
        load_dotenv()
        self._cluster = cluster if cluster in cfg.KNOWN_CLUSTERS else None
        self._model = model or cfg.PRIMARY_MODEL
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._fallback = fallback_store or fb.FallbackStore()
        self._verifier = literature_verifier or _default_literature_verifier
        self._checker = GroundingChecker(literature_verifier=self._verifier)
        self._client: Optional[Any] = None

    # -- cluster context ---------------------------------------------------- #
    def set_cluster(self, cluster: str) -> None:
        """Set the active cluster (c1..c9). Unknown ids raise KeyError."""
        if cluster not in cfg.KNOWN_CLUSTERS:
            raise KeyError(
                f"[loop] unknown cluster {cluster!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}"
            )
        self._cluster = cluster

    @property
    def cluster(self) -> Optional[str]:
        return self._cluster

    # -- Anthropic client (lazy, guarded) ---------------------------------- #
    def _get_client(self) -> Optional[Any]:
        """Return a warm Anthropic client, or None if unavailable (-> fallback)."""
        if not _ANTHROPIC_IMPORTED or not self._api_key:
            return None
        if self._client is None:
            try:
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except Exception:  # pragma: no cover - client construction failure
                return None
        return self._client

    # -- opening interpretation (posted before any question) --------------- #
    def opening_interpretation(self, cluster: Optional[str] = None) -> AgentResponse:
        """Post the grounded opening interpretation for ``cluster``.

        The base is the deterministic, verdict-grounded opening
        (:func:`agent.fallback.fallback_opening`) — the call, confidence, driving
        markers with their real glm_coef/pearson, and the panel-absence notes.
        When the connector is warm, it is enriched with ONE real literature
        citation for the leading driver (PMID fetched live, never from memory).
        The opening ALWAYS passes the grounding floor; on any enrichment failure
        it degrades cleanly to the pure verdict opening.
        """
        target = cluster or self._cluster
        if target is None:
            raise KeyError("[loop] no cluster set; call set_cluster() first")
        if target not in cfg.KNOWN_CLUSTERS:
            raise KeyError(f"[loop] unknown cluster {target!r}")

        base = self._fallback.opening(target)  # deterministic, grounded

        # Fast path: the pipeline's precomputed, live-cited cell-type note. Its PMID
        # was verified live at build time and the base is grounded by construction,
        # so this is trusted — NO network, no live re-verify at open time (matching
        # the Pathways opening). The live CHAT is unaffected and stays fully live.
        pre = self._precomputed_opening(base, target)
        if pre is not None:
            return pre

        enriched = self._enrich_opening(base, target)
        candidate = enriched if enriched is not None else base

        # The opening must clear the floor. If enrichment somehow broke it (it
        # should not — real PMIDs only), fall back to the pure verdict opening.
        result = self._checker.check(candidate.text, candidate.grounding, target)
        if result.ok:
            return candidate
        return base

    def _enrich_opening(
        self, base: AgentResponse, cluster: str
    ) -> Optional[AgentResponse]:
        """Add ONE real literature citation to the opening via a LIVE lookup, or None
        on any failure.

        This is the fallback path used only when no precomputed cell-type note exists
        (``opening_interpretation`` tries :meth:`_precomputed_opening` first). Uses the
        loop's own literature search (the same tool the chat loop uses), so a PMID only
        ever comes from a real connector lookup. Everything is wrapped: a down/slow
        connector yields None and the caller keeps the pure verdict opening.
        """
        op = base  # alias
        pin = op.pin_marker
        if not pin:
            return None
        cell_type = cfg.CLUSTER_KEY[cluster]["cell_type"].replace("_", " ")
        query = f"{pin} {cell_type} marker"
        env = self._safe_tool(
            "literature_search", {"query": query, "max_results": _OPENING_LIT_MAX}
        )
        if not env.get("ok"):
            return None
        results = (env.get("data") or {}).get("results") or []
        real = [r for r in results if str(r.get("pmid", "")).strip().isdigit()]
        if not real:
            return None
        top = real[0]
        pmid = str(top["pmid"]).strip()
        # Verify the PMID resolves for real before we cite it (fail closed).
        if not self._verifier(pmid):
            return None

        cite = Citation(
            pmid=pmid,
            title=str(top.get("title", "")),
            authors=str(top.get("authors", "")),
            year=int(top["year"]) if str(top.get("year", "")).isdigit() else 0,
            journal=str(top.get("journal", "")),
            url=str(top.get("url", "")) or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            is_real=True,
        )
        lit_line = (
            f" Literature: {pin} as a {cell_type} marker — PMID:{pmid}"
            f"{' (' + cite.title + ')' if cite.title else ''}."
        )
        new_sidecar = GroundingSidecar(
            numbers=op.grounding.numbers,
            markers=op.grounding.markers,
            pmids=op.grounding.pmids + (pmid,),
            notes_used=op.grounding.notes_used,
        )
        lit_source = Source(
            kind="lit",
            ref=pmid,
            value=cite.title,
            detail=f"{cite.authors} ({cite.year}) {cite.journal}".strip(),
        )
        return AgentResponse(
            text=op.text + lit_line,
            sources=op.sources + (lit_source,),
            verify=op.verify,
            grounding=new_sidecar,
            pin_marker=op.pin_marker,
            citations=op.citations + (cite,),
            note_written=None,
            used_fallback=False,
            opening=True,
        )

    @staticmethod
    def _load_celltype_note(cluster: str) -> Optional[dict]:
        """The pipeline's precomputed cell-type note for ``cluster`` (keys: summary,
        pmid, citation), or None. Fail-soft: no tree / import issue -> None (live path)."""
        try:
            from pipeline import store  # lazy: avoids an import cycle with pipeline

            notes = store.load_celltype_notes(cfg.DATASET_ID)
        except Exception:  # noqa: BLE001 - no tree / not importable -> live lookup
            return None
        note = (notes or {}).get(cluster)
        return note if isinstance(note, dict) else None

    def _precomputed_opening(
        self, base: AgentResponse, cluster: str
    ) -> Optional[AgentResponse]:
        """Enrich the opening from the precomputed, live-cited cell-type note — no
        network. None if there is no precomputed note with a real PMID (the caller
        then does a live lookup). The PMID is real: it was fetched live in the
        pipeline's notes stage, so this keeps the confident floor (never from memory)."""
        note = self._load_celltype_note(cluster)
        if not note:
            return None
        pmid = str(note.get("pmid", "") or "").strip()
        if not pmid.isdigit():
            return None

        op = base
        cinfo = note.get("citation") or {}
        cell_type = cfg.CLUSTER_KEY[cluster]["cell_type"].replace("_", " ")
        cite = Citation(
            pmid=pmid,
            title=str(cinfo.get("title", "")),
            authors=str(cinfo.get("authors", "")),
            year=int(cinfo["year"]) if str(cinfo.get("year", "")).isdigit() else 0,
            journal=str(cinfo.get("journal", "")),
            url=str(cinfo.get("url", "")) or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            is_real=True,
        )
        lit_line = (
            f" Literature: {cell_type} biology — PMID:{pmid}"
            f"{' (' + cite.title + ')' if cite.title else ''}."
        )
        new_sidecar = GroundingSidecar(
            numbers=op.grounding.numbers,
            markers=op.grounding.markers,
            pmids=op.grounding.pmids + (pmid,),
            notes_used=op.grounding.notes_used,
        )
        lit_source = Source(
            kind="lit",
            ref=pmid,
            value=cite.title,
            detail=f"{cite.authors} ({cite.year}) {cite.journal}".strip(),
        )
        return AgentResponse(
            text=op.text + lit_line,
            sources=op.sources + (lit_source,),
            verify=op.verify,
            grounding=new_sidecar,
            pin_marker=op.pin_marker,
            citations=op.citations + (cite,),
            note_written=None,
            used_fallback=False,
            opening=True,
        )

    # -- chat --------------------------------------------------------------- #
    def chat(
        self,
        query: str,
        cluster: Optional[str] = None,
        history: Optional[list] = None,
        skill: str = _DEFAULT_SKILL,
    ) -> AgentResponse:
        """Answer ``query`` about the active cluster. Always grounded, never raises.

        Runs the Anthropic tool-use loop, accumulates the grounded facts each tool
        returned, and finalizes through the grounding gate. On any model/connector
        failure, on an empty answer, or on a floor failure that a single repair
        cannot fix, returns a deterministic grounded fallback. An exception NEVER
        reaches the caller. ``skill`` selects the interpretation contract (marker vs
        enrichment) loaded into the system prompt.
        """
        target = cluster or self._cluster
        if target is None or target not in cfg.KNOWN_CLUSTERS:
            # No valid cluster context -> generic grounded fallback (never crash).
            target = target if target in cfg.KNOWN_CLUSTERS else cfg.CLUSTER_ORDER[0]
            return self._fallback.match(query or "", target)

        client = self._get_client()
        if client is None:
            # No live model -> deterministic grounded fallback for this query.
            return self._fallback.match(query or "", target)

        try:
            return self._run_loop(client, query or "", target, history, skill)
        except Exception:  # noqa: BLE001 - the loop must never raise into the UI
            return self._fallback.match(query or "", target)

    # -- the model loop ----------------------------------------------------- #
    def _run_loop(
        self,
        client: Any,
        query: str,
        cluster: str,
        history: Optional[list],
        skill: str = _DEFAULT_SKILL,
    ) -> AgentResponse:
        system = build_system_prompt(cluster, skill)
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": query})

        ledger = _Ledger()
        final_text = ""

        for _round in range(MAX_TOOL_ROUNDS):
            reply = self._safe_create(client, system, messages)
            if reply is None:
                # Model call failed -> deterministic fallback.
                return self._fallback.match(query, cluster)

            text_now, tool_uses = _split_reply(reply)
            if text_now:
                final_text = text_now

            if not tool_uses:
                break  # model produced a final answer

            # Append the assistant's tool-use turn, then run each tool.
            messages.append({"role": "assistant", "content": _reply_content(reply)})
            tool_results = self._run_tools(tool_uses, cluster, ledger)
            messages.append({"role": "user", "content": tool_results})
        else:
            # Ran out of rounds without a final text turn — ask once for a wrap-up.
            reply = self._safe_create(client, system, messages)
            if reply is not None:
                text_now, _ = _split_reply(reply)
                if text_now:
                    final_text = text_now

        return self._finalize(final_text, ledger, cluster, query, client, system, messages)

    def _run_tools(
        self, tool_uses: list[dict[str, Any]], cluster: str, ledger: "_Ledger"
    ) -> list[dict[str, Any]]:
        """Dispatch each requested tool, record it, and shape tool_result blocks."""
        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            envelope = self._safe_dispatch(tu["name"], tu["input"], cluster)
            ledger.record(tu["name"], envelope)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": json.dumps(envelope, default=str),
                    "is_error": not envelope.get("ok", False),
                }
            )
        return tool_results

    # -- finalize + grounding gate ----------------------------------------- #
    def _finalize(
        self,
        text: str,
        ledger: _Ledger,
        cluster: str,
        query: str,
        client: Any,
        system: str,
        messages: list[dict[str, Any]],
    ) -> AgentResponse:
        """Build the AgentResponse and run the grounding floor. Repair once, else fall back.

        The sidecar the ledger built localizes the grounded facts, but the checker
        re-derives every prose number from source, so a hallucinated number that
        never came from a tool is still caught. A CRITICAL/HIGH violation triggers
        one repair round (violations fed back to the model); if the repaired answer
        still fails, the deterministic fallback is returned — never an ungrounded
        answer, never a spinner.
        """
        text = (text or "").strip()
        if not text:
            return self._fallback.match(query, cluster)

        candidate = self._assemble(text, ledger, cluster)
        result = self._checker.check(candidate.text, candidate.grounding, cluster)
        if result.ok:
            return candidate

        # One repair attempt: tell the model exactly what failed, re-run once.
        repaired = self._repair(result, ledger, cluster, query, client, system, messages)
        if repaired is not None:
            return repaired

        # Repair failed the floor too -> deterministic grounded fallback.
        return self._fallback.match(query, cluster)

    def _repair(
        self,
        result,
        ledger: _Ledger,
        cluster: str,
        query: str,
        client: Any,
        system: str,
        messages: list[dict[str, Any]],
    ) -> Optional[AgentResponse]:
        """One repair round: feed the grounding violations back, re-check once."""
        feedback = (
            "Your previous answer FAILED the confident-floor grounding check with "
            "these violations:\n"
            + result.summary()
            + "\n\nRewrite the answer so EVERY number matches a tool result exactly, "
            "every PMID is one you fetched live (no PMID from memory), and every "
            "note you cite is in scope. Only state jazzPanda numbers you read from "
            "marker_lookup. Do not state a statistic for an off-panel gene."
        )
        repair_messages = list(messages)
        repair_messages.append({"role": "user", "content": feedback})

        for _ in range(_MAX_REPAIRS):
            reply = self._safe_create(client, system, repair_messages)
            if reply is None:
                return None
            text_now, tool_uses = _split_reply(reply)
            if tool_uses:
                # Allow the model to re-pull facts during repair.
                repair_messages.append({"role": "assistant", "content": _reply_content(reply)})
                repair_messages.append(
                    {"role": "user", "content": self._run_tools(tool_uses, cluster, ledger)}
                )
                continue
            text_now = (text_now or "").strip()
            if not text_now:
                return None
            candidate = self._assemble(text_now, ledger, cluster)
            if self._checker.check(candidate.text, candidate.grounding, cluster).ok:
                return candidate
            return None
        return None

    def _assemble(self, text: str, ledger: _Ledger, cluster: str) -> AgentResponse:
        """Build the AgentResponse from the model text + the accumulated ledger."""
        verify = _text_asks_verify(text) or _cluster_verify_default(cluster)
        return AgentResponse(
            text=text,
            sources=ledger.source_chips(),
            verify=verify,
            grounding=ledger.sidecar(),
            pin_marker=ledger.pin_marker,
            citations=ledger.citation_tuple(),
            note_written=None,
            note_draft=ledger.note_draft,
            used_fallback=False,
            opening=False,
        )

    # -- guarded primitives (never raise) ---------------------------------- #
    def _safe_create(
        self, client: Any, system: str, messages: list[dict[str, Any]]
    ) -> Optional[Any]:
        """One Anthropic messages.create call, wrapped. None on any failure/timeout."""
        try:
            return client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                system=system,
                tools=agent_tools.TOOL_SCHEMAS,
                messages=messages,
                timeout=AGENT_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 - model failure -> caller falls back
            return None

    def _safe_dispatch(self, name: str, args: Any, cluster: str) -> dict[str, Any]:
        """Dispatch a tool call, wrapped. Always returns the uniform envelope.

        Injects the active cluster for a cluster-scoped ``memory_read`` when the
        model omits it, so reading in-scope notes for the current cluster works.
        """
        kwargs = dict(args) if isinstance(args, dict) else {}
        if name == "memory_read" and "cluster" not in kwargs:
            kwargs["cluster"] = cluster
        try:
            return agent_tools.dispatch(name, kwargs)
        except Exception as exc:  # noqa: BLE001 - dispatch guards too; belt+braces
            return {"ok": False, "data": None, "sources": [], "error": f"{name} failed: {exc!r}"}

    def _safe_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Direct guarded tool call used by opening enrichment (no cluster inject)."""
        try:
            return agent_tools.dispatch(name, args)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "data": None, "sources": [], "error": f"{name} failed: {exc!r}"}


# --------------------------------------------------------------------------- #
# Reply parsing helpers (tolerant of SDK block shapes)
# --------------------------------------------------------------------------- #
def _reply_content(reply: Any) -> Any:
    """Return the assistant message content to echo back into the next turn.

    The SDK's ``reply.content`` is a list of content blocks; passing it straight
    back preserves tool_use blocks so the tool_result turn references them.
    """
    return getattr(reply, "content", []) or []


def _split_reply(reply: Any) -> tuple[str, list[dict[str, Any]]]:
    """Split a model reply into (concatenated text, list of tool_use dicts)."""
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in getattr(reply, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            t = getattr(block, "text", "")
            if t:
                text_parts.append(t)
        elif btype == "tool_use":
            tool_uses.append(
                {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                }
            )
    return "\n".join(text_parts).strip(), tool_uses


def _text_asks_verify(text: str) -> bool:
    """True iff the model itself flagged the answer for re-checking."""
    low = (text or "").lower()
    return "verify=true" in low or "re-check this" in low


def _cluster_verify_default(cluster: str) -> bool:
    """The verdict engine's verify flag for a cluster (fragile/low -> True).

    Wrapped so a verdict failure never breaks a chat answer (defaults to False).
    """
    try:
        from agent.verdict import verdict_for_cluster

        return bool(verdict_for_cluster(cluster).verify)
    except Exception:  # pragma: no cover
        return False


# --------------------------------------------------------------------------- #
# Module-level convenience (BLUEPRINT §3 signatures)
# --------------------------------------------------------------------------- #
_DEFAULT_AGENT: Optional[PanoscopeAgent] = None


def _default_agent() -> PanoscopeAgent:
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = PanoscopeAgent()
    return _DEFAULT_AGENT


def set_cluster(cluster_id: str) -> None:
    """Set the active cluster on the module-level default agent."""
    _default_agent().set_cluster(cluster_id)


def opening_interpretation(cluster: str) -> AgentResponse:
    """Opening interpretation for ``cluster`` via the default agent."""
    return _default_agent().opening_interpretation(cluster)


def chat(
    user_msg: str,
    cluster: Optional[str] = None,
    history: Optional[list] = None,
    skill: str = _DEFAULT_SKILL,
) -> AgentResponse:
    """Answer ``user_msg`` about ``cluster`` via the default agent.

    ``skill`` selects the interpretation contract (marker vs enrichment) — the
    pathway-notes stage passes ``skill="geneset-enrichment"``.
    """
    return _default_agent().chat(user_msg, cluster=cluster, history=history, skill=skill)


def sources_of(resp: AgentResponse) -> list[Source]:
    """Return the Source chips of a response (UI helper)."""
    return list(resp.sources)


__all__ = [
    "PanoscopeAgent",
    "build_system_prompt",
    "set_cluster",
    "opening_interpretation",
    "chat",
    "sources_of",
]
