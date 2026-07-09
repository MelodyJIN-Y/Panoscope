"""The seven agent tools — Anthropic tool schemas + grounded Python impls.

The tool-use loop (``agent/loop.py``) hands the model :data:`TOOL_SCHEMAS` and,
for every tool call the model makes, invokes :func:`dispatch`. Each tool returns
the SAME envelope so the loop never special-cases a tool::

    {"ok": bool, "data": <json-safe>, "sources": [<Source dict>, ...], "error": str|None}

Grounding discipline (the confident floor, enforced here):

* **Numbers come ONLY from :mod:`agent.data`.** No tool invents a marker, a
  statistic, or a confidence value. ``panel_lookup`` / ``marker_lookup`` /
  ``get_spatial`` read precomputed jazzPanda / panel / density files and return
  exactly what is on disk (or a clean "not found").
* **Citations are real or absent.** ``literature_search`` / ``literature_fetch``
  go through the live PubMed MCP client (:mod:`agent.mcp_client`). When the
  connector is down they return ``ok=False`` (never a remembered PMID), so the
  caller falls back to the frozen citation cache — it never fabricates.
* **Memory is scoped and cited.** ``memory_read`` returns only the notes whose
  scope fires for the cluster (via :func:`agent.memory.apply_notes`, the single
  choke point). ``memory_write`` creates a note AND reconciles it against the
  literature — injecting :func:`literature_search` as the reconciler — so the
  note is born with its real-citation tension attached.

Every tool is wrapped by :func:`dispatch` in a ``try/except`` that converts any
exception into ``ok=False`` with a message, so a tool error can never crash the
loop — it degrades to a fallback instead.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

from agent import config as cfg
from agent import data
from agent import memory
from agent.types import Citation, Source

# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #
# A tool result envelope. ``sources`` is a list of Source-shaped dicts (the loop
# turns them into chips + a grounding sidecar). ``data`` is the tool payload.
Envelope = dict[str, Any]

_SPATIAL_VIEWS: tuple[str, ...] = ("cell_map", "umap", "density")


def _source_dict(src: Source) -> dict[str, Any]:
    """Serialize a :class:`~agent.types.Source` to a JSON-safe dict for the loop."""
    return {"kind": src.kind, "ref": src.ref, "value": src.value, "detail": src.detail}


def _ok(data_payload: Any, sources: Iterable[Source] = ()) -> Envelope:
    """Build a success envelope."""
    return {
        "ok": True,
        "data": data_payload,
        "sources": [_source_dict(s) for s in sources],
        "error": None,
    }


def _fail(error: str, data_payload: Any = None) -> Envelope:
    """Build a failure envelope. ``data`` may still carry context for the caller."""
    return {"ok": False, "data": data_payload, "sources": [], "error": error}


# --------------------------------------------------------------------------- #
# Live literature callable (injected into memory reconcile; used by lit tools)
# --------------------------------------------------------------------------- #
# Overridable at module level so tests can stub the network. The default lazily
# imports the MCP client so importing this module never spawns a server.
_LITERATURE_SEARCH_FN: Optional[Callable[[str], list[Citation]]] = None

# Where memory notes are written. Overridable (tests point it at a tmp dir).
_MEMORY_BASE_DIR: Optional[str] = os.getenv("PANOSCOPE_CONTEXT_DIR")

_LIT_SEARCH_DEFAULT_MAX = 5


def _citation_from_mcp(rec: dict[str, Any]) -> Citation:
    """Turn one flat MCP search/fetch dict into a real :class:`Citation`.

    Every field comes from the live server response; ``is_real=True`` because the
    PMID was resolved by the connector (never written from memory).
    """
    pmid = str(rec.get("pmid", "")).strip()
    year_raw = rec.get("year", 0)
    try:
        year = int(year_raw)
    except (TypeError, ValueError):
        year = 0
    url = rec.get("url") or (
        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    )
    return Citation(
        pmid=pmid,
        title=str(rec.get("title", "")).strip(),
        authors=str(rec.get("authors", "")).strip(),
        year=year,
        journal=str(rec.get("journal", "")).strip(),
        abstract=str(rec.get("abstract", "")).strip(),
        url=url,
        stance="context",
        is_real=bool(pmid),
    )


def _live_literature_search(query: str, max_results: int = _LIT_SEARCH_DEFAULT_MAX) -> list[Citation]:
    """Default reconciler: real PubMed search via the warm MCP client.

    Returns real :class:`Citation` objects (possibly empty). On any connector
    failure returns ``[]`` — memory reconcile then records thin literature rather
    than inventing a reference. Lazily imported so module import is side-effect
    free.
    """
    try:
        from agent import mcp_client
    except Exception:  # pragma: no cover - mcp module import guard
        return []
    try:
        client = mcp_client.get_mcp_client()
        hits = client.search_articles(query, max_results=max_results)
    except Exception:  # pragma: no cover - never raise into reconcile
        return []
    return [_citation_from_mcp(h) for h in hits if str(h.get("pmid", "")).strip()]


def set_literature_search(fn: Optional[Callable[[str], list[Citation]]]) -> None:
    """Inject the literature-search callable (tests stub the network with this)."""
    global _LITERATURE_SEARCH_FN
    _LITERATURE_SEARCH_FN = fn


def set_memory_base_dir(path: Optional[str]) -> None:
    """Point memory writes/reads at a base dir (tests isolate the real context/)."""
    global _MEMORY_BASE_DIR
    _MEMORY_BASE_DIR = path


def memory_base_dir() -> Optional[str]:
    """The base dir memory tools currently read/write (None = memory's default).

    Exposed so other layers (e.g. the loop's in-scope-notes context injection)
    read notes from the exact directory these tools write them to.
    """
    return _MEMORY_BASE_DIR


def _literature_search_fn() -> Callable[[str], list[Citation]]:
    """The active reconciler: injected stub if set, else the live MCP search."""
    return _LITERATURE_SEARCH_FN or _live_literature_search


# --------------------------------------------------------------------------- #
# 1. panel_lookup
# --------------------------------------------------------------------------- #
def panel_lookup(gene: str) -> Envelope:
    """Is ``gene`` on the analyzed panel, and what is its panel annotation?

    THE panel-absence primitive surfaced as a tool. ``on_panel=False`` means the
    gene was never measured — its absence is NOT evidence against any cell type,
    and the caller must say so. No statistic is invented; only membership + the
    panel's own annotation string are returned.
    """
    g = str(gene or "").strip()
    if not g:
        return _fail("panel_lookup requires a non-empty gene symbol")
    on_panel = data.panel_contains(g)
    annotation = data.panel_annotation(g) if on_panel else None
    payload = {"gene": g.upper(), "on_panel": on_panel, "annotation": annotation}
    detail = (
        f"on panel (annotation: {annotation})"
        if on_panel
        else "off-panel — never measured; absence is not evidence against a cell type"
    )
    src = Source(kind="panel", ref=g.upper(), value=str(on_panel), detail=detail)
    return _ok(payload, (src,))


# --------------------------------------------------------------------------- #
# 2. marker_lookup
# --------------------------------------------------------------------------- #
def _marker_source(payload: dict[str, Any]) -> Source:
    return Source(
        kind="jz",
        ref=payload["gene"],
        value=f"{payload['glm_coef']:.2f}",
        detail=(
            f"top_cluster {payload['top_cluster']}, glm_coef {payload['glm_coef']:.4f}, "
            f"pearson {payload['pearson']:.4f}"
        ),
    )


def marker_lookup(cluster: Optional[str] = None, gene: Optional[str] = None) -> Envelope:
    """Return jazzPanda numbers for a cluster's markers and/or a specific gene.

    * ``gene`` given, no ``cluster`` -> that gene's top-marker row (case-insensitive).
    * ``cluster`` given, no ``gene`` -> all of that cluster's assigned markers,
      glm_coef descending.
    * both -> that gene's row within that cluster (or a clean not-found).
    * neither -> error.

    Every number is read from ``agent.data`` (the precomputed top-marker table);
    nothing is computed here. Unknown cluster / unknown gene degrade to a clean
    envelope, never an exception past :func:`dispatch`.
    """
    cl = str(cluster).strip() if cluster else None
    gn = str(gene).strip() if gene else None
    if not cl and not gn:
        return _fail("marker_lookup requires at least one of cluster or gene")

    if cl and cl not in cfg.KNOWN_CLUSTERS:
        return _fail(f"unknown cluster {cl!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}")

    # gene-only: that gene's top-marker row
    if gn and not cl:
        row = data.get_marker(gn)
        if row is None:
            return _ok({"gene": gn.upper(), "found": False, "marker": None})
        payload = {
            "gene": str(row["gene"]),
            "top_cluster": str(row["top_cluster"]),
            "glm_coef": float(row["glm_coef"]),
            "pearson": float(row["pearson"]),
            "max_gg_corr": float(row["max_gg_corr"]),
            "max_gc_corr": float(row["max_gc_corr"]),
        }
        return _ok({"gene": gn.upper(), "found": True, "marker": payload}, (_marker_source(payload),))

    # cluster (optionally filtered to one gene)
    rows = data.get_cluster_markers(cl)  # type: ignore[arg-type]
    markers = [
        {
            "gene": str(r["gene"]),
            "top_cluster": str(r["top_cluster"]),
            "glm_coef": float(r["glm_coef"]),
            "pearson": float(r["pearson"]),
            "max_gg_corr": float(r["max_gg_corr"]),
            "max_gc_corr": float(r["max_gc_corr"]),
        }
        for _, r in rows.iterrows()
    ]
    if gn:
        markers = [m for m in markers if m["gene"].upper() == gn.upper()]
        if not markers:
            return _ok({"cluster": cl, "gene": gn.upper(), "found": False, "markers": []})

    sources = [_marker_source(m) for m in markers[:5]]
    payload: dict[str, Any] = {
        "cluster": cl,
        "cell_type": data.cell_type_for(cl),  # type: ignore[arg-type]
        "n_markers": len(markers),
        "markers": markers,
    }
    if gn:
        payload["gene"] = gn.upper()
        payload["found"] = True
    return _ok(payload, sources)


# --------------------------------------------------------------------------- #
# 3. get_spatial
# --------------------------------------------------------------------------- #
def get_spatial(view: str, marker: Optional[str] = None, bin_um: int = 50) -> Envelope:
    """Report availability + a summary of a PRECOMPUTED spatial view. Never recomputes.

    Views:

    * ``cell_map`` — segmented cells at tissue locations (marker optional; the
      pinned marker colours them). Available iff ``data.load_cells`` reads.
    * ``umap`` — expression-space embedding. Available iff ``data.load_umap`` reads.
    * ``density`` — hex-binned raw transcript density for ``marker`` at ``bin_um``.
      Available iff the precomputed frame ``{marker}_{bin_um}um.parquet`` exists.

    Returns a reference + summary (row counts, bin size) — NOT the pixels and NOT
    any recomputed statistic. A different ``bin_um`` reads a different precomputed
    frame. A missing frame yields ``available=False`` (the caller falls back to the
    cell map), never an exception.
    """
    v = str(view or "").strip().lower()
    if v not in _SPATIAL_VIEWS:
        return _fail(f"unknown spatial view {view!r}; valid views: {list(_SPATIAL_VIEWS)}")
    mk = str(marker).strip().upper() if marker else None

    if v == "cell_map":
        try:
            cells = data.load_cells()
        except FileNotFoundError as exc:
            return _ok({"view": v, "available": False, "reason": str(exc), "marker": mk})
        payload = {
            "view": v,
            "available": True,
            "marker": mk,
            "n_cells": int(len(cells)),
            "summary": f"cell map: {len(cells)} segmented cells at tissue coordinates"
            + (f", coloured by {mk}" if mk else ""),
        }
        return _ok(payload)

    if v == "umap":
        try:
            umap = data.load_umap()
        except FileNotFoundError as exc:
            return _ok({"view": v, "available": False, "reason": str(exc), "marker": mk})
        payload = {
            "view": v,
            "available": True,
            "marker": mk,
            "n_cells": int(len(umap)),
            "summary": f"UMAP embedding: {len(umap)} cells in expression space"
            + (f", coloured by {mk}" if mk else ""),
        }
        return _ok(payload)

    # density
    if not mk:
        return _fail("get_spatial(view='density') requires a marker")
    try:
        binned = data.get_density(mk, bin_um=int(bin_um))
    except FileNotFoundError as exc:
        return _ok(
            {
                "view": v,
                "available": False,
                "marker": mk,
                "bin_um": int(bin_um),
                "reason": str(exc),
            }
        )
    src = Source(
        kind="jz",
        ref=mk,
        value=f"density@{int(bin_um)}um",
        detail=f"precomputed transcript density, {len(binned)} bins at {int(bin_um)}um",
    )
    payload = {
        "view": v,
        "available": True,
        "marker": mk,
        "bin_um": int(bin_um),
        "n_bins": int(len(binned)),
        "summary": (
            f"transcript density for {mk}: {len(binned)} bins at {int(bin_um)}um "
            f"(area-normalized, precomputed)"
        ),
    }
    return _ok(payload, (src,))


# --------------------------------------------------------------------------- #
# 4. literature_search
# --------------------------------------------------------------------------- #
def _citation_payload(c: Citation) -> dict[str, Any]:
    """Flatten a Citation to a JSON-safe dict for the tool payload."""
    return {
        "pmid": c.pmid,
        "title": c.title,
        "authors": c.authors,
        "year": c.year,
        "journal": c.journal,
        "url": c.url,
    }


def _citation_source(c: Citation) -> Source:
    return Source(
        kind="lit",
        ref=c.pmid,
        value=c.title,
        detail=f"{c.authors} ({c.year}) {c.journal}".strip(),
    )


def literature_search(query: str, max_results: int = _LIT_SEARCH_DEFAULT_MAX) -> Envelope:
    """Search PubMed for real PMIDs via the live MCP connector.

    Returns ``ok=True`` with a list of real PMID records when the connector is up
    (even if zero hits — that is honest "thin literature"), and ``ok=False`` when
    the connector is unavailable, so the caller falls back to the frozen citation
    cache. Never returns a remembered / fabricated PMID.

    When a stub is injected via :func:`set_literature_search`, it is used instead
    of the network (tests), and its returned citations are treated as the result.
    """
    q = str(query or "").strip()
    if not q:
        return _fail("literature_search requires a non-empty query")

    fn = _LITERATURE_SEARCH_FN
    if fn is not None:
        # Injected stub path (tests / deterministic reconcile).
        try:
            cites = list(fn(q))
        except Exception as exc:  # noqa: BLE001
            return _fail(f"literature_search stub failed: {exc!r}")
        real = [c for c in cites if isinstance(c, Citation) and c.is_real and c.pmid]
        payload = {"query": q, "n_results": len(real), "results": [_citation_payload(c) for c in real]}
        return _ok(payload, tuple(_citation_source(c) for c in real))

    # Live MCP path.
    try:
        from agent import mcp_client
    except Exception as exc:  # pragma: no cover
        return _fail(f"literature connector unavailable: {exc!r}")

    client = mcp_client.get_mcp_client()
    if not client.available:
        return _fail(
            f"PubMed connector unavailable ({client.last_error or 'not started'}); "
            f"fall back to frozen citation cache"
        )
    hits = client.search_articles(q, max_results=max_results)
    cites = [_citation_from_mcp(h) for h in hits if str(h.get("pmid", "")).strip()]
    payload = {"query": q, "n_results": len(cites), "results": [_citation_payload(c) for c in cites]}
    return _ok(payload, tuple(_citation_source(c) for c in cites))


# --------------------------------------------------------------------------- #
# 5. literature_fetch
# --------------------------------------------------------------------------- #
def literature_fetch(pmids: list[str] | str) -> Envelope:
    """Fetch full metadata (incl. abstract) for real PMIDs via the MCP connector.

    ``ok=False`` when the connector is down (caller falls back to the frozen
    cache). Only PMIDs the server actually resolves are returned — a PMID the
    connector cannot resolve is dropped, never faked.
    """
    if isinstance(pmids, str):
        ids = [p.strip() for p in pmids.replace(",", " ").split() if p.strip()]
    else:
        ids = [str(p).strip() for p in (pmids or []) if str(p).strip()]
    ids = [p for p in ids if p.isdigit()]
    if not ids:
        return _fail("literature_fetch requires at least one numeric PMID")

    try:
        from agent import mcp_client
    except Exception as exc:  # pragma: no cover
        return _fail(f"literature connector unavailable: {exc!r}")

    client = mcp_client.get_mcp_client()
    if not client.available:
        return _fail(
            f"PubMed connector unavailable ({client.last_error or 'not started'}); "
            f"fall back to frozen citation cache"
        )
    recs = client.fetch_articles(ids)
    cites = [_citation_from_mcp(r) for r in recs if str(r.get("pmid", "")).strip()]
    resolved = {c.pmid for c in cites}
    unresolved = [p for p in ids if p not in resolved]
    payload = {
        "requested": ids,
        "n_resolved": len(cites),
        "unresolved": unresolved,
        "articles": [{**_citation_payload(c), "abstract": c.abstract} for c in cites],
    }
    return _ok(payload, tuple(_citation_source(c) for c in cites))


# --------------------------------------------------------------------------- #
# 6. memory_read
# --------------------------------------------------------------------------- #
def _note_payload(note) -> dict[str, Any]:
    """Flatten a Note (with its tension) to a JSON-safe dict for the tool payload."""
    tension = note.tension
    return {
        "id": note.id,
        "claim": note.claim,
        "scope": note.scope,
        "cluster": note.scope_ref.cluster,
        "dataset": note.scope_ref.dataset,
        "basis": note.basis,
        "status": note.status,
        "subject_cell_type": note.subject_cell_type,
        "subject_markers": list(note.subject_markers),
        "author": note.author,
        "created_at": note.created_at,
        "tension": {
            "thin": tension.thin,
            "agree": [c.pmid for c in tension.agree if c.pmid],
            "dissent": [c.pmid for c in tension.dissent if c.pmid],
            "query": tension.query,
        },
    }


def memory_read(cluster: Optional[str] = None, dataset: str = cfg.DATASET_ID) -> Envelope:
    """Return the lab notes whose scope FIRES for ``cluster`` (the scope choke point).

    Goes through :func:`agent.memory.apply_notes`, so a cluster-scoped note fires
    ONLY for its own cluster; dataset/lab notes fire for all. Each returned note
    carries a ``[note:<id>]`` citation Source — the caller MUST cite a note to use
    it (cite-on-use), and any attached tension is surfaced.
    """
    cl = str(cluster).strip() if cluster else None
    if cl and cl not in cfg.KNOWN_CLUSTERS:
        return _fail(f"unknown cluster {cl!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}")

    notes = memory.apply_notes(cl, dataset=dataset, base_dir=_MEMORY_BASE_DIR)
    payload = {
        "cluster": cl,
        "dataset": dataset,
        "n_notes": len(notes),
        "notes": [_note_payload(n) for n in notes],
    }
    sources = [memory.cite_note(n) for n in notes]
    return _ok(payload, sources)


# --------------------------------------------------------------------------- #
# 7. memory_write
# --------------------------------------------------------------------------- #
def memory_write(
    claim: str,
    scope: str,
    basis: str,
    status: str = "firm",
    cluster: Optional[str] = None,
    subject_cell_type: Optional[str] = None,
    subject_markers: Optional[list[str]] = None,
    dataset: str = cfg.DATASET_ID,
    trigger: str = "override",
) -> Envelope:
    """Create a lab note AND reconcile it against the literature (real PMIDs only).

    Delegates to :func:`agent.memory.create_note`, INJECTING the active literature
    search (the same callable :func:`literature_search` uses) so the note is born
    with its agree/dissent tension attached from real citations. The biologist's
    call is kept WITH the disagreement visible — never a bare "got it".

    Scope is enforced at birth (a cluster note must name a cluster). Returns the
    written note (with tension) + its ``[note:<id>]`` citation Source.
    """
    if not claim or not str(claim).strip():
        return _fail("memory_write requires a non-empty claim")
    if scope not in ("cluster", "dataset", "lab"):
        return _fail(f"invalid scope {scope!r}; must be cluster|dataset|lab")
    if basis not in ("paper", "own_validation", "convention"):
        return _fail(f"invalid basis {basis!r}; must be paper|own_validation|convention")

    lit_fn = _literature_search_fn()
    try:
        note = memory.create_note(
            claim=str(claim).strip(),
            scope=scope,  # type: ignore[arg-type]
            basis=basis,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            cluster=cluster,
            subject_cell_type=subject_cell_type,
            subject_markers=subject_markers,
            dataset=dataset,
            trigger=trigger,
            literature_search=lit_fn,
            base_dir=_MEMORY_BASE_DIR,
        )
    except ValueError as exc:
        return _fail(str(exc))

    src = memory.cite_note(note)
    return _ok(_note_payload(note), (src,))


# --------------------------------------------------------------------------- #
# Dispatch table + Anthropic tool schemas
# --------------------------------------------------------------------------- #
_DISPATCH: dict[str, Callable[..., Envelope]] = {
    "panel_lookup": panel_lookup,
    "marker_lookup": marker_lookup,
    "get_spatial": get_spatial,
    "literature_search": literature_search,
    "literature_fetch": literature_fetch,
    "memory_read": memory_read,
    "memory_write": memory_write,
}


def dispatch(name: str, args: dict[str, Any] | None = None) -> Envelope:
    """Invoke tool ``name`` with keyword ``args``, returning the uniform envelope.

    Any unknown tool, bad argument shape, or impl exception is converted to an
    ``ok=False`` envelope — a tool call can never crash the loop; it degrades to a
    fallback. This is the ONLY entry point the loop uses.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return _fail(f"unknown tool {name!r}; tools: {sorted(_DISPATCH)}")
    kwargs = dict(args or {})
    try:
        return fn(**kwargs)
    except TypeError as exc:
        # Bad/missing argument names from the model.
        return _fail(f"{name} bad arguments: {exc}")
    except KeyError as exc:
        return _fail(f"{name} not found: {exc}")
    except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
        return _fail(f"{name} failed: {exc!r}")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "panel_lookup",
        "description": (
            "Check whether a gene is on the analyzed spatial panel and return its "
            "panel annotation. THE panel-absence primitive: if on_panel is false the "
            "gene was NEVER measured, so its absence is not evidence against any cell "
            "type — say 'not measured', never 'not expressed'. Use before down-weighting "
            "any missing canonical marker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene": {"type": "string", "description": "Gene symbol, e.g. 'ERBB2' or 'COL1A1'."}
            },
            "required": ["gene"],
        },
    },
    {
        "name": "marker_lookup",
        "description": (
            "Return jazzPanda's precomputed marker numbers (glm_coef, pearson, "
            "max_gg_corr, max_gc_corr) for a cluster's assigned markers and/or a specific "
            "gene. Provide a cluster (c1..c9) to list its markers glm_coef-descending, a "
            "gene to get that gene's top-marker row, or both to look one gene up inside a "
            "cluster. These are the ONLY numbers you may state; never invent a statistic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster": {
                    "type": "string",
                    "description": "Cluster id c1..c9. Optional if gene is given.",
                },
                "gene": {
                    "type": "string",
                    "description": "Gene symbol. Optional if cluster is given.",
                },
            },
        },
    },
    {
        "name": "get_spatial",
        "description": (
            "Report availability and a summary of a PRECOMPUTED spatial view — it never "
            "recomputes and never changes a value. Views: 'cell_map' (segmented cells at "
            "tissue locations, default), 'umap' (expression space), 'density' (hex-binned "
            "raw transcript density for a marker at a bin size in um). The bin size and view "
            "are viewing controls, not analysis knobs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": list(_SPATIAL_VIEWS),
                    "description": "Which spatial view to reference.",
                },
                "marker": {
                    "type": "string",
                    "description": "Marker to colour/bin by. Required for 'density'.",
                },
                "bin_um": {
                    "type": "integer",
                    "description": "Density bin size in um (e.g. 25, 50, 100). Density only.",
                    "default": 50,
                },
            },
            "required": ["view"],
        },
    },
    {
        "name": "literature_search",
        "description": (
            "Search PubMed live for REAL PMIDs supporting or contesting an interpretive "
            "claim. Returns real PMID records (title, authors, year, journal). If the "
            "connector is down the call fails and you must say the literature lookup is "
            "unavailable — NEVER write a PMID from memory. If it returns zero hits, say the "
            "literature is thin; do not invent a reference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text query, e.g. 'LUM lumican fibroblast marker breast'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max PMIDs to return.",
                    "default": _LIT_SEARCH_DEFAULT_MAX,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "literature_fetch",
        "description": (
            "Fetch full metadata (including abstract) for one or more real PMIDs via the "
            "PubMed connector. Use to expand a citation you already found with "
            "literature_search. Only PMIDs the server resolves are returned; unresolved ids "
            "are reported, never faked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more numeric PMIDs.",
                }
            },
            "required": ["pmids"],
        },
    },
    {
        "name": "memory_read",
        "description": (
            "Return the lab's stored notes that are IN SCOPE for a cluster. A cluster-scoped "
            "note fires only for its own cluster; dataset/lab notes fire for all. You MUST "
            "cite a note ([note:<id>]) to use it, and you must surface any tension it carries "
            "(agreeing vs dissenting citations). Never apply a note out of its scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster": {
                    "type": "string",
                    "description": "Cluster id c1..c9 to read in-scope notes for.",
                }
            },
        },
    },
    {
        "name": "memory_write",
        "description": (
            "Capture a biologist override/correction as a scoped lab note AND cross-check it "
            "against the literature (real PMIDs), attaching the agree/dissent tension. Use at "
            "an override: record the claim, its scope (cluster|dataset|lab), and basis "
            "(paper|own_validation|convention). Keeps the biologist's call WITH the "
            "disagreement visible — never a bare acknowledgement, never a silent overrule."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The correction/claim in plain words."},
                "scope": {
                    "type": "string",
                    "enum": ["cluster", "dataset", "lab"],
                    "description": "How widely the note applies.",
                },
                "basis": {
                    "type": "string",
                    "enum": ["paper", "own_validation", "convention"],
                    "description": "What the claim rests on.",
                },
                "status": {
                    "type": "string",
                    "enum": ["firm", "tentative"],
                    "default": "firm",
                },
                "cluster": {
                    "type": "string",
                    "description": "Required when scope is 'cluster'. The cluster c1..c9.",
                },
                "subject_cell_type": {
                    "type": "string",
                    "description": "Cell type the note is about (optional).",
                },
                "subject_markers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Markers the note is about (optional).",
                },
            },
            "required": ["claim", "scope", "basis"],
        },
    },
]

# Sanity: schema list and dispatch table must name exactly the same 7 tools.
assert {s["name"] for s in TOOL_SCHEMAS} == set(_DISPATCH), (
    "TOOL_SCHEMAS and dispatch table disagree on the tool set"
)


__all__ = [
    "TOOL_SCHEMAS",
    "dispatch",
    "panel_lookup",
    "marker_lookup",
    "get_spatial",
    "literature_search",
    "literature_fetch",
    "memory_read",
    "memory_write",
    "set_literature_search",
    "set_memory_base_dir",
]
