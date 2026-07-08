# PANOSCOPE — Authoritative Implementation Blueprint

Lead architect synthesis of six subsystem specs into one buildable plan. Every load-bearing value below is re-verified against the on-disk files this session. Where subsystems disagreed, one contract is chosen and marked **[RECONCILED]**.

## Key reconciliations (read first)

| Conflict | Subsystems | **Decision** |
|---|---|---|
| Tidy data location | data-layer used `data/*.parquet`; agent-core used `data/tidy/*.parquet`; verdict used `data/jazzpanda/*.csv` | **[RECONCILED]** All tidy outputs under `data/` per CLAUDE.md subdirs: `data/jazzpanda/`, `data/panels/`, `data/cells/`, `data/embeddings/`, `data/density/`. Format: **parquet** for large frames, **CSV** for the two small marker tables (human-diffable, no pyarrow risk), **JSON** for keys/manifests. |
| jazzPanda source | verdict wanted CSV copy; data-layer wanted `res_lst.Rds` | **[RECONCILED]** R prep reads `xenium_hbreast_jazzPanda_res_lst.Rds` as the single authoritative source for BOTH top+full, writes `data/jazzpanda/markers_top.csv` + `markers_full.csv`. The existing `jazzPanda_top_marker.csv` is a cross-check only. |
| Note format | memory used YAML; grounding-tests + agent-core used JSON | **[RECONCILED]** **JSON** (`.json`) under `context/corrections/`. Unambiguous to parse, matches grounding-test contract. One file per note, git-tracked. |
| Loader module name | data-layer `agent/data.py`; UI `ui/data_access.py`; grounding `SourceIndex` | **[RECONCILED]** One pure loader `agent/data.py` owns all tidy-file reads + panel/marker/absence primitives. `ui/data_access.py` is a **thin `@st.cache_data` wrapper** over `agent/data.py`. Grounding's `SourceIndex` **imports `agent/data.py`** (no second parser). |
| Verdict output type | verdict `ClusterVerdict`; agent-core `Verdict`; UI `Verdict` | **[RECONCILED]** One frozen dataclass **`ClusterVerdict`** in `agent/types.py`. Agent-core's `verdict.assess()` = alias for `verdict_for_cluster()`. |
| Confidence percentile axis | verdict computes at runtime; data-layer precomputes `coef_pctl_in_cluster` | **[RECONCILED]** Precompute `coef_pctl_in_cluster` + `n_markers_in_cluster` in R prep AND expose them on `MarkerRow`; `verdict.py` reads them (single source, never re-derives). |
| Inline citation/tag convention | agent-core `⟦jz:…⟧`; grounding `PMID:xxxx`/`[note:id]` | **[RECONCILED]** Agent emits a **machine-readable sidecar JSON** on `AgentResponse.grounding` (exact numbers/PMIDs/note-ids used) AND renders `PMID:xxxxxxx` + `[note:id]` inline in prose. Grounding checker prefers the sidecar, falls back to prose. Inline `⟦…⟧` tags dropped (sidecar replaces them). |
| Cluster count in UI | wireframe 6 clusters | **[RECONCILED]** 9 clusters c1..c9, authoritative key. Wireframe is layout reference only. |

---

## 1. FILE TREE

```
Panoscope/
  requirements.txt                      # [NEW] anthropic, mcp, python-dotenv, pandas, pyarrow, plotly, streamlit, pytest, pyyaml
  app.py                                # Streamlit entrypoint: config, state init, 3-pane shell, drawers
  .mcp.json                             # [EXISTS] pubmed stdio server
  .env / .env.example                   # [EXISTS] ANTHROPIC_API_KEY, NCBI_API_KEY, NCBI_ADMIN_EMAIL

  scripts/
    prep_data.R                         # ONE-TIME: .Rds -> tidy data/ files (sample1 only); emits PREP_MANIFEST.json
    precompute_density.py               # ONE-TIME: transcripts.csv.gz -> data/density/{gene}_{bin}um.parquet
    build_fallbacks.py                  # ONE-TIME: run real loop per cluster/prompt -> data/precomp/fallbacks.json + citations.json

  agent/
    __init__.py
    types.py                            # ALL shared frozen dataclasses (MarkerRow, MarkerEvidence, ClusterVerdict, Note, Citation, OpeningInterpretation, AgentResponse, Source)
    data.py                             # pure loader: markers/panel/cells/umap/density + panel_contains/panel_annotation (THE absence primitive)
    verdict.py                          # deterministic cell-type call + confidence band + verify (reads jazzPanda numbers + panel; no network/LLM)
    memory.py                           # note create/read/reconcile/scope-enforce/cite; writes context/corrections/*.json + decisions log
    tools.py                            # tool schemas + impls: panel_lookup, marker_lookup, get_spatial, literature_search/fetch, memory_read/write
    mcp_client.py                       # persistent PubMed MCP stdio client (background asyncio loop, warm session)
    loop.py                             # tool-use loop: chat(), set_cluster(), opening_interpretation(); finalize+grounding gate
    fallback.py                         # FallbackStore: pre-baked grounded AgentResponses; generic verdict-based fallback
    config.py                           # constants: PRIMARY_MODEL, DATA_DIR, thresholds, cluster key, demo markers
    grounding_check.py                  # GroundingChecker + SourceIndex + Extractor + LiteratureVerifier (shared by loop finalize AND tests)

  ui/
    __init__.py
    state.py                            # session_state schema + init_state() + typed accessors
    theme.py                            # inject_css() — design tokens from wireframe
    data_access.py                      # @st.cache_data / @st.cache_resource wrappers over agent/data.py + agent/verdict.py
    format.py                           # pure formatters: role_chip, confidence_chip, CLUSTER_COLORS, num_fmt
    cluster_rail.py                     # render_rail() — left pane, 9 clusters
    verdict_header.py                   # render_verdict() — call + confidence chip + rationale + confirm/override
    evidence_table.py                   # render_evidence_table() — role column + click-to-pin
    spatial_stage.py                    # render_spatial_stage() — 3 linked Plotly views + bin/view controls
    conversation.py                     # render_conversation() — chat thread + opening interp + capture-at-override
    lab_knowledge.py                    # render_lab_panel() — notes + tension drawer
    paper_drawer.py                     # render_paper_drawer() — citation paper modal

  data/                                 # ALL tidy outputs (git-ignored except keys/manifest); produced by scripts/
    jazzpanda/markers_top.csv           # gene, top_cluster, glm_coef, pearson, max_gg_corr, max_gc_corr, cell_type, coef_pctl_in_cluster, n_markers_in_cluster
    jazzpanda/markers_full.csv          # gene, cluster, glm_coef, p_value, pearson, max_gg_corr, max_gc_corr, is_covariate
    panels/panel.parquet                # gene, ensembl_id, annotation (312 rows)
    cluster_key.json                    # {c1:{cell_type,cell_type_short,category,lineage}, ...} 9 entries
    cells/cells.parquet                 # cell_id, cluster, x, y (158,379 sample1 rows)
    embeddings/umap.parquet             # cell_id, umap_1, umap_2, cluster
    embeddings/marker_expr.parquet      # cell_id + one col per demo marker (narrow, for UMAP feature coloring)
    density/{GENE}_{bin}um.parquet      # hx, hy, count, density (area-normalized); + density/_index.json
    precomp/fallbacks.json              # frozen real AgentResponses for demo beats
    precomp/citations.json              # frozen real PMIDs (title/authors/year/journal/abstract/url/fetched_at)
    PREP_MANIFEST.json                  # per-file row counts + sha256 + generated_at

  context/
    corrections/                        # lab notes, one JSON per note, git-tracked
    decisions/decision_log.jsonl        # append-only event ledger
    README.md                           # note format doc

  skills/jazzpanda-markers/             # [EXISTS] SKILL.md + references/output_template.md

  tests/
    conftest.py                         # session fixtures: SourceIndex, GroundingChecker, MCP mode switch
    fixtures/answers/*.json             # clean + poisoned recorded answers
    fixtures/cassettes/pubmed_efetch.json   # recorded MCP responses keyed by PMID
    fixtures/notes/*.json               # sample notes for scope tests
    data/calibration_set.yaml           # calibration cases + expected verdicts (renders to README table)
    test_data_loader.py                 # loader schema + row-count + provenance asserts
    test_verdict.py                     # band re-derivability + small-n + panel-absence invariant
    test_memory.py                      # scope enforcement + cite-on-use + tension
    test_extract.py                     # gene/number/PMID extraction precision incl. AR/KIT/LIF false-positives
    test_grounding_numbers.py           # invented number FAILS
    test_grounding_markers.py           # invented marker FAILS
    test_grounding_citations.py         # fabricated PMID FAILS (cassette)
    test_grounding_notes.py             # uncited/out-of-scope note FAILS
    test_panel_absence.py               # off-panel absence never down-weights (the headline invariant)
    test_calibration.py                 # parametrized over calibration_set.yaml
    test_live_pubmed.py                 # ONE real MCP lookup (@pytest.mark.live, skipped offline)

  README.md                             # quickstart, calibration table, install
  .github/workflows/grounding.yml       # CI: pytest -m "not live" (deterministic, green badge)
```

---

## 2. SHARED CONTRACTS (`agent/types.py`)

All frozen (immutability rule). These cross every module boundary.

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

Confidence  = Literal["Very High", "High", "Medium-High", "Medium", "Low"]
MarkerRole  = Literal["supports", "expected_absent", "off_panel"]
Scope       = Literal["cluster", "dataset", "lab"]
Basis       = Literal["paper", "own_validation", "convention"]
Status      = Literal["firm", "tentative"]

# --- raw jazzPanda row (from data/jazzpanda/markers_top.csv) ---
@dataclass(frozen=True)
class MarkerRow:
    gene: str
    top_cluster: str                 # c1..c9 | NoSig
    glm_coef: float
    pearson: float
    max_gg_corr: float
    max_gc_corr: float
    cell_type: Optional[str]         # joined from cluster_key; None for NoSig
    coef_pctl_in_cluster: Optional[float]   # 0..100, PRECOMPUTED in R prep; None for NoSig
    n_markers_in_cluster: Optional[int]     # PRECOMPUTED; drives small-n branch

# --- evidence row the verdict/UI use (role column = panel-absence rule made visible) ---
@dataclass(frozen=True)
class MarkerEvidence:
    gene: str
    top_cluster: str
    glm_coef: float
    pearson: float
    max_gg_corr: float
    max_gc_corr: float
    p_value: Optional[float]         # from markers_full c\d+ term only, else None
    within_cluster_pctile: float     # 0..1 (1.0 = strongest in cluster)
    is_canonical: bool
    is_on_panel: bool
    role: MarkerRole
    caveats: tuple[str, ...] = ()
    source: str = "jazzpanda:top_result"

@dataclass(frozen=True)
class OffPanelNote:
    gene: str                        # e.g. "COL1A1"
    cell_type: str                   # "Stromal"
    message: str                     # "COL1A1 is off-panel (never measured); its absence is not evidence against Stromal."
    source: str = "panel:absence"

@dataclass(frozen=True)
class Citation:
    pmid: str
    title: str
    authors: str
    year: int
    journal: str
    abstract: str = ""
    url: str = ""                    # https://pubmed.ncbi.nlm.nih.gov/{pmid}/
    stance: str = "context"          # agree | dissent | context | unclassified
    is_real: bool = True             # True only if resolved via live MCP or frozen-real cache
    fetched_at: str = ""             # iso; honest snapshot stamp for cached citations

@dataclass(frozen=True)
class LiteratureHook:                # engine emits WHAT to look up; loop fills live. Engine writes ZERO citations.
    claim: str
    marker: str
    cell_type: str
    query_terms: tuple[str, ...]
    status: Literal["unfilled"] = "unfilled"

@dataclass(frozen=True)
class OpeningInterpretation:
    cluster: str
    cell_type: str
    confidence: Confidence
    headline: str
    driving_markers: tuple[MarkerEvidence, ...]
    offpanel_notes: tuple[OffPanelNote, ...]
    literature_hooks: tuple[LiteratureHook, ...]
    verify: bool

@dataclass(frozen=True)
class ClusterVerdict:                # == the CSV output contract + UI affordances
    cluster: str
    cell_type: str
    cell_type_short: str
    confidence: Confidence
    confidence_score: float          # fixed band anchor {0.95,0.85,0.70,0.55,0.30}
    key_markers: tuple[str, ...]     # top 3-5 by glm_coef
    notes: str                       # grounded rationale, cites glm_coef/pearson
    category: str
    lineage: str
    exclude: bool
    verify: bool
    # UI / audit extras (not in CSV):
    small_n: bool
    evidence: tuple[MarkerEvidence, ...]
    offpanel_notes: tuple[OffPanelNote, ...]
    opening: OpeningInterpretation
    band_basis: str                  # "percentile" | "small-n absolute"
    demotions: tuple[str, ...]       # audit trail of band changes
    source_trace: tuple[str, ...]    # every (gene,stat,value) used — grounding tests read this

# --- memory ---
@dataclass(frozen=True)
class ScopeRef:
    dataset: str
    cluster: Optional[str]           # set iff scope=="cluster"

@dataclass(frozen=True)
class Tension:
    agree: tuple[Citation, ...]
    dissent: tuple[Citation, ...]
    thin: bool
    query: str
    looked_up_at: str

@dataclass(frozen=True)
class Note:
    id: str
    claim: str
    scope: Scope
    scope_ref: ScopeRef
    basis: Basis
    status: Status
    subject_cell_type: Optional[str]
    subject_markers: tuple[str, ...]
    tension: Tension
    author: str
    created_at: str
    trigger: Literal["override", "manual_add", "holistic_review"]
    supersedes: Optional[str]

# --- agent I/O ---
@dataclass(frozen=True)
class Source:
    kind: Literal["jz", "panel", "lit", "mem"]
    ref: str                         # gene | pmid | note_id
    value: Optional[str]             # glm_coef value | title | claim
    detail: str = ""

@dataclass(frozen=True)
class GroundingSidecar:              # machine-readable claim manifest the checker prefers
    numbers: tuple[tuple[str, str, float], ...]   # (gene, stat, value)
    markers: tuple[str, ...]
    pmids: tuple[str, ...]
    notes_used: tuple[str, ...]

@dataclass(frozen=True)
class AgentResponse:
    text: str                        # markdown, PMID:xxx + [note:id] inline
    sources: tuple[Source, ...]
    verify: bool
    grounding: GroundingSidecar
    pin_marker: Optional[str] = None
    citations: tuple[Citation, ...] = ()
    note_written: Optional[Note] = None
    used_fallback: bool = False
    opening: bool = False
```

---

## 3. MODULE SPECS (final reconciled signatures)

### `agent/config.py` — constants
```python
DATA_DIR = Path(__file__).parent.parent / "data"
PRIMARY_MODEL = os.getenv("PANOSCOPE_MODEL", "claude-sonnet-4-6")
DATASET_ID = "xenium_hbreast_sample1"
CLUSTER_KEY = {"c1":"Tumor","c2":"Stromal","c3":"Macrophages","c4":"Myoepithelial",
               "c5":"T_Cells","c6":"B_Cells","c7":"Endothelial","c8":"Dendritic","c9":"Mast_Cells"}
KNOWN_CLUSTERS = frozenset(CLUSTER_KEY)
DEMO_MARKERS = ["LUM","POSTN","PDGFRA","CD3D","CD8A","EPCAM","ERBB2","ACTA2","KRT14","PECAM1","CD68","MS4A1","CPA3"]
# verdict thresholds
SMALL_N_THRESHOLD = 8
STRONG_ABS, STRONG_PEARSON = 1.5, 0.60
WEAK_ABS, WEAK_PEARSON = 0.75, 0.40
LOW_PEARSON, GG_NOT_UNIQUE, EPS = 0.30, 0.98, 1e-9
SCORE_MAP = {"Very High":0.95,"High":0.85,"Medium-High":0.70,"Medium":0.55,"Low":0.30}
NUMBER_ABS_TOL, NUMBER_REL_TOL = 1e-3, 0.005
MAX_TOOL_ROUNDS, AGENT_TIMEOUT_S = 6, 25
```

### `agent/data.py` — pure loader (the interface everything reads)
```python
def load_markers() -> pd.DataFrame          # markers_top.csv, 280 rows, cached
def load_markers_full() -> pd.DataFrame      # markers_full.csv, incl is_covariate
def get_cluster_markers(cluster: str, include_nosig=False) -> pd.DataFrame  # sorted glm_coef desc; KeyError if bad
def get_marker(gene: str) -> pd.Series | None       # case-insensitive
def load_cluster_key() -> dict[str, dict]           # 9 entries
def cell_type_for(cluster: str) -> str
def load_cells() -> pd.DataFrame                     # cell_id,cluster,x,y
def get_cluster_cells(cluster: str) -> pd.DataFrame
def load_umap() -> pd.DataFrame                      # cell_id,umap_1,umap_2,cluster
def marker_expression(gene: str) -> pd.Series | None # per-cell, demo markers only; None if not exported
def load_panel() -> pd.DataFrame                     # gene,ensembl_id,annotation (312)
def panel_contains(gene: str) -> bool                # O(1) frozenset, case-insensitive — THE absence primitive
def panel_annotation(gene: str) -> str | None
def available_density_markers() -> list[str]
def get_density_hexbins(gene: str, bin_um: int = 50) -> pd.DataFrame   # FileNotFoundError w/ clear msg -> caller falls back to cell map
def density_meta() -> dict
def load_manifest() -> dict
```
Guarantees: `panel_contains` is the sole absence source; loader never computes a NEW statistic (percentile/count precomputed in R); different `bin_um` returns a different precomputed frame, never a recomputed value.

### `agent/verdict.py` — deterministic engine
```python
def verdict_for_cluster(cluster: str, notes: list[Note] | None = None) -> ClusterVerdict
def assess(cluster: str) -> ClusterVerdict          # alias for agent-core naming
def all_verdicts(notes: list[Note] | None = None) -> list[ClusterVerdict]   # c1..c9 order
def holistic_review(verdicts: list[ClusterVerdict]) -> list[ClusterVerdict] # pure flags; relabel surfaced, never auto-applied
def to_csv(verdicts: list[ClusterVerdict], header=True) -> str              # 11-col contract, csv.writer
```

### `agent/memory.py`
```python
def create_note(*, claim, scope, basis, status="firm", cluster=None, subject_cell_type=None,
                subject_markers=None, attributed_to="melody.xyjin@gmail.com",
                trigger="override", supersedes=None, literature_search=None) -> Note
def apply_notes(cluster: str | None, dataset=DATASET_ID) -> list[Note]      # THE scope choke point
def reconcile(note: Note, literature_search=None) -> Tension               # agree/dissent split, real PMIDs only
def note_in_scope(note: Note, *, cluster, dataset) -> bool                  # lab⊇dataset⊇cluster, fail-closed
def render_citation(note: Note, *, refresh=None) -> str                     # cite-on-use markdown, shows tension
def list_notes(dataset=None) -> list[Note]
def supersede_note(old_id, **new_fields) -> Note                            # immutable "edit"
def log_decision(*, kind, cluster=None, note_id=None, actor=..., detail=None) -> None
```

### `agent/tools.py` — 7 tools, uniform envelope `{ok, data, sources, error}`
```python
TOOL_SCHEMAS: list[dict]                             # JSON schemas for the loop
def dispatch(name: str, args: dict) -> dict          # try/except -> ok:false on failure
# impls: panel_lookup, marker_lookup, get_spatial, literature_search, literature_fetch, memory_read, memory_write
```

### `agent/mcp_client.py`
```python
class PubMedMCP:
    def call_tool(self, name: str, args: dict, timeout: float = 8) -> dict   # thread-safe over background asyncio loop
    def health(self) -> bool
def get_mcp_client() -> PubMedMCP                     # singleton, @st.cache_resource-friendly
MCP_LIVE: bool
```

### `agent/loop.py`
```python
def chat(user_msg, cluster, history=None, scope="cluster", live=True) -> AgentResponse
def set_cluster(cluster_id: str) -> None
def opening_interpretation(cluster: str) -> AgentResponse      # posts before any question, live citation
def sources_of(resp) -> list[Source]
```
`finalize()` runs the in-process grounding gate (§5): if `GroundingChecker.check()` finds CRITICAL/HIGH, discard the answer → `FallbackStore`. No exception ever reaches Streamlit.

### `agent/grounding_check.py` (shared by loop + tests)
```python
class SourceIndex:                                   # wraps agent/data.py, adds notes index
    def number_matches(self, gene, cluster, stat, value) -> bool
    def is_on_panel(self, gene) -> bool
    def is_modeled(self, gene) -> bool
class Extractor:
    def extract(self, answer: str, sidecar: GroundingSidecar | None) -> list[Claim]
class LiteratureVerifier:                            # cassette (default) | live
    def resolves(self, ident: str) -> bool
class GroundingChecker:
    def check(self, answer, *, cluster_ctx=None, allowed_notes=None, sidecar=None) -> GroundingReport
```

### `ui/data_access.py` — thin cached wrappers
```python
@st.cache_data def load_all_verdicts() -> list[ClusterVerdict]   # wraps verdict.all_verdicts()
@st.cache_data def cells_df(), umap_df(), hexbins(gene,bin_um), panel_names() ...
@st.cache_resource def mcp() -> PubMedMCP
# read_notes NOT cached (mutates on save)
```

---

## 4. VERDICT ALGORITHM (final pseudocode)

```
function verdict_for_cluster(cluster, notes=None):
    meta = cluster_key[cluster]                       # cell_type, short, category, lineage — 1:1, KeyError if unknown
    rows = get_cluster_markers(cluster)               # assigned markers, glm_coef desc (NoSig excluded)
    n = len(rows)

    # --- build evidence (percentile precomputed; role; caveats) ---
    evidence = []
    for r in rows:
        pctile   = r.coef_pctl_in_cluster / 100        # PRECOMPUTED, 0..1, 1.0=strongest
        canonical = gene_is_canonical(r.gene, meta.cell_type)   # from panel Annotation-derived list, grounded
        caveats = []
        if r.max_gc_corr > r.pearson + EPS: caveats += ["localizes better with another cluster"]
        if r.max_gg_corr >= GG_NOT_UNIQUE:  caveats += ["spatial pattern not unique"]
        if r.pearson < LOW_PEARSON:         caveats += ["low spatial specificity"]
        role = "supports"        if canonical and r.pearson >= LOW_PEARSON
             else "expected_absent" if canonical                  # on-panel canonical but weak -> REAL down-weight
             else "supports"                                      # non-canonical on-panel supporter
             # off_panel never reaches here (assigned rows are on-panel by construction)
        evidence.append(MarkerEvidence(..., within_cluster_pctile=pctile, is_canonical=canonical, role=role, caveats=tuple(caveats)))

    drivers = [e for e in evidence if e.is_canonical and e.role=="supports"]

    # --- BAND ---
    if n < SMALL_N_THRESHOLD:                          # c6=7, c8=4, c9=2 -> small-n
        band, score, basis, verify = small_n_band(drivers)
    else:
        band, score, basis, verify = percentile_band(drivers)
        band, score, verify, demotions = apply_modifiers(band, score, verify, drivers)

    # --- PANEL-ABSENCE (notes ONLY; NEVER touches band/score/verify) ---
    offpanel = [OffPanelNote(g, meta.cell_type, f"{g} is off-panel (never measured); its absence is not evidence against {meta.cell_type}.")
                for g in offpanel_canonical(meta.cell_type)]   # asserts each is truly off-panel

    exclude = decide_exclude(cluster, evidence)        # FALSE for all in P0 unless NoSig-dominated / lab note

    # --- lab-note override (scoped, cited; may relabel; records tension; NEVER silent) ---
    meta, band, score, verify, note_trace = apply_lab_notes(meta, band, score, verify, cluster, notes)

    opening = build_opening(cluster, meta, band, drivers, offpanel, verify)   # lit_hooks UNFILLED (loop fills live)
    return ClusterVerdict(..., confidence=band, confidence_score=score, verify=verify, exclude=exclude,
                          key_markers=tuple(top 3-5 genes by glm_coef), notes=compose_notes(...),
                          band_basis=basis, source_trace=collect_trace(evidence, offpanel, note_trace))


function percentile_band(drivers):                    # n >= 8. top_pct = 1 - within_cluster_pctile
    if not drivers:  return ("Low", 0.30, "percentile", True)
    best_top_pct = min(1 - d.within_cluster_pctile for d in drivers)   # strongest canonical supporter's rank
    if   best_top_pct > 0.85: band="Low"
    elif best_top_pct <=0.15: band="Very High"
    elif best_top_pct <=0.35: band="High"
    elif best_top_pct <=0.60: band="Medium-High"
    else:                     band="Medium"
    # coherence: Very High/High require >=2 canonical drivers in-band; else cap one band (record demotion)
    if band in ("Very High","High") and len(drivers) < 2:
        band = demote(band)                            # "single canonical marker; multiple-agree not met"
    verify = (band == "Low")
    return (band, SCORE_MAP[band], "percentile", verify)


function small_n_band(drivers):                       # n < 8. percentiles unstable -> ABSOLUTE strength. CAP Medium-High. verify ALWAYS True.
    if not drivers: return ("Low", 0.30, "small-n absolute", True)
    top = max(drivers, key=glm_coef)
    if   top.glm_coef >= STRONG_ABS and top.pearson >= STRONG_PEARSON: band="Medium-High"   # CAP
    elif top.glm_coef >= WEAK_ABS   and top.pearson >= WEAK_PEARSON:   band="Medium"
    else:                                                              band="Low"
    return (band, SCORE_MAP[band], "small-n absolute", True)           # verify TRUE always


function apply_modifiers(band, score, verify, drivers):   # percentile branch only; each demotes <=1 band, logged
    BANDS=["Very High","High","Medium-High","Medium","Low"]; demotions=[]
    top = max(drivers, key=glm_coef)  if drivers else None
    for cond, msg in [(top and top.pearson<LOW_PEARSON,"low pearson on driver"),
                      (top and top.max_gc_corr>top.pearson+EPS,"driver localizes better elsewhere"),
                      (top and top.max_gg_corr>=GG_NOT_UNIQUE,"driver not spatially unique")]:
        if cond: band=BANDS[min(BANDS.index(band)+1,4)]; demotions.append(msg)
    verify = verify or (band=="Low")
    return band, SCORE_MAP[band], verify, tuple(demotions)
```

**Verified anchors this produces:** c1 Tumor — ERBB2 glm 21.44/pearson 0.91 (top of 84) → **Very High, verify=FALSE**. c2 Stromal — LUM glm 18.00/pearson 0.91 + POSTN glm 15.80/pearson 0.80 (2 drivers agree) → **High/Very-High, verify=FALSE**, + 5 off-panel notes (COL1A1/COL1A2/DCN/VIM/FAP). c9 Mast — CPA3 glm 1.95/pearson 0.42 (n=2, pearson<0.60) → small-n → **Medium, verify=TRUE**.

---

## 5. MCP + AGENT INTEGRATION (runtime + deterministic fallback)

**MCP client** (`mcp_client.py`): The Streamlit app is its own MCP client (does NOT inherit Claude Code's servers). One persistent asyncio event loop on a dedicated daemon thread, started once (`@st.cache_resource`). Spawns the same stdio server `.mcp.json` declares (`npx -y @cyanheads/pubmed-mcp-server@latest`) with `NCBI_*` from `.env` via python-dotenv, runs `initialize()` once, keeps the session warm. `call_tool(name,args,timeout=8)` submits a coroutine via `asyncio.run_coroutine_threadsafe(...).result(timeout)`. On app start, one background probe (`pubmed_search_articles("CD3D T cell", max=1)`) sets `MCP_LIVE`. Tools normalize MCP `content` blocks into flat `{pmid,title,authors,year,journal}`.

**Loop** (`loop.py`): system prompt = cached SKILL.md + output_template.md + GROUNDING_CONTRACT + CLUSTER_KEY + active-cluster context. Standard Anthropic tool-use loop, `temperature=0`, `max_tokens=2048`. Guards: `MAX_TOOL_ROUNDS=6`, `AGENT_TIMEOUT_S=25`, all API calls in `try/except → fallback`. `finalize()` builds the `GroundingSidecar`, runs `GroundingChecker.check()`; any CRITICAL/HIGH → discard → `FallbackStore`.

**Three-layer deterministic fallback** (demo never breaks, no spinner):
1. **Pre-baked responses** — `scripts/build_fallbacks.py` runs the real loop offline per cluster + scripted prompt, freezes to `data/precomp/fallbacks.json` with real citations. `FallbackStore.match(user_msg, cluster)` keys on cluster + normalized intent.
2. **Frozen citation cache** — `data/precomp/citations.json` holds real PMIDs fetched offline (stamped `fetched_at`), so opening interpretations show real clickable PubMed links even with network off. The one live demo call (override cross-check) still hits MCP with this as its net.
3. **Generic grounded fallback** — no match → run `verdict_for_cluster()` (pure local) and template the call + confidence + top markers with glm_coef. 100% grounded, cited to jazzPanda.

Every fallback carries the same `sources`/`verify`/`grounding` envelope, so grounding tests pass on fallbacks too.

---

## 6. DEMO-FLOW MAPPING

| Demo beat | Modules | Screen |
|---|---|---|
| **Cold open** — plain LLM mis-calls a cluster (off-panel marker looks absent) | (narration; contrast slide) | — |
| **Open the cluster** — opening interpretation already posted with cited markers | `loop.opening_interpretation(c2)` → `verdict.build_opening` + live `literature_fetch`; frozen citation cache net | `conversation.py` (opening bubble) + `verdict_header.py` |
| **The catch** — missing canonical (COL1A1/DCN/VIM/FAP) is off-panel → uninformative; confidence holds on LUM/POSTN | `verdict.offpanel_notes` + `data.panel_contains`; band computed before notes | `evidence_table.py` role column (`⊘ off-panel — not measured`) + offnote line |
| **Calibration** — clean Very-High (c1 Tumor, ERBB2) vs shaky verify (c9 Mast, small-n) | `verdict_for_cluster` percentile vs small_n_band | `cluster_rail.py` (verify flag) + `verdict_header.py` (confidence chip + verify badge) |
| **Holistic review** — agent revises one call after seeing all clusters | `verdict.holistic_review(all_verdicts())`; `memory.log_decision(kind="holistic_revision")` | `conversation.py` (revision message) |
| **Memory** — biologist overrides; agent asks scope+basis (two taps), cross-checks literature, keeps call with tension visible, cites back | `memory.create_note` + `memory.reconcile` (live `literature_search`) → next turn `memory.render_citation` + `apply_notes` | `conversation.py` capture panel + `lab_knowledge.py` tension drawer |
| **Click-a-marker-to-pin** — one pin drives 3 linked views | `state.pinned_marker` → `data.get_cluster_cells`/`load_umap`/`get_density_hexbins` | `evidence_table.py` pin button + `spatial_stage.py` (cell map default, UMAP, density) |
| **Live literature call** (P0) — one real MCP lookup, real PMID | `mcp_client.call_tool("pubmed_fetch_articles")` during override cross-check | `paper_drawer.py` |
| **CSV export** | `verdict.to_csv(all_verdicts())` | header/verdict download button |

---

## 7. BUILD DAG (parallel groups)

**GROUP 0 — Scaffolding** (no deps; do first, serially quick)
- `requirements.txt`, `agent/__init__.py`, `agent/config.py`, `agent/types.py`, `context/README.md`, `.github/workflows/grounding.yml`

**GROUP 1 — Data foundation** (deps: G0) — *the R prep is the critical-path long pole; start immediately*
- `scripts/prep_data.R` (long: reads 328M seu.Rds; produces all tidy `data/` files + manifest)
- `scripts/precompute_density.py` (long: streams 1.4G transcripts; needs `DEMO_MARKERS` locked — G0)
- Run both once; verify `data/PREP_MANIFEST.json` before dependents build against real files.

**GROUP 2 — Pure loaders + engine** (deps: G0 types/config; G1 tidy files for integration test) — **fully parallel internally**
- `agent/data.py` (pure loader)
- `agent/verdict.py` (needs `data.py` interface — build against the contract, integration-test after G1)
- `agent/memory.py` (needs types + a `literature_search` callable it accepts as injection; testable with stub)

**GROUP 3 — MCP + tools** (deps: G0, G2 `data.py`) — parallel internally
- `agent/mcp_client.py` (standalone; smoke-test against live server)
- `agent/tools.py` (wraps `data.py` + `memory.py` + `mcp_client.py`)
- `agent/grounding_check.py` (wraps `data.py`; `SourceIndex`/`Extractor`/`LiteratureVerifier`/`GroundingChecker`)

**GROUP 4 — Agent loop + fallback** (deps: G3 all)
- `agent/loop.py` (uses tools + grounding_check finalize + fallback)
- `agent/fallback.py`
- `scripts/build_fallbacks.py` (runs the real loop offline → precomp jsons; needs loop working)

**GROUP 5 — UI** (deps: G2 verdict/data, G4 loop) — **highly parallel** (each pane independent)
- `ui/state.py`, `ui/theme.py`, `ui/format.py` (parallel, no cross-dep)
- `ui/data_access.py` (deps: data.py, verdict.py)
- `ui/cluster_rail.py`, `ui/verdict_header.py`, `ui/evidence_table.py`, `ui/spatial_stage.py`, `ui/conversation.py`, `ui/lab_knowledge.py`, `ui/paper_drawer.py` (parallel; each deps state/format/data_access)
- `app.py` (deps: all ui/*; build last)

**GROUP 6 — Tests + docs** (deps: everything they cover; write test contracts in parallel with G2–G5, wire fixtures after)
- `tests/conftest.py`, `tests/fixtures/*`, `tests/data/calibration_set.yaml` (can start after G2)
- `test_data_loader.py`, `test_verdict.py`, `test_memory.py` (after G2)
- `test_extract.py`, `test_grounding_*.py`, `test_panel_absence.py`, `test_calibration.py` (after G3 grounding_check)
- `test_live_pubmed.py` (after G3 mcp_client)
- `README.md` (calibration table from G6 yaml; after verdict values settle)

Critical path: **G0 → G1 (R prep, hours of wall-clock) → G2 verdict → G3 grounding → G4 loop → G5 app**. G1 must kick off first because it is the slowest and everything reads its outputs. G6 test contracts can be authored in parallel against the frozen dataclasses from G0.

---

## 8. TOP RISKS + OPEN DECISIONS

### Risks + mitigations
1. **R prep is the long pole** (328M seu.Rds + 1.4G transcripts single-pass). *Mitigate:* start G1 first; `fread(select=4 cols)` + immediate `DEMO_MARKERS` filter on transcripts; one-time cost cached to small parquets; runtime never touches raw files. Fail loudly at app start if a tidy file is missing (name the prep step), never fake a view.
2. **Prose extraction is lossy** (fabricated number could escape the stat-binding window). *Mitigate:* the agent emits the `GroundingSidecar` (mandatory in `finalize()`); checker prefers it, prose pass is secondary. This collapses false-negatives to near-zero.
3. **MCP cold-start / down during demo.** *Mitigate:* warm singleton + probe at launch; frozen real-citation cache (`citations.json`) so opening interps always show real clickable PMIDs; never fall back to remembered PMIDs.
4. **`cell_id` reconciliation across 3 conventions** (`1`, `"1_1"`, `"_1_1"`). *Mitigate:* R prep asserts every derived `cell_id` is positive int with exact `_1` suffix; fail loudly. All tidy files join on `cell_id: int`.
5. **Streamlit + persistent asyncio thread under `@st.cache_resource` reruns** (common footgun). *Mitigate:* smoke-test the background-loop singleton survives reruns before relying on the warm session; `@st.fragment` for spatial stage is optional — correctness never depends on it.
6. **Fallback staleness** if data changes but `fallbacks.json` not regenerated. *Mitigate:* CI re-runs grounding suite over fallbacks; regenerate on any data/skill change.
7. **`st.dataframe` cannot return per-row clicks** for pin. *Mitigate:* render evidence rows as `st.columns` + per-row pin `st.button`. Hover-preview of spatial panels is not server-side possible → ship **pin-only** drive + Plotly-native tooltips for the number (honest divergence from wireframe).

### Open decisions needing a human call
1. **`cell_type_short` values** — clusters.Rds `anno` gives one label (Stromal, T_Cells…). Confirm the short form convention (`Str_Fib`, `Tum_…`) and whether it lives as a fixed column in `cluster_key.json` (recommended: yes, author it once, grounded). **User: provide the 9 short names or approve `short == anno`.**
2. **`markers_full` p_value export** — do we export `full_result` with `p_value`? If NOT, the agent must not state p-values (grounding fails closed on any p-value claim). Recommend: export it, keep p_value corroboration-only (never a band driver). **User/data-owner: confirm export.**
3. **`firm own_validation` note authority** — may a firm lab note clear a `verify=TRUE` flag? Spec leans **NO** (verify stays; note shown alongside with tension). **User: confirm the biologist-decides boundary — note annotates, never silently flips confidence.**
4. **Canonical-marker source** — canonical lists derived from the panel `Annotation` column (grounded) + a curated off-panel list per type. Who owns `offpanel_canonical` per cell type? Recommend: author once in `cluster_key.json` (validated at load: each off-panel gene asserted truly absent). **User: approve the c2 spine {COL1A1,COL1A2,DCN,VIM,FAP} and equivalents for other types, or restrict off-panel notes to c2 only for P0.**
5. **Which single demo step makes the live (non-cached) PubMed call** for the P0 "one live literature call" item. Recommend: the override→cross-check step, frozen cache as net. **User: confirm this is the scripted live moment.**