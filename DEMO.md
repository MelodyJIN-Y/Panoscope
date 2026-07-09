# Panoscope — demo walkthrough

A ~3-minute guided run. Every beat below was verified end-to-end in the live app.
Nothing is staged: the numbers come from jazzPanda's precomputed output, and every
citation is fetched live from PubMed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

Live citations need the PubMed MCP: `.mcp.json` is configured; set `NCBI_API_KEY`
and `NCBI_ADMIN_EMAIL` in `.env` (see `.env.example`). If the connector is slow or
absent, every agent call has a **deterministic grounded fallback** — the demo never
stalls and never fabricates.

## The flow

**[0:00] Cold open — the trap.** Ask a plain LLM "is this cluster fibroblasts?" with
cluster 2's present markers. It hedges, because the canonical fibroblast genes
(COL1A1, VIM) look *absent* — but it has no idea they were never on the panel. Hold
on that.

**[0:15] Cut to the tool.** Select **cluster 2 (Stromal)**. Before any question, the
agent's interpretation is already posted: **Stromal, Very High**, driven by
**LUM (glm_coef 18.00, pearson 0.91)** and **POSTN (15.80, 0.80)** — each number a
jazzPanda source chip — plus a live literature citation (**PMID:39147169**). Click the
PMID to open the paper.

**[0:35] The catch, with receipts.** The evidence table has a
**"CANONICAL MARKERS NOT ON THE PANEL"** section: COL1A1, COL1A2, DCN, VIM, FAP show
**"— not measured / ⊘ expected, absent"**, with the line *"Their absence is not
evidence against the call — a missing off-panel gene tells us nothing about the cell
type."* This is the mistake the plain LLM made, caught mechanically.

**[1:15] Calibration — it doesn't rubber-stamp.** Click **cluster 1 (Tumor)**: a clean
**Very High** call (ERBB2 glm_coef 21.44), it commits. Then **cluster 9 (Mast)**:
**Medium confidence, `verify` flagged (⚑)** — only 167 of 158,379 cells, CPA3 at low
spatial specificity. It says *re-check this*. Not crying wolf, not rubber-stamping.

**[1:45] Holistic review.** Click **"Review all clusters"**. It re-reads the whole set:
all breast-TME compartments present, proportions plausible (Tumor largest at 62,755
cells; Mast rarest at 167), no redundancy — and surfaces **one refinement to consider:
c8 Dendritic → Plasmacytoid DC (pDC)**, because its markers LILRA4/TCL1A/SPIB are
pDC-specific. Tagged **"subtype — you decide"**; the jazzPanda numbers don't change.

**[2:15] Memory as reconciliation.** On cluster 2, click **Override / Capture a note**:
*"In our breast TME, this stromal signature is established."* Pick scope (this cluster),
basis (a paper), status (firm). **Save.** The agent keeps your call **and** cross-checks
the literature, reporting agreement/dissent with real citations — or, when the
literature is thin, saying so plainly instead of inventing one. The note appears in
**"What this tool knows about your lab"** with its scope, basis, attribution, date, and
any tension, and is cited back on the next answer. Notes are versioned files under
`context/` — the lab owns them.

**[2:35] Spatial evidence.** Click a marker's pin (e.g. **LUM**). One pinned marker drives
three linked views — **Cell map** (segmented cells at tissue coordinates), **UMAP**
(expression space), **Density** (raw transcripts hex-binned before cell calling,
area-normalized, 25/50/100 µm). The bin size and view toggle change the picture, **never
a value**.

**[2:50] Close.** The markers come straight from jazzPanda, and the grounding test suite
(green in CI) fails if the agent ever invents a number or a citation. Literature
reasoning is labeled a direction, not a fact. The `jazzpanda-markers` skill installs
standalone. Self-hosted and private — which matters for unpublished data.

## The guarantee

Every marker, number, and citation traces to source (jazzPanda output, the panel list,
or a stored lab note). The grounding gate rejects any answer that states an ungrounded
number, uses a lab note without citing it, or produces a citation that does not resolve.
See the calibration table in the [README](README.md) and `tests/` (133 tests, green in CI).
