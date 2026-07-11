"""Summary page — the two-pane report editor (the final, integrated step).

Both workflows land here. The page reads like a manuscript you finalise:

* a **contents rail** on the left — ``Overall`` (the whole-dataset map + the
  cross-cluster check), then every cluster, then ``Caveats`` and ``My notes``;
* a **right pane** that shows exactly one section at a time: the Overall page is
  the merged overview table + the editable global check; a cluster page is that
  cluster's editable identity+programs write-up; the dataset pages are their own
  editors / note cards.

Grounding is preserved end to end — nothing here computes a value. The overview is
a straight projection of :func:`ui.data_access.all_verdicts` +
:func:`~ui.data_access.all_enrichments`; each draft is
:func:`ui.report.default_cluster_summary` / ``global_check_text`` / ``caveats_text``
(live-cited, PMID-carrying); the exports are byte-for-byte what you edit.

"Auto-seed, edits win": each editable region seeds once from the latest draft and
then keeps your edits (held in plain ``wsval_*`` session keys, so switching sections
never drops one). One "refresh all" re-drafts every region from the freshest calls,
programs, and my notes.

Streamlit is imported lazily inside the render function so ``import ui.summary``
works with no server running.
"""

from __future__ import annotations

import datetime
import html
import math

from agent.types import ClusterVerdict

from ui import data_access as da
from ui import format as fmt
from ui import report
from ui import state

# --------------------------------------------------------------------------- #
# Copy + export metadata. Constants so prose never drifts.
# --------------------------------------------------------------------------- #
_TITLE = "Interpretation summary"
_SUB = (
    "Both workflows in one report — the marker-gene calls and the gene-set programs, "
    "drafted per cluster for you to edit and export. Every number is jazzPanda's, every "
    "biology claim is live-cited, nothing here recomputes a value."
)

# Two exports of DIFFERENT kinds: the editable narrative (Word), and the
# machine-readable verdict table (CSV, R-importable — the structured call, NOT the
# free-text edits).
_DOCX_LABEL = "⬇  Report (Word)"
_DOCX_NAME = "panoscope_interpretation.docx"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_CSV_LABEL = "⬇  Annotations (CSV)"
_CSV_NAME = "panoscope_annotations.csv"
_CSV_MIME = "text/csv"

# Editor auto-height: 13.5px font * 1.65 line-height ≈ 22px/line; the wide 0.76 pane
# fits ~150 chars/line, so we estimate at 110 (a safe under-count → a little headroom,
# never a clip). Sizes each box to its content so nothing hides below an inner scroll.
_WRAP_CHARS = 110
_PX_PER_LINE = 22
_ED_PAD = 28
_ED_MAX_PX = 1600

# Rail sections. "overall" hosts the overview + the (editable) cross-cluster check;
# then one entry per cluster; then the two dataset-level editors/cards.
_SEC_OVERALL = "overall"
_SEC_CAVEATS = "caveats"
_SEC_NOTES = "labnotes"
_DATASET_SECTIONS = ((_SEC_CAVEATS, "Caveats"), (_SEC_NOTES, "My notes"))
_ED_GLOBAL = "global"   # editor name for the cross-cluster check (lives on Overall)
_K_ACTIVE = "sum_active_section"  # which rail item is focused

# Overview table columns (merged marker + enrichment). (label, width%)
_OVERVIEW_COLS = (
    ("Cluster", 9), ("Cell type", 16), ("Conf.", 11),
    ("Key markers", 28), ("Enriched programs", 36),
)

# --------------------------------------------------------------------------- #
# Styling. Reuses the theme tokens (var(--sans) etc.) + the global .cf-* pills.
# --------------------------------------------------------------------------- #
_SUMMARY_CSS = """
.pano-sum-title { font-family: var(--sans); font-size: 23px; font-weight: 700;
                  letter-spacing: -.02em; color: var(--ink); margin: 2px 0 5px; }
.pano-sum-sub { font-size: 12.5px; color: var(--muted); line-height: 1.5;
                max-width: 74ch; margin: 0 0 10px; }
.pano-sum-meta { font-family: var(--mono); font-size: 11px; color: var(--faint);
                 display: flex; gap: 9px; align-items: center; margin: 0; }
.pano-sum-meta .n { color: var(--ink); font-weight: 600; }
.pano-sum-meta .sep { color: var(--hair); }
.pano-sum-saved { color: var(--faint); }
.pano-sum-saved .err { color: var(--absent); }

/* Export cluster: two downloads (Word report, CSV table). */
.st-key-dl_docx button, .st-key-dl_csv button {
  border-radius: 9px !important; font-size: 12px !important; white-space: nowrap;
  min-height: 0 !important; padding: 7px 10px !important; }
/* Per-region Save button — right-aligned directly under its editor. */
div[class*="st-key-savrow_"] { display: flex; justify-content: flex-end; margin: 8px 0 2px; }
div[class*="st-key-savrow_"] button { background: var(--accent) !important; color: #fff !important;
  border: 0 !important; box-shadow: none !important; min-height: 0 !important; padding: 6px 22px !important;
  border-radius: 8px !important; font-size: 12px !important; font-weight: 600 !important; }
div[class*="st-key-savrow_"] button:hover { filter: brightness(1.06); }
/* The re-draft control — quiet, left-aligned, clearly separate from the exports. */
.st-key-pano_redraft { display: flex; justify-content: flex-start; margin: 8px 0 0; }
.st-key-pano_redraft button { background: transparent !important; border: 1px solid var(--hair) !important;
  color: var(--faint) !important; box-shadow: none !important; min-height: 0 !important;
  padding: 4px 11px !important; border-radius: 7px !important; font-family: var(--mono) !important;
  font-size: 10.5px !important; }
.st-key-pano_redraft button:hover { color: var(--absent) !important; border-color: var(--absent) !important;
  background: var(--absent-bg) !important; }

/* Left contents rail — reuses the app's cluster-rail look (theme.py owns the
   button de-chrome + the coloured per-cluster dots via the rail_cN keys). Here we
   only add the sticky card, the group labels, and hide the dot on non-cluster items. */
.st-key-pano_rail { position: sticky; top: 68px; align-self: start;
  border: 1px solid var(--hair); border-radius: 14px; background: var(--paper);
  padding: 7px 10px 14px; }
/* Group labels are wrapped in keyed containers FORCED to a real row (Streamlit
   otherwise collapses a custom-markdown element to ~0 height in the flex rail). The
   label sits at the TOP of its row (flex-start) with clear empty space BELOW it, so
   the next item's selected box can never rise up and cover the label. */
.st-key-pano_rail div[class*="st-key-raillbl_"] { min-height: 30px !important; margin: 8px 0 0 !important;
  display: flex !important; flex-direction: column !important; justify-content: flex-start !important;
  overflow: hidden !important; }
.pano-rail-lbl { font-family: var(--mono); font-size: 9px; text-transform: uppercase;
  letter-spacing: .14em; color: var(--faint); font-weight: 700; padding: 2px 9px 0; margin: 0; }
/* Non-cluster rows keep the dot's SLOT (transparent) so every label lines up with
   the dotted cluster rows below — no ragged left edge. */
.st-key-nav_overall button::before, .st-key-nav_caveats button::before,
.st-key-nav_labnotes button::before { background: transparent !important; }

/* Per-cluster reference (marker + enrichment stats, above the editor). */
.pano-ref-cap { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .09em; color: var(--ink); font-weight: 700; margin: 18px 0 7px;
  display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
.pano-ref-cap .meta { color: var(--faint); font-weight: 500; letter-spacing: .03em; font-size: 9.5px; }
.pano-num { font-family: var(--mono); font-size: 11.5px; color: var(--ink); }
.pano-num-dim { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.pano-canon { color: var(--accent); font-weight: 700; }
.pano-role { font-size: 11px; color: var(--muted); text-transform: capitalize; }
.pano-tier { font-family: var(--mono); font-size: 8.5px; text-transform: uppercase; letter-spacing: .05em;
  padding: 2px 6px; border-radius: 5px; font-weight: 700; margin-left: 7px; vertical-align: middle; }
.pano-tier.enriched { background: var(--accent-soft); color: var(--accent); }
.pano-tier.suggestive { background: #FBF3E3; color: #9A6B12; }
/* A note anchored beneath its driver / program row. */
.pano-sum-table tr.pano-anchor-row td { padding-top: 0 !important; padding-bottom: 10px !important;
  border-bottom: 1px solid var(--hair2); }
.pano-anchornote { font-size: 11.5px; line-height: 1.5; color: var(--muted);
  background: #FAFBFC; border: 1px solid var(--hair); border-radius: 8px;
  padding: 6px 10px; }
.pano-notecite { font-family: var(--mono); font-size: 10px; color: var(--accent); }
/* Holistic-review refinements (capture as an override note). */
.pano-refine { font-size: 12.5px; color: var(--ink); margin: 12px 0 4px; }
.pano-refine .cid { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-right: 6px; }
.pano-refine .rat { font-size: 11.5px; color: var(--muted); line-height: 1.5; margin-top: 3px; max-width: 84ch; }
div[class*="st-key-refbtn_"] { margin: 2px 0 10px; }
div[class*="st-key-refbtn_"] button { background: transparent !important; border: 1px solid var(--accent) !important;
  color: var(--accent) !important; box-shadow: none !important; min-height: 0 !important; padding: 5px 14px !important;
  border-radius: 8px !important; font-size: 12px !important; }
div[class*="st-key-refbtn_"] button:hover { background: var(--accent-soft) !important; }

/* Overview table (marker + programs merged). */
.pano-ov-cap { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); font-weight: 600; margin: 2px 0 8px; }
.pano-sum-tablewrap { background: var(--paper); border: 1px solid var(--hair);
                      border-radius: 14px; overflow: hidden; }
.pano-sum-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-family: var(--sans); }
.pano-sum-table thead th { text-align: left; font-family: var(--mono); font-size: 9.5px;
  text-transform: uppercase; letter-spacing: .06em; color: var(--faint); font-weight: 500;
  padding: 11px 13px; border-bottom: 1px solid var(--hair); background: #FAFBFB; }
.pano-sum-table tbody td { padding: 12px 13px; border-bottom: 1px solid var(--hair2); vertical-align: top; }
.pano-sum-table tbody tr:last-child td { border-bottom: 0; }
.pano-sum-table tbody tr:hover td { background: #FAFCFC; }
.pano-sum-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                margin-right: 8px; vertical-align: middle; }
.pano-sum-cid { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink); vertical-align: middle; }
.pano-sum-ct { font-size: 13.5px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.pano-sum-km { font-family: var(--mono); font-size: 11px; color: var(--muted); line-height: 1.6; word-break: break-word; }
.pano-sum-desc { font-size: 12px; color: var(--muted); line-height: 1.5; }
.pano-enr-le { font-family: var(--mono); font-size: 10px; color: var(--faint); line-height: 1.55;
  word-break: break-word; display: block; margin-top: 3px; }
.pano-sum-dash { color: var(--faint); }
.pano-sum-flag { font-family: var(--mono); font-size: 9.5px; font-weight: 600; color: var(--absent);
  background: var(--absent-bg); padding: 2px 7px; border-radius: 6px; white-space: nowrap; }
.pano-sum-table .cf { font-size: 10px; padding: 2px 8px; }
.pano-ovflag { color: var(--absent); font-size: 11px; margin-left: 6px; }
.pano-ov-lab { font-family: var(--mono); font-size: 8.5px; text-transform: uppercase; letter-spacing: .05em;
  font-weight: 700; color: var(--accent); background: var(--accent-soft); padding: 1px 5px;
  border-radius: 4px; margin-left: 6px; vertical-align: middle; }
.pano-ov-was { font-family: var(--mono); font-size: 9.5px; color: var(--faint); margin-top: 3px; }

/* Focused editor. */
.pano-ed-hr { border: 0; border-top: 1px solid var(--hair); margin: 24px 0 4px; }
.pano-ed-eyebrow { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .11em; color: var(--accent); font-weight: 600; margin: 4px 0 10px; }
.pano-ws-head { display: flex; align-items: baseline; gap: 10px; padding: 0 0 10px; flex-wrap: wrap; }
.pano-ws-head .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.pano-ws-head .cid { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.pano-ws-head .ct { font-size: 18px; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.pano-ws-head .cf { font-size: 10px; padding: 2px 8px; }
.pano-ed-hint { font-size: 11.5px; color: var(--faint); margin: 6px 0 0; }
div[class*="st-key-wsw_"] textarea { font-family: var(--sans) !important; font-size: 13.5px !important;
  line-height: 1.65 !important; border-radius: 12px !important; border: 1px solid var(--hair) !important;
  background: #FCFCFD !important; color: var(--ink) !important; }
div[class*="st-key-wsw_"] textarea:focus { border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important; background: var(--paper) !important; }

/* Note cards (read-only, inside the My notes section). */
.pano-lk-card { border: 1px solid var(--hair); border-radius: 11px; padding: 11px 13px;
  margin: 8px 0; background: var(--paper); }
.pano-lk-claim { font-size: 13px; color: var(--ink); line-height: 1.5; }
.pano-lk-meta { font-family: var(--mono); font-size: 10px; color: var(--faint); margin-top: 7px; }
.pano-lk-meta .sco { background: var(--accent-soft); color: var(--accent); padding: 2px 7px; border-radius: 5px; }
.pano-lk-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); border: 1px dashed var(--hair);
  border-radius: 10px; padding: 16px; text-align: center; }

/* --- Clinical review table (the Overall pane) --------------------------- *
 * Swiss/clinical: one contained surface, tight columns, strong hierarchy,
 * restrained colour. Depth via a hairline + soft shadow, never boxes-in-boxes. */
/* Section sub-nav (Overall · Caveats · My notes) — the clusters live in the
   table, so the nav is only the dataset-level sections. */
.st-key-pano_subnav { max-width: 360px; margin: 0 0 16px; background: var(--paper);
  border: 1px solid var(--hair); border-radius: 10px; padding: 3px; }
.st-key-pano_subnav [data-testid="stHorizontalBlock"] { gap: 3px; }
.st-key-pano_subnav div[data-testid="stButton"] > button { background: transparent !important;
  border: 0 !important; box-shadow: none !important; border-radius: 7px !important; min-height: 0 !important;
  padding: 6px 10px !important; font-family: var(--sans) !important; font-size: 12.5px !important;
  font-weight: 500 !important; color: var(--muted) !important; }
.st-key-pano_subnav div[data-testid="stButton"] > button:hover { background: var(--hair2) !important; color: var(--ink) !important; }
.st-key-pano_subnav button[data-testid="stBaseButton-primary"] { background: var(--accent-soft) !important;
  color: var(--accent) !important; font-weight: 600 !important; }
/* One clean stat strip (replaces the two duplicate status lines). */
.pano-stat { display: flex; align-items: baseline; gap: 22px; margin: 0 0 13px; max-width: 1060px; }
.pano-stat .item { font-family: var(--sans); font-size: 14px; font-weight: 700; color: var(--ink);
  letter-spacing: -.01em; }
.pano-stat .item .lab { font-size: 11px; font-weight: 500; color: var(--faint); margin-left: 5px;
  font-family: var(--mono); text-transform: uppercase; letter-spacing: .05em; }
.pano-stat .item.warn { color: var(--absent); }
.pano-stat .item.dim { color: var(--faint); font-weight: 500; }
/* Reconciliation: one calm line, neutral surface, no coloured left border. */
.pano-recon-line { font-size: 11.5px; line-height: 1.55; color: var(--muted); background: var(--paper);
  border: 1px solid var(--hair); border-radius: 10px; padding: 9px 13px; margin: 0 0 14px; max-width: 1060px; }
.pano-recon-line .ic { color: var(--faint); font-weight: 700; margin-right: 6px; }
.pano-recon-line .cid { font-family: var(--mono); font-size: 10.5px; color: var(--faint); }
.pano-recon-line b { color: var(--ink); font-weight: 600; }
/* The table's contained surface: a white card with a soft shadow, width-capped so
   the columns stay tight instead of sprawling across the viewport. */
.st-key-pano_tblcard { background: var(--paper); border: 1px solid var(--hair); border-radius: 14px;
  box-shadow: 0 1px 2px rgba(20,27,32,.05), 0 6px 22px rgba(20,27,32,.035);
  padding: 4px 20px 8px; max-width: 1060px; }
/* Column header. */
.pano-th { font-family: var(--mono); font-size: 9px; text-transform: uppercase; letter-spacing: .1em;
  color: var(--faint); font-weight: 600; padding: 6px 0 8px; }
.st-key-pano_thead { border-bottom: 1px solid var(--hair); }
/* Rows: hairline separators, a real row rhythm, subtle state tints (never a side border). */
div[class*="st-key-trow_"] { border-bottom: 1px solid var(--hair2); padding: 2px 0; }
div[class*="st-key-trowneed_"] { background: #FCFAF3; }               /* needs action — a warm wash */
div[class*="st-key-trowdone_"] .pano-tc, div[class*="st-key-trowdone_"] .pano-tcell { opacity: .42; }
div[class*="st-key-trow_"]:hover, div[class*="st-key-trowneed_"]:hover { background: var(--accent-soft); }
/* Cells. */
.pano-tc { display: flex; align-items: center; gap: 9px; min-height: 34px; }
.pano-tc .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }
.pano-tc .ct { font-size: 13.5px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.pano-tc .yours { font-family: var(--mono); font-size: 8px; text-transform: uppercase; letter-spacing: .04em;
  font-weight: 700; color: var(--accent); background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; }
.pano-tc .flag { color: var(--absent); font-size: 11px; }
.pano-tcell { font-size: 12px; color: var(--muted); display: flex; align-items: center; min-height: 34px; }
.pano-tcell.mono { font-family: var(--mono); font-size: 11.5px; }
.pano-tcell .g { color: var(--ink); font-weight: 600; margin-right: 7px; }
.pano-tcell .r { color: var(--faint); margin-left: 3px; }
.pano-tsub { font-size: 11px; color: var(--muted); line-height: 1.5; margin: -4px 0 6px 0; padding-left: 1px; }
.pano-tsub .warn { color: var(--absent); font-weight: 600; }
.pano-tsub b { color: var(--ink); font-weight: 600; }
/* Cluster id — a borderless mono link that drills into the evidence. */
div[class*="st-key-tdrill_"] { display: flex; align-items: center; }
div[class*="st-key-tdrill_"] button { min-height: 0 !important; padding: 4px 0 !important; border: 0 !important;
  background: transparent !important; box-shadow: none !important; color: var(--muted) !important;
  font-family: var(--mono) !important; font-size: 12px !important; font-weight: 600 !important; text-align: left !important; }
div[class*="st-key-tdrill_"] button:hover { color: var(--accent) !important; }
/* Sign-off sits at the right edge of the row. */
div[class*="st-key-tact_"] { display: flex; justify-content: flex-end; align-items: center; }
/* Checkbox-style toggle: a bare glyph (☐ faint / ✓ teal), matching the app's dot language. */
div[class*="st-key-tact_"] button { min-height: 0 !important; padding: 2px 7px !important;
  border: 0 !important; background: transparent !important; box-shadow: none !important;
  color: var(--faint) !important; font-size: 17px !important; line-height: 1 !important; border-radius: 6px !important; }
div[class*="st-key-tact_"] button:hover { color: var(--accent) !important; background: var(--accent-soft) !important; }
div[class*="st-key-tact_done_"] button { color: var(--accent) !important; }
/* Confirm (flagged) — a small amber outline pill, never a stretched box. */
div[class*="st-key-tact_confirm_"] button { border: 1px solid var(--absent) !important; color: var(--absent) !important;
  background: var(--absent-bg) !important; font-family: var(--mono) !important; font-size: 10px !important;
  font-weight: 700 !important; padding: 5px 11px !important; text-transform: uppercase; letter-spacing: .05em; }
div[class*="st-key-tact_confirm_"] button:hover { background: var(--absent) !important; color: #fff !important; }
/* Drill-down back link. */
div[class*="st-key-backlink"] button { min-height: 0 !important; padding: 3px 0 !important; border: 0 !important;
  background: transparent !important; box-shadow: none !important; color: var(--faint) !important;
  font-family: var(--mono) !important; font-size: 11px !important; font-weight: 600 !important; }
div[class*="st-key-backlink"] button:hover { color: var(--accent) !important; }
/* Kept for the tested evidence-strength helper (drill-down / reuse). */
.pano-str { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-top: 4px; line-height: 1.5; }
.pano-str-gene { color: var(--ink); font-weight: 600; }
.pano-str-num { color: var(--muted); }
.pano-str-n { color: var(--faint); }
.pano-str-cav { color: var(--muted); }
/* Contested sign-off confirm card (basis toggle + confirm). Per-cluster key
   (sofcard_<cluster>) so several can render at once without a duplicate-key crash. */
div[class*="st-key-sofcard_"] { border: 1px solid var(--hair); border-radius: 11px;
  background: #FAFBFC; padding: 12px 14px; margin: 4px 0 12px; }
.sof-eyebrow { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); font-weight: 700; margin-bottom: 6px; }
.sof-claim { font-size: 12.5px; line-height: 1.5; color: var(--ink); font-weight: 500; margin-bottom: 8px; }
/* Cross-cluster write-up, demoted below the table (still exports). */
.pano-export-note { font-size: 11.5px; color: var(--faint); margin: 2px 0 8px; line-height: 1.5; }
"""


# --------------------------------------------------------------------------- #
# Small pure HTML cell builders (grounded projections — no value invented).
# --------------------------------------------------------------------------- #
def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


def _dot_id(cluster: str) -> str:
    color = fmt.cluster_color(cluster)
    return (f'<span class="pano-sum-dot" style="background:{color}"></span>'
            f'<span class="pano-sum-cid">{html.escape(cluster)}</span>')


def _conf_pill(confidence: str) -> str:
    css, _ = fmt.confidence_chip(confidence)
    return f'<span class="cf {css}">{html.escape(confidence)}</span>'


def _markers_cell(v: ClusterVerdict) -> str:
    if not v.key_markers:
        return '<span class="pano-sum-dash">—</span>'
    return f'<span class="pano-sum-km">{" · ".join(html.escape(str(g)) for g in v.key_markers)}</span>'


def _programs_cell(ce) -> str:
    """Top enriched programs + their leading-edge genes, compact. Dash if none."""
    if ce is None or not ce.enriched:
        return '<span class="pano-sum-dash">no program clears the gate</span>'
    progs = " · ".join(_short(p.gene_set) for p in ce.enriched[:4])
    le: list[str] = []
    for p in ce.enriched[:3]:
        for g in p.leading_edge:
            if g not in le:
                le.append(g)
    le_html = f'<span class="pano-enr-le">{" · ".join(html.escape(g) for g in le[:9])}</span>' if le else ""
    return f'<span class="pano-sum-desc">{html.escape(progs)}</span>{le_html}'


def _overview_table_html(verdicts: list[ClusterVerdict], enr_map: dict, overrides: dict = None) -> str:
    """The one scannable map: per cluster, the marker call and the enriched programs.
    A confirmed cell-type override shows the new call with a 'yours' tag and the computed
    call it replaced (tension visible), never a silent swap."""
    overrides = overrides or {}
    cols = "".join(f'<col style="width:{w}%">' for _, w in _OVERVIEW_COLS)
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in _OVERVIEW_COLS)
    rows = []
    for v in verdicts:
        ce = enr_map.get(v.cluster)
        flag = ' <span class="pano-ovflag" title="flagged for re-check">⚑</span>' if v.verify else ""
        ct = f'<span class="pano-sum-ct">{html.escape(v.cell_type)}</span>'
        ov = overrides.get(v.cluster)
        if ov:
            ct += (f' <span class="pano-ov-lab" title="your override; computed: '
                   f'{html.escape(ov["computed_call"])}">yours</span>')
            was = f'was {html.escape(ov["computed_call"])}'
            if ov["dissent"]:
                was += f' · {ov["dissent"]} lit. dissent'
            ct += f'<div class="pano-ov-was">{was}</div>'
        rows.append(
            "<tr>"
            f"<td>{_dot_id(v.cluster)}</td>"
            f"<td>{ct}{flag}</td>"
            f"<td>{_conf_pill(v.confidence)}</td>"
            f"<td>{_markers_cell(v)}</td>"
            f"<td>{_programs_cell(ce)}</td>"
            "</tr>"
        )
    return (
        '<div class="pano-sum-tablewrap"><table class="pano-sum-table">'
        f"<colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# Sign-off board — the Overall pane's hero. A review cockpit: the whole call-set
# triaged into what needs the biologist vs what is signed off, each row grounded
# in the verdict evidence and acted on inline. Nothing here computes a value.
# --------------------------------------------------------------------------- #
_BUCKET_NEEDS = "needs"
_BUCKET_REVIEW = "review"
_BUCKET_SIGNED = "signed"
_BUCKET_LABELS = {
    _BUCKET_NEEDS: "Needs you",
    _BUCKET_REVIEW: "To review",
    _BUCKET_SIGNED: "Signed off",
}
_SIGNOFF_SLOT = "signoff::{}"  # pending-draft slot for a contested sign-off's note


def _rerun(st) -> None:
    """Rerun the script (new + legacy Streamlit API)."""
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def _refinements_by_cluster() -> dict:
    """The holistic pass's proposed refinements, keyed by cluster (grounded, may be
    empty). A refinement is a within-lineage subtype sharpening to consider."""
    try:
        hol = da.holistic()
        refs = list(getattr(hol, "refinements", ()) or []) if hol else []
        return {r.cluster: r for r in refs}
    except Exception:  # noqa: BLE001 - no holistic slice -> no refinements
        return {}


def _evidence_strength_html(v: ClusterVerdict) -> str:
    """A compact, grounded read of how strong a call's evidence is: the top driver
    (gene, glm coef, pearson), the assigned-marker count, and the strongest caveat if
    any. A projection of ``v.evidence`` — never a computed or invented value."""
    ev = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)
    if not ev:
        return '<div class="pano-str">no marker evidence recorded</div>'
    top = ev[0]
    caveat = ""
    for e in ev[:6]:
        if e.caveats:
            caveat = f' · <span class="pano-str-cav">{html.escape(str(e.caveats[0]))}</span>'
            break
    return (
        '<div class="pano-str">'
        f'<span class="pano-str-gene">{html.escape(top.gene)}</span> '
        f'<span class="pano-str-num">glm {top.glm_coef:.1f} · r {top.pearson:.2f}</span> · '
        f'<span class="pano-str-n">{len(v.evidence)} genes</span>{caveat}'
        "</div>"
    )


def _flag_reason(v: ClusterVerdict) -> str:
    """The grounded reason a call is flagged for re-check, projected from the verdict —
    NOT a blanket 'thin'. jazzPanda's `verify` has distinct causes: a fragile cluster
    (few assigned genes), or a low-specificity demotion recorded as a driver caveat
    (localizes better with another cluster / spatial pattern not unique). We surface the
    real one so the reason line and any saved sign-off note never misstate why."""
    n = len(v.evidence)
    if n <= 2:
        return f"rests on {n} marker{'s' if n != 1 else ''}"
    ev = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)
    if ev and ev[0].caveats:
        return str(ev[0].caveats[0])
    return "flagged for re-check"


def _triage_bucket(v: ClusterVerdict, signed, refinements, overrides) -> str:
    """Which pile a call sits in: signed off, needs-you (flagged for re-check / has a
    proposed refinement / an override the literature dissents from), or to-review."""
    if v.cluster in signed:
        return _BUCKET_SIGNED
    if v.verify or v.cluster in refinements:
        return _BUCKET_NEEDS
    ov = overrides.get(v.cluster)
    if ov and ov.get("dissent"):
        return _BUCKET_NEEDS
    return _BUCKET_REVIEW


def _evidence_compact(v: ClusterVerdict) -> str:
    """The 'key evidence' cell: the top driver's gene + glm·pearson, grounded and compact."""
    ev = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)
    if not ev:
        return '<span class="pano-tcell">—</span>'
    top = ev[0]
    return (f'<span class="pano-tcell mono"><span class="g">{html.escape(top.gene)}</span> '
            f'{top.glm_coef:.1f}<span class="r"> · r{top.pearson:.2f}</span></span>')


def _stat_strip_html(n_signed: int, n_total: int, n_needs: int, n_panel: int, status: str) -> str:
    """One clean stat strip above the table — the single at-a-glance status (grounded
    counts), replacing the two duplicate status lines the redesign removed."""
    need = (f'<span class="item warn">{n_needs}<span class="lab">need you</span></span>'
            if n_needs else '<span class="item dim">none flagged</span>')
    status_html = f'<span class="item dim">{status}</span>' if status else ""
    return ('<div class="pano-stat">'
            f'<span class="item">{n_signed}<span class="lab">/ {n_total} signed off</span></span>'
            f'{need}'
            f'<span class="item dim">{n_panel}<span class="lab">gene panel</span></span>'
            f'{status_html}</div>')


def _table_subline_html(v: ClusterVerdict, override, refinement, is_signed: bool) -> str:
    """A grounded reason line beneath a flagged / to-refine / dissenting row (none when the
    row is clean or already signed off)."""
    if is_signed:
        return ""
    bits: list[str] = []
    if refinement is not None:
        bits.append(f'refine &rarr; <b>{html.escape(refinement.to_call)}</b> · '
                    f'{html.escape(refinement.rationale)}')
    if v.verify:
        bits.append(f'<span class="warn">flagged</span> — {html.escape(_flag_reason(v))}')
    if override is not None and override.get("dissent"):
        bits.append(f'override — {override["dissent"]} lit. dissent '
                    f'(was {html.escape(override["computed_call"])})')
    if not bits:
        return ""
    return '<div class="pano-tsub">' + " &nbsp;·&nbsp; ".join(bits) + "</div>"


def _signoff_claim(v: ClusterVerdict, override) -> str:
    """The grounded validation claim a contested sign-off records in My notes."""
    if override is not None and override.get("dissent"):
        return (f"Accepted the {v.cell_type} call for {v.cluster} despite "
                f"{override['dissent']} literature dissent. Reviewed and confirmed.")
    ev = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)
    top = f"top marker {ev[0].gene}" if ev else "its top marker"
    return (f"Confirmed {v.cell_type} for {v.cluster} despite being flagged "
            f"({_flag_reason(v)}; {top}). Reviewed and accepted.")


# --- Reconciliation strip (grounded cross-cluster cues) -------------------- #
def _reconciliation_items(verdicts: list[ClusterVerdict], themes, overrides: dict) -> list:
    """Grounded cross-cluster cues as compact ``(icon, html)`` items for a one-line strip.
    Every cue projects real evidence — spillover caveats, enrichment themes, override
    dissent — nothing invented. Returns [] when there is nothing cross-cluster to weigh."""
    items: list = []
    spill = [
        e.gene
        for v in verdicts
        for e in sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)[:8]
        if e.gene in v.key_markers and any("localizes better" in str(c) for c in e.caveats)
    ]
    if spill:
        genes = ", ".join(html.escape(g) for g in spill[:5])
        items.append(("⇄", f'<b>{len(spill)}</b> marker(s) may localize to another cluster '
                           f'<span class="cid">({genes})</span>'))
    recurring = [
        r for r in (list(getattr(themes, "recurring", ()) or []) if themes else [])
        if getattr(r, "n_clusters", 0) >= 3
    ]
    if recurring:
        names = ", ".join(html.escape(_short(r.gene_set)) for r in recurring[:3])
        items.append(("◇", f'<b>{len(recurring)}</b> program(s) recur across clusters '
                           f'<span class="cid">({names})</span>'))
    for c, o in sorted(overrides.items()):
        if o.get("dissent"):
            items.append(("⚖", f'<span class="cid">{html.escape(c)}</span> '
                               f'{html.escape(o["new_call"])} override — '
                               f'<b>{o["dissent"]}</b> lit. dissent'))
    return items


def _reconciliation_line_html(items: list) -> str:
    """The reconciliation as ONE calm line on a neutral surface (no coloured border), or
    '' when there is nothing to reconcile."""
    if not items:
        return ""
    body = ' &nbsp;·&nbsp; '.join(f'<span class="ic">{ic}</span> {txt}' for ic, txt in items)
    return f'<div class="pano-recon-line">{body}</div>'


# --- Sign-off actions (on_click handlers) ---------------------------------- #
def _lit_search():
    """The literature connector (or None -> an honest thin tension on the note)."""
    try:
        from agent import tools

        return tools._literature_search_fn()
    except Exception:  # noqa: BLE001 - no connector -> thin
        return None


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _sign_off_clean(cluster: str) -> None:
    """on_click: a plain review checkmark for a solid call (no note, nothing contested)."""
    da.mark_signed_off(cluster, note_id=None, at=_now_iso())


def _undo_sign_off(cluster: str) -> None:
    """on_click: reopen a call for review. The validation note (if any) is kept in My
    notes — undo reopens the review, it does not erase the recorded basis."""
    da.clear_signoff(cluster, at=_now_iso())


def _begin_signoff(cluster: str, claim: str) -> None:
    """on_click: draft a validation note for a contested sign-off (a flagged call or an
    override the literature dissents from), reconciled live against the literature, and
    stash it so the confirm card renders. Nothing persists until the biologist confirms."""
    from agent import memory

    try:
        draft = memory.draft_note(
            claim=claim, scope="cluster", basis="convention", cluster=cluster,
            note_type="validation", literature_search=_lit_search(),
        )
        state.set_pending_draft(_SIGNOFF_SLOT.format(cluster), draft)
    except Exception:  # noqa: BLE001 - never crash the board on a bad draft
        pass


def _render_signoff_card(st, cluster: str) -> None:
    """The contested-sign-off confirm card: the drafted validation claim + its live
    literature check + a basis toggle. On confirm it saves the note AND records the
    sign-off (which clears the re-check flag at composition). Cancel drops the draft."""
    import dataclasses

    from ui import conversation as convo

    slot = _SIGNOFF_SLOT.format(cluster)
    draft = state.get_pending_draft(slot)
    if draft is None:
        return
    # Per-cluster container key so two pending sign-off cards never collide, and a
    # draft-nonce on the basis toggle so each fresh draft re-seeds from its proposal.
    nonce = convo._draft_nonce(draft)
    with st.container(key=f"sofcard_{cluster}"):
        st.markdown(
            '<div class="sof-eyebrow">Sign off &amp; clear flag · confirm basis</div>'
            f'<div class="sof-claim">{html.escape(draft.claim)}</div>',
            unsafe_allow_html=True,
        )
        basis = convo._seg(st, "basis", convo._BASIS_OPTS, draft.basis, f"sof_basis_{cluster}_{nonce}")
        st.markdown(convo._draft_tension_html(draft), unsafe_allow_html=True)
        c_ok, c_no = st.columns([0.6, 0.4])
        ok = c_ok.button("Confirm sign-off", key=f"sof_ok_{cluster}", type="primary",
                         use_container_width=True)
        no = c_no.button("Cancel", key=f"sof_no_{cluster}", use_container_width=True)
    if no:
        state.clear_pending_draft(slot)
        _rerun(st)
    elif ok:
        edited = dataclasses.replace(draft, basis=basis, scope="cluster", cluster=cluster,
                                     status="firm")
        try:
            note = da.save_note_draft(edited, trigger="manual_add")
            da.mark_signed_off(cluster, note_id=note.id, at=_now_iso())
        except Exception:  # noqa: BLE001 - surface nothing, never crash the board
            pass
        state.clear_pending_draft(slot)
        _rerun(st)


# --- Row + board rendering -------------------------------------------------- #
# Quiet-table columns: (label, width fraction). Sums to 1.0.
_TBL_COLS = (("Cluster", 0.13), ("Cell type", 0.31), ("Conf.", 0.15),
             ("Key evidence", 0.29), ("", 0.12))


def _accept_refine_button(st, refinement) -> None:
    with st.container(key=f"tact_refine_{refinement.cluster}"):
        st.button(f"Accept {refinement.to_call}", key=f"acc_{refinement.cluster}",
                  on_click=_draft_refinement, args=(refinement,),
                  help="Draft this refinement as a cell-type override; confirm below.")


def _sign_control(st, v: ClusterVerdict, override, signed_rec, contested: bool) -> None:
    """The row's sign-off control (called inside its column): a checkbox-style toggle for a
    clean call, a 'Confirm' for a flagged/contested one, or a checked ✓ (click to reopen)."""
    if signed_rec is not None:
        at = str(signed_rec.get("at", "") or "")
        tip = f"Signed {at[11:16]} — click to reopen" if len(at) >= 16 else "Signed — click to reopen"
        with st.container(key=f"tact_done_{v.cluster}"):
            st.button("✓", key=f"sgn_{v.cluster}", on_click=_undo_sign_off, args=(v.cluster,), help=tip)
    elif contested:
        with st.container(key=f"tact_confirm_{v.cluster}"):
            st.button("Confirm", key=f"sgn_{v.cluster}", on_click=_begin_signoff,
                      args=(v.cluster, _signoff_claim(v, override)),
                      help="Confirm this flagged call — writes a validation note and clears the flag.")
    else:
        with st.container(key=f"tact_{v.cluster}"):
            st.button("☐", key=f"sgn_{v.cluster}", on_click=_sign_off_clean, args=(v.cluster,),
                      help="Sign off — mark reviewed and accepted.")


def _render_table_head(st) -> None:
    with st.container(key="pano_thead"):
        cols = st.columns([w for _, w in _TBL_COLS])
        for col, (label, _w) in zip(cols, _TBL_COLS):
            col.markdown(f'<div class="pano-th">{html.escape(label)}</div>', unsafe_allow_html=True)


def _render_table_row(st, v: ClusterVerdict, override, signed_rec, refinement) -> None:
    """One table row: cluster (drill link) · cell type · confidence · key evidence ·
    sign-off control, with a grounded reason sub-line and any pending confirm card below.
    Signed rows recede; rows that need action carry a subtle warm wash (never a border)."""
    is_signed = signed_rec is not None
    contested = bool(v.verify or (override and override.get("dissent")))
    needs = contested or refinement is not None
    rowkey = (f"trowdone_{v.cluster}" if is_signed
              else f"trowneed_{v.cluster}" if needs else f"trow_{v.cluster}")
    with st.container(key=rowkey):
        cols = st.columns([w for _, w in _TBL_COLS], vertical_alignment="center")
        with cols[0], st.container(key=f"tdrill_{v.cluster}"):
            st.button(v.cluster, key=f"drill_{v.cluster}", on_click=_set_active, args=(v.cluster,),
                      help="See this cluster's evidence + write-up")
        color = fmt.cluster_color(v.cluster)
        yours = ' <span class="yours">yours</span>' if override else ""
        flag = ' <span class="flag" title="flagged for re-check">⚑</span>' if v.verify else ""
        cols[1].markdown(
            f'<div class="pano-tc"><span class="dot" style="background:{color}"></span>'
            f'<span class="ct">{html.escape(v.cell_type.replace("_", " "))}</span>{yours}{flag}</div>',
            unsafe_allow_html=True)
        cols[2].markdown(f'<div class="pano-tcell">{_conf_pill(v.confidence)}</div>',
                         unsafe_allow_html=True)
        cols[3].markdown(f'<div class="pano-tc">{_evidence_compact(v)}</div>', unsafe_allow_html=True)
        with cols[4]:
            _sign_control(st, v, override, signed_rec, contested)
    sub = _table_subline_html(v, override, refinement, is_signed)
    if sub:
        st.markdown(sub, unsafe_allow_html=True)
    _render_signoff_card(st, v.cluster)
    if refinement is not None:
        from ui import conversation as convo

        convo._render_draft_card(v.cluster, thread_key=f"holistic::{v.cluster}",
                                 trigger="holistic_review")


def _render_table(st, verdicts: list[ClusterVerdict], overrides: dict, signed: dict,
                  refinements: dict) -> None:
    """The clinical review table, held in one contained surface: every call in stable
    cluster order, one sign-off control per row, action rows washed and flagged with a
    grounded reason. ``verdicts`` are the COMPOSED verdicts."""
    with st.container(key="pano_tblcard"):
        _render_table_head(st)
        for v in verdicts:
            _render_table_row(st, v, overrides.get(v.cluster), signed.get(v.cluster),
                              refinements.get(v.cluster))


def _render_drilldown_actions(st, v: ClusterVerdict, override, signed_rec, refinement) -> None:
    """Sign-off + chat (+ accept refinement) on the cluster drill-down, so review-then-accept
    is one screen — the same controls as the table row, laid out inline above the evidence."""
    contested = bool(v.verify or (override and override.get("dissent")))
    c_sign, c_chat, _sp = st.columns([0.16, 0.16, 0.68])
    with c_sign:
        _sign_control(st, v, override, signed_rec, contested)
    with c_chat, st.container(key=f"tact_chat_{v.cluster}"):
        st.button("Chat", key=f"dchat_{v.cluster}", on_click=state.open_cluster_chat,
                  args=(v.cluster,), use_container_width=True,
                  help="Open this cluster's conversation to question or override the call.")
    if refinement is not None:
        _accept_refine_button(st, refinement)


def _ws_head_html(cluster: str, cell_type: str, confidence: str, verify: bool) -> str:
    color = fmt.cluster_color(cluster)
    css, _ = fmt.confidence_chip(confidence)
    flag = ' <span class="pano-sum-flag">⚑ re-check</span>' if verify else ""
    return (
        f'<div class="pano-ws-head"><span class="dot" style="background:{color}"></span>'
        f'<span class="cid">{html.escape(cluster)}</span>'
        f'<span class="ct">{html.escape(cell_type)}</span>'
        f'<span class="cf {css}">{html.escape(confidence)}</span>{flag}</div>'
    )


def _anchored_note_html(notes: list) -> str:
    """A caveat line for my notes anchored to a gene/program: the claim + [note:id] +
    the literature tension, rendered beneath the driver row it modifies."""
    if not notes:
        return ""
    bits = []
    for n in notes[:2]:
        t = n.tension
        if t.dissent:
            tension = f' · {len(t.dissent)} lit. dissent'
        elif t.agree:
            tension = f' · {len(t.agree)} lit. agree'
        else:
            tension = ""
        bits.append(
            f'⚑ note: {html.escape(n.claim)} '
            f'<span class="pano-notecite">[note:{html.escape(n.id[:6])}]</span>{tension}'
        )
    return '<div class="pano-anchornote">' + "<br>".join(bits) + "</div>"


def _marker_evidence_table_html(v: ClusterVerdict, notes: dict = None) -> str:
    """The cluster's marker evidence with jazzPanda stats — top genes by glm_coef.
    A grounded projection of ``ClusterVerdict.evidence``; nothing invented. Any note
    anchored to a gene renders as a caveat row directly beneath that gene's driver row."""
    gene_notes = (notes or {}).get("gene", {})
    ev = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)[:8]
    if not ev:
        return '<div class="pano-lk-empty">No marker evidence recorded for this cluster.</div>'
    cols = (("Gene", 26), ("glm coef", 20), ("Pearson r", 20), ("Role", 34))
    colgroup = "".join(f'<col style="width:{w}%">' for _, w in cols)
    head = "".join(f"<th>{html.escape(lbl)}</th>" for lbl, _ in cols)
    rows = []
    for e in ev:
        canon = ' <span class="pano-canon" title="canonical marker">★</span>' if e.is_canonical else ""
        gene = f'<span class="pano-sum-cid">{html.escape(e.gene)}</span>{canon}'
        rows.append(
            f'<tr><td>{gene}</td>'
            f'<td><span class="pano-num">{e.glm_coef:.2f}</span></td>'
            f'<td><span class="pano-num">{e.pearson:.2f}</span></td>'
            f'<td><span class="pano-role">{html.escape(str(e.role))}</span></td></tr>'
        )
        anch = _anchored_note_html(gene_notes.get(e.gene))
        if anch:
            rows.append(f'<tr class="pano-anchor-row"><td colspan="4">{anch}</td></tr>')
    return (
        '<div class="pano-sum-tablewrap"><table class="pano-sum-table">'
        f"<colgroup>{colgroup}</colgroup><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _enrichment_evidence_table_html(ce, notes: dict = None) -> str:
    """The cluster's enriched (and suggestive) programs with jazzPanda stats +
    panel coverage. A grounded projection of ``ClusterEnrichment``. A note anchored
    to a gene set renders as a caveat row directly beneath that program's row."""
    set_notes = (notes or {}).get("gene_set", {})
    if ce is None:
        return '<div class="pano-lk-empty">No enrichment slice for this dataset.</div>'
    graded = [(p, "enriched") for p in ce.enriched] + [(p, "suggestive") for p in ce.suggestive]
    if not graded:
        return '<div class="pano-lk-empty">No gene set clears the enrichment gate for this cluster.</div>'
    cols = (("Program", 33), ("Score", 12), ("Cov", 14), ("Leading edge", 41))
    colgroup = "".join(f'<col style="width:{w}%">' for _, w in cols)
    head = "".join(f"<th>{html.escape(lbl)}</th>" for lbl, _ in cols)
    rows = []
    for p, tier in graded:
        name = (f'<span class="pano-sum-ct">{html.escape(_short(p.gene_set))}</span>'
                f'<span class="pano-tier {tier}">{tier}</span>')
        le = " · ".join(html.escape(g) for g in p.leading_edge[:8]) or "—"
        rows.append(
            f'<tr><td>{name}</td>'
            f'<td><span class="pano-num">{p.score:.2f}</span></td>'
            f'<td><span class="pano-num-dim" title="set genes on the panel / set size">{p.panel_hits}/{p.set_size_full}</span></td>'
            f'<td><span class="pano-enr-le">{le}</span></td></tr>'
        )
        anch = _anchored_note_html(set_notes.get(p.gene_set))
        if anch:
            rows.append(f'<tr class="pano-anchor-row"><td colspan="4">{anch}</td></tr>')
    return (
        '<div class="pano-sum-tablewrap"><table class="pano-sum-table">'
        f"<colgroup>{colgroup}</colgroup><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# Editable working space — one section rendered at a time; edits held in plain
# ``wsval_*`` keys so navigating away never drops them.
# --------------------------------------------------------------------------- #
def _reset_ws_all(defaults: dict, dataset: str) -> None:
    """on_click: re-draft EVERY region from its freshest auto-seed AND drop the saved
    edits on disk, so 're-draft from latest' is a true reset. Pops the mounted widget
    key so the visible editor re-seeds from the new default."""
    import streamlit as st

    for name, default in defaults.items():
        st.session_state[f"wsval_{name}"] = default
        st.session_state.pop(f"wsw_{name}", None)
    st.session_state.pop("_sum_saved_snapshot", None)
    st.session_state.pop("_sum_saved_at", None)
    try:
        from pipeline import store

        store.save_summary_edits({}, dataset, saved_at="")
    except Exception:  # noqa: BLE001 - clearing the disk copy is best-effort
        pass


def _autoheight(text: str, *, min_lines: int) -> int:
    """Pixel height that fits ``text``: hard newlines plus ~``_WRAP_CHARS``-char soft
    wraps, floored at ``min_lines`` and capped so a huge paste can't make a page-tall
    box. Sizes each editor to its real content so nothing hides below an inner scroll."""
    lines = sum(max(1, math.ceil(len(line) / _WRAP_CHARS)) for line in text.split("\n"))
    return min(_ED_MAX_PX, max(min_lines, lines) * _PX_PER_LINE + _ED_PAD)


def _set_active(section: str) -> None:
    """on_click: focus a rail section."""
    import streamlit as st

    st.session_state[_K_ACTIVE] = section


def _save_now() -> None:
    """on_click for the Save button: force the next render's autosave to write and
    refresh the 'saved HH:MM' status, even if nothing changed since the last autosave."""
    import streamlit as st

    st.session_state.pop("_sum_saved_snapshot", None)


def _draft_refinement(r) -> None:
    """on_click: draft a celltype_override from a holistic refinement (its from_call ->
    to_call), reconciled against the literature, and stash it on the cluster's holistic
    thread so the same two-tap confirm card renders. Saved with trigger=holistic_review.
    A refinement is a within-lineage subtype sharpening, so lineage/category are left to
    the computed values (subject_cell_type carries the new call)."""
    from agent import memory
    from ui import state

    lit = None
    try:
        from agent import tools

        lit = tools._literature_search_fn()
    except Exception:  # noqa: BLE001 - no connector -> honest thin tension
        lit = None
    try:
        draft = memory.draft_note(
            claim=f"Refine {r.cluster} from {r.from_call} to {r.to_call} — {r.rationale}",
            scope="cluster", basis="convention", cluster=r.cluster,
            note_type="celltype_override", subject_cell_type=r.to_call,
            literature_search=lit,
        )
        state.set_pending_draft(f"holistic::{r.cluster}", draft)
    except Exception:  # noqa: BLE001 - never crash the page on a bad refinement
        pass


def _editor(st, name: str, default: str, height: int) -> None:
    """Render one editable region. Canonical text lives in ``wsval_{name}`` (a plain
    key that survives when the widget is unmounted); the ``wsw_{name}`` widget seeds
    from it. The caller reconciles ``wsval`` from ``wsw`` before this runs."""
    vkey = f"wsval_{name}"
    if vkey not in st.session_state:
        st.session_state[vkey] = default
    val = st.text_area("edit", value=st.session_state[vkey], key=f"wsw_{name}",
                       height=height, label_visibility="collapsed")
    st.session_state[vkey] = val


def _save_button(st, name: str) -> None:
    """A per-region Save button, right under its editor. Edits also autosave as you
    type; this is the explicit affordance for that one region."""
    with st.container(key=f"savrow_{name}"):
        st.button("Save", key=f"btn_ws_save_{name}", on_click=_save_now,
                  help="Save your edits now — they also autosave as you type and reload after a refresh.")


def _eyebrow(st, text: str) -> None:
    st.markdown(f'<div class="pano-ed-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def _rail_label(st, text: str) -> None:
    """A rail group label in a keyed container (so its spacing margin is honoured
    and never overflows onto the next button)."""
    with st.container(key=f"raillbl_{text.lower()}"):
        st.markdown(f'<div class="pano-rail-lbl">{html.escape(text)}</div>', unsafe_allow_html=True)


def _render_lab_note_cards(st, notes: list) -> None:
    if not notes:
        st.markdown('<div class="pano-lk-empty">No notes yet. Override or confirm a call in the '
                    "chat and it is saved here with its basis and any literature tension.</div>",
                    unsafe_allow_html=True)
        return
    for n in sorted(notes, key=lambda x: (x.created_at, x.id), reverse=True):
        scope = (f"cluster {n.scope_ref.cluster}" if n.scope == "cluster" and n.scope_ref.cluster
                 else {"dataset": "this dataset", "lab": "all datasets"}.get(n.scope, n.scope))
        date = n.created_at.split("T", 1)[0] if n.created_at else "n/a"
        st.markdown(
            f'<div class="pano-lk-card"><div class="pano-lk-claim">{html.escape(n.claim)}</div>'
            f'<div class="pano-lk-meta"><span class="sco">{html.escape(scope)}</span> · '
            f'basis: {html.escape(n.basis)} · {html.escape(n.status)} · '
            f'{html.escape(n.author or "you")} · {html.escape(date)}</div></div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render_summary_page() -> None:
    """Two-pane report editor: contents rail (Overall + clusters + dataset) on the
    left, one focused section on the right."""
    import streamlit as st

    st.markdown(f"<style>{_SUMMARY_CSS}</style>", unsafe_allow_html=True)

    verdicts = da.composed_verdicts()  # confirmed overrides/excludes reflected everywhere
    overrides = {v.cluster: da.override_info(v.cluster) for v in verdicts}
    overrides = {c: o for c, o in overrides.items() if o}
    sec_by_id = {v.cluster: v for v in verdicts}
    # Sign-off state drives the board triage AND the header finish line.
    signed = da.signed_off()
    refinements = _refinements_by_cluster()
    n_signed = sum(1 for v in verdicts if v.cluster in signed)
    n_needs = sum(1 for v in verdicts
                  if _triage_bucket(v, signed, refinements, overrides) == _BUCKET_NEEDS)
    try:
        n_panel = len(da.panel_names())
    except Exception:  # noqa: BLE001 - a missing panel count is not fatal to the page
        n_panel = 0
    try:
        enrichments = da.all_enrichments()
    except Exception:  # noqa: BLE001 - no enrichment slice -> marker-only summary
        enrichments = []
    enr_map = {ce.cluster: ce for ce in enrichments}

    # Latest per-section drafts (grounded). Computed up front so one button can
    # re-draft everything and the exports read the freshest text.
    rep = report.build_report_from_sources(generated_at=datetime.date.today().isoformat())
    try:
        themes = da.pathway_themes()
    except Exception:  # noqa: BLE001
        themes = None
    ws_defaults: dict[str, str] = {
        s.cluster: report.default_cluster_summary(s) for s in rep.sections
    }
    ws_defaults[_ED_GLOBAL] = report.global_check_text(da.holistic(), themes)
    ws_defaults[_SEC_CAVEATS] = report.caveats_text(verdicts, enr_map, n_panel)

    # Restore the biologist's saved edits (edits win over the fresh auto-draft), so a
    # browser refresh brings the edited text back. Seed session state once per session.
    try:
        from pipeline import store

        saved = store.load_summary_edits(rep.dataset)
    except Exception:  # noqa: BLE001 - no tree / import issue -> just use auto-drafts
        store, saved = None, {}

    def _seed(name: str) -> str:
        return saved.get(name) or ws_defaults[name]

    if not st.session_state.get("_sum_seeded"):
        st.session_state["_sum_seeded"] = True
        for name in ws_defaults:
            st.session_state.setdefault(f"wsval_{name}", _seed(name))

    # Reconcile canonical text from the (possibly just-edited) mounted widget BEFORE
    # exports/autosave, so both always reflect the latest keystrokes.
    for name in ws_defaults:
        wkey = f"wsw_{name}"
        if wkey in st.session_state:
            st.session_state[f"wsval_{name}"] = st.session_state[wkey]

    def _val(name: str) -> str:
        return st.session_state.get(f"wsval_{name}", _seed(name))

    # Autosave: persist only regions the biologist changed from the auto-draft
    # (self-cleaning). Idempotent — writes only when the edit set actually changed.
    edits = {n: _val(n) for n in ws_defaults if _val(n) != ws_defaults[n]}
    save_error = ""
    if store is not None and edits != st.session_state.get("_sum_saved_snapshot"):
        try:
            now = datetime.datetime.now()
            store.save_summary_edits(edits, rep.dataset,
                                     saved_at=now.isoformat(timespec="seconds"))
            st.session_state["_sum_saved_snapshot"] = dict(edits)
            st.session_state["_sum_saved_at"] = now.strftime("%H:%M")
        except Exception as exc:  # noqa: BLE001 - surface, never crash the page
            save_error = str(exc)

    active = st.session_state.get(_K_ACTIVE, _SEC_OVERALL)

    export_sections = [
        (s.cluster, s.cell_type, s.confidence, s.verify, _val(s.cluster))
        for s in rep.sections
    ]
    ga = datetime.date.today().isoformat()
    exp_kw = dict(dataset=rep.dataset, generated_at=ga,
                  global_check=_val(_ED_GLOBAL), caveats=_val(_SEC_CAVEATS),
                  lab_notes=rep.dataset_notes)

    # Save-status line (autosave + the manual Save button both feed it).
    if save_error:
        status = f'<span class="err">save failed — {html.escape(save_error)}</span>'
    elif not edits:
        status = "no edits yet"
    elif edits == st.session_state.get("_sum_saved_snapshot") and st.session_state.get("_sum_saved_at"):
        status = f'saved {html.escape(st.session_state["_sum_saved_at"])}'
    else:
        status = "editing…"

    # ---- Top bar: title + meta + save status | [Report] [CSV] [Save] ----- #
    head_col, act_col = st.columns([0.60, 0.40], vertical_alignment="center")
    with head_col:
        st.markdown(
            f'<div class="pano-sum-title">{html.escape(_TITLE)}</div>'
            '<div class="pano-sum-sub">Review each call, sign it off, and export — every '
            "number is jazzPanda's, every biology claim live-cited.</div>",
            unsafe_allow_html=True,
        )
    with act_col:
        b_doc, b_csv = st.columns(2)
        with b_doc:
            st.download_button(
                _DOCX_LABEL, report.working_docx(export_sections, **exp_kw),
                _DOCX_NAME, _DOCX_MIME, key="dl_docx", use_container_width=True,
                help="Your edited narrative — every cluster, the global check, caveats, and my notes.")
        with b_csv:
            st.download_button(
                _CSV_LABEL, da.verdict_csv(), _CSV_NAME, _CSV_MIME,
                key="dl_csv", use_container_width=True,
                help="The 11-column verdict table (R-importable). The structured call, not your free-text edits.")

    # A quiet, page-level re-draft control — kept OUT of the export cluster because it
    # REPLACES your edits (rebuilds every region from the latest calls + programs).
    with st.container(key="pano_redraft"):
        st.button("↻ re-draft all from latest", key="btn_ws_refresh",
                  on_click=_reset_ws_all, args=(ws_defaults, rep.dataset),
                  help="Rebuild every region from the latest calls, programs, and my notes — replaces your current edits.")

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ---- Section sub-nav (the clusters live IN the table now, not a rail) --- #
    with st.container(key="pano_subnav"):
        nav_cols = st.columns(3)
        for col, (sec, label) in zip(
            nav_cols,
            ((_SEC_OVERALL, "Overall"), (_SEC_CAVEATS, "Caveats"), (_SEC_NOTES, "My notes")),
        ):
            with col:
                st.button(label, key=f"snav_{sec}", use_container_width=True,
                          type="primary" if active == sec else "secondary",
                          on_click=_set_active, args=(sec,))

    with st.container(key="pano_pane"):
        if active == _SEC_OVERALL:
            # The whole call-set as one clinical table: a stat strip, the one-line
            # reconciliation cues (grounded), then a row per call to sign off inline.
            st.markdown(_stat_strip_html(n_signed, len(verdicts), n_needs, n_panel, status),
                        unsafe_allow_html=True)
            line = _reconciliation_line_html(_reconciliation_items(verdicts, themes, overrides))
            if line:
                st.markdown(line, unsafe_allow_html=True)
            _render_table(st, verdicts, overrides, signed, refinements)

            # The cross-cluster write-up stays for the export, demoted below the board —
            # it is report prose, not the review action.
            st.markdown('<hr class="pano-ed-hr"/>', unsafe_allow_html=True)
            _eyebrow(st, "Cross-cluster write-up · opens the exported report")
            st.markdown('<div class="pano-export-note">A short synthesis across all calls — '
                        "edit freely; this is what the report opens with.</div>",
                        unsafe_allow_html=True)
            _editor(st, _ED_GLOBAL, _seed(_ED_GLOBAL),
                    height=_autoheight(_val(_ED_GLOBAL), min_lines=6))
            _save_button(st, _ED_GLOBAL)

        elif active in sec_by_id:
            with st.container(key="backlink"):
                st.button("← All calls", key="back_overall",
                          on_click=_set_active, args=(_SEC_OVERALL,))
            s = sec_by_id[active]
            v = da.verdict_for(s.cluster)
            ce = enr_map.get(s.cluster)
            _eyebrow(st, "Cluster · evidence + write-up")
            st.markdown(_ws_head_html(s.cluster, s.cell_type, s.confidence, s.verify),
                        unsafe_allow_html=True)

            # Sign off right here — the same controls the table row offers, so the
            # biologist can read the evidence below and accept the call on one screen
            # instead of bouncing back to the table.
            _render_drilldown_actions(st, s, overrides.get(s.cluster),
                                      signed.get(s.cluster), refinements.get(s.cluster))
            _render_signoff_card(st, s.cluster)
            if s.cluster in refinements:
                from ui import conversation as convo

                convo._render_draft_card(s.cluster, thread_key=f"holistic::{s.cluster}",
                                         trigger="holistic_review")

            # Evidence first — this drill-down is reached from the board's "Evidence"
            # action, so the grounded jazzPanda numbers the call rests on lead. Any note
            # anchored to a driver / program renders beneath the exact row it modifies.
            anch = da.anchored_notes(s.cluster)
            st.markdown(
                '<div class="pano-ref-cap">Marker evidence'
                f'<span class="meta">jazzPanda · top {min(8, len(v.evidence))} of {len(v.evidence)} by glm coef</span></div>',
                unsafe_allow_html=True)
            st.markdown(_marker_evidence_table_html(v, anch), unsafe_allow_html=True)
            st.markdown(
                '<div class="pano-ref-cap">Enriched programs'
                '<span class="meta">panel-scoped · score = jazzPanda test statistic · cov = set genes on panel</span></div>',
                unsafe_allow_html=True)
            st.markdown(_enrichment_evidence_table_html(ce, anch), unsafe_allow_html=True)

            # ...then the editable write-up (what exports for this cluster) — demoted
            # below the evidence, no longer the first thing the biologist faces.
            st.markdown('<hr class="pano-ed-hr"/>', unsafe_allow_html=True)
            _eyebrow(st, "Write-up · exports for this cluster")
            _editor(st, s.cluster, _seed(s.cluster),
                    height=_autoheight(_val(s.cluster), min_lines=10))
            st.markdown('<div class="pano-ed-hint">Edit freely — this is exactly what exports for '
                        f"{html.escape(s.cluster)}.</div>", unsafe_allow_html=True)
            _save_button(st, s.cluster)

        elif active == _SEC_CAVEATS:
            _eyebrow(st, "Dataset · caveats")
            _editor(st, _SEC_CAVEATS, _seed(_SEC_CAVEATS),
                    height=_autoheight(_val(_SEC_CAVEATS), min_lines=13))
            _save_button(st, _SEC_CAVEATS)

        else:  # _SEC_NOTES (or any stale value) -> my notes
            _eyebrow(st, "My notes · your saved notes")
            st.markdown('<div class="pano-ed-hint" style="margin:0 0 10px">Captured when you override '
                        "or confirm a call in the chat; compiled into the write-up and the exported "
                        "report.</div>", unsafe_allow_html=True)
            _render_lab_note_cards(st, da.read_notes())


__all__ = ["render_summary_page"]
