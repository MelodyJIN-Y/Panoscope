"""Summary page — the two-pane report editor (the final, integrated step).

Overall (sign-off board + cross-cluster synthesis), one drill-down per cluster
(evidence, then the editable write-up), Caveats and My notes. Nothing here computes
a value: the board projects the composed verdicts and every draft comes from
``ui.report``. Editable regions auto-seed from the latest draft and keep the
biologist's edits; the exports are what was edited. Streamlit is imported lazily.
"""

from __future__ import annotations

import datetime
import hashlib
import html
import math

from panoscope.agent import config as cfg
from panoscope.agent.enrichment_themes import pretty_gene_set
from panoscope.agent.types import ClusterVerdict

from panoscope.ui import data_access as da
from panoscope.ui import format as fmt
from panoscope.ui import report
from panoscope.ui import state

# --------------------------------------------------------------------------- #
# Copy + export metadata. Constants so prose never drifts.
# --------------------------------------------------------------------------- #
_TITLE = "Interpretation summary"

# The report as an editable Word document and a locked PDF.
_DOCX_LABEL = "⬇  Report (Word)"
_DOCX_NAME = "panoscope_interpretation.docx"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PDF_LABEL = "⬇  Report (PDF)"
_PDF_NAME = "panoscope_interpretation.pdf"
_PDF_MIME = "application/pdf"

# Editor auto-height: ~22px/line; 110 chars/line under-counts the pane, so never a clip.
_WRAP_CHARS = 110
_PX_PER_LINE = 22
_ED_PAD = 28
_ED_MAX_PX = 1600

# Sections: overall, one per cluster, then the dataset-level caveats and notes.
_SEC_OVERALL = "overall"
_SEC_CAVEATS = "caveats"
_SEC_NOTES = "labnotes"
_ED_GLOBAL = "global"   # editor name for the cross-cluster check (lives on Overall)
_K_ACTIVE = "sum_active_section"  # which rail item is focused

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

/* Export cluster: two downloads of the same report (Word, PDF). */
.st-key-dl_docx button, .st-key-dl_pdf button {
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
.pano-sum-cid { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink); vertical-align: middle; }
.pano-sum-ct { font-size: 13.5px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.pano-enr-le { font-family: var(--mono); font-size: 10px; color: var(--faint); line-height: 1.55;
  word-break: break-word; display: block; margin-top: 3px; }
.pano-sum-flag { font-family: var(--mono); font-size: 9.5px; font-weight: 600; color: var(--absent);
  background: var(--absent-bg); padding: 2px 7px; border-radius: 6px; white-space: nowrap; }
.pano-sum-table .cf { font-size: 10px; padding: 2px 8px; }

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
/* Lead: one plain-language instruction + progress — says what to do, no mono noise. */
.pano-lead { font-size: 14px; line-height: 1.5; color: var(--muted); margin: 2px 0 14px; max-width: 100%; }
.pano-lead b { color: var(--ink); font-weight: 700; }
.pano-lead .prog { color: var(--muted); }
.pano-lead .need { color: var(--absent); font-weight: 600; }
.pano-lead .sep { color: var(--hair); margin: 0 8px; }
/* The table's contained surface: a white card with a soft shadow, width-capped so
   the columns stay tight instead of sprawling across the viewport. */
.st-key-pano_tblcard { background: var(--paper); border: 1px solid var(--hair); border-radius: 14px;
  box-shadow: 0 1px 2px rgba(20,27,32,.05), 0 6px 22px rgba(20,27,32,.035);
  padding: 4px 22px 8px; max-width: 100%; }
/* Confidence as a SOFT badge (uniform light chip, ramp-coloured text) — reads the
   gradient without the heavy solid pills. */
.pano-cfb { font-family: var(--mono); font-size: 10px; font-weight: 600; padding: 2px 8px;
  border-radius: 5px; letter-spacing: .02em; white-space: nowrap; background: #EFF2F3; }
.pano-cfb-vh { color: #0F5B65; }
.pano-cfb-h  { color: #237E89; }
.pano-cfb-mh { color: #3B7C84; }
.pano-cfb-m  { color: #737B82; }
.pano-cfb-l  { color: #99A2A9; }

/* === The review table: a REAL HTML table (auto-sized columns locked to their
 *     headers, tight even rhythm — what st.columns cannot give). Actions are
 *     query-param links the app dispatches on click. =========================== */
.ptbl { width: 100%; border-collapse: collapse; font-family: var(--sans); table-layout: auto; }
/* Reset Streamlit's default markdown-table borders (they add the boxy vertical rules). */
.ptbl, .ptbl th, .ptbl td, .ptbl tr { border: 0 !important; background: transparent; }
.ptbl thead th { text-align: left; font-family: var(--mono); font-size: 9px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); font-weight: 600; padding: 4px 18px 10px 0; white-space: nowrap;
  border-bottom: 1px solid var(--hair) !important; }
.ptbl thead th:first-child { padding-left: 2px; }
.ptbl td { padding: 9px 18px 9px 0; border-bottom: 1px solid var(--hair2) !important; vertical-align: middle; }
.ptbl td:first-child { padding-left: 2px; }
.ptbl tbody tr:last-child td { border-bottom: 0; }
.ptbl tbody tr.prow:hover td { background: #FAFCFC; }
.ptbl tr.need td { background: #FCFAF3; }
/* Columns: the checklist tick + id stay narrow; the rest auto-size to content. */
.ptbl .c-sign { width: 30px; text-align: center; }
.ptbl .c-cid  { width: 42px; white-space: nowrap; }
.ptbl .c-ct   { white-space: nowrap; }
.ptbl .c-conf { white-space: nowrap; }
/* Checklist tick (a query-param link). */
.ptbl a.tick { text-decoration: none; font-size: 18px; color: var(--muted); line-height: 1; }
.ptbl a.tick:hover { color: var(--accent); }
.ptbl a.tick.done { color: var(--accent); }
.ptbl a.warnflag { color: var(--absent); font-size: 15px; text-decoration: none; }
.ptbl a.warnflag:hover { filter: brightness(.85); }
/* Cluster id link. */
.ptbl a.cid { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--muted); text-decoration: none; }
.ptbl a.cid:hover { color: var(--accent); }
/* Cell type — the row's hero. */
.ptbl .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 10px; vertical-align: middle; }
.ptbl .ct { font-size: 13.5px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; white-space: nowrap; }
.ptbl .yours { font-family: var(--mono); font-size: 8px; text-transform: uppercase; letter-spacing: .04em;
  font-weight: 700; color: var(--accent); background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; margin-left: 7px; }
/* Key markers — the top driver (bold + coef) then the next key markers. */
.ptbl .km { font-family: var(--mono); font-size: 11px; color: var(--faint); white-space: nowrap; }
.ptbl .km b { color: var(--ink); font-weight: 600; }
.ptbl .km .n { color: var(--muted); }
.ptbl .km-dash { color: var(--hair); margin: 0 5px; }
/* Signed rows recede. */
.ptbl tr.done td .ct, .ptbl tr.done td .km, .ptbl tr.done td .dot, .ptbl tr.done td .pano-cfb { opacity: .5; }
/* Sub-row (flagged / to-refine): reason on the left, its action pill on the right,
 * kept ON the table grid via a colspan. */
.ptbl tr.subrow td { padding-top: 0; padding-bottom: 9px; }
.ptbl .subwrap { display: flex; align-items: center; gap: 16px; }
.ptbl .subtext { font-size: 11.5px; color: var(--muted); line-height: 1.5; flex: 1; }
.ptbl .subtext .warn { color: var(--absent); font-weight: 600; }
.ptbl .subtext b { color: var(--ink); font-weight: 600; }
.ptbl a.act { text-decoration: none; font-family: var(--mono); font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .04em; padding: 6px 13px; border-radius: 8px; white-space: nowrap; flex: 0 0 auto; }
.ptbl a.act.confirm { border: 1px solid var(--absent); color: var(--absent); background: var(--absent-bg); }
.ptbl a.act.confirm:hover { background: var(--absent); color: #fff; }
.ptbl a.act.accept { border: 1px solid var(--accent); color: var(--accent); background: var(--accent-soft); }
.ptbl a.act.accept:hover { background: var(--accent); color: #fff; }
/* --- Reference-style rich rows: big GLM/pearson stat + cited biology + caveat pill --- */
.ptbl th .cited { text-transform: none; letter-spacing: 0; color: var(--faint); font-weight: 500; }
.ptbl .c-stat { white-space: nowrap; }
.ptbl .stat-big { font-family: var(--mono); font-size: 15px; font-weight: 700; color: var(--ink); line-height: 1.05; }
.ptbl .stat-sub { font-family: var(--mono); font-size: 9.5px; color: var(--faint); margin-top: 3px; }
.ptbl .stat-sub b { color: var(--muted); font-weight: 600; }
.ptbl .bio { font-size: 12px; color: var(--muted); line-height: 1.55; }
.ptbl .bio-dim { color: var(--faint); }
.ptbl a.pmid { font-family: var(--mono); font-size: 10.5px; color: var(--accent); text-decoration: none; white-space: nowrap; margin-left: 5px; }
.ptbl a.pmid:hover { text-decoration: underline; }
.ptbl .cav { font-family: var(--mono); font-size: 9px; font-weight: 700; color: var(--absent); border: 1px solid var(--absent);
  border-radius: 5px; padding: 2px 7px; margin-left: 8px; white-space: nowrap; }
/* Sign-off: an open circle (○) unsigned, a teal check (✓) when signed off. */
.ptbl a.tick { font-size: 14px; }
.ptbl a.tick.done { font-size: 15px; font-weight: 700; }
/* Read-only cross-cluster overview (bullets) at the top of the page. */
.pano-ovw { background: var(--paper); border: 1px solid var(--hair); border-radius: 12px;
  padding: 13px 20px; margin: 2px 0 16px; }
.pano-ovw-hd { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .1em;
  color: var(--faint); font-weight: 600; margin: 0 0 7px; }
.pano-ovw ul { margin: 0; padding-left: 19px; }
.pano-ovw li { font-size: 12.5px; color: var(--muted); line-height: 1.6; margin: 5px 0; }
.pano-ovw li::marker { color: var(--accent); }
/* Drill-down back link. */
div[class*="st-key-backlink"] button { min-height: 0 !important; padding: 3px 0 !important; border: 0 !important;
  background: transparent !important; box-shadow: none !important; color: var(--faint) !important;
  font-family: var(--mono) !important; font-size: 11px !important; font-weight: 600 !important; }
div[class*="st-key-backlink"] button:hover { color: var(--accent) !important; }
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


# --------------------------------------------------------------------------- #
# Sign-off board: the call set triaged into needs-you / to-review / signed off.
# --------------------------------------------------------------------------- #
_BUCKET_NEEDS = "needs"
_BUCKET_REVIEW = "review"
_BUCKET_SIGNED = "signed"
_SIGNOFF_SLOT = "signoff::{}"  # pending-draft slot for a contested sign-off's note


def _refinements_by_cluster() -> dict:
    """The holistic pass's proposed refinements, keyed by cluster (grounded, may be
    empty). A refinement is a within-lineage subtype sharpening to consider."""
    try:
        hol = da.holistic()
        refs = list(getattr(hol, "refinements", ()) or []) if hol else []
        return {r.cluster: r for r in refs}
    except Exception:  # noqa: BLE001 - no holistic slice -> no refinements
        return {}


def _flag_reason(v: ClusterVerdict) -> str:
    """The real reason a call is flagged for re-check (few assigned genes, or a
    low-specificity demotion), projected from the verdict; never a blanket 'thin'."""
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


def _lead_html(n_signed: int, n_total: int, n_needs: int) -> str:
    """One plain-language instruction + progress — tells the biologist what to do, in
    sentence case (no mono-uppercase noise). Grounded counts."""
    if n_signed >= n_total:
        return ('<div class="pano-lead"><b>All calls signed off.</b> '
                '<span class="prog">Export the report or the annotations table.</span></div>')
    need = (f'<span class="sep">·</span><span class="need">{n_needs} need a closer look first</span>'
            if n_needs else "")
    return ('<div class="pano-lead"><b>Tick each call to sign it off.</b> '
            f'<span class="prog">{n_signed} of {n_total} done</span>{need}</div>')


def _signoff_claim(v: ClusterVerdict, override) -> str:
    """The grounded validation claim a contested sign-off records in My notes."""
    if override is not None and override.get("dissent"):
        return (f"Accepted the {v.cell_type} call for {v.cluster} despite "
                f"{override['dissent']} literature dissent. Reviewed and confirmed.")
    ev = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)
    top = f"top marker {ev[0].gene}" if ev else "its top marker"
    return (f"Confirmed {v.cell_type} for {v.cluster} despite being flagged "
            f"({_flag_reason(v)}; {top}). Reviewed and accepted.")


# --- Sign-off actions (on_click handlers) ---------------------------------- #
def _lit_search():
    """The literature connector (or None -> an honest thin tension on the note)."""
    try:
        from panoscope.agent import tools

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
    """on_click: draft a validation note for a contested sign-off, reconciled against
    the literature. Nothing persists until the biologist confirms."""
    from panoscope.agent import memory

    try:
        draft = memory.draft_note(
            claim=claim, scope="cluster", basis="convention", cluster=cluster,
            note_type="validation", literature_search=_lit_search(),
        )
        state.set_pending_draft(_SIGNOFF_SLOT.format(cluster), draft)
    except Exception:  # noqa: BLE001 - never crash the board on a bad draft
        pass


def _render_signoff_card(st, cluster: str) -> None:
    """The contested-sign-off confirm card. Confirm saves the note and records the
    sign-off; Cancel drops the draft."""
    import dataclasses

    from panoscope.ui import conversation as convo

    slot = _SIGNOFF_SLOT.format(cluster)
    draft = state.get_pending_draft(slot)
    if draft is None:
        return
    # Per-cluster container key; draft-nonce so each fresh draft re-seeds its toggle.
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
                         width="stretch")
        no = c_no.button("Cancel", key=f"sof_no_{cluster}", width="stretch")
    if no:
        state.clear_pending_draft(slot)
        convo._rerun(st)
    elif ok:
        edited = dataclasses.replace(draft, basis=basis, scope="cluster", cluster=cluster,
                                     status="firm")
        try:
            note = da.save_note_draft(edited, trigger="manual_add")
            da.mark_signed_off(cluster, note_id=note.id, at=_now_iso())
        except Exception:  # noqa: BLE001 - surface nothing, never crash the board
            pass
        state.clear_pending_draft(slot)
        convo._rerun(st)


# --- Row + board rendering -------------------------------------------------- #
def _conf_badge(confidence: str) -> str:
    """Confidence as a soft, ramp-coloured badge; none for a unit with no band."""
    if not confidence:
        return ""
    css, _ = fmt.confidence_chip(confidence)          # e.g. "cf-vh"
    level = css.replace("cf-", "") or "m"
    return f'<span class="pano-cfb pano-cfb-{html.escape(level)}">{html.escape(confidence)}</span>'


# --------------------------------------------------------------------------- #
# The review table: pure HTML; every action is a query-param link.
# --------------------------------------------------------------------------- #
def _attention_note(v: ClusterVerdict, override, refinement) -> str:
    """The short 'needs attention' reason for a caveat pill (grounded), or '' if none."""
    if v.verify:
        return _flag_reason(v)
    if refinement is not None:
        return f"refine → {refinement.to_call}?"
    if override is not None and override.get("dissent"):
        return f'{override["dissent"]} lit. dissent'
    return ""


def _biology_cell(v: ClusterVerdict, ct_notes: dict, needs_note: str) -> str:
    """Biology cell: the live-cited cell-type note (plus a key-markers sentence when
    it is one sentence long), then a caveat pill if the cluster needs a closer look."""
    # Through the guard: no note under a call it was not written about.
    note = da.celltype_note_for(v) if ct_notes else {}
    summary = str(note.get("summary") or "")
    pmid = note.get("pmid")
    body: list[str] = []
    if summary:
        body.append(f'<span class="bio">{html.escape(summary)}</span>')
        # Key-markers sentence only when the stored note is a single sentence.
        if summary.count(".") <= 1 and v.key_markers:
            km = ", ".join(str(g) for g in v.key_markers[:4])
            body.append(f'<span class="bio">Key markers: {html.escape(km)}.</span>')
    if pmid:
        body.append(f'<a class="pmid" href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(str(pmid))}/" '
                    f'target="_blank">PMID:{html.escape(str(pmid))}</a>')
    if needs_note:
        body.append(f'<span class="cav">⚠ {html.escape(needs_note)}</span>')
    if body:
        return " ".join(body)
    from panoscope.corpus.build import header

    h = header(cfg.current_study().id)     # unbuilt or incomplete: say it cannot settle
    empty = h if h.endswith("cannot settle") else "—"
    return f'<span class="bio-dim">{html.escape(empty)}</span>'


def _table_html(verdicts: list[ClusterVerdict], overrides: dict, signed: dict,
                refinements: dict) -> str:
    """The review table, reference-style: identity · big glm/pearson stat · cited biology.
    A cluster that needs a closer look carries a small ⚠ (no inline actions) — the full
    reason + Confirm/Accept live on its click-through drill-down."""
    try:
        ct_notes = da.celltype_notes()
    except Exception:  # noqa: BLE001 - no notes -> the biology column just shows a dash
        ct_notes = {}
    head = ('<thead><tr><th class="c-sign"></th><th class="c-cid">Cluster</th>'
            '<th class="c-ct">Cell type</th><th class="c-conf">Confidence</th>'
            '<th>Biology · relevance <span class="cited">(cited)</span></th></tr></thead>')
    rows: list[str] = []
    for v in verdicts:
        c = v.cluster
        is_signed = c in signed
        override = overrides.get(c)
        refinement = refinements.get(c)
        contested = bool(v.verify or (override and override.get("dissent")))
        needs = (contested or refinement is not None) and not is_signed
        cls = "prow" + (" need" if needs else "") + (" done" if is_signed else "")
        if is_signed:
            sign = f'<a class="tick done" href="?page=summary&undo={c}" target="_self" title="signed, reopen">&#10003;</a>'
        elif needs:
            sign = ('<a class="warnflag" href="?page=summary&drill=' + c + '" target="_self" '
                    'title="needs a closer look, click for details">⚠</a>')
        else:
            sign = f'<a class="tick" href="?page=summary&sign={c}" target="_self" title="sign off">○</a>'
        color = fmt.cluster_color(c)
        yours = ' <span class="yours">override</span>' if override else ""
        needs_note = _attention_note(v, override, refinement) if needs else ""
        rows.append(
            f'<tr class="{cls}"><td class="c-sign">{sign}</td>'
            f'<td class="c-cid"><a class="cid" href="?page=summary&drill={c}" target="_self">{html.escape(c)}</a></td>'
            f'<td class="c-ct"><span class="dot" style="background:{color}"></span>'
            f'<span class="ct">{html.escape(v.cell_type.replace("_", " "))}</span>{yours}</td>'
            f'<td class="c-conf">{_conf_badge(v.confidence)}</td>'
            f'<td>{_biology_cell(v, ct_notes, needs_note)}</td></tr>'
        )
    return f'<table class="ptbl">{head}<tbody>{"".join(rows)}</tbody></table>'


def _overview_bullets_html(text: str) -> str:
    """The cross-cluster synthesis as read-only bullet points (each paragraph -> a bullet).
    Grounded prose from ``report.global_check_text`` — shown, never edited here."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not paras:
        return ""
    items = "".join(f"<li>{html.escape(p)}</li>" for p in paras)
    return ('<div class="pano-ovw"><div class="pano-ovw-hd">Cross-cluster overview</div>'
            f"<ul>{items}</ul></div>")


def _handle_table_actions(st, verdicts: list[ClusterVerdict], overrides: dict,
                          refinements: dict) -> None:
    """Dispatch a table click that arrived as a query param, then clear it.

    Only ``sign``, ``undo`` and ``drill`` are currently sent; ``confirm`` and
    ``accept`` are kept as landing points for the contested-sign-off card.
    """
    qp = st.query_params
    vmap = {v.cluster: v for v in verdicts}

    def _pop(k: str) -> None:
        try:
            del qp[k]
        except Exception:  # noqa: BLE001 - clearing a stale param is best-effort
            pass

    if (c := qp.get("sign")) in vmap:
        da.mark_signed_off(c, note_id=None, at=_now_iso()); _pop("sign")
    elif (c := qp.get("undo")) in vmap:
        da.clear_signoff(c, at=_now_iso()); _pop("undo")
    elif (c := qp.get("drill")) in vmap:
        _set_active(c); _pop("drill")
    elif (c := qp.get("confirm")) in vmap:
        _begin_signoff(c, _signoff_claim(vmap[c], overrides.get(c))); _pop("confirm")
    elif (c := qp.get("accept")) in vmap:
        r = refinements.get(c)
        if r is not None:
            _draft_refinement(r)
        _pop("accept")


def _render_drilldown_actions(st, v: ClusterVerdict, override, signed_rec, refinement) -> None:
    """Sign-off + chat (+ accept refinement) on the cluster drill-down, so review-then-accept
    is one screen. Labeled buttons here (unlike the table's bare checklist tick)."""
    contested = bool(v.verify or (override and override.get("dissent")))
    cols = st.columns([0.24, 0.18, 0.24, 0.34])
    with cols[0]:
        if signed_rec is not None:
            st.button("✓ Signed — reopen", key=f"ddsign_{v.cluster}", width="stretch",
                      on_click=_undo_sign_off, args=(v.cluster,))
        elif contested:
            st.button("Confirm & sign off", key=f"ddsign_{v.cluster}", type="primary",
                      width="stretch", on_click=_begin_signoff,
                      args=(v.cluster, _signoff_claim(v, override)))
        else:
            st.button("Sign off", key=f"ddsign_{v.cluster}", type="primary",
                      width="stretch", on_click=_sign_off_clean, args=(v.cluster,))
    with cols[1]:
        st.button("Chat", key=f"ddchat_{v.cluster}", width="stretch",
                  on_click=state.open_cluster_chat, args=(v.cluster,))
    if refinement is not None:
        with cols[2]:
            st.button(f"Accept {refinement.to_call}", key=f"ddrefine_{v.cluster}",
                      width="stretch", on_click=_draft_refinement, args=(refinement,))


def _ws_head_html(cluster: str, cell_type: str, confidence: str, verify: bool) -> str:
    color = fmt.cluster_color(cluster)
    css, _ = fmt.confidence_chip(confidence)
    flag = ' <span class="pano-sum-flag">⚑ re-check</span>' if verify else ""
    return (
        f'<div class="pano-ws-head"><span class="dot" style="background:{color}"></span>'
        f'<span class="cid">{html.escape(cluster)}</span>'
        f'<span class="ct">{html.escape(cell_type)}</span>'
        + (f'<span class="cf {css}">{html.escape(confidence)}</span>' if confidence else "")
        + f'{flag}</div>'
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
    """The cluster's marker evidence (top genes by glm_coef), with any note anchored
    to a gene as a caveat row beneath it."""
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
    """The cluster's enriched and suggestive programs with stats and panel coverage,
    with any note anchored to a gene set as a caveat row beneath it."""
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
        name = (f'<span class="pano-sum-ct">{html.escape(pretty_gene_set(p.gene_set))}</span>'
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
def _skey(name: str) -> str:
    """``name``, scoped to the current study's tree path (hashed short, since it is
    also a widget key), so one study's edits can never be saved into another's."""
    tag = hashlib.sha1(str(cfg.current_study().dir).encode("utf-8")).hexdigest()[:8]
    return f"{name}__{tag}"


def _autoheight(text: str, *, min_lines: int) -> int:
    """Pixel height that fits ``text`` (newlines plus soft wraps), floored at
    ``min_lines`` and capped."""
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

    st.session_state.pop(_skey("_sum_saved_snapshot"), None)


def _draft_refinement(r) -> None:
    """on_click: draft a celltype_override from a holistic refinement, reconciled
    against the literature, for the two-tap confirm card (trigger=holistic_review)."""
    from panoscope.agent import memory
    from panoscope.ui import state

    lit = None
    try:
        from panoscope.agent import tools

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
    """Render one editable region: canonical text in ``wsval_{name}`` (survives
    unmount), seeding the ``wsw_{name}`` widget."""
    vkey = _skey(f"wsval_{name}")
    if vkey not in st.session_state:
        st.session_state[vkey] = default
    val = st.text_area("edit", value=st.session_state[vkey], key=_skey(f"wsw_{name}"),
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


def _render_lab_note_cards(st, notes: list) -> None:
    if not notes:
        st.markdown('<div class="pano-lk-empty">No notes yet. Override or confirm a call in the '
                    "chat and it is saved here with its basis and any literature tension.</div>",
                    unsafe_allow_html=True)
        return
    for n in sorted(notes, key=lambda x: (x.created_at, x.id), reverse=True):
        scope = report.note_scope_label(n)
        date = n.created_at.split("T", 1)[0] if n.created_at else "n/a"
        st.markdown(
            f'<div class="pano-lk-card"><div class="pano-lk-claim">{html.escape(n.claim)}</div>'
            f'<div class="pano-lk-meta"><span class="sco">{html.escape(scope)}</span> · '
            f'basis: {html.escape(n.basis)} · {html.escape(n.status)} · '
            f'{html.escape(n.author or "you")} · {html.escape(date)}</div></div>',
            unsafe_allow_html=True,
        )


def _render_withheld_notes(st, notes: list) -> None:
    """The notes this study must NOT apply (no study named; fail-closed), said out
    loud instead of silently dropped. Renders nothing in the normal case."""
    if not notes:
        return
    st.markdown(
        '<div class="pano-ed-eyebrow" style="margin-top:16px">Held back · names no study</div>'
        '<div class="pano-ed-hint" style="margin:0 0 10px">Stored, but not applied here: '
        "a note that names no dataset cannot be assumed to belong to this one. Add its "
        "study to the file and it returns to the board.</div>",
        unsafe_allow_html=True,
    )
    for n in sorted(notes, key=lambda x: (x.created_at, x.id), reverse=True):
        date = n.created_at.split("T", 1)[0] if n.created_at else "n/a"
        st.markdown(
            f'<div class="pano-lk-card"><div class="pano-lk-claim">{html.escape(n.claim)}</div>'
            f'<div class="pano-lk-meta"><span class="sco">no study named</span> · '
            f'note:{html.escape(n.id)} · {html.escape(n.author or "you")} · '
            f"{html.escape(date)}</div></div>",
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

    # No readable method output is a message, not a traceback.
    try:
        verdicts = da.composed_verdicts()  # overrides/excludes reflected everywhere
    except Exception as exc:  # noqa: BLE001 - no method output is a message, not a crash
        st.warning(
            "This study has no readable method output, so there is nothing to "
            f"review or sign off here yet: {exc}"
        )
        return
    overrides = {v.cluster: da.override_info(v.cluster) for v in verdicts}
    overrides = {c: o for c, o in overrides.items() if o}
    sec_by_id = {v.cluster: v for v in verdicts}
    refinements = _refinements_by_cluster()
    # Dispatch a table click before reading sign-off state.
    _handle_table_actions(st, verdicts, overrides, refinements)
    # Re-read: a sign/undo click changes ``verify`` on the composed verdicts.
    verdicts = da.composed_verdicts()
    sec_by_id = {v.cluster: v for v in verdicts}
    # Sign-off state drives the board triage AND the header finish line.
    signed = da.signed_off()
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

    # Latest per-section drafts, computed up front. Guarded like ``n_panel``: a
    # missing input is a message, not a traceback.
    try:
        rep = report.build_report_from_sources(
            generated_at=datetime.date.today().isoformat())
    except Exception as exc:  # noqa: BLE001 - say what is missing; do not die
        st.warning(
            "This study is missing an input the write-up needs, so the report and "
            f"its exports are unavailable: {exc}"
        )
        st.markdown(_table_html(verdicts, overrides, signed, refinements),
                    unsafe_allow_html=True)
        return
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
        from panoscope.pipeline import store

        saved = store.load_summary_edits(rep.dataset)
    except Exception:  # noqa: BLE001 - no tree / import issue -> just use auto-drafts
        store, saved = None, {}

    def _seed(name: str) -> str:
        return saved.get(name) or ws_defaults[name]

    if not st.session_state.get(_skey("_sum_seeded")):
        st.session_state[_skey("_sum_seeded")] = True
        for name in ws_defaults:
            st.session_state.setdefault(_skey(f"wsval_{name}"), _seed(name))
            # Remember which draft each region was seeded from, so an untouched
            # region is never mistaken for an edit when the draft changes.
            st.session_state.setdefault(_skey(f"wsseed_{name}"), _seed(name))

    # A region still identical to the draft it was seeded from was never edited, so
    # re-seed it from the current draft rather than freezing the stale one.
    for name in ws_defaults:
        vkey, skey_ = _skey(f"wsval_{name}"), _skey(f"wsseed_{name}")
        if (skey_ in st.session_state
                and st.session_state.get(vkey) == st.session_state[skey_]
                and st.session_state[skey_] != ws_defaults[name]
                and name not in saved):
            st.session_state[vkey] = ws_defaults[name]
            st.session_state[skey_] = ws_defaults[name]
            st.session_state.pop(_skey(f"wsw_{name}"), None)

    # Reconcile canonical text from the (possibly just-edited) mounted widget BEFORE
    # exports/autosave, so both always reflect the latest keystrokes.
    for name in ws_defaults:
        wkey = _skey(f"wsw_{name}")
        if wkey in st.session_state:
            st.session_state[_skey(f"wsval_{name}")] = st.session_state[wkey]

    def _val(name: str) -> str:
        return st.session_state.get(_skey(f"wsval_{name}"), _seed(name))

    # Autosave only regions changed from the auto-draft, only when the edit set
    # changed. ``edits or saved``: viewing never writes; reverting all writes ``{}``.
    edits = {n: _val(n) for n in ws_defaults if _val(n) != ws_defaults[n]}
    save_error = ""
    if (store is not None and (edits or saved)
            and edits != st.session_state.get(_skey("_sum_saved_snapshot"))):
        try:
            now = datetime.datetime.now()
            store.save_summary_edits(edits, rep.dataset,
                                     saved_at=now.isoformat(timespec="seconds"))
            st.session_state[_skey("_sum_saved_snapshot")] = dict(edits)
            st.session_state[_skey("_sum_saved_at")] = now.strftime("%H:%M")
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
    elif (edits == st.session_state.get(_skey("_sum_saved_snapshot"))
            and st.session_state.get(_skey("_sum_saved_at"))):
        status = f'saved {html.escape(st.session_state[_skey("_sum_saved_at")])}'
    else:
        status = "editing…"

    # ---- Top bar: title + meta + save status | [Report] [CSV] [Save] ----- #
    head_col, act_col = st.columns([0.60, 0.40], vertical_alignment="center")
    ctx = " · ".join(p for p in (rep.tissue, rep.platform) if p) or rep.dataset
    with head_col:
        st.markdown(
            f'<div class="pano-sum-title">{html.escape(_TITLE)}</div>'
            f'<div class="pano-sum-sub">{html.escape(ctx)} · every number is '
            "jazzPanda's, every biology claim live-cited.</div>",
            unsafe_allow_html=True,
        )
    # Export needs a terminal decision on every cluster (interface §5.2); a Summary
    # sign-off counts as accepting the call shown.
    from panoscope.pipeline import decisions as dec

    refusal = dec.export_refusal(cfg.CLUSTER_ORDER, da.decisions())
    if refusal:
        act_col.caption(refusal)
    else:
        with act_col:
            b_doc, b_pdf = st.columns(2)
            _dl_help = ("The interpretation report: a confirmed-call overview table, per-cluster "
                        "detail with programs, the cross-cluster review, and caveats.")
            with b_doc:
                st.download_button(
                    _DOCX_LABEL, report.working_docx(export_sections, report_model=rep, **exp_kw),
                    _DOCX_NAME, _DOCX_MIME, key="dl_docx", width="stretch",
                    help=_dl_help + " Editable Word.")
            with b_pdf:
                st.download_button(
                    _PDF_LABEL, report.working_pdf(export_sections, report_model=rep, **exp_kw),
                    _PDF_NAME, _PDF_MIME, key="dl_pdf", width="stretch",
                    help=_dl_help + " Locked PDF.")

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ---- Section sub-nav (the clusters live IN the table now, not a rail) --- #
    with st.container(key="pano_subnav"):
        nav_cols = st.columns(3)
        for col, (sec, label) in zip(
            nav_cols,
            ((_SEC_OVERALL, "Overall"), (_SEC_CAVEATS, "Caveats"), (_SEC_NOTES, "My notes")),
        ):
            with col:
                st.button(label, key=f"snav_{sec}", width="stretch",
                          type="primary" if active == sec else "secondary",
                          on_click=_set_active, args=(sec,))

    with st.container(key="pano_pane"):
        if active == _SEC_OVERALL:
            bullets = _overview_bullets_html(ws_defaults.get(_ED_GLOBAL, ""))
            if bullets:
                st.markdown(bullets, unsafe_allow_html=True)
            st.markdown(_lead_html(n_signed, len(verdicts), n_needs), unsafe_allow_html=True)
            with st.container(key="pano_tblcard"):
                st.markdown(_table_html(verdicts, overrides, signed, refinements),
                            unsafe_allow_html=True)

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

            _render_drilldown_actions(st, s, overrides.get(s.cluster),
                                      signed.get(s.cluster), refinements.get(s.cluster))
            _render_signoff_card(st, s.cluster)
            if s.cluster in refinements:
                from panoscope.ui import conversation as convo

                convo._render_draft_card(s.cluster, thread_key=f"holistic::{s.cluster}",
                                         trigger="holistic_review")

            # Evidence first, then the editable write-up.
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
            _render_withheld_notes(st, da.withheld_notes())
