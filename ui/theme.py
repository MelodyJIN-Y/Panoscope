"""Design-token CSS for the Panoscope UI.

``inject_css()`` writes one ``<style>`` block into the Streamlit page. Tokens
are lifted verbatim from the wireframe (``dashboard_wireframe_panels.html``):
teal accent ``#0F7B87``, the confidence bands (Very-High -> Low), the three
role-chip colors (support / expected-absent / off-panel), and the Space Grotesk
+ IBM Plex Mono pairing. Fonts load via a Google Fonts ``@import`` with a full
system fallback stack, so the app still renders if fonts are blocked.

The chip class names (``.cf-vh``…, ``.role-sup``…, ``.src-jz``…) are the exact
classes ``ui.format`` returns, so a formatter output drops straight into a
styled span. The 3-pane grid tokens match the wireframe shell
(rail | evidence+spatial stage | conversation).

Streamlit is imported lazily inside ``inject_css`` so importing ``ui.theme``
never requires a running server.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Design tokens (single source; mirrors the wireframe :root block)
# --------------------------------------------------------------------------- #
TOKENS: dict[str, str] = {
    # surfaces / ink
    "bg": "#F3F4F6",
    "paper": "#FFFFFF",
    "ink": "#161B20",
    "muted": "#606A73",
    "faint": "#9AA3AB",
    "hair": "#E4E7EA",
    "hair2": "#EEF0F2",
    # brand
    "accent": "#0F7B87",
    "accent_soft": "#EBF5F6",
    # role chips
    "support": "#1F6FEB",
    "support_bg": "#EAF1FE",
    "absent": "#BE7A1E",
    "absent_bg": "#FAF1E1",
    "offpanel": "#98A0A7",
    "offpanel_bg": "#F0F2F3",
    # confidence bands
    "c_vh": "#0F5B65",
    "c_h": "#2E8C97",
    "c_mh": "#5FA7AE",
    "c_m": "#A9C7CB",
    "c_l": "#E2E6E7",
    "c_l_ink": "#5A6167",
    # panes
    "rail_w": "222px",
    "conv_w": "372px",
    "header_h": "52px",
    # type
    "sans": '"Space Grotesk",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
    "mono": '"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace',
}

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600&display=swap');"
)


def _css() -> str:
    """Return the full stylesheet as a string (pure; no Streamlit)."""
    t = TOKENS
    return f"""
{_FONT_IMPORT}

:root {{
  --bg:{t['bg']}; --paper:{t['paper']}; --ink:{t['ink']}; --muted:{t['muted']};
  --faint:{t['faint']}; --hair:{t['hair']}; --hair2:{t['hair2']};
  --accent:{t['accent']}; --accent-soft:{t['accent_soft']};
  --support:{t['support']}; --support-bg:{t['support_bg']};
  --absent:{t['absent']}; --absent-bg:{t['absent_bg']};
  --offpanel:{t['offpanel']}; --offpanel-bg:{t['offpanel_bg']};
  --c-vh:{t['c_vh']}; --c-h:{t['c_h']}; --c-mh:{t['c_mh']}; --c-m:{t['c_m']}; --c-l:{t['c_l']};
  --rail-w:{t['rail_w']}; --conv-w:{t['conv_w']}; --header-h:{t['header_h']};
  --sans:{t['sans']}; --mono:{t['mono']};
}}

/* ---- base ---- */
html, body, .stApp {{
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}}
.stApp [data-testid="stAppViewContainer"] {{ background: var(--bg); }}
.mono, .pano-mono {{ font-family: var(--mono); }}

/* Tighten Streamlit's default block padding so the 3-pane shell fills the width */
.block-container {{ padding: 12px 20px 24px; max-width: 100%; }}

/* ---- brand / header ---- */
.pano-brand {{
  font-family: var(--sans); font-weight: 700; font-size: 16px;
  letter-spacing: -.01em; display: flex; align-items: center; color: var(--ink);
}}
.pano-brand .d {{ color: var(--accent); }}
.pano-mark {{
  display: inline-block; width: 18px; height: 18px; border-radius: 5px; flex: none;
  background: linear-gradient(135deg, var(--accent) 0%, #0B5B65 100%);
  margin-right: 9px;
}}
/* The real Panoscope logo mark (assets/panoscope_logo.png), embedded inline. */
.pano-logo {{ width: 30px; height: 30px; margin-right: 10px; flex: none; object-fit: contain; }}
.pano-ctx {{ font-family: var(--mono); font-size: 10.5px; color: var(--faint); margin-top: 3px; }}
/* Right-aligned dataset context — a bordered pill with a live-status dot. */
.pano-ctx-wrap {{ display: flex; justify-content: flex-end; }}
.pano-ctx-chip {{
  display: inline-flex; align-items: center; gap: 9px;
  background: var(--paper); border: 1px solid var(--hair);
  border-radius: 10px; padding: 6px 12px;
}}
.pano-ctx-chip::before {{
  content: ''; width: 7px; height: 7px; border-radius: 50%; flex: none;
  background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
}}
.pano-ctx-text {{ display: flex; flex-direction: column; line-height: 1.35; }}
.pano-ctx-main {{ font-family: var(--mono); font-size: 11px; color: var(--ink); font-weight: 600; }}
.pano-ctx-sub {{ font-family: var(--mono); font-size: 10px; color: var(--faint); }}
/* App bar — a real header strip with a hairline under it. */
.st-key-pano_appbar {{ border-bottom: 1px solid var(--hair); padding: 4px 0 11px; margin-bottom: 14px; }}
.pano-eyebrow {{
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); font-weight: 500; margin: 0 0 12px;
}}
.pano-sect {{
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); font-weight: 500;
  margin: 0 0 12px; display: flex; align-items: center; gap: 10px;
}}
.pano-sect .r {{ margin-left: auto; text-transform: none; letter-spacing: 0; color: var(--faint); }}

/* ---- 3-pane shell (rail | stage | conversation) ---- */
.pano-shell {{
  display: grid;
  grid-template-columns: var(--rail-w) 1fr var(--conv-w);
  gap: 0;
  min-height: calc(100vh - var(--header-h));
}}
.pano-rail {{ background: var(--paper); border-right: 1px solid var(--hair); padding: 16px 12px; }}
.pano-center {{ padding: 8px 24px; }}
.pano-conv {{ background: var(--paper); border-left: 1px solid var(--hair); padding: 12px 14px; }}
@media (max-width: 1120px) {{ .pano-shell {{ grid-template-columns: 200px 1fr 330px; }} }}
@media (max-width: 860px)  {{ .pano-shell {{ grid-template-columns: 1fr; }} }}

/* ---- verdict header ---- */
.pano-idline {{ font-family: var(--mono); font-size: 11px; color: var(--faint); margin-bottom: 6px; }}
.pano-verdict {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; flex-wrap: wrap; }}
.pano-verdict h1 {{ font-size: 25px; font-weight: 600; letter-spacing: -.02em; margin: 0; }}
.pano-rat {{ color: var(--muted); max-width: 62ch; margin-bottom: 6px; }}
.pano-hint {{
  font-family: var(--mono); font-size: 11px; color: var(--faint);
  max-width: 64ch; margin: 2px 0 14px; line-height: 1.5;
}}
.pano-hint a {{ color: var(--accent); }}

/* ---- confidence chips (classes returned by ui.format.confidence_chip) ---- */
.cf {{
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  padding: 4px 9px; border-radius: 6px; color: #fff; white-space: nowrap;
  display: inline-block;
}}
.cf-vh {{ background: var(--c-vh); }}
.cf-h  {{ background: var(--c-h); }}
.cf-mh {{ background: var(--c-mh); }}
.cf-m  {{ background: var(--c-m); color: {t['ink']}; }}
.cf-l  {{ background: var(--c-l); color: {t['c_l_ink']}; }}

/* ---- verify flag ---- */
.pano-verify {{
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  color: var(--absent); background: var(--absent-bg);
  padding: 3px 8px; border-radius: 6px; white-space: nowrap;
}}

/* ---- role chips (classes returned by ui.format.role_chip) ---- */
.role {{
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 11px; padding: 3px 9px;
  border-radius: 20px; white-space: nowrap;
}}
.role-sup {{ background: var(--support-bg); color: var(--support); }}
.role-abs {{ background: var(--absent-bg);  color: var(--absent); }}
.role-off {{ background: var(--offpanel-bg); color: var(--offpanel); }}

/* ---- evidence numbers ---- */
.num {{ font-family: var(--mono); font-size: 13px; }}
.num.dim {{ color: var(--faint); }}
.gene {{ font-family: var(--mono); font-weight: 600; font-size: 14px; }}
.pin {{ color: var(--accent); margin-left: 7px; font-size: 11px; }}

/* ---- source chips (chat) ---- */
.srcchip {{ font-family: var(--mono); font-size: 10px; padding: 3px 8px; border-radius: 5px; }}
.src-jz  {{ background: var(--accent-soft); color: var(--accent); }}
.src-panel {{ background: var(--accent-soft); color: var(--accent); }}
.src-lit {{ background: var(--offpanel-bg); color: var(--muted); }}
.src-mem {{ background: var(--absent-bg); color: var(--absent); }}

/* ---- chat bubbles ---- */
.bubble {{ padding: 11px 13px; border-radius: 11px; font-size: 13px; line-height: 1.5; }}
.bubble.a   {{ background: #F5FAFB; }}
.bubble.u   {{ background: var(--support-bg); }}
.bubble.sys {{ background: var(--absent-bg); font-family: var(--mono); font-size: 11px; color: #7a5b1e; }}
.who {{
  font-family: var(--mono); font-size: 9px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--faint); margin-bottom: 4px;
}}

/* ---- citation + tension ---- */
.pcite {{
  color: var(--accent); cursor: pointer;
  border-bottom: 1px dotted var(--accent); white-space: nowrap;
}}
.tension {{
  margin-top: 8px; border-left: 2px solid var(--absent);
  padding: 4px 0 4px 10px; font-size: 12px; color: var(--muted);
}}

/* ---- density/expression legend grad (plasma: low dark purple -> high bright yellow) ---- */
.grad {{
  display: inline-block; width: 64px; height: 8px; border-radius: 3px;
  background: linear-gradient(90deg,
    #0D0887 0%, #7E03A8 20%, #CC4778 40%, #F1605D 60%, #FCA636 80%, #F0F921 100%);
}}

/* ---- accent buttons (primary actions) ---- */
.stButton > button[kind="primary"],
.stButton > button.pano-go {{
  background: var(--accent); color: #fff; border-color: var(--accent);
}}
/* --- demo chrome + Streamlit 1.59 primary button (post-review polish) --- */
[data-testid="stToolbar"], [data-testid="stDecoration"],
header[data-testid="stHeader"], #MainMenu, footer {{ display: none !important; }}
[data-testid="stAppViewContainer"] .block-container,
.main .block-container {{ padding-top: 1.3rem !important; }}
button[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"],
button[kind="primary"] {{
  background: var(--accent) !important; color: #fff !important;
  border-color: var(--accent) !important;
}}

/* ===================================================================== *
 * MINIMAL / BORDERLESS CONTROLS (redesign)
 *
 * De-chroming is opt-in and scoped: wrap ONLY the evidence-table select
 * dots and the cluster-rail rows in a `.pano-dechrome` container so the
 * Confirm / Ask / Save primary buttons keep their chrome. Streamlit
 * renders every button inside `div[data-testid="stButton"] > button`, so
 * every rule below is namespaced under a wrapper class and never leaks.
 * ===================================================================== */

/* (1) SELECT-DOT — a borderless button whose whole surface is a single
 * teal `●` (selected) or muted `○` (add). Zero border / background / box;
 * the glyph is the affordance. Used per evidence row and any dot toggle.
 * The button *label* carries the `●`/`○` glyph, so we color the button
 * text and strip all chrome. Wrap the dot's button in `.pano-select-dot`
 * (single control) or `.pano-dechrome` (a region of such controls). */
.pano-dechrome div[data-testid="stButton"] > button,
.pano-select-dot div[data-testid="stButton"] > button {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 2px 4px !important;
  min-height: 0 !important;
  line-height: 1 !important;
  border-radius: 6px !important;
  color: var(--faint) !important;             /* ○ add — muted by default */
  font-size: 15px !important;
  transition: color 140ms ease, background 140ms ease;
}}
.pano-dechrome div[data-testid="stButton"] > button:hover,
.pano-select-dot div[data-testid="stButton"] > button:hover {{
  color: var(--accent) !important;            /* previewing a pick reads teal */
  background: var(--accent-soft) !important;
}}
.pano-dechrome div[data-testid="stButton"] > button:focus-visible,
.pano-select-dot div[data-testid="stButton"] > button:focus-visible {{
  outline: 2px solid var(--accent) !important; outline-offset: 1px;
}}
/* Selected dot: the wrapper adds `.is-selected` so the glyph locks teal
 * regardless of the button's own label/hover state. */
.pano-select-dot.is-selected div[data-testid="stButton"] > button,
.pano-dechrome .is-selected div[data-testid="stButton"] > button {{
  color: var(--accent) !important;
}}
/* Robust variant: each dot lives in a real st.container(key="seldot_<on|off>_<gene>")
 * so these class-substring rules actually reach the button. `_off_` = muted ○,
 * `_on_` = teal ●. This is the mechanism the evidence table uses. */
/* Descendant `button` (not `> button`): the dot uses help= which inserts tooltip
 * wrapper spans between div[stButton] and the button, breaking a direct-child rule. */
div[class*="st-key-seldot_"] button {{
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  padding: 0 !important; min-height: 0 !important; line-height: 1 !important;
  border-radius: 6px !important; color: var(--faint) !important;
  font-size: 16px !important; transition: color 140ms ease;
}}
div[class*="st-key-seldot_"] button:hover {{
  color: var(--accent) !important; background: var(--accent-soft) !important;
}}
/* Selected dot = type="primary" (stable key; selection carried by type, not the
 * container class). Override the global solid-teal primary back to a bare glyph. */
div[class*="st-key-seldot_"] button[kind="primary"],
div[class*="st-key-seldot_"] button[data-testid="stBaseButton-primary"] {{
  color: var(--accent) !important; background: transparent !important;
  border: 0 !important; box-shadow: none !important;
}}

/* (2) CLUSTER-RAIL — a light, borderless clickable list. Rendered inside a real
 * st.container(key="pano_rail") -> `.st-key-pano_rail`, which genuinely wraps its
 * child buttons (a markdown <div> block auto-closes empty and never contains the
 * later widgets). Each row is a confidence dot + a chromeless cell-type button;
 * the selected row uses type="primary", restyled here to a LIGHT accent (not the
 * solid-teal primary), scoped tightly enough to beat the global primary rule. */
.st-key-pano_rail {{ display: flex; flex-direction: column; gap: 1px; }}
/* One chromeless, LEFT-aligned button per cluster. The colour dot is the button's
 * ::before (coloured per cluster via the st-key-rail_cN class below), so dot + name
 * sit on one line, aligned. Shrinking the label child (flex 0 1 auto, width auto)
 * lets justify-content:flex-start pack dot+name left. Selected row = a simple
 * accent-soft tint (like a selected gene row), no left bar/bracket.
 * NB: the per-cluster dot colours mirror ui/format.py CLUSTER_COLORS. */
.st-key-pano_rail div[data-testid="stButton"] > button {{
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  min-height: 0 !important; padding: 6px 10px !important; border-radius: 7px !important;
  font-family: var(--sans) !important; font-size: 13px !important; font-weight: 500 !important;
  color: var(--ink) !important;
  display: flex !important; align-items: center !important; justify-content: flex-start !important;
  transition: background 140ms ease;
}}
.st-key-pano_rail div[data-testid="stButton"] > button > div {{
  flex: 0 1 auto !important; width: auto !important;
  justify-content: flex-start !important; text-align: left !important;
}}
.st-key-pano_rail div[data-testid="stButton"] > button::before {{
  content: ''; width: 9px; height: 9px; border-radius: 50%; flex: none;
  margin-right: 9px; background: var(--faint);
}}
.st-key-rail_c1 button::before {{ background: #FC8D62 !important; }}
.st-key-rail_c2 button::before {{ background: #66C2A5 !important; }}
.st-key-rail_c3 button::before {{ background: #8DA0CB !important; }}
.st-key-rail_c4 button::before {{ background: #E78AC3 !important; }}
.st-key-rail_c5 button::before {{ background: #A6D854 !important; }}
.st-key-rail_c6 button::before {{ background: #87CEEB !important; }}
.st-key-rail_c7 button::before {{ background: #7D26CD !important; }}
.st-key-rail_c8 button::before {{ background: #E5C498 !important; }}
.st-key-rail_c9 button::before {{ background: #0000FF !important; }}
.st-key-pano_rail div[data-testid="stButton"] > button:hover {{ background: var(--hair2) !important; }}
.st-key-pano_rail div[data-testid="stButton"] > button[kind="primary"],
.st-key-pano_rail button[data-testid="stBaseButton-primary"] {{
  background: var(--accent-soft) !important; color: var(--accent) !important; font-weight: 700 !important;
}}
.st-key-pano_rail div[data-testid="stButton"] > button:focus-visible {{
  outline: 2px solid var(--accent) !important; outline-offset: -2px;
}}

/* (2b) TOP NAV — a compact, centered segmented control; the active tab
 * (type="primary") is a soft-accent pill. Modern, professional, app-like. */
.st-key-pano_topnav {{
  max-width: 440px; margin: 0 auto;
  background: var(--paper); border: 1px solid var(--hair);
  border-radius: 11px; padding: 3px;
}}
.st-key-pano_topnav [data-testid="stHorizontalBlock"] {{ gap: 3px; }}
.st-key-pano_topnav div[data-testid="stButton"] > button {{
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  border-radius: 8px !important; padding: 7px 12px !important; min-height: 0 !important;
  font-family: var(--sans) !important; font-size: 13px !important; font-weight: 500 !important;
  color: var(--muted) !important; transition: background 120ms ease, color 120ms ease;
}}
.st-key-pano_topnav div[data-testid="stButton"] > button:hover {{
  color: var(--ink) !important; background: var(--hair2) !important;
}}
.st-key-pano_topnav div[data-testid="stButton"] > button[kind="primary"],
.st-key-pano_topnav button[data-testid="stBaseButton-primary"] {{
  background: var(--accent-soft) !important; color: var(--accent) !important;
  border-radius: 8px !important; font-weight: 600 !important;
}}

/* (3) CONDENSED SOURCES LINE — one small muted mono line under a chat turn,
 * replacing the per-number chip stack. Numbers stay in prose; this line
 * only names the provenance (jazzPanda · PubMed · lab note). */
.pano-sources {{
  font-family: var(--mono); font-size: 10px; color: var(--faint);
  letter-spacing: .02em; margin-top: 7px; line-height: 1.5;
  display: flex; flex-wrap: wrap; gap: 5px 8px; align-items: baseline;
}}
.pano-sources .lbl {{ text-transform: uppercase; letter-spacing: .08em; }}
.pano-sources .sep {{ color: var(--hair); }}      /* faint middot separator */
.pano-sources .src {{ color: var(--muted); white-space: nowrap; }}
.pano-sources a {{ color: var(--accent); text-decoration: none; }}
.pano-sources a:hover {{ text-decoration: underline; }}

/* (4) QUIET CAPTIONS — recede Streamlit's default caption chrome so
 * whitespace carries the layout instead of grey helper text. Captions kept
 * (scope/basis/status labels) stay legible but visually recessive. */
[data-testid="stCaptionContainer"], .stCaption, .pano-quiet-caption {{
  color: var(--faint) !important;
  font-size: 10px !important;
  font-family: var(--mono) !important;
  text-transform: uppercase; letter-spacing: .08em;
  margin: 2px 0 !important;
}}
"""


def inject_css() -> None:
    """Inject the Panoscope stylesheet into the current Streamlit page.

    Re-injecting the same CSS across reruns is harmless. Imports Streamlit
    lazily so the module stays import-safe with no server.
    """
    import streamlit as st

    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


__all__ = ["TOKENS", "inject_css"]
