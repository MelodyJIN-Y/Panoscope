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
.pano-brand {{ font-family: var(--sans); font-weight: 700; font-size: 15px; }}
.pano-brand .d {{ color: var(--accent); }}
.pano-ctx {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
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

/* ---- density legend grad (area-normalized teal ramp) ---- */
.grad {{
  display: inline-block; width: 70px; height: 8px; border-radius: 3px;
  background: linear-gradient(90deg, #EAF3F4, var(--c-vh));
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
"""


def inject_css() -> None:
    """Inject the Panoscope stylesheet into the current Streamlit page.

    Re-injecting the same CSS across reruns is harmless. Imports Streamlit
    lazily so the module stays import-safe with no server.
    """
    import streamlit as st

    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


__all__ = ["TOKENS", "inject_css"]
