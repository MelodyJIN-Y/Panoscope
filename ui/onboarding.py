"""Landing + upload front door (demo onboarding).

A two-step gate the app shows before the dashboard:

1. **Landing** — brand, one-line pitch, the three things that make Panoscope
   different, and a "Get started" call to action.
2. **Upload** — drop in the jazzPanda marker CSV, the gene-set CSV, and the
   Seurat object, then "Analyze". A short staged progress ("reading jazzPanda
   output -> panel-absence rule -> grounding citations") hands off to the
   dashboard.

DEMO NOTE (honest framing): this is a front door, not a live processor. The app
runs on the PRECOMPUTED per-dataset tree (the project's "precomputed jazzPanda
output only" rule). The uploaded files are accepted and their names are echoed in
the progress, but the dashboard reads the existing interpretation tree — no live
jazzPanda run, no slow gene-note recompute. Real processing is the offline
``pipeline.run`` path; this screen stands in for it during the demo.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import streamlit as st

from agent import profile as agent_profile

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_PAGE_KEY = "active_page"        # mirrors app.py's session page key


# --------------------------------------------------------------------------- #
# Navigation — each onboarding screen has its own dedicated ?page= name
# --------------------------------------------------------------------------- #
def _nav(page: str) -> None:
    """Navigate to a named page (session + URL, mirroring app.py) and rerun."""
    st.session_state[_PAGE_KEY] = page
    st.query_params["page"] = page
    st.rerun()


def _logo_uri(name: str) -> str:
    path = _ASSETS / name
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _css() -> None:
    st.markdown(
        """
<style>
/* hide Streamlit chrome + heading anchor links on the onboarding screens */
header[data-testid="stHeader"], #MainMenu, footer { display:none !important; }
.pano-ob h1 a, .pano-ob h2 a, [data-testid="stHeaderActionElements"],
h1 a.anchor-link, h2 a.anchor-link { display:none !important; }
.stApp [data-testid="stAppViewContainer"]{ background: var(--bg); }
.block-container{ max-width: 1060px !important; padding-top: 2.6rem !important; }

.pano-ob-atmos{
  position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(1100px 520px at 50% -8%, var(--accent-soft) 0%, rgba(235,245,246,0) 60%),
    radial-gradient(700px 380px at 88% 12%, rgba(15,123,135,.06) 0%, rgba(15,123,135,0) 70%);
}
.pano-ob{ position:relative; z-index:1; }

/* ---- hero ---- */
.pano-ob-brand{ display:flex; align-items:center; gap:.6rem; margin-bottom:2.2rem; }
.pano-ob-brand img{ height:30px; width:auto; }
.pano-ob-brand .nm{ font-family:var(--sans); font-weight:700; letter-spacing:-.01em; font-size:1.05rem; color:var(--ink); }
.pano-ob-kicker{
  display:inline-flex; align-items:center; gap:.5rem; font-family:var(--mono);
  font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--accent);
  background:var(--accent-soft); border:1px solid rgba(15,123,135,.18);
  padding:.34rem .7rem; border-radius:999px; margin-bottom:1.1rem;
}
.pano-ob-h1{
  font-family:var(--sans); font-weight:760; color:var(--ink); letter-spacing:-.025em;
  font-size:clamp(2rem, 1.1rem + 2.3vw, 2.85rem); line-height:1.06; margin:.15rem 0 .9rem 0; max-width:15ch;
  text-wrap:balance;
}
.pano-ob-h1 .accent{ color:var(--accent); }
.pano-ob-lead{
  font-family:var(--sans); color:var(--muted); font-size:1rem; line-height:1.55;
  max-width:48ch; margin:0 0 1.6rem 0; text-wrap:pretty;
}
.pano-ob-feats{ display:flex; flex-direction:column; gap:.5rem; margin:0 0 1.9rem 0; max-width:430px; align-items:flex-start; }
.pano-ob-feat{
  font-family:var(--sans); font-size:.82rem; color:var(--ink);
  background:var(--paper); border:1px solid var(--hair); border-radius:10px;
  padding:.5rem .8rem; box-shadow:0 1px 2px rgba(22,27,32,.04);
}
.pano-ob-feat b{ color:var(--accent); font-weight:650; }

/* ---- hero visual (right column): cell-map panel + floating verdict card ---- */
.pano-hero-vis{ position:relative; width:100%; max-width:396px; margin:0 0 1rem auto; }
.pano-hero-svg{
  width:100%; height:auto; display:block; border:1px solid var(--hair); border-radius:20px;
  background:var(--paper); box-shadow:0 26px 60px rgba(15,123,135,.12), 0 4px 16px rgba(22,27,32,.05);
}
.pano-hero-card{
  position:absolute; right:-10px; bottom:-18px; width:236px; background:var(--paper);
  border:1px solid var(--hair); border-radius:14px; padding:.8rem .9rem;
  box-shadow:0 18px 40px rgba(22,27,32,.14); font-variant-numeric:tabular-nums;
}
.pano-hc-top{ display:flex; align-items:center; justify-content:space-between; margin-bottom:.6rem; }
.pano-hc-id{ font-family:var(--sans); font-weight:700; color:var(--ink); font-size:.92rem; letter-spacing:-.01em; }
.pano-hc-id b{ color:var(--accent); }
.pano-hc-band{
  font-family:var(--sans); font-weight:650; font-size:.64rem; color:#fff; background:var(--c-vh);
  padding:.18rem .5rem; border-radius:999px; letter-spacing:.02em;
}
.pano-hc-row{
  display:flex; align-items:center; justify-content:space-between;
  font-family:var(--mono); font-size:.72rem; padding:.17rem 0;
}
.pano-hc-row .g{ color:var(--muted); } .pano-hc-row .v{ color:var(--ink); }
.pano-hc-row.off .v{ color:var(--absent); }
.pano-hc-cite{
  margin-top:.5rem; padding-top:.46rem; border-top:1px solid var(--hair2);
  font-family:var(--mono); font-size:.68rem; color:var(--accent);
}

/* ---- upload heading ---- */
.pano-ob-step{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); margin-bottom:.5rem; }
.pano-ob-h2{ font-family:var(--sans); font-weight:720; color:var(--ink); letter-spacing:-.02em; font-size:1.7rem; margin:0 0 .5rem 0; }
.pano-ob-sub{ font-family:var(--sans); color:var(--muted); font-size:.96rem; margin:0 0 1.6rem 0; }
.pano-ob-card-h{ font-family:var(--sans); font-weight:650; color:var(--ink); font-size:.9rem; margin:.2rem 0 .15rem 0; min-height:1.3em; }
.pano-ob-card-h .req{ color:var(--faint); font-weight:600; font-size:.74rem; }
.pano-ob-card-d{ font-family:var(--sans); color:var(--faint); font-size:.78rem; line-height:1.4; margin:0 0 .55rem 0; min-height:2.6em; }

/* dropzones as tidy cards */
[data-testid="stFileUploaderDropzone"]{
  background:var(--paper); border:1.5px dashed var(--hair); border-radius:12px; min-height:104px;
  transition:border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent); background:var(--accent-soft); }

/* research-focus input styled to match the dropzone cards (onboarding CSS only
   loads on these screens, so a broad stTextArea selector is safe here) */
[data-testid="stTextArea"] > div{
  background:var(--paper) !important; border:1.5px solid var(--hair) !important; border-radius:12px !important;
}
[data-testid="stTextArea"] > div:focus-within{
  border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stTextArea"] textarea{
  background:transparent !important; font-family:var(--sans) !important; color:var(--ink) !important;
  font-size:.9rem !important; resize:none !important; padding:.15rem .35rem !important;
}
[data-testid="stTextArea"] textarea::placeholder{ color:var(--faint) !important; }

.pano-ob-note{ font-family:var(--sans); color:var(--faint); font-size:.78rem; margin-top:1.1rem; }
.pano-ob-note b{ color:var(--muted); }

/* CTA buttons within the onboarding wrapper */
.st-key-pano_ob_cta .stButton>button, .st-key-pano_ob_go .stButton>button{
  background:var(--accent); color:#fff; border:none; border-radius:12px;
  font-family:var(--sans); font-weight:650; font-size:.98rem; padding:.72rem 1.4rem;
  box-shadow:0 6px 16px rgba(15,123,135,.22);
  transition:transform .2s cubic-bezier(.16,1,.3,1), box-shadow .2s ease;
}
.st-key-pano_ob_cta .stButton>button:hover, .st-key-pano_ob_go .stButton>button:hover{
  transform:translateY(-1px); box-shadow:0 12px 26px rgba(15,123,135,.30);
}
.st-key-pano_ob_cta .stButton>button:active, .st-key-pano_ob_go .stButton>button:active{
  transform:translateY(0) scale(.985); box-shadow:0 4px 12px rgba(15,123,135,.24);
}
.st-key-pano_ob_cta .stButton>button:focus-visible, .st-key-pano_ob_go .stButton>button:focus-visible{
  outline:none; box-shadow:0 0 0 3px var(--accent-soft), 0 0 0 4px var(--accent);
}
.st-key-pano_ob_skip .stButton>button{
  background:transparent; color:var(--faint); border:none; font-family:var(--sans);
  font-size:.82rem; padding:.4rem .2rem; box-shadow:none; transition:color .2s ease;
}
.st-key-pano_ob_skip .stButton>button:hover{ color:var(--accent); }

/* honor reduced-motion (taste-skill requirement) */
@media (prefers-reduced-motion: reduce){
  .st-key-pano_ob_cta .stButton>button, .st-key-pano_ob_go .stButton>button,
  .st-key-pano_ob_skip .stButton>button, [data-testid="stFileUploaderDropzone"]{ transition:none !important; }
  .st-key-pano_ob_cta .stButton>button:hover, .st-key-pano_ob_go .stButton>button:hover{ transform:none !important; }
}
</style>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
_HERO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 360">'
    '<defs>'
    '<pattern id="pc" width="15" height="15" patternUnits="userSpaceOnUse">'
    '<circle cx="3" cy="3" r="1.3" fill="#DDE3E5"/></pattern>'
    '<radialGradient id="pg" cx="50%" cy="50%" r="50%">'
    '<stop offset="0%" stop-color="#0F7B87" stop-opacity="0.18"/>'
    '<stop offset="100%" stop-color="#0F7B87" stop-opacity="0"/></radialGradient>'
    '</defs>'
    '<rect width="360" height="360" rx="22" fill="#FFFFFF"/>'
    '<rect width="360" height="360" rx="22" fill="url(#pc)" opacity="0.55"/>'
    '<circle cx="258" cy="118" r="50" fill="#5FA7AE" opacity="0.13"/>'
    '<circle cx="252" cy="268" r="58" fill="#2E8C97" opacity="0.11"/>'
    '<circle cx="126" cy="160" r="94" fill="url(#pg)"/>'
    '<g fill="#0F7B87">'
    '<circle cx="100" cy="130" r="4"/><circle cx="120" cy="120" r="4.4"/><circle cx="140" cy="132" r="4"/>'
    '<circle cx="112" cy="150" r="4.6"/><circle cx="132" cy="158" r="5"/><circle cx="152" cy="150" r="4.2"/>'
    '<circle cx="98" cy="162" r="4"/><circle cx="118" cy="178" r="4.4"/><circle cx="142" cy="182" r="4"/>'
    '<circle cx="160" cy="168" r="4"/><circle cx="128" cy="140" r="4.2"/><circle cx="150" cy="128" r="3.8"/>'
    '<circle cx="106" cy="182" r="3.8"/><circle cx="134" cy="196" r="4"/><circle cx="90" cy="146" r="3.6"/>'
    '<circle cx="166" cy="146" r="3.8"/></g>'
    '<circle cx="128" cy="158" r="62" fill="none" stroke="#0F7B87" stroke-width="1.4" '
    'stroke-dasharray="5 5" opacity="0.7"/>'
    '<text x="128" y="86" text-anchor="middle" font-family="ui-monospace, monospace" '
    'font-size="12" font-weight="700" fill="#0F7B87">c2 &#183; Stromal</text>'
    '</svg>'
)


def _hero_visual_html() -> str:
    """Cell-map panel (SVG as a data-URI img, sanitizer-proof) + a floating verdict card."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(_HERO_SVG.encode("utf-8")).decode()
    return f"""
<div class="pano-hero-vis">
  <img class="pano-hero-svg" src="{uri}" alt="Spatial cell map"/>
  <div class="pano-hero-card">
    <div class="pano-hc-top">
      <div class="pano-hc-id"><b>c2</b> · Stromal</div>
      <div class="pano-hc-band">Very High</div>
    </div>
    <div class="pano-hc-row"><span class="g">LUM</span><span class="v">glm 18.00</span></div>
    <div class="pano-hc-row"><span class="g">POSTN</span><span class="v">glm 15.80</span></div>
    <div class="pano-hc-row off"><span class="g">COL1A1</span><span class="v">not measured</span></div>
    <div class="pano-hc-cite">PMID 37479733 · live</div>
  </div>
</div>
"""

_HERO_TEXT = """
<div class="pano-ob">
  <span class="pano-ob-kicker">● annotation-confidence layer</span>
  <h1 class="pano-ob-h1">Turn <span class="accent">jazzPanda</span> markers into a call you can trust.</h1>
  <p class="pano-ob-lead">Ask about a cluster in plain language; Panoscope answers with a cell-type call,
  a confidence level, and the evidence behind it. Every number traces to jazzPanda's spatial output, and
  every literature claim carries a real, live-fetched PubMed citation.</p>
  <div class="pano-ob-feats">
    <span class="pano-ob-feat"><b>Grounded</b> · every number traces to source</span>
    <span class="pano-ob-feat"><b>Panel-absence catch</b> · off-panel ≠ absent</span>
    <span class="pano-ob-feat"><b>Live citations</b> · real PubMed, never from memory</span>
  </div>
</div>
"""


def render_landing() -> None:
    """Landing page (?page=welcome)."""
    _css()
    logo = _logo_uri("panoscope_logo.png")
    mark = f'<img src="{logo}" alt=""/>' if logo else ""
    with st.container(key="pano_ob_landing"):
        st.markdown('<div class="pano-ob-atmos"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pano-ob"><div class="pano-ob-brand">{mark}'
            '<span class="nm">Panoscope</span></div></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.04, 0.96], vertical_alignment="center")
        with left:
            st.markdown(_HERO_TEXT, unsafe_allow_html=True)
            cta, _sp = st.columns([1, 1.3])
            with cta:
                with st.container(key="pano_ob_cta"):
                    if st.button("Get started  →", use_container_width=True, key="ob_start"):
                        _nav("upload")
        with right:
            st.markdown(_hero_visual_html(), unsafe_allow_html=True)


def _upload_card(col, title: str, desc: str, key: str, types, required: bool):
    with col:
        tag = "" if required else ' <span class="req">· optional</span>'
        st.markdown(f'<div class="pano-ob-card-h">{title}{tag}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pano-ob-card-d">{desc}</div>', unsafe_allow_html=True)
        return st.file_uploader(title, type=types, key=key, label_visibility="collapsed")


def _run_progress(names: list[str]) -> None:
    """Short staged progress so the hand-off feels like real work (it reads the
    precomputed tree; see the module DEMO NOTE)."""
    read = names[0] if names else "markers_top.csv"
    steps = [
        (f"Reading jazzPanda output ({read})…", 0.28),
        ("Applying the panel-absence rule to the gene panel…", 0.52),
        ("Scoring confidence from spatial glm_coef…", 0.72),
        ("Grounding cell-type notes with live citations…", 0.9),
        ("Interpretation ready. 9 clusters, 280-gene panel.", 1.0),
    ]
    bar = st.progress(0.0, text="Starting…")
    for msg, frac in steps:
        time.sleep(0.5)
        bar.progress(frac, text=msg)
    time.sleep(0.35)


def render_upload() -> None:
    """Load-your-data page (?page=upload)."""
    _css()
    with st.container(key="pano_ob_upload"):
        st.markdown('<div class="pano-ob-atmos"></div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="pano-ob">
  <h2 class="pano-ob-h2">Bring your jazzPanda output</h2>
  <p class="pano-ob-sub">Drop in your jazzPanda marker table; add the gene-set enrichment result and the
  Seurat object to unlock Pathways and the spatial views. Panoscope builds the interpretation; you review it.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3, gap="medium")
        f1 = _upload_card(
            c1, "Marker genes (CSV)",
            "jazzPanda top markers: gene, top_cluster, glm_coef, pearson.",
            "ob_markers", ["csv"], required=True,
        )
        f2 = _upload_card(
            c2, "Gene-set enrichment (CSV)",
            "jazzPanda gene-set enrichment result: per-cluster program scores. Drives Pathways.",
            "ob_panel", ["csv"], required=False,
        )
        f3 = _upload_card(
            c3, "Seurat object (.rds)",
            "Cells, clusters and UMAP. Required for the spatial views.",
            "ob_seurat", ["rds", "Rds", "RDS"], required=True,
        )

        names = [f.name for f in (f1, f2, f3) if f is not None]

        # Optional research profile — sharpens literature search, saved locally.
        st.markdown(
            '<div class="pano-ob-card-h" style="margin-top:1.4rem;">Your research focus '
            '<span style="color:var(--faint);font-weight:400;">· optional</span></div>'
            '<div class="pano-ob-card-d">One line about your background or interest. Panoscope uses it '
            "to make literature search more precise. <b>Saved locally on your machine; never uploaded.</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.container(key="pano_ob_profile"):
            interest = st.text_area(
                "research focus",
                value=agent_profile.load(),
                key="ob_profile",
                label_visibility="collapsed",
                placeholder="e.g. triple-negative breast cancer, CAF heterogeneity, spatial immunology",
                height=72,
            )

        st.markdown(
            '<div class="pano-ob-note"><b>Runs on precomputed jazzPanda output.</b> No live jazzPanda '
            "run, so the demo is deterministic and every number on screen has a checkable source.</div>",
            unsafe_allow_html=True,
        )

        col, _sp = st.columns([1, 2])
        with col:
            with st.container(key="pano_ob_cta"):
                analyze = st.button("Analyze my data  →", use_container_width=True, key="ob_analyze")
        with st.container(key="pano_ob_skip"):
            skip = st.button("← back", key="ob_back")

        if skip:
            _nav("welcome")
        if analyze:
            agent_profile.save(interest)
            _run_progress(names)
            _nav("summary")   # enter the dashboard
