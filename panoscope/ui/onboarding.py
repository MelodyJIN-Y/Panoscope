"""Landing + upload front door (demo onboarding).

A two-step gate the app shows before the dashboard:

1. **Landing** — brand, one-line pitch, the three things that make Panoscope
   different, and a "Get started" call to action.
2. **Upload** — a results-folder path box. Panoscope links the folder's files into
   a study under the user data root, recognises each through the method registry,
   shows the §6.4 intake checklist, asks only what is missing, and builds the study
   deterministically (``pipeline.run``) for this session. "Explore the demo" opens
   the bundled precomputed study instead.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import shutil
import time
from pathlib import Path

import streamlit as st

from panoscope import methods, platforms
from panoscope.agent import intake
from panoscope.agent import loop as agent_loop
from panoscope.data import layout
from panoscope.data import study as study_mod
from panoscope.data.study import Study
from panoscope.reference import sources as ref_sources
from panoscope.reference import view as ref_view
from panoscope.types.contracts import UndeclaredClustering

from panoscope.agent import profile as agent_profile
from panoscope.ui import nav, state


def _detected_tissue() -> str:
    """The active dataset's tissue and platform from its manifest ("" if unreadable)."""
    from panoscope.ui import data_access as da

    return " · ".join(p for p in da.tissue_platform() if p)


# --------------------------------------------------------------------------- #
# Navigation — each onboarding screen has its own dedicated ?page= name
# --------------------------------------------------------------------------- #
def _nav(page: str) -> None:
    """Navigate to a named page (session + URL, as app.py does) and rerun."""
    nav.set_page(page)
    st.rerun()


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
.pano-ob-card-d{ font-family:var(--sans); color:var(--faint); font-size:.78rem; line-height:1.4; margin:0 0 .55rem 0; min-height:2.6em; }

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

/* Detected-tissue chip: read-only, grounded from the dataset manifest. */
.pano-ob-detected{
  display:inline-flex; align-items:center; gap:.4rem; margin:.35rem 0 .1rem;
  font-family:var(--sans); font-size:.92rem; font-weight:600; color:var(--ink);
  background:var(--accent-soft); border:1px solid var(--accent); border-radius:999px;
  padding:.28rem .8rem;
}
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
  .st-key-pano_ob_skip .stButton>button{ transition:none !important; }
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
    <div class="pano-hc-cite">cited to real literature</div>
  </div>
</div>
"""

_HERO_TEXT = """
<div class="pano-ob">
  <span class="pano-ob-kicker">● annotation-confidence layer</span>
  <h1 class="pano-ob-h1">Turn <span class="accent">jazzPanda</span> markers into a call you can trust.</h1>
  <p class="pano-ob-lead">Ask about a cluster in plain language; Panoscope answers with a cell-type call,
  a confidence level, and the evidence behind it. Every number traces to jazzPanda's spatial output, and
  every literature claim carries a real PubMed citation, never one from memory.</p>
  <div class="pano-ob-feats">
    <span class="pano-ob-feat"><b>Grounded</b> · every number traces to source</span>
    <span class="pano-ob-feat"><b>Panel-absence catch</b> · off-panel ≠ absent</span>
    <span class="pano-ob-feat"><b>Real citations</b> · looked up in PubMed, never from memory</span>
  </div>
</div>
"""


def render_landing() -> None:
    """Landing page (?page=welcome)."""
    _css()
    logo = nav.logo_data_uri("panoscope_logo.png")
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
                    if st.button("Get started  →", width="stretch", key="ob_start"):
                        _nav("upload")
        with right:
            st.markdown(_hero_visual_html(), unsafe_allow_html=True)


def _ready_line() -> str:
    """The final progress line, with the study's own cluster count and panel size.

    Both are derived from the same readers the plots use, never from the manifest
    or a constant. A number that cannot be derived is left out, never defaulted.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import data as agent_data
    from panoscope.ui import data_access as da

    bits: list[str] = []
    try:
        # Gated on the study's own marker table, so a study without one is never
        # counted from the demo's fallback.
        n_clusters = len(cfg.CLUSTER_ORDER) if agent_data._markers_top().exists() else 0
        if n_clusters:
            bits.append(f"{n_clusters} clusters")
    except Exception:  # noqa: BLE001 - underivable is stated by omission, never faked
        pass
    try:
        n_panel = len(da.panel_names())
        if n_panel:
            bits.append(f"{n_panel}-gene panel")
    except Exception:  # noqa: BLE001 - same
        pass
    return "Interpretation ready." + (f" {', '.join(bits)}." if bits else "")


def _run_progress() -> None:
    """Short staged progress for the demo hand-off; it reads the precomputed tree, runs nothing."""
    steps = [
        ("Reading jazzPanda output (markers_top.csv)…", 0.28),
        ("Applying the panel-absence rule to the gene panel…", 0.52),
        ("Scoring confidence from spatial glm_coef…", 0.72),
        ("Grounding cell-type notes with looked-up citations…", 0.9),
        (_ready_line(), 1.0),
    ]
    bar = st.progress(0.0, text="Starting…")
    for msg, frac in steps:
        time.sleep(0.5)
        bar.progress(frac, text=msg)
    time.sleep(0.35)


_KEY_OF = {"assay": "platform"}  # checklist field -> manifest key, where they differ


def _stage(folder: Path) -> Study:
    """Link the folder's files (and its ``samples/`` subfolder) into a study under the user root,
    and copy its ``reference/lab_markers.csv`` (the lab list is the study's own release).

    The id carries a hash of the resolved path, so two folders sharing a name never share
    a study.
    """
    folder = folder.resolve()
    sid = f"{folder.name}-{hashlib.sha1(str(folder).encode()).hexdigest()[:8]}"
    store = Study(id=sid, data_root=layout.user_data_root())
    store.inputs.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        dst = store.inputs / p.name
        wanted = (p.is_file() and p.name != "manifest.json") or (p.is_dir() and p.name == "samples")
        if wanted and not (dst.is_symlink() or dst.exists()):
            dst.symlink_to(p.resolve())
    lab = folder / "reference" / "lab_markers.csv"
    if lab.is_file():
        (store.dir / ref_view.LAB_FILE).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lab, store.dir / ref_view.LAB_FILE)
    if not store.exists:
        own = folder / "manifest.json"
        man = {**(json.loads(own.read_text()) if own.is_file() else {}), "dataset_id": store.id}
        if (folder / "samples").is_dir() and "samples" not in man:  # read off the folder names
            man["samples"] = sorted(d.name for d in (folder / "samples").iterdir() if d.is_dir())
        _write_manifest(store, man)
    return store


def _write_manifest(store: Study, man: dict) -> None:
    layout.manifest_json(store.id, store.data_root).write_text(json.dumps(man, indent=2) + "\n")
    study_mod.clear_cache()


def _state(man: dict, fields: dict[str, str]) -> dict:
    """``man`` with the biologist's answers set and named in ``stated``, so they read back STATED."""
    keys = {_KEY_OF.get(f, f): v for f, v in fields.items()}
    return {**man, **keys, "stated": sorted({*(man.get("stated") or ()), *keys})}


def _apply(store: Study, draft: intake.Draft, answers: dict[str, str]) -> bool:
    """The biologist's answers, validated by the draft, written to the staged manifest."""
    stated, ok = {}, True
    for field, value in answers.items():
        many = field in intake.BACKGROUND[1:]  # list-valued: comma-separated entries
        try:
            for v in (value.split(",") if many else [value]):
                if v.strip():
                    draft = draft.stated(field, v)
        except (ValueError, UndeclaredClustering) as exc:
            st.error(f"{field}: {exc}")
            ok = False
            continue
        if value.strip() and field not in intake.BACKGROUND:
            stated[field] = draft.condition(field).value
    man = {**store.manifest, "background": draft.background.to_manifest()}
    if "tissue" in stated:                 # a re-stated tissue drops the old ontology id
        gone = ("tissue_id", "tissue_label")
        man = {k: v for k, v in man.items() if k not in gone}
        man["stated"] = [k for k in man.get("stated") or () if k not in gone]
    _write_manifest(store, _state(man, stated))
    return ok


def _render_tissue_id(store: Study) -> None:
    """Optional: pin the stated tissue to a UBERON id, which keys the organ-scoped sources.
    Looked up live in EBI OLS4 on a click; without one the view is the lab list and CL."""
    if tid := store.manifest.get("tissue_id"):
        st.caption(f"Tissue ontology id: {tid} ({store.manifest.get('tissue_label', '')}), stated")
        return
    q = st.text_input("Tissue ontology id (optional): tissue words or a UBERON id",
                      key="ob_uberon_q", placeholder=store.tissue or "")
    if st.button("Look up UBERON", key="ob_uberon_find"):
        try:
            st.session_state["ob_uberon"] = ref_sources.uberon_candidates(q or store.tissue or "")
        except ref_sources.SourceUnavailable as exc:
            st.warning(str(exc))
    found = st.session_state.get("ob_uberon")
    if found == []:
        st.caption("No UBERON term matched; the study keeps its tissue words.")
    if not found:
        return
    pick = st.selectbox("UBERON term", found, index=None, key="ob_uberon_pick",
                        format_func=lambda c: f"{c[1]} ({c[0]})")
    if pick and st.button("Use this tissue id", key="ob_uberon_use"):
        _write_manifest(store, _state(store.manifest,
                                      {"tissue_id": pick[0], "tissue_label": pick[1]}))
        st.session_state.pop("ob_uberon", None)
        st.rerun()


def _build(store: Study) -> None:
    """Build into the user data root, then open the study for this session."""
    from panoscope.pipeline import run as pipeline_run

    layout.set_session_root(store.data_root)
    try:
        with st.spinner("Building the interpretation…"):
            pipeline_run.run(store.id)
    except Exception as exc:  # noqa: BLE001 - a blocked build is shown, never a traceback
        state.apply_session_study()
        st.error(str(exc))
        return
    study_mod.clear_cache()
    state.open_study(store)
    _nav(nav.SUMMARY_PAGE)


def _render_folder_intake() -> None:
    """§6.4: read the folder, show the checklist, ask only what is missing, build."""
    path = st.text_input("Results folder", key="ob_folder",
                         placeholder="/path/to/your/results")
    folder = Path(path).expanduser() if path.strip() else None
    if folder is None:
        return
    if not folder.is_dir():
        st.error(f"{folder} is not a folder.")
        return
    for p, m in sorted(methods.recognise_inputs(folder).items()):
        st.markdown(f"- `{p.name}`: " + (f"{m.display_name} ({m.workflow})" if m
                                         else "not recognised by any registered method"))
    store = _stage(folder)
    if not store.platform:
        choice = st.selectbox("Platform (blocks the study until declared)", platforms.names(),
                              index=None, key="ob_platform")
        if choice:
            _write_manifest(store, _state(store.manifest, {"assay": choice}))
            st.rerun()
        return
    draft = intake.derive(platforms.for_study(store), store)
    st.code(draft.checklist().render(), language=None)

    with st.form("ob_intake", clear_on_submit=True):
        answers = {f: st.text_input(f, key=f"ob_f_{f}")
                   for f in intake.ITEMS[1:] if draft.status(f) is not intake.Status.KNOWN}
        answers |= {f: st.text_input(f"{f} (optional)", key=f"ob_f_{f}")
                    for f in intake.BACKGROUND}
        if st.form_submit_button("Apply", key="ob_apply") and _apply(store, draft, answers):
            st.rerun()
    if store.tissue:
        _render_tissue_id(store)
    if not agent_loop.has_model_key():
        st.caption("no proposal: no model key. Every cluster opens not yet called, yours to call.")
    if draft.missing:
        st.warning("Blocked until declared: " + ", ".join(draft.missing))
    elif st.button("Build  →", key="ob_build", type="primary"):
        _build(store)


def render_upload() -> None:
    """Load-your-data page (?page=upload)."""
    _css()
    with st.container(key="pano_ob_upload"):
        st.markdown('<div class="pano-ob-atmos"></div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="pano-ob">
  <h2 class="pano-ob-h2">Bring your own results</h2>
  <p class="pano-ob-sub">Point Panoscope at the folder holding your panel and method results. It reads what
  is there, asks only about what is missing, and builds the interpretation; you review it.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        _render_folder_intake()

        # Tissue is read from the dataset (grounded) — the agent reasons within it and
        # prefers tissue-relevant literature (a real cross-tissue paper is still fine).
        tissue = _detected_tissue()
        if tissue:
            st.markdown(
                '<div class="pano-ob-card-h" style="margin-top:1.4rem;">Detected from your dataset</div>'
                f'<div class="pano-ob-detected">🧬 {html.escape(tissue)}</div>'
                '<div class="pano-ob-card-d">Read from your data. Panoscope reasons within this tissue and '
                "prefers tissue-relevant papers; a real cross-tissue reference is still valid.</div>",
                unsafe_allow_html=True,
            )

        # Optional personal sub-focus — sharpens WHICH real paper is cited, saved locally.
        st.markdown(
            '<div class="pano-ob-card-h" style="margin-top:1.2rem;">Anything more specific? '
            '<span style="color:var(--faint);font-weight:400;">· optional</span></div>'
            '<div class="pano-ob-card-d">One line about your angle — a subtype, a program, a marker you '
            "care about. It sharpens which real paper gets cited, on top of the tissue above. "
            "<b>Saved locally on your machine; never uploaded.</b></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="pano_ob_profile"):
            interest = st.text_area(
                "research focus",
                value=agent_profile.load(),
                key="ob_profile",
                label_visibility="collapsed",
                placeholder="e.g. CAF heterogeneity, HER2+ subtypes, spatial immunology",
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
                analyze = st.button("Explore the demo  →", width="stretch", key="ob_analyze")
        with st.container(key="pano_ob_skip"):
            skip = st.button("← back", key="ob_back")

        if skip:
            _nav("welcome")
        if analyze:
            agent_profile.save(interest)
            state.open_study(None)
            _run_progress()
            _nav("summary")   # enter the dashboard
