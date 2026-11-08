"""Evidence-first annotation review (interface §5.3): the candidate types' reference
lists and corpus cards under the markers, the proposal one tap away, the commit bar,
and after a call the biologist's assessment beside the proposal's.

Reference data selects which genes are shown; every state is the lattice's. Corpus
dissent is a display flag: nothing here reaches ``verify``.
"""
from __future__ import annotations

import datetime
import html
from typing import Any

from panoscope.agent import config as cfg
from panoscope.core import measure
from panoscope.data.study import NOT_YET_CALLED
from panoscope.pipeline import decisions as D
from panoscope.types.measured import MeasuredState
from panoscope.types.subject import gene, subject
from panoscope.ui import data_access as da
from panoscope.ui import evidence_table as et
from panoscope.ui import surface

_ACTOR = "you@example.com"     # memory.save_draft's own default author
#: The rail's mark per decision state.
MARK = {D.OPEN: "○", D.ACCEPTED: "✓", D.OVERRIDDEN: "✎", D.DEFERRED: "…"}
_DISSENT = "In window, dissent:"
_CSS = ("<style>.pano-rv-cand{font-size:12px;line-height:1.7;padding:2px 0}"
        ".pano-rv-assess{border:1px solid var(--hair2);border-radius:6px;padding:8px 10px;"
        "font-size:13px}</style>")


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _view() -> Any:
    from panoscope.reference import view

    return view.for_study(cfg.current_study())


def revealed(cluster: str) -> bool:
    """May the proposal show: tapped open, decided, or already overridden by the biologist."""
    import streamlit as st

    return bool(st.session_state.get(f"rv_reveal_{cluster}") or da.decision(cluster))


def gene_states(built: Any, genes: tuple[str, ...]) -> list[tuple[str, MeasuredState]]:
    """Each gene's state for this unit: the record's own item, else the same lattice."""
    have = {i.subject.display().upper(): i.measured.state for i in built.evidence.items}
    return [(g, have.get(g.upper()) or measure.resolve(
        subject("gene", g=gene(g)), built.world, built.reported).value.state) for g in genes]


def candidates_html(built: Any, view: Any) -> str:
    """One line per candidate type: its canonical markers, each with its state glyph."""
    rows = [(ct, gene_states(built, view.canonical(ct))) for ct in view.candidates()]
    if not rows:
        return ('<div class="pano-sf-empty">no candidate cell types in reference data '
                'for this tissue</div>')
    # A viewing order only: types with more observed canonical markers first.
    rows.sort(key=lambda r: -sum(s is MeasuredState.OBSERVED for _, s in r[1]))
    out = ['<div class="pano-sf-block">reference canonical markers, per candidate type '
           '&middot; reference selects, the measurement answers</div>']
    for ct, genes in rows:
        chips = " ".join(
            f'<span class="pano-sf-glyph s-{s.value}" title="{_esc(surface._GLYPH[s][1])}" '
            f'data-claim="{_esc(subject("gene", g=gene(g)).key)}">'
            f'{surface._GLYPH[s][0]} {_esc(g)}</span>' for g, s in genes)
        out.append(f'<div class="pano-rv-cand"><b>{_esc(ct)}</b> '
                   f'{chips or "<i>no canonical marker listed</i>"}</div>')
    return "".join(out)


def card_lines(study_id: str, cell_types: tuple[str, ...], platform: str = "") -> list[str]:
    """Each candidate's corpus card as text; dissent marked as a flag. Not built: one line."""
    from panoscope.corpus import cards
    from panoscope.corpus.build import NOT_BUILT
    from panoscope.corpus.store import corpus_path, load

    try:
        corpus = load(corpus_path(study_id))
    except Exception:  # noqa: BLE001 - CorpusUnavailable, or a cache inside the repo
        return [NOT_BUILT]
    out: list[str] = []
    for ct in cell_types:
        lines = cards.cell_type_card(corpus, ct, platform).lines()
        out.append(f"{ct}: {lines[0]}")
        out += [("⚑ dissent (a flag, never sets verify):" if ln == _DISSENT else ln)
                for ln in lines[1:]]
    return out


def _assess_html(title: str, label: str, v: Any, claim: str, reads_as: str = "") -> str:
    """One side of the side-by-side: the band function's output on one label."""
    driver = v.driver.subject.display() if v.driver is not None else "none"
    flag = " &#9873; verify" if v.verify else ""
    note = f'<div class="pano-sf-dim">{_esc(reads_as)}</div>' if reads_as else ""
    return (f'<div class="pano-rv-assess" data-claim="{_esc(surface.anchored(claim, what=title))}">'
            f'<div class="pano-eyebrow">{_esc(title)}</div><b>{_esc(label)}</b>'
            f'<div>{_esc(v.band)}{flag}</div>'
            f'<div class="pano-sf-dim">{_esc(v.basis)} · driver {_esc(driver)}</div>{note}</div>')


def _tension_line(note_id: str) -> str:
    """The override note's literature check, derived at its draft: a display flag only."""
    from panoscope.agent import memory, tools
    from panoscope.claims.notes import lookup_state
    from panoscope.corpus.store import Resolution

    n = memory.get_note(note_id, tools.memory_base_dir())
    if n is None:
        return ""
    t = n.tension
    state = lookup_state(t)
    body = (f"{len(t.agree)} agree / {len(t.dissent)} dissent (dissent is a flag, "
            "never sets verify)" if state is Resolution.RESOLVED
            else "literature thin" if state is Resolution.NOT_FOUND
            else "not looked up, so it cannot settle this")
    return f"literature check on your call: {body} [note:{note_id}]"


def _override(cluster: str, label: str, basis: str, built: Any, view: Any) -> None:
    """Draft the override as a cluster note (corpus tension at draft), then commit it."""
    from panoscope.agent import memory, tools

    observed = tuple(g for g, s in gene_states(built, view.canonical(label))
                     if s is MeasuredState.OBSERVED)
    draft = memory.draft_note(
        claim=f"{cluster} is {label} (the biologist's call)", scope="cluster", basis=basis,
        cluster=cluster, subject_cell_type=label, subject_markers=observed,
        dataset=cfg.DATASET_ID, lookup_markers=view.canonical(label))
    D.override(cluster, draft, by=_ACTOR, at=_now(), base_dir=tools.memory_base_dir())


def render_review(cluster: str) -> None:
    """Everything under the marker table for ``cluster``; proposal hidden until tapped."""
    import streamlit as st

    from panoscope.agent import tools
    from panoscope.ui import conversation as convo

    built = et.unit_build(cluster)
    if built is None:
        return
    view = _view()
    st.markdown(_CSS + candidates_html(built, view), unsafe_allow_html=True)
    lines = card_lines(cfg.DATASET_ID, view.candidates(), da.tissue_platform()[1])
    st.markdown('<div class="pano-sf-block">literature, per candidate type</div>'
                + "".join(f'<div class="pano-ev-conc">{_esc(ln)}</div>' for ln in lines),
                unsafe_allow_html=True)

    proposal = da.verdict_for(cluster).cell_type
    shown = st.toggle("Show the proposal (call and band)", key=f"rv_reveal_{cluster}")
    if shown:
        # The proposal as built, never the override-aware display label.
        hv = {**et._header_values(cluster), "label": proposal}
        if proposal == NOT_YET_CALLED:
            hv["confidence"] = ""
        st.markdown(surface.header_html(et.spec_for(built), hv), unsafe_allow_html=True)

    decs = {cluster: da.decision(cluster)}
    s = D.state(decs, cluster)
    st.markdown(f'<div class="pano-sf-block">decision &middot; {_esc(s.lower())}</div>',
                unsafe_allow_html=True)
    if s == D.OPEN:
        a, b, c = st.columns(3)
        if a.button("Accept proposal", key=f"rv_accept_{cluster}", width="stretch",
                    disabled=proposal == NOT_YET_CALLED or not shown):
            D.accept(cluster, proposal, by=_ACTOR, at=_now(), base_dir=tools.memory_base_dir())
            st.rerun()
        b.toggle("Call it yourself", key=f"rv_calling_{cluster}")
        if c.button("Defer", key=f"rv_defer_{cluster}", width="stretch"):
            D.defer(cluster, by=_ACTOR, at=_now(), base_dir=tools.memory_base_dir())
            st.rerun()
        if st.session_state.get(f"rv_calling_{cluster}"):
            cands, expected = D.label_space(view)
            pick = st.selectbox("Cell type", cands + expected, index=None,
                                key=f"rv_pick_{cluster}", placeholder="a candidate type")
            free = st.text_input("or your own label", key=f"rv_free_{cluster}")
            label = (free or "").strip() or pick
            st.markdown(f'<div class="draft-scope"><span class="lbl">scope</span>'
                        f'this cluster ({_esc(cluster)})</div>', unsafe_allow_html=True)
            basis = convo._seg(st, "basis", convo._BASIS_OPTS, "own_validation",
                               f"rv_basis_{cluster}")
            if st.button("Save call", key=f"rv_save_{cluster}", type="primary",
                         disabled=not label):
                _override(cluster, label, basis, built, view)
                st.rerun()
        return

    label = D.call(decs, cluster)
    if label is None:          # DEFERRED: no call, nothing to assess
        return
    got = D.assessments(built, label, view)
    note_id = decs[cluster].get("note_id")
    left, right = st.columns(2)
    left.markdown(_assess_html("Proposal", proposal, got.proposal, et._unit_key(cluster)),
                  unsafe_allow_html=True)
    right.markdown(_assess_html("Your call", label, got.biologist,
                                f"note:{note_id}" if note_id else et._unit_key(cluster),
                                got.reads_as), unsafe_allow_html=True)
    if note_id and (line := _tension_line(note_id)):
        st.markdown(f'<div class="pano-ev-conc">{_esc(line)}</div>', unsafe_allow_html=True)
