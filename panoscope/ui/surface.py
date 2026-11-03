"""The generic renderer: a ``UISpec`` in, the review surface out.

``panoscope/types/uispec.py`` made the review surface declarable. Three workflows
declared one. Nothing read them: ``app.py`` hardcoded three tabs and every page was
hand-built per method, which is the thing ARCHITECTURE §13 says Panoscope does not
do. This module is the half that was missing -- *"the review surface is part of the
interpretation contract, declared by the workflow"* with pixels behind it.

**The split: pure HTML functions, one thin Streamlit wrapper.** Everything that
decides what a cell says is a pure ``*_html`` function over a spec and rows, so the
guards below are testable with no server running and the surface can be diffed as a
string. :func:`render_surface` is the only function that imports Streamlit, and all
it does is ``st.markdown`` what the pure half returned (plus the per-row pin button,
which has to be a widget).

**Four rules this renderer enforces rather than documents.**

1. *Degradation goes DOWN.* :func:`label_band_html` refuses a surface whose task
   kind has no label space, and :func:`provenance_html` refuses one that declared
   a task kind. ``UISpec.__post_init__`` already makes the illegal *declaration*
   unconstructable; this is the same rule at the other end, because the bug it is
   named for -- ``ui/enrichment_table.py`` importing the assignment header -- was a
   renderer reaching for another task kind's header, not a bad declaration.
2. *A never-measured item is not a blank row and not a zero.* A numeric cell with
   no number hatches, with the state and its reason on the hover. A number arriving
   on a row whose state forbids one is refused outright rather than drawn.
3. *Click-back.* Every row and every claim-bearing header sentence carries a
   ``Subject.key`` anchor; one that does not is not rendered, it raises.
4. *Plots render, they do not derive.* The spec types have nowhere to write a
   computation down; :func:`_no_code` checks that nothing put one there anyway
   before a single cell is drawn, so a ``MethodSpec.status_rule`` leaking into
   ``extra_columns`` fails here instead of running in the renderer.

**What the renderer does not decide.** Not the measurement state (it arrives on the
row, resolved server-side), not a number's value (it formats what it is handed, to
the declared precision), not the block order or the sort key (declared), and not
which rows exist. Sorting happens *within* a block, never across one, because
sorting across blocks is how a top-N hides every absence at the bottom of a list
nobody scrolls.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from panoscope.types.measured import MeasuredState
from panoscope.types.outputs import TaskKind
from panoscope.types.uispec import (
    BlockSpec,
    CensusHeader,
    ColumnKind,
    ColumnSpec,
    ConfoundChecklistHeader,
    DegradedUpward,
    HeaderKind,
    LabelBandHeader,
    NothingToClick,
    ProvenanceOnlyHeader,
    SpecCarriesCode,
    UISpec,
    anchored,
)
# Imported, never re-listed. uispec.py derives both from `Measured.carries_number`,
# and a second copy here would be the exact bug its own comment warns about: a
# widened property would move the hatching in one file and leave this one behind.
from panoscope.types.uispec import _NO_NUMBER, _NUMERIC_KINDS
from panoscope.ui import format as fmt


# --------------------------------------------------------------------------- #
# The row. Resolved server-side: the renderer is handed a state, never a rule.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SurfaceRow:
    """One item row, already adjudicated.

    ``key`` is a ``Subject.key`` and is the click-back anchor -- the same string
    reconciliation joins on, so a row in the table and a claim in a record resolve
    to each other without a second identity scheme.

    ``values`` maps a declared ``ColumnSpec.key`` to an already-resolved value.
    ``None`` means *there is no number here*, which is a different fact from zero
    and is rendered differently.
    """

    key: str
    state: MeasuredState
    values: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""          # why this state -- the hover on the hatch
    #: The capability frame that produced ``state``, verbatim from
    #: ``Measured.frame`` ("panel:<study>", "detect:<study>"). Carried as its own
    #: field rather than parsed back out of ``reason``, because a block that
    #: selects on it (``BlockSpec.frame_prefix``) must not break when that
    #: sentence is copy-edited -- the defect this table already has a comment
    #: about, one layer up.
    frame: str = ""


# ● measured  ◐ measured, not here  ◌ below the floor  ⊘ never measured.
# Four glyphs because there are four states; the two-glyph version of this table is
# the collapse the whole lattice exists to stop.
_GLYPH: dict[MeasuredState, tuple[str, str]] = {
    MeasuredState.OBSERVED: ("&#9679;", "observed"),
    MeasuredState.TESTED_ABSENT: ("&#9680;", "measured, not here"),
    MeasuredState.UNDERPOWERED: ("&#9676;", "too little signal"),
    MeasuredState.NEVER_MEASURED: ("&#8856;", "never measured"),
}

# `width=0` on a ColumnSpec means "the renderer decides" (uispec.py). It decides by
# kind: prose needs room, a number does not.
_DEFAULT_FR: dict[ColumnKind, int] = {
    ColumnKind.CITATION: 3,
    ColumnKind.NOTE_CHIP: 2,
    ColumnKind.TEXT: 2,
    # A numeric column is not narrow, because on a non-observed row it does not
    # hold a number -- it holds the hatch, "◐&nbsp;measured, not here", whose
    # non-breaking space binds the glyph to its label so the whole run cannot
    # wrap. Measured in a browser: it wants 78px and a 1fr track gave it 42 at
    # 1024 and 63 at 1440, so the text spilled into the next column. The header
    # "SPECIFICITY" wants 72 and got 42.
    #
    # This only became visible once the tracks started honouring their fractions
    # (``minmax(0, Nfr)``). Before that each box sized to its own content, which
    # is exactly why the header and the rows disagreed and the headings sat over
    # the wrong columns. The fractions are the fix; they just have to be big
    # enough for what the cells actually carry.
    ColumnKind.NUMBER: 2,
    ColumnKind.SPECIFICITY_PAIR: 2,
    ColumnKind.GLYPH: 2,
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _no_code(spec: UISpec) -> None:
    """Refuse a declaration that carries a computation. Rule 4, before any pixel."""
    for col in spec.columns:
        for name in col.__slots__:
            if callable(getattr(col, name, None)):
                raise SpecCarriesCode(
                    f"{spec.workflow_id}: column {col.key!r} carries a callable on "
                    f"{name!r}. Plots and columns render; they do not derive, and a "
                    "renderer that can run a rule can decide what counts as measured."
                )


def _anchor(inner: str, key: str, cls: str) -> str:
    """Wrap a rendered claim in its click-back anchor, or refuse to render it.

    The check is ``uispec.anchored`` rather than an ``if`` here: a row emits its
    anchor and its target in one span, so it resolves to itself by construction and
    has nothing to be checked against -- but a sentence ABOUT a subject does not,
    and that half of the rule was missing from both renderers' private copies of it.
    """
    return (f'<span class="{cls}" data-claim="{_esc(anchored(key, what=cls))}">'
            f'{inner}</span>')


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def _hatch(row: SurfaceRow) -> str:
    """The cell for a row that has no number. Not blank, not a zero."""
    glyph, label = _GLYPH[row.state]
    title = row.reason or f"{row.state.value}: absence here is not evidence"
    return (f'<span class="pano-sf-hatch" title="{_esc(title)}">'
            f'{glyph}&nbsp;{_esc(label)}</span>')


def cell_html(col: ColumnSpec, row: SurfaceRow) -> str:
    """One cell. The hatching rule lives here and nowhere else."""
    value = row.values.get(col.key)

    if col.kind is ColumnKind.GLYPH:
        glyph, label = _GLYPH[row.state]
        title = row.reason or label
        return (f'<span class="pano-sf-glyph s-{row.state.value}" '
                f'title="{_esc(title)}">{glyph} {_esc(label)}</span>')

    if col.kind in _NUMERIC_KINDS:
        if row.state in _NO_NUMBER and value is not None:
            raise DegradedUpward(
                f"column {col.key!r} was handed {value!r} on a row whose state is "
                f"{row.state.value}. A statistic may exist only where something was "
                "observed; quoting one for an unmeasured thing claims more than the "
                "measurement permits."
            )
        if value is None or row.state in col.hatch_when:
            return _hatch(row)
        return f'<span class="pano-sf-num">{_esc(_number(col, value))}</span>'

    if value is None or value == "" or value == ():
        return '<span class="pano-sf-dim">&mdash;</span>'

    if col.kind is ColumnKind.CITATION:
        text, href = value if isinstance(value, tuple) else (value, "")
        cite = (f'<a class="pano-sf-cite" href="{_esc(href)}" target="_blank">'
                f'{_esc(href.rstrip("/").rsplit("/", 1)[-1])}</a>') if href else ""
        return f'<span class="pano-sf-prose">{_esc(text)} {cite}</span>'

    if col.kind is ColumnKind.NOTE_CHIP:
        chips = value if isinstance(value, (tuple, list)) else (value,)
        return "".join(f'<span class="pano-sf-chip">{_esc(c)}</span>' for c in chips)

    if col.kind is ColumnKind.DIRECTION:
        arrow = {"up": "&#9650;", "down": "&#9660;"}.get(str(value).lower(), "&#8213;")
        return f'<span class="pano-sf-dir">{arrow} {_esc(value)}</span>'

    return f'<span class="pano-sf-text">{_esc(value)}</span>'


def _number(col: ColumnSpec, value: Any) -> str:
    """Format a number to the DECLARED precision. Never rounds a pair differently."""
    if col.kind is ColumnKind.SPECIFICITY_PAIR and isinstance(value, (tuple, list)):
        return " / ".join(fmt.num_fmt(v, col.precision) for v in value)
    if col.kind is ColumnKind.COVERAGE_BAR and isinstance(value, (tuple, list)):
        return "/".join(str(v) for v in value)
    return fmt.num_fmt(value, col.precision)


def _fr(col: ColumnSpec) -> int:
    return col.width or _DEFAULT_FR.get(col.kind, 1)


def _grid_style(cols: Sequence[ColumnSpec], *, skip_first: bool = False) -> str:
    """The shared track list. ``minmax(0, Nfr)``, never a bare ``Nfr``.

    A bare ``1fr`` is ``minmax(auto, 1fr)``, so a track never shrinks below its own
    content and the SAME declaration resolves differently in every box it is
    applied to. The header and the rows are separate grids carrying one string, and
    measured in a browser that string produced three different track sets on one
    page -- header ``101 103 117 51 101 152``, one row group ``117 78 78 59 117
    176``, another ``125 63 63 63 125 188``. "observed" rendered under the SPATIAL
    SPECIFICITY heading with no overlap of its own, and the cited biology under
    CAVEATS.

    ``minmax(0, ...)`` lets a track shrink below its content, so the fractions are
    honoured identically everywhere and the columns line up. Verified in the same
    browser: all six offsets go to 0.0px.
    """
    used = cols[1:] if skip_first else cols
    return ("grid-template-columns:"
            + " ".join(f"minmax(0,{_fr(c)}fr)" for c in used) + ";")


def row_html(spec: UISpec, row: SurfaceRow, *, skip_first: bool = False) -> str:
    """One table row, anchored on its ``Subject.key``."""
    cols = spec.columns[1:] if skip_first else spec.columns
    cells = "".join(
        f'<div class="pano-sf-cell a-{c.align}">{cell_html(c, row)}</div>'
        for c in cols
    )
    grid = _grid_style(spec.columns, skip_first=skip_first)
    return _anchor(f'<div class="pano-sf-row" style="{grid}">{cells}</div>',
                   row.key, "pano-sf-anchor")


# --------------------------------------------------------------------------- #
# Blocks and the coverage ribbon
# --------------------------------------------------------------------------- #
def rows_for(block: BlockSpec, rows: Sequence[SurfaceRow]) -> list[SurfaceRow]:
    """This block's rows, sorted WITHIN the block. Never truncated, never merged."""
    got = [r for r in rows
           if r.state in block.statuses
           and r.frame.startswith(block.frame_prefix)
           and not any(r.frame.startswith(x) for x in block.frame_excludes)]
    if block.sort_by:
        got.sort(key=lambda r: (r.values.get(block.sort_by) is None,
                                r.values.get(block.sort_by) or 0),
                 reverse=block.sort_desc)
    return got


def block_label_html(block: BlockSpec, n: int) -> str:
    return (f'<div class="pano-sf-block">'
            f'{_esc(block.label_template.format(count=n))}</div>')


def block_html(spec: UISpec, block: BlockSpec, rows: Sequence[SurfaceRow]) -> str:
    """A block header and its rows -- the header even at zero, with its empty text.

    A block that disappears at zero is the panel-absence rule disappearing at the
    moment it had something to say.
    """
    got = rows_for(block, rows)
    out = [block_label_html(block, len(got))]
    if not got:
        out.append(f'<div class="pano-sf-empty">{_esc(block.empty_text)}</div>'
                   if block.empty_text else "")
    else:
        out.extend(row_html(spec, r) for r in got)
    return "".join(out)


def ribbon_html(spec: UISpec, rows: Sequence[SurfaceRow]) -> str:
    """The table's own conditionality: one stacked bar, every count labelled.

    What stops a screen of twelve supporting programs from reading as twelve of
    twelve.
    """
    if not spec.coverage_ribbon.enabled or not rows:
        return ""
    total = len(rows)
    segs, labels = [], []
    for state in spec.coverage_ribbon.order:
        n = sum(1 for r in rows if r.state is state)
        if not n:
            continue
        glyph, label = _GLYPH[state]
        segs.append(f'<span class="pano-sf-seg s-{state.value}" '
                    f'style="width:{100 * n / total:.4g}%" '
                    f'title="{_esc(f"{n} of {total} {label}")}"></span>')
        labels.append(f'<span class="pano-sf-key s-{state.value}">{glyph} {n} '
                      f'{_esc(label)}</span>')
    return (f'<div class="pano-sf-ribbon">{"".join(segs)}</div>'
            f'<div class="pano-sf-keys">{"".join(labels)} '
            f'<span class="pano-sf-key">of {total}</span></div>')


def column_header_html(spec: UISpec, *, skip_first: bool = False) -> str:
    cols = spec.columns[1:] if skip_first else spec.columns
    cells = "".join(
        f'<div class="pano-sf-cell a-{c.align}">{_esc(c.label)}'
        + (f'<span class="pano-sf-unit">{_esc(c.unit)}</span>' if c.unit else "")
        + "</div>"
        for c in cols
    )
    grid = _grid_style(spec.columns, skip_first=skip_first)
    return f'<div class="pano-sf-head" style="{grid}">{cells}</div>'


def table_html(spec: UISpec, rows: Sequence[SurfaceRow]) -> str:
    _no_code(spec)
    return (ribbon_html(spec, rows) + column_header_html(spec)
            + "".join(block_html(spec, b, rows) for b in spec.blocks))


# --------------------------------------------------------------------------- #
# Headers. One function per variant, each refusing the task kinds it may not serve.
# --------------------------------------------------------------------------- #
def _idline(values: Mapping[str, Any]) -> str:
    scope = values.get("unit_scope") or ""
    return f'<div class="pano-idline">{_esc(scope)}</div>' if scope else ""


def label_band_html(spec: UISpec, values: Mapping[str, Any]) -> str:
    """ASSIGNMENT only. The band, its driver, and the rejected alternative.

    The guard is the point of the function. ``ui/enrichment_table.py`` reached for
    this header from a characterisation page and inherited a confidence band it had
    no right to; a renderer that cannot be reached that way is the fix, and a
    comment saying "do not import this" is not.
    """
    h = spec.header
    if not isinstance(h, LabelBandHeader) or spec.task_kind is not TaskKind.ASSIGNMENT:
        raise DegradedUpward(
            f"{spec.workflow_id}: {spec.task_kind} has no label space, so a "
            "unit-level confidence band is a category error, not merely unhelpful. "
            "Degradation goes toward making no claim, never toward borrowing one."
        )
    band = values.get(h.band_field) or ""
    css, text = fmt.confidence_chip(band) if band else ("", "")
    verify = ('<span class="pano-verify">&#9873; re-check this</span>'
              if values.get(h.verify_field) else "")
    label = values.get("label") or ""
    audit = " &middot; ".join(
        str(x) for f in h.band_audit_fields for x in _seq(values.get(f))
    )
    drivers = ", ".join(_seq(values.get(h.driver_field)))
    runner_up = values.get("runner_up") or ""
    return (
        _idline(values)
        + f'<div class="pano-verdict"><h1>{_esc(label)}</h1>'
        + (f'<span class="cf {css}" title="{_esc(audit)}">{_esc(text)}</span>'
           if band else "")
        + verify + "</div>"
        + (f'<div class="pano-rat">Driven by {_esc(drivers)}.</div>' if drivers else "")
        + (f'<div class="pano-sf-scope">rejected: {_esc(runner_up)}</div>'
           if runner_up else "")
        + _gates_html(values.get("gates"))
    )


def census_html(spec: UISpec, values: Mapping[str, Any]) -> str:
    """CHARACTERISATION-shaped. Per-item counts, each with its denominator; no band.

    There is no band slot to fill, which is the structural form of the rule rather
    than a reminder about it.
    """
    h = spec.header
    if not isinstance(h, CensusHeader):
        raise DegradedUpward(f"{spec.workflow_id}: census_html called on a "
                             f"{spec.header.kind.value} header")
    out = [_idline(values)]

    for f in h.imported_context_fields:
        if values.get(f):
            out.append(f'<div class="pano-verdict"><h1>{_esc(values[f])}</h1>'
                       f'<span class="pano-sf-imported" '
                       f'title="{_esc(h.imported_context_note)}">imported</span>'
                       "</div>")

    for line in h.lines:
        counts = " &middot; ".join(
            f'<b>{_esc(values.get(fieldname, 0))}</b> {_esc(label)}'
            for label, fieldname in line.counts
        )
        denom = line.denominator_template.format(
            value=values.get(line.denominator_field, "?"))
        out.append(f'<div class="pano-sf-census"><span class="lbl">'
                   f'{_esc(line.label)}</span> {counts} '
                   f'<span class="den">{_esc(denom)}</span></div>')

    out.append(_top_item_html(h.top_item_template, values))
    if values.get(h.scope_line_field):
        out.append(f'<div class="pano-sf-scope">'
                   f'{_esc(values[h.scope_line_field])}</div>')
    out.append(_gates_html(values.get(h.gates_field)))
    return "".join(out)


def checklist_html(spec: UISpec, values: Mapping[str, Any]) -> str:
    """COMPARISON-shaped. The checks and what blocked them; no band here either."""
    h = spec.header
    if not isinstance(h, ConfoundChecklistHeader):
        raise DegradedUpward(f"{spec.workflow_id}: checklist_html called on a "
                             f"{spec.header.kind.value} header")
    out = [_idline(values)]
    for template in (h.design_template, h.results_template,
                     h.testable_denominator_template):
        line = _fill(template, values)
        if line:
            out.append(f'<div class="pano-sf-census">{_esc(line)}</div>')
    checks = "".join(
        f'<span class="pano-sf-check st-{_esc(values.get(c.state_field, "unknown"))}" '
        f'title="{_esc(values.get(c.detail_field, ""))}">{_esc(c.label)}: '
        f'{_esc(values.get(c.state_field, "not assessed"))}</span>'
        for c in h.checks
    )
    if checks:
        out.append(f'<div class="pano-sf-checks">{checks}</div>')
    if values.get(h.scope_line_field):
        out.append(f'<div class="pano-sf-scope">'
                   f'{_esc(values[h.scope_line_field])}</div>')
    return "".join(out)


def provenance_html(spec: UISpec, values: Mapping[str, Any]) -> str:
    """The degraded header. A banner and the file -- no claim of any kind.

    Refuses a spec that DID declare a task kind: degrading a registered workflow to
    "we know nothing" hides a working interpretation behind a disclaimer, which is
    the mirror-image failure of degrading upward and just as dishonest.
    """
    h = spec.header
    if not isinstance(h, ProvenanceOnlyHeader) or spec.task_kind is not None:
        raise DegradedUpward(
            f"{spec.workflow_id}: ProvenanceOnlyHeader iff the task kind is "
            f"undeclared (task_kind={spec.task_kind})."
        )
    prov = " &middot; ".join(_esc(x) for x in _seq(values.get("provenance")))
    return (f'<div class="pano-sf-banner">{_esc(h.banner)}</div>'
            f'<div class="pano-verdict"><h1>{_esc(spec.title)}</h1></div>'
            + (f'<div class="pano-sf-scope">{prov}</div>' if prov else ""))


_HEADERS: dict[HeaderKind, Callable[[UISpec, Mapping[str, Any]], str]] = {
    HeaderKind.LABEL_BAND: label_band_html,
    HeaderKind.CENSUS: census_html,
    HeaderKind.CONFOUND_CHECKLIST: checklist_html,
    HeaderKind.PROVENANCE_ONLY: provenance_html,
}


def header_html(spec: UISpec, values: Mapping[str, Any]) -> str:
    """The declared header for this surface. Dispatch only; the guards are below."""
    _no_code(spec)
    return _HEADERS[spec.header.kind](spec, values)


def _seq(value: Any) -> tuple:
    if value is None or value == "":
        return ()
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _fill(template: str, values: Mapping[str, Any]) -> str:
    """Fill a declared template, or render nothing. Never a half-filled sentence."""
    if not template:
        return ""
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return ""


def _top_item_html(template: str, values: Mapping[str, Any]) -> str:
    """The header's one sentence about a specific item -- so it carries an anchor."""
    line = _fill(template, values)
    if not line:
        return ""
    return ('<div class="pano-rat">'
            + _anchor(_esc(line), values.get("top_key", ""), "pano-sf-anchor")
            + "</div>")


def _gates_html(gates: Any) -> str:
    """The bars a result had to clear, named. A gate nobody can read is a rumour."""
    items = _seq(gates)
    if not items:
        return ""
    return ('<div class="pano-sf-gates">'
            + "".join(f'<span class="pano-sf-chip">{_esc(g)}</span>' for g in items)
            + "</div>")


# --------------------------------------------------------------------------- #
# The one Streamlit entry point
# --------------------------------------------------------------------------- #
_CSS = """
<style>
.pano-sf-ribbon { display:flex; height:7px; border-radius:4px; overflow:hidden;
  background:var(--hair2); margin:14px 0 6px; }
.pano-sf-seg { display:block; height:100%; }
.pano-sf-seg.s-observed { background:var(--accent); }
.pano-sf-seg.s-tested_absent { background:var(--muted); }
.pano-sf-seg.s-underpowered { background:var(--absent); opacity:.55; }
.pano-sf-seg.s-never_measured {
  background:repeating-linear-gradient(45deg,var(--faint),var(--faint) 2px,
    transparent 2px,transparent 5px); }
.pano-sf-keys { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px; }
.pano-sf-key { font-family:var(--mono); font-size:10px; color:var(--faint); }
.pano-sf-key.s-observed { color:var(--accent); }
.pano-sf-head, .pano-sf-row { display:grid; gap:10px; align-items:start; }
.pano-sf-head { font-family:var(--mono); font-size:10px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--faint); padding:0 0 5px;
  border-bottom:1px solid var(--hair); }
/* Labels wrap between words, never inside one. The tracks are minmax(0,Nfr) so a
   heading may be narrower than its own text -- that is what keeps the columns
   aligned with the rows -- and without this "SPECIFICITY" split as SPECIFICI/TY
   and "glm_coef" broke at its underscore. */
.pano-sf-head .pano-sf-cell, .pano-sf-row .pano-sf-cell {
  word-break:normal; overflow-wrap:break-word; }
.pano-sf-unit { display:block; margin-left:0; }
.pano-sf-row { padding:7px 0; border-bottom:1px solid var(--hair2); }
.pano-sf-cell.a-right { text-align:right; }
.pano-sf-unit { font-size:9px; opacity:.7; margin-left:4px; }
.pano-sf-num, .pano-sf-text { font-family:var(--mono); font-size:13px; color:var(--ink); }
.pano-sf-prose { font-size:12px; color:var(--muted); line-height:1.5; }
.pano-sf-dim { color:var(--faint); }
.pano-sf-cite { font-family:var(--mono); font-size:10px; color:var(--accent);
  text-decoration:none; white-space:nowrap; }
.pano-sf-chip { display:inline-block; font-family:var(--mono); font-size:9px;
  color:var(--muted); background:var(--hair2); border-radius:4px; padding:1px 6px;
  margin:0 4px 2px 0; }
/* A never-measured cell is HATCHED, never blank and never a zero: a blank reads as
   "we looked and found nothing", a hatch reads as "we could not look". */
/* max-width so the pill cannot exceed its own column. An inline-block with no
   constraint takes its max-content width, and the glyph is bound to its label by
   a non-breaking space -- so "◐ measured, not here" spilled 13px into the next
   column at 1024 while the cell around it reported no overflow at all. */
.pano-sf-hatch { display:inline-block; max-width:100%; font-family:var(--mono);
  font-size:10px;
  color:var(--faint); border-radius:4px; padding:1px 6px; cursor:help;
  background:repeating-linear-gradient(45deg,var(--hair2),var(--hair2) 3px,
    transparent 3px,transparent 6px); }
.pano-sf-glyph { font-family:var(--mono); font-size:10px; color:var(--faint);
  cursor:help; }
.pano-sf-glyph.s-observed { color:var(--accent); }
.pano-sf-glyph.s-tested_absent { color:var(--muted); }
.pano-sf-glyph.s-underpowered, .pano-sf-glyph.s-never_measured { color:var(--absent); }
.pano-sf-block { font-family:var(--mono); font-size:10px; text-transform:uppercase;
  letter-spacing:.07em; color:var(--muted); margin:16px 0 2px; padding-top:8px;
  border-top:1px solid var(--hair); }
.pano-sf-empty { font-family:var(--mono); font-size:11px; color:var(--faint);
  padding:6px 0 2px; }
.pano-sf-census { font-family:var(--mono); font-size:12px; color:var(--ink);
  margin:3px 0; }
.pano-sf-census .lbl { color:var(--faint); letter-spacing:.07em; margin-right:7px; }
.pano-sf-census .den { color:var(--faint); }
.pano-sf-scope { font-family:var(--mono); font-size:10.5px; color:var(--faint);
  line-height:1.5; margin:6px 0 2px; }
.pano-sf-gates { display:flex; flex-wrap:wrap; gap:4px; margin:6px 0 2px; }
.pano-sf-imported { font-family:var(--mono); font-size:9px; color:var(--muted);
  background:var(--hair2); border-radius:4px; padding:2px 7px; cursor:help; }
.pano-sf-banner { font-size:12px; color:var(--absent); background:var(--absent-bg);
  border-left:2px solid var(--absent); border-radius:0 6px 6px 0; padding:9px 12px;
  margin-bottom:10px; line-height:1.5; }
.pano-sf-check { font-family:var(--mono); font-size:10px; color:var(--muted);
  border:1px solid var(--hair); border-radius:4px; padding:1px 6px; margin-right:5px;
  cursor:help; }
.pano-sf-checks { margin:6px 0; }
/* Pin button: chromeless, a dot as ::before, matching the cluster rail. */
div[class*="st-key-sfrow_"] [data-testid="stColumn"] { padding:0 !important; }
div[class*="st-key-sfrow_"] div[data-testid="stButton"] > button {
  background:transparent !important; border:0 !important; box-shadow:none !important;
  min-height:0 !important; padding:5px 4px !important; border-radius:6px !important;
  display:flex !important; align-items:center !important;
  justify-content:flex-start !important; font-family:var(--sans) !important;
  font-weight:600 !important; font-size:13px !important; color:var(--ink) !important;
  text-align:left !important; }
div[class*="st-key-sfrow_"] div[data-testid="stButton"] > button > div {
  flex:0 1 auto !important; width:auto !important; }
div[class*="st-key-sfrow_"] div[data-testid="stButton"] > button::before {
  content:''; width:10px; height:10px; border-radius:50%; flex:none; margin-right:9px;
  box-sizing:border-box; border:1.5px solid var(--faint); background:transparent; }
div[class*="st-key-sfrow_"] button[kind="primary"]::before,
div[class*="st-key-sfrow_"] button[data-testid="stBaseButton-primary"]::before {
  background:var(--accent) !important; border-color:var(--accent) !important; }
div[class*="st-key-sfrow_"] button[kind="primary"],
div[class*="st-key-sfrow_"] button[data-testid="stBaseButton-primary"] {
  color:var(--accent) !important; background:transparent !important; }
div[class*="st-key-sfrow_"] div[data-testid="stButton"] > button:hover {
  background:var(--hair2) !important; }
</style>
"""


def render_surface(
    spec: UISpec,
    *,
    header: Mapping[str, Any],
    rows: Sequence[SurfaceRow],
    on_select: Callable[[SurfaceRow], None] | None = None,
    selected: Callable[[SurfaceRow], bool] | None = None,
) -> None:
    """Draw the declared surface into the current Streamlit container.

    ``on_select`` makes the first column a pin button. It is honoured only where
    pinning is declared (``spec.pin_kinds``) and only on rows that carry a
    measurement: a never-scored program has no leading edge to map, so offering to
    pin it would be a control that promises a figure it cannot draw.
    """
    import streamlit as st

    _no_code(spec)
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(header_html(spec, header), unsafe_allow_html=True)

    pinnable = on_select is not None and bool(spec.pin_kinds)
    st.markdown(ribbon_html(spec, rows), unsafe_allow_html=True)
    # The header keeps ALL its columns, pinnable or not, because all three row
    # shapes now put the first column in the same place: a non-observed row spans
    # the full grid, and a pinnable row's button sits in an ``st.columns`` split
    # using the SAME fraction (first / sum-of-rest). A header that skipped the
    # first column matched the rows only while those rows also skipped it -- and
    # once they stopped, it labelled every remaining column one place to the left,
    # putting "glm_coef" over the gene names.
    st.markdown(column_header_html(spec), unsafe_allow_html=True)

    for block in spec.blocks:
        got = rows_for(block, rows)
        st.markdown(block_label_html(block, len(got)), unsafe_allow_html=True)
        if not got:
            if block.empty_text:
                st.markdown(f'<div class="pano-sf-empty">{_esc(block.empty_text)}'
                            "</div>", unsafe_allow_html=True)
            continue
        for row in got:
            if pinnable and row.state is MeasuredState.OBSERVED:
                _render_pinnable_row(st, spec, row, on_select, selected)
            else:
                # NOT skip_first, even when the table is pinnable. A pinnable
                # table draws the first column as a button, which only measured
                # rows get -- but dropping the column from every other row drops
                # the NAME with it, and a never-measured row that does not say
                # WHICH entity was never measured is exactly the blank row the
                # interface contract forbids. The button is withheld here; the
                # identifier never is.
                st.markdown(row_html(spec, row), unsafe_allow_html=True)


def _render_pinnable_row(st, spec: UISpec, row: SurfaceRow, on_select, selected) -> None:
    first = spec.columns[0]
    rest_fr = sum(_fr(c) for c in spec.columns[1:]) or 1
    is_on = bool(selected and selected(row))
    with st.container(key=f"sfrow_{'on' if is_on else 'off'}_{row.key}"):
        c_name, c_rest = st.columns([_fr(first), rest_fr], vertical_alignment="top")
        with c_name:
            if st.button(str(row.values.get(first.key, row.key)),
                         key=f"sfsel_{row.key}",
                         type="primary" if is_on else "secondary",
                         width="stretch"):
                on_select(row)
                st.rerun()
        with c_rest:
            st.markdown(row_html(spec, row, skip_first=True), unsafe_allow_html=True)


__all__ = [
    "SurfaceRow", "NothingToClick", "SpecCarriesCode",
    "render_surface", "header_html", "table_html", "row_html", "cell_html",
    "ribbon_html", "block_html", "column_header_html", "rows_for",
    "label_band_html", "census_html", "checklist_html", "provenance_html",
]
