"""The generic renderer: a ``UISpec`` in, the review surface out.

ARCHITECTURE §13: the review surface is declared by the workflow. Pure ``*_html``
functions decide every cell (testable without a server); :func:`render_surface`
is the only Streamlit entry point.

Rules enforced here:

1. *Degradation goes down.* :func:`label_band_html` refuses a task kind with no
   label space; :func:`provenance_html` refuses one that declared a task kind.
2. *A never-measured item is not a blank row and not a zero.* A numeric cell with
   no number hatches; a number on a row whose state forbids one is refused.
3. *Click-back.* Every row and claim-bearing header sentence carries a
   ``Subject.key`` anchor, or it raises.
4. *Plots render, they do not derive.* :func:`_no_code` refuses a computation in
   the spec before any cell is drawn.

State, values, block order and sort are all handed in. Sorting is within a block,
never across, so absences cannot sink below a top-N.
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
# Imported, never re-listed: uispec.py derives both from `Measured.carries_number`.
from panoscope.types.uispec import _NO_NUMBER, _NUMERIC_KINDS
from panoscope.ui import format as fmt


# --------------------------------------------------------------------------- #
# The row. Resolved server-side: the renderer is handed a state, never a rule.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SurfaceRow:
    """One item row, already adjudicated.

    ``key`` is a ``Subject.key`` (the click-back anchor). ``values`` maps a declared
    ``ColumnSpec.key`` to a resolved value; ``None`` means no number, never zero.
    """

    key: str
    state: MeasuredState
    values: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""          # why this state -- the hover on the hatch
    #: The capability frame that produced ``state`` (``Measured.frame``), its own
    #: field so ``BlockSpec.frame_prefix`` never parses prose.
    frame: str = ""


# ● measured  ◐ measured, not here  ◌ below the floor  ⊘ never measured.
# Four glyphs for four states; never collapse to two.
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
    # Numeric columns must fit the hatch text ("◐ measured, not here"), not just a number.
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
    """Wrap a rendered claim in its click-back anchor (``uispec.anchored``), or
    refuse to render it."""
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
    """The shared track list. ``minmax(0, Nfr)``, never a bare ``Nfr``: a bare
    ``1fr`` sizes to content, so the header and row grids would not line up."""
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


def method_limits_html(spec: UISpec) -> str:
    """What the methods registered into this workflow do **not** measure
    (``does_not_measure``), shown where the number is read. ``""`` when none.

    The surface id is looked up through its workflow, never restated as a mapping.
    """
    from panoscope import methods, workflows

    owner = next((w for w in workflows.WORKFLOWS.values()
                  if w.uispec().workflow_id == spec.workflow_id), None)
    if owner is None:
        return ""

    out: list[str] = []
    for m in methods.for_workflow(owner.name):
        name = _esc(getattr(m, "display_name", None) or m.name)
        parts = [f'<span class="pano-sf-ml-name">{name}</span>'
                 f'<span class="pano-sf-ml-stat">{_esc(m.statistic_name)}</span>']
        measures = str(getattr(m, "measures", "") or "").strip()
        if measures:
            stop = "" if measures.endswith((".", "!", "?")) else "."
            parts.append(f'<div class="pano-sf-ml-is">Measures {_esc(measures)}{stop}</div>')
        parts.append('<div class="pano-sf-ml-not"><b>Does not measure</b> '
                     f'{_esc(str(m.does_not_measure).strip())}</div>')
        for a in getattr(m, "assumptions", ()) or ():
            parts.append(f'<div class="pano-sf-ml-as">&#8226; {_esc(str(a).strip())}</div>')
        out.append(f'<div class="pano-sf-ml-item">{"".join(parts)}</div>')
    if not out:
        return ""
    return ('<div class="pano-sf-ml"><div class="pano-sf-block">'
            "What produced these numbers</div>" + "".join(out) + "</div>")


def block_html(spec: UISpec, block: BlockSpec, rows: Sequence[SurfaceRow]) -> str:
    """A block header and its rows -- the header even at zero, with its empty text."""
    got = rows_for(block, rows)
    out = [block_label_html(block, len(got))]
    if not got:
        out.append(f'<div class="pano-sf-empty">{_esc(block.empty_text)}</div>'
                   if block.empty_text else "")
    else:
        out.extend(row_html(spec, r) for r in got)
    return "".join(out)


def ribbon_html(spec: UISpec, rows: Sequence[SurfaceRow]) -> str:
    """The table's own conditionality: one stacked bar, every count labelled."""
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


def column_header_html(spec: UISpec) -> str:
    cells = "".join(
        f'<div class="pano-sf-cell a-{c.align}">{_esc(c.label)}'
        + (f'<span class="pano-sf-unit">{_esc(c.unit)}</span>' if c.unit else "")
        + "</div>"
        for c in spec.columns
    )
    grid = _grid_style(spec.columns)
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
    """ASSIGNMENT only (refuses any other task kind). The band, its driver, and the
    rejected alternative."""
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
    """CHARACTERISATION-shaped. Per-item counts, each with its denominator; no band."""
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
    """The degraded header. A banner and the file -- no claim of any kind. Refuses a
    spec that declared a task kind."""
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
/* max-width for the hatch's reason, one column over: an inline-block takes its
   max-content width and PAINTS past the cell without extending its scroll box,
   so the cell reports no overflow and a check on scrollWidth sees nothing.
   "low spatial specificity" spilled 7.5px out of a 64px CAVEATS cell at 1024. */
.pano-sf-chip { display:inline-block; max-width:100%; font-family:var(--mono);
  font-size:9px;
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
.pano-sf-ml-item { padding:4px 0 2px; }
.pano-sf-ml-name { font-family:var(--mono); font-size:11px; color:var(--ink); }
.pano-sf-ml-stat { font-family:var(--mono); font-size:10px; color:var(--faint);
  margin-left:6px; }
.pano-sf-ml-is { font-size:11px; line-height:1.5; color:var(--muted); margin-top:2px; }
.pano-sf-ml-not { font-size:11px; line-height:1.5; color:var(--ink); margin-top:2px; }
.pano-sf-ml-not b { color:var(--absent); font-weight:600; }
.pano-sf-ml-as { font-size:10px; line-height:1.5; color:var(--faint);
  margin-left:8px; }
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
    header: Mapping[str, Any] | None,
    rows: Sequence[SurfaceRow],
    on_select: Callable[[SurfaceRow], None] | None = None,
    selected: Callable[[SurfaceRow], bool] | None = None,
) -> None:
    """Draw the declared surface into the current Streamlit container.

    ``on_select`` makes the first column a pin button, only where ``spec.pin_kinds``
    declares it and only on rows that carry a measurement. ``header=None``: the caller
    places the header (the marker proposal sits one tap behind the evidence).
    """
    import streamlit as st

    _no_code(spec)
    st.markdown(_CSS, unsafe_allow_html=True)
    if header is not None:
        st.markdown(header_html(spec, header), unsafe_allow_html=True)

    pinnable = on_select is not None and bool(spec.pin_kinds)
    st.markdown(ribbon_html(spec, rows), unsafe_allow_html=True)
    # The header keeps all its columns; every row shape puts the first column in
    # the same place.
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
                # Not skip_first: an unpinnable row withholds the button, never
                # the entity's name.
                st.markdown(row_html(spec, row), unsafe_allow_html=True)

    # Under the numbers it qualifies.
    limits = method_limits_html(spec)
    if limits:
        st.markdown(limits, unsafe_allow_html=True)


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
