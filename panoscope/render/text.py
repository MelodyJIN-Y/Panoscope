"""The second renderer: the same ``UISpec``, in plain text.

``ui/surface.py`` reads a declared surface and emits HTML for Streamlit. It is the
only thing that has ever read a ``UISpec``, and while that is true, *"the review
surface is declared by the workflow, not a UI concern"* (ARCHITECTURE §13) and
*"the review surface is hand-built in exactly one place"* produce identical
evidence. This module is the difference: a renderer in a different medium, sharing
no code with the first, consuming the same declaration and the same rows.

It imports nothing from ``panoscope.ui`` -- deliberately, and
``tests/test_render_text.py`` pins it. A wrapper around the HTML renderer would
prove that the HTML renderer is reusable, which was never in doubt; what is in
doubt is whether the *declaration* is enough to render from. So the glyph table,
the number formatting, the hatching rule and the four guards are written again
here from the declaration alone. Where the two renderers agree, they agree because
the spec said so; where they would disagree, the cross-renderer test fails.

**The four properties are of the declaration, not of the medium**, so each is
enforced here independently rather than inherited:

1. *Degradation goes DOWN.* :func:`label_band_text` refuses a surface whose task
   kind has no label space; :func:`provenance_text` refuses one that declared a
   task kind.
2. *A never-measured cell is neither blank nor zero.* In a terminal there is no
   hover, so the cell is bracketed -- ``[⊘ never measured]`` -- and the reason the
   HTML puts in a ``title`` attribute takes its own continuation line. That is the
   one thing the HTML surface can say in a place text does not have.
3. *Click-back.* Every row line ends in its ``Subject.key``. A row with no key
   raises rather than printing as unanchored prose, which in a terminal looks
   exactly like anchored prose.
4. *Plots render, they do not derive.* :func:`_no_code` refuses a declaration
   carrying a callable before a line is built.

**What it does not do.** No plots: a ``PlotSpec`` is a promise to draw a figure,
and none of the three workflows declares one (R4 -- no conditionality builder has
landed). If one did, this renderer would have to print the conditionality caption
and refuse the figure, which is the honest terminal rendering of a plot and is not
written until there is a plot to refuse.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

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
    ProvenanceOnlyHeader,
    UISpec,
)
# Derived in `uispec.py` from `Measured.carries_number`, imported rather than
# re-listed for the reason the HTML renderer imports them: a second copy would let
# a widened property move the hatching in one renderer and not the other -- which
# is exactly the disagreement this module exists to make visible.
from panoscope.types.uispec import _NO_NUMBER, _NUMERIC_KINDS


class NothingToClick(ValueError):
    """Raised for a rendered claim that resolves to nothing.

    The same rule as ``ui.surface.NothingToClick`` and deliberately NOT that class:
    importing it would make this renderer depend on the other one. That the
    exception type for a property of the *declaration* lives in a renderer at all
    is a smell the report names; it is not fixable from here, because ``types/`` is
    frozen.
    """


class SpecCarriesCode(ValueError):
    """Raised when a declaration handed to this renderer carries a callable."""


class Row(Protocol):
    """What a renderer needs of a row, structurally.

    Not a new dataclass. ``ui.surface.SurfaceRow`` already has exactly this shape,
    and defining a second row type would mean the cross-renderer test feeds the two
    renderers two different objects -- at which point it is comparing adapters, not
    renderers. This way both are handed the same rows.
    """

    key: str
    state: MeasuredState
    values: Mapping[str, Any]
    reason: str


# ● measured  ◐ measured, not here  ◌ below the floor  ⊘ never measured.
# The same four glyphs and the same four labels the HTML renderer emits as entities
# -- written again here from the four states, not imported. The cross-renderer test
# is what holds them to the same wording.
_GLYPH: dict[MeasuredState, tuple[str, str]] = {
    MeasuredState.OBSERVED: ("●", "observed"),
    MeasuredState.TESTED_ABSENT: ("◐", "measured, not here"),
    MeasuredState.UNDERPOWERED: ("◌", "too little signal"),
    MeasuredState.NEVER_MEASURED: ("⊘", "never measured"),
}

# The ribbon's fill per state, darkest for the thing actually measured.
_FILL: dict[MeasuredState, str] = {
    MeasuredState.OBSERVED: "█",
    MeasuredState.TESTED_ABSENT: "▓",
    MeasuredState.UNDERPOWERED: "▒",
    MeasuredState.NEVER_MEASURED: "░",
}

SEP: str = " | "          # the column separator; a text table's only chrome
ANCHOR: str = "  ← "  # the click-back: a row line ends in its Subject.key
RIBBON_WIDTH: int = 40
MAX_WIDTH: int = 34       # wider than every glyph and hatch string, by construction


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def _no_code(spec: UISpec) -> None:
    """Refuse a declaration that carries a computation, before a line is built."""
    for col in spec.columns:
        for name in col.__slots__:
            if callable(getattr(col, name, None)):
                raise SpecCarriesCode(
                    f"{spec.workflow_id}: column {col.key!r} carries a callable on "
                    f"{name!r}. Columns render; they do not derive, and a renderer "
                    "that can run a rule can decide what counts as measured."
                )


def _anchored(line: str, key: str, what: str) -> str:
    """End a rendered claim in the key it resolves to, or refuse to render it."""
    if not str(key).strip():
        raise NothingToClick(
            f"a {what} was rendered with no Subject.key to resolve to. A sentence "
            "with nothing to click should not have been rendered."
        )
    return f"{line}{ANCHOR}{key}"


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def _num(value: Any, digits: int) -> str:
    """Fixed decimals, so a column aligns and a rounded value is not a lie."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if f != f else f"{f:.{digits}f}"


def _number(col: ColumnSpec, value: Any) -> str:
    if col.kind is ColumnKind.SPECIFICITY_PAIR and isinstance(value, (tuple, list)):
        return " / ".join(_num(v, col.precision) for v in value)
    if col.kind is ColumnKind.COVERAGE_BAR and isinstance(value, (tuple, list)):
        return "/".join(str(v) for v in value)
    return _num(value, col.precision)


def _hatch(row: Row) -> str:
    """The cell for a row that has no number. Bracketed: not blank, not a zero.

    The brackets are the terminal's hatching. A blank cell reads as *we looked and
    found nothing*; ``[⊘ never measured]`` reads as *we could not look*, and the
    difference is the difference between evidence and no evidence.
    """
    glyph, label = _GLYPH[row.state]
    return f"[{glyph} {label}]"


def cell_text(col: ColumnSpec, row: Row) -> str:
    """One cell, unpadded. The hatching rule lives here and nowhere else."""
    value = row.values.get(col.key)

    if col.kind is ColumnKind.GLYPH:
        glyph, label = _GLYPH[row.state]
        return f"{glyph} {label}"

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
        return _number(col, value)

    if value is None or value == "" or value == ():
        return "—"

    if col.kind is ColumnKind.CITATION:
        text, href = value if isinstance(value, tuple) else (value, "")
        tail = href.rstrip("/").rsplit("/", 1)[-1] if href else ""
        return f"{text} ({tail})" if tail else str(text)

    if col.kind is ColumnKind.NOTE_CHIP:
        chips = value if isinstance(value, (tuple, list)) else (value,)
        return ", ".join(str(c) for c in chips)

    if col.kind is ColumnKind.DIRECTION:
        arrow = {"up": "▲", "down": "▼"}.get(str(value).lower(), "―")
        return f"{arrow} {value}"

    return str(value)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _pad(text: str, width: int, align: str) -> str:
    return text.rjust(width) if align == "right" else text.ljust(width)


# --------------------------------------------------------------------------- #
# Rows and blocks
# --------------------------------------------------------------------------- #
def row_text(spec: UISpec, row: Row,
             widths: Sequence[int] = ()) -> list[str]:
    """One row: the cell line, anchored, plus the reason line where a cell hatched.

    Two lines rather than a hover. The reason is the half of the hatch that says
    *why we could not look*, and a renderer that drops it leaves a row saying
    "never measured" with no way to find out what frame refused it.
    """
    cells = [cell_text(c, row) for c in spec.columns]
    w = tuple(widths) or tuple(len(c) for c in cells)
    line = SEP.join(
        _pad(_clip(cell, w[i]), w[i], col.align)
        for i, (col, cell) in enumerate(zip(spec.columns, cells))
    )
    out = [_anchored("  " + line.rstrip(), row.key, "row")]
    if row.reason and any(c.startswith("[") for c in cells):
        out.append(f"      reason: {row.reason}")
    return out


def rows_for(block: BlockSpec, rows: Sequence[Row]) -> list[Row]:
    """This block's rows, sorted WITHIN the block. Never truncated, never merged.

    Written again rather than imported, and it has to produce the same order as the
    HTML renderer's: a text surface that sorted differently would be a second
    opinion about which result is the headline.
    """
    got = [r for r in rows if r.state in block.statuses]
    if block.sort_by:
        got.sort(key=lambda r: (r.values.get(block.sort_by) is None,
                                r.values.get(block.sort_by) or 0),
                 reverse=block.sort_desc)
    return got


def column_header_text(spec: UISpec, widths: Sequence[int]) -> list[str]:
    labels = [f"{c.label} ({c.unit})" if c.unit else c.label for c in spec.columns]
    line = SEP.join(
        _pad(_clip(lab, widths[i]), widths[i], spec.columns[i].align)
        for i, lab in enumerate(labels)
    )
    return ["  " + line.rstrip(), "  " + "-" * len(line.rstrip())]


def block_text(spec: UISpec, block: BlockSpec, rows: Sequence[Row],
               widths: Sequence[int]) -> list[str]:
    """A block header and its rows -- the header even at zero, with its empty text.

    A block that disappears at zero is the panel-absence rule disappearing at the
    moment it had something to say.
    """
    got = rows_for(block, rows)
    out = ["", block.label_template.format(count=len(got))]
    if not got:
        if block.empty_text:
            out.append(f"  ({block.empty_text})")
        return out
    for row in got:
        out.extend(row_text(spec, row, widths))
    return out


def ribbon_text(spec: UISpec, rows: Sequence[Row]) -> list[str]:
    """The table's own conditionality: one stacked bar, every count labelled.

    What stops a screen of twelve supporting rows from reading as twelve of twelve.
    The bar is decoration; the counts beneath it are the claim, which is why they
    are spelled out rather than left to be read off the widths.
    """
    if not spec.coverage_ribbon.enabled or not rows:
        return []
    total = len(rows)
    bar, keys = "", []
    for state in spec.coverage_ribbon.order:
        n = sum(1 for r in rows if r.state is state)
        if not n:
            continue
        bar += _FILL[state] * max(1, round(RIBBON_WIDTH * n / total))
        glyph, label = _GLYPH[state]
        keys.append(f"{glyph} {n} {label}")
    return ["  " + bar, "  " + " · ".join(keys) + f" · of {total}"]


def _widths(spec: UISpec, rows: Sequence[Row]) -> tuple[int, ...]:
    """One width per declared column: the widest thing that has to fit in it."""
    out = []
    for i, col in enumerate(spec.columns):
        longest = len(f"{col.label} ({col.unit})" if col.unit else col.label)
        for row in rows:
            longest = max(longest, len(cell_text(col, row)))
        out.append(min(max(longest, 3), MAX_WIDTH))
    return tuple(out)


def table_text(spec: UISpec, rows: Sequence[Row]) -> list[str]:
    """The ribbon, the column header and every block. Declaration order throughout."""
    _no_code(spec)
    w = _widths(spec, rows)
    out = ribbon_text(spec, rows) + column_header_text(spec, w)
    for block in spec.blocks:
        out.extend(block_text(spec, block, rows, w))
    return out


# --------------------------------------------------------------------------- #
# Headers. One function per variant, each refusing the task kinds it may not serve.
# --------------------------------------------------------------------------- #
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


def _field(label: str, body: str) -> str:
    return f"  {label:<13}{body}"


def _idline(values: Mapping[str, Any]) -> list[str]:
    scope = values.get("unit_scope") or ""
    return [str(scope)] if scope else []


def _gates(values: Any) -> list[str]:
    """The bars a result had to clear, named. A gate nobody can read is a rumour."""
    items = _seq(values)
    return [_field("gates", " · ".join(str(g) for g in items))] if items else []


def label_band_text(spec: UISpec, values: Mapping[str, Any]) -> list[str]:
    """ASSIGNMENT only. The band, its driver, and the rejected alternative."""
    h = spec.header
    if not isinstance(h, LabelBandHeader) or spec.task_kind is not TaskKind.ASSIGNMENT:
        raise DegradedUpward(
            f"{spec.workflow_id}: {spec.task_kind} has no label space, so a "
            "unit-level confidence band is a category error, not merely unhelpful. "
            "Degradation goes toward making no claim, never toward borrowing one."
        )
    out = _idline(values)
    band = values.get(h.band_field) or ""
    verify = "  ⚑ re-check this" if values.get(h.verify_field) else ""
    out.append(f"{values.get('label') or ''}"
               + (f"   [{band}]" if band else "") + verify)
    audit = " · ".join(
        str(x) for f in h.band_audit_fields for x in _seq(values.get(f)))
    if audit:
        out.append(_field("band basis", audit))
    drivers = ", ".join(str(d) for d in _seq(values.get(h.driver_field)))
    if drivers:
        out.append(_field("driven by", drivers))
    if values.get("runner_up"):
        out.append(_field("rejected", str(values["runner_up"])))
    return out + _gates(values.get("gates"))


def census_text(spec: UISpec, values: Mapping[str, Any]) -> list[str]:
    """CHARACTERISATION-shaped. Per-item counts, each with its denominator; no band.

    There is no band slot to fill, which is the structural form of the rule rather
    than a reminder about it.
    """
    h = spec.header
    if not isinstance(h, CensusHeader):
        raise DegradedUpward(f"{spec.workflow_id}: census_text called on a "
                             f"{spec.header.kind.value} header")
    out = _idline(values)
    for f in h.imported_context_fields:
        if values.get(f):
            out.append(f"{values[f]}   ({h.imported_context_note})")
    for line in h.lines:
        counts = " · ".join(
            f"{values.get(fieldname, 0)} {label}" for label, fieldname in line.counts)
        denom = line.denominator_template.format(
            value=values.get(line.denominator_field, "?"))
        out.append(f"  {line.label:<11}{counts}   {denom}")
    top = _fill(h.top_item_template, values)
    if top:
        # The one header sentence about a specific item, so it carries an anchor.
        out.append(_anchored("  " + top, values.get("top_key", ""),
                             "header sentence"))
    if values.get(h.scope_line_field):
        out.append(f"  {values[h.scope_line_field]}")
    return out + _gates(values.get(h.gates_field))


def checklist_text(spec: UISpec, values: Mapping[str, Any]) -> list[str]:
    """COMPARISON-shaped. The checks and what blocked them; no band here either."""
    h = spec.header
    if not isinstance(h, ConfoundChecklistHeader):
        raise DegradedUpward(f"{spec.workflow_id}: checklist_text called on a "
                             f"{spec.header.kind.value} header")
    out = _idline(values)
    for template in (h.design_template, h.results_template,
                     h.testable_denominator_template):
        line = _fill(template, values)
        if line:
            out.append(f"  {line}")
    for c in h.checks:
        state = values.get(c.state_field, "not assessed")
        detail = values.get(c.detail_field, "")
        out.append(f"  [{state:^16}] {c.label}" + (f" -- {detail}" if detail else ""))
    if values.get(h.scope_line_field):
        out.append(f"  {values[h.scope_line_field]}")
    return out


def provenance_text(spec: UISpec, values: Mapping[str, Any]) -> list[str]:
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
    out = [f"! {h.banner}", "", str(spec.title)]
    prov = " · ".join(str(x) for x in _seq(values.get("provenance")))
    return out + ([f"  {prov}"] if prov else [])


_HEADERS = {
    HeaderKind.LABEL_BAND: label_band_text,
    HeaderKind.CENSUS: census_text,
    HeaderKind.CONFOUND_CHECKLIST: checklist_text,
    HeaderKind.PROVENANCE_ONLY: provenance_text,
}


def header_text(spec: UISpec, values: Mapping[str, Any]) -> list[str]:
    """The declared header for this surface. Dispatch only; the guards are above."""
    _no_code(spec)
    return _HEADERS[spec.header.kind](spec, values)


# --------------------------------------------------------------------------- #
# The whole surface
# --------------------------------------------------------------------------- #
def render(spec: UISpec, *, header: Mapping[str, Any],
           rows: Sequence[Row]) -> str:
    """One declared surface, as a block of text ready for a terminal."""
    lines = header_text(spec, header)
    lines.append("=" * max((len(x) for x in lines), default=0))
    lines.extend(table_text(spec, rows))
    return "\n".join(lines)


__all__ = [
    "Row", "NothingToClick", "SpecCarriesCode",
    "render", "header_text", "table_text", "row_text", "cell_text",
    "ribbon_text", "block_text", "column_header_text", "rows_for",
    "label_band_text", "census_text", "checklist_text", "provenance_text",
]
