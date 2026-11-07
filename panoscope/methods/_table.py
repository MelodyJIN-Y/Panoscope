"""Neutral table readers shared by the method modules.

Holds no method schema: each caller names its own expected table in the refusal,
so the error still says which method could not read what.
"""

from __future__ import annotations

from typing import Any

from panoscope.types.contracts import UnrecognisedResult


def columns(table: Any, expected: str) -> tuple[str, ...]:
    """Column names off anything table-shaped, without importing a table library.

    ``MethodModule.detect_unit`` takes ``Any`` on purpose -- a method registered out
    of tree may hand up a polars frame, a list of dicts or an arrow table -- so this
    reads the two shapes that actually occur and refuses the rest rather than
    duck-typing its way into a wrong answer. ``expected`` finishes the refusal.
    """
    cols = getattr(table, "columns", None)
    if cols is not None:
        return tuple(str(c) for c in cols)
    if isinstance(table, (list, tuple)) and table and isinstance(table[0], dict):
        return tuple(str(k) for k in table[0])
    raise UnrecognisedResult(
        f"cannot read column names off a {type(table).__name__}; {expected}"
    )


def rows(table: Any) -> tuple[dict[str, Any], ...]:
    """Rows as plain dicts, in the table's own order. Never re-sorted: the order is
    the method's, and re-sorting here would be a second, invented one."""
    to_dict = getattr(table, "to_dict", None)
    if to_dict is not None:
        return tuple(to_dict(orient="records"))
    return tuple(dict(r) for r in table)


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)
