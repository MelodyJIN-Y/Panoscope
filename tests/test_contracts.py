"""Contract invariants, written BEFORE ``core/`` exists.

That ordering is deliberate. These tests inspect *signatures* and *source*, not
behaviour, so they are live the moment the first ``core/`` module lands -- which
means it lands already conforming instead of being retrofitted. Until then most of
them pass over an empty package, and ``test_the_core_scan_is_wired`` keeps that from
being mistaken for coverage.

The rule they exist to defend: **decisions flow out, never in.** A core function
returns ``Traced[T]`` and accepts no ``Trace``, so there is no seam through which a
caller could inject a recorder into the analysis layer.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import typing
from pathlib import Path

import pytest

from panoscope.types.trace import (
    RULES, Decision, Trace, Traced, UnregisteredRule, fmt_num, ref,
)
from panoscope.types.subject import gene, subject

PKG = Path(__file__).resolve().parents[1] / "panoscope"
CORE = PKG / "core"


def _core_modules():
    """Every module under ``panoscope.core``; empty until the package exists."""
    if not CORE.is_dir():
        return []
    out = []
    for m in pkgutil.walk_packages([str(CORE)], prefix="panoscope.core."):
        out.append(importlib.import_module(m.name))
    return out


def _public_functions(mod):
    for name, fn in vars(mod).items():
        if name.startswith("_") or not inspect.isfunction(fn):
            continue
        if fn.__module__ != mod.__name__:      # re-export, not this module's contract
            continue
        yield name, fn


# --------------------------------------------------------------------------- #
# The direction rule
# --------------------------------------------------------------------------- #
def test_every_core_function_returns_traced():
    """A core function that returns a bare value has dropped its reasoning on the
    floor, and the record cannot be reconstructed afterwards."""
    bad = []
    for mod in _core_modules():
        for name, fn in _public_functions(mod):
            hints = typing.get_type_hints(fn)
            ret = hints.get("return")
            if typing.get_origin(ret) is not Traced and ret is not Traced:
                bad.append(f"{mod.__name__}.{name} -> {ret!r}")
    assert not bad, "core functions not returning Traced[...]: " + "; ".join(bad)


def _mentions_trace(t) -> bool:
    """True if ``Trace`` appears anywhere in an annotation, including inside a union.

    The naive ``t is Trace`` missed ``Trace | None = None`` -- which is not a corner
    case, it is the shape the seam would ACTUALLY take. Nobody adds a required Trace
    parameter to an existing signature; they add an optional one with a default, and
    every call site keeps working. That version passed the first draft of this test.
    """
    if t is Trace:
        return True
    return any(_mentions_trace(a) for a in typing.get_args(t))


def test_no_core_function_accepts_a_trace():
    """The whole design. A Trace parameter is an injection seam: a caller could pass
    in a recorder that narrates, and ``test_record_not_generated`` would then rest on
    nobody having exploited it."""
    bad = []
    for mod in _core_modules():
        for name, fn in _public_functions(mod):
            hints = typing.get_type_hints(fn)
            for param, t in hints.items():
                if param != "return" and _mentions_trace(t):
                    bad.append(f"{mod.__name__}.{name}({param}: {t})")
    assert not bad, "core functions accepting a Trace: " + "; ".join(bad)


def test_the_trace_detector_sees_through_a_union():
    """Pins the hole above, since no core function currently has one to catch."""
    assert _mentions_trace(Trace)
    assert _mentions_trace(typing.Optional[Trace])
    assert _mentions_trace(Trace | None)
    assert _mentions_trace(dict[str, Trace | None])
    assert not _mentions_trace(int)
    assert not _mentions_trace(Traced[int] | None)


def test_the_core_scan_is_wired():
    """Guard against silent vacuity.

    Every test above passes trivially while ``core/`` is empty. This one fails the
    moment the package exists but the scan finds nothing in it -- which would mean
    the walk broke, not that the code is clean.
    """
    if not CORE.is_dir():
        pytest.skip("panoscope/core/ does not exist yet (expected at this stage)")
    mods = _core_modules()
    assert mods, "panoscope/core/ exists but the module scan found nothing"
    assert any(list(_public_functions(m)) for m in mods), (
        "panoscope/core/ has modules but no public functions -- the scan is not biting"
    )


# --------------------------------------------------------------------------- #
# The rule registry
# --------------------------------------------------------------------------- #
def test_every_decision_rule_is_registered():
    """Static scan: every literal rule id passed to ``.note(...)`` anywhere in the
    package is in RULES. An unregistered id means the record carries a decision that
    nothing can explain, render or test against -- which is how English crept back
    into the matching logic last time."""
    unknown = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "note"):
                continue
            lit = next(
                (a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)),
                next((kw.value.value for kw in node.keywords
                      if kw.arg == "rule" and isinstance(kw.value, ast.Constant)), None),
            )
            if lit is not None and lit not in RULES:
                unknown.append(f"{path.relative_to(PKG)}:{node.lineno} {lit!r}")
    assert not unknown, "unregistered decision rules: " + "; ".join(unknown)


def test_registry_is_self_consistent():
    for rid, rule in RULES.items():
        assert rule.id == rid, f"RULES key {rid!r} != Rule.id {rule.id!r}"
        assert rule.step and rule.what, f"{rid} is missing step or description"


def test_an_unregistered_rule_raises():
    with pytest.raises(UnregisteredRule):
        Trace().note("measure:invented_just_now", "x")


# --------------------------------------------------------------------------- #
# Determinism -- these strings are hashed into the golden record
# --------------------------------------------------------------------------- #
def test_decision_order_is_code_order():
    """Append, splice on absorb, never sort. A caller's own decisions made before a
    call must precede the callee's, and the callee's must precede what follows."""
    inner = Trace()
    inner.note("measure:observed", "OBSERVED")
    inner.note("measure:coverage_floor", "UNDERPOWERED")

    outer = Trace()
    outer.note("band:from_coef", "High")
    outer.absorb(inner.done(None))
    outer.note("band:fragile_cap", "Medium-High")

    assert [d.rule for d in outer.done(None).decisions] == [
        "band:from_coef", "measure:observed", "measure:coverage_floor",
        "band:fragile_cap",
    ]


def test_absorb_returns_the_callee_value():
    t = Trace()
    assert t.absorb(Traced(value=42, decisions=())) == 42


def test_decisions_contain_no_wallclock():
    """Two halves: the type has no timestamp field, and core may not reach for one.

    A clock in a ``because`` string makes the golden hash move on every run, which
    would destroy the reproducibility claim the record exists to make.
    """
    assert not any(
        f in Decision.__slots__ for f in ("at", "when", "timestamp", "created_at")
    ), "Decision grew a timestamp field"

    banned = {"datetime", "time", "date"}
    offenders = []
    for path in (CORE.rglob("*.py") if CORE.is_dir() else []):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{path.name}: import {a.name}" for a in node.names
                              if a.name.split(".")[0] in banned]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in banned:
                    offenders.append(f"{path.name}: from {node.module}")
    assert not offenders, "clock access in core/: " + "; ".join(offenders)


def test_fmt_num_is_fixed_precision_and_total():
    assert fmt_num(21.43781) == "21.4378"
    assert fmt_num(21.4) == "21.4000"          # padded, so widths never shift
    assert fmt_num(None) == "na"
    assert fmt_num(3) == "3"                   # ints stay exact; counts are not 3.0000
    assert fmt_num(-0.00001) == "0.0000"       # never "-0.0000"
    assert fmt_num(float("nan")) == "nan"
    assert fmt_num(float("inf")) == "inf"


def test_ref_anchors_on_the_join_key():
    """A ref must resolve BACK to its subject, or click-back cannot work for any
    shape with more than one role."""
    s = subject("gene", g=gene("ERBB2"))
    assert ref(s, "glm_coef", 21.43781) == f"{s.key}#glm_coef=21.4378"
    assert ref(s) == s.key
    assert ref("claim:abc123") == "claim:abc123"


def test_traced_is_frozen_and_decisions_are_hashable():
    d = Decision("measure:observed", "OBSERVED", ("in panel",), ("gene|g=gene:X",))
    hash(d)
    with pytest.raises((AttributeError, TypeError)):
        d.outcome = "mutated"                   # type: ignore[misc]
