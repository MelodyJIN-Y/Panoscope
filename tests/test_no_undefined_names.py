"""No function may reference a name that is bound nowhere.

This exists because of a specific failure. Threading a new ``skill`` argument
through the chat loop updated two call sites but not the two enclosing
function signatures, so ``_finalize`` and ``_repair`` referenced a name that
was not a parameter, not a closure variable and not a module global. Every
live-model answer raised ``NameError``, was caught by the broad ``except`` that
protects the chat from crashing, and was silently replaced by the
deterministic fallback.

The whole suite stayed green. The deterministic path does not go through those
functions, and the paths that do are marked ``live`` and need an API key, so
2252 tests could not see it. A silent degradation to the fallback is the worst
shape of bug this project has: it looks like the system working.

``ruff`` and ``mypy`` are declared in the ``dev`` extra and would both catch
this, but neither is installed in the environment the suite actually runs in,
so the check that exists on paper has never run. This one uses the standard
library only, so it cannot be skipped by a missing tool.
"""

from __future__ import annotations

import builtins
import pathlib
import symtable

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "panoscope"

#: Bound by the interpreter in every module, so ``symtable`` cannot see them.
_MODULE_DUNDERS = frozenset({
    "__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__path__", "__debug__",
})


def _unbound(path: pathlib.Path) -> list[tuple[str, str]]:
    """``(qualified function, name)`` for every name a function cannot resolve.

    ``symtable`` marks a name used in a function and assigned nowhere enclosing
    it as global. If it is then absent from the module's own symbols and from
    builtins, nothing binds it and the reference raises at runtime.
    """
    src = path.read_text(encoding="utf-8")
    top = symtable.symtable(src, str(path), "exec")
    module_names = {s.get_name() for s in top.get_symbols()} | _MODULE_DUNDERS

    out: list[tuple[str, str]] = []

    def walk(table, trail: str) -> None:
        for child in table.get_children():
            qual = f"{trail}.{child.get_name()}" if trail else child.get_name()
            if child.get_type() == "function":
                for sym in child.get_symbols():
                    name = sym.get_name()
                    if (sym.is_global() and name not in module_names
                            and not hasattr(builtins, name)):
                        out.append((qual, name))
            walk(child, qual)

    walk(top, "")
    return out


def _modules() -> list[pathlib.Path]:
    return sorted(PKG.rglob("*.py"))


def test_the_package_has_modules_to_check():
    """Guards the whole file: a bad glob would make every assertion below vacuous."""
    mods = _modules()
    assert len(mods) > 50, f"only {len(mods)} modules found under {PKG}"
    assert any(m.name == "loop.py" for m in mods)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_function_references_an_unbound_name(path: pathlib.Path):
    bad = _unbound(path)
    assert not bad, (
        f"{path.relative_to(PKG.parent)} references names nothing binds:\n  "
        + "\n  ".join(f"{fn}: {name}" for fn, name in bad)
    )


def test_the_check_catches_the_bug_it_was_written_for(tmp_path):
    """The check itself, checked. An invariant nobody has seen fail is a guess."""
    good = tmp_path / "good.py"
    good.write_text(
        "def outer(skill):\n"
        "    def inner():\n"
        "        return skill\n"
        "    return inner()\n"
    )
    assert _unbound(good) == []

    bad = tmp_path / "bad.py"
    bad.write_text(
        "def finalize(text):\n"
        "    return assemble(text, skill)\n"
        "def assemble(a, b):\n"
        "    return a\n"
    )
    assert ("finalize", "skill") in _unbound(bad)
