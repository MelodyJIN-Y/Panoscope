"""Platform adapters: how one technology's store answers the questions
interpretation depends on.

**The registry is a dict of import paths, not of adapters**: an adapter carries its
platform's read dependencies, so :data:`ADAPTERS` maps a platform id to a
``"module:attr"`` string and :func:`get` imports it on demand, verifying the resolved
object declares the ``name`` it was registered under. Adding a platform is one entry
plus the adapter module.

**Nothing here runs at import** (``tests/test_import_surface.py``): no dataset file
is read and no study is resolved at module scope.
"""

from __future__ import annotations

from typing import Any

from panoscope.types.uispec import DuplicateRegistration

__all__ = ["ADAPTERS", "UnregisteredPlatform", "for_study", "get", "names",
           "register"]


class UnregisteredPlatform(LookupError):
    """Raised for a platform nobody registered. Never defaulted to another one:
    the adapter decides the capability frames, so a wrong one would answer about a
    panel that was never run.
    """


#: Platform id -> ``"module:attribute"``, resolved on first use by :func:`get`.
#: The key is the adapter's own declared ``name``, case-folded, and it is what a
#: study manifest's ``platform`` field is matched against.
ADAPTERS: dict[str, str] = {
    "xenium": "panoscope.platforms.xenium:XeniumAdapter",
}


def register(name: str, path: str) -> None:
    """Add one platform. Raises on a collision; never overwrites.

    ``DuplicateRegistration`` is reused from ``types/uispec.py`` rather than
    redefined, exactly as the workflow registry reuses it: one exception for one
    failure, so a caller can catch registration collisions across every registry
    with one name.
    """
    key = str(name).strip().lower()
    if key in ADAPTERS:
        raise DuplicateRegistration(
            f"platform {key!r} is already registered ({ADAPTERS[key]}); two "
            "registrations for one id are resolved by import order, which is "
            "decided by nothing"
        )
    ADAPTERS[key] = path


def names() -> tuple[str, ...]:
    """Registered ids, sorted -- dict order is import order, which is not a fact
    about anything."""
    return tuple(sorted(ADAPTERS))


def get(name: str) -> Any:
    """A fresh adapter for ``name``. Imports its module on first use.

    A new instance per call rather than a cached singleton: adapters are stateless
    readers whose caches are keyed on the store, so sharing one would buy nothing
    and would let a test that monkeypatches an adapter leak into the next.
    """
    key = str(name).strip().lower()
    try:
        path = ADAPTERS[key]
    except KeyError:
        raise UnregisteredPlatform(
            f"no adapter registered for platform {key!r}; known: "
            f"{', '.join(names()) or '(none)'}. The adapter decides the capability "
            "frames, so there is no safe default to fall back to"
        ) from None

    from importlib import import_module

    module, _, attr = path.partition(":")
    adapter = getattr(import_module(module), attr)()
    declared = str(getattr(adapter, "name", "")).strip().lower()
    if declared != key:
        raise UnregisteredPlatform(
            f"{path} declares name {declared!r} but is registered as {key!r}; the "
            "registry would hand this study the wrong technology's reader"
        )
    return adapter


def for_study(store: Any) -> Any:
    """The adapter this store's manifest declares. Raises rather than guessing.

    Platform blocks the study (§6.3); it is never inferred from a filename.
    """
    declared = getattr(store, "platform", None)
    if not declared:
        raise UnregisteredPlatform(
            f"study {getattr(store, 'id', '?')!r} declares no platform in its "
            "manifest, so no adapter can be chosen. Platform is one of the four "
            "facts ARCHITECTURE 6.3 blocks the study on; it is not inferred"
        )
    return get(str(declared))
