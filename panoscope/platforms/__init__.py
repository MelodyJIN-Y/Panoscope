"""Platform adapters: how one technology's store answers the questions
interpretation depends on.

**The registry is a dict of import paths, not of adapters.** Every other registry in
the package (``workflows``, ``methods``) imports its members at the bottom of its own
``__init__`` and registers live objects. This one cannot: an adapter carries its
platform's read dependencies, so importing the package to resolve *one* study would
drag in every platform's -- the opposite of what a registry is for
(``types/contracts.py``: "adding a platform cannot break a method"). So
:data:`ADAPTERS` maps a platform id to a ``"module:attr"`` string and :func:`get`
imports it on demand. Adding a platform is still one entry plus the adapter module,
and the entry is still checked: :func:`get` verifies the resolved object declares the
``name`` it was registered under, so a typo fails loudly at first use rather than
resolving to the wrong technology's reader.

**Nothing here runs at import**, which is the same constraint as before and matters
more now: every value is a string, so the dict literal costs nothing and imports
nothing. ``tests/test_import_surface.py`` requires that importing the package read no
dataset file, and an adapter that resolved a study at module scope would freeze every
path to whichever study happened to be active first -- the bug ``data/study.py`` was
written to kill.

**Why this exists at all.** ``pipeline/records.py`` and ``pipeline/stages/enrichment``
both constructed ``XeniumAdapter()`` as a literal, each with a comment saying no
registry existed. ARCHITECTURE §4 claims the core does not change when a member is
added; with the adapter hardcoded at both composition roots that claim was false for
platforms specifically, and a second platform would have been a pipeline edit rather
than a registration.
"""

from __future__ import annotations

from typing import Any

from panoscope.types.uispec import DuplicateRegistration

__all__ = ["ADAPTERS", "UnregisteredPlatform", "for_study", "get", "names",
           "register"]


class UnregisteredPlatform(LookupError):
    """Raised for a platform nobody registered. Never defaulted to another one.

    Falling back to "the only registered adapter" is the failure this class exists
    to prevent. An adapter decides the capability frames, which decide every
    measurement state in the study -- so reading a MERSCOPE store through the Xenium
    adapter would not raise, it would answer, and it would answer about a panel that
    was never run. A study whose platform is unknown has to stop.
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

    The platform is read from the manifest, which is Panoscope's own file and not a
    vendor one, so this is not a chicken-and-egg: nothing platform-specific has to be
    parsed to discover which platform it is. A store declaring none stops here --
    ARCHITECTURE §6.3 already lists platform among the four facts that block a study,
    and inferring it from a filename is the guess that rule exists to forbid.
    """
    declared = getattr(store, "platform", None)
    if not declared:
        raise UnregisteredPlatform(
            f"study {getattr(store, 'id', '?')!r} declares no platform in its "
            "manifest, so no adapter can be chosen. Platform is one of the four "
            "facts ARCHITECTURE 6.3 blocks the study on; it is not inferred"
        )
    return get(str(declared))
