"""The v1 ``ClusterVerdict`` for a unit of the CURRENT study, built through ``core/``.

The one producer of the v1 verdict outside ``pipeline/stages/verdicts.py``: the chat
(``agent/loop.py``, ``agent/skeptic.py``), the prose fallback (``agent/compose.py``),
the cited notes (``pipeline/stages/notes.py``), the UI fallback (``ui/data_access.py``)
and the golden projection all read it, so the band the chat speaks is the band the
table shows.

**Why this is a projection and not a second engine.** Every value comes up through
``records.build_all`` -> ``records.to_v1_dict`` -> ``serialize.verdict_from_dict``:
the same three calls ``stages/verdicts.py`` makes before it writes
``interp/clusters/<unit>.json``. Nothing here computes a statistic or a band, so a
caller of this module and a reader of that file cannot disagree.

**Cost.** ``build_all`` composes the whole v2 path: ~0.14 s cold, ~0 warm. It is
cached per study with ``data._per_study`` -- the decorator every zero-arg reader
uses, so ``tests/conftest.py``'s autouse fixture clears it with the rest and a study
switch cannot serve another study's bands. ``loop.py`` calls this once per chat
turn; the cache is what makes that free.
"""

from __future__ import annotations

from panoscope.agent import config as cfg
from panoscope.agent import data as agent_data
from panoscope.agent.types import ClusterVerdict


@agent_data._per_study
def verdicts() -> dict[str, ClusterVerdict]:
    """Every unit of the current study, as the v1 object the UI and chat both read.

    Imports are function-local for the reason ``agent/discriminate.py``'s are:
    ``agent -> pipeline`` is a cycle risk, and importing this module must perform no
    filesystem I/O. There is deliberately no ``try``/``except`` -- a band that is
    quietly a default is the failure shape this whole cutover exists to remove.
    """
    from panoscope.pipeline import records
    from panoscope.pipeline.serialize import verdict_from_dict

    built = records.build_all(cfg.current_study())
    return {unit: verdict_from_dict(records.to_v1_dict(b)) for unit, b in built.items()}


def verdict_for(unit: str) -> ClusterVerdict:
    """One unit's verdict. ``KeyError`` if the method produced no evidence for it.

    A unit the study names
    and the method never saw raises rather than returning an empty call.
    """
    built = verdicts()
    if unit not in built:
        raise KeyError(
            f"[v1_verdicts] no interpretation for unit {unit!r} in study "
            f"{cfg.current_study().id!r}; the method found: {sorted(built)}"
        )
    return built[unit]
