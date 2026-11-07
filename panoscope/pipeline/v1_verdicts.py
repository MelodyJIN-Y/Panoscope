"""The v1 ``ClusterVerdict`` for a unit of the CURRENT study, built through ``core/``.

Read by the chat, the prose fallback, the cited notes, the UI fallback and the
golden projection, so the band the chat speaks is the band the table shows.

A projection, not a second engine: ``records.build_all`` -> ``records.to_v1_dict``
-> ``serialize.verdict_from_dict``, the same calls ``stages/verdicts.py`` makes.
Cached per study with ``data._per_study`` (cleared by ``tests/conftest.py``).
"""

from __future__ import annotations

from panoscope.agent import config as cfg
from panoscope.agent import data as agent_data
from panoscope.agent.types import ClusterVerdict


@agent_data._per_study
def verdicts() -> dict[str, ClusterVerdict]:
    """Every unit of the current study, as the v1 object the UI and chat both read.
    Imports are function-local: ``agent -> pipeline`` is a cycle risk, and importing
    this module must perform no filesystem I/O. No ``try``/``except``: a band is
    never quietly a default.
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
