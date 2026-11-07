"""The v1 ``ClusterVerdict`` for a unit of the CURRENT study, built through ``core/``.

``pipeline/stages/verdicts.py`` was cut over to ``core/verdict.py``; five call sites
outside it were not, and kept calling ``agent/verdict.py`` directly:

  * ``agent/loop.py`` -- the band injected into the system prompt every chat turn,
    the deterministic skeptic's evidence, and the ``verify`` flag on every answer;
  * ``agent/skeptic.py`` -- the call the second opinion argues against;
  * ``pipeline/stages/notes.py`` -- the cell type and markers the cited notes are
    written from;
  * ``ui/data_access.py`` -- the fallback when a study has no built tree.

``tests/test_core_compat.py`` gates the two engines against each other on the NINE
DEMO UNITS. On any other study nothing pins them, so the first place a divergence
would show is the chat telling the biologist a different band than the table in
front of them. One engine removes the question.

**Why this is a projection and not a second engine.** Every value comes up through
``records.build_all`` -> ``records.to_v1_dict`` -> ``serialize.verdict_from_dict``:
the same three calls ``stages/verdicts.py`` makes before it writes
``interp/clusters/<unit>.json``. Nothing here computes a statistic or a band, so a
caller of this module and a reader of that file cannot disagree.

``agent/verdict.py`` stays exactly where it is. It is the independent reference
implementation ``test_core_compat`` compares ``core/`` against, and it cannot be
that gate and a shim over ``core/`` at the same time.

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

    Same contract as ``agent.verdict.verdict_for_cluster``: a unit the study names
    and the method never saw raises rather than returning an empty call.
    """
    built = verdicts()
    if unit not in built:
        raise KeyError(
            f"[v1_verdicts] no interpretation for unit {unit!r} in study "
            f"{cfg.current_study().id!r}; the method found: {sorted(built)}"
        )
    return built[unit]
