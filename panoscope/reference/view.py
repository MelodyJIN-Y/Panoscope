"""The per-study reference view (ARCHITECTURE §7.1): the sources this study's tissue
registers, read into one graph. THE canonical-list source for the marker rubric.

Sources: the lab's own list (``inputs/reference/lab_markers.csv``, rows for this
tissue), the Cell Ontology (synonyms only), and each organ-scoped source the tissue
maps to (``sources.organ_key``). No cross-tissue fallback: a label no source marks in
this tissue gets an empty canonical set and reads "not in reference data".
Labels match exactly, or through an unambiguous Cell Ontology exact synonym; never fuzzily.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from panoscope.reference import graph, sources
from panoscope.reference.graph import Edge, Graph
from panoscope.types.subject import Entity, EntityKind

#: The lab-curated list, relative to the study directory. Lab data, not code.
LAB_FILE = Path("inputs") / "reference" / "lab_markers.csv"


@dataclass(frozen=True, slots=True)
class View:
    """One study's consulted reference subgraph, plus the release of each source read."""

    graph: Graph
    releases: tuple[tuple[str, str], ...] = ()     # (source name, release), read order

    def _primary(self) -> dict[str, str]:
        """synonym -> its one Cell Ontology name; an ambiguous synonym resolves nowhere."""
        seen: dict[str, set[str]] = {}
        for e in self.graph.edges:
            if e.predicate == "synonym":
                seen.setdefault(e.subject.id, set()).add(e.object.id)
        return {s: next(iter(n)) for s, n in seen.items() if len(n) == 1}

    def marks(self, label: str) -> tuple[Edge, ...]:
        """Every ``marks`` edge onto ``label`` (or its synonym), with source and licence."""
        p = self._primary()
        key = p.get(label, label)
        return tuple(e for e in self.graph.edges
                     if e.predicate == "marks" and p.get(e.object.id, e.object.id) == key)

    def markers(self, label: str) -> tuple[str, ...]:
        """Every marker the sources list for ``label``, on- and off-panel, in edge order."""
        return tuple(dict.fromkeys(e.subject.id for e in self.marks(label)))

    def canonical(self, label: str) -> tuple[str, ...]:
        """The canonical ones (an edge not flagged ``canonical=no``): the band's list."""
        return tuple(dict.fromkeys(e.subject.id for e in self.marks(label)
                                   if e.qualifier("canonical") != "no"))

    def candidates(self) -> tuple[str, ...]:
        """The cell types this tissue's sources mark: the label and rival space."""
        return tuple(dict.fromkeys(e.object.id for e in self.graph.edges
                                   if e.predicate == "marks"))

    def cl_id(self, label: str) -> str:
        """The label's Cell Ontology id, or "" if the ontology holds no such name."""
        name = self._primary().get(label, label)
        return next((a for n in self.graph.nodes if n.entity.id == name
                     for a in sorted(n.aliases) if a.startswith("CL:")), "")

    def reads_as(self, label: str) -> str:
        """The sentence for a label's reference lookup; never "does not exist"."""
        return graph.lookup(self.graph, Entity(EntityKind.CELL_TYPE, label)).value.reads_as


def _sources(study) -> tuple:
    """What this study's tissue registers, lab list first. Unread ones stay recorded."""
    tissue = str(study.tissue or "")
    lab = study.dir / LAB_FILE
    out: list = [sources.LabCurated(lab, tissue)] if lab.is_file() else []
    out.append(sources.CellOntology())
    if organ := sources.organ_key(tissue, "panglaodb"):
        out.append(sources.PanglaoDB(organ))
    if organ := sources.organ_key(tissue, "cellmarker"):
        out.append(sources.CellMarker(organ))
    return tuple(out)


def for_study(study) -> View:
    """The study's reference view; rebuilt when its lab list changes."""
    return _view(study, sources.LabCurated(study.dir / LAB_FILE, "").release)


@lru_cache(maxsize=8)
def _view(study, _lab_release: str) -> View:
    srcs = _sources(study)
    g = sources.build(srcs=srcs).value
    return View(g, tuple((s.name, s.release) for s in srcs if s.name in g.consulted))


def check_pinned(study, pinned: dict[str, str]) -> None:
    """Refuse to run an eval whose expected outcomes were set against another release."""
    now = dict(for_study(study).releases)
    if now != pinned:
        raise SystemExit(f"reference release changed: pinned {pinned}, now {now}; "
                         "re-derive the expected outcomes, then re-pin")
