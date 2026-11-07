"""The per-study reference view (ARCHITECTURE §7.1): the sources this study's tissue
registers, read into one graph. THE canonical-list source for the marker rubric.

Sources: the lab's own list (``inputs/reference/lab_markers.csv``, rows for this
tissue), the Cell Ontology (synonyms only), and each organ-scoped source the tissue
maps to (``sources.organ_key``). No cross-tissue fallback: a label no source marks in
this tissue gets an empty canonical set and reads "not in reference data".
Names join across sources deterministically: through a CL id (CellMarker's ontology
column), an unambiguous CL exact synonym, and one logged plural/case normaliser
(``Macrophages`` -> ``macrophage``); never fuzzily. Disease state and species narrow a
source that carries them; one that cannot says so in ``View.scope``, never silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from panoscope.reference import graph, sources
from panoscope.reference.graph import Edge, Graph
from panoscope.types.subject import Entity, EntityKind

#: The lab-curated list, relative to the study directory. Lab data, not code.
LAB_FILE = Path("inputs") / "reference" / "lab_markers.csv"
LAB_SYNONYMS = LAB_FILE.with_name("lab_synonyms.csv")

log = logging.getLogger(__name__)
_NORMAL = frozenset({"normal", "healthy", "control", "none"})   # disease_state words for "no disease"
_SPECIES = {"human": ("Hs", "Human"), "homo sapiens": ("Hs", "Human"),
            "mouse": ("Mm", "Mouse"), "mus musculus": ("Mm", "Mouse")}


def _norm(name: str) -> str:
    """The one plural/case normaliser: casefold, single spaces, drop a plural "s"."""
    n = " ".join(name.split()).casefold()
    if n.endswith("s") and not n.endswith(("ss", "us", "is")):
        log.info("reference join: %r read as %r (plural/case)", name, n[:-1])
        return n[:-1]
    return n


def _index(g: Graph) -> dict[str, str]:
    """normalised name or CL synonym -> its join key (a CL id when the ontology has it).
    A synonym naming two keys is ambiguous and resolves nowhere. Lab synonyms name
    rivals (``synonyms``) but never borrow another label's canonical list."""
    ids = {_norm(n.entity.id): a for n in g.nodes if n.source.startswith("cl@")
           for a in n.aliases if a.startswith("CL:")}
    seen: dict[str, set[str]] = {}
    for e in g.edges:
        if e.predicate == "synonym" and e.source.startswith("cl@"):
            o = _norm(e.object.id)
            seen.setdefault(_norm(e.subject.id), set()).add(ids.get(o, o))
    return {**{s: next(iter(k)) for s, k in seen.items() if len(k) == 1}, **ids,
            **{i.casefold(): i for i in ids.values()}}       # a CL id is its own key


def _listed(e: Edge) -> bool:
    """A marker the source itself stands behind: PanglaoDB's canonical flag, a cited
    CellMarker row. Keeps hundreds of never-measured genes out of the off-panel list."""
    src = e.source.split("@")[0]
    if src == "panglaodb":
        return e.qualifier("canonical") == "yes"
    if src == "cellmarker":
        return e.citation.startswith("pmid:")
    return True


@dataclass(frozen=True, slots=True)
class View:
    """One study's consulted reference subgraph, plus the release of each source read."""

    graph: Graph
    releases: tuple[tuple[str, str], ...] = ()     # (source name, release), read order
    scope: tuple[str, ...] = ()                    # how each source was (not) narrowed
    index: dict[str, str] = field(default_factory=dict)

    def key(self, name: str, cl_id: str | None = None) -> str:
        """The join key for a cell-type name: a CL id when resolvable, else its normal form."""
        if cl_id and self.index.get(cl_id.casefold()) == cl_id:
            return cl_id
        n = _norm(name)
        return self.index.get(n, n)

    def marks(self, label: str) -> tuple[Edge, ...]:
        """Every ``marks`` edge onto ``label`` (or its synonym), with source and licence."""
        k = self.key(label)
        return tuple(e for e in self.graph.edges if e.predicate == "marks"
                     and self.key(e.object.id, e.qualifier("cl_id")) == k)

    def markers(self, label: str) -> tuple[str, ...]:
        """Every marker the sources list for ``label``, on- and off-panel, in edge order."""
        return tuple(dict.fromkeys(e.subject.id for e in self.marks(label) if _listed(e)))

    def canonical(self, label: str) -> tuple[str, ...]:
        """The canonical ones (an edge not flagged ``canonical=no``): the band's list."""
        return tuple(dict.fromkeys(e.subject.id for e in self.marks(label)
                                   if _listed(e) and e.qualifier("canonical") != "no"))

    def synonyms(self) -> tuple[tuple[str, str], ...]:
        """``(synonym, cell type)`` pairs from the ontology and the lab's list."""
        return tuple((e.subject.id, e.object.id) for e in self.graph.edges
                     if e.predicate == "synonym")

    def candidates(self) -> tuple[str, ...]:
        """The cell types this tissue's sources mark: the label and rival space."""
        out: dict[str, str] = {}
        for e in self.graph.edges:
            if e.predicate == "marks":
                out.setdefault(self.key(e.object.id, e.qualifier("cl_id")), e.object.id)
        return tuple(out.values())

    def reads_as(self, label: str) -> str:
        """The sentence for a label's reference lookup; never "does not exist"."""
        return graph.lookup(self.graph, Entity(EntityKind.CELL_TYPE, label)).value.reads_as


_SCOPE_KEYS = ("tissue", "tissue_id", "species", "disease_state")


def _sources(m: dict, sdir: Path) -> tuple[tuple, tuple[str, ...]]:
    """What this study's tissue registers, lab list first, and how each was narrowed."""
    tissue = str(m.get("tissue") or "")
    tid = str(m.get("tissue_id") or tissue)          # an intake-chosen UBERON id, else the words
    out: list = [sources.LabCurated(sdir / LAB_FILE, tissue)] if (sdir / LAB_FILE).is_file() else []
    if (sdir / LAB_SYNONYMS).is_file():
        out.append(sources.LabSynonyms(sdir / LAB_SYNONYMS))
    out.append(sources.CellOntology())
    disease = str(m.get("disease_state") or "").strip().casefold()
    normal = None if disease in ("", "unknown") else disease in _NORMAL
    sp = str(m.get("species") or "").strip().casefold()
    species = _SPECIES.get(sp or "human")
    scope: list[str] = []
    if sp and species is None:
        scope.append(f"species {sp!r}: no organ-keyed source carries it, none read")
    organ_srcs: list = []
    if species and (organ := sources.organ_key(tid, "panglaodb")):
        organ_srcs.append(sources.PanglaoDB(organ, species[0]))
    if species and (organ := sources.organ_key(tid, "cellmarker")):
        organ_srcs.append(sources.CellMarker(organ, normal, species[1]))
    for s in organ_srcs:
        sp_note = f"species {s.species}" + ("" if sp else " (undeclared; read as human)")
        if isinstance(s, sources.CellMarker):
            dz = ("all rows (disease state undeclared)" if normal is None
                  else "Normal rows" if normal else "disease rows (cancer_type not Normal)")
        else:
            dz = "carries no disease state, not narrowed" if normal is not None else ""
        scope.append(f"{s.name}: {sp_note}" + (f"; {dz}" if dz else ""))
    if normal is not None and out and isinstance(out[0], sources.LabCurated):
        scope.append(f"{out[0].name}: carries no disease state, not narrowed")
    return tuple(out + organ_srcs), tuple(scope)


def for_study(study) -> View:
    """The study's reference view; rebuilt when its scope, cache or lab files change."""
    # ponytail: a source fetched into the same cache mid-process is seen on restart.
    m = study.manifest
    return _view(study.dir, tuple(str(m.get(k) or "") for k in _SCOPE_KEYS),
                 sources.cache_root(), sources.LabCurated(study.dir / LAB_FILE, "").release,
                 sources.LabSynonyms(study.dir / LAB_SYNONYMS).release)


@lru_cache(maxsize=8)
def _view(sdir: Path, scope_vals: tuple, _cache_root, _lab: str, _syn: str) -> View:
    srcs, scope = _sources(dict(zip(_SCOPE_KEYS, scope_vals)), sdir)
    g = sources.build(srcs=srcs).value
    return View(g, tuple((s.name, s.release) for s in srcs if s.name in g.consulted),
                scope, _index(g))


def check_pinned(study, pinned: dict[str, str]) -> None:
    """Refuse to run an eval whose expected outcomes were set against another release."""
    now = dict(for_study(study).releases)
    if now != pinned:
        raise SystemExit(f"reference release changed: pinned {pinned}, now {now}; "
                         "re-derive the expected outcomes, then re-pin")
