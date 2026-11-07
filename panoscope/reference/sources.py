"""The source registry, the licence gate, and PanglaoDB.

**The licence gate is behaviour, not documentation.** :func:`cache_dir` refuses to
resolve a cache directory inside the repository for a source with
``redistributable=False``, even with ``PANOSCOPE_REFERENCE_DIR`` pointed at the
checkout, so restrictively-licensed data cannot be bundled.

**Registration is data.** ``SOURCES`` is a plain dict, **empty at import**: PanglaoDB
is instantiated per organ, so the study build registers what its tissue needs.

**PanglaoDB is organ-scoped** (ARCHITECTURE §8.1: the join key is the
``(gene, cell type, tissue)`` triple), read from the file's ``organ`` column.

**Offline is a first-class outcome.** Only :meth:`PanglaoDB.fetch` touches the
network. ``nodes()``/``edges()`` read the cache or raise :class:`SourceUnavailable`,
and :func:`build` records the source as unavailable rather than returning an empty
graph (see ``graph.Presence``).
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterator

from panoscope.reference.graph import Edge, Graph, Node
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import ReferenceSource
from panoscope.types.measured import Frame
from panoscope.types.subject import Entity, EntityKind
from panoscope.types.trace import Trace, Traced


class SourceUnavailable(RuntimeError):
    """Raised when a source's data could not be read -- never fetched, or offline.

    Distinct from a plain error because it is an expected outcome, not a bug: the
    caller's correct response is to record the source as unconsulted, which is what
    :func:`build` does. Swallowing it into an empty result would turn a failed
    download into the finding "reference data knows nothing about any of this".
    """


class UnrecognisedSource(ValueError):
    """Raised when a cached file does not have the columns this reader needs.

    The same refusal as ``MethodModule.detect_unit``: never guess a schema. A reader
    that positionally guesses its way through a changed release silently emits edges
    between the wrong columns, and every answer under them is confidently wrong.
    """


class NotRedistributable(RuntimeError):
    """Raised when a non-redistributable source is pointed at the repository tree.

    This is the licence gate. It fires at *path resolution*, before any byte is
    written, so there is no window in which the data exists inside the checkout.
    """


class DuplicateSource(ValueError):
    """Raised when two different sources claim one registry name.

    Overwriting silently would swap the graph's provenance underneath a running
    study, and every edge would still carry the *old* source string.
    """


# --------------------------------------------------------------------------- #
# Where reference data lives. ARCHITECTURE §20's proposed default.
# --------------------------------------------------------------------------- #
def cache_root() -> Path:
    """``$PANOSCOPE_REFERENCE_DIR``, else ``~/.panoscope/reference``.

    Outside the repo by default, which is the point: the gate below then only ever
    fires for someone who deliberately overrode it to a checkout.
    """
    env = os.getenv("PANOSCOPE_REFERENCE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(".panoscope", "reference").resolve()


def _repo_root() -> Path:
    """The working tree this package is checked out in. Resolved lazily, because
    importing a module must not touch the filesystem (``test_import_surface``)."""
    return Path(__file__).resolve().parents[2]


def cache_dir(name: str, release: str, *, redistributable: bool) -> Path:
    """Where one source's release is cached -- and the licence gate.

    A source that may not be redistributed may not resolve a directory inside the
    repository.
    """
    root = cache_root()
    if not redistributable:
        repo = _repo_root()
        if root == repo or repo in root.parents:
            raise NotRedistributable(
                f"{name} is not redistributable ({release}), and the reference cache "
                f"resolves to {root}, inside the repository at {repo}. Fetch it, "
                "never bundle it: point PANOSCOPE_REFERENCE_DIR outside the checkout."
            )
    return root / name / release


# --------------------------------------------------------------------------- #
# The registry. DATA, not types. Empty at import -- see the module docstring.
# --------------------------------------------------------------------------- #
SOURCES: dict[str, ReferenceSource] = {}


def register(src: ReferenceSource) -> None:
    """Add a source to the registry. Refuses a name collision rather than clobbering."""
    if not isinstance(src, ReferenceSource):
        raise TypeError(
            f"{src!r} does not satisfy the ReferenceSource contract: it must declare "
            "name, release, licence, redistributable, conditions(), nodes(), edges()"
        )
    existing = SOURCES.get(src.name)
    # Equality, not identity: re-registering an equal source is idempotent (frozen
    # dataclasses compare by value), while a *different* source under one name is the
    # collision worth refusing.
    if existing is not None and existing != src:
        raise DuplicateSource(
            f"{src.name!r} is already registered to {existing!r}; two sources under "
            "one name would silently swap the provenance on every edge"
        )
    SOURCES[src.name] = src


def get(name: str) -> ReferenceSource | None:
    return SOURCES.get(name)


def build(names: tuple[str, ...] = ()) -> Traced[Graph]:
    """Read the named sources (all registered ones by default) into one graph.

    Never merges: two sources holding a node for one gene contribute two nodes. An
    unreadable source lands in ``Graph.unavailable`` with its reason.
    """
    t = Trace()
    chosen = names or tuple(sorted(SOURCES))
    nodes: list[Node] = []
    edges: list[Edge] = []
    consulted: list[str] = []
    unavailable: list[str] = []

    for name in chosen:
        src = SOURCES.get(name)
        if src is None:
            unavailable.append(name)
            t.note("reference:source_unavailable", name,
                   because=("not registered",), refs=(f"source:{name}",))
            continue
        try:
            ns = tuple(src.nodes())
            es = tuple(src.edges())
        except SourceUnavailable as exc:
            unavailable.append(name)
            t.note("reference:source_unavailable", name,
                   because=(str(exc),), refs=(f"source:{name}",))
            continue
        nodes.extend(ns)
        edges.extend(es)
        consulted.append(name)
        t.note("reference:source_read", name,
               because=(f"release {src.release}; licence {src.licence}",
                        f"{len(ns)} nodes, {len(es)} edges"),
               refs=(f"source:{name}",))

    return t.done(Graph(
        nodes=tuple(nodes), edges=tuple(edges),
        consulted=tuple(consulted), unavailable=tuple(unavailable),
    ))


# --------------------------------------------------------------------------- #
# PanglaoDB
# --------------------------------------------------------------------------- #
_PANGLAO_COLUMNS = (
    "species", "official gene symbol", "cell type", "nicknames",
    "ubiquitousness index", "product description", "gene type",
    "canonical marker", "germ layer", "organ",
    "sensitivity_human", "sensitivity_mouse",
    "specificity_human", "specificity_mouse",
)
"""The 2020 release's 14 columns. Read by NAME, never by position; a release that
renames a required one fails loudly (:class:`UnrecognisedSource`).

Five of them are numbers, and none has anywhere to go: ``Edge`` has no numeric field
and its qualifiers reject numeric values.
"""

_REQUIRED_COLUMNS = frozenset(
    {"species", "official gene symbol", "cell type", "organ"}
)


@dataclass(frozen=True, slots=True)
class PanglaoDB:
    """PanglaoDB's curated cell-type marker table, scoped to one organ.

    Franzén O, Gan L-M, Björkegren JLM. *PanglaoDB: a web server for exploration of
    mouse and human single-cell RNA sequencing data.* Database 2019: baz046.
    ``doi:10.1093/database/baz046`` -- resolved live against Crossref, not recalled.

    **Licence: the site states none** (the CC-BY-4.0 Crossref reports is the
    article's). So ``redistributable`` is ``False``, fail-closed.
    """

    organ: str

    release: ClassVar[str] = "27_Mar_2020"
    licence: ClassVar[str] = (
        "unstated by the source (panglaodb.se states only that the data are public "
        "and asks for citation); treated as not redistributable"
    )
    redistributable: ClassVar[bool] = False
    citation: ClassVar[str] = "doi:10.1093/database/baz046"
    filename: ClassVar[str] = "PanglaoDB_markers_27_Mar_2020.tsv.gz"
    url: ClassVar[str] = (
        "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
    )
    species: ClassVar[str] = "Hs"          # human; the file's own token

    @property
    def name(self) -> str:
        """Organ-scoped, because the reference view is per tissue (§7.1, §8.1)."""
        return f"panglaodb:{self.organ.lower()}"

    @property
    def provenance(self) -> str:
        """What every node and edge from this source is stamped with."""
        return f"panglaodb@{self.release}"

    @property
    def path(self) -> Path:
        """The cached file. One download serves every organ, so the cache directory
        is keyed on the source, not on :attr:`name`."""
        return cache_dir(
            "panglaodb", self.release, redistributable=self.redistributable
        ) / self.filename

    # -- reading ----------------------------------------------------------- #
    def _rows(self) -> Iterator[dict[str, str]]:
        """Rows for this organ and species, keys lowercased and stripped.

        Raises :class:`SourceUnavailable` when the file is not cached, which is the
        honest offline answer, and :class:`UnrecognisedSource` when it is cached but
        does not have the columns this reader needs.
        """
        path = self.path
        if not path.is_file():
            raise SourceUnavailable(
                f"{self.name}: {path} is not present. PanglaoDB is not "
                "redistributable, so it is fetched, never bundled -- run "
                "PanglaoDB(...).fetch() or point PANOSCOPE_REFERENCE_DIR at a cache "
                "that has it. Until then the graph records this source as unread."
            )
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = {(c or "").strip().lower() for c in (reader.fieldnames or ())}
            missing = sorted(_REQUIRED_COLUMNS - header)
            if missing:
                raise UnrecognisedSource(
                    f"{self.name}: {path} is missing column(s) {missing}. Expected "
                    f"the {len(_PANGLAO_COLUMNS)}-column {self.release} release; "
                    f"found {sorted(header)}. Refusing rather than guessing."
                )
            want = self.organ.strip().lower()
            for raw in reader:
                row = {(k or "").strip().lower(): (v or "").strip()
                       for k, v in raw.items() if k is not None}
                if row.get("organ", "").lower() != want:
                    continue
                if self.species not in row.get("species", "").split():
                    continue
                if not row.get("official gene symbol") or not row.get("cell type"):
                    continue
                yield row

    def nodes(self) -> Iterator[Node]:
        """One node per gene and one per cell type, in file order, deduplicated.

        Gene nicknames become :attr:`Node.aliases` -- §7.2's "is this symbol an
        alias". An alias is navigation; it asserts nothing and carries no citation.
        """
        seen: set[Entity] = set()
        for row in self._rows():
            gene = Entity(EntityKind.GENE, row["official gene symbol"])
            if gene not in seen:
                seen.add(gene)
                aliases = frozenset(
                    a.strip() for a in row.get("nicknames", "").split("|") if a.strip()
                )
                yield Node(entity=gene, source=self.provenance, aliases=aliases)
            ct = Entity(EntityKind.CELL_TYPE, row["cell type"])
            if ct not in seen:
                seen.add(ct)
                yield Node(entity=ct, source=self.provenance)

    def edges(self) -> Iterator[Edge]:
        """One ``gene --marks--> cell type`` edge per row.

        Qualifiers carry organ, species and the curation flag only. ``canonical
        marker`` is re-spelled ``1``/``0`` -> ``yes``/``no``: a curation flag
        (canonical lists are what reference data is *for*, §7.2), which also keeps
        it past ``Edge``'s numeric guard.
        """
        for row in self._rows():
            quals = [
                ("organ", self.organ),
                ("species", self.species),
                ("canonical", "yes" if row.get("canonical marker") == "1" else "no"),
            ]
            germ = row.get("germ layer", "")
            if germ and germ.upper() not in ("NA", "N/A"):
                quals.append(("germ_layer", germ))
            yield Edge(
                subject=Entity(EntityKind.GENE, row["official gene symbol"]),
                predicate="marks",
                object=Entity(EntityKind.CELL_TYPE, row["cell type"]),
                source=self.provenance,
                citation=self.citation,
                licence=self.licence,
                qualifiers=tuple(quals),
            )

    def conditions(self) -> Conditions:
        """The conditions THIS SOURCE's edges were established under (§7.5's stamp).

        Analysis method, fixation and segmentation are ``UNKNOWN`` (PanglaoDB
        aggregates over many studies). The required ``FrameRef`` describes the set of
        symbols the source holds an opinion about; it carries no members, so it
        cannot enter a ``MeasuredWorld`` or decide whether anything was measured.
        """
        symbols = frozenset(
            n.entity.id for n in self.nodes() if n.entity.kind is EntityKind.GENE
        )
        frame = Frame(
            kind=EntityKind.GENE,
            name=f"panglaodb:{self.organ.lower()}:curated_symbols",
            members=symbols,
        )
        return Conditions(
            assay=Condition("scRNA-seq", Provenance.STATED),
            modality=Condition("transcriptomics", Provenance.STATED),
            tissue=Condition(f"panglaodb:organ:{self.organ}", Provenance.READ),
            method=UNKNOWN,          # >1000 aggregated studies; no single method
            segmentation=UNKNOWN,    # dissociated scRNA-seq; none to declare
            fixation=UNKNOWN,        # varies per contributing study
            frames=(frame.ref(),),
        )

    # -- fetching (explicit, never implicit) -------------------------------- #
    def fetch(self, *, timeout: float = 120.0) -> Path:
        """Download the markers file into the cache. Called by a human or by setup.

        Never called from a read path. The licence gate in :attr:`path` refuses a
        destination inside the repository, and the write is atomic.
        """
        dest = self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            self.url, headers={"User-Agent": "Panoscope/0.1 (reference cache)"},
        )
        tmp = dest.with_name(dest.name + ".part")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                tmp.write_bytes(resp.read())
        except OSError as exc:                 # URLError and friends are OSErrors
            tmp.unlink(missing_ok=True)
            raise SourceUnavailable(
                f"{self.name}: could not fetch {self.url}: {exc}. The graph will "
                "record this source as unread rather than as empty."
            ) from exc
        # A gzip file that does not open is a 403 page or a truncated body; either
        # would parse as "this release has no rows", which reads as a finding.
        try:
            with gzip.open(io.BytesIO(tmp.read_bytes()), "rb") as fh:
                fh.read(1)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise SourceUnavailable(
                f"{self.name}: {self.url} did not return a gzip file ({exc}). "
                "panglaodb.se rejects some automated requests; fetch it in a browser "
                f"and place it at {dest}."
            ) from exc
        tmp.replace(dest)
        return dest
