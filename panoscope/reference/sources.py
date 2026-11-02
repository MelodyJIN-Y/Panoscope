"""The source registry, the licence gate, and PanglaoDB.

A :class:`~panoscope.types.contracts.ReferenceSource` is the only thing allowed to
put nodes and edges into the graph, and ``graph.py`` is where "it may never supply a
value" is made structural. This module is the other half: **where the data lives, and
whether it may live in the repo at all.**

**The licence gate is behaviour, not documentation.** CLAUDE.md's compliance rule is
"MIT. Everything shown must be open source. Fetch restrictively-licensed data, never
bundle it", and ARCHITECTURE §12 adds "restrictively-licensed sources are fetched,
never bundled; CI greps ``data/`` and fails on known marker-database filenames". A CI
grep catches the mistake after someone makes it. :func:`cache_dir` refuses to make
it: a source with ``redistributable=False`` cannot resolve a cache directory inside
the repository working tree, so there is no path by which ``fetch`` could write one
there -- not even with ``PANOSCOPE_REFERENCE_DIR`` pointed at the checkout, which is
precisely the shortcut someone takes on a laptop and then commits.

**Registration is data.** ``SOURCES`` is a plain dict, the ``SHAPES``/``RULES``/
``FLAGS`` pattern, and it is **empty at import**. That is deliberate and it is not
laziness: PanglaoDB is instantiated per organ (see below), so a prepopulated registry
would have to name a tissue, and a tissue constant in library code is exactly the
dataset constant CLAUDE.md forbids -- the demo is breast and the second study is
kidney. The study build registers what its tissue needs.

**Why PanglaoDB is organ-scoped.** ARCHITECTURE §8.1: the join key is the
``(gene, cell type, tissue)`` triple, not the gene -- "POSTN in breast is a different
question from POSTN in kidney". The marker table carries an ``organ`` column, so the
scoping is *read from the file* rather than imposed. It also makes
``ReferenceSource.conditions()`` constructible at all: ``Conditions`` blocks on an
unknown tissue, and a pan-organ database has no single one. See
:meth:`PanglaoDB.conditions` for the one place that contract forced a judgement call.

**Offline is a first-class outcome.** Nothing here touches the network except
:meth:`PanglaoDB.fetch`, which a human or a setup step calls explicitly. ``nodes()``
and ``edges()`` read the cache or raise :class:`SourceUnavailable`; :func:`build`
catches it and records the source as unavailable rather than silently producing an
empty graph. An empty graph and an unread graph are the same bytes and completely
different claims -- see ``graph.Presence``.
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

    ``redistributable`` is a boolean on the contract rather than a licence string
    precisely so that something can *act* on it. This acts on it: a source that may
    not be redistributed may not resolve a directory inside the repository, so it
    cannot be committed by accident, by a mis-set environment variable, or by a
    well-meaning "just cache it next to the code".
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

    Never merges: two sources that both hold a node for ERBB2 contribute two nodes,
    for ``claim.py``'s reason -- claims are ordered and never merged, and a graph that
    merged them would have silently resolved a disagreement nobody saw.

    A source that cannot be read lands in ``Graph.unavailable`` with its reason, and
    the returned graph is honest about having been built with a hole in it.
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
"""The 2020 release's 14 columns. Read by NAME, never by position, and the four this
reader needs are required explicitly -- a release that renames one must fail loudly
(:class:`UnrecognisedSource`) rather than emit edges between the wrong columns.

**Five of these fourteen are numbers** -- the ubiquitousness index and the four
sensitivity/specificity columns -- and not one of them has anywhere to go. ``Edge``
has no numeric field and ``Edge.qualifiers`` rejects a numeric value, so the rule
"reference data may never supply a value" is enforced by the shape of the output
rather than by this reader's restraint. That is the whole demonstration, on a real
file: a careless rewrite of :meth:`PanglaoDB.edges` that tried to carry
``specificity_human`` would raise, not ship.
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

    **Licence: the site states none.** panglaodb.se says only "all data used in this
    database are public" and asks to be cited; there is no licence, no terms-of-use
    page and no redistribution grant. The CC-BY-4.0 that Crossref reports is the
    *article's* licence, not the database's. So ``redistributable`` is ``False``,
    fail-closed: an unstated licence is not permission, and the cost of being wrong
    in that direction is one download instead of a licence violation shipped in a
    public MIT repo.
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

        The qualifiers carry organ, species and the curation flag, and nothing else.
        ``canonical marker`` is a *curation* flag spelled ``1``/``0`` in the file and
        re-spelled ``yes``/``no`` here -- deliberately, and it is the one judgement
        call in this reader. Supplying canonical lists is explicitly what reference
        data is *for* (§7.2), while the five continuous columns are measurements of
        somebody else's experiment, which is what the rule forbids. The re-spelling is
        also what keeps it past ``Edge``'s numeric guard, so it is written down here
        rather than left to look like an accident.
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

        Three fields are honestly ``UNKNOWN`` and stay that way: PanglaoDB aggregates
        over a thousand studies, so there is no single analysis method, no single
        fixation and no segmentation at all. ``declared_unknown`` propagates them,
        which is the material for §7.5's class banner -- *"curated edges in this
        corpus were established in dissociated scRNA-seq; you are in FFPE in-situ"* --
        rendered once rather than repeated on every edge.

        **The judgement call.** ``Conditions`` requires at least one ``FrameRef``, and
        a curated database has no capability frame in the measurement sense. What it
        does have is the set of symbols it holds an opinion about, which is the honest
        reading of "what this apparatus was capable of saying something about", so
        that is what the ref describes. It cannot leak into a measurement: a
        ``FrameRef`` carries a digest and a size but no members, so it answers no
        ``contains()`` and cannot be put in a ``MeasuredWorld``. Reference coverage
        therefore still cannot decide whether anything was measured -- which is §7.3's
        rule, enforced by a type rather than by a reviewer.
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

        Never called from :meth:`nodes`, :meth:`edges` or :meth:`conditions`: a read
        path that silently reaches for the network turns an offline test suite into a
        flaky one and a slow link into a hung demo. The licence gate in
        :attr:`path` has already refused any destination inside the repository before
        a byte is written, and the write is atomic so an interrupted download never
        leaves a half file that the reader would parse as a short release.
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
