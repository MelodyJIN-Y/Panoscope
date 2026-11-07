"""Reference data: the rule it exists to obey, proved structurally.

IMPLEMENTATION §9 names three things ``test_reference_*`` must defend -- determinism,
no bundled restricted data, and a missing node reading as unknown. This file defends
those, and one more that matters more than all three:

> Reference data may select and order evidence, generate rivals, supply canonical
> lists and carry citations; it may never **supply a value, count as agreement, or
> corroborate a measurement**.

The tests below are written to fail if that becomes *merely documented*. Each one
targets a specific route a number could take out of a source and into an answer -- a
typed field, the qualifier channel, the authority ladder, an import edge -- rather
than asserting that today's code happens not to do it.

**No reference data is read from disk here.** Every fixture writes a synthetic
14-column file into ``tmp_path``, so the suite is offline, deterministic, and bundles
nothing: ``test_no_reference_data_is_bundled_in_the_repo`` is the in-suite form of
ARCHITECTURE §12's CI grep.
"""

from __future__ import annotations

import ast
import dataclasses
import gzip
import inspect
import re
import typing
from pathlib import Path

import pytest

from panoscope.reference import graph as G
from panoscope.reference import sources as S
from panoscope.reference.graph import (
    RELATIONS, Edge, Graph, Lookup, Node, Presence, ReferenceValueLeak,
    UnregisteredRelation, lookup,
)
from panoscope.reference.sources import (
    NotRedistributable, PanglaoDB, SourceUnavailable, UnrecognisedSource,
)
from panoscope.types.claim import PREDICATES, Authority, Claim
from panoscope.types.conditions import DISEASE_STATE, Provenance
from panoscope.types.contracts import ReferenceSource
from panoscope.types.measured import FrameRef
from panoscope.types.subject import Entity, EntityKind, cell_type, gene, subject
from panoscope.types.trace import RULES, Traced

REPO = Path(__file__).resolve().parents[1]

CITE = "doi:10.1093/database/baz046"

# The real 2020 release's 14 columns, in order. Five of them are numbers, and that
# is the point of ``test_the_real_schema_carries_five_numbers_...``.
HEADER = (
    "species", "official gene symbol", "cell type", "nicknames",
    "ubiquitousness index", "product description", "gene type",
    "canonical marker", "germ layer", "organ",
    "sensitivity_human", "sensitivity_mouse",
    "specificity_human", "specificity_mouse",
)

# Synthetic rows in the real shape. Values are invented; only the SCHEMA is real.
ROWS = (
    ("Hs", "ERBB2", "Epithelial cells", "NEU|HER2", "0.31", "erb-b2 receptor",
     "protein-coding", "1", "Ectoderm", "Breast", "0.42", "0.00", "0.95", "0.00"),
    ("Hs", "POSTN", "Fibroblasts", "OSF2", "0.19", "periostin",
     "protein-coding", "1", "Mesoderm", "Breast", "0.55", "0.00", "0.88", "0.00"),
    ("Hs", "COL1A1", "Fibroblasts", "", "0.44", "collagen type I alpha 1",
     "protein-coding", "0", "Mesoderm", "Breast", "0.61", "0.00", "0.70", "0.00"),
    ("Mm", "Erbb2", "Epithelial cells", "", "0.30", "erb-b2 receptor",
     "protein-coding", "1", "Ectoderm", "Breast", "0.00", "0.40", "0.00", "0.90"),
    ("Hs", "NPHS2", "Podocytes", "", "0.05", "podocin",
     "protein-coding", "1", "Mesoderm", "Kidney", "0.77", "0.00", "0.99", "0.00"),
)


def _write_markers(root: Path, rows=ROWS, header=HEADER) -> Path:
    """Write a synthetic markers file where ``PanglaoDB`` expects the cached one."""
    dest = root / "panglaodb" / PanglaoDB.release / PanglaoDB.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = "\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    dest.write_bytes(gzip.compress(body.encode()))
    return dest


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A reference cache in tmp_path -- outside the repo, which the gate requires."""
    root = tmp_path / "refcache"
    root.mkdir()
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(root))
    return root.resolve()


@pytest.fixture
def registry():
    """Save and restore ``SOURCES``; other tests in this process share the dict."""
    before = dict(S.SOURCES)
    S.SOURCES.clear()
    yield S.SOURCES
    S.SOURCES.clear()
    S.SOURCES.update(before)


def _edge(subj=None, obj=None, **kw) -> Edge:
    return Edge(
        subject=subj or gene("ERBB2"),
        predicate=kw.pop("predicate", "marks"),
        object=obj or cell_type("Epithelial cells"),
        source=kw.pop("source", "panglaodb@27_Mar_2020"),
        citation=kw.pop("citation", CITE),
        licence=kw.pop("licence", "unstated"),
        **kw,
    )


# =========================================================================== #
# 0. The cross-stream contract. contracts.py forward-references these names.
# =========================================================================== #
def test_contracts_forward_reference_resolves_to_this_module():
    """``types/contracts.py`` imports ``Node`` and ``Edge`` from here under
    TYPE_CHECKING, so the names never resolve at runtime and nothing would complain
    until something called ``get_type_hints``. ``types/`` is frozen, so if this module
    had named them anything else the contract could not be repaired at the other end.

    Resolved against ``vars(G)`` -- this module's real namespace -- rather than a
    hand-supplied mapping, which would pass whatever the names happened to be.
    """
    import panoscope.types.contracts as C

    tree = ast.parse(Path(C.__file__).read_text(), filename=C.__file__)
    imported = {(n.module, a.name) for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module for a in n.names}
    assert {("panoscope.reference.graph", "Node"),
            ("panoscope.reference.graph", "Edge")} <= imported

    hints = typing.get_type_hints(ReferenceSource.nodes, localns=vars(G))
    assert typing.get_args(hints["return"])[0] is Node
    hints = typing.get_type_hints(ReferenceSource.edges, localns=vars(G))
    assert typing.get_args(hints["return"])[0] is Edge


def test_panglaodb_satisfies_the_reference_source_contract():
    src = PanglaoDB("Breast")
    assert isinstance(src, ReferenceSource)
    assert src.name and src.release and src.licence
    assert isinstance(src.redistributable, bool)


# =========================================================================== #
# 1. THE RULE: a source cannot supply a value. Four routes, four guards.
# =========================================================================== #
_NUMERIC = re.compile(r"\b(int|float|complex|Decimal|Fraction)\b")


def test_no_numeric_field_anywhere_on_the_reference_graph():
    """Guard 1, and the strongest one: there is no slot to put a statistic in.

    ``EvidenceItem`` needs a ``__post_init__`` to keep a statistic and a measurement
    status coherent because it legitimately carries numbers. The reference graph never
    does, so the honest enforcement is the absence of the field -- a source cannot
    return a value it has nowhere to put.
    """
    bad = []
    for name in dir(G):
        obj = getattr(G, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        if obj.__module__ != G.__name__:
            continue
        for f in dataclasses.fields(obj):
            if _NUMERIC.search(str(f.type)):
                bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, (
        "numeric field(s) on the reference graph -- reference data may never supply "
        "a value: " + "; ".join(bad)
    )


def test_the_numeric_field_scan_is_wired():
    """Guard against silent vacuity: the scan passes trivially over no dataclasses."""
    found = [n for n in dir(G)
             if isinstance(getattr(G, n), type)
             and dataclasses.is_dataclass(getattr(G, n))
             and getattr(G, n).__module__ == G.__name__]
    assert len(found) >= 4, f"the graph dataclass scan found only {found}"
    assert _NUMERIC.search("tuple[int, ...]") and not _NUMERIC.search("tuple[str, ...]")


@pytest.mark.parametrize(
    "value", ["0.87", "3", "-1", "1e-4", "nan", "inf", "0", "  2.5  "],
)
def test_a_qualifier_that_reads_as_a_number_is_refused(value):
    """Guard 2: the one open channel. ``qualifiers`` is where a value would actually
    be smuggled -- it takes free text, so a ``("specificity", "0.87")`` pair would
    sail through every other check and render beside a measured statistic."""
    with pytest.raises(ReferenceValueLeak):
        _edge(qualifiers=(("specificity", value),))


def test_a_textual_qualifier_still_passes():
    """The guard must bite on numbers only; a reference source that cannot say
    "Breast" is a reference source that cannot scope anything."""
    e = _edge(qualifiers=(("organ", "Breast"), ("species", "Hs")))
    assert e.qualifier("organ") == "Breast"


def test_the_real_schema_carries_five_numbers_and_the_graph_has_nowhere_for_them(
    cache,
):
    """The demonstration on the real file shape, not on a toy.

    PanglaoDB's 14 columns include a ubiquitousness index and four
    sensitivity/specificity values. Every one of them is a measurement of somebody
    else's experiment. None of them reaches the built graph -- and the reason is not
    this reader's restraint, it is that ``Edge`` would raise.
    """
    _write_markers(cache)
    src = PanglaoDB("Breast")
    numbers = {r[4] for r in ROWS} | {r[10] for r in ROWS} | {r[12] for r in ROWS}

    rendered = []
    for e in src.edges():
        rendered.append(str(e))
        rendered.extend(f"{k}={v}" for k, v in e.qualifiers)
    blob = " ".join(rendered)
    leaked = sorted(n for n in numbers if n in blob)
    assert not leaked, f"a statistic reached the graph: {leaked}"

    # ... and the reason it is structural: trying to carry one raises.
    with pytest.raises(ReferenceValueLeak):
        _edge(qualifiers=(("specificity_human", "0.95"),))


def test_an_edge_cannot_be_minted_above_curated():
    """Guard 3, the "never counts as agreement" half.

    Claims are ordered and never merged. An edge that cannot be minted above CURATED
    can only ever be displayed *beside* the measurement, never grouped as one -- and
    ``to_claim`` takes no authority parameter, so there is no call site that could
    ask for MEASUREMENT.
    """
    params = set(inspect.signature(Edge.to_claim).parameters)
    assert params == {"self", "claim_id"}, (
        f"to_claim grew a parameter: {sorted(params)}. An authority argument is the "
        "seam through which a reference edge becomes a measurement."
    )
    claim = _edge().to_claim("ref:1")
    assert claim.authority is Authority.CURATED
    assert claim.authority < Authority.MEASUREMENT
    assert claim.authority < Authority.LAB
    assert claim.conditions is None       # a provenance stamp, not a transfer verdict


def test_the_verdict_predicate_is_not_a_registerable_relation():
    """``is_cell_type`` is the answer, and the answer is the measurement's to give.

    It is a perfectly valid ``Claim`` predicate -- the deterministic engine writes it
    -- which is exactly why its absence from ``RELATIONS`` has to be checked: the two
    vocabularies overlap, and the overlap is where a source would acquire the power
    to assert a verdict.
    """
    assert "is_cell_type" in PREDICATES
    assert "is_cell_type" not in RELATIONS
    with pytest.raises(UnregisteredRelation):
        _edge(subj=Entity(EntityKind.UNIT, "c3"), predicate="is_cell_type")


def test_the_graph_cannot_construct_a_measurement():
    """Guard 4: an import scan, because the other three only cover what exists today.

    ``EvidenceItem`` is the only object in the system that carries a statistic beside
    a measurement status. ``graph.py`` does not import it, ``Measured``, or anything
    that builds one -- so there is no code path from a reference edge to a thing that
    renders as a measurement, and adding one is a visible import, not a quiet field.
    """
    tree = ast.parse(Path(G.__file__).read_text(), filename=G.__file__)
    imported = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"panoscope.types.evidence", "panoscope.types.measured",
                 "panoscope.types.outputs", "panoscope.core.measure"}
    assert not (imported & forbidden), (
        f"graph.py imports {sorted(imported & forbidden)} -- a reference module that "
        "can build evidence can corroborate a measurement"
    )


@pytest.mark.parametrize("subj, obj", [
    (cell_type("Fibroblasts"), gene("POSTN")),            # fully reversed
    (Entity(EntityKind.UNIT, "c3"), cell_type("Fibroblasts")),   # subject only
    (gene("POSTN"), gene("COL1A1")),                      # object only
])
def test_an_edges_endpoints_must_be_the_declared_kinds(subj, obj):
    """A backwards ``marks`` edge would hand a caller the canonical *cell types* for
    a gene while the call site believed it had asked for markers. Silent, and wrong
    in the direction that looks plausible.

    Parametrised over each end separately: a reversed edge trips both checks at once,
    so one case alone would keep passing with either guard deleted.
    """
    with pytest.raises(ValueError):
        _edge(subj=subj, obj=obj)


def test_a_citation_must_resolve():
    """A citation that is prose renders, exports and survives, and the grounding gate
    cannot tell it from a fabrication."""
    from panoscope.types.claim import UnknownSource
    with pytest.raises(UnknownSource):
        _edge(citation="Franzen et al. 2019")
    assert _edge(citation=CITE).citation == CITE


# =========================================================================== #
# 2. A missing node reads as "not in reference data", never "does not exist"
# =========================================================================== #
def test_a_missing_node_reads_as_not_in_reference_data():
    g = Graph(nodes=(Node(gene("ERBB2"), "src"),), consulted=("src",))
    out = lookup(g, gene("FAP"))
    assert out.value.presence is Presence.NOT_IN_REFERENCE
    said = out.value.reads_as
    assert "not in reference data" in said
    for forbidden in ("does not exist", "no such", "absent from", "is not a"):
        assert forbidden not in said, f"the sentence reads as non-existence: {said!r}"


def test_reference_silence_never_argues_in_any_state():
    """The mirror of ``Measured.absence_is_evidence``, and the difference is the
    point: a *measured* absence is the productive half of the lattice and does argue.
    A reference source's silence is only ever coverage, so there is no state in which
    it may be read as evidence against anything."""
    for state in Presence:
        assert Lookup(gene("FAP"), state).absence_argues is False
    src = inspect.getsource(Lookup.absence_argues.fget)
    assert "if" not in src and "self.presence" not in src, (
        "absence_argues grew a branch; it must be a constant False"
    )


def test_a_present_node_comes_back_with_its_edges_and_a_traced_decision():
    g = Graph(
        nodes=(Node(gene("POSTN"), "src"), Node(cell_type("Fibroblasts"), "src")),
        edges=(_edge(subj=gene("POSTN"), obj=cell_type("Fibroblasts")),),
        consulted=("src",),
    )
    out = lookup(g, cell_type("Fibroblasts"))
    assert isinstance(out, Traced)
    assert out.value.presence is Presence.IN_REFERENCE
    assert out.value.subjects("marks") == (gene("POSTN"),)
    assert [d.rule for d in out.decisions] == ["reference:in_reference"]
    assert all(d.step == "reference" for d in out.decisions)


def test_every_reference_rule_is_registered():
    """A decision whose rule id is not in RULES carries a reason nothing can render."""
    for rid in ("reference:in_reference", "reference:not_in_reference",
                "reference:no_source", "reference:source_read",
                "reference:source_unavailable"):
        assert rid in RULES, f"{rid} is not registered in trace.RULES"
        assert RULES[rid].step == "reference"


def test_a_canonical_list_is_a_selection_not_a_value():
    """What reference data IS for: naming the markers worth looking at. Note what
    comes back -- entities, never statistics, so the measurement still answers."""
    g = Graph(
        nodes=(Node(cell_type("Fibroblasts"), "src"),),
        edges=(_edge(subj=gene("POSTN"), obj=cell_type("Fibroblasts")),
               _edge(subj=gene("COL1A1"), obj=cell_type("Fibroblasts"))),
        consulted=("src",),
    )
    markers = lookup(g, cell_type("Fibroblasts")).value.subjects("marks")
    assert markers == (gene("POSTN"), gene("COL1A1"))
    assert all(isinstance(m, Entity) for m in markers)


# =========================================================================== #
# 3. Offline degrades honestly: an unread source is not an empty one
# =========================================================================== #
def test_an_unfetched_source_is_unavailable_not_empty(cache, registry):
    """The failure this exists to prevent: an empty graph and an unread graph are the
    same bytes. Rendered as ``NOT_IN_REFERENCE`` for every gene on the panel, a failed
    download reads as "the literature knows nothing about any of this"."""
    S.register(PanglaoDB("Breast"))          # nothing written to the cache
    out = S.build()
    g = out.value
    assert g.consulted == ()
    assert g.unavailable == ("panglaodb:breast",)
    assert [d.rule for d in out.decisions] == ["reference:source_unavailable"]

    said = lookup(g, gene("ERBB2"))
    assert said.value.presence is Presence.NO_SOURCE
    assert "not consulted" in said.value.reads_as
    assert "not in reference data" not in said.value.reads_as


@dataclasses.dataclass(frozen=True, slots=True)
class _Unreachable:
    """A registered source whose data cannot be read -- the offline case, isolated.

    Needed because an *unregistered* name takes a different branch in ``build``, so a
    test built on one would keep passing with the ``SourceUnavailable`` handler gone.
    """

    name: str = "atlas:offline"
    release: str = "v1"
    licence: str = "CC-BY-4.0"
    redistributable: bool = True

    def conditions(self):
        raise SourceUnavailable("never fetched")

    def nodes(self):
        raise SourceUnavailable("never fetched")
        yield                                     # pragma: no cover - generator shape

    def edges(self):
        raise SourceUnavailable("never fetched")
        yield                                     # pragma: no cover - generator shape


def test_a_partial_read_fails_closed(cache, registry):
    """One source readable, one not, and the entity in neither. The honest answer is
    NO_SOURCE: saying "not in reference data" while a source failed to load reports a
    download error as knowledge, and the two sentences are indistinguishable."""
    _write_markers(cache)
    S.register(PanglaoDB("Breast"))
    S.register(_Unreachable())
    out = S.build()
    g = out.value
    assert g.consulted == ("panglaodb:breast",)
    assert g.unavailable == ("atlas:offline",)
    assert sorted(d.rule for d in out.decisions) == [
        "reference:source_read", "reference:source_unavailable"]

    said = lookup(g, gene("NPHS2"))          # a kidney gene: absent from breast rows
    assert said.value.presence is Presence.NO_SOURCE, (
        "a partial read must not report an absence as a finding"
    )
    assert "not in reference data" not in said.value.reads_as


def test_a_fully_read_graph_can_still_say_not_in_reference(cache, registry):
    """The other half: once everything was read, "we looked and it is not there" is a
    legitimate answer, and must not be downgraded to NO_SOURCE."""
    _write_markers(cache)
    S.register(PanglaoDB("Breast"))
    g = S.build().value
    assert g.unavailable == ()
    assert lookup(g, gene("FAP")).value.presence is Presence.NOT_IN_REFERENCE


def test_the_read_path_never_reaches_for_the_network():
    """A read that silently downloads turns an offline suite flaky and a slow link
    into a hung demo. Fetching is explicit, and only ``fetch`` does it."""
    for fn in (PanglaoDB._rows, PanglaoDB.nodes, PanglaoDB.edges,
               PanglaoDB.conditions):
        src = inspect.getsource(fn)
        assert "urlopen" not in src and "urllib" not in src, (
            f"{fn.__qualname__} reaches for the network on a read path"
        )
    assert "urlopen" in inspect.getsource(PanglaoDB.fetch), (
        "the network scan above is vacuous if nothing fetches"
    )


def test_build_with_an_empty_registry_says_no_source_not_nothing_exists(registry):
    g = S.build().value
    assert g.consulted == () and g.nodes == ()
    assert lookup(g, gene("ERBB2")).value.presence is Presence.NO_SOURCE


# =========================================================================== #
# 4. The licence gate: a non-redistributable source cannot land in the repo
# =========================================================================== #
def test_panglaodb_is_declared_not_redistributable():
    """panglaodb.se states no licence -- only that the data are public and should be
    cited. An unstated licence is not permission, so the boolean is fail-closed."""
    assert PanglaoDB.redistributable is False
    assert "unstated" in PanglaoDB.licence


@pytest.mark.parametrize("where", ["", "data", "panoscope/reference", "studies/x"])
def test_a_non_redistributable_source_cannot_cache_inside_the_repo(
    monkeypatch, where,
):
    """The gate fires at PATH RESOLUTION, before a byte is written, so there is no
    window in which the data exists inside the checkout. A CI grep catches this after
    someone does it; this stops them doing it."""
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(REPO / where))
    with pytest.raises(NotRedistributable):
        S.cache_dir("panglaodb", "27_Mar_2020", redistributable=False)
    with pytest.raises(NotRedistributable):
        PanglaoDB("Breast").path


def test_a_redistributable_source_may_cache_inside_the_repo(monkeypatch):
    """The boolean has to mean something in both directions, or it is decoration."""
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(REPO / "data"))
    p = S.cache_dir("open_source_atlas", "v1", redistributable=True)
    assert REPO in p.parents


def test_no_reference_data_is_bundled_in_the_repo():
    """ARCHITECTURE §12's CI grep, run in the suite so it fails at the commit that
    adds the file rather than at the job that runs nightly."""
    patterns = ("panglaodb", "cellmarker", "cell_marker", "celltypist", "cellxgene")
    data_suffixes = (".gz", ".tsv", ".csv", ".parquet", ".json", ".zip", ".db",
                     ".sqlite", ".xlsx", ".h5", ".h5ad")
    hits = []
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith((".git/", ".venv/", "docs/archive/")):
            continue
        low = p.name.lower()
        if any(k in low for k in patterns) and p.suffix.lower() in data_suffixes:
            hits.append(rel)
    assert not hits, f"bundled reference data in the repo: {hits}"


def test_the_cache_destination_is_outside_the_repo(cache):
    """``fetch`` resolves its destination through the gate, so the licence check runs
    before any directory is created."""
    dest = PanglaoDB("Breast").path
    assert cache in dest.parents
    assert REPO not in dest.parents


# =========================================================================== #
# 5. Reading the real schema (synthetic rows), and determinism
# =========================================================================== #
def test_rows_are_scoped_to_organ_and_species(cache):
    _write_markers(cache)
    breast = {(e.subject.id, e.object.id) for e in PanglaoDB("Breast").edges()}
    assert breast == {("ERBB2", "Epithelial cells"), ("POSTN", "Fibroblasts"),
                      ("COL1A1", "Fibroblasts")}
    assert ("NPHS2", "Podocytes") not in breast          # kidney row
    assert not any(s == "Erbb2" for s, _ in breast)      # mouse row
    kidney = {(e.subject.id, e.object.id) for e in PanglaoDB("Kidney").edges()}
    assert kidney == {("NPHS2", "Podocytes")}


def test_a_renamed_column_is_refused_rather_than_guessed(cache):
    """The ``detect_unit`` refusal, one layer out. A reader that positionally guesses
    its way through a changed release emits edges between the wrong columns, and every
    answer under them is confidently wrong."""
    renamed = tuple("gene_symbol" if c == "official gene symbol" else c
                    for c in HEADER)
    _write_markers(cache, header=renamed)
    with pytest.raises(UnrecognisedSource) as exc:
        list(PanglaoDB("Breast").edges())
    assert "official gene symbol" in str(exc.value)


def test_nodes_and_edges_join_on_the_subject_key(cache):
    """Reconciliation is a join on ``Subject.key`` across task kinds. An edge that
    kept its own identity scheme could not take part in it."""
    _write_markers(cache)
    src = PanglaoDB("Breast")
    nodes = {n.key: n for n in src.nodes()}
    assert nodes["gene:ERBB2"].aliases == frozenset({"NEU", "HER2"})
    assert nodes["cell_type:Fibroblasts"].aliases == frozenset()
    e = next(e for e in src.edges() if e.subject.id == "ERBB2")
    assert e.as_subject().key == subject("gene", g=gene("ERBB2")).key


def test_the_curation_flag_travels_and_the_statistics_do_not(cache):
    """Supplying canonical lists is what reference data is FOR (§7.2). The flag is a
    curation fact; the sensitivity columns beside it are somebody else's measurement.
    Only one of the two is in the output."""
    _write_markers(cache)
    by_gene = {e.subject.id: e for e in PanglaoDB("Breast").edges()}
    assert by_gene["ERBB2"].qualifier("canonical") == "yes"
    assert by_gene["COL1A1"].qualifier("canonical") == "no"
    assert by_gene["ERBB2"].qualifier("germ_layer") == "Ectoderm"
    assert {k for k, _ in by_gene["ERBB2"].qualifiers} == {
        "canonical", "germ_layer", "organ", "species"}


def test_the_built_graph_is_deterministic(cache, registry):
    """These bytes are hashed into the record; two builds must not differ by the
    order a reader happened to walk a file."""
    _write_markers(cache)
    S.register(PanglaoDB("Breast"))
    a, b = S.build().value, S.build().value
    assert a == b
    assert a.edges[0].qualifiers == tuple(sorted(a.edges[0].qualifiers))


def test_conditions_are_a_provenance_stamp_with_its_unknowns_declared(cache):
    """§7.5: a database-curated source gets a provenance stamp, not a verdict. The
    three unknowns are the material for the class-level banner, declared rather than
    guessed."""
    _write_markers(cache)
    c = PanglaoDB("Breast").conditions()
    assert c.assay.value == "scRNA-seq"
    assert c.tissue.provenance is Provenance.READ
    assert "Breast" in (c.tissue.value or "")
    assert c.declared_unknown == ("method", "segmentation", "fixation", *DISEASE_STATE)
    assert len(c.frames) == 1 and isinstance(c.frames[0], FrameRef)


def test_a_reference_frame_cannot_decide_whether_anything_was_measured():
    """The one place ``ReferenceSource.conditions() -> Conditions`` forced a call.

    A curated database has no capability frame, but ``Conditions`` requires a
    ``FrameRef``, so the source describes the symbols it holds an opinion about. That
    cannot leak into a measurement: a ``FrameRef`` has a digest and a size and **no
    members**, so it answers no ``contains()`` and cannot be put in a
    ``MeasuredWorld`` -- reference coverage still cannot decide measurability, which
    is §7.3's rule enforced by a type.
    """
    names = {f.name for f in dataclasses.fields(FrameRef)}
    assert "members" not in names, (
        "FrameRef grew a member set; reference coverage could now be read as a "
        "capability frame"
    )
    assert not hasattr(FrameRef, "contains")


def test_conditions_are_unavailable_offline_rather_than_invented(cache):
    """No file, no stamp. A ``Conditions`` built over nothing would claim a frame
    digest over an empty set, which reads as a real, tiny panel."""
    with pytest.raises(SourceUnavailable):
        PanglaoDB("Breast").conditions()


def test_registration_refuses_a_non_source_and_a_name_collision(registry):
    """A silent overwrite would swap the graph's provenance underneath a running
    study, and every edge would still carry the old source string."""
    S.register(PanglaoDB("Breast"))
    S.register(PanglaoDB("Breast"))                 # equal value: idempotent
    with pytest.raises(TypeError):
        S.register(object())                        # type: ignore[arg-type]

    @dataclasses.dataclass(frozen=True, slots=True)
    class Impostor:
        name: str = "panglaodb:breast"
        release: str = "27_Mar_2020"
        licence: str = "MIT"
        redistributable: bool = True

        def conditions(self): ...
        def nodes(self): ...
        def edges(self): ...

    with pytest.raises(S.DuplicateSource):
        S.register(Impostor())


def test_claims_minted_from_edges_carry_their_licence(cache):
    _write_markers(cache)
    e = next(iter(PanglaoDB("Breast").edges()))
    claim = e.to_claim("ref:panglaodb:1")
    assert isinstance(claim, Claim)
    assert claim.source == CITE
    assert claim.licence == PanglaoDB.licence
