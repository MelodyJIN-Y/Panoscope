"""Negative-control probes are not panel genes, and the count has one source.

Xenium writes its controls into ``panel.parquet`` beside the genes.
``XeniumAdapter.frames`` dropped them and ``agent.data.load_panel`` did not, so the
capability frame said one number and every consumer of the tidy reader said another:
``manifest.panel_gene_count`` (``pipeline/run.py`` counts these rows),
``agent.enrichment._panel_size`` and so every "...on this N-gene panel" caveat, and
``ui.conditionality._panel_scope``, the line printed under every plot. All three are
coverage denominators, which is the one place a panel size is load-bearing.

The bundled demo's panel was pre-stripped, so the disagreement was unobservable on it
and only appears on a store whose panel file still carries a control row -- the
multi-sample fixture does, and a real 10x panel ships dozens.
"""

from __future__ import annotations

import pandas as pd
import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import enrichment as E
from panoscope.data.study import Study
from panoscope.platforms.xenium import (
    XeniumAdapter,
    is_control_annotation,
    is_control_probe,
)
from panoscope.types import measured as M
from panoscope.types.subject import Entity, EntityKind, subject
from panoscope.ui import conditionality as C

FIXTURE = "synthetic_multisample_fixture"
CONTROL = "NegControlProbe_00001"


@pytest.fixture
def store() -> Study:
    s = Study.open(FIXTURE)
    if not s.exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    return s


def test_the_control_row_is_still_in_the_file(store: Study) -> None:
    """The guard for this whole module: a fixture quietly cleaned of its control row
    would make every assertion below pass for the wrong reason."""
    import pandas as pd

    raw = pd.read_parquet(store.inputs / "panel.parquet", columns=["gene"])
    assert CONTROL in set(raw["gene"].astype(str))


def test_the_reader_and_the_capability_frame_agree_on_the_assay_size(store: Study) -> None:
    """One assay, one number. The frame was right; the reader was one row larger."""
    frame = next(f for f in XeniumAdapter().frames(store) if f.name.startswith("panel:"))
    with cfg.use_study(store):
        panel = data.load_panel()
    assert len(panel) == len(frame.members) == 12
    assert set(panel["gene"].astype(str)) == set(frame.members)


def test_a_control_probe_is_not_on_the_panel(store: Study) -> None:
    """``panel_contains`` is THE absence primitive; a control answering True to it
    means a probe can never be reported as off-panel."""
    with cfg.use_study(store):
        assert data.panel_contains(CONTROL) is False
        assert data.panel_annotation(CONTROL) is None
        assert data.panel_contains("SYNG01") is True
    assert is_control_probe(CONTROL) and not is_control_probe("SYNG01")


def test_no_coverage_denominator_counts_a_control_probe(store: Study) -> None:
    """The three places the inflated count surfaced, checked where they are read."""
    with cfg.use_study(store):
        assert E._panel_size() == 12
        assert "12-gene panel" in E._panel_scope_caveat("HALLMARK_X", 2, 30)
        assert "a 12-gene targeted panel" in C._panel_scope()
        # What pipeline/run.py writes into manifest.panel_gene_count.
        assert int(len(data.load_panel())) == store.manifest["panel_gene_count"] == 12


def test_the_demo_panel_is_unchanged(demo_id: str) -> None:
    """The demo panel carries no control rows, so a correct filter is a no-op there.
    If this moves, the filter is dropping a real gene and the golden would move too."""
    with cfg.use_study(Study.open(demo_id)):
        assert len(data.load_panel()) == 280


# --------------------------------------------------------------------------- #
# A control probe is MEASURED. Two ways that was being denied.
# --------------------------------------------------------------------------- #
def _store_with_a_measured_control(src: Study, root, *, names: tuple[str, ...]) -> Study:
    """``src``, copied, with its control row named ``names[0]`` AND a row for it in the
    method output -- a real unit, the largest statistic in the file. Any further
    ``names`` are written into ``panel.parquet`` as additional control rows, which is
    what lets one store carry every control family at once.

    Two constructions in one, because the defect needed both to show its worst face
    and last round built them separately. A vendor who writes ``NEGCONTROLPROBE_00001``
    is shipping the same probe and nothing stops a kidney ``panel.parquet`` arriving
    that way; and a control reaches ``markers_top.csv`` with a number on it for the
    ordinary reason that the method wrote its output over every probe the instrument
    ran. Built under ``tmp_path`` because the fixture on disk must stay byte-identical
    -- and because a test that writes into ``data/studies/`` is gitignored into silence.

    The statistic is deliberately the largest in the table: if the lattice ever admits
    this row, it does not merely appear, it drives the band.
    """
    import shutil

    data_root = root / "data"
    (data_root / "studies").mkdir(parents=True)
    shutil.copytree(src.dir, data_root / "studies" / src.id)
    study = data_root / "studies" / src.id

    named = names[0]
    pq = study / "inputs" / "panel.parquet"
    df = pd.read_parquet(pq)
    assert (df["gene"].astype(str) == CONTROL).sum() == 1
    df.loc[df["gene"].astype(str) == CONTROL, "gene"] = named
    for extra in names[1:]:
        df.loc[len(df)] = {"gene": extra, "ensembl_id": "", "annotation": "negative control"}
    df.to_parquet(pq, index=False)

    csv = study / "inputs" / "markers_top.csv"
    markers = pd.read_csv(csv)
    assert not markers["gene"].astype(str).str.upper().str.startswith("NEGCONTROL").any()
    # EVERY spelling gets a method-output row, not only ``names[0]``.
    #
    # This is the hole an independent verifier found and measured: the extra
    # spellings went into ``panel.parquet`` alone, and ``names[0]`` is
    # ``NegControlProbe_00001`` -- the one spelling the PRE-fix rule already matched.
    # So the whole suite stayed green with the reader half of the fix narrowed back
    # to the decorated spellings, because no BLANK- or ANTISENSE-family name had a
    # ``markers_top.csv`` row anywhere in the repo. The frame half was pinned through
    # the panel file and the reader half through the one easy name, and the two
    # discriminators never met in the same row.
    #
    # Descending coefficients keep ``names[0]`` the largest statistic in the file, so
    # the existing "the control leads the method output" assertions still mean that.
    for i, name in enumerate(names):
        markers.loc[len(markers)] = {
            "gene": name, "top_cluster": "c1", "glm_coef": 99.0 - i, "pearson": 0.99,
            "max_gg_corr": 0.99, "max_gc_corr": 0.99, "cell_type": "Synthetic Type A",
        }
    markers.to_csv(csv, index=False)

    return Study.open(src.id, data_root=data_root)


@pytest.fixture(params=[CONTROL, CONTROL.upper()], ids=["as_written", "mixed_case"])
def control_store(request, store: Study, tmp_path) -> tuple[Study, str]:
    """The cross product: the whole section below runs in both cases.

    Last round's two tests each pinned one half -- the case-insensitivity test never
    called ``panel_lookup`` and the ``panel_lookup`` test never used a mixed-case
    store -- so the combination, which is where the frame and the readers actually
    disagreed, was untested. Parametrising the fixture is what makes the hard half
    cost nothing to cover.
    """
    return _store_with_a_measured_control(store, tmp_path, names=(request.param,)), request.param


def test_a_control_is_excluded_from_every_gene_frame_in_any_case(
    control_store: tuple[Study, str],
) -> None:
    """One rule, one place: the capability frames and the tidy reader agree because
    they call the same function, upper-cased like every other gene comparison here.

    ``platforms.xenium.is_control_probe`` was the one case-SENSITIVE comparison in the
    package, and it was the one the frames used, so on a mixed-case panel file
    ``panel:<id>`` admitted the control and read 13 for a 12-gene assay while
    ``load_panel`` read 12. ``_detectable_genes`` filtered nothing at all, so
    ``detect:<id>`` admitted it in *either* case and went 9 -> 10.

    The numbers below are asserted against the constructed file -- 12 gene rows and
    one control row, written three lines up -- and never against another caller of the
    rule. A cross-check between two users of one broken rule passes while both are
    wrong, which is exactly how this survived a round.

    Both frames refusing is also what enforces the exclusion: ``_state_for`` requires
    *every* frame of a kind to admit an entity, so with the control outside both there
    is no store, and no case, in which it can be ``OBSERVED`` (and carry its count
    forward as a spatial marker) or ``TESTED_ABSENT`` (and argue against a cell type).
    """
    st, named = control_store
    raw = pd.read_parquet(st.inputs / "panel.parquet", columns=["gene"])["gene"].astype(str)
    assert named in set(raw) and len(raw) == 13   # the file really still has it

    frames = XeniumAdapter().frames(st)
    panel = next(f for f in frames if f.name.startswith("panel:"))
    detect = next(f for f in frames if f.name.startswith("detect:"))

    assert named not in panel.members and len(panel.members) == 12
    assert named not in detect.members and len(detect.members) == 9

    with cfg.use_study(st):
        assert len(data.load_panel()) == 12
        assert set(data.load_panel()["gene"].astype(str)) == set(panel.members)
        assert data.panel_contains(named) is False
        assert data.is_control(named) is True
        # The three coverage denominators, read where they are read.
        assert E._panel_size() == 12
        assert "a 12-gene targeted panel" in C._panel_scope()
        assert "12-gene panel" in E._panel_scope_caveat("HALLMARK_X", 2, 30)


def test_a_control_can_reach_neither_state_that_does_anything(
    control_store: tuple[Study, str],
) -> None:
    """The failure the lattice exists to make unrepresentable, pinned on both sides.

    The control sits in ``markers_top.csv`` with the largest statistic in the file, so
    ``reported`` contains it -- which is the live situation, not a contrived one. With
    the frames admitting it, that row resolves ``OBSERVED`` and ``carries_number``, and
    a negative control's count becomes a spatial marker with a number on it; with it
    absent from the method output instead, it resolves ``TESTED_ABSENT`` and its
    silence argues against a cell type. Neither may ever happen, in any case.

    A real panel gene is asserted beside it so the test fails if the frames simply
    stopped admitting anything.
    """
    st, named = control_store
    world = M.MeasuredWorld(XeniumAdapter().frames(st))

    def resolve(gene_id: str, reported: frozenset[str]):
        subj = subject("gene", g=Entity(EntityKind.GENE, gene_id))
        return M.resolve(subj, world, reported)

    assert resolve(named, frozenset({named})).carries_number is False
    assert resolve(named, frozenset()).absence_is_evidence is False
    assert resolve("SYNG01", frozenset({"SYNG01"})).carries_number is True
    assert resolve("SYNG01", frozenset()).absence_is_evidence is True


def test_panel_lookup_names_a_control_in_any_case(
    control_store: tuple[Study, str],
) -> None:
    """The sentence the biologist reads, on the store that used to break it.

    ``load_panel`` dropping the control row made ``panel_annotation`` return None, which
    sent ``tools.panel_lookup`` down its off-panel branch: a ``Source``-backed
    "never measured" about a probe whose entire function is to be measured -- a
    fabricated measurement status with click-back intact. That was fixed for a
    correctly-cased store only; on a mixed-case one the probe was back ON the panel, so
    the tool reported a negative control as a measured panel gene with an annotation.

    On an off-target panel this is the one fact that matters: the controls are how you
    decide whether a flat kidney-resident signal is biology or the assay floor.
    """
    from panoscope.agent import tools

    st, named = control_store
    with cfg.use_study(st):
        env = tools.panel_lookup(named)
        assert env["ok"]
        assert env["data"]["on_panel"] is False   # the denominator does not move
        assert env["data"]["control"] is True
        assert env["data"]["annotation"] == "negative control"
        detail = env["sources"][0]["detail"]
        assert "never measured" not in detail
        assert "control probe" in detail and "negative control" in detail

        # The third answer must not blunt the real one: a genuinely off-panel gene
        # still gets the never-measured sentence.
        off = tools.panel_lookup("COL1A1")
        assert off["data"]["on_panel"] is False and off["data"]["control"] is False
        assert "never measured" in off["sources"][0]["detail"]

        # ...and a control is still not a PANEL annotation, in any case.
        assert data.control_annotation(named) == "negative control"
        assert data.panel_annotation(named) is None


def test_control_annotation_is_reachable_and_only_for_controls(store: Study) -> None:
    """Dropped from the panel, not from the file -- that is what keeps the sentence true."""
    with cfg.use_study(store):
        assert data.control_annotation(CONTROL) == "negative control"
        assert data.control_annotation(CONTROL.lower()) == "negative control"
        assert data.control_annotation("SYNG01") is None   # a panel gene is not a control
        assert data.control_annotation("COL1A1") is None   # nor is an off-panel gene
        assert data.panel_annotation(CONTROL) is None      # still not a PANEL annotation


# --------------------------------------------------------------------------- #
# The rule is a list of control FAMILIES, not of the spellings someone had to hand
# --------------------------------------------------------------------------- #
CONTROL_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("neg control probe", ("NegControlProbe_00001", "negcontrolprobe_00002",
                           "NEGCONTROLPROBE_00003")),
    ("neg probe short",   ("NegPrb_00004", "negprb00005")),
    ("neg codeword",      ("NegControlCodeword_00006", "NEGCONTROLCODEWORD_00007")),
    ("blank",             ("BLANK_0001", "Blank-0002", "BlankCodeword_0003",
                           "blankcodeword_0004", "BLANKCODEWORD_0005")),
    ("antisense",         ("antisense_ABC", "antisenseXYZ", "AntiSense_DEF",
                           "ANTISENSE_0001")),
    ("unassigned",        ("UnassignedCodeword_0009", "unassignedcodeword_0010")),
    ("deprecated",        ("DeprecatedCodeword_0011", "deprecatedcodeword_0012")),
)
"""Every 10x control family, each in several of the spellings a vendor file really uses.

Parametrised over FAMILY, because the defect this pins was not a case bug and a
case-only parametrisation would have missed it twice. ``CONTROL_PREFIXES`` carried the
decorated spellings ``"BLANK_"``, ``"Blank-"`` and ``"antisense_"``, so the families
10x emits on *every* run -- ``BlankCodeword_xxxx``, bare ``antisenseXYZ`` -- matched
nothing, entered ``panel:<id>`` and ``detect:<id>`` as genes, and on an off-target
panel (where few rows carry any statistic at all) could come out as the top driver
with a glm_coef on it.
"""

ALL_CONTROL_SPELLINGS: tuple[str, ...] = tuple(
    name for _, names in CONTROL_FAMILIES for name in names
)

REAL_SYMBOLS_NEAR_A_FAMILY: tuple[str, ...] = (
    "NEGR1", "NEK2", "BLNK", "BLM", "BLVRA", "ANTXR1", "ANKRD1", "ANGPT1",
    "UNC5B", "DEPDC1", "BLOC1S1", "ANXA1",
)
"""Real HGNC symbols that start the way a control family starts, and must survive.

The other direction, which a filter this blunt has to be checked in: a rule that
deletes a measured gene from the assay is the same error as one that admits a control,
pointed the other way, and it is quieter -- the gene simply becomes NEVER_MEASURED and
its absence stops being evidence. **NEGR1 is why no entry is a bare ``NEG``.** No HGNC
symbol begins with ``BLANK`` or ``ANTISENSE`` (antisense transcripts are named
``<GENE>-AS1``), which is what makes those two safe as bare families.
"""


@pytest.mark.parametrize(
    "name",
    [n for _, names in CONTROL_FAMILIES for n in names],
    ids=[f"{fam}:{n}" for fam, names in CONTROL_FAMILIES for n in names],
)
def test_every_control_family_is_matched_in_every_spelling(name: str) -> None:
    """The rule itself, asserted against the family list and nothing else.

    Not a cross-check between two callers: last round's test compared the capability
    frame against the tidy reader and passed while both were wrong, because both were
    asking the same narrow rule. The only thing that catches a narrow rule is naming
    the probes the instrument emits and asserting the verdict directly.
    """
    assert is_control_probe(name) is True


@pytest.mark.parametrize("symbol", REAL_SYMBOLS_NEAR_A_FAMILY)
def test_the_widened_rule_drops_no_real_gene(symbol: str) -> None:
    """A control filter that eats a gene has removed a measurement, silently."""
    assert is_control_probe(symbol) is False


def test_a_panel_file_of_every_control_family_reads_as_the_assay_it_is(
    store: Study, tmp_path,
) -> None:
    """One constructed ``panel.parquet``: 12 genes and every control family beside them.

    The assertions are against the file written three lines up -- 12 gene rows,
    ``len(ALL_CONTROL_SPELLINGS)`` control rows -- so this fails if the rule narrows
    again, whichever caller does the narrowing. ``BlankCodeword_0003`` also carries the
    largest statistic in ``markers_top.csv``, which is the live shape of the failure:
    a control does not merely appear in the method output, it can lead it.
    """
    from panoscope.agent import tools

    st = _store_with_a_measured_control(store, tmp_path, names=ALL_CONTROL_SPELLINGS)
    raw = pd.read_parquet(st.inputs / "panel.parquet", columns=["gene"])["gene"].astype(str)
    assert len(raw) == 12 + len(ALL_CONTROL_SPELLINGS)
    assert set(ALL_CONTROL_SPELLINGS) <= set(raw)    # the file really carries them all

    frames = XeniumAdapter().frames(st)
    panel = next(f for f in frames if f.name.startswith("panel:"))
    detect = next(f for f in frames if f.name.startswith("detect:"))
    assert len(panel.members) == 12 and len(detect.members) == 9
    assert not (set(ALL_CONTROL_SPELLINGS) & (set(panel.members) | set(detect.members)))

    with cfg.use_study(st):
        # What pipeline/run.py counts into manifest.panel_gene_count, and the two
        # coverage denominators read off it.
        assert len(data.load_panel()) == 12
        assert E._panel_size() == 12
        assert "a 12-gene targeted panel" in C._panel_scope()

        # The control that leads the method output: neither state that does anything,
        # and a sentence that says what it is instead of what it is not.
        world = M.MeasuredWorld(frames)
        lead = ALL_CONTROL_SPELLINGS[0]
        subj = subject("gene", g=Entity(EntityKind.GENE, lead))
        assert M.resolve(subj, world, frozenset({lead})).carries_number is False
        assert M.resolve(subj, world, frozenset()).absence_is_evidence is False

        for name in ALL_CONTROL_SPELLINGS:
            assert data.panel_contains(name) is False
            assert data.is_control(name) is True
            env = tools.panel_lookup(name)
            assert env["data"]["on_panel"] is False and env["data"]["control"] is True
            assert "never measured" not in env["sources"][0]["detail"]

        # The method wrote its output over every probe the instrument ran, so the
        # control sits in markers_top.csv with the largest statistic in the file.
        # One filter in load_markers is what stops any caller quoting that number:
        # the agent tool, pipeline/stages/annotate, discriminate, and v1's
        # detectability frame all read through it.
        assigned = set(data.get_cluster_markers("c1")["gene"].astype(str))
        got = tools.marker_lookup(cluster="c1")["data"]["markers"]
        # Every spelling, not just ``lead``. ``lead`` is the one the pre-fix rule
        # already caught, so asserting on it alone stays green against a rule
        # narrowed back to the decorated spellings -- measured, by plugin, at 2095
        # passed with the reader half of the fix defeated.
        for name in ALL_CONTROL_SPELLINGS:
            assert name not in assigned, f"{name} reached get_cluster_markers"
            assert data.get_marker(name) is None, f"{name} reached get_marker"
        assert [m["gene"] for m in got if data.is_control(m["gene"])] == []
        assert got[0]["gene"] == "SYNG01"    # the real top driver, not the 99.0 control

        # ...and a real panel gene is still measured, so the test fails if the rule
        # widened into the genes instead.
        assert data.panel_contains("SYNG01") is True


def test_a_control_named_outside_every_family_is_caught_by_its_annotation(
    store: Study, tmp_path,
) -> None:
    """The half the NAME rule cannot reach, and the one that shipped a wrong answer.

    :data:`CONTROL_PREFIXES` matches the probe's name, and the name is the one thing
    a vendor is free to change between panel versions. A negative control shipped as
    ``IntergenicRegion_0001`` matches no family, enters ``panel:<id>`` as a gene and
    can then be ``OBSERVED`` -- which on an off-target panel, where few rows carry a
    statistic at all, is plausibly the top one. Measured before this rule existed:
    two such probes led ``key_markers`` in ``interp/verdicts.csv`` with
    ``role: supports`` and ``caveats: []``, read as *"Driven by IntergenicRegion_0001"*
    on the Markers page, and reached the model with ``glm_coef 89.0`` and a Source --
    while the row's own ``annotation`` column said ``negative control``, a column
    ``agent/data.panel_annotation`` already read and ``panel_lookup`` already printed.

    The store is built so the two rules cannot cover for each other: these names match
    no family, and the test asserts ``is_control_probe`` is False for both before
    asserting the store refuses them anyway.
    """
    import shutil

    from panoscope.agent import tools

    names = ("IntergenicRegion_0001", "GenomicControl_02")
    for n in names:
        assert is_control_probe(n) is False, f"{n} must not be caught by the NAME rule"

    data_root = tmp_path / "data"
    (data_root / "studies").mkdir(parents=True)
    shutil.copytree(store.dir, data_root / "studies" / store.id)
    st_dir = data_root / "studies" / store.id

    pq = st_dir / "inputs" / "panel.parquet"
    df = pd.read_parquet(pq)
    for n in names:
        df.loc[len(df)] = {"gene": n, "ensembl_id": "(synthetic, no identifier)",
                           "annotation": "negative control"}
    df.to_parquet(pq, index=False)

    csv = st_dir / "inputs" / "markers_top.csv"
    markers = pd.read_csv(csv)
    for i, n in enumerate(names):
        markers.loc[len(markers)] = {
            "gene": n, "top_cluster": "c1", "glm_coef": 89.0 - i, "pearson": 0.99,
            "max_gg_corr": 0.99, "max_gc_corr": 0.99, "cell_type": "Synthetic Type A",
        }
    markers.to_csv(csv, index=False)

    st = Study.open(store.id, data_root=data_root)
    adapter = XeniumAdapter()
    frames = adapter.frames(st)
    panel = next(f for f in frames if f.name.startswith("panel:"))
    detect = next(f for f in frames if f.name.startswith("detect:"))
    assert len(panel.members) == 12, "the annotated probes must not inflate the assay"
    assert not (set(names) & (set(panel.members) | set(detect.members)))

    for n in names:
        assert adapter.is_control(st, n) is True

    with cfg.use_study(st):
        assert len(data.load_panel()) == 12
        for n in names:
            assert data.is_control(n) is True
            assert data.panel_contains(n) is False
            assert data.get_marker(n) is None
            env = tools.panel_lookup(n)
            assert env["data"]["control"] is True
            assert env["data"]["annotation"] == "negative control"
        got = tools.marker_lookup(cluster="c1")["data"]["markers"]
        assert [m["gene"] for m in got][:1] == ["SYNG01"]
        assert not (set(names) & {m["gene"] for m in got})
        # The other direction: a real gene whose annotation is a cell type stays.
        assert data.panel_contains("SYNG01") is True


def test_the_annotation_rule_reads_only_the_two_tokens_it_declares() -> None:
    """The other direction for the annotation half: it must not eat a cell type.

    ``CONTROL_ANNOTATION_TOKENS`` is deliberately two words, both of which this
    codebase already treats as vendor control vocabulary (``negative control`` is the
    annotation on this repo's own fixture control row; ``CODEWORD`` is already a name
    family). A rule that widened into tissue or cell-type words would delete real
    genes from the assay -- the same error as admitting a probe, pointed the other
    way, and the quieter one. Every string below appears in a ``panel.parquet``
    committed to this repo.
    """
    for control in ("negative control", "Negative Control Probe", "Genomic Control",
                    "Unassigned Codeword", "Deprecated Codeword", "NEGATIVE CONTROL"):
        assert is_control_annotation(control) is True, control
    for real in ("Breast cancer", "Fibroblasts", "T cells", "Macrophages",
                 "Breast glandular cells", "synthetic fixture gene", "Adipocytes",
                 "NK cells", "Breast myoepithelial cells", "B cells", "", None):
        assert is_control_annotation(real) is False, real
