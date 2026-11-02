"""The second-study gate: the multi-sample code paths, exercised on data.

``test_study_plurality.py`` proved two studies can be open in one process without
leaking into each other. It could not prove anything about *plurality inside a
study*, because neither store on disk declares ``samples`` -- so four things had
never run on data at all:

- ``PlatformAdapter.samples`` had never returned more than one
- ``clustering_mode`` had never had to decide anything
- ``UndeclaredClustering``, ARCHITECTURE §6.3's fourth and only conditional
  blocker, fired only against hand-built manifests in ``test_platforms.py``
- ``core/concordance.py`` had no data, and -- the part that only showed up once a
  multi-sample store existed -- **no producer**: nothing in the package could
  resolve a per-sample reading, so the one function §9.10 is written around had no
  caller that was not a test

``data/studies/synthetic_multisample_fixture`` exists for exactly these. It is a
FIXTURE, in the sense ``demo_mini`` is one: every gene name, label, sample id and
number in it is made up and means nothing biologically, and its manifest says so in
its first field. It is deliberately not named after, and carries no content from,
the real second study. What it reproduces is that study's *shape* -- several
samples, joint clustering, and a panel that mostly does not cover the annotation's
cell types -- and nothing else.

**The number this file pins, and why it moved.** The band.
``test_the_band_is_the_finding`` used to record a finding rather than a fix: a unit
with one canonical marker on panel and five never measured read **Very High, verify
False, no demotion**, while zero canonical markers read Low and set verify. The
rubric had a cliff at one. On the regime this fixture reproduces -- an
immuno-oncology panel against a tissue it was not designed for -- most resident
types land on zero or one, so roughly half of them would have come back confidently
wrong on the study built to stress exactly this.

The fix is ``band:demote_canonical_coverage``: K of N of the label's canonical
markers that a capability frame admitted, against a declared floor of one in three.
It demotes this fixture (1/6) and no demo cluster (worst is c7 at 4/7), so the
golden hash does not move. The test below now pins the corrected behaviour and the
census it rests on, so the floor cannot drift without a reader seeing the K of N it
was measured against.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent.intake import CLUSTERING, Status, derive
from panoscope.core.concordance import (
    Comparison, IndependentClustering, Replicate, concordance,
)
from panoscope.data.study import Study
from panoscope.methods.jazzpanda import MODULE as JAZZPANDA
from panoscope.platforms.xenium import XeniumAdapter
from panoscope.types.contracts import UndeclaredClustering
from panoscope.types.measured import MeasuredState, MeasuredWorld, resolve
from panoscope.types.subject import EntityKind, cell_type, gene, subject
from panoscope.types.subject import unit as unit_entity

FIXTURE = "synthetic_multisample_fixture"
ADAPTER = XeniumAdapter()


@pytest.fixture(scope="module", autouse=True)
def _fixture_present() -> None:
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")


@pytest.fixture
def store() -> Study:
    return Study.open(FIXTURE)


def _copy(store: Study, root: Path, drop_key: str | None = None) -> Study:
    """A copy of the fixture under ``root``, optionally minus one manifest key."""
    dst = root / "studies" / "copy"
    shutil.copytree(store.dir, dst)
    if drop_key is not None:
        manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop(drop_key, None)
        (dst / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    return Study(id="copy", data_root=root)


def _evidence(store: Study, sample: str | None = None):
    """Normalised evidence for one sample, or for the pooled table."""
    path = (store.inputs / "markers_top.csv" if sample is None
            else ADAPTER.sample_inputs(store)[sample])
    return JAZZPANDA.load(str(path), ADAPTER, store).value


def _verdicts():
    from panoscope.agent import verdict as V
    return V.all_verdicts()


# --------------------------------------------------------------------------- #
# 0. It is a fixture and says so
# --------------------------------------------------------------------------- #
def test_the_fixture_declares_itself_synthetic(store: Study) -> None:
    """A fake dataset that does not announce itself becomes the thing someone cites.

    The check is on the manifest rather than on a README, because the manifest is
    what every reader of this study already opens.
    """
    assert store.manifest["synthetic"] is True
    assert "NOT REAL DATA" in store.manifest["SYNTHETIC_FIXTURE"]
    assert store.manifest["exercises"], "a fixture must say what it exists to exercise"
    assert "kidney" not in FIXTURE, "it must not read as the real second study"


# --------------------------------------------------------------------------- #
# 1. Multi-sample with joint clustering declared
# --------------------------------------------------------------------------- #
def test_a_study_on_disk_finally_holds_more_than_one_sample(store: Study) -> None:
    """``samples`` had never returned more than one from a real store."""
    assert ADAPTER.samples(store) == ["sampleA", "sampleB"]
    assert ADAPTER.clustering_mode(store) == "joint"
    assert ADAPTER.validate(store) == []


def test_the_frames_are_derived_from_the_files_not_the_manifest(store: Study) -> None:
    """The panel count in the manifest is a claim; ``panel.parquet`` is the frame.

    The fixture's panel file carries a negative-control probe, which the adapter
    drops and which ``panel_gene_count`` never counted -- so the two numbers agree
    here only because the file was read. ``detect`` is three smaller: the panel
    genes that produced no usable signal.
    """
    sizes = {f.name.split(":")[0]: len(f.members) for f in ADAPTER.frames(store)}
    assert sizes == {"panel": 12, "detect": 9, "annotation": 2, "clusters": 3}
    assert {f.kind for f in ADAPTER.frames(store)} == {
        EntityKind.GENE, EntityKind.CELL_TYPE, EntityKind.UNIT}
    panel = next(f for f in ADAPTER.frames(store) if f.name.startswith("panel:"))
    assert not any(g.startswith("NegControl") for g in panel.members)


# --------------------------------------------------------------------------- #
# 2. Multi-sample with NO declared mode -> blocker, as a checklist line
# --------------------------------------------------------------------------- #
def test_undeclared_clustering_fires_on_a_real_store(store, tmp_path) -> None:
    """§6.3's fourth blocker, against a store on disk rather than a dict literal."""
    blind = _copy(store, tmp_path, drop_key="clustering_mode")
    assert ADAPTER.samples(blind) == ["sampleA", "sampleB"]
    with pytest.raises(UndeclaredClustering) as exc:
        ADAPTER.clustering_mode(blind)
    msg = str(exc.value)
    assert "holds 2 samples (sampleA, sampleB)" in msg
    assert "manifest.json" in msg, "the biologist's next action is to open that file"
    assert "joint | independent | integrated" in msg


def test_intake_renders_the_blocker_as_a_checklist_line(store, tmp_path) -> None:
    """A traceback is not a checklist (§6.4). Intake must catch and render it."""
    blind = _copy(store, tmp_path, drop_key="clustering_mode")
    draft = derive(ADAPTER, blind)

    assert draft.status(CLUSTERING) is Status.BLOCKING
    assert CLUSTERING in draft.missing
    assert not draft.complete

    text = draft.checklist().render()
    assert "Missing and BLOCKING" in text
    line = next(l for l in text.splitlines() if CLUSTERING in l)
    assert "holds 2 samples" in line
    # Everything else the store does say is still known -- an incomplete study
    # costs one row, not six.
    assert {i.field for i in draft.checklist().of(Status.KNOWN)} >= {
        "measured_set", "assay", "tissue"}

    # And the declared fixture resolves, so the blocker is conditional, not a wall.
    assert derive(ADAPTER, store).complete


def test_the_adapter_reports_a_multisample_store_with_no_per_sample_tables(
        store, tmp_path) -> None:
    """Added because the fixture proved it missing.

    With the per-sample tables removed, ``validate`` returned [] for a store whose
    concordance check could never run: ``samples`` said two, and the only readable
    marker table was the pooled one.
    """
    copy = _copy(store, tmp_path)
    shutil.rmtree(copy.inputs / "samples")
    assert ADAPTER.sample_inputs(copy) == {}
    problem = next(p for p in ADAPTER.validate(copy) if "per-sample" in p)
    assert "sampleA, sampleB" in problem
    assert "concordance" in problem


# --------------------------------------------------------------------------- #
# 3. Panel absence at severity
# --------------------------------------------------------------------------- #
def test_most_canonical_markers_are_never_measured(store: Study) -> None:
    """The regime the real second study is in, as a census rather than a vibe.

    Two frames do the refusing and they say different things: ``CANON_*`` was never
    on the panel, and ``SYNG10`` was on the panel and produced no usable signal.
    Both are NEVER_MEASURED and neither absence argues -- but only the second is a
    fact about this section rather than about the panel design, which is why the
    adapter declares two gene frames and not one.
    """
    world = MeasuredWorld(ADAPTER.frames(store))
    ev = _evidence(store)

    with cfg.use_study(FIXTURE):
        ann = annotation.load_annotation()
        census: Counter = Counter()
        refused_by: Counter = Counter()
        for uid, meta in ann.items():
            reported = frozenset(i.subject.one("g").id for i in ev[uid].items
                                 if i.measured.carries_number)
            for g in meta["canonical_markers"]:
                m = resolve(subject("gene", g=gene(g)), world, reported)
                census[m.state] += 1
                if m.state is MeasuredState.NEVER_MEASURED:
                    refused_by[m.frame.split(":")[0]] += 1

    assert sum(census.values()) == 18
    assert census[MeasuredState.NEVER_MEASURED] == 16
    assert census[MeasuredState.OBSERVED] == 1
    assert census[MeasuredState.TESTED_ABSENT] == 1
    # off the panel entirely, vs on the panel with no usable signal
    assert dict(refused_by) == {"panel": 15, "detect": 1}


def test_a_cell_type_can_have_no_canonical_support_at_all(store: Study) -> None:
    """Synthetic Type B's only on-panel canonical marker is undetectable.

    This is the shape that matters: a positive call for a type the panel cannot
    corroborate at all. Here the rubric does the honest thing.
    """
    world = MeasuredWorld(ADAPTER.frames(store))
    ev = _evidence(store)
    with cfg.use_study(FIXTURE):
        canon = annotation.canonical_markers("Synthetic Type B")
    reported = frozenset(i.subject.one("g").id for i in ev["c2"].items
                         if i.measured.carries_number)
    states = {resolve(subject("gene", g=gene(g)), world, reported).state for g in canon}
    assert states == {MeasuredState.NEVER_MEASURED}

    with cfg.use_study(FIXTURE):
        v = next(v for v in _verdicts() if v.cluster == "c2")
    assert v.confidence == "Low" and v.verify is True
    assert "no canonical marker supports the call" in v.demotions


def test_no_breast_knowledge_reaches_a_study_that_is_not_breast(store: Study) -> None:
    """``test_study_plurality`` notes both studies on disk are breast-derived, so
    tissue is deliberately not a discriminator there. Here it is one."""
    assert "breast" not in str(store.manifest["tissue"]).lower()
    with cfg.use_study(FIXTURE):
        assert annotation.canonical_markers("Tumor") == ()
        assert annotation.canonical_markers("Stromal") == ()
        assert annotation.offpanel_canonical("Stromal") == ()
        # and the labels this study DOES carry resolve from its own annotation
        assert annotation.canonical_markers("Synthetic Type A")[0] == "SYNG01"


def test_the_cliff_at_one_canonical_marker_is_gone(store: Study) -> None:
    """The correction, with the number it is conditioned on.

    c1 has ONE canonical marker in frame of six. It used to read Very High (0.95)
    with ``verify=False`` and no demotion at all, while the off-panel notes sat
    beside that number saying how little had been measurable. It now reads one rung
    lower AND carries verify, because a single rung off Very High is still High,
    which reads as settled.

    Both halves are asserted. A demote alone would leave a confident-looking band on
    a call the panel could barely ask for; a verify alone would leave the number
    saying something the rubric no longer believes.
    """
    from panoscope.agent import verdict as V

    with cfg.use_study(FIXTURE):
        by_unit = {v.cluster: v for v in _verdicts()}
        coverage = V.canonical_coverage("Synthetic Type A")

    # The K of N the demote rests on, stated rather than implied.
    assert (coverage.k, coverage.n) == (1, 6)
    assert coverage.floor == 2 and not coverage.ok

    c1 = by_unit["c1"]
    assert c1.confidence == "High" and c1.confidence_score == 0.85
    assert c1.verify is True
    assert c1.demotions == ("canonical coverage 1/6 below floor 2 -> -1 band",)
    assert c1.band_basis == "glm_coef direct"
    # The conditionality is still recorded per gene, and now also reaches the band.
    assert len(c1.offpanel_notes) == 5
    assert all("absence is not evidence against" in n.message
               for n in c1.offpanel_notes)


def test_the_coverage_demote_is_one_rung_and_not_a_collapse(store: Study) -> None:
    """Thin canonical support is not the same answer as none, and must not read as it.

    c1 (one canonical marker in frame) and c2 (none at all) both now carry verify,
    which is the point -- but they are different findings and the band has to keep
    them apart. Collapsing both to Low would make "the panel could barely test this"
    indistinguishable from "nothing canonical supports this", which is the same
    collapse the four measurement states exist to prevent.
    """
    with cfg.use_study(FIXTURE):
        by_unit = {v.cluster: v for v in _verdicts()}

    assert by_unit["c1"].confidence == "High"
    assert by_unit["c2"].confidence == "Low"
    assert by_unit["c1"].verify is by_unit["c2"].verify is True
    assert by_unit["c1"].demotions != by_unit["c2"].demotions


def test_no_demo_cluster_is_demoted_by_the_coverage_floor(store: Study) -> None:
    """The floor was chosen inside a measured band; this is the measurement.

    Every demo unit's canonical coverage clears the floor and the fixture's does
    not, which is what makes the floor a decision about studies rather than a number
    that happened to leave the golden hash alone. Stated as the census it is, so a
    future tuning pass sees how much headroom there is: the worst demo unit is c7 at
    4/7 and the fixture sits at 1/6.
    """
    from panoscope.agent import verdict as V

    with cfg.use_study(cfg.BUNDLED_DEMO_ID):
        demo = {u: V.canonical_coverage(annotation.cell_type_for(u))
                for u in cfg.CLUSTER_ORDER}
    assert len(demo) == 9, "nine units, never one"
    assert all(c.ok for c in demo.values()), {
        u: str(c) for u, c in demo.items() if not c.ok}
    worst = min(demo.values(), key=lambda c: c.fraction)
    assert round(worst.fraction, 3) == 0.571, str(worst)

    with cfg.use_study(FIXTURE):
        thin = V.canonical_coverage("Synthetic Type A")
    assert not thin.ok and round(thin.fraction, 3) == 0.167

    # The floor sits strictly between the two, and not on either edge.
    assert thin.fraction < 1 / 3 < worst.fraction


# --------------------------------------------------------------------------- #
# 4. Two units, one label
# --------------------------------------------------------------------------- #
def test_two_units_sharing_a_label_stay_two_subjects(store: Study) -> None:
    """What ``SHAPES["unit"]`` was added for.

    c1 and c3 are both called "Synthetic Type A". Keyed on the label they are one
    subject and their evidence silently pools; keyed on the unit they are two,
    which is what the ASSIGNMENT task kind requires -- the label is the thing being
    decided, so a subject keyed on it would be circular.
    """
    with cfg.use_study(FIXTURE):
        labels = {u: annotation.cell_type_for(u) for u in store.units}
    assert labels["c1"] == labels["c3"] == "Synthetic Type A"

    ev = _evidence(store)
    keys = {u: ev[u].unit_subject.key for u in ("c1", "c3")}
    assert keys == {"c1": "unit|u=unit:c1", "c3": "unit|u=unit:c3"}

    # The label-keyed subject is the collapse this avoids -- shown, not assumed.
    collapsed = {subject("cell_type", ct=cell_type(labels[u])).key for u in ("c1", "c3")}
    assert len(collapsed) == 1

    # And their evidence stays apart: c1 observed SYNG01, c3 tested it and did not.
    def state(u: str, g: str) -> MeasuredState:
        return next(i for i in ev[u].items
                    if i.subject.one("g").id == g).measured.state

    assert state("c1", "SYNG01") is MeasuredState.OBSERVED
    assert state("c3", "SYNG01") is MeasuredState.TESTED_ABSENT

    # The cell-type frame DOES collapse them, and that is correct: it is the label
    # space, and the label space has two labels for three units.
    ct_frame = next(f for f in ADAPTER.frames(store) if f.kind is EntityKind.CELL_TYPE)
    unit_frame = next(f for f in ADAPTER.frames(store) if f.kind is EntityKind.UNIT)
    assert len(ct_frame.members) == 2 and len(unit_frame.members) == 3


def test_the_unit_frame_admits_each_unit_separately(store: Study) -> None:
    world = MeasuredWorld(ADAPTER.frames(store))
    for u in store.units:
        m = resolve(subject("unit", u=unit_entity(u)), world, frozenset(store.units))
        assert m.state is MeasuredState.OBSERVED, (u, m)


# --------------------------------------------------------------------------- #
# 5. Concordance, with data under it at last
# --------------------------------------------------------------------------- #
def test_concordance_runs_across_the_samples(store: Study) -> None:
    """§9.10 on real evidence: a flag, a denominator, and no band anywhere.

    sampleB assigns SYNG03 to c2 instead of c1. For unit c1 that is sampleA
    reporting a coefficient and sampleB having had the gene in frame, tested it and
    not reported it -- a measured absence, so the two genuinely conflict. The three
    undetectable genes are not comparable in either sample and are reported as such
    rather than as agreement.
    """
    assert set(ADAPTER.sample_inputs(store)) == {"sampleA", "sampleB"}

    ev = {s: _evidence(store, s) for s in ADAPTER.samples(store)}
    reps = tuple(Replicate(s, ev[s]["c1"].items) for s in ADAPTER.samples(store))

    c = concordance("c1", reps, joint=ADAPTER.clustering_mode(store) == "joint").value

    assert c.samples == ("sampleA", "sampleB") and c.reference == "sampleA"
    assert c.divergent is True
    assert [s.key for s in c.diverging_subjects] == ["gene|g=gene:SYNG03"]

    assert Counter(s.outcome for s in c.per_subject) == {
        Comparison.CONSISTENT: 8,
        Comparison.CONFLICT: 1,
        Comparison.NOT_COMPARABLE: 3,
    }
    assert (c.coverage.k, c.coverage.n) == (9, 12) and c.coverage.ok

    # No band, no score, no weight -- agreement must have nowhere to raise one.
    assert not hasattr(c, "band") and not hasattr(c, "score")

    why = next(s.because for s in c.per_subject if s.outcome is Comparison.CONFLICT)
    assert "a measured absence" in " ".join(why)

    nc = next(s for s in c.per_subject if s.outcome is Comparison.NOT_COMPARABLE)
    assert "never_measured" in " ".join(nc.because)


def test_concordance_refuses_under_independent_clustering(store: Study) -> None:
    """The same data, the other mode: a traceback is the only honest output."""
    ev = {s: _evidence(store, s) for s in ADAPTER.samples(store)}
    reps = tuple(Replicate(s, ev[s]["c1"].items) for s in ADAPTER.samples(store))
    with pytest.raises(IndependentClustering):
        concordance("c1", reps, joint=False)


def test_every_unit_is_concordance_checkable(store: Study) -> None:
    """Not just c1. Each unit keeps its own denominator, and the one unit the two
    samples agree about reports that as the absence of a flag, not as a score."""
    ev = {s: _evidence(store, s) for s in ADAPTER.samples(store)}
    divergent = {}
    for u in store.units:
        reps = tuple(Replicate(s, ev[s][u].items) for s in ADAPTER.samples(store))
        c = concordance(u, reps, joint=True).value
        assert c.coverage.n == 12
        divergent[u] = c.divergent
    # SYNG03 moved out of c1 and into c2, so exactly those two diverge.
    assert divergent == {"c1": True, "c2": True, "c3": False}


# --------------------------------------------------------------------------- #
# The bundled breast fallback belongs to the bundled demo and to nothing else
# --------------------------------------------------------------------------- #
def test_a_study_with_no_marker_table_does_not_inherit_the_demos(tmp_path):
    """``data/jazzpanda/markers_top.csv`` is the DEMO's flat file, not a default.

    ``config._active_input`` fell back to it for any study, so a store missing its
    own table read the demo's: ``cfg.CLUSTER_ORDER`` came back as nine clusters and
    ``load_panel`` as 280 genes, under the other study's id. The onboarding line
    showed it first -- "9 clusters, 280-gene panel" over a kidney store -- but
    ``CLUSTER_ORDER`` is also what Stage 0's unknown-label guard compares against,
    and the panel is the capability frame every absence is judged in.

    The honest answer for a study with no file is the path that does not exist, so
    the reader raises and names it.
    """
    import pytest

    from panoscope.agent import data as agent_data
    from panoscope.data.study import Study

    root = tmp_path / "studies" / "kidney_like"
    (root / "inputs").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"dataset_id": "kidney_like", "platform": "Xenium",
                    "tissue": "kidney"}))
    st = Study.open("kidney_like", data_root=tmp_path)

    with cfg.use_study(st):
        resolved = cfg.ACTIVE_MARKERS_CSV
        assert "kidney_like" in str(resolved), (
            f"a non-demo study resolved to {resolved} -- the bundled demo's file"
        )
        assert not resolved.exists()
        with pytest.raises((FileNotFoundError, ValueError)) as exc:
            _ = cfg.CLUSTER_ORDER
        assert "kidney_like" in str(exc.value)

    # ...and the bundled demo still gets its fallback, which is what it is for.
    with cfg.use_study(cfg.BUNDLED_DEMO_ID):
        assert len(cfg.CLUSTER_ORDER) == 9
