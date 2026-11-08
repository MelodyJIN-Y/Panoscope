"""§9.10 on the PRODUCTION path: the build reads the per-sample tables, or says why not.

``core/concordance.py`` had no caller outside tests. The cost was specific and worse
than a missing feature: ``data/studies/synthetic_multisample_fixture`` has two
per-sample marker tables, ``XeniumAdapter.sample_inputs`` resolves both, and
``validate`` passes clean *because those files are present* -- and then nothing opened
them. Every record's ``reading.path`` was the pooled table, so a unit whose profile
differs between sections came out as one confident number with no caveat anywhere on
it. A confident wrong answer, not a refusal.

So these tests are about the record a biologist reads, not about the function. Each
one builds through ``pipeline.records.build_all`` -- the single root behind both
``panoscope build`` and ``panoscope explain`` -- and asserts on what came out the
other end. A test that called ``concordance`` directly would have passed before this
wiring existed and after it was removed again.

The pairing is the point: the multi-sample study must carry the comparison, and the
single-sample study must carry the sentence saying there was none to make. Silence
is the bug; either answer is fine.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest

from panoscope import methods
from panoscope.core.reconcile import Comparison
from panoscope.data.study import Study
from panoscope.pipeline import records
from panoscope.platforms.xenium import XeniumAdapter

DEMO = "xenium_hbreast_sample1"
FIXTURE = "synthetic_multisample_fixture"
ADAPTER = XeniumAdapter()


def _method():
    return methods.for_workflow(records.WORKFLOW_ID)[0]


def _concordance_rules(record) -> Counter:
    return Counter(d.rule for d in record.decisions
                   if d.rule.startswith("concordance:"))


# --------------------------------------------------------------------------- #
# 1. The multi-sample half: both tables are opened, and the divergence arrives
# --------------------------------------------------------------------------- #
def test_the_build_opens_both_per_sample_tables_and_carries_the_divergence() -> None:
    """sampleB assigns SYNG03 to c2; sampleA assigns it to c1. That must reach c1.

    The numbers here are the ones ``tests/test_second_study.py`` already pins off a
    direct call -- 8 consistent, 1 divergent, 3 not comparable. Asserting them again
    off ``build_all`` is deliberate duplication: it is the only way to show the
    production root reaches the same answer rather than skipping the step.
    """
    built = records.build_all(Study.open(FIXTURE))

    assert _concordance_rules(built["c1"].record) == {
        "concordance:consistent": 8,
        "concordance:divergent": 1,
        "concordance:not_comparable": 3,
    }
    divergent = [d for d in built["c1"].record.decisions
                 if d.rule == "concordance:divergent"]
    assert [d.outcome for d in divergent] == ["gene|g=gene:SYNG03"]

    # And the biologist's own surface, not just the trace: ``panoscope explain``
    # prints ``undecidable`` as its own section, and §9.10 says the cause genuinely
    # is undecided -- batch effect, section quality or real biology.
    why = " ".join(built["c1"].record.undecidable)
    assert "SYNG03" in why and "sampleA, sampleB" in why
    assert "does not claim which" in why


def test_the_unit_the_samples_agree_about_carries_no_divergence_caveat() -> None:
    """c3 is the control. Agreement must leave NO trace that could be read as a lift.

    §9.10: agreement never raises confidence, and the one thing worse than missing a
    divergence is manufacturing a reassurance. c3's record carries the per-subject
    consistency decisions and nothing in ``undecidable`` at all.
    """
    rec = records.build_all(Study.open(FIXTURE))["c3"].record
    assert _concordance_rules(rec)["concordance:divergent"] == 0
    assert _concordance_rules(rec)["concordance:consistent"] == 9
    # The ONE sentence agreement is allowed here is the denominator: how many
    # subjects were comparable out of how many. It carries no flag, no reassurance
    # and no word that could be read as a lift -- and it travels unconditionally,
    # because "9 of 12" and "1 of 12" must not both render as silence.
    assert rec.undecidable == (
        "replicate concordance compared 9 of 12 subject(s) across sampleA, sampleB",
    )


def test_a_subject_outside_the_frame_is_not_a_divergence() -> None:
    """The lattice, on the production path. The half that makes this not noise.

    The fixture's three undetectable genes are NEVER_MEASURED in both samples. A
    check that read "no number in either sample" as "the samples disagree" would
    report a divergence for every one of them -- and on the kidney panel, where most
    canonical markers are off-panel, that is the whole result. They must land on
    not-comparable, and nothing about them may reach ``undecidable``.
    """
    traced, because, _unc = records._concordance(
        Study.open(FIXTURE), ADAPTER, _method(), "joint", ("c1", "c2", "c3"))
    c = traced["c1"].value

    unmeasured = [s for s in c.per_subject
                  if s.outcome is Comparison.NOT_COMPARABLE]
    assert len(unmeasured) == 3
    for s in unmeasured:
        assert "never_measured" in " ".join(s.because), s
        assert s.subject not in c.diverging_subjects
        assert s.subject.display() not in " ".join(because.get("c1", ()))


def _sections_disagree_about_what_they_saw(tmp_path: Path) -> Study:
    """The fixture, rewritten so the two sections have DIFFERENT detectable sets.

    Two edits, one per direction of the leak, and both are the ordinary case rather
    than a contrived one -- gene dropout between sections is what an IO panel on a
    mismatched tissue does all day, and the demo already runs 17.5% of its panel at
    no usable signal.

    * ``SYNG09``: a coefficient in sampleA, ``NoSig`` in sampleB. Asymmetric. The
      pooled table still has the coefficient, so a pooled detectability frame says
      sampleB tested it -- and sampleB's silence then argues.
    * ``SYNG06``: ``NoSig`` in BOTH sections, still visible in the pooled table,
      which is what pooling two weak sections does. Symmetric, and the one the
      panel frame cannot catch, because the gene IS on the panel.

    The fixture on disk is not touched: it is copied into ``tmp_path`` and edited
    there, so this test can describe a study the repo does not have to ship.
    """
    dst = tmp_path / "studies" / "copy"
    shutil.copytree(Study.open(FIXTURE).dir, dst)
    for sample, line, replacement in (
        ("sampleB", "SYNG09,c3,11.0,0.9,0.9,0.9,Synthetic Type A",
                    "SYNG09,NoSig,0.0,0.0,0.0,0.0,"),
        ("sampleA", "SYNG06,c2,14.0,0.9,0.9,0.9,Synthetic Type B",
                    "SYNG06,NoSig,0.0,0.0,0.0,0.0,"),
        ("sampleB", "SYNG06,c2,14.0,0.9,0.9,0.9,Synthetic Type B",
                    "SYNG06,NoSig,0.0,0.0,0.0,0.0,"),
    ):
        p = dst / "inputs" / "samples" / sample / "markers_top.csv"
        text = p.read_text(encoding="utf-8")
        assert line in text, (sample, line)   # the edit must actually land
        p.write_text(text.replace(line, replacement), encoding="utf-8")
    return Study(id="copy", data_root=tmp_path)


def test_a_gene_one_section_never_saw_is_not_a_divergence(tmp_path: Path) -> None:
    """The ASYMMETRIC half: observed in sampleA, ``NoSig`` in sampleB.

    This is the case the symmetric test above cannot reach and the panel frame
    cannot catch, because SYNG09 is on the panel and both sections were capable of
    it in principle -- only sampleB's own file says it produced no usable signal.
    Under one pooled detectability frame the record said, in these words, that
    sampleB "had it in frame ... tested it, and did not report it -- a measured
    absence, so they genuinely conflict". Sample B tested nothing of the sort. That
    is a fabricated measurement status reaching the biologist's surface, which is
    the one failure the four-state lattice exists to make unrepresentable.

    Asserted on the record, not on the function: the frames are chosen by
    ``_concordance`` when it loads each replicate, so a test that built the
    replicates itself would pass with the leak still in the build.
    """
    study = _sections_disagree_about_what_they_saw(tmp_path)
    rec = records.build_all(study)["c3"].record

    assert _concordance_rules(rec)["concordance:divergent"] == 0
    assert "SYNG09" not in " ".join(rec.undecidable)

    traced, _, _unc = records._concordance(study, ADAPTER, _method(), "joint", ("c3",))
    s9 = next(s for s in traced["c3"].value.per_subject
              if s.subject.key.endswith("SYNG09"))
    assert s9.outcome is Comparison.NOT_COMPARABLE
    why = " ".join(s9.because)
    # The sample's OWN frame is named, which is also how the record stays clickable:
    # one frame name for two different detectable sets would make the two sections
    # cite the same evidence for different capabilities.
    assert "sampleB" in why and "detect:copy:sampleB" in why


def test_a_gene_neither_section_saw_is_not_counted_as_agreement(tmp_path: Path) -> None:
    """The quiet half: the leak also INFLATES the denominator it was added to guard.

    SYNG06 is ``NoSig`` in both sections and still carries a coefficient in the
    pooled table. Under a pooled frame both sections were credited with having
    tested it and neither reporting it -- CONSISTENT, and a tick in ``coverage.k``.
    Agreement manufactured out of two measurements that were never made, in the one
    number the kidney case depends on: K of N comparable is what stops "not
    divergent" reading the same at 1-of-200 and at 200-of-200.

    Nine of twelve before, seven after: SYNG06 and SYNG09 both leave the numerator.
    """
    study = _sections_disagree_about_what_they_saw(tmp_path)
    traced, because, _unc = records._concordance(
        study, ADAPTER, _method(), "joint", ("c3",))
    c = traced["c3"].value

    s6 = next(s for s in c.per_subject if s.subject.key.endswith("SYNG06"))
    assert s6.outcome is Comparison.NOT_COMPARABLE
    assert sorted(s6.blocked) == ["sampleA", "sampleB"]

    assert (c.coverage.k, c.coverage.n) == (7, 12)
    assert because["c3"] == (
        "replicate concordance compared 7 of 12 subject(s) across sampleA, sampleB",
    )


# --------------------------------------------------------------------------- #
# 2. The half that is absence: every refusal names itself in the same field
# --------------------------------------------------------------------------- #
def test_a_single_sample_study_says_concordance_does_not_apply() -> None:
    """The demo. No concordance decision anywhere, and a sentence in every record.

    This is the leak guard as much as the honesty guard: the demo is single-sample,
    so a ``concordance:*`` rule appearing on any of its nine records means the
    multi-sample wiring is running on a join that does not exist -- which is also
    what would move the golden.
    """
    built = records.build_all(Study.open(DEMO))
    assert len(built) == 9

    for unit, b in built.items():
        assert _concordance_rules(b.record) == {}, unit
        why = " ".join(b.record.undecidable)
        assert "9.10" in why and "one sample" in why, (unit, why)
        # And the band engine must have been told *not assessed* rather than
        # *assessed and empty*. This is the golden's discriminator: the demo is
        # the study tests/test_golden.py pins, so a replicate demote firing on
        # one sample is the inverted-loudness bug arriving in the committed
        # record -- a study that cannot support the check losing a rung to it.
        assert b.verdict.replicate_coverage is None, unit
        assert "band:demote_replicate_coverage" not in b.verdict.demotions, unit


@pytest.mark.req("ARCH-5.3-01")
def test_independent_clustering_is_refused_with_the_reason_in_the_record() -> None:
    """Joint clustering is what licenses the join; the mode is read, never assumed.

    ``core.concordance`` raises ``IndependentClustering`` rather than returning a
    verdict, which is right for a library and wrong for a build -- a traceback is not
    something a biologist can read off a record. The root catches the condition
    before the call and writes the refusal down instead.
    """
    traced, because, _unc = records._concordance(
        Study.open(FIXTURE), ADAPTER, _method(), "independent", ("c1", "c3"))
    assert traced == {}
    for unit in ("c1", "c3"):
        why = " ".join(because[unit])
        assert "'independent'" in why and "never joined" in why


def test_a_declared_sample_with_no_table_is_named_not_quietly_dropped() -> None:
    """``sample_inputs`` omits a sample with no table. That omission must be audible.

    ``XeniumAdapter.validate`` already refuses this store, but ``validate`` is the
    intake checklist and a record built anyway must not read as though the
    comparison was made across every declared sample.
    """
    class _Adapter:
        def samples(self, store):      return ["sampleA", "sampleB", "sampleC"]
        def sample_inputs(self, store): return {"sampleA": Path("a.csv")}

    traced, because, _unc = records._concordance(
        Study.open(FIXTURE), _Adapter(), _method(), "joint", ("c1",))
    assert traced == {}
    assert "sampleB, sampleC" in because["c1"][0]
    assert "Only the pooled table is readable" in because["c1"][0]


def test_a_unit_only_one_sample_reported_drops_that_sample_and_says_so() -> None:
    """A unit missing from a sample's table is not an empty replicate.

    ``concordance`` adjudicates a missing row as not-comparable naming the sample, so
    joining a zero-item replicate would turn every subject of that unit into one --
    a check seen to run and find nothing, which reads as agreement. The sample is
    dropped and named; with fewer than two left, the unit gets the refusal.
    """
    class _Method:
        def load(self, path, adapter, store):
            from panoscope.types.trace import Traced
            return Traced({"c1": object()} if path.endswith("A.csv") else {"c2": object()})

    class _Adapter:
        def samples(self, store):      return ["sampleA", "sampleB"]
        def sample_inputs(self, store): return {"sampleA": Path("A.csv"),
                                                "sampleB": Path("B.csv")}
        # Each replicate is loaded under its OWN frames, so the root asks the
        # adapter for a sample-bound view. This stub declares no frames at all.
        def for_sample(self, sample):  return self

    traced, because, _unc = records._concordance(
        Study.open(FIXTURE), _Adapter(), _Method(), "joint", ("c1",))
    assert traced == {}
    why = because["c1"][0]
    assert "sampleB reported no row under it" in why
    assert "1 sample(s) to compare" in why


# --------------------------------------------------------------------------- #
# 3. Coverage reaching the BAND -- the half that was missing entirely
# --------------------------------------------------------------------------- #
#: SampleB's rows for the six genes a bad second section loses. Gene dropout
#: between sections is the ordinary failure, not a contrived one; this is the same
#: edit ``_sections_disagree_about_what_they_saw`` makes, at the scale where it
#: takes the comparison below half of the unit's subjects.
_SECTION_DROPOUT: tuple[str, ...] = (
    "SYNG04,c2,16.0", "SYNG05,c2,15.0", "SYNG06,c2,14.0",
    "SYNG07,c3,13.0", "SYNG08,c3,12.0", "SYNG09,c3,11.0",
)


def _one_unit_built_two_ways(tmp_path: Path, name: str, *, dropout: bool) -> Study:
    """The fixture with c1's canonical list ON PANEL, and sampleB optionally blind.

    Two edits, and the first is what makes the test discriminate rather than merely
    pass. The shipped fixture gives c1 five off-panel canonical markers, so every
    build of it is already demoted by ``band:demote_canonical_coverage`` and already
    carries ``verify`` -- a replicate rule asserted against that baseline would be
    asserted against a band and a flag it did not move. Putting SYNG01-03 on panel
    makes c1 a clean Very High with ``verify`` false, which is exactly the shape the
    defect report measured: *9 of 12 comparable -> Very High, verify False* and
    *1 of 12 comparable -> Very High, verify False*, byte-identical.

    The second edit is the only difference between the two studies this builds.
    ``dropout`` NoSigs six genes in sampleB's own table, so sampleB never tested
    them and ``_compare`` lands them on NOT_COMPARABLE -- 3 of 12 instead of 9 of 12,
    with the pooled table, the annotation, the panel and sampleA all untouched.
    """
    dst = tmp_path / "studies" / name
    shutil.copytree(Study.open(FIXTURE).dir, dst)

    lab = dst / "inputs" / "reference" / "lab_markers.csv"   # c1's label's canonical list
    rows = [r for r in lab.read_text(encoding="utf-8").splitlines()
            if not r.startswith("Synthetic Type A,")]
    tissue = rows[1].split(",", 2)[2]
    lab.write_text("\n".join(rows + [f"Synthetic Type A,SYNG0{i},{tissue}"
                                     for i in (1, 2, 3)]) + "\n", encoding="utf-8")

    if dropout:
        p = dst / "inputs" / "samples" / "sampleB" / "markers_top.csv"
        text = p.read_text(encoding="utf-8")
        for head in _SECTION_DROPOUT:
            line = next(ln for ln in text.splitlines() if ln.startswith(head))
            text = text.replace(line, f"{head.split(',')[0]},NoSig,0.0,0.0,0.0,0.0,")
        p.write_text(text, encoding="utf-8")
    return Study(id=name, data_root=tmp_path)


def test_thin_replicate_coverage_demotes_the_band_and_sets_verify(tmp_path: Path) -> None:
    """The headline defect: one unit built two ways, differing only in coverage.

    Before this wiring, ``records.build_all`` computed the band from the POOLED
    items and spliced the concordance decisions into the record afterwards, so the
    two builds below came out byte-identical -- same band, same ``verify``, and the
    only trace of the difference was one unflagged sentence in ``undecidable`` that
    read the same at 1 of 12 as at 9 of 12, because the floor was a placeholder 1
    that ``Coverage.ok`` cleared everywhere.

    So the assertions are a PAIR. The rich build must still commit, or a rule that
    demoted everything would pass the thin half and be worse than none at all; the
    thin build must lose a rung and gain the flag. The two studies differ in six
    rows of one sample's table and in nothing else.
    """
    rich = records.build_all(_one_unit_built_two_ways(tmp_path, "rich", dropout=False))["c1"]
    thin = records.build_all(_one_unit_built_two_ways(tmp_path, "thin", dropout=True))["c1"]

    assert (rich.verdict.replicate_coverage.k, rich.verdict.replicate_coverage.n) == (9, 12)
    assert rich.verdict.band == "Very High" and rich.verdict.verify is False
    assert "band:demote_replicate_coverage" not in rich.verdict.demotions

    assert (thin.verdict.replicate_coverage.k, thin.verdict.replicate_coverage.n) == (3, 12)
    assert thin.verdict.band == "High" and thin.verdict.verify is True
    assert "band:demote_replicate_coverage" in thin.verdict.demotions

    # One rung, and the rung is recorded with the K of N that caused it -- the
    # record says WHY, so the sentence on screen can click back to it.
    (d,) = [x for x in thin.record.decisions
            if x.rule == "band:demote_replicate_coverage"]
    assert d.outcome == "High" and d.because == ("subjects_comparable 3/12 (floor 6)",)


@pytest.mark.req("ARCH-9.10-01")
def test_agreement_across_sections_never_raises_anything(tmp_path: Path) -> None:
    """§9.10's asymmetry, on the production path: the rich build gains NOTHING.

    Replicates share a panel, a protocol and usually a day, so their agreeing is
    close to tautological. The rule may subtract and may set ``verify``; a build
    whose sections agree about every comparable subject must record no decision at
    all for it, not a decision saying the call survived a check.
    """
    rich = records.build_all(_one_unit_built_two_ways(tmp_path, "rich", dropout=False))["c1"]
    assert rich.verdict.replicate_coverage.ok is True
    assert not [d for d in rich.record.decisions
                if d.rule == "band:demote_replicate_coverage"]
    assert "replicate" not in " ".join(rich.verdict.demotions)


def test_the_thin_sentence_does_not_read_like_the_rich_one(tmp_path: Path) -> None:
    """"3 of 12" and "9 of 12" are the same shape of sentence. Make thin LOOK thin.

    The number alone is a caveat a reader skips, in the same neutral field as every
    other caveat. The thin sentence has to name its consequence -- that the band
    moved and the call is flagged -- because that is the part the biologist acts on.
    """
    rich = records.build_all(_one_unit_built_two_ways(tmp_path, "rich", dropout=False))["c1"]
    thin = records.build_all(_one_unit_built_two_ways(tmp_path, "thin", dropout=True))["c1"]

    (rich_line,) = [s for s in rich.record.undecidable if "compared" in s]
    (thin_line,) = [s for s in thin.record.undecidable if "compared" in s]
    assert rich_line == (
        "replicate concordance compared 9 of 12 subject(s) across sampleA, sampleB")
    assert thin_line.startswith(
        "replicate concordance compared 3 of 12 subject(s) across sampleA, sampleB:")
    assert "fewer than the floor of 6" in thin_line
    assert "demoted one rung and flagged for re-check" in thin_line


def test_a_sample_dropped_from_a_comparison_that_still_ran_is_named(tmp_path: Path) -> None:
    """sampleC exists, has no table, and the comparison runs anyway across A and B.

    ``test_a_declared_sample_with_no_table_is_named_not_quietly_dropped`` covers the
    case where the drop leaves fewer than two samples and the whole check is refused.
    This is the quiet one: two readable samples are enough, so the check runs and
    reports "across sampleA, sampleB" -- a sentence that reads as the whole study
    unless the study's third section is named beside it.
    """
    class _Adapter(XeniumAdapter):
        def samples(self, store):
            return [*super().samples(store), "sampleC"]

    study = _one_unit_built_two_ways(tmp_path, "rich", dropout=False)
    _, because, _unc = records._concordance(study, _Adapter(), _method(), "joint", ("c1",))

    dropped = [s for s in because["c1"] if s.startswith("not compared:")]
    assert dropped == [
        "not compared: sampleC -- declared by the study but with no per-sample "
        "marker table, so it was left out of every unit's comparison"
    ]


def test_a_sample_silent_about_one_unit_is_named_on_that_unit(tmp_path: Path) -> None:
    """The other drop, kept apart from the first because the fix differs.

    sampleC HAS a table here; it just holds no row under c1. That is a fact about
    the method's output, not a missing file, and the record must not guess which --
    so it gets its own sentence on the unit it affects rather than being folded into
    the study-wide one.
    """
    class _Adapter(XeniumAdapter):
        def samples(self, store):
            return [*super().samples(store), "sampleC"]

        def sample_inputs(self, store):
            got = dict(super().sample_inputs(store))
            got["sampleC"] = got["sampleA"].parent.parent / "sampleC" / "markers_top.csv"
            return got

    study = _one_unit_built_two_ways(tmp_path, "rich", dropout=False)
    src = study.dir / "inputs" / "samples" / "sampleA" / "markers_top.csv"
    dst = study.dir / "inputs" / "samples" / "sampleC" / "markers_top.csv"
    dst.parent.mkdir(parents=True)
    # sampleC saw everything except c1: its rows are relabelled onto c3.
    dst.write_text(src.read_text(encoding="utf-8").replace(",c1,", ",c3,"),
                   encoding="utf-8")

    _, because, _unc = records._concordance(study, _Adapter(), _method(), "joint", ("c1",))
    silent = [s for s in because["c1"] if s.startswith("not compared for this unit:")]
    assert silent == [
        "not compared for this unit: sampleC -- readable, but reported no row "
        "under it. Whether the method found nothing there or the unit is absent "
        "from that section is not in the table"
    ]


# --------------------------------------------------------------------------- #
# 6. Zero comparisons is the THINNEST coverage, not the absence of a question
# --------------------------------------------------------------------------- #
def _no_second_section_speaks(tmp_path: Path, name: str) -> Study:
    """``_one_unit_built_two_ways``'s study, with sampleB reporting no c1 row at all.

    Not the same as the dropout case. There, sampleB tested the genes and reported
    nothing for six of them. Here sampleB's table carries no row under c1 whatsoever,
    which is what a section that simply does not contain a unit looks like -- and on
    an off-target panel against a tissue it was not designed for, it is the ordinary
    case rather than the edge one.
    """
    study = _one_unit_built_two_ways(tmp_path, name, dropout=False)
    p = study.dir / "inputs" / "samples" / "sampleB" / "markers_top.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    kept = [lines[0]] + [ln for ln in lines[1:] if ",c1," not in ln]
    p.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return study


def test_a_unit_no_second_section_reported_is_demoted_not_waved_through(
    tmp_path: Path,
) -> None:
    """0 of N comparable must demote, exactly as 1 of N does.

    This is the last place the loudness gradient inverted. ``_concordance`` dropped
    a unit fewer than two samples spoke about out of ``conc`` entirely, so ``assess``
    received ``replicate=None`` -- which means *the question was never askable*. That
    is true of the single-sample demo and false here: the study is multi-sample and
    joint, both tables are readable, the question WAS asked, and the answer is that
    nothing could be compared. Measured before the fix: 1-of-12 came out Medium-High
    and demoted while 0-of-12 came out High with no demotion and shipped. The band
    went UP as the evidence ran out.

    In the project's own vocabulary that is ``UNDERPOWERED`` reported as
    ``NEVER_MEASURED`` -- and ``None`` must stay reserved for the genuinely
    unaskable, which ``test_a_single_sample_study_says_concordance_does_not_apply``
    pins from the other side.
    """
    rich = records.build_all(_one_unit_built_two_ways(tmp_path, "rich0", dropout=False))["c1"]
    none = records.build_all(_no_second_section_speaks(tmp_path, "none0"))["c1"]

    assert rich.verdict.replicate_coverage.k == 9
    assert "band:demote_replicate_coverage" not in rich.verdict.demotions

    cov = none.verdict.replicate_coverage
    assert cov is not None, "zero comparisons is a coverage, not 'not assessed'"
    assert cov.k == 0 and not cov.ok
    # Same denominator down both paths: the method's own subjects for this unit,
    # never the reference-selected canonical absences appended after it.
    assert cov.n == rich.verdict.replicate_coverage.n
    assert "band:demote_replicate_coverage" in none.verdict.demotions
    assert none.verdict.verify is True
    # Monotone: more coverage never reads as less confident. ``BANDS`` is ordered
    # BEST FIRST, so a demotion moves the index up, not down.
    from panoscope.core.verdict import BANDS
    assert BANDS.index(none.verdict.band) > BANDS.index(rich.verdict.band)


def test_the_zero_comparison_unit_still_projects_to_v1(tmp_path: Path) -> None:
    """``band:demote_replicate_coverage`` needs a v1 sentence or no study builds.

    ``pipeline/stages/verdicts.py`` projects every unit through ``to_v1_dict`` in an
    unguarded loop, so a rule with no entry in ``core/compat._demotion_sentence``
    aborted the whole verdicts stage with a bare ``NoV1Shape`` and took every OTHER
    unit's file down with it. The rule and its projection were added by different
    streams; each was correct alone, and together they meant a multi-sample study
    with one thin unit could not be built at all through the documented command.
    """
    none = records.build_all(_no_second_section_speaks(tmp_path, "none1"))["c1"]
    v1 = records.to_v1_dict(none)
    assert any("replicate coverage 0/" in d for d in v1["demotions"]), v1["demotions"]
    # And it must not borrow the canonical sentence: the two K-of-Ns count
    # different things and both can fire on one unit.
    assert sum("canonical coverage" in d for d in v1["demotions"]) <= 1
