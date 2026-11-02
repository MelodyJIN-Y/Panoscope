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

from collections import Counter
from pathlib import Path

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
    assert rec.undecidable == ()


def test_a_subject_outside_the_frame_is_not_a_divergence() -> None:
    """The lattice, on the production path. The half that makes this not noise.

    The fixture's three undetectable genes are NEVER_MEASURED in both samples. A
    check that read "no number in either sample" as "the samples disagree" would
    report a divergence for every one of them -- and on the kidney panel, where most
    canonical markers are off-panel, that is the whole result. They must land on
    not-comparable, and nothing about them may reach ``undecidable``.
    """
    traced, because = records._concordance(
        Study.open(FIXTURE), ADAPTER, _method(), "joint", ("c1", "c2", "c3"))
    c = traced["c1"].value

    unmeasured = [s for s in c.per_subject
                  if s.outcome is Comparison.NOT_COMPARABLE]
    assert len(unmeasured) == 3
    for s in unmeasured:
        assert "never_measured" in " ".join(s.because), s
        assert s.subject not in c.diverging_subjects
        assert s.subject.display() not in " ".join(because.get("c1", ()))


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


def test_independent_clustering_is_refused_with_the_reason_in_the_record() -> None:
    """Joint clustering is what licenses the join; the mode is read, never assumed.

    ``core.concordance`` raises ``IndependentClustering`` rather than returning a
    verdict, which is right for a library and wrong for a build -- a traceback is not
    something a biologist can read off a record. The root catches the condition
    before the call and writes the refusal down instead.
    """
    traced, because = records._concordance(
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

    traced, because = records._concordance(
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

    traced, because = records._concordance(
        Study.open(FIXTURE), _Adapter(), _Method(), "joint", ("c1",))
    assert traced == {}
    why = because["c1"][0]
    assert "sampleB reported no row under it" in why
    assert "1 sample(s) to compare" in why
