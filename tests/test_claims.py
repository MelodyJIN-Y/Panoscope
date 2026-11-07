"""Invariants for the claim store and ARCHITECTURE §7.5's split.

As in ``test_types_claim.py``, the tests are the deliverable. Four are the point of
the package:

* **scope is fail-closed.** Retrieval filters and says why; *use* raises. A claim
  applied outside its scope is an error, not a caveat.
* **claims are never merged.** Two claims about one subject from different
  authorities both survive, ordered, with the contradiction intact.
* **the split behaves differently by class.** A database-curated claim gets a
  provenance stamp and no verdict; a lab claim gets named condition deltas. §7.4
  measured the single-check version at 0.000 bits, and this is what replaced it.
* **detectability varies where the retracted rules did not.** An on-panel gene with
  no usable signal is a third outcome, distinguishable from panel-absence.

**The canonical-marker seed is synthetic here.** ``data/claims/canonical_markers_seed.json``
is produced by a separate live-lookup pass and was not on disk when these were
written, so :func:`_seed` builds a fixture against the documented schema. Its PMIDs
are placeholder digit strings and are deliberately NOT real identifiers -- a test
that hardcoded a real one would be a fabricated citation with a green tick next to
it, which is the failure mode the whole project is pointed at. Nothing here asserts
on the real file's contents.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import sys

import pytest

from panoscope.claims.evaluability import Evaluable, evaluability, rests_on
from panoscope.claims.store import (
    SEED_RELPATH, Assessment, BadSeed, ClaimStore, NotCommittable, OutOfScope,
    Retracted, assess, from_dict, load_canonical_markers, load_dir,
    thin_literature, to_dict, write_claim,
)
from panoscope.claims.transfer import (
    BLOCKED, CANNOT_BE_SETTLED, MISMATCH, RETURN, WITH_CONDITIONS,
    manifest_sourced, transfer_check,
)
from panoscope.types.claim import Authority, Claim
from panoscope.types.conditions import DISEASE_STATE, UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.measured import Frame, MeasuredWorld
from panoscope.types.subject import EntityKind, cell_type, gene, subject
from panoscope.types.trace import RULES

# --------------------------------------------------------------------------- #
# The world: a panel, a detectability frame, an annotation label set.
#
# POSTN is the interesting one -- on the panel and refused by the detectability
# frame. That is §7.5's "49 of 280 panel genes yield no usable spatial signal",
# and it is the gate the retracted seven rules were blind to.
# --------------------------------------------------------------------------- #
PANEL = Frame(EntityKind.GENE, "panel:demo_v1", frozenset({"ERBB2", "POSTN", "LUM"}))
DETECT = Frame(EntityKind.GENE, "detect:nosig", frozenset({"ERBB2", "LUM"}))
LABELS = Frame(EntityKind.CELL_TYPE, "annotation:leiden_r0.8",
               frozenset({"Tumor", "Stromal"}))
WORLD = MeasuredWorld(frames=(PANEL, DETECT, LABELS))

ERBB2 = subject("gene", g=gene("ERBB2"))
POSTN = subject("gene", g=gene("POSTN"))
FAP = subject("gene", g=gene("FAP"))          # off panel entirely


def _conditions(**over) -> Conditions:
    base = dict(
        assay=Condition("xenium", Provenance.READ),
        modality=Condition("in_situ", Provenance.READ),
        tissue=Condition("UBERON:0000310", Provenance.STATED),
        method=Condition("jazzpanda", Provenance.READ),
        segmentation=Condition("v2", Provenance.READ),
        fixation=Condition("ffpe", Provenance.STATED),
        frames=(PANEL.ref(), DETECT.ref(), LABELS.ref()),
        disease_state=Condition("breast carcinoma", Provenance.STATED),
        disease_subtype=Condition("HER2+", Provenance.STATED),
        disease_treatment=Condition("naive", Provenance.STATED),
        disease_stage=Condition("II", Provenance.STATED),
    )
    return Conditions(**{**base, **over})


HERE = _conditions()


def _curated(**over) -> Claim:
    """A database-curated edge: lab-scoped, no manifest behind it."""
    base = dict(
        id="curated:marks:ERBB2:Tumor", subject=ERBB2, predicate="marks",
        object="Tumor", authority=Authority.CURATED, source="pmid:00000001",
        depends_on=(cell_type("Tumor"),), scope="lab",
    )
    return Claim(**{**base, **over})


def _lab(**over) -> Claim:
    """A lab claim, which cannot exist without the conditions it was formed under."""
    base = dict(
        id="note-0001", subject=ERBB2, predicate="means_here",
        object="bleeds in dense stroma on this panel", authority=Authority.LAB,
        source="note:0001", conditions=HERE, depends_on=(cell_type("Tumor"),),
        scope="dataset:demo", author="a.biologist",
        created="2026-01-01T00:00:00+00:00",
    )
    return Claim(**{**base, **over})


# --------------------------------------------------------------------------- #
# 1. Scope is fail-closed: retrieval filters WITH A REASON, use RAISES
# --------------------------------------------------------------------------- #
def test_using_a_claim_out_of_scope_raises():
    """CLAUDE.md: "Scope is enforced fail-closed." An error, not a warning.

    A downgrade would mean the sentence still renders, just with a hedge -- and a
    hedged sentence about the wrong tissue block is indistinguishable, in a report,
    from knowledge.
    """
    store = ClaimStore((_lab(),))
    with pytest.raises(OutOfScope):
        store.cite("note-0001", "dataset:other")
    with pytest.raises(OutOfScope):
        assess(_lab(), WORLD, HERE, "dataset:other")


def test_the_scope_gate_runs_before_anything_else():
    """IMPLEMENTATION §5: "scope gate first (a filter, not a downgrade)".

    Pinned with a claim that would ALSO fail evaluability. If the gates ran in the
    other order the caller would be told the gene is off-panel, which is true and
    beside the point: the claim was never applicable here at all.
    """
    off_and_out = _lab(id="note-0002", subject=FAP, scope="dataset:other")
    with pytest.raises(OutOfScope):
        assess(off_and_out, WORLD, HERE, "dataset:demo")


def test_retrieval_filters_but_never_silently():
    """§10.3: "silent dropping is the failure mode, not a safe default."

    Without the decision, "the lab has said nothing about ERBB2" and "the lab's
    judgement about ERBB2 belongs to another dataset" are the same empty result.
    """
    mine, theirs = _lab(), _lab(id="note-0009", scope="dataset:other")
    got = ClaimStore((mine, theirs)).about(ERBB2.key, "dataset:demo")
    assert tuple(c.id for c in got.value) == ("note-0001",)
    withheld = [d for d in got.decisions if d.rule == "claims:out_of_scope"]
    assert len(withheld) == 1
    assert "note-0009" in withheld[0].refs[0]
    assert "dataset:other" in withheld[0].because[0]


def test_a_retracted_claim_is_shown_but_cannot_be_cited():
    """The lab changing its mind is part of the record; citing the old call is not."""
    old = _lab(id="note-0003", valid="superseded", superseded_by="note-0004")
    store = ClaimStore((old,))
    assert store.about(ERBB2.key, "dataset:demo").value == (old,)
    with pytest.raises(Retracted):
        store.cite("note-0003", "dataset:demo")


# --------------------------------------------------------------------------- #
# 2. Never merged
# --------------------------------------------------------------------------- #
def test_two_authorities_about_one_subject_both_survive_ordered():
    """§10.2: conflicts are grouped and displayed, never resolved.

    The curated edge and the lab's contradiction of it both come back, measurement
    first. A store that returned one of them would have resolved a disagreement that
    only the biologist may resolve -- and the disagreement is the product.
    """
    store = ClaimStore((_curated(), _lab()))
    got = store.about(ERBB2.key, "dataset:demo").value
    assert [c.authority for c in got] == [Authority.LAB, Authority.CURATED]
    assert len(got) == 2


def test_ordering_is_stable_within_an_authority():
    """No recency, no score. The store's order is what the biologist sees."""
    a = _curated(id="curated:a", source="pmid:00000001")
    b = _curated(id="curated:b", source="pmid:00000002")
    assert [c.id for c in ClaimStore((a, b)).about(ERBB2.key, "study").value] \
        == ["curated:a", "curated:b"]
    assert [c.id for c in ClaimStore((b, a)).about(ERBB2.key, "study").value] \
        == ["curated:b", "curated:a"]


def test_no_claim_field_can_hold_a_number():
    """"Reference data may never supply a value" -- structurally, not by convention.

    There is no numeric field on a Claim, so a curated edge has nothing to supply.
    The day someone adds ``Claim.score`` or ``Claim.weight``, this fails, and that is
    the day reference data could start corroborating a measurement.
    """
    numeric = [f.name for f in dataclasses.fields(Claim)
               if any(x in str(f.type) for x in ("int", "float"))]
    assert numeric == [], f"Claim now carries a number: {numeric}"


# --------------------------------------------------------------------------- #
# 3. Evaluability: the capability half, and detectability
# --------------------------------------------------------------------------- #
def test_an_in_frame_claim_is_evaluable():
    got = evaluability(_curated(), WORLD)
    assert got.value.state is Evaluable.EVALUABLE
    assert got.value.usable and got.value.blocked_by == ()
    assert [d.rule for d in got.decisions] == ["claims:evaluable"]


def test_panel_absence_and_undetectability_are_different_answers():
    """§7.5's addition, and the reason it is worth adding.

    FAP was never on the instrument; POSTN was, and produced nothing usable. Both
    block the claim, and a reader must not be shown the same sentence for them --
    one says "buy a different panel", the other says "this gene failed here".
    """
    off = evaluability(_curated(id="c:off", subject=FAP), WORLD).value
    dead = evaluability(_curated(id="c:dead", subject=POSTN), WORLD).value
    assert off.state is Evaluable.NEVER_MEASURED
    assert off.frame == "panel:demo_v1"
    assert dead.state is Evaluable.UNDETECTABLE
    assert dead.frame == "detect:nosig"
    assert not off.usable and not dead.usable


def test_a_claim_about_an_unassigned_cell_type_is_blocked_by_the_label_frame():
    """``depends_on`` is gated against ITS OWN kind's frame, not the panel.

    A marks-edge for a label this study never assigned is not evaluable here, and
    checking the cell type against the gene panel would have called it fine.
    """
    got = evaluability(
        _curated(id="c:ct", depends_on=(cell_type("Mast_Cells"),)), WORLD).value
    assert got.state is Evaluable.NEVER_MEASURED
    assert got.frame == "annotation:leiden_r0.8"
    assert [str(e) for e in got.blocked_by] == ["cell_type:Mast_Cells"]


def test_no_frame_is_not_the_same_as_absent():
    """A study that declared no cell-type frame has not *refused* the cell type."""
    got = evaluability(_curated(), MeasuredWorld(frames=(PANEL,)))
    assert got.value.state is Evaluable.NO_FRAME
    assert [d.rule for d in got.decisions] == ["claims:no_frame"]


def test_rests_on_is_subject_plus_depends_on_and_not_the_object():
    """``object`` is a bare string whose entity kind the claim does not carry.

    Including it would mean guessing a kind -- which is how the cell-type label
    "Tumor" ends up checked against the gene panel and reported as off-panel.
    """
    assert [str(e) for e in rests_on(_curated())] == ["gene:ERBB2", "cell_type:Tumor"]


def test_evaluability_applies_to_every_authority_class():
    """§7.5: the capability gate "applies to every claim in every class"."""
    states = {
        a: evaluability(c, WORLD).value.state
        for a, c in (
            (Authority.CURATED, _curated(id="c:1", subject=FAP)),
            (Authority.LAB, _lab(id="n:1", subject=FAP)),
            (Authority.LITERATURE,
             Claim(id="l:1", subject=FAP, predicate="marks", object="Stromal",
                   authority=Authority.LITERATURE, source="pmid:00000003",
                   depends_on=(cell_type("Stromal"),), scope="study")),
            (Authority.MEASUREMENT,
             Claim(id="m:1", subject=FAP, predicate="marks", object="Stromal",
                   authority=Authority.MEASUREMENT, source="file:sha256:" + "0" * 64,
                   depends_on=(cell_type("Stromal"),), scope="study")),
        )
    }
    assert set(states.values()) == {Evaluable.NEVER_MEASURED}


# --------------------------------------------------------------------------- #
# 4. Transfer: the verdict-bearing half, and who never gets a verdict
# --------------------------------------------------------------------------- #
def test_no_verdict_for_database_curated():
    """§7.5, and the direct consequence of §7.4's 0.000 bits.

    A stamp, not a verdict, and ``checked`` is what tells the two apart -- an empty
    delta tuple alone would read as "checked, matched everywhere", which is the same
    collapse §7.4 caught.
    """
    got = transfer_check(_curated(), HERE)
    assert not manifest_sourced(_curated())
    assert got.value.action == RETURN
    assert got.value.deltas == ()
    assert got.value.stamp and not got.value.checked
    assert [d.rule for d in got.decisions] == ["transfer:no_verdict"]


def test_no_verdict_for_literature_either():
    lit = Claim(id="l:2", subject=ERBB2, predicate="marks", object="Tumor",
                authority=Authority.LITERATURE, source="pmid:00000004", scope="study")
    assert not manifest_sourced(lit)
    assert transfer_check(lit, HERE).value.stamp


def test_a_lab_claim_gets_named_deltas_not_a_label():
    """The half that actually varies: §7.4 measured 1.561 bits on lab notes."""
    formed = _conditions(tissue=Condition("UBERON:0002113", Provenance.STATED),
                         fixation=Condition("fresh_frozen", Provenance.STATED))
    got = transfer_check(_lab(conditions=formed), HERE)
    assert got.value.checked
    named = {d.field: d.severity for d in got.value.deltas}
    assert named == {"tissue": MISMATCH, "fixation": MISMATCH}
    assert got.value.action == BLOCKED      # tissue is a §6.3 blocker
    assert {d.rule for d in got.decisions} == {"transfer:delta"}


def test_the_split_behaves_differently_on_curated_and_lab():
    """The whole point of §7.5, in one assertion.

    Same subject, same context. The curated claim is stamped; the lab claim is
    adjudicated by named deltas. Before the split both went through one function
    that returned ``conditional`` for every row of both sets.
    """
    formed = _conditions(tissue=Condition("UBERON:0002113", Provenance.STATED))
    cur = transfer_check(_curated(), HERE).value
    lab = transfer_check(_lab(conditions=formed), HERE).value
    assert (cur.stamp, cur.deltas) != (lab.stamp, lab.deltas)
    assert cur.stamp and not cur.deltas
    assert lab.deltas and not lab.stamp
    # ...while the CAPABILITY gate treats them identically, because it is a fact
    # about the instrument and not about who is speaking.
    assert evaluability(_curated(), WORLD).value.state \
        == evaluability(_lab(conditions=formed), WORLD).value.state


def test_a_dataset_scoped_curated_set_does_get_a_verdict():
    """§7.5 names "dataset-scoped curated sets such as a vendor panel annotation".

    Proves the discriminator is the SCOPE, not the authority label: the same curated
    claim scoped to a dataset is manifest-sourced and is really checked.
    """
    vendor = _curated(id="curated:panel:ERBB2", scope="dataset:demo",
                      source="panel:demo_v1",
                      conditions=_conditions(
                          tissue=Condition("UBERON:0002113", Provenance.STATED)))
    assert manifest_sourced(vendor)
    got = transfer_check(vendor, HERE).value
    assert got.checked and got.stamp == ""
    assert [d.field for d in got.deltas] == ["tissue"]


def test_unknown_condition_never_silently_applies():
    """§5: an ``UNKNOWN`` field produces a *cannot be settled* delta that NAMES it.

    "segmentation unknown" is something the biologist can go and fix; treating it as
    a match is the silent half of the panel-absence failure, moved to conditions.
    """
    formed = _conditions(segmentation=UNKNOWN)
    got = transfer_check(_lab(conditions=formed), HERE)
    named = {d.field: d.severity for d in got.value.deltas}
    assert named == {"segmentation": CANNOT_BE_SETTLED}
    assert got.value.action == WITH_CONDITIONS     # unsettled is not a mismatch
    assert [d.rule for d in got.decisions] == ["transfer:cannot_be_settled"]


def test_blocked_is_surfaced_with_its_reason():
    """§10.3. A blocked claim still comes back, carrying what blocked it."""
    formed = _conditions(assay=Condition("cosmx", Provenance.READ))
    got = transfer_check(_lab(conditions=formed), HERE).value
    assert got.action == BLOCKED
    assert str(got.deltas[0]) == "assay: cosmx -> xenium (mismatch)"


def test_a_matching_lab_claim_transfers_cleanly():
    got = transfer_check(_lab(), HERE)
    assert (got.value.action, got.value.deltas) == (RETURN, ())
    assert [d.rule for d in got.decisions] == ["transfer:transferable"]


def test_transfer_never_compares_frame_digests():
    """Deliberate omission: it would fire on nearly everything and be the §7.4
    constant again. Capability is answered per entity by ``evaluability``."""
    other = _conditions(frames=(Frame(EntityKind.GENE, "panel:other",
                                      frozenset({"ERBB2"})).ref(),))
    assert transfer_check(_lab(conditions=other), HERE).value.deltas == ()


# --------------------------------------------------------------------------- #
# 5. The ordered assessment
# --------------------------------------------------------------------------- #
def test_assess_runs_both_halves_in_order():
    got = assess(_lab(), WORLD, HERE, "dataset:demo")
    assert isinstance(got.value, Assessment)
    assert [d.rule for d in got.decisions] == ["claims:evaluable",
                                               "transfer:transferable"]
    assert got.value.usable


def test_capability_outranks_a_clean_transfer():
    """§7.6: the measurement adjudicates the claim, never the claim's own conditions.

    Every condition matches and the gene was never measured here. Usable must be
    False, or formation conditions have been allowed to vouch for a measurement.
    """
    got = assess(_lab(id="note-0005", subject=FAP), WORLD, HERE, "dataset:demo").value
    assert got.transfer.action == RETURN
    assert got.evaluable.state is Evaluable.NEVER_MEASURED
    assert not got.usable


# --------------------------------------------------------------------------- #
# 6. §7.4 made checkable: the constant, and the gate that is not constant
# --------------------------------------------------------------------------- #
def test_the_curated_verdict_is_constant_and_detectability_is_not():
    """§7.4's finding, as a runnable check rather than an assertion in prose.

    Over a set of database-curated claims the transfer outcome is the same every
    time -- which is exactly why it is no longer computed as a verdict. Over the
    same claims the capability gate returns three distinct states, because the panel
    and the detectability frame disagree about different genes. That is §7.5's
    "unlike the seven original rules, it actually varies".

    Numbers are not asserted: §7.4's 0.000 and 0.652 bits were measured over 280
    real edges and this fixture is three. The *shape* is what is pinned.
    """
    claims = (_curated(id="c:a", subject=ERBB2),
              _curated(id="c:b", subject=POSTN),
              _curated(id="c:c", subject=FAP))
    verdicts = {(transfer_check(c, HERE).value.action,
                 transfer_check(c, HERE).value.deltas) for c in claims}
    capability = {evaluability(c, WORLD).value.state for c in claims}
    assert len(verdicts) == 1, "the curated transfer verdict is supposed to be constant"
    assert len(capability) == 3, capability


# --------------------------------------------------------------------------- #
# 7. The canonical-marker seed -> CURATED claims
# --------------------------------------------------------------------------- #
def _seed(tmp_path, pairs, name="canonical_markers_seed.json"):
    """A SYNTHETIC seed file. Placeholder PMIDs, never real identifiers."""
    p = tmp_path / name
    p.write_text(json.dumps({"pairs": pairs, "summary": {"n": len(pairs)}}),
                 encoding="utf-8")
    return p


SUPPORTED = {"cell_type": "Tumor", "gene": "ERBB2", "off_panel": False,
             "pmid": "00000001", "title": "SYNTHETIC - not a real record",
             "year": 2000, "journal": "SYNTHETIC", "query": "ERBB2 Tumor marker",
             "support": "supported", "note": ""}
WEAK = {**SUPPORTED, "gene": "LUM", "cell_type": "Stromal", "pmid": "00000002",
        "support": "weak"}
UNCITED = {"cell_type": "Stromal", "gene": "FAP", "off_panel": True, "pmid": None,
           "title": "", "year": 0, "journal": "", "query": "FAP Stromal marker",
           "support": "none", "note": "no resolving record found"}


def test_a_cited_pair_carries_its_citation(tmp_path):
    (c,) = load_canonical_markers(_seed(tmp_path, [SUPPORTED])).claims
    assert c.id == "curated:marks:ERBB2:Tumor"
    assert (c.authority, c.predicate, c.object) == (Authority.CURATED, "marks", "Tumor")
    assert c.source == "pmid:00000001"
    assert c.status == "firm"
    assert not thin_literature(c)
    assert c.depends_on == (cell_type("Tumor"),)


def test_a_weak_pair_is_tentative(tmp_path):
    """``status`` kept from the note store (§10): how firmly it was said."""
    (c,) = load_canonical_markers(_seed(tmp_path, [WEAK])).claims
    assert (c.status, c.source) == ("tentative", "pmid:00000002")
    assert not thin_literature(c)


def test_an_uncited_pair_is_kept_sourced_to_the_file_and_reads_as_thin(tmp_path):
    """The design decision this module had to make, pinned.

    ``support: "none"`` is neither dropped (which silently changes the canonical set
    and therefore the band, with no record) nor given a borrowed PMID. It is sourced
    to the bytes that actually assert it, content-addressed, marked tentative, and
    ``thin_literature`` is True -- "say the literature is thin; never invent a ref".
    """
    path = _seed(tmp_path, [UNCITED])
    (c,) = load_canonical_markers(path).claims
    assert c.source.startswith("file:sha256:")
    assert c.status == "tentative"
    assert thin_literature(c)
    # The digest resolves to the exact bytes read, so the source is CHECKABLE.
    assert c.source == "file:sha256:" + hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def test_offpanel_never_penalises(tmp_path):
    """IMPLEMENTATION §5 names this test. The claim does not know about the panel.

    "canonical whether or not it is on the panel; the frame decides measurability,
    the claim does not." Flipping ``off_panel`` must change nothing about the claim
    -- the frame is consulted later, by ``evaluability``, against the study at hand.
    """
    on = load_canonical_markers(
        _seed(tmp_path, [{**SUPPORTED, "off_panel": False}], "on.json"))
    off = load_canonical_markers(
        _seed(tmp_path, [{**SUPPORTED, "off_panel": True}], "off.json"))
    assert on.claims == off.claims


def test_a_seed_pair_claiming_no_support_but_carrying_a_pmid_is_refused(tmp_path):
    """The exact shape of a fabricated citation. Neither half is guessed at."""
    with pytest.raises(BadSeed, match="support='none' but carries"):
        load_canonical_markers(_seed(tmp_path, [{**UNCITED, "pmid": "00000005"}]))


def test_a_seed_pair_with_an_unknown_support_value_is_refused(tmp_path):
    with pytest.raises(BadSeed, match="support"):
        load_canonical_markers(_seed(tmp_path, [{**SUPPORTED, "support": "maybe"}]))


def test_an_identical_repeated_row_is_one_claim(tmp_path):
    """A repeated ROW is not a repeated CLAIM, and this is not a merge.

    The real seed carries DCN and COL1A1 under Stromal twice, identical but for
    ``off_panel`` -- because ``annotation.py`` lists both genes in its on-panel AND
    its off-panel Stromal map. Since ``off_panel`` is deliberately not part of a
    claim, the two rows produce the same claim and there is nothing to merge.
    """
    twice = _seed(tmp_path, [SUPPORTED, {**SUPPORTED, "off_panel": True}])
    assert len(load_canonical_markers(twice)) == 1


def test_two_different_claims_for_one_pair_are_refused(tmp_path):
    """The failure the id check actually exists for.

    Two citations for one pair are two claims, and two claims may not share an id.
    Dropping one would be resolution by file order, which §10.5 refuses outright.
    """
    with pytest.raises(BadSeed, match="collides"):
        load_canonical_markers(
            _seed(tmp_path, [SUPPORTED, {**SUPPORTED, "pmid": "00000009"}]))


def test_a_truncated_seed_is_refused(tmp_path):
    """``summary.total`` disagreeing with the array means a half-written file, and
    a short canonical set silently changes every confidence band."""
    p = tmp_path / "short.json"
    p.write_text(json.dumps({"pairs": [SUPPORTED], "summary": {"total": 65}}),
                 encoding="utf-8")
    with pytest.raises(BadSeed, match="summary says"):
        load_canonical_markers(p)


def test_a_missing_seed_is_an_error_not_an_empty_set(tmp_path):
    """An empty canonical set would silently change every confidence band."""
    with pytest.raises(BadSeed, match="never invents one"):
        load_canonical_markers(tmp_path / "nope.json")


def test_seed_claims_are_lab_scoped_and_get_a_stamp(tmp_path):
    """Canonical markers are not about any one dataset, so no transfer verdict."""
    (c,) = load_canonical_markers(_seed(tmp_path, [SUPPORTED])).claims
    assert c.scope == "lab" and c.applies_in("dataset:anything")
    assert not manifest_sourced(c)
    assert transfer_check(c, HERE).value.stamp


def test_the_loader_is_deterministic(tmp_path):
    """Same bytes, same claims, same order -- the ids are hashed into the record."""
    p = _seed(tmp_path, [SUPPORTED, WEAK, UNCITED])
    assert load_canonical_markers(p).claims == load_canonical_markers(p).claims


REAL_SEED = (pathlib.Path(__file__).resolve().parents[1]
             / "data" / SEED_RELPATH)


@pytest.mark.skipif(not REAL_SEED.is_file(),
                    reason="the canonical-marker seed is produced by a separate "
                           "live-lookup pass and is not on disk")
def test_the_real_seed_loads_and_every_claim_carries_a_source():
    """The shipped file, against the loader. No PMID is asserted here.

    Checking a specific identifier would mean writing one into the test suite from
    something other than a live lookup, which is the failure this whole file is
    pointed at. What is pinned is structural: every claim has a resolvable source,
    the uncited ones say so, and nothing was invented to fill a gap.
    """
    store = load_canonical_markers(REAL_SEED)
    assert len(store) > 50
    for c in store:
        assert c.authority is Authority.CURATED and c.predicate == "marks"
        assert c.source.partition(":")[0] in ("pmid", "doi", "file")
        # The two halves of the honesty rule.
        assert thin_literature(c) is c.source.startswith("file:")
        if c.status == "firm":
            assert c.source.startswith("pmid:"), f"{c.id} is firm with no citation"
    assert any(thin_literature(c) for c in store), \
        "a seed with no uncited pair means the honest-absence path is untested"
    assert any(not thin_literature(c) for c in store)


# --------------------------------------------------------------------------- #
# 8. The write path and round-trip
# --------------------------------------------------------------------------- #
def test_only_the_biologist_commits(tmp_path):
    """§10.5: two writers, and only one of them uses this path.

    A curated or measurement claim arriving down the biologist's commit path is
    knowledge with the wrong provenance stamped on it -- "if a lab claim renders as
    a curated fact, or a curated edge as evidence, the design collapses".
    """
    with pytest.raises(NotCommittable):
        write_claim(_curated(), tmp_path)
    assert write_claim(_lab(), tmp_path).name == "note-0001.json"


def test_a_written_claim_round_trips(tmp_path):
    original = _lab(tension=("curated:marks:ERBB2:Tumor",), spans=("s1",))
    write_claim(original, tmp_path)
    assert load_dir(tmp_path).claims == (original,)


def test_an_undeclared_disease_state_round_trips_without_new_keys():
    """Stores written before disease state existed must stay byte-stable."""
    c = _lab(conditions=_conditions(**{f: UNKNOWN for f in DISEASE_STATE}))
    d = to_dict(c)
    assert not set(DISEASE_STATE) & set(d["conditions"])
    assert from_dict(d) == c


def test_round_trip_preserves_the_join_key():
    """``Subject.key`` is what reconciliation joins on across task kinds."""
    c = _lab()
    assert from_dict(to_dict(c)).subject.key == c.subject.key


def test_a_hand_edited_claim_file_is_validated_on_read(tmp_path):
    """Git-tracked JSON means hand edits, which is where a bad scope enters."""
    write_claim(_lab(), tmp_path)
    path = tmp_path / "note-0001.json"
    d = json.loads(path.read_text())
    d["scope"] = "sample:s1:oops"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_dir(tmp_path)


def test_a_claim_file_with_no_scope_key_is_refused_not_widened(tmp_path):
    """A MISSING scope is the dangerous one, because nothing about the file looks wrong.

    ``"study"`` is the unkeyed rung, so a file that merely forgot its scope key would
    have applied in EVERY study -- the silent widening ``memory._note_from_dict`` was
    fixed for one module over. Fail closed, loudly, naming the claim.
    """
    write_claim(_lab(), tmp_path)
    path = tmp_path / "note-0001.json"
    d = json.loads(path.read_text())
    del d["scope"]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_dir(tmp_path)


def test_load_dir_is_empty_not_an_error_for_a_new_lab(tmp_path):
    assert load_dir(tmp_path / "never_created").claims == ()


def test_thin_literature_only_speaks_about_knowledge_claims():
    """A lab claim does not become thin for want of a PMID, and a measurement
    sourced to a file is not literature at all. Calling either thin would turn an
    honest signal into noise the interface learns to ignore."""
    assert not thin_literature(_lab())
    assert not thin_literature(
        Claim(id="m:2", subject=ERBB2, predicate="marks", object="Tumor",
              authority=Authority.MEASUREMENT, source="file:sha256:" + "0" * 64, scope="study"))
    assert thin_literature(_curated(source="panel:demo_v1"))


# --------------------------------------------------------------------------- #
# 9. Module hygiene
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["panoscope.claims.evaluability",
                                 "panoscope.claims.transfer"])
def test_the_decision_halves_are_pure(mod):
    """``test_transfer_pure`` (§5): both halves run at retrieval time, on every
    claim, in a process whose output is hashed. Neither may touch a disk or a clock."""
    path = sys.modules[mod].__file__
    assert path is not None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for forbidden in ("import time", "datetime", "import random", "open(",
                      "Path(", "json."):
        assert forbidden not in text, f"{forbidden!r} in {mod}"


def test_every_rule_this_package_emits_is_registered():
    """Belt to ``test_contracts``'s braces, naming the ids rather than scanning."""
    for rid in ("claims:out_of_scope", "claims:evaluable", "claims:never_measured",
                "claims:undetectable", "claims:no_frame", "transfer:no_verdict",
                "transfer:transferable", "transfer:delta",
                "transfer:cannot_be_settled"):
        assert rid in RULES, rid


def test_no_mutable_fields_in_the_claims_package():
    """dict/list/set fields break byte-determinism and the frozen contract."""
    from panoscope.claims import evaluability as E
    from panoscope.claims import store as S
    from panoscope.claims import transfer as T

    bad = []
    for module in (E, T, S):
        for name in dir(module):
            obj = getattr(module, name)
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            for f in dataclasses.fields(obj):
                t = str(f.type)
                if any(x in t for x in ("dict[", "list[", "set[")) \
                        and "frozenset" not in t:
                    bad.append(f"{module.__name__}.{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable contract fields: " + "; ".join(bad)
