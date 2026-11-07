"""Study intake: read the files first, ask about what is left, block on four things.

The step that populates ``Conditions`` (ARCHITECTURE §6.2). Every applicability check
in the system reads that object, so this is the one place where a guess could enter
the *foundation* rather than an answer -- and the one place where an over-eager
interview costs the biologist their patience before the tool has said anything useful.
§6.4 names that failure mode exactly: *a biologist losing patience in a twenty-question
interview*. So the order is fixed and this module implements it in that order:

1. :func:`derive` -- everything the store already says, with ``Provenance.READ``.
   **Zero questions for anything on disk.**
2. :meth:`Draft.checklist` -- what is known, what is missing and blocking, what is
   optional. The biologist ticks; he is not driven through a script.
3. :meth:`Draft.stated` / :meth:`Draft.confirm` -- the short targeted chat, which
   belongs to ``agent/loop.py``. Nothing here calls a model.

**The partial-state problem, and why ``Conditions`` is not weakened to solve it.**
``Conditions.__post_init__`` raises ``Blocked`` on a blocking unknown, so a
half-filled ``Conditions`` cannot exist -- correct for the finished object, and the
reason every downstream check may read it without a defensive branch. But §6.4's
checklist has to *show* an incomplete study, which means intake needs a shape that
can hold one. That shape is :class:`Draft`: the same six fields, no completeness rule,
and exactly one way out -- :meth:`Draft.resolve`, which constructs the real object or
raises ``Blocked`` naming every gap at once. The validation lives in one place still;
``Draft`` simply does not pretend to be the thing it is on the way to becoming.

**A model guess is not a condition.** ``INFERRED`` means *the model proposed it and
the biologist confirmed it*. An unconfirmed proposal is neither a condition nor a
weaker condition -- it is not in the object at all. :class:`Proposal` is therefore a
separate type with no ``provenance`` attribute and no path into a ``Draft`` except
:meth:`Draft.confirm`, and ``Draft.__post_init__`` rejects anything that is not a
``Condition`` by type. "Probably breast, from the gene names" cannot be stored while
nobody has agreed to it, which is the §6.2 failure this module exists to prevent.

**Blocking means blocking, by an explicit decision.** The alternative -- record
``UNKNOWN`` for tissue and propagate it -- was rejected: tissue is the corpus scoping
key, so a propagated unknown there is a study whose literature is scoped to nothing,
reported as a caveat. :meth:`Draft.declare_unknown` therefore *refuses* a blocking
field. The four declared-unknown fields propagate instead, and they **name
themselves**: the checklist line for a missing ``segmentation`` says
``propagates as "segmentation unknown"``, because that is a sentence a biologist can
act on and "insufficient information" is not.

**Clustering mode is §6.3's fourth blocker and is not a ``Conditions`` field.** It
blocks only when the store holds more than one sample, the sample count lives on the
adapter, and ``PlatformAdapter.clustering_mode`` raises ``UndeclaredClustering``
rather than guessing. Intake catches that exception and turns it into a checklist
line -- the whole point of §6.4 is that a traceback is not a checklist. The declared
mode rides on the ``Draft`` and not on the ``Conditions`` it resolves to, because
``Conditions`` deliberately cannot express it; the caller writes it back to the
manifest, which is where the adapter will read it next time.

**What this module does NOT do.** No LLM call: the chat layer drives this machinery,
not the other way round. No UBERON lookup -- mapping ``"human breast cancer"`` to an
ontology id is inference, and inference arrives here as a :class:`Proposal` the
biologist confirms. No writes: resolving produces an object, and persisting a study
artifact belongs to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

from panoscope.platforms.base import CLUSTERING_MODES
from panoscope.types.conditions import (
    BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, UNKNOWN, Blocked, Condition, Conditions,
    Provenance, gene_frames_admit,
)
from panoscope.types.contracts import PlatformAdapter, UndeclaredClustering
from panoscope.types.measured import FrameRef

FIELDS: Final[tuple[str, ...]] = BLOCKS_THE_STUDY + DECLARED_UNKNOWN_OK
"""The six ``Condition`` fields, from §6.3's two columns. No third list to drift."""

MEASURED_SET: Final[str] = "measured_set"
"""The structural blocker. Not a ``Condition``: it IS the panel file, so it has no
provenance to record and no answer a biologist can type into a chat."""

CLUSTERING: Final[str] = "clustering_mode"
"""§6.3's fourth blocker. Not a ``Conditions`` field; see the module docstring.
Spelled as the manifest key, because the biologist's next action is to declare it
there."""

ITEMS: Final[tuple[str, ...]] = (
    (MEASURED_SET,) + BLOCKS_THE_STUDY + (CLUSTERING,) + DECLARED_UNKNOWN_OK
)
"""Checklist order: structural, blocking, conditional-blocking, optional."""

#: The phrase PlatformAdapter.validate uses when the store declares no
#: measured set. Matched rather than re-derived: the adapter owns which of its
#: frames IS the measured set, and intake must not name panel: itself.
_MEASURED_SET_GAP = "no measured set"

_WHY_BLOCKS: Final[dict[str, str]] = {
    MEASURED_SET: (
        "the panel file IS the capability frame -- with no frame every absence is "
        "NEVER_MEASURED and the panel-absence rule discriminates nothing"
    ),
    "assay": "with no declared assay there is no apparatus to audit the output against",
    "tissue": (
        "the corpus scoping key -- an unknown tissue scopes the literature to no "
        "organ, and a guessed one scopes it to the wrong organ"
    ),
    CLUSTERING: (
        "more than one sample and no declared mode: joint and independent clustering "
        "give the same cluster labels two different meanings"
    ),
}
"""Why each blocker blocks, in the biologist's terms. A blocking line that does not
say what it costs reads as bureaucracy and gets guessed at."""

_HOW: Final[dict[Provenance, str]] = {
    Provenance.READ: "read from the store",
    Provenance.STATED: "stated by the biologist",
    Provenance.INFERRED: "proposed, and confirmed by the biologist",
    Provenance.UNKNOWN: "not known",
}


class Status(StrEnum):
    """Which of §6.4's three checklist columns an item is in."""

    KNOWN = "known"        # on disk or supplied; no question
    BLOCKING = "blocking"  # missing, and §6.3 blocks the study on it
    OPTIONAL = "optional"  # missing, declared-unknown allowed; it propagates


@dataclass(frozen=True, slots=True)
class Proposal:
    """A model's guess, before anyone has agreed to it. **Not a condition.**

    No ``provenance`` field, on purpose: there is no spelling of this object that a
    downstream check could read as known. It becomes a ``Condition`` with
    ``INFERRED`` only through :meth:`Draft.confirm`, which is what makes
    "the model proposed and the biologist confirmed" structural rather than a
    convention somebody has to remember at the call site.
    """

    field: str
    value: str
    because: str
    """What the model offers as grounds. Shown to the biologist at the confirm, so
    he is agreeing to a reason rather than to a word."""

    def confirmed(self) -> Condition:
        """Only ever called from :meth:`Draft.confirm`."""
        return Condition(self.value.strip(), Provenance.INFERRED)


@dataclass(frozen=True, slots=True)
class Item:
    """One checklist line. Rendered, never computed from: the ticks drive the chat."""

    field: str
    status: Status
    value: str | None
    provenance: Provenance
    why: str


@dataclass(frozen=True, slots=True)
class Checklist:
    """§6.4 step 2, as data. ``render`` is a view of it and derives nothing."""

    study: str
    items: tuple[Item, ...]
    unread_because: str = ""
    problems: tuple[str, ...] = ()
    """``PlatformAdapter.validate``, verbatim. It reports gaps the condition fields
    cannot -- a missing detectability frame, an annotation label set that is not
    there -- and intake does not re-derive what the adapter already says."""

    def of(self, status: Status) -> tuple[Item, ...]:
        return tuple(i for i in self.items if i.status is status)

    @property
    def blocking(self) -> tuple[Item, ...]:
        return self.of(Status.BLOCKING)

    def render(self) -> str:
        width = max(len(i.field) for i in self.items)
        out = [f"Study intake -- {self.study}", ""]
        if self.unread_because:
            out += [
                "  ! the adapter did not report on every field, so one shown below "
                "as unknown",
                "    may still be declared in the files. Reason: "
                + self.unread_because,
                "",
            ]

        def section(title: str, items: tuple[Item, ...], tick: str) -> None:
            if not items:
                return
            out.append(title)
            for i in items:
                shown = f"{i.value}  ({_HOW[i.provenance]})" if i.value else i.why
                out.append(f"  [{tick}] {i.field.ljust(width)}  {shown}")
            out.append("")

        section("Known from the files -- nothing to ask:",
                self.of(Status.KNOWN), "x")
        section("Missing and BLOCKING -- the study cannot start without these:",
                self.of(Status.BLOCKING), " ")
        section("Optional -- tick what you can supply; skippable, addable later:",
                self.of(Status.OPTIONAL), " ")
        if self.problems:
            out.append("The store itself reports:")
            out.extend(f"  - {p}" for p in self.problems)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def __str__(self) -> str:  # so a print() of a checklist is the checklist
        return self.render()


@dataclass(frozen=True, slots=True)
class Draft:
    """Conditions under construction: the shape ``Conditions`` refuses to be.

    Every mutator returns a new ``Draft``, so the chat layer can hold the state it
    started from and a half-applied answer cannot exist.
    """

    assay: Condition
    modality: Condition
    tissue: Condition
    method: Condition
    segmentation: Condition
    fixation: Condition
    frames: tuple[FrameRef, ...]
    clustering: Condition
    unread_because: str = ""
    """Which fields the adapter did not report on, and why, or ``""``.

    ``PlatformAdapter.reads`` is a partial read, so a blocking unknown no longer
    costs the fields that *were* readable. What is left is the adapter refusing the
    store outright, or reporting fewer fields than the contract names: those fields
    are shown as unknown *because intake could not read them*, which is a different
    statement from the files being silent, and the checklist must not collapse the
    two."""
    clustering_blocked_because: str = ""
    """The ``UndeclaredClustering`` message, verbatim, or ``""`` when the adapter
    returned a mode. Says why *and* which manifest to open -- the half a bare flag
    throws away. Kept after the biologist answers, because what the store declares is
    unchanged by what he said: it is the note that tells the caller to write it back."""
    study: str = ""
    problems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Every condition slot holds a ``Condition``. A ``Proposal`` does not.

        Frozen dataclasses do not check annotations at runtime, so without this a
        ``replace(draft, tissue=proposal)`` would store an unconfirmed guess that
        every ``.known`` read downstream would then crash on -- or worse, a type with
        a truthy ``value`` that something read positionally. Rejecting by type is what
        makes "a model guess is not a condition" structural.
        """
        for name in FIELDS + ("clustering",):
            got = getattr(self, name)
            if not isinstance(got, Condition):
                raise TypeError(
                    f"{name} is {type(got).__name__}, not a Condition. A proposal is "
                    "not a condition until the biologist confirms it: use "
                    "Draft.confirm(proposal), which is the only path to INFERRED."
                )
        if not isinstance(self.frames, tuple):
            raise TypeError("frames must be a tuple of FrameRef")

    # ---- reading the draft ------------------------------------------------- #
    def condition(self, field: str) -> Condition:
        """The ``Condition`` behind a checklist field. ``CLUSTERING`` is a slot alias."""
        if field == CLUSTERING:
            return self.clustering
        if field not in FIELDS:
            raise ValueError(f"{field!r} is not a condition field; one of {FIELDS}")
        return getattr(self, field)

    def status(self, field: str) -> Status:
        if field == MEASURED_SET:
            # The same test ``Conditions.__post_init__`` applies, which is what
            # ``resolve()`` has to clear.
            if not gene_frames_admit(self.frames):
                return Status.BLOCKING
            # ...and the adapter's own verdict, because a frame that is ABSENT
            # leaves nothing here to measure. With panel.parquet gone there is no
            # ``panel:`` ref at all, so the size test above sees only ``detect:``
            # and passes -- while ``validate`` says "no measured set: the panel IS
            # the capability frame (§6.3)". ``problems`` is that verbatim, and the
            # adapter is the layer that knows which of its frames is the measured
            # set; naming ``panel:`` here would put platform knowledge in intake.
            if any(_MEASURED_SET_GAP in str(p).lower() for p in self.problems):
                return Status.BLOCKING
            return Status.KNOWN
        cond = self.condition(field)
        if cond.known:
            return Status.KNOWN
        if field == CLUSTERING:
            return Status.BLOCKING if self.clustering_blocked_because else Status.OPTIONAL
        return Status.BLOCKING if field in BLOCKS_THE_STUDY else Status.OPTIONAL

    @property
    def missing(self) -> tuple[str, ...]:
        """Every blocking gap, in checklist order. Empty iff :meth:`resolve` succeeds."""
        return tuple(f for f in ITEMS if self.status(f) is Status.BLOCKING)

    @property
    def complete(self) -> bool:
        return not self.missing

    # ---- the three ways a field gets filled --------------------------------- #
    def stated(self, field: str, value: str) -> "Draft":
        """The biologist said it. The ordinary outcome of the targeted chat."""
        return self._set(field, Condition(self._clean(field, value), Provenance.STATED))

    def confirm(self, proposal: Proposal) -> "Draft":
        """The model proposed it and the biologist confirmed: the only route to INFERRED.

        Takes the ``Proposal`` rather than a value and a provenance, so a caller
        cannot write ``INFERRED`` without one in hand.
        """
        if not isinstance(proposal, Proposal):
            raise TypeError("confirm takes a Proposal; INFERRED has no other route")
        self._clean(proposal.field, proposal.value)  # same validation as stated
        return self._set(proposal.field, proposal.confirmed())

    def declare_unknown(self, field: str) -> "Draft":
        """"I genuinely do not know." Allowed for §6.3's right-hand column only.

        Refusing it on a blocker is the decision §6.3 records: an unknown tissue
        propagated as a caveat is a study scoped to no organ, reported as a footnote.
        """
        if self.status(field) is Status.BLOCKING or field in BLOCKS_THE_STUDY:
            raise Blocked(
                f"{field} blocks the study (§6.3) and cannot be declared unknown: "
                f"{_WHY_BLOCKS.get(field, 'it is a blocking condition')}"
            )
        return self._set(field, UNKNOWN)

    def _set(self, field: str, cond: Condition) -> "Draft":
        if field == MEASURED_SET:
            raise ValueError(
                "the measured set is not a chat answer: it IS the panel file (§6.3). "
                "Point the study at a panel the adapter can read."
            )
        if field == CLUSTERING:
            return replace(self, clustering=cond)
        if field not in FIELDS:
            raise ValueError(f"{field!r} is not a condition field; one of {FIELDS}")
        return replace(self, **{field: cond})

    @staticmethod
    def _clean(field: str, value: str) -> str:
        v = str(value).strip()
        if not v:
            raise ValueError(
                f"{field} was given an empty value. Leave it unknown rather than "
                "recording a blank as known."
            )
        if field == CLUSTERING and v.lower() not in CLUSTERING_MODES:
            raise UndeclaredClustering(
                f"clustering mode {value!r} is not one of {', '.join(CLUSTERING_MODES)}. "
                "An unrecognised word is not a declaration; say which of the three."
            )
        return v.lower() if field == CLUSTERING else v

    # ---- the one way out ---------------------------------------------------- #
    def resolve(self) -> Conditions:
        """The finished object, or ``Blocked`` naming **every** gap at once.

        All of them, not the first: §6.4 exists so a study is not fixed one round
        trip at a time. ``Conditions`` would raise on its own two, and would say
        nothing about clustering, which it cannot express.
        """
        gaps = self.missing
        if gaps:
            raise Blocked(
                "intake is not finished -- "
                + "; ".join(f"{f}: {_WHY_BLOCKS.get(f, 'blocking')}" for f in gaps)
                + ". Ask the biologist; do not record a guess as inferred (§6.3)."
            )
        return Conditions(
            assay=self.assay, modality=self.modality, tissue=self.tissue,
            method=self.method, segmentation=self.segmentation,
            fixation=self.fixation, frames=self.frames,
        )

    # ---- §6.4 step 2 --------------------------------------------------------- #
    def checklist(self) -> Checklist:
        """The ticked clustering mode drops out of ``problems``.

        ``validate`` is a snapshot of the store and the store has not changed, but a
        line saying "does not declare how they were clustered" printed under the mode
        the biologist just supplied reads as intake not having listened. It is still
        true of the manifest, which is why :meth:`resolve`'s caller writes it back.
        """
        answered = self.clustering_blocked_because if self.clustering.known else ""
        return Checklist(
            study=self.study or "(unnamed study)",
            items=tuple(self._item(f) for f in ITEMS),
            unread_because=self.unread_because,
            problems=tuple(p for p in self.problems if p != answered),
        )

    def _item(self, field: str) -> Item:
        status = self.status(field)
        if field == MEASURED_SET:
            value = ", ".join(f"{f.name} ({f.size})" for f in self.frames) or None
            prov = Provenance.READ if self.frames else Provenance.UNKNOWN
        else:
            cond = self.condition(field)
            value, prov = cond.value, cond.provenance
        if status is Status.BLOCKING:
            why = (self.clustering_blocked_because if field == CLUSTERING
                   and self.clustering_blocked_because else _WHY_BLOCKS[field])
        elif status is Status.OPTIONAL:
            why = f'propagates as "{field} unknown"'
        else:
            why = _HOW[prov]
        return Item(field=field, status=status, value=value, provenance=prov, why=why)



def derive(adapter: PlatformAdapter, store: Any) -> Draft:
    """§6.4 step 1: everything the files already say. **Asks nothing.**

    Through ``PlatformAdapter.reads``, the partial read: a store that declares its
    assay and not its tissue hands over the assay, and the missing tissue is one
    blocking row on the checklist rather than six unknown ones. Intake never opens the
    manifest behind the adapter's back -- which key carries the assay is adapter
    knowledge, and a second reader of it is a second thing to keep right and a
    guaranteed liar for the next platform.

    :attr:`Draft.unread_because` survives for the two cases that still degrade: an
    adapter that refuses the store outright, and one that reports fewer fields than
    the contract names. Both show every unreported field as unknown, and *could not
    read* is a different statement from *the files are silent*.
    """
    frames = tuple(f.ref() for f in adapter.frames(store))
    unread = ""
    try:
        got = dict(adapter.reads(store))
    except Blocked as exc:
        got, unread = {}, str(exc)
    silent = tuple(f for f in FIELDS if f not in got)
    if silent and not unread:
        unread = (f"the platform adapter reported no read for {', '.join(silent)}; "
                  "shown as unknown because intake could not read it, which is not "
                  "the same as the files being silent")
    read = {f: got.get(f, UNKNOWN) for f in FIELDS}

    try:
        clustering = Condition(str(adapter.clustering_mode(store)), Provenance.READ)
        because = ""
    except UndeclaredClustering as exc:
        clustering, because = UNKNOWN, str(exc)

    return Draft(
        frames=frames, clustering=clustering, clustering_blocked_because=because,
        unread_because=unread,
        study=str(getattr(store, "id", "") or ""),
        problems=tuple(adapter.validate(store)), **read,
    )
