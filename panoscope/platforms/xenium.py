"""The Xenium adapter: what the 10x Xenium store was CAPABLE of seeing, and what it
says about itself.

Every member here is a read. None computes a statistic -- the adapter declares the
apparatus, the method declares what it found, and that split is what makes the
confidence assessment an audit rather than a self-assessment.

**Two gene frames, not one.** ``measured.py`` says there is one absence primitive per
entity kind and that several frames may share a kind; this is the first place that
matters in practice. The panel says which genes the assay *could* report.
Detectability says which of those produced usable spatial signal at all -- 49 of the
demo's 280 (17.5%) did not. They are different facts with different consequences: a
gene off the panel was never probed, a gene on the panel with no signal was probed
and produced nothing usable, neither absence argues anything, and only the second is
a property of this tissue section rather than of the panel design. ``_state_for``
requires *every* frame of a kind to admit an entity, so declaring both is what makes
an undetectable gene resolve to ``NEVER_MEASURED`` instead of quietly reading as a
tested absence that argues against a rival. Folding detectability into a footnote --
which is what the codebase did before -- is the four-states-into-a-boolean mistake,
one layer down.

**The panel size is derived and never asserted.** ``manifest.json`` carries
``panel_gene_count``, and this module ignores it in favour of counting the rows of
``panel.parquet``. A count in a manifest is a claim about a file; the file is the
measured set. The second study's immuno-oncology panel is a different size, and a
constant 280 would turn it into 280 genes that are not there -- while a stale
manifest count would do the same thing with a plausible number on it.

**The clustering blocker is real here.** ``clustering_mode`` delegates to
``base.declared_clustering``, which raises ``UndeclaredClustering`` when the store
holds more than one sample and declares nothing. The demo is one sample, so it
returns without raising; the kidney study is not, so it will not.

**What this adapter refuses to infer.** ``tissue`` arrives as the manifest's prose
("human breast cancer") even though ``Conditions.tissue`` wants an ontology id,
because mapping prose to UBERON is inference and
``PlatformAdapter.standing_conditions`` may only write ``READ``. ``fixation`` is FFPE
in every description of this dataset and is still ``UNKNOWN`` here, because no file in
the store says so and a condition that merely sounds right is exactly what
``Provenance`` exists to stop. Intake fills both, with a provenance saying who filled
them in.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import pandas as pd

from panoscope.data.study import Study
from panoscope.platforms.base import (
    ClusteringMode,
    UndeclaredSamples,
    declared_clustering,
)
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import UndeclaredClustering
from panoscope.types.measured import Frame
from panoscope.types.subject import EntityKind

NO_SIGNAL: Final[str] = "NoSig"
"""The unit-column sentinel for a gene that produced no usable spatial signal.

A value in the unit column that is not a unit. It has to be named somewhere, and the
store is where it appears, so the adapter owns it: the tidy reader takes it as
``not_units`` and the detectability frame is its complement. Two readings of one
sentinel, from one declaration.
"""


def is_no_signal(value: object) -> bool:
    """True if a unit-column ``value`` says *no usable spatial signal*, in ANY case.

    One predicate, because four places test this sentinel and they must agree about
    it: this adapter's detectability frame, ``agent/config._derive_cluster_order``
    (which builds the cluster label set as the sentinel's complement),
    ``agent/grounding_check.is_detectable`` and ``agent/verdict._detectable_genes``.
    They tested ``!= "NoSig"`` independently and case-SENSITIVELY, and
    ``markers_top.csv`` is a consumed input whose spelling is whatever the upstream
    jazzPanda run emitted. A table spelling it ``nosig`` therefore put every
    undetectable gene INTO ``detect:<id>``, which makes its absence argue -- the
    panel-absence rule inverted, and inverted hardest on an off-target panel where
    the undetectable set is most of the assay. The Stage 0 guard that exists to
    catch an unknown label shared the same comparison, so it passed the store clean.

    **Blank and NaN count as no signal**, and that is the conservative direction
    rather than a convenience: an empty unit column is not a declaration that the
    gene was detected, and reading it as one both admits the gene to the
    detectability frame and (at ``agent/config.py``) makes the empty string a
    cluster id.
    """
    s = "" if value is None else str(value).strip()
    return not s or s.upper() == NO_SIGNAL.upper() or s.lower() == "nan"


CONTROL_PREFIXES: Final[tuple[str, ...]] = (
    "NEGCONTROLPROBE", "NEGCONTROLCODEWORD", "NEGPRB", "BLANK", "ANTISENSE",
    "UNASSIGNEDCODEWORD", "DEPRECATEDCODEWORD",
)
"""Xenium puts its negative controls in the same table as the panel.

**Prefix FAMILIES, not spellings.** Each entry names a control family and stops
where the family name stops, because the vendor decorates the rest: ``BLANK`` is
``BLANK_0003`` and ``BlankCodeword_0001`` and ``Blank-0002``, ``ANTISENSE`` is
``antisense_ABC`` and ``antisenseXYZ``. The list used to carry the decorated
spellings -- ``"BLANK_"``, ``"Blank-"``, ``"antisense_"`` -- which is the same
mistake as a case-sensitive comparison one level down: it answers the probe the
author happened to have in front of them rather than the family. ``BlankCodeword``
is emitted by *every* Xenium run and matched none of them, so a blank codeword
entered ``panel:<id>`` and ``detect:<id>`` as a gene, and a probe measuring nothing
could be ``OBSERVED`` -- on an off-target panel, where a handful of rows carry any
statistic at all, plausibly the top one, with a ``Source``-backed sentence calling
it on-panel while quoting "negative control" as its annotation.

The families are deliberately the longest string that is still only the family.
``NEG`` is not one: **NEGR1 is a real human gene** and a bare ``NEG`` would delete
it from the assay, which is the same class of error as admitting a control, pointed
the other way. ``BLANK`` and ``ANTISENSE`` are safe in that direction because no
HGNC symbol begins with either -- antisense transcripts are named ``<GENE>-AS1``,
never ``ANTISENSE*``. ``tests/test_panel_controls.py`` pins both directions.

Upper-cased in the source because the comparison is (see ``is_control_probe``);
written out rather than lower-cased at use so the one list reads as what is matched.

This is the platform-specific half of "how measured is derived": on another platform
the rule is different, which is why it lives in an adapter and not in ``measured.py``.
A control probe left in the frame makes the panel look larger than the assay and
makes a control's absence read as a biological absence.

Custom add-on panels are NOT filtered, because nothing in the tidy store marks them.
A study that mixes a base panel with add-ons and needs them separated has to declare
that separation; guessing it from a gene name is how a real gene gets dropped.

Public, and imported by ``agent/data.py`` (as ``NO_SIGNAL`` already is by
``agent/grounding_check.py``), because the frame and the tidy reader have to drop the
same rows or they disagree about the size of the assay. They did disagree: the
adapter filtered and ``load_panel`` did not, so the capability frame said 12 while
``manifest.panel_gene_count``, every "...on this N-gene panel" caveat and every
plot's conditionality line said 13. A second copy of this list is a second answer.
"""


_CONTROL_PREFIXES_UPPER: Final[tuple[str, ...]] = tuple(
    p.upper() for p in CONTROL_PREFIXES
)
"""The list above, upper-cased once. Not a second list -- a second comparison."""


def is_control_probe(name: str) -> bool:
    """True if ``name`` is a Xenium control probe rather than a panel gene, in ANY case.

    The named half of "excluded, not invisible": controls are dropped from the panel
    everywhere (see ``agent/data.load_panel``), and this is what lets a caller say
    *why* a name is off-panel instead of reporting a probe the instrument really did
    run as something that was never measured.

    **Case-insensitive, because every other gene-identity comparison here is.**
    ``panel_contains`` and ``panel_annotation`` upper-case both sides -- a vendor who
    writes ``erbb2`` means ERBB2, and a vendor who writes ``NEGCONTROLPROBE_00001``
    means the same probe. This was the one case-SENSITIVE comparison left, and it was
    the one the capability frame used: on a ``panel.parquet`` whose controls arrive in
    any other case it put every control back into ``panel:<id>``, so the frame said 13
    where the tidy reader said 12 and the two layers that must agree about the size of
    the assay disagreed. ``agent.data.is_control`` delegates here rather than carrying
    its own upper-cased copy, so the frame and the readers share one rule and one
    comparison -- they agree by construction, not because two filters happen to match.
    """
    return str(name).strip().upper().startswith(_CONTROL_PREFIXES_UPPER)


CONTROL_ANNOTATION_TOKENS: Final[tuple[str, ...]] = ("CONTROL", "CODEWORD")
"""Words in ``panel.parquet``'s ``annotation`` column that mark a row as a probe.

**The second half of the control rule, and the half that is not the vendor's to
rename.** :data:`CONTROL_PREFIXES` matches the probe's NAME, and a name is exactly
what a vendor is free to change between panel versions: a negative control shipped
as ``IntergenicRegion_0001`` or ``GenomicControl_02`` matches no family here, enters
``panel:<id>`` as a gene, and can then be ``OBSERVED`` -- which on an off-target
panel, where few rows carry any statistic, is plausibly the top one. Measured on a
kidney-shaped rehearsal: two such probes led ``key_markers`` in ``interp/verdicts.csv``
with ``role: supports`` and ``caveats: []``, read as *"Driven by IntergenicRegion_0001"*
on the Markers page, and were handed to the model with ``glm_coef 89.0`` and a
``jz`` Source -- while the row's own ``annotation`` column said ``negative control``
and ``agent/data.panel_annotation`` already read that column.

So the store already carried the answer and nothing asked it. These two tokens are
the smallest rule that reads it. They are matched as substrings of the upper-cased
annotation, and they are deliberately only two: both are terms this codebase already
treats as vendor control vocabulary (``negative control`` is the annotation on this
repo's own fixture control row; ``CODEWORD`` is already a prefix family above), and
neither is a word a cell-type or tissue annotation uses. Widening this list guesses
at strings no file here contains, and guessing wrong here deletes a real gene from
the assay -- the same error as admitting a probe, pointed the other way.

Name OR annotation, never name AND annotation: an unannotated control still matches
by family, and an un-family-named control still matches by annotation.
"""


def is_control_annotation(annotation: object) -> bool:
    """True if a panel row's ``annotation`` says the row is a control probe."""
    if annotation is None:
        return False
    return any(t in str(annotation).strip().upper()
               for t in CONTROL_ANNOTATION_TOKENS)

_VIEW_ELEMENTS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("cell_map", ("viz/cells.parquet", "viz/cells.csv")),
    ("umap", ("viz/umap.parquet", "viz/umap.csv")),
    ("expr", ("viz/expr.parquet",)),
    ("density", ("viz/hexbin",)),
)
"""Which elements a view needs. A view with no element behind it is a plot that would
invent its data, so it is not offered."""


def _present(store: Study, rel: str) -> bool:
    """True if the element exists; a directory must also be non-empty.

    ``viz/hexbin/`` exists and holds zero frames in ``demo_mini``, and an empty
    directory offering a density view is the same lie as a missing one.
    """
    p = store.dir / rel
    if p.is_dir():
        return any(p.iterdir())
    return p.is_file()


@lru_cache(maxsize=16)
def _annotated_controls_cached(path: str, mtime_ns: int, size: int) -> frozenset[str]:
    """``panel.parquet``'s control rows BY ANNOTATION, memoised on the file itself.

    ``is_control`` is asked once per row of every marker table a build reads, so an
    uncached parquet read here is one file read per gene -- measured at roughly
    three times the suite's runtime before this cache went in.

    Keyed on ``(path, mtime_ns, size)`` rather than on the path alone: a build
    rewrites study inputs in-process, and a cache keyed on the name would answer a
    later study's question with an earlier study's panel. A rewrite changes at least
    one of the two stat fields, so the key changes with the file.
    """
    df = pd.read_parquet(path)
    # Both columns, not just the annotation: a panel spelling its gene column
    # anything else raised a bare KeyError here as well as in ``_panel_genes``,
    # and this one is reached first through ``frames``.
    if "annotation" not in df.columns or "gene" not in df.columns:
        return frozenset()
    return frozenset(
        str(g).strip().upper()
        for g, note in zip(df["gene"], df["annotation"])
        if is_control_annotation(note)
    )


class XeniumAdapter:
    """``PlatformAdapter`` for a 10x Xenium store in the tidy layout.

    Structural conformance only -- no base class, per ``types/contracts.py``. Reads
    at call time and holds no path, no table and no study, so one instance is safe
    to share and a study opened later is never served an earlier study's frames.

    ``sample`` is the one thing an instance carries, and it is a *sample id*, not a
    resolved file: :meth:`for_sample` binds which section's reading this adapter
    speaks for, and the path is still resolved against whichever store a call is
    given. Detectability is the only frame that moves -- see
    :meth:`_detectable_genes` for why it must.
    """

    name = "xenium"

    def __init__(self, sample: str | None = None) -> None:
        self.sample = sample

    def for_sample(self, sample: str) -> "XeniumAdapter":
        """This adapter, reading ``sample``'s own tables rather than the pooled ones.

        A new instance rather than a mutation, because the pooled adapter is shared
        and a replicate loop that re-pointed it would change what every other caller
        in the process means by "detectable".

        Not on the ``PlatformAdapter`` Protocol yet (``types/contracts.py``), which
        is a gap and not a decision: a second platform with per-sample tables needs
        the same method under the same name, and today's one caller --
        ``pipeline/records.py::_concordance`` -- would meet it as an ``AttributeError``
        rather than as a missing declaration.
        """
        return XeniumAdapter(sample)

    # ---- frames ----------------------------------------------------------- #
    def _panel_genes(self, store: Study) -> frozenset[str] | None:
        """``panel.parquet``'s gene column, minus negative controls: the measured set.

        ``None`` when the file is absent, for the same reason ``_detectable_genes``
        returns it: "no panel file" and "a panel file listing nothing but negative
        controls" are different facts about a store, and only the second is the
        apparatus declaring that it measured nothing. Returning a bare
        ``frozenset()`` for both made the second unrepresentable, and ``frames``
        then dropped the ``panel:`` frame and left ``detect:`` governing the gene
        kind alone -- a tested absence against a measured set that is empty.
        """
        p = store.inputs / "panel.parquet"
        if not p.is_file():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001 - an unreadable panel is "no measured set"
            return None
        if "gene" not in df.columns:
            # UNREADABLE, not empty -- the same answer a missing file gets, so the
            # distinction this docstring protects (None = no frame at all,
            # frozenset() = a frame that admits nothing) still holds.
            #
            # This indexed df["gene"] unguarded, so a panel whose column is spelled
            # "Gene" or "gene_name" raised a bare KeyError out of ``frames``. And
            # ``intake.derive`` wraps ``reads`` in try/except Blocked but not
            # ``frames``, so the biologist got a traceback instead of the checklist
            # line written for precisely this case -- which already says "missing
            # or has no usable gene column" four hundred lines below.
            return None
        notes = (df["annotation"] if "annotation" in df.columns
                 else [None] * len(df))
        return frozenset(
            g for g, note in zip((str(v).strip() for v in df["gene"]), notes)
            if g and not is_control_probe(g) and not is_control_annotation(note)
        )

    def is_control(self, store: Study, entity: str) -> bool:
        """``PlatformAdapter.is_control``: the store's own answer about one name.

        The public half of the rule the two gene frames already apply, exposed
        because the frames are not enough on their own. A control row in
        ``markers_top.csv`` is refused by both frames and therefore resolves
        ``NEVER_MEASURED`` -- which is a real state, so the row becomes a real
        subject: it is enumerated as a never-measured entity, projected to v1 as an
        OffPanelNote reading *"X is off-panel (never measured); its absence is not
        evidence against T"* in a container whose contract is *canonical markers
        never measured*, and counted into the §9.10 replicate denominator as part of
        "this unit's profile". None of those sentences is true of a probe, and all
        three are click-backable. ``agent/data.is_control`` is this rule on the v1
        side of the agent/core line; they read the same two columns.
        """
        return (is_control_probe(entity)
                or str(entity).strip().upper() in self._annotated_controls(store))

    def _annotated_controls(self, store: Study) -> frozenset[str]:
        """Upper-cased names the panel file's ANNOTATION marks as control probes.

        The detectability frame is read off ``markers_top.csv``, which carries no
        annotation column, so without this a probe the panel file annotates as a
        control but whose name matches no family stays a member of ``detect:<id>``.
        The lattice would still refuse it -- ``_state_for`` requires every frame of
        a kind to admit an entity and ``panel:<id>`` does not -- but the frame's
        size is reported, and a detectability count inflated by probes is a derived
        number that disagrees with the panel beside it.
        """
        p = store.inputs / "panel.parquet"
        if not p.is_file():
            return frozenset()
        st = p.stat()
        return _annotated_controls_cached(str(p), st.st_mtime_ns, st.st_size)

    def _markers_path(self, store: Study) -> Path:
        """The marker table THIS adapter speaks for: the bound sample's, or pooled.

        Factored out so the detectability frame and the sentence ``validate`` prints
        when that frame comes back empty name the same file. They did not: the
        blocker named ``inputs/markers_top.csv`` on a per-sample adapter, sending a
        biologist to a pooled table full of signal to explain why one section had
        none.
        """
        if self.sample is None:
            return store.inputs / "markers_top.csv"
        p = self.sample_inputs(store).get(self.sample)
        if p is None:
            raise KeyError(
                f"study {store.id!r} has no per-sample marker table for "
                f"{self.sample!r} ({store.inputs / 'samples' / self.sample / 'markers_top.csv'}), "
                "so this sample's detectability cannot be read. Falling back to "
                "the pooled table would hand this section another section's "
                "detectability and report its NoSig genes as tested absences"
            )
        return p

    def _why_detect_empty(self, store: Study) -> str:
        """Why the detectability frame came back empty, COUNTED from the table.

        The sentence this replaces asserted one cause -- "every gene is marked
        NoSig" -- for three different stores, and was false on the one the empty
        frame was built to represent: a table whose every row is a control probe
        has no NoSig row in it at all, and a zero-row table has no gene to mark.
        A blocker is the sentence a biologist acts on, so it says what the file
        actually holds.
        """
        p = self._markers_path(store)
        df = pd.read_csv(p, usecols=["gene", "top_cluster"])
        total = len(df)
        if not total:
            return "it has no rows at all"
        names = [str(g).strip() for g in df["gene"]]
        by_note = self._annotated_controls(store)
        controls = sum(1 for g in names
                       if is_control_probe(g) or g.upper() in by_note)
        nosig = sum(1 for u in df["top_cluster"] if is_no_signal(u))
        parts = []
        if nosig:
            parts.append(f"{nosig} of {total} row(s) carry the no-signal sentinel")
        if controls:
            parts.append(f"{controls} of {total} name a control probe rather than "
                         "a gene")
        blank = sum(1 for g in names if not g)
        if blank:
            parts.append(f"{blank} have no gene name")
        return "; ".join(parts) or f"none of its {total} row(s) names a panel gene"

    def _detectable_genes(self, store: Study) -> frozenset[str] | None:
        """Panel genes with usable spatial signal; ``None`` when nothing declares it.

        ``None`` rather than an empty frozenset: "no detectability file" and "no gene
        was detectable" are different statements, and an empty frame would make every
        gene in the study ``NEVER_MEASURED``.

        **Detectability is per sample, and this is the frame that proves it.** It is
        never *declared* anywhere; it is read off a markers table, as the genes that
        table did not mark ``NoSig``. So a pooled table answers the pooled question,
        and handing its answer to one section says that section tested a gene its own
        file says produced no usable signal -- a fabricated measurement status, and
        the one the four-state lattice exists to make unrepresentable. It fabricates
        in both directions: the section is then credited with a *measured* absence
        that argues against the other section, and with an agreement when both
        sections were silent about a gene only the pooled read could see. Gene
        dropout between sections is the normal case (the demo runs 17.5% no usable
        signal on a panel designed for its tissue), not an edge.

        The panel frame deliberately does NOT move: one instrument panel was run
        across the sections, so what the apparatus was *capable* of is the store's,
        and only what each section actually got signal from is the sample's.

        A bound sample with no table on disk raises rather than returning ``None``:
        ``None`` here means "nothing declares detectability", which turns every
        ``NoSig`` gene into a tested absence that argues -- exactly the fabrication
        above, arriving as a silent fallback instead of a leak.

        **Controls are dropped here too, and this frame is the half that mattered.**
        A method writes its own output over whatever probes the instrument ran, so a
        negative control reaches ``markers_top.csv`` with a real unit and a real
        number on it. ``_panel_genes`` filtered and this did not, so the control sat
        in ``detect:<id>`` and the panel frame was the only thing keeping it out of
        the lattice -- and on a mixed-case panel file that frame let it in too. Both
        gene frames now refuse it, and because ``_state_for`` requires *every* frame
        of a kind to admit an entity, that refusal is what makes a control
        unadmittable rather than merely unlikely: it can never be ``OBSERVED``, so
        its count can never be promoted to a spatial marker carrying a number, and
        never ``TESTED_ABSENT``, so its absence can never argue against a cell type.
        See ``agent.data.is_control`` for where a control sits in the four states and
        what the lattice still cannot say about one.
        """
        p = self._markers_path(store)
        if not p.is_file():
            return None
        df = pd.read_csv(p, usecols=["gene", "top_cluster"])
        by_note = self._annotated_controls(store)
        return frozenset(
            str(g).strip() for g, u in zip(df["gene"], df["top_cluster"])
            if str(g).strip() and not is_no_signal(u)
            and not is_control_probe(g)
            and str(g).strip().upper() not in by_note
        )

    def frames(self, store: Study) -> tuple[Frame, ...]:
        """Every capability frame this store declares. Derived, never constant.

        An undeclarable frame is omitted rather than returned empty. ``_state_for``
        reads "no frame for this kind" as ``NEVER_MEASURED`` with a reason naming the
        gap, whereas an empty frame says the apparatus measured nothing at all -- the
        same verdict with a false explanation behind it. ``validate`` is where the gap
        becomes something a biologist can act on.

        **Which is why the two gene frames test ``is not None`` and not truthiness.**
        Both helpers already separated "nothing declares this" from "this is declared
        and empty", and both tests threw the distinction away: ``frozenset()`` is
        falsy, so a *declared* empty frame was dropped exactly as an undeclarable one
        was. The gene kind then had only ``panel:`` left, every frame of the kind
        admitted every panel gene, and a gene whose own ``markers_top.csv`` row says
        ``NoSig`` came back ``TESTED_ABSENT`` with ``absence_is_evidence`` -- handed
        to the rival machinery as "measured and did not mark this unit", with no flag
        and nothing for ``verify`` to read. The frame built to stop that fabrication
        was the thing producing it. (Reachable only since the controls filter above:
        a ``markers_top.csv`` in which the one gene with a unit is a control probe
        now yields an empty detectable set from a non-empty table.)

        Emitted instead, an empty ``detect:`` frame admits nothing, so every gene in
        the study resolves ``NEVER_MEASURED`` naming it, and no absence anywhere in
        the study argues against anything -- which is precisely what a study whose
        method marked every gene ``NoSig`` supports. ``validate`` blocks such a store
        at Stage 0 as well, and the two do different jobs: the blocker is what reaches
        the biologist, the frame is what keeps the lattice honest for the callers that
        never run Stage 0 (``agent.discriminate``, ``methods.*``, ``pipeline.records``,
        ``agent.intake``), which read ``frames`` directly on whatever is on disk.

        The two kinds below stay on truthiness deliberately. Each declares exactly one
        frame, so an omitted empty one already resolves ``NEVER_MEASURED`` -- the same
        state, with a reason ("no frame declared for cell_type") that is not false.
        Neither is read through a sentinel either: there is no "declared and empty" to
        tell apart from "not declared", because the store declaring no cell types and
        the store declaring no units are the one fact in both cases.
        """
        out: list[Frame] = []
        panel = self._panel_genes(store)
        if panel is not None:
            out.append(Frame(EntityKind.GENE, f"panel:{store.id}", panel))
        detect = self._detectable_genes(store)
        if detect is not None:
            # The sample is in the NAME, not only in the members, because the frame
            # name is what ``reconcile._compare`` quotes back at the biologist ("had
            # it in frame panel:X+detect:X"). Two sections whose detectability
            # genuinely differs must not cite one name for two frames.
            suffix = f":{self.sample}" if self.sample else ""
            out.append(Frame(EntityKind.GENE, f"detect:{store.id}{suffix}", detect))
        labels = frozenset(
            str(meta["cell_type"]).strip()
            for meta in store.unit_labels.values()
            if str(meta.get("cell_type", "")).strip()
        )
        if labels:
            out.append(Frame(EntityKind.CELL_TYPE, f"annotation:{store.id}", labels))
        units = frozenset(store.units)
        if units:
            out.append(Frame(EntityKind.UNIT, f"clusters:{store.id}", units))
        return tuple(out)

    # ---- what the store says about itself --------------------------------- #
    def unit_labels(self, store: Study) -> dict[str, str]:
        """Unit id -> display label. Display only; never load-bearing."""
        return {u: store.label_for(u) for u in store.units}

    def samples(self, store: Study) -> list[str]:
        """The samples in this store, first-hand. The count lives here and nowhere else.

        Three reads, in descending order of how explicit they are: a declared
        ``samples`` list, the distinct values of a ``sample`` column in the cell
        table, and -- for a flat cell table with no such column -- the single sample
        the store evidently holds, named after the store. The last is a count, not a
        guess: a table with no sample column holds one sample, and returning an empty
        list instead would let the clustering blocker fall through on a store that
        simply never mentioned the word.

        **Unless the store contradicts it.** ``inputs/samples/<sample>/`` holding
        per-sample marker tables is the store saying, in files, that it has more than
        one reading. Answering "one sample, named after the store" over the top of
        that is not a count and not a default: it is a false measurement claim, and
        the one that costs most, because a single sample makes
        ``declared_clustering`` return ``"independent"`` as a tautology and every
        ``ReadingRecord`` then carries a clustering mode nobody declared. So the
        fallback refuses, naming the tables it found and the file to fix.
        """
        declared = store.manifest.get("samples")
        if isinstance(declared, (list, tuple)) and declared:
            return [str(s) for s in declared]

        for rel in ("viz/cells.parquet", "viz/cells.csv"):
            p = store.dir / rel
            if not p.is_file():
                continue
            if p.suffix == ".parquet":
                head = pd.read_parquet(p)
                col = head["sample"] if "sample" in head.columns else None
            else:
                names = pd.read_csv(p, nrows=0).columns
                col = (pd.read_csv(p, usecols=["sample"])["sample"]
                       if "sample" in names else None)
            if col is None:
                break
            found = sorted({str(v).strip() for v in col if str(v).strip()})
            if found:
                return found
            break

        on_disk = self.sample_inputs(store)
        if on_disk:
            raise UndeclaredSamples(
                f"study {store.id!r} has per-sample marker tables for "
                f"{', '.join(sorted(on_disk))} under {store.inputs / 'samples'}, and "
                f"neither {store.dir / 'manifest.json'} nor the cell table says which "
                "samples the store holds. Answering with one sample named after the "
                "store would report a multi-sample study as single-sample and stamp "
                "'independent' clustering on every record without anyone declaring "
                "it. Declare `samples` in manifest.json (and, with more than one, "
                "`clustering_mode`)."
            )
        return [store.id]

    def sample_inputs(self, store: Study) -> dict[str, Path]:
        """Per-sample marker tables, by sample id. Empty when the store has none.

        ``samples`` says how many readings exist; this says where they are, and
        until it did, nothing in the codebase could produce the
        ``core.concordance.Replicate`` list §9.10 is defined over -- the function
        had no caller that was not a test, because a caller would have had to guess
        a path. Resolved from the filesystem under one convention,
        ``inputs/samples/<sample>/markers_top.csv``, and NOT from a manifest key:
        the same rule as ``available_views``, because a manifest may say a file is
        there and only the file settles it.

        A sample with no table is simply absent from the mapping rather than
        mapped to the pooled table. Handing back the pooled numbers under two
        sample names would make every replicate agree with itself, and §9.10's one
        informative outcome -- divergence -- would become unreachable while the
        check went on reporting that it had run.

        Scans the directory rather than iterating :meth:`samples`, which is both the
        honest reading of "resolved from the filesystem" and what lets
        :meth:`samples` consult this to catch its own fallback: a store whose sample
        list is undeclared is exactly the store whose per-sample tables nobody has
        asked about, and asking :meth:`samples` first would have made that
        unanswerable. A directory here with no table under it is not a sample.
        """
        base = store.inputs / "samples"
        if not base.is_dir():
            return {}
        out: dict[str, Path] = {}
        for d in sorted(base.iterdir()):
            p = d / "markers_top.csv"
            if d.is_dir() and p.is_file():
                out[d.name] = p
        return out

    def clustering_mode(self, store: Study) -> ClusteringMode:
        """§6.3's fourth blocker. Raises ``UndeclaredClustering``; never guesses."""
        return declared_clustering(
            self.samples(store),
            store.manifest.get("clustering_mode"),
            where=f"study {store.id!r} ({store.dir / 'manifest.json'})",
        )

    def available_views(self, store: Study) -> frozenset[str]:
        """Views the elements actually present can support.

        The manifest may veto a view whose element exists -- a deliberate "do not
        draw this" survives -- but it may never grant one whose element does not. A
        claim in a manifest is not a file.
        """
        declared = store.views_available
        return frozenset(
            name for name, elements in _VIEW_ELEMENTS
            if declared.get(name, True) and any(_present(store, e) for e in elements)
        )

    def coordinate_system(self, store: Study) -> tuple[str, float]:
        """Xenium writes cell centroids in micrometres, so the scale is 1.

        A vendor fact about the instrument's output, which is precisely what a
        platform adapter is for -- and the reason it is declared rather than measured
        is that a spatial statistic read at the wrong scale is wrong by a constant
        nobody can see in the figure.
        """
        return ("xenium_microns", 1.0)

    def reads(self, store: Study) -> dict[str, Condition]:
        """What the store states about the experiment, with provenance. Asserts
        nothing about whether it is enough.

        ``READ`` for everything parsed out of a file and ``UNKNOWN`` for everything
        else. The adapter never writes ``INFERRED`` -- that provenance means a model
        proposed it and the biologist confirmed, and neither has happened here.

        Never raises on a missing field: a store with a readable assay and no tissue
        still yields the assay. Deciding that the missing tissue blocks the study is
        :class:`Conditions`' job, through :meth:`standing_conditions`.
        """
        def read(key: str) -> Condition:
            v = store.manifest.get(key)
            return Condition(str(v), Provenance.READ) if v else UNKNOWN

        # Modality follows from the platform and is not a guess. A Xenium store is
        # imaging-based spatial transcriptomics by construction -- the same kind of
        # vendor fact as coordinate_system()'s micron scale, which nobody calls an
        # inference. It derives from `platform`, which IS read from the manifest, so
        # it is READ. Reporting it UNKNOWN threw away something certain and made
        # every downstream check carry "modality unknown" for no reason.
        modality = read("modality")
        if modality is UNKNOWN and (store.manifest.get("platform") or "").strip().lower() == "xenium":
            modality = Condition("imaging_spatial_transcriptomics", Provenance.READ)

        return {
            "assay": read("platform"),
            "modality": modality,
            "tissue": read("tissue"),
            "method": read("method"),
            "segmentation": read("segmentation"),
            "fixation": read("fixation"),
        }

    def standing_conditions(self, store: Study) -> Conditions:
        """:meth:`reads`, finished with the frames into a ``Conditions``.

        Raises ``Blocked`` (from ``Conditions``) when the store declares no frames,
        no tissue or no assay. That is the right layer to fail at: a study with no
        capability frame makes every absence ``NEVER_MEASURED`` and the panel-absence
        rule a machine that says nothing. ``validate`` reports the same gaps as
        strings, for the intake checklist, and ``reads`` hands over the fields that
        *were* readable without asserting the study is complete.
        """
        return Conditions(frames=tuple(f.ref() for f in self.frames(store)),
                          **self.reads(store))

    # ---- validation -------------------------------------------------------- #
    def validate(self, store: Study) -> list[str]:
        """Problems that would make this store uninterpretable; empty means usable.

        Returns rather than raises because §6.4's intake surface is a checklist the
        biologist ticks, and a traceback is not a checklist -- it reports the first
        problem and hides the rest, which is how a study gets fixed one round trip at
        a time.
        """
        if not store.exists:
            return [f"no manifest at {store.dir / 'manifest.json'}"]

        problems: list[str] = []
        platform = str(store.manifest.get("platform", "")).strip()
        if platform.lower() != "xenium":
            problems.append(
                f"manifest platform is {platform or '(unset)'!r}, not Xenium; this "
                "adapter declares the capability frame of a Xenium store"
            )

        frames = self.frames(store)
        kinds = {f.kind for f in frames}
        # Absent and empty are separate blockers, because they are separate facts and
        # the fix is different: one is a missing file, the other is a study in which
        # the method found nothing. ``frames`` now emits the empty frame rather than
        # dropping it, so a check written as "is there a frame?" would pass on the
        # worse of the two -- the store that measured nothing would validate clean.
        panel = next((f for f in frames if f.name.startswith("panel:")), None)
        detect = next((f for f in frames if f.name.startswith("detect:")), None)
        if panel is None:
            problems.append(
                f"no measured set: {store.inputs / 'panel.parquet'} is missing or has "
                "no usable gene column. The panel IS the capability frame (§6.3)"
            )
        elif not panel.members:
            problems.append(
                f"the measured set is empty: {store.inputs / 'panel.parquet'} lists no "
                "gene that is not a negative control, so the store declares no "
                "capability for any gene (§6.3)"
            )
        if detect is None:
            problems.append(
                f"no detectability frame: {store.inputs / 'markers_top.csv'} is "
                "missing, so a gene that produced no usable signal is "
                "indistinguishable from one tested and not reported"
            )
        elif not detect.members:
            problems.append(
                f"the detectability frame is empty: {self._markers_path(store)} "
                f"leaves no gene with usable spatial signal -- "
                f"{self._why_detect_empty(store)}. That is a fact about the study "
                "and it blocks it (§6.3): no absence here is evidence of anything, "
                "every gene resolves NEVER_MEASURED, and the result is "
                "uninterpretable rather than negative"
            )
        if EntityKind.CELL_TYPE not in kinds:
            problems.append(
                "no annotation label set, so no capability frame for cell types"
            )
        if EntityKind.UNIT not in kinds:
            problems.append("the store declares no units")

        for field in ("tissue", "platform"):
            if not str(store.manifest.get(field, "")).strip():
                problems.append(
                    f"{field} is not declared, and it blocks the study (§6.3)"
                )

        # Both sample-shaped blockers come out of one call, because ``samples``
        # itself refuses when the store has per-sample tables and declares no list.
        # ``UndeclaredSamples`` subclasses ``UndeclaredClustering``, so the one
        # handler below renders either as a checklist line rather than a traceback.
        try:
            samples = self.samples(store)
        except UndeclaredClustering as exc:
            problems.append(str(exc))
            return problems

        if len(samples) > 1:
            on_disk = self.sample_inputs(store)
            missing = [s for s in samples if s not in on_disk]
            if missing:
                # What the MISSING tables cost, counted rather than assumed. The
                # sentence here used to say "only the pooled table is readable, so
                # replicate concordance has nothing to run on" for every shape of
                # this problem -- which is false as soon as two of the declared
                # sections do have a table, and false in the direction that makes a
                # biologist stop reading a study that can still be compared. On a
                # kidney-shaped store, where one section failing QC is the ordinary
                # case rather than the edge, that is the difference between a
                # blocked study and a narrower denominator.
                readable = [s for s in samples if s in on_disk]
                if len(readable) >= 2:
                    cost = (
                        f"{len(readable)} of {len(samples)} sections can still be "
                        f"compared ({', '.join(readable)}), so every replicate "
                        f"coverage on this study is out of {len(readable)} and not "
                        f"{len(samples)}, and nothing these sections report can be "
                        "checked against the missing one"
                    )
                else:
                    cost = (
                        "fewer than two sections are readable, so replicate "
                        "concordance (ARCHITECTURE §9.10) has nothing to run on and "
                        "a divergence between samples cannot be seen at all"
                    )
                problems.append(
                    f"the store declares {len(samples)} samples and has no per-sample "
                    f"marker table for {', '.join(missing)} "
                    f"({store.inputs / 'samples' / '<sample>' / 'markers_top.csv'}). "
                    + cost
                )

        try:
            self.clustering_mode(store)
        except UndeclaredClustering as exc:
            problems.append(str(exc))
        return problems
