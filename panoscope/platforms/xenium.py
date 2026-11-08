"""The Xenium adapter: what the 10x Xenium store was CAPABLE of seeing, and what it
says about itself.

Every member is a read; none computes a statistic. The adapter declares the
apparatus and the method declares what it found.

**Two gene frames, not one.** The panel says which genes the assay *could* report;
detectability says which of those produced usable spatial signal (49 of the demo's
280 did not). ``_state_for`` requires *every* frame of a kind to admit an entity, so
an undetectable gene resolves ``NEVER_MEASURED`` rather than a tested absence.

**The panel size is derived, never asserted**: counted from ``panel.parquet``, not
read from the manifest's ``panel_gene_count``.

**The clustering blocker** delegates to ``base.declared_clustering``.

**Not inferred:** ``tissue`` stays the manifest's prose (mapping it to UBERON would
be inference) and ``fixation`` stays ``UNKNOWN`` since no file states it. Intake
fills both, with provenance.
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
from panoscope.types.conditions import (
    DISEASE_STATE, UNKNOWN, Condition, Conditions, Provenance,
)
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

    The one predicate every reader of this sentinel uses, so they agree; the
    spelling is whatever the upstream run emitted. **Blank and NaN count as no
    signal** (the conservative direction): an empty unit column does not declare
    the gene detected.
    """
    s = "" if value is None else str(value).strip()
    return not s or s.upper() == NO_SIGNAL.upper() or s.lower() == "nan"


CONTROL_PREFIXES: Final[tuple[str, ...]] = (
    "NEGCONTROLPROBE", "NEGCONTROLCODEWORD", "NEGPRB", "BLANK", "ANTISENSE",
    "UNASSIGNEDCODEWORD", "DEPRECATEDCODEWORD",
)
"""Xenium puts its negative controls in the same table as the panel.

**Prefix FAMILIES, not spellings** (``BLANK`` matches ``BLANK_0003`` and
``BlankCodeword_0001``), each the longest string that is still only the family:
``NEG`` is not one because NEGR1 is a real gene. ``tests/test_panel_controls.py`` pins
both directions. Upper-cased because the comparison is (``is_control_probe``).

Custom add-on panels are NOT filtered; nothing in the tidy store marks them.

Public and shared with ``agent/data.py`` so the frame and the tidy reader drop the
same rows and agree on the size of the assay.
"""


_CONTROL_PREFIXES_UPPER: Final[tuple[str, ...]] = tuple(
    p.upper() for p in CONTROL_PREFIXES
)
"""The list above, upper-cased once. Not a second list -- a second comparison."""


def is_control_probe(name: str) -> bool:
    """True if ``name`` is a Xenium control probe rather than a panel gene, in ANY case.

    Lets a caller say *why* a name is off-panel. Case-insensitive like every other
    gene-identity comparison here; ``agent.data.is_control`` delegates here so the
    frame and the readers share one rule.
    """
    return str(name).strip().upper().startswith(_CONTROL_PREFIXES_UPPER)


CONTROL_ANNOTATION_TOKENS: Final[tuple[str, ...]] = ("CONTROL", "CODEWORD")
"""Words in ``panel.parquet``'s ``annotation`` column that mark a row as a probe.

The half of the control rule a vendor cannot rename: a control named
``IntergenicRegion_0001`` matches no prefix family but is annotated
``negative control``. Matched as substrings of the upper-cased annotation and kept to
two terms already used as control vocabulary; widening it risks deleting real genes.

Name OR annotation: either one marks a control.
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


class UnreadableStoreFile(ValueError):
    """A file in the store cannot be read, named.

    pyarrow reports a corrupt parquet as ``Could not open Parquet input source
    '<Buffer>'`` -- it never says WHICH file, because pandas hands it a buffer.
    A biologist whose panel was truncated in transfer got that traceback at the
    intake screen, naming nothing they could act on.
    """


def _read_parquet(path):
    """``pd.read_parquet`` that names the file when it cannot be read."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - re-raised, named
        raise UnreadableStoreFile(
            f"{path} could not be read as parquet ({type(exc).__name__}: "
            f"{exc}). The file is corrupt, truncated, or not a parquet file."
        ) from exc


@lru_cache(maxsize=16)
def _annotated_controls_cached(path: str, mtime_ns: int, size: int) -> frozenset[str]:
    """``panel.parquet``'s control rows BY ANNOTATION, memoised on the file itself.

    Asked once per marker-table row, so cached. Keyed on ``(path, mtime_ns, size)``
    so an in-process rewrite of study inputs changes the key.
    """
    df = _read_parquet(path)
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

    Structural conformance only (``types/contracts.py``). Reads at call time and
    holds no path, table or study, so one instance is safe to share. ``sample`` is
    the one thing an instance carries (see :meth:`for_sample`): it moves only the
    detectability frame.
    """

    name = "xenium"

    def __init__(self, sample: str | None = None) -> None:
        self.sample = sample

    def for_sample(self, sample: str) -> "XeniumAdapter":
        """This adapter, reading ``sample``'s own tables rather than the pooled ones.

        A new instance, not a mutation: the pooled adapter is shared. Not on the
        ``PlatformAdapter`` Protocol yet; its one caller is
        ``pipeline/records.py::_concordance``.
        """
        return XeniumAdapter(sample)

    # ---- frames ----------------------------------------------------------- #
    def _panel_genes(self, store: Study) -> frozenset[str] | None:
        """``panel.parquet``'s gene column, minus negative controls: the measured set.

        ``None`` when the file is absent or unreadable (no frame at all); an empty
        frozenset is a panel declaring it measured nothing.
        """
        p = store.inputs / "panel.parquet"
        if not p.is_file():
            return None
        try:
            df = _read_parquet(p)
        except Exception:  # noqa: BLE001 - an unreadable panel is "no measured set"
            return None
        if "gene" not in df.columns:
            # UNREADABLE, not empty -- the same answer a missing file gets, so
            # intake shows its checklist line instead of a KeyError.
            return None
        notes = (df["annotation"] if "annotation" in df.columns
                 else [None] * len(df))
        return frozenset(
            g for g, note in zip((str(v).strip() for v in df["gene"]), notes)
            if g and not is_control_probe(g) and not is_control_annotation(note)
        )

    def is_control(self, store: Study, entity: str) -> bool:
        """``PlatformAdapter.is_control``: the store's own answer about one name.

        Needed beyond the frames: a control row in ``markers_top.csv`` would
        otherwise become a real never-measured subject (off-panel notes, §9.10
        denominator). ``agent/data.is_control`` is the v1-side twin; both read the
        same two columns.
        """
        return (is_control_probe(entity)
                or str(entity).strip().upper() in self._annotated_controls(store))

    def _annotated_controls(self, store: Study) -> frozenset[str]:
        """Upper-cased names the panel file's ANNOTATION marks as control probes.

        ``markers_top.csv`` has no annotation column, so this keeps
        annotation-marked probes out of ``detect:<id>`` and its reported size.
        """
        p = store.inputs / "panel.parquet"
        if not p.is_file():
            return frozenset()
        st = p.stat()
        return _annotated_controls_cached(str(p), st.st_mtime_ns, st.st_size)

    def _markers_path(self, store: Study) -> Path:
        """The marker table THIS adapter speaks for: the bound sample's, or pooled.

        Shared so the detectability frame and ``validate``'s blocker name the same
        file.
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

        Says what the file actually holds (all NoSig, all control probes, or no
        rows) rather than asserting one cause.
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
        was detectable" are different statements.

        **Per sample.** Read off the bound sample's markers table as the genes it did
        not mark ``NoSig``; a pooled answer handed to one section would fabricate a
        measured absence (or agreement). The panel frame does NOT move: one panel was
        run across the sections. A bound sample with no table raises.

        **Controls are dropped here too**, so both gene frames refuse them and a
        control can never be ``OBSERVED`` or ``TESTED_ABSENT``.
        """
        p = self._markers_path(store)
        if not p.is_file():
            return None
        if not {"gene", "top_cluster"} <= set(pd.read_csv(p, nrows=0).columns):
            return None      # a table nothing recognises declares no detectability
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

        An undeclarable frame is omitted (``_state_for`` reads that as
        ``NEVER_MEASURED`` with a reason naming the gap); ``validate`` turns the gap
        into a blocker.

        The two gene frames test ``is not None``, not truthiness: a *declared* empty
        frame is emitted, so it admits nothing and every gene resolves
        ``NEVER_MEASURED`` -- dropping it would leave ``panel:`` alone and turn a
        ``NoSig`` gene into a tested absence. Callers that never run Stage 0 read
        ``frames`` directly, so the frame itself must stay honest. The cell-type and
        unit kinds stay on truthiness: an omitted empty one is the same state.
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
        units = frozenset(store.units) or self._result_units(store)
        if units:
            out.append(Frame(EntityKind.UNIT, f"clusters:{store.id}", units))
        return tuple(out)

    def _result_units(self, store: Study) -> frozenset[str]:
        """Units read off every recognised method result: a fresh study declares no
        other list yet."""
        from panoscope import methods

        return frozenset(u for p, m in methods.recognise_inputs(store.inputs).items()
                         if m is not None
                         for u in m.detect_unit(pd.read_csv(p)).units)

    # ---- what the store says about itself --------------------------------- #
    def unit_labels(self, store: Study) -> dict[str, str]:
        """Unit id -> display label. Display only; never load-bearing."""
        return {u: store.label_for(u) for u in store.units}

    def samples(self, store: Study) -> list[str]:
        """The samples in this store, first-hand. The count lives here and nowhere else.

        In order: a declared ``samples`` list, the distinct values of a ``sample``
        column in the cell table, else the single sample the store holds, named
        after the store. That fallback refuses (``UndeclaredSamples``) when
        ``inputs/samples/<sample>/`` tables say there is more than one reading.
        """
        declared = store.manifest.get("samples")
        if isinstance(declared, (list, tuple)) and declared:
            return [str(s) for s in declared]

        for rel in ("viz/cells.parquet", "viz/cells.csv"):
            p = store.dir / rel
            if not p.is_file():
                continue
            if p.suffix == ".parquet":
                head = _read_parquet(p)
                col = head["sample"] if "sample" in head.columns else None
            else:
                names = pd.read_csv(p, nrows=0).columns
                col = (pd.read_csv(p, usecols=["sample"])["sample"]
                       if "sample" in names else None)
            if col is None:
                break
            # ``pd.notna`` first: a null reads back as the float NaN, and
            # ``str(nan)`` is the non-empty string "nan", which passed this
            # filter. A vendor export whose sample column is present but
            # unfilled was accepted as a study with one sample literally named
            # "nan", and every per-sample view then described it.
            found = sorted({str(v).strip() for v in col
                            if pd.notna(v) and str(v).strip()})
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

        Resolved from the filesystem under ``inputs/samples/<sample>/markers_top.csv``,
        not from a manifest key. A sample with no table is absent, never mapped to the
        pooled table (which would make every replicate agree with itself). Scans the
        directory rather than :meth:`samples`, which consults this.
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
        """What the store states about the experiment, with provenance.

        ``READ`` for everything parsed out of a file, ``STATED`` for a key the manifest's
        ``stated`` list names, ``UNKNOWN`` otherwise; never ``INFERRED``. Never raises on
        a missing field: blocking is :class:`Conditions`' job, through
        :meth:`standing_conditions`.
        """
        stated = store.manifest.get("stated") or ()  # keys the biologist stated at intake

        def read(key: str) -> Condition:
            v = store.manifest.get(key)
            prov = Provenance.STATED if key in stated else Provenance.READ
            return Condition(str(v), prov) if v else UNKNOWN

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
            **{f: read(f) for f in DISEASE_STATE},
        }

    def standing_conditions(self, store: Study) -> Conditions:
        """:meth:`reads`, finished with the frames into a ``Conditions``.

        Raises ``Blocked`` (from ``Conditions``) when the store declares no frames,
        no tissue or no assay. ``validate`` reports the same gaps as strings.
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
        from panoscope import methods

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
        if not any(methods.recognise_inputs(store.inputs).values()):
            problems.append(
                f"no method result: nothing in {store.inputs} is a table a registered "
                "method reads, so there is no result to interpret"
            )
        if detect is not None and not detect.members:
            problems.append(
                f"the detectability frame is empty: {self._markers_path(store)} "
                f"leaves no gene with usable spatial signal -- "
                f"{self._why_detect_empty(store)}. That is a fact about the study "
                "and it blocks it (§6.3): no absence here is evidence of anything, "
                "every gene resolves NEVER_MEASURED, and the result is "
                "uninterpretable rather than negative"
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

        on_disk = self.sample_inputs(store)
        # Per-sample marker tables feed the marker workflow only: a study with no
        # marker result at all (DE-only, enrichment-only) is not missing them.
        if len(samples) > 1 and (on_disk or (store.inputs / "markers_top.csv").is_file()):
            missing = [s for s in samples if s not in on_disk]
            if missing:
                # What the MISSING tables cost, counted: two readable sections
                # still give a (narrower) comparison.
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
