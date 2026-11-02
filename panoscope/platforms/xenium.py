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

from pathlib import Path
from typing import Final

import pandas as pd

from panoscope.data.study import Study
from panoscope.platforms.base import ClusteringMode, declared_clustering
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

_CONTROL_PREFIXES: Final[tuple[str, ...]] = (
    "NegControlProbe_", "NegControlCodeword_", "NegPrb", "BLANK_", "Blank-",
    "antisense_", "UnassignedCodeword_", "DeprecatedCodeword_",
)
"""Xenium puts its negative controls in the same table as the panel.

This is the platform-specific half of "how measured is derived": on another platform
the rule is different, which is why it lives in an adapter and not in ``measured.py``.
A control probe left in the frame makes the panel look larger than the assay and
makes a control's absence read as a biological absence.

Custom add-on panels are NOT filtered, because nothing in the tidy store marks them.
A study that mixes a base panel with add-ons and needs them separated has to declare
that separation; guessing it from a gene name is how a real gene gets dropped.
"""

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


class XeniumAdapter:
    """``PlatformAdapter`` for a 10x Xenium store in the tidy layout.

    Structural conformance only -- no base class, per ``types/contracts.py``. Holds
    no state and reads at call time, so one instance is safe to share and a study
    opened later is never served an earlier study's frames.
    """

    name = "xenium"

    # ---- frames ----------------------------------------------------------- #
    def _panel_genes(self, store: Study) -> frozenset[str]:
        """``panel.parquet``'s gene column, minus negative controls: the measured set."""
        p = store.inputs / "panel.parquet"
        if not p.is_file():
            return frozenset()
        col = pd.read_parquet(p, columns=["gene"])["gene"]
        return frozenset(
            g for g in (str(v).strip() for v in col)
            if g and not g.startswith(_CONTROL_PREFIXES)
        )

    def _detectable_genes(self, store: Study) -> frozenset[str] | None:
        """Panel genes with usable spatial signal; ``None`` when nothing declares it.

        ``None`` rather than an empty frozenset: "no detectability file" and "no gene
        was detectable" are different statements, and an empty frame would make every
        gene in the study ``NEVER_MEASURED``.
        """
        p = store.inputs / "markers_top.csv"
        if not p.is_file():
            return None
        df = pd.read_csv(p, usecols=["gene", "top_cluster"])
        return frozenset(
            str(g).strip() for g, u in zip(df["gene"], df["top_cluster"])
            if str(g).strip() and str(u).strip() != NO_SIGNAL
        )

    def frames(self, store: Study) -> tuple[Frame, ...]:
        """Every capability frame this store declares. Derived, never constant.

        An undeclarable frame is omitted rather than returned empty. ``_state_for``
        reads "no frame for this kind" as ``NEVER_MEASURED`` with a reason naming the
        gap, whereas an empty frame says the apparatus measured nothing at all -- the
        same verdict with a false explanation behind it. ``validate`` is where the gap
        becomes something a biologist can act on.
        """
        out: list[Frame] = []
        panel = self._panel_genes(store)
        if panel:
            out.append(Frame(EntityKind.GENE, f"panel:{store.id}", panel))
        detect = self._detectable_genes(store)
        if detect:
            out.append(Frame(EntityKind.GENE, f"detect:{store.id}", detect))
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
            return sorted({str(v).strip() for v in col if str(v).strip()})
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
        """
        out: dict[str, Path] = {}
        for s in self.samples(store):
            p = store.inputs / "samples" / s / "markers_top.csv"
            if p.is_file():
                out[s] = p
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
        if not any(f.name.startswith("panel:") for f in frames):
            problems.append(
                f"no measured set: {store.inputs / 'panel.parquet'} is missing or has "
                "no usable gene column. The panel IS the capability frame (§6.3)"
            )
        if not any(f.name.startswith("detect:") for f in frames):
            problems.append(
                f"no detectability frame: {store.inputs / 'markers_top.csv'} is "
                "missing, so a gene that produced no usable signal is "
                "indistinguishable from one tested and not reported"
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

        samples = self.samples(store)
        if len(samples) > 1:
            missing = [s for s in samples if s not in self.sample_inputs(store)]
            if missing:
                problems.append(
                    f"the store declares {len(samples)} samples and has no per-sample "
                    f"marker table for {', '.join(missing)} "
                    f"({store.inputs / 'samples' / '<sample>' / 'markers_top.csv'}). "
                    "Only the pooled table is readable, so replicate concordance "
                    "(ARCHITECTURE §9.10) has nothing to run on and a divergence "
                    "between samples cannot be seen"
                )

        try:
            self.clustering_mode(store)
        except UndeclaredClustering as exc:
            problems.append(str(exc))
        return problems
