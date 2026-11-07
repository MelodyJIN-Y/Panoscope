"""The corpus window: the input contract, the vocabulary and the ladder, as pure code.

harness v2 §2-§4. The model never computes anything here: the study's frame, the
alias restriction rule, the rung W0..W5, the modality and the species state are all
decided by this module from text and declared conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from panoscope.corpus.store import CorpusUnavailable

_PLACEHOLDER = re.compile(r"\{[A-Za-z_. ]*\}")
RELEASE_STATUS: Final = ("confirmed", "unpublished", "unresolved")


# --------------------------------------------------------------------------- #
# §2 Input contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CellTypeEntry:
    """One frame cell type: its label, CL id when one resolves, and (synonym, origin)."""

    label: str
    ontology_id: str | None = None
    synonyms: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StudyContext:
    """What a corpus build reads about the study. Refuses to exist incomplete."""

    study: str
    cell_types: tuple[CellTypeEntry, ...]
    tissue: str
    tissue_id: str
    platform: str
    modality_class: str
    disease_state: str = ""                         # "" = undeclared, window is W0*
    disease_qualifiers: tuple[tuple[str, str], ...] = ()
    release_pmid: str = ""
    release_status: str = "unresolved"

    def __post_init__(self) -> None:
        missing = [n for n in ("tissue", "tissue_id", "platform", "modality_class")
                   if not getattr(self, n).strip()]
        if missing:
            raise CorpusUnavailable(
                f"corpus build refused: {', '.join(missing)} missing"
                + (" (pick the tissue's UBERON term at intake)" if "tissue_id" in missing
                   else ""))
        if not self.cell_types:
            raise CorpusUnavailable("corpus build refused: the cell-type frame is empty "
                                    "(no annotation labels and no reference candidates)")
        if self.release_status not in RELEASE_STATUS:
            raise CorpusUnavailable(f"release paper status {self.release_status!r} is "
                                    f"not one of {RELEASE_STATUS}")
        if self.release_status == "confirmed" and not self.release_pmid.isdigit():
            raise CorpusUnavailable("a confirmed release paper needs its PMID")
        texts = [self.study, self.tissue, self.tissue_id, self.platform,
                 self.modality_class, self.disease_state, self.release_pmid,
                 *(v for q in self.disease_qualifiers for v in q),
                 *(c.label for c in self.cell_types),
                 *(s for c in self.cell_types for s, _ in c.synonyms)]
        if left := [t for t in texts if _PLACEHOLDER.search(t)]:
            raise CorpusUnavailable(f"corpus build refused: unrendered placeholder in {left}")

    @property
    def w0_star(self) -> bool:
        """The disease state is undeclared, so the exact window is W0*."""
        return not self.disease_state


def cell_type_frame(view: Any, labels: tuple[str, ...]) -> tuple[CellTypeEntry, ...]:
    """The full frame: annotation labels plus reference candidates, joined by view key."""
    entries: dict[str, str] = {}
    for name in (*labels, *view.candidates()):
        entries.setdefault(view.key(name), name)
    syns: dict[str, list[tuple[str, str]]] = {}
    for e in view.graph.edges:
        if e.predicate == "synonym":
            syns.setdefault(view.key(e.object.id), []).append(
                (e.subject.id, e.source.split("@")[0]))
    return tuple(CellTypeEntry(label, k if k.startswith("CL:") else None,
                               tuple(dict.fromkeys(syns.get(k, ()))))
                 for k, label in entries.items())


def context_for(study: Any) -> StudyContext:
    """Read the StudyContext off a built study: manifest, conditions, view and labels."""
    from panoscope import platforms
    from panoscope.data.study import NOT_YET_CALLED
    from panoscope.reference import view as ref_view

    m = study.manifest
    try:
        cond = platforms.for_study(study).reads(study)
    except Exception as exc:  # noqa: BLE001 - an unreadable study is a refused build
        raise CorpusUnavailable(f"corpus build refused: {exc}") from exc
    labels = tuple(dict.fromkeys(
        v.get("cell_type") for v in study.unit_labels.values()
        if v.get("cell_type") and v.get("cell_type") != NOT_YET_CALLED))

    def val(name: str) -> str:
        c = cond.get(name)
        return c.value if c is not None and c.known else ""

    rel = m.get("release_paper") or {}
    return StudyContext(
        study=study.id,
        cell_types=cell_type_frame(ref_view.for_study(study), labels),
        tissue=val("tissue"), tissue_id=str(m.get("tissue_id") or ""),
        platform=val("assay"), modality_class=val("modality"),
        disease_state=val("disease_state"),
        disease_qualifiers=tuple((k.removeprefix("disease_"), val(k))
                                 for k in ("disease_subtype", "disease_treatment",
                                           "disease_stage") if val(k)),
        release_pmid=str(rel.get("pmid") or ""),
        release_status=str(rel.get("status") or "unresolved"),
    )


# --------------------------------------------------------------------------- #
# §3 Vocabulary
# --------------------------------------------------------------------------- #
# ponytail: small hand lists; swap for a word list if census shows noisy aliases slip by.
_ENGLISH: Final = frozenset({
    "act", "camp", "clock", "face", "fast", "fish", "gaps", "impact", "light", "mice",
    "nail", "pole", "rest", "ring", "spin", "spring", "stop", "trap", "wars", "wasp",
    "chip", "coil", "damage", "early", "great", "hole", "large", "late", "mask", "past",
})
_ACRONYMS: Final = frozenset({
    "AIDS", "SARS", "COPD", "ELISA", "MALDI", "CRISPR", "HPLC", "NMDA", "GABA", "MRSA",
})
_BARE_CD = re.compile(r"CD\d+[A-Za-z]?", re.I)


def restricted(term: str) -> bool:
    """Proximity-only: <=3 chars, an English word, a common acronym or a bare CD number."""
    t = term.strip()
    return (len(t) <= 3 or t.casefold() in _ENGLISH or t.upper() in _ACRONYMS
            or bool(_BARE_CD.fullmatch(t)))


@dataclass(frozen=True, slots=True)
class Term:
    text: str
    origin: str          # hgnc_symbol | hgnc_prev | hgnc_alias | hgnc_name | mesh | study
    restricted: bool


def gene_terms(symbol: str, hgnc: dict[str, Any] | None,
               mesh: tuple[str, ...] = ()) -> tuple[Term, ...]:
    """Every alias of a gene, with origin; HGNC then MeSH, first origin wins, no memory."""
    rows = [(symbol, "study")]
    if hgnc:
        rows = [(hgnc.get("symbol") or symbol, "hgnc_symbol"), (symbol, "study")]
        rows += [(a, "hgnc_prev") for a in hgnc.get("prev_symbol") or ()]
        rows += [(a, "hgnc_alias") for a in hgnc.get("alias_symbol") or ()]
        rows += [(hgnc["name"], "hgnc_name")] if hgnc.get("name") else []
    rows += [(a, "mesh") for a in mesh]
    out: dict[str, Term] = {}
    for text, origin in rows:
        out.setdefault(text.casefold(), Term(text, origin, restricted(text)))
    return tuple(out.values())


# --------------------------------------------------------------------------- #
# §4 Ladder, species and modality
# --------------------------------------------------------------------------- #
class Rung(StrEnum):
    W0 = "W0"            # exact window: tissue, disease state, cell type
    W0_STAR = "W0*"      # exact window while the study's disease state is undeclared
    W1 = "W1"            # the setting, no cell-type resolution
    W2 = "W2"            # same tissue, other disease state or treatment
    W3 = "W3"            # same cell type, other tissue
    W4 = "W4"            # other species
    W5 = "W5"            # in vitro

    @property
    def width(self) -> int:
        return int(self.value[1])


class Species(StrEnum):
    HUMAN_STATED = "HUMAN_STATED"
    HUMAN_INFERRED = "HUMAN_INFERRED"
    UNSTATED = "UNSTATED"
    NON_HUMAN = "NON_HUMAN"


@dataclass(frozen=True, slots=True)
class Facets:
    """Verified facets of one statement: True matches the study, False is a verified
    mismatch, None was not verified."""

    tissue: bool | None = None
    cell_type: bool | None = None
    disease: bool | None = None
    in_vivo: bool | None = None          # False = in vitro
    species: Species = Species.UNSTATED
    biofluid: bool = False
    resolution: str = "UNKNOWN"          # BULK caps at W1


def rung(f: Facets, *, disease_declared: bool) -> Rung | None:
    """The one rung a statement sits at; None is beyond the ladder (no rung)."""
    ct = f.cell_type is True and f.resolution != "BULK"
    if f.biofluid:
        return None
    if f.in_vivo is False:
        return Rung.W5 if (f.tissue is True or ct) else None
    if f.in_vivo is not True:
        return None
    if f.species is Species.NON_HUMAN:
        return Rung.W4 if (f.tissue is True or ct) else None
    if f.tissue is True:
        if f.disease is True or not disease_declared:
            if not ct:
                return Rung.W1
            return Rung.W0 if disease_declared else Rung.W0_STAR
        return Rung.W2
    return Rung.W3 if ct else None


_NON_HUMAN = re.compile(r"\b(mouse|mice|murine|rats?|zebrafish|porcine|pigs?|canine|dogs?|"
                        r"bovine|macaques?|non-human primates?|xenografts?|drosophila)\b", re.I)
_HUMAN = re.compile(r"\bhuman(?:\s+[\w-]+){0,2}?\s+(tissues?|samples?|donors?|patients?|"
                    r"specimens?|biops(?:y|ies)|cohorts?|subjects?)\b|\b(patients?|women|"
                    r"donors?|TCGA|METABRIC)\b", re.I)
_CLINICAL = re.compile(r"\b(biops(?:y|ies)|resections?|nephrectom(?:y|ies)|mastectom(?:y|ies)|"
                       r"surgical specimens?|cohorts?|clinicopathological|TNM|pT[0-4])\b", re.I)


def species_state(span: str, methods: str = "", read_text: str = "") -> Species:
    """Tri-state species (plus NON_HUMAN) from text; "human HER2" is not a species claim."""
    local = f"{span}\n{methods}"
    if _NON_HUMAN.search(local):
        return Species.NON_HUMAN
    if _HUMAN.search(local):
        return Species.HUMAN_STATED
    whole = f"{local}\n{read_text}"
    if not _NON_HUMAN.search(whole) and _CLINICAL.search(whole):
        return Species.HUMAN_INFERRED
    return Species.UNSTATED


# (pattern, analyte, resolution); the earliest term in the text wins.
_LEXICON: Final = tuple((re.compile(p, re.I), a, r) for p, a, r in (
    (r"\b(xenium|cosmx|merfish|smfish|rnascope|in situ hybridi[sz]ation|ISH)\b",
     "RNA", "CELL_IN_SITU"),
    (r"\b(scRNA-seq|snRNA-seq|single[- ](?:cell|nucle(?:us|i)) RNA)", "RNA", "CELL_DISSOCIATED"),
    (r"\b(visium|geomx|spatial transcriptomics)\b", "RNA", "SPOT"),
    (r"\b(copy number|amplification|amplified)\b", "DNA", "UNKNOWN"),
    (r"\b(immunohistochemi\w*|IHC|immunofluorescen\w*|imaging mass cytometry|IMC|"
     r"CycIF|multiplex immuno\w*)\b", "PROTEIN", "CELL_IN_SITU"),
    (r"\b(flow cytometry|mass cytometry|CyTOF)\b", "PROTEIN", "CELL_DISSOCIATED"),
    (r"\b(western blot\w*|ELISA)\b", "PROTEIN", "BULK"),
    (r"\b(bulk RNA|RNA-seq|microarray|qPCR|RT-PCR|TCGA)\b", "RNA", "BULK"),
))


def modality(text: str) -> tuple[str, str]:
    """(analyte, resolution) from the earliest lexicon term in ``text``; else UNKNOWN."""
    # ponytail: earliest term wins in mixed-method text; pass the span before Methods.
    hits = [(m.start(), a, r) for rx, a, r in _LEXICON if (m := rx.search(text))]
    return (min(hits)[1], min(hits)[2]) if hits else ("UNKNOWN", "UNKNOWN")


def modality_match(text: str, platform: str) -> str:
    """SAME_PLATFORM > SAME > ADJACENT_RNA > OTHER_ANALYTE > SPOT > BULK, else UNKNOWN."""
    if platform and re.search(rf"\b{re.escape(platform)}\b", text, re.I):
        return "SAME_PLATFORM"
    analyte, res = modality(text)
    if analyte != "UNKNOWN" and (analyte, res) == modality(platform):
        return "SAME"
    if res in ("SPOT", "BULK"):
        return res
    if res.startswith("CELL_"):
        return "ADJACENT_RNA" if analyte == "RNA" else "OTHER_ANALYTE"
    return "UNKNOWN"
