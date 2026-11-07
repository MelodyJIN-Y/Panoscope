"""Cards: what the corpus says per (gene, cell type) and per cell type, rendered by code.

Headline first (the in-window count at the study's modality), then in-window
statements with consensus and dissent both cited, then wider rungs, each labelled.
The study's own dataset papers render apart and are never counted. Absence reads
"thin" only after a complete search; never as novelty. A card supplies no value to the
measurement, and dissent is a display flag: it never reaches ``verify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panoscope.corpus.retrieve import KeyState, Support, key_state
from panoscope.corpus.store import Corpus, Statement

WINDOW: Final = ("W0", "W0*")
STUDY_MODALITY: Final = ("SAME_PLATFORM", "SAME")
NOT_COUNTED: Final = ("SAME_DATASET", "SPECIES_UNSTATED")
RUNG_LABELS: Final = {
    "W1": "W1 the setting, no cell-type resolution",
    "W2": "W2 same tissue, other disease state or treatment",
    "W3": "W3 same cell type, other tissue",
    "W4": "W4 other species",
    "W5": "W5 in vitro",
}
READS_AS: Final = ("Literature topically relevant to this gene in this setting. It is not "
                   "confirmation that the marker works in this assay, and it supplies no "
                   "value to the measurement.")


@dataclass(frozen=True, slots=True)
class Card:
    cell_type: str
    platform: str
    state: KeyState

    @property
    def own(self) -> tuple[Statement, ...]:
        """This dataset's own papers: shown apart, never counted."""
        return tuple(s for s in self.state.statements if "SAME_DATASET" in s.flags)

    @property
    def window(self) -> tuple[Statement, ...]:
        return tuple(s for s in self.state.statements
                     if s.rung in WINDOW and "SAME_DATASET" not in s.flags)

    @property
    def consensus(self) -> tuple[Statement, ...]:
        return tuple(s for s in self.window if s.stance in ("supports", "qualifies"))

    @property
    def dissent(self) -> tuple[Statement, ...]:
        return tuple(s for s in self.window if s.stance == "argues_against")

    @property
    def widened(self) -> dict[str, tuple[Statement, ...]]:
        return {label: got for r, label in RUNG_LABELS.items()
                if (got := tuple(s for s in self.state.statements
                                 if s.rung == r and "SAME_DATASET" not in s.flags))}

    @property
    def headline(self) -> str:
        if self.state.state is not Support.FOUND:
            return self.state.reads_as
        counted = [s for s in self.window if not set(s.flags) & set(NOT_COUNTED)]
        k = len({s.uri for s in counted if s.modality_match in STUDY_MODALITY})
        line = (f"{k} in-window paper(s) at the study modality ({self.platform or 'this assay'})"
                f", {len({s.uri for s in counted})} in window in all")
        if self.own:
            line += (f"; {len({s.uri for s in self.own})} from this dataset's own papers "
                     "(not counted)")
        if self.state.gaps:
            line += f"; search incomplete ({'; '.join(self.state.gaps)})"
        return line

    def lines(self) -> list[str]:
        """The card as text: headline, in window, wider rungs, own dataset papers."""
        out = [self.headline]
        groups = [("In window, consensus", self.consensus), ("In window, dissent", self.dissent),
                  *((f"Widened to {label}", got) for label, got in self.widened.items()),
                  ("This dataset's own papers", self.own)]
        for title, got in groups:
            if got:
                out.append(f"{title}:")
                out += [cite(s) for s in got]
        return out


def cite(s: Statement) -> str:
    badges = "".join(f" [{f}]" for f in s.flags
                     if f in ("SPECIES_UNSTATED", "PREPRINT", "ABSTRACT_ONLY"))
    return (f'{s.gene} in {s.cell_type or "the tissue"}, {s.predicate.lower()}: '
            f'"{s.span}" [{s.uri}]{badges}')


def card(corpus: Corpus, gene: str, cell_type: str, platform: str = "") -> Card:
    return Card(cell_type, platform, key_state(corpus, gene, cell_type))


def cell_type_card(corpus: Corpus, cell_type: str, platform: str = "") -> Card:
    """Every gene's statements about one cell type in this tissue, gaps unioned."""
    c = cell_type.casefold()
    genes = tuple(dict.fromkeys(q.gene for q in corpus.queries
                                if q.cell_type.casefold() == c and q.gene))
    states = [key_state(corpus, g, cell_type) for g in genes]
    found = tuple(s for k in states for s in k.statements)
    gaps = tuple(dict.fromkeys(f"{k.gene}: {g}" for k in states for g in k.gaps))
    state = Support.NOT_BUILT if not genes else Support.FOUND if found else Support.NONE_FOUND
    return Card(cell_type, platform, KeyState(", ".join(genes), cell_type, state, found, gaps))


def note(c: Card, corpus: Corpus) -> dict:
    """A notes-file entry from a card: summary, first cited pmid, the card lines.

    ``verify`` is always False here: literature, dissent included, never sets it (D1).
    """
    first = next(iter(c.consensus or c.dissent), None)   # never this dataset's own paper
    rec = next(iter(corpus.find(first.uri)), None) if first else None
    pmid = first.uri[5:] if first and first.uri.startswith("pmid:") else None
    summary = c.headline + (f'. "{first.span}"' if first else "")
    if c.dissent and first is not c.dissent[0]:
        summary += f' Dissent: "{c.dissent[0].span}" [{c.dissent[0].uri}]'
    return {
        "summary": summary,
        "pmid": pmid,
        "citation": ({"pmid": pmid, "title": rec.title, "authors": rec.authors,
                      "year": rec.year} if rec and pmid else None),
        "verify": False,
        "dissent": len(c.dissent),
        "card": c.lines(),
        "reads_as": READS_AS,
    }


def stance_toward(corpus: Corpus, genes: tuple[str, ...], cell_type: str
                  ) -> tuple[tuple[Statement, ...], tuple[Statement, ...], tuple[KeyState, ...]]:
    """(agree, dissent, key states) toward a call of ``cell_type`` on ``genes``, at read time.

    In-window statements only; the study's own dataset papers never count either way.
    """
    cards = [card(corpus, g, cell_type) for g in genes]
    return (tuple(s for c in cards for s in c.consensus),
            tuple(s for c in cards for s in c.dissent), tuple(c.state for c in cards))
