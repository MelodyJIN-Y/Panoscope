"""The confident-floor ENFORCER.

Every number, marker, citation, and lab note in an agent answer must trace back
to a real source before the answer is allowed to reach the biologist. This module
is that gate. It is deliberately paranoid and it is INDEPENDENT OF THE SIDECAR:

- The prose is the ground truth of what the agent *said*. Every numeric claim is
  extracted from the prose and re-checked against :mod:`agent.data` (the real
  jazzPanda values). A number that appears in prose but NOT in the sidecar must
  STILL be checked — the sidecar is a localization aid, never a whitelist.
- Every PMID/DOI is extracted and must resolve through an INJECTED
  ``literature_verifier`` callable. An unresolved identifier is a violation. A
  fabricated citation is the worst possible failure, so this fails closed.
- Every referenced lab note must exist and be cited -- and must be IN SCOPE. The
  scope half reads the real note store (``agent/memory.apply_notes``, the single
  gate every other caller already goes through) when no note set is injected, so
  it fires on what the agent actually says rather than only under test.
- Every ``[claim:id]`` resolves through ``ClaimStore.cite`` -- the store's own USE
  gate, which refuses a missing, retracted or out-of-scope claim -- and every
  ``[edge:...]`` must match an edge of the consulted reference graph by identity.
- A sentence that reasons from the absence of something the capability frame says
  was NEVER MEASURED is a violation. That is the panel-absence rule, and it is the
  one class here with no number in it to anchor on.
- ``check_record`` gates an ``InterpretationRecord``'s GENERATED layer, named by
  ``types.record.GENERATED_FIELDS`` rather than by the string ``"prose"``.

If the checker cannot positively confirm a claim traces to source, the claim is a
violation and the answer is rejected (``ok == False``). Fail closed, never open.

Consumers
---------
- ``agent/loop.py`` — ``finalize()`` builds a ``GroundingSidecar`` and calls
  ``GroundingChecker.check()``; any violation discards the answer and falls back
  (BLUEPRINT sections 3 and 5).
- ``tests/test_grounding.py`` — exercises the gate on clean and poisoned answers.

Design notes
------------
``SourceIndex`` imports :mod:`agent.data` and reuses its loaders — there is NO
second parser of the marker files. The only numeric source of truth is what
``agent.data.get_marker`` returns (a ``pd.Series`` with fields ``gene``,
``top_cluster``, ``glm_coef``, ``pearson``, ``max_gg_corr``, ``max_gc_corr``,
``cell_type``). This module never reads a data file directly and never writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Union

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.types.record import GENERATED_FIELDS

# --------------------------------------------------------------------------- #
# Tolerances
# --------------------------------------------------------------------------- #
# A number in prose is usually rounded (e.g. "17.998" or "18.0" for 17.9978...).
# We accept a match within an absolute OR relative tolerance. The relative band
# covers claims like "glm 18" for a true 17.998; the absolute band covers small
# stats near zero (e.g. pearson 0.42). Values come from config if present so the
# whole app shares one tolerance; otherwise sensible defaults.
_NUMBER_ABS_TOL: float = getattr(cfg, "NUMBER_ABS_TOL", 1e-2)
_NUMBER_REL_TOL: float = getattr(cfg, "NUMBER_REL_TOL", 5e-3)

# The four per-gene statistics that live in markers_top.csv and are therefore
# groundable against agent.data. A numeric claim whose stat is one of these is
# checked; a numeric claim tied to a stat we cannot resolve is reported as
# UNVERIFIABLE (fail closed) rather than silently passed.
_GROUNDABLE_STATS: frozenset[str] = frozenset(
    {"glm_coef", "pearson", "max_gg_corr", "max_gc_corr"}
)

# Map of prose stat keywords -> canonical stat name in markers_top.csv.
# Order matters: longer / more specific phrases first so "gene-gene correlation"
# wins over a bare "correlation", and "glm_coef" wins over a bare "glm".
_STAT_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"max[\s_-]*gg[\s_-]*corr", "max_gg_corr"),
    (r"max[\s_-]*gc[\s_-]*corr", "max_gc_corr"),
    (r"gene[\s_-]*gene\s+corr(?:elation)?", "max_gg_corr"),
    (r"gene[\s_-]*cluster\s+corr(?:elation)?", "max_gc_corr"),
    (r"glm[\s_-]*coef(?:ficient)?", "glm_coef"),
    (r"glm", "glm_coef"),
    (r"coef(?:ficient)?", "glm_coef"),
    (r"pearson", "pearson"),
)
_STAT_KEYWORD_RE = re.compile(
    "|".join(f"(?:{pat})" for pat, _ in _STAT_SYNONYMS), re.IGNORECASE
)


def _canonical_stat(keyword: str) -> Optional[str]:
    """Return the canonical stat name for a matched prose keyword, or None."""
    low = keyword.strip().lower()
    for pat, canon in _STAT_SYNONYMS:
        if re.fullmatch(pat, low, re.IGNORECASE):
            return canon
    return None


# --------------------------------------------------------------------------- #
# Extraction regexes
# --------------------------------------------------------------------------- #
# A gene token: 2-10 chars, starts with a letter, upper-case letters + digits.
# Real symbols look like ERBB2, LUM, COL1A1, MS4A1, CD3D, PECAM1.
_GENE_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

# A signed decimal number (no thousands separators expected in these stats).
_NUMBER = r"[-+]?\d+(?:\.\d+)?"

# A gene-shaped token used inside the stat-claim pattern (not anchored on \b at
# both ends here; the surrounding pattern anchors it).
_GENE_SHAPE = r"[A-Z][A-Z0-9]{1,9}"

# The stat keyword alternation, reused inside the stat-claim pattern.
_STAT_ALT = "|".join(f"(?:{pat})" for pat, _ in (
    (r"max[\s_-]*gg[\s_-]*corr", "max_gg_corr"),
    (r"max[\s_-]*gc[\s_-]*corr", "max_gc_corr"),
    (r"gene[\s_-]*gene\s+corr(?:elation)?", "max_gg_corr"),
    (r"gene[\s_-]*cluster\s+corr(?:elation)?", "max_gc_corr"),
    (r"glm[\s_-]*coef(?:ficient)?", "glm_coef"),
    (r"glm", "glm_coef"),
    (r"coef(?:ficient)?", "glm_coef"),
    (r"pearson", "pearson"),
))

# The stat-claim pattern: a gene token, then (within a short gap of connective
# words like "at"/"of"/"="/":") a stat keyword, then (within a short gap) a
# number. This models "LUM glm 17.998", "LUM at glm_coef 17.998",
# "LUM pearson = 0.91" while refusing to reach across a whole sentence into a
# trailing PMID. The gaps allow a few connective tokens but not arbitrary prose.
_GAP = r"(?:[\s:=,()\-]|of|at|is|was|=)*"
# The gene group is case-SENSITIVE (real symbols are upper-case: ERBB2, LUM),
# via the inline (?-i:...) flag, so lowercase connective words ("with", "at")
# never match as a gene. The stat keyword and gap stay case-insensitive.
_STAT_CLAIM_RE = re.compile(
    rf"(?-i:\b(?P<gene>{_GENE_SHAPE})\b)"
    rf"{_GAP}"
    rf"(?P<stat>{_STAT_ALT})"
    rf"{_GAP}"
    rf"(?P<value>{_NUMBER})\b",
    re.IGNORECASE,
)

# PMID: "PMID:12345678", "PMID 12345678", "PMID12345678".
_PMID_RE = re.compile(r"\bPMID[:\s]*?(\d{4,9})\b", re.IGNORECASE)

# DOI: 10.xxxx/....  (stop at whitespace, closing bracket/paren, or trailing punct)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\)\]\}<>\"]+)", re.IGNORECASE)

# Lab-note reference: [note:some_id]  or  [note: some_id]
_NOTE_REF_RE = re.compile(r"\[note:\s*([A-Za-z0-9_.\-]+)\s*\]")

# Claim reference: [claim:curated:marks:LUM:Stromal]. The id namespace is the
# claim store's, and ``trace.ref`` already spells a claim anchor "claim:<id>", so
# the bracketed prose form is the same string the decision trail carries.
_CLAIM_REF_RE = re.compile(r"\[claim:\s*([^\]\s][^\]]*?)\s*\]")

# Reference-graph edge: [edge:gene:LUM marks cell_type:Stromal]. The payload is
# exactly ``str(Edge)`` -- subject, predicate, object in canonical entity
# spelling -- so an edge citation resolves by identity and not by re-parsing.
_EDGE_REF_RE = re.compile(r"\[edge:\s*([^\]\s][^\]]*?)\s*\]")

# --------------------------------------------------------------------------- #
# The panel-absence rule, in prose
# --------------------------------------------------------------------------- #
# "A negative observation is evidence only if the measurement was CAPABLE of
# producing a positive one" (CLAUDE.md). A sentence that asserts an OFF-PANEL
# entity is absent AND draws a conclusion from that absence violates the rule,
# and it does so with no number in it for the numeric pass to anchor on.
#
# The load-bearing guard is NOT the word lists: it is
# ``SourceIndex.is_on_panel`` / ``is_modeled``. The pass fires only for a token
# the capability frame says was never measured here, so every sentence about an
# on-panel gene -- the overwhelming majority, including the productive
# TESTED_ABSENT half the rule exists to protect -- is out of its reach by
# construction.
#
# The exemption below is not an ad-hoc softener either. By the project's own
# rule the ONLY legitimate sentence about an off-panel entity's absence is one
# that says it was never measured, so "says it was never measured" is the rule's
# own escape hatch rather than a second opinion about English.
#
# ponytail: word lists over a sentence. The ceiling is an uppercase non-gene
# abbreviation (a histology term, say) written in an absence-plus-inference
# sentence -- it is off-panel too, so it fires. Upgrade path is a real symbol
# universe to gate the token on; there is none in this repo today.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")

_ABSENCE_RE = re.compile(
    r"\b(?:not\s+detected|undetected|not\s+expressed|no\s+expression"
    r"|absen(?:t|ce)|not\s+observed|not\s+seen|not\s+found|not\s+reported"
    r"|no\s+signal|missing|lack(?:s|ing|ed)?|negative)\b",
    re.IGNORECASE,
)

_INFERENCE_RE = re.compile(
    r"\b(?:argues?|arguing|rules?\s+out|ruling\s+out|excludes?|excluding"
    r"|rules?\s+against|weighs?\s+against|counts?\s+against|against\s+(?:a|an|the)"
    r"|supports?|supporting|confirms?|contradicts?|inconsistent\s+with"
    r"|consistent\s+with|unlikely|therefore|hence|which\s+is\s+why"
    r"|so\s+(?:this|it|the|c\d))\b",
    re.IGNORECASE,
)

_NEVER_MEASURED_RE = re.compile(
    r"\b(?:never\s+measured|not\s+measured|was\s+not\s+measured|never_measured"
    r"|off[-\s]?panel|not\s+on\s+the\s+panel|not\s+in\s+the\s+panel"
    r"|outside\s+the\s+(?:panel|frame)|out\s+of\s+frame|not\s+in\s+frame"
    r"|not\s+in\s+the\s+(?:capability\s+)?frame|no\s+marker\s+row"
    r"|argues?\s+nothing|says?\s+nothing|nothing\s+follows|tells\s+us\s+nothing"
    r"|no\s+evidence|not\s+evidence|cannot\s+argue|does\s+not\s+argue"
    r"|uninformative|not\s+informative|underpowered|below\s+the\s+floor)\b",
    re.IGNORECASE,
)

# English/stat words that tokenise as upper-case gene-shaped tokens but are NOT
# genes. These are excluded from gene extraction so prose like "AND", "THE",
# "PMID" never triggers a false marker binding. We do NOT rely on this list to
# decide validity — a token is a valid gene ONLY if it resolves against
# agent.data / the panel. This set only suppresses obvious noise so the
# gene-number binding does not attach a number to a non-gene word.
_STOPWORD_TOKENS: frozenset[str] = frozenset(
    {
        "PMID", "DOI", "GLM", "CV", "FDR", "BH", "DE", "ID", "OK", "NO", "AND",
        "THE", "FOR", "NOT", "BUT", "ALL", "ANY", "ARE", "WAS", "HAS", "CSV",
        "UMAP", "MCP", "NCBI", "TSV", "JSON", "URL", "API", "TRUE", "FALSE",
        "NOSIG", "HIGH", "LOW", "VERY", "P", "R", "N", "UM", "SE", "TME",
    }
)


# --------------------------------------------------------------------------- #
# Result / violation dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Violation:
    """One grounding failure, with enough detail to explain the rejection."""

    kind: str          # "number" | "citation" | "note" | "marker" | "unverifiable"
    severity: str      # "CRITICAL" | "HIGH"
    detail: str        # human-readable explanation
    ref: str = ""      # gene | pmid | doi | note_id
    claimed: str = ""  # what the prose stated
    expected: str = "" # what the source actually holds (if known)


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of a grounding check. ``ok`` is False iff any violation exists."""

    ok: bool
    violations: tuple[Violation, ...] = ()

    def summary(self) -> str:
        if self.ok:
            return "grounded: every claim traced to source"
        lines = [f"{len(self.violations)} grounding violation(s):"]
        for v in self.violations:
            lines.append(f"  [{v.severity}] {v.kind}: {v.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SourceIndex — the single numeric source of truth (wraps agent.data)
# --------------------------------------------------------------------------- #
class SourceIndex:
    """Thin index over :mod:`agent.data`. Never parses marker files itself.

    Everything numeric is resolved through ``agent.data.get_marker`` so there is
    exactly one place the real jazzPanda values come from.
    """

    # -- gene identity ------------------------------------------------------ #
    def is_modeled(self, gene: str) -> bool:
        """True iff jazzPanda has a marker row for ``gene`` (case-insensitive)."""
        return data.get_marker(gene) is not None

    def is_on_panel(self, gene: str) -> bool:
        """True iff ``gene`` is on the measured panel (the absence primitive)."""
        try:
            return data.panel_contains(gene)
        except Exception:
            return False

    def gene_exists(self, gene: str) -> bool:
        """A token counts as a real gene if it is modeled OR on the panel."""
        return self.is_modeled(gene) or self.is_on_panel(gene)

    def panel_known(self) -> bool:
        """True iff the capability frame could actually be read.

        ``is_on_panel`` answers False when the panel file cannot be loaded, which
        is the right answer for a membership question and the WRONG premise for
        the panel-absence pass: with no panel, every token is "off-panel" and the
        pass would reject every absence sentence in the answer. So the pass asks
        this first and declines to run rather than refuse on a frame it does not
        have. Panel size is per study and derived from the file, never asserted.
        """
        try:
            return len(data.load_panel()) > 0
        except Exception:
            return False

    # -- numeric grounding -------------------------------------------------- #
    def stat_value(self, gene: str, stat: str) -> Optional[float]:
        """Return the real value of ``stat`` for ``gene``, or None if unavailable."""
        if stat not in _GROUNDABLE_STATS:
            return None
        row = data.get_marker(gene)
        if row is None:
            return None
        if stat not in row.index:
            return None
        try:
            return float(row[stat])
        except (TypeError, ValueError):
            return None

    def number_matches(self, gene: str, stat: str, value: float) -> bool:
        """True iff prose ``value`` matches the real ``gene.stat`` within tolerance.

        A claim about a stat we cannot resolve (unknown gene, non-groundable
        stat, missing column) returns False — the caller treats that as an
        unverifiable claim and fails closed.
        """
        truth = self.stat_value(gene, stat)
        if truth is None:
            return False
        return _close(value, truth)


def _close(a: float, b: float) -> bool:
    """Absolute-or-relative tolerance match (handles rounded prose numbers)."""
    diff = abs(a - b)
    if diff <= _NUMBER_ABS_TOL:
        return True
    scale = max(abs(a), abs(b), 1e-9)
    return diff / scale <= _NUMBER_REL_TOL


# --------------------------------------------------------------------------- #
# Extracted-claim dataclasses (internal)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NumberClaim:
    gene: str
    stat: Optional[str]   # canonical stat name, or None if no stat keyword bound
    value: float
    text: str             # the raw prose fragment, for diagnostics


@dataclass(frozen=True)
class AbsenceClaim:
    """One sentence that asserts an entity is absent AND concludes from that."""

    entity: str
    sentence: str


@dataclass(frozen=True)
class ExtractedClaims:
    genes: tuple[str, ...]
    numbers: tuple[NumberClaim, ...]
    pmids: tuple[str, ...]
    dois: tuple[str, ...]
    note_refs: tuple[str, ...]
    claim_refs: tuple[str, ...] = ()
    edge_refs: tuple[str, ...] = ()
    absences: tuple[AbsenceClaim, ...] = ()


# --------------------------------------------------------------------------- #
# Extractor — pulls claims FROM THE PROSE (sidecar is a hint only)
# --------------------------------------------------------------------------- #
class Extractor:
    """Extract genes, gene-bound numbers, PMIDs/DOIs and note refs from prose.

    The sidecar (if given) is used ONLY to widen the gene vocabulary so a marker
    named in the manifest is recognised even when the prose casing is odd. It is
    NEVER used to skip a prose number: everything checkable is pulled from the
    prose so a poisoned number that is absent from the sidecar still surfaces.
    """

    def __init__(self, source: SourceIndex) -> None:
        self._src = source

    def extract(
        self, text: str, sidecar_markers: tuple[str, ...] = ()
    ) -> ExtractedClaims:
        # Mask citation/note spans so their digits never become numeric claims
        # (a PMID's digits must never be read as a "gene stat").
        masked = _mask_spans(
            text, (_PMID_RE, _DOI_RE, _NOTE_REF_RE, _CLAIM_REF_RE, _EDGE_REF_RE)
        )

        numbers = tuple(self._extract_stat_claims(masked))
        genes = tuple(dict.fromkeys(self._candidate_genes(masked)))
        pmids = tuple(dict.fromkeys(_PMID_RE.findall(text)))
        dois = tuple(dict.fromkeys(m.rstrip(".,;)") for m in _DOI_RE.findall(text)))
        note_refs = tuple(dict.fromkeys(_NOTE_REF_RE.findall(text)))
        claim_refs = tuple(dict.fromkeys(_CLAIM_REF_RE.findall(text)))
        edge_refs = tuple(dict.fromkeys(_EDGE_REF_RE.findall(text)))

        return ExtractedClaims(
            genes=genes,
            numbers=numbers,
            pmids=pmids,
            dois=dois,
            note_refs=note_refs,
            claim_refs=claim_refs,
            edge_refs=edge_refs,
            absences=tuple(self._extract_absences(masked)),
        )

    # -- absence-used-as-evidence sentences --------------------------------- #
    def _extract_absences(self, text: str) -> list[AbsenceClaim]:
        """Entities whose ABSENCE a sentence asserts and then reasons from.

        Extraction only: this says nothing about whether the entity was in the
        capability frame. :meth:`GroundingChecker._check_panel_absence` is what
        consults the frame, because that is the half of the rule that decides,
        and keeping it out of here is what stops the word lists from being the
        thing that rejects an answer.
        """
        out: list[AbsenceClaim] = []
        seen: set[str] = set()
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            if not _ABSENCE_RE.search(sentence):
                continue
            if not _INFERENCE_RE.search(sentence):
                continue
            if _NEVER_MEASURED_RE.search(sentence):
                # The sentence names the absence as unmeasured -- which is the
                # rule being OBEYED, loudly. Nothing to flag.
                continue
            for m in _GENE_TOKEN_RE.finditer(sentence):
                tok = m.group(1)
                if tok in _STOPWORD_TOKENS or tok in seen:
                    continue
                seen.add(tok)
                out.append(AbsenceClaim(entity=tok, sentence=sentence.strip()))
        return out

    # -- gene tokens (names actually asserted as real genes) ---------------- #
    def _candidate_genes(self, text: str) -> list[str]:
        """Gene-shaped tokens that resolve as real genes (modeled or on-panel).

        Used for the reported ``genes`` vocabulary. It is NOT the gate for
        numeric claims — a stat claim on an UNKNOWN gene is still extracted (see
        :meth:`_extract_stat_claims`) so a fabricated marker with a fabricated
        number is caught rather than silently dropped.
        """
        out: list[str] = []
        for m in _GENE_TOKEN_RE.finditer(text):
            tok = m.group(1)
            if tok in _STOPWORD_TOKENS:
                continue
            if self._src.gene_exists(tok):
                out.append(tok)
        return out

    # -- gene + stat + number claims ---------------------------------------- #
    def _extract_stat_claims(self, text: str) -> list[NumberClaim]:
        """Extract every "GENE <stat> NUMBER" triple.

        Uses one adjacency pattern so a number is bound to a stat only when the
        stat keyword sits directly between the gene and the number (separated by
        connective tokens, not a whole sentence). The gene need NOT resolve —
        an unknown/off-panel gene carrying a jazzPanda stat is a fabrication the
        downstream checker must reject, so we surface it here. Stopword tokens
        (PMID, GLM, THE, ...) are never treated as the gene.
        """
        claims: list[NumberClaim] = []
        seen: set[tuple[str, str, float, int]] = set()
        for m in _STAT_CLAIM_RE.finditer(text):
            gene = m.group("gene")
            if gene in _STOPWORD_TOKENS:
                continue
            stat = _canonical_stat(m.group("stat"))
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            key = (gene, stat or "", value, m.start())
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                NumberClaim(
                    gene=gene,
                    stat=stat,
                    value=value,
                    text=m.group(0),
                )
            )
        return claims


# --------------------------------------------------------------------------- #
# LiteratureVerifier — wraps the injected resolver callable
# --------------------------------------------------------------------------- #
class LiteratureVerifier:
    """Resolve PMIDs/DOIs through an injected callable.

    The callable takes an identifier string and returns a truthy value iff the
    identifier resolves to a REAL literature record (via a live connector or a
    frozen-real cassette). If no callable is injected, EVERY identifier is
    unresolved — we never assume a citation is real. Fail closed.
    """

    def __init__(self, resolver: Optional[Callable[[str], bool]] = None) -> None:
        self._resolver = resolver

    def resolves(self, ident: str) -> bool:
        if self._resolver is None:
            return False
        try:
            return bool(self._resolver(ident))
        except Exception:
            # A resolver that errors cannot vouch for the citation -> unresolved.
            return False


# --------------------------------------------------------------------------- #
# GroundingChecker — the gate
# --------------------------------------------------------------------------- #
class GroundingChecker:
    """Reject any answer that states something not traceable to source.

    Parameters
    ----------
    literature_verifier:
        Callable ``(identifier) -> bool`` that resolves a PMID/DOI to a real
        record. Injected so tests (cassette) and the live loop share one gate.
    source:
        Optional :class:`SourceIndex`; a fresh one is built if omitted.
    known_notes:
        Optional callable ``() -> set[str]`` returning the ids of existing lab
        notes, or a set. Used to confirm a referenced note actually exists. When
        it is omitted the gate reads the REAL note store instead, so the note
        existence and note scope passes fire in production rather than only for
        a test that remembered to pass a parameter.
    claims:
        Optional claim store (anything with ``cite(claim_id, context)``). A
        ``[claim:id]`` in prose is resolved through it, so the store's own
        fail-closed refusals -- missing, retracted, out of scope -- are the ones
        the gate reports. With no store, a claim citation cannot be resolved and
        is a violation: fail closed, never open.
    claim_context:
        The scope context ``[claim:id]`` citations are resolved in; defaults to
        this install's dataset.
    graph:
        Optional reference graph (anything with an ``edges`` tuple). An
        ``[edge:...]`` citation must match one of its edges by identity.
    """

    def __init__(
        self,
        literature_verifier: Optional[Callable[[str], bool]] = None,
        *,
        source: Optional[SourceIndex] = None,
        known_notes: Optional[Union[Callable[[], set[str]], set[str]]] = None,
        claims: Optional[Any] = None,
        claim_context: Optional[str] = None,
        graph: Optional[Any] = None,
    ) -> None:
        self._src = source or SourceIndex()
        self._extractor = Extractor(self._src)
        self._lit = LiteratureVerifier(literature_verifier)
        self._known_notes = known_notes
        self._claims = claims
        self._claim_context = claim_context or f"dataset:{cfg.DATASET_ID}"
        self._graph = graph

    # -- public API --------------------------------------------------------- #
    def check(
        self,
        answer_text: str,
        sidecar=None,
        cluster_ctx: Optional[str] = None,
        *,
        allowed_notes: Optional[set[str]] = None,
    ) -> GroundingResult:
        """Verify ``answer_text``. ``sidecar`` is a localization aid, not a
        whitelist. Returns a :class:`GroundingResult`; ``ok`` iff no violations.

        ``allowed_notes`` (if given) is the set of note ids in scope for this
        turn — a referenced note outside this set is a violation even if the
        note file exists elsewhere. ``cluster_ctx`` is accepted for interface
        symmetry and future per-cluster checks; numeric grounding does not
        depend on it because ``get_marker`` is keyed by gene.
        """
        text = answer_text or ""
        sidecar_markers = _sidecar_markers(sidecar)
        claims = self._extractor.extract(text, sidecar_markers)

        violations: list[Violation] = []
        violations += self._check_numbers(claims)
        violations += self._check_markers(claims)
        violations += self._check_citations(claims)
        violations += self._check_notes(claims, cluster_ctx, allowed_notes)
        violations += self._check_claim_refs(claims)
        violations += self._check_edge_refs(claims)
        violations += self._check_panel_absence(claims)

        return GroundingResult(ok=not violations, violations=tuple(violations))

    def check_record(
        self,
        record: Any,
        *,
        allowed_notes: Optional[set[str]] = None,
    ) -> GroundingResult:
        """Gate the GENERATED layer of an :class:`InterpretationRecord`.

        Every other field on a record is deterministic output from ``core/`` and
        is hashed; the fields :data:`~panoscope.types.record.GENERATED_FIELDS`
        names are the ones a model may have written, and they are the ones gated
        here. The field list is read off the TYPE rather than spelled ``"prose"``
        in this module, for the reason ``types/record.py`` gives: a gate that
        hardcodes the name silently stops covering the second generated field the
        day one is added.

        A record whose generated fields are all ``None`` is complete and valid --
        it is what the demo falls back to -- and passes with nothing to check.
        """
        violations: list[Violation] = []
        for field in GENERATED_FIELDS:
            value = getattr(record, field, None)
            if value is None:
                continue
            result = self.check(
                str(value),
                None,
                getattr(record, "unit", None),
                allowed_notes=allowed_notes,
            )
            violations += [
                replace(v, detail=f"{field}: {v.detail}") for v in result.violations
            ]
        return GroundingResult(ok=not violations, violations=tuple(violations))

    # -- numeric claims ----------------------------------------------------- #
    def _check_numbers(self, claims: ExtractedClaims) -> list[Violation]:
        out: list[Violation] = []
        for c in claims.numbers:
            if c.stat is None:
                # A bare number next to a gene with no stat keyword is not a
                # statistical claim we can (or should) bind — skip it. Only
                # gene+stat+number triples are grounded.
                continue
            truth = self._src.stat_value(c.gene, c.stat)
            if truth is None:
                out.append(
                    Violation(
                        kind="unverifiable",
                        severity="CRITICAL",
                        detail=(
                            f"claim about {c.gene} {c.stat}={c.value} cannot be "
                            f"resolved against jazzPanda output"
                        ),
                        ref=c.gene,
                        claimed=f"{c.stat}={c.value}",
                    )
                )
                continue
            if not _close(c.value, truth):
                out.append(
                    Violation(
                        kind="number",
                        severity="CRITICAL",
                        detail=(
                            f"{c.gene} {c.stat} stated as {c.value} but jazzPanda "
                            f"has {truth}"
                        ),
                        ref=c.gene,
                        claimed=f"{c.stat}={c.value}",
                        expected=f"{c.stat}={truth}",
                    )
                )
        return out

    # -- marker existence --------------------------------------------------- #
    def _check_markers(self, claims: ExtractedClaims) -> list[Violation]:
        """A gene cited WITH a jazzPanda statistic must be MODELED (have a row).

        Panel-only genes are legitimate to name for the panel-absence rule, but
        stating a jazzPanda number for a gene with no marker row is fabrication.
        The numeric pass already catches the unresolved value; this pass names
        the root cause (no marker row) so the audit trail is clear.
        """
        out: list[Violation] = []
        stat_genes = {c.gene for c in claims.numbers if c.stat is not None}
        for gene in stat_genes:
            if not self._src.is_modeled(gene):
                out.append(
                    Violation(
                        kind="marker",
                        severity="CRITICAL",
                        detail=(
                            f"{gene} is cited with a jazzPanda statistic but has "
                            f"no marker row (off-panel absence is not a number)"
                        ),
                        ref=gene,
                    )
                )
        return out

    # -- citations ---------------------------------------------------------- #
    def _check_citations(self, claims: ExtractedClaims) -> list[Violation]:
        out: list[Violation] = []
        for pmid in claims.pmids:
            if not self._lit.resolves(pmid):
                out.append(
                    Violation(
                        kind="citation",
                        severity="CRITICAL",
                        detail=f"PMID:{pmid} does not resolve to a real record",
                        ref=pmid,
                        claimed=f"PMID:{pmid}",
                    )
                )
        for doi in claims.dois:
            if not self._lit.resolves(doi):
                out.append(
                    Violation(
                        kind="citation",
                        severity="CRITICAL",
                        detail=f"DOI {doi} does not resolve to a real record",
                        ref=doi,
                        claimed=doi,
                    )
                )
        return out

    # -- lab notes ---------------------------------------------------------- #
    def _check_notes(
        self,
        claims: ExtractedClaims,
        cluster_ctx: Optional[str],
        allowed_notes: Optional[set[str]],
    ) -> list[Violation]:
        if not claims.note_refs:
            return []
        existing, allowed = self._note_universe(cluster_ctx, allowed_notes)
        out: list[Violation] = []
        for note_id in claims.note_refs:
            if existing is not None and note_id not in existing:
                out.append(
                    Violation(
                        kind="note",
                        severity="CRITICAL",
                        detail=f"referenced lab note [note:{note_id}] does not exist",
                        ref=note_id,
                    )
                )
                continue
            if allowed is not None and note_id not in allowed:
                out.append(
                    Violation(
                        kind="note",
                        severity="HIGH",
                        detail=(
                            f"lab note [note:{note_id}] is out of scope for this "
                            f"answer"
                        ),
                        ref=note_id,
                    )
                )
        return out

    def _note_universe(
        self, cluster_ctx: Optional[str], allowed_notes: Optional[set[str]]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """``(notes that exist, notes in scope here)`` -- from ONE source.

        Either the caller owns the note universe (it injected ``known_notes``,
        and then ``allowed_notes`` is its business too) or the gate reads the
        real store for both halves. Mixing an injected existence set with a
        filesystem-derived scope set would answer two questions from two
        different worlds, and the answer to the second would be noise.

        The real half is ``memory.apply_notes`` -- the single scope gate every
        other caller already goes through (cluster-scoped notes fire only for
        their own cluster; dataset and lab notes fire everywhere) -- read from
        the same base dir the memory tools write to. It is what makes "a claim
        used outside its scope is an error" true of agent output in production
        and not only under test.
        """
        if self._known_notes is not None:
            return self._resolve_known_notes(), allowed_notes
        try:
            from panoscope.agent import memory
            from panoscope.agent import tools as agent_tools

            base = agent_tools.memory_base_dir()
            existing = {n.id for n in memory.read_notes(base)}
            if allowed_notes is None:
                allowed_notes = {
                    n.id for n in memory.apply_notes(cluster_ctx, base_dir=base)
                }
            return existing, allowed_notes
        except Exception:
            # The store could not be read AT ALL -- which is not the same as it
            # being empty (an empty store is read fine and rejects every note
            # reference). Nothing can be concluded, so neither half is asserted.
            return None, allowed_notes

    def _resolve_known_notes(self) -> Optional[set[str]]:
        if self._known_notes is None:
            return None
        if callable(self._known_notes):
            try:
                return set(self._known_notes())
            except Exception:
                return None
        return set(self._known_notes)

    # -- claim citations ---------------------------------------------------- #
    def _check_claim_refs(self, claims: ExtractedClaims) -> list[Violation]:
        """A ``[claim:id]`` must resolve through the store, IN THIS SCOPE.

        The resolution is ``ClaimStore.cite``, not a membership test: cite is the
        USE gate, and its three refusals are the three ways a cited claim can be
        wrong without looking wrong in the rendered sentence -- it does not
        exist, it was retracted, or it belongs to another scope. Re-implementing
        any of those here would be a second, weaker opinion about the same rule.
        """
        if not claims.claim_refs:
            return []
        out: list[Violation] = []
        for cid in claims.claim_refs:
            if self._claims is None:
                out.append(Violation(
                    kind="claim",
                    severity="CRITICAL",
                    detail=(
                        f"[claim:{cid}] cannot be resolved: no claim store is "
                        f"wired into this gate, so the citation is unverifiable"
                    ),
                    ref=cid,
                    claimed=f"claim:{cid}",
                ))
                continue
            try:
                self._claims.cite(cid, self._claim_context)
            except KeyError:
                out.append(Violation(
                    kind="claim",
                    severity="CRITICAL",
                    detail=f"[claim:{cid}] does not exist in the claim store",
                    ref=cid,
                    claimed=f"claim:{cid}",
                ))
            except ValueError as exc:
                # OutOfScope and Retracted are both ValueError; the store names
                # which one and why, and that sentence is the audit trail.
                out.append(Violation(
                    kind="claim",
                    severity="CRITICAL",
                    detail=f"[claim:{cid}] refused by the claim store: {exc}",
                    ref=cid,
                    claimed=f"claim:{cid}",
                ))
        return out

    # -- reference-graph edges ---------------------------------------------- #
    def _check_edge_refs(self, claims: ExtractedClaims) -> list[Violation]:
        """An ``[edge:...]`` must match an edge of the consulted graph by identity.

        Matched on ``str(Edge)`` -- ``subject predicate object`` in canonical
        entity spelling -- rather than re-parsed into parts here, so a citation
        that resolves is a citation to an edge that was actually read, and a
        reference source still has no channel through which a value could reach
        the answer.
        """
        if not claims.edge_refs:
            return []
        known = {str(e) for e in getattr(self._graph, "edges", ()) or ()}
        out: list[Violation] = []
        for spec in claims.edge_refs:
            if spec in known:
                continue
            out.append(Violation(
                kind="edge",
                severity="CRITICAL",
                detail=(
                    f"[edge:{spec}] is not an edge of the consulted reference "
                    f"graph" + ("" if self._graph is not None else
                                " (no graph is wired into this gate)")
                ),
                ref=spec,
                claimed=f"edge:{spec}",
            ))
        return out

    # -- the panel-absence rule, in prose ----------------------------------- #
    def _check_panel_absence(self, claims: ExtractedClaims) -> list[Violation]:
        """Absence used as evidence for something that was never measured.

        CLAUDE.md's headline rule: "The absence of an out-of-frame entity is not
        evidence against anything." The numeric passes cannot see this one -- the
        sentence carries no number -- so the frame is consulted directly.

        Only an entity the frame says was NEVER MEASURED is flagged. A gene that
        is on the panel and was tested and not reported is ``TESTED_ABSENT``, and
        its absence is the productive half of the lattice: it argues, and this
        pass must never touch it.
        """
        if not claims.absences or not self._src.panel_known():
            return []
        out: list[Violation] = []
        for a in claims.absences:
            if self._src.is_on_panel(a.entity) or self._src.is_modeled(a.entity):
                continue
            out.append(Violation(
                kind="panel_absence",
                severity="CRITICAL",
                detail=(
                    f"{a.entity} is not in the capability frame (never measured "
                    f"here), so its absence is not evidence for or against "
                    f"anything; the sentence reasons from it anyway"
                ),
                ref=a.entity,
                claimed=a.sentence,
                expected=f"{a.entity} NEVER_MEASURED: absence_is_evidence = False",
            ))
        return out


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _mask_spans(text: str, patterns) -> str:
    """Replace matched spans with spaces (same length) so their digits are not
    re-read as numeric claims, while keeping every other offset stable."""
    if not text:
        return text
    chars = list(text)
    for pat in patterns:
        for m in pat.finditer(text):
            for i in range(m.start(), m.end()):
                chars[i] = " "
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Sidecar helpers
# --------------------------------------------------------------------------- #
def _sidecar_markers(sidecar) -> tuple[str, ...]:
    """Best-effort read of marker names from a sidecar (localization aid only)."""
    if sidecar is None:
        return ()
    markers = getattr(sidecar, "markers", None)
    if markers is None and isinstance(sidecar, dict):
        markers = sidecar.get("markers")
    if not markers:
        return ()
    return tuple(str(m) for m in markers)
