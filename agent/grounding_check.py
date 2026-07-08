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
- Every referenced lab note must exist and be cited.

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
from dataclasses import dataclass
from typing import Callable, Optional, Union

from agent import config as cfg
from agent import data

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
class ExtractedClaims:
    genes: tuple[str, ...]
    numbers: tuple[NumberClaim, ...]
    pmids: tuple[str, ...]
    dois: tuple[str, ...]
    note_refs: tuple[str, ...]


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
        masked = _mask_spans(text, (_PMID_RE, _DOI_RE, _NOTE_REF_RE))

        numbers = tuple(self._extract_stat_claims(masked))
        genes = tuple(dict.fromkeys(self._candidate_genes(masked)))
        pmids = tuple(dict.fromkeys(_PMID_RE.findall(text)))
        dois = tuple(dict.fromkeys(m.rstrip(".,;)") for m in _DOI_RE.findall(text)))
        note_refs = tuple(dict.fromkeys(_NOTE_REF_RE.findall(text)))

        return ExtractedClaims(
            genes=genes,
            numbers=numbers,
            pmids=pmids,
            dois=dois,
            note_refs=note_refs,
        )

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
        notes, or a set. Used to confirm a referenced note actually exists.
    """

    def __init__(
        self,
        literature_verifier: Optional[Callable[[str], bool]] = None,
        *,
        source: Optional[SourceIndex] = None,
        known_notes: Optional[Union[Callable[[], set[str]], set[str]]] = None,
    ) -> None:
        self._src = source or SourceIndex()
        self._extractor = Extractor(self._src)
        self._lit = LiteratureVerifier(literature_verifier)
        self._known_notes = known_notes

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
        violations += self._check_notes(claims, allowed_notes)

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
        self, claims: ExtractedClaims, allowed_notes: Optional[set[str]]
    ) -> list[Violation]:
        if not claims.note_refs:
            return []
        existing = self._resolve_known_notes()
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
            if allowed_notes is not None and note_id not in allowed_notes:
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

    def _resolve_known_notes(self) -> Optional[set[str]]:
        if self._known_notes is None:
            return None
        if callable(self._known_notes):
            try:
                return set(self._known_notes())
            except Exception:
                return None
        return set(self._known_notes)


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
