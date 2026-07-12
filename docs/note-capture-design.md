# Note capture — design spec

Status: **design locked (2026-07-10); built on `feat/discriminator-and-interpretation-summary`
(steps 1–6, note-capture commits).** This remains the spec of record for how the lab's knowledge
is captured as structured notes; the `## 11. Suggested build order` below is now largely done.

Grounds in `CLAUDE.md`: the confident floor (never fabricate; every claim cites a real source),
and *"Memory is a reconciliation layer, not a memory of the user… the value is in the disagreement…
capture at override, ask scope and basis in two taps, not a form… the biologist decides, always."*

---

## 1. The problem

A biologist says **eight kinds of things worth remembering** in an annotation session. Today the
tool captures **one** — a full cell-type override (`agent` proposes `note_draft` → the two-tap
`_render_draft_card` confirm → fail-closed save). The other seven evaporate on cluster switch, and
the **Pathways / enrichment chat has no note capture at all** (its loop has no `memory_draft` tool).

The fix is **not a new subsystem**: widen the one path that already works
(`draft_note → reconcile → confirm card → apply_notes`) to cover all eight, reusing the same
two-tap card, the same literature cross-check, the same fail-closed scope gate.

This also resolves two earlier concerns:
- The free-text Summary editor feels "dumb" → knowledge enters as typed notes; the Summary becomes a
  **composed view** of grounded facts + notes (the current free-text editors are interim).
- The edit-vs-agent **conflict** dissolves → one knowledge store (notes), the report is *derived*, so
  a note from chat re-composes the report with its tension visible. Nothing to clobber, either way.

## 2. The note taxonomy (what to capture)

| # | Trigger utterance (example) | `type` | Anchor | Default scope | Basis hint | Literature check |
|---|---|---|---|---|---|---|
| 1 | "It's CAF, not stroma" *(changes the call)* | `celltype_override` *(exists)* | the call | cluster | infer: "our data"→own_validation, "we always"→convention, else paper | **Yes** — asserted call vs drivers |
| 2 | "POSTN here is tumor bleed — down-weight it" *(call unchanged)* | `marker_reinterpretation` | a **gene** | cluster (dataset if section-wide) | paper / own_validation | **Yes** — the gene's meaning in this context |
| 3 | "This EMT program is co-infiltration, not theirs" | `program_reinterpretation` | a **gene set** | cluster | paper / convention | **Yes** — gene-set + leading-edge biology |
| 4 | "KRT7 is unreliable in this panel — never trust it" | `marker_convention` | a **gene** (no cluster) | dataset / lab | convention | **Yes** — marker reliability/specificity; say if thin |
| 5 | "We confirmed c4 by p63 IHC" | `validation` | the call | cluster (dataset if assay covered section) | own_validation, status=firm | **Yes** — literature is the *tension*, the assay is the *basis* |
| 6 | "CPA3 is enough — I'm comfortable at High" *(numbers unchanged)* | `confidence_adjustment` | the call | cluster | the stated reason | **Yes** — over/under-confidence vs literature |
| 7 | "Exclude c8, it's a doublet" | `exclude` | the cluster | cluster | own_validation / convention, firm | **Light** — a QC/spatial judgment; never block save on a thin lookup |
| 8 | "c5 and c7 are one population" | `cross_cluster` | a **set of clusters** | dataset (surfaces on each) | own_validation / convention | **Light** — a clustering judgment about this dataset |

The two highest-value currently-lost moments are **#2 (marker reinterpretation)** — the everyday
"this number means something different here" — and **#3 (program reinterpretation)** — the entire
Pathways side that remembers nothing.

## 3. What we do NOT capture (silence is the default)

- **Questions** ("what does POSTN mark?", "why is this Very High?") — pulling info *out*, not putting
  judgment *in*. Answer it, cite it, capture nothing.
- **Bare acknowledgements** ("got it", "makes sense") — no new claim.
- **Mid-thought hedges with no landing** ("could be CAF, could be pericyte… let me look at the map") —
  no decision yet; freezing an un-made call violates "the biologist decides, always". Wait for the
  landing; if it resolves to a call, capture *that*.
- **Viewing controls** ("show UMAP", "pin LUM", "100µm bins") — a view/bin never changes a value, so it
  never mints a note (same rule as CLAUDE.md's viewing-controls).
- **Queries about memory** ("what did I say about c2?") — serve from the notes drawer; don't mint a
  note about asking for notes.
- **Verbatim duplicates** of an existing in-scope note — reinforce/timestamp (or offer *supersede* if
  it changed), don't fragment the layer.

Rule: **precision over recall.** When in doubt, do not draft — a missed note the biologist can
restate; a drawer of junk notes erodes trust in the whole layer.

## 4. The logic flow (same two taps, widened)

1. **Detect the divergence, not a keyword.** Widen the agent's existing override-judgment to classify
   which of the 8 types a turn is, and infer `{type, anchor, scope, basis, status}`. Questions,
   acknowledgements, view commands, and hedges never trip it. The Pathways loop gets this detection
   too (new wiring).
2. **Draft silently and reconcile FIRST.** The agent calls `memory_draft` with the inferred fields;
   `draft_note` runs the literature cross-check into a `Tension` **before anything renders**. Nothing
   hits disk. "The value is in the disagreement" holds for *every* type (a validation note still gets
   its literature agree/dissent; `exclude`/`cross_cluster` reconcile thin/optional and never block).
3. **Pre-fill the SAME two-tap card** (`_render_draft_card`): scope + basis + status controls, the
   tension with clickable real PMIDs, Save / Discard. The **only** addition is a one-line **editable
   header** showing the inferred type + anchor — e.g. `Marker note · POSTN · c2`,
   `Program note · HALLMARK_EMT · c2`, `Cross-cluster · c5 + c7` — so the biologist confirms the agent
   read the right subject. No new form, no new surface. Friction budget stays at two taps.
4. **Confirm-time edits are free** (no re-lookup): the tension is already attached, so changing scope
   cluster→dataset or basis paper→own_validation, or fixing the type/anchor, is metadata-only.
5. **Save is scope-enforced and fail-closed** — `apply_notes` stays the single choke point. Nothing
   persists on the model's classification; only the explicit **Save** tap writes one JSON note.
6. **Surface on next open, ANCHORED.** `_inscope_notes_block` already injects in-scope notes each turn;
   extend the opening so an anchored note renders **next to its subject**:
   - a `marker_reinterpretation` / `marker_convention` note as a caveat row beneath that gene's driver
     row (`POSTN — glm 15.8 · lab note: tumor-adjacent bleed, down-weight [note:id]`);
   - a `program_reinterpretation` note beside its gene-set row in the Pathways table;
   - a `validation` note beside the call, citable as `[note:id]` alongside the jazzPanda numbers;
   - a `confidence_adjustment` as the **dual band** `lab: High · computed Medium` (never overwriting
     the computed value);
   - an `exclude` greys the cluster header with the reason;
   - a `cross_cluster` note on **every** cluster in its anchor set (`same population as c7 [note:id]`).
   Every applied note shows `[note:id]` with its tension — cite-on-use, no silent application, ever.

## 5. Capture surfaces (all three)

Every surface writes through the **identical** `draft_note → save_draft → apply_notes` pipeline with
the identical two-tap card; only the entry point and the `trigger` literal differ. **Capture is never
automatic — the explicit Save tap is always required.**

- **Marker-genes chat** — the per-cluster marker chat (the one place capture works today). Widen to
  all types. `trigger = "override"` (or the specific type).
- **Pathways chat** — currently has **no memory tool at all**. Give the enrichment loop the same
  `memory_draft` tool and `enrichment_conversation` the same confirm card; anchor = gene set. This is
  the one place needing genuinely new wiring and the single largest missing capability.
- **Summary / holistic review** — when the holistic pass proposes a refinement (e.g. c8 → pDC,
  "exclude c8", "c5 + c7 are one") or the biologist accepts/edits it, route it through the same
  draft→confirm path with `trigger = "holistic_review"`.

## 6. The Note object changes (the only new shape)

Three additions to the frozen dataclasses in `agent/types.py`, **all backward-compatible**
(optional-with-defaults, so every existing note JSON still parses), none loosening the confident floor.
Mirror all three onto `NoteDraft` and carry through `save_draft` into `Note`.

```python
NoteType = Literal[
    "celltype_override", "marker_reinterpretation", "program_reinterpretation",
    "marker_convention", "validation", "confidence_adjustment", "exclude", "cross_cluster",
]

@dataclass(frozen=True)
class Note:
    id: str
    claim: str
    type: NoteType = "celltype_override"          # (1) NEW — default keeps old notes parsing
    scope: Scope                                   # cluster | dataset | lab
    scope_ref: ScopeRef                            # unchanged (single optional firing cluster)
    basis: Basis                                   # paper | own_validation | convention
    status: Status                                 # firm | tentative
    subject_cell_type: str = ""
    subject_markers: tuple[str, ...] = ()          # existing gene anchor
    subject_gene_sets: tuple[str, ...] = ()        # (2) NEW — the enrichment analog (HALLMARK_*)
    subject_clusters: tuple[str, ...] = ()         # (2) NEW — the anchor SET for cross_cluster
    tension: Tension
    author: str = ""
    created_at: str = ""
    trigger: Literal["override", "manual_add", "holistic_review"] = "override"
    supersedes: str = ""
```

- **`type`** is the enabling change: today `{claim, subject_markers=[POSTN]}` cannot say whether the
  note means "POSTN means contamination here" or "the whole call is wrong because of POSTN" — so the
  opening can't render them differently. `type` routes a note to render next to its anchor.
- **`subject_gene_sets`** reuses the existing `PathwayEvidence.gene_set` vocabulary.
- **`subject_clusters`** is the only genuinely new shape: `scope_ref.cluster` stays the single *firing*
  cluster for cluster-scope; `subject_clusters` is the set a dataset-scoped `cross_cluster` note
  surfaces on. `apply_notes`/`note_in_scope` gains one branch: a `cross_cluster` note fires when the
  requested cluster is in `subject_clusters`. **Do not** loosen `ScopeRef`.
- **`confidence_adjustment` stores NO numeric field.** The biologist's asserted label + reason live in
  `claim`/`status`; the UI renders the overlay `lab: High · computed Medium`. A field that could
  overwrite `confidence_score` would let a note fabricate a grounded number — forbidden.

## 7. Report composition & reconciliation

On cluster re-open the opening re-composes from three grounded sources, tension kept visible:
(a) the deterministic `OpeningInterpretation` (call, confidence, drivers with jazzPanda numbers,
off-panel notes, live citations) — the confident floor, unchanged; (b) the in-scope notes from
`apply_notes`, now **anchored** so each renders where its subject lives (§4.6); (c) every applied note
shown as `[note:id]` with its attached tension (agree/dissent PMIDs).

So the report = the computed floor + the lab's owned divergences rendered next to the exact rows they
modify + the disagreement between the biologist's call and the literature made **visible, not smoothed**.

**No conflict is possible:** notes are the single knowledge store; the report is derived. A note saved
in chat re-composes the report to include it (with tension). No stale free-text edit, no merge, no
silent overwrite in either direction.

## 8. Decisions locked (2026-07-10)

- **`exclude` applies at report composition.** A saved `exclude` note flips the exported `exclude`
  flag when the report/CSV is composed — **without mutating the deterministic jazzPanda verdict**
  underneath. This is the sanctioned exception (the one type with a real downstream effect), applied at
  composition, clearly attributed.
- **Capture from all three surfaces:** marker-genes chat, Pathways chat, and Summary/holistic review.
- **Build later** — this spec is the deliverable for now.

## 9. Guardrails (the confident floor holds)

- **Never fabricate a number through a note.** `confidence_adjustment` is overlay-only; no note may
  rewrite `confidence_score` or any jazzPanda statistic. *(Load-bearing.)*
- **Reconcile every type**, not just overrides — marker/program/convention/validation all get their
  literature agree/dissent; if literature is thin, say so honestly (never fake it).
- **Scope stays fail-closed** for the new anchors — a c2 marker note must never surface on c3;
  `apply_notes` is the single gate, no path bypasses it. Test the negative.
- **Two taps, never a form** — the agent infers type + anchor and pre-fills; the biologist confirms
  scope + basis (and can correct type/anchor inline). If a type needs more than the existing card, it
  is out of scope.
- **Cite on use, always** — a note is applied only if surfaced as `[note:id]` with its tension.
- **Panel-absence still governs** — a marker note about an off-panel gene must still say the gene was
  never measured; a note may not turn off-panel absence into evidence.

## 10. Explicitly NOT building

A notes-editor UI · free-text notes · auto-save without the confirm tap · any note that can mutate a
grounded number. The whole point: the lab owns **more** of its divergences with the **same** two-tap
friction and the **same** fail-closed, cite-on-use, tension-visible discipline that already governs the
override.

## 11. Suggested build order (when we build)

1. Add `type` + `subject_gene_sets` + `subject_clusters` to `Note`/`NoteDraft` (frozen,
   backward-compatible) and thread through `serialize`/`memory`/`apply_notes` (+ the `cross_cluster`
   firing branch, + anchored-dataset-convention firing where the gene is in evidence).
2. Widen the agent's override-detection to classify the 8 types + infer the anchor; add the editable
   type/anchor header to `_render_draft_card`.
3. Close the Pathways hole: give the enrichment loop `memory_draft` and `enrichment_conversation` the
   confirm card (anchor = gene set).
4. Anchored rendering in the opening (marker/program caveat rows, dual confidence band, exclude grey).
5. `exclude` applied at report composition (CSV + UI), not by mutating the verdict.
6. Route holistic-review refinements through the same draft→confirm path (`trigger=holistic_review`).
