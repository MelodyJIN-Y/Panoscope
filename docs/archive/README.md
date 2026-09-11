# Archive

Superseded documents, kept for provenance. **Nothing here is current.** If you are looking for how
Panoscope works, read `ARCHITECTURE.md` and `IMPLEMENTATION.md` at the repo root.

Archived 2026-09-11, at the point the project stopped being a hackathon artifact and became a
research tool targeted at publication.

| File | What it was | Superseded by |
|---|---|---|
| `ARCHITECTURE_v4.md` | The architecture of record, Sept 2026. Still the source of most of v5's structure — §2's core principle, the reconciliation split, task kinds, the interpretation record, and the refused list all survive largely intact | `ARCHITECTURE.md` (v5) |
| `IMPLEMENTATION_SPEC_v2.md` | Developer companion to v4 | `IMPLEMENTATION.md` (v3) |
| `BLUEPRINT.md` | The original build blueprint — file tree, type contracts, verdict pseudocode, build DAG. Already explicitly superseded by v4 | `ARCHITECTURE.md`, `IMPLEMENTATION.md` |
| `AGENTS.md` | "How Panoscope was built and how it reasons" — the hackathon build narrative, describing the agent team that produced the first version | nothing; the build story is not part of the tool's design |
| `HACKATHON_STRATEGY.md` | Hackathon positioning and demo strategy. Never tracked in git | nothing |
| `dashboard_wireframe_panels.html` | The original dashboard wireframe | `docs/interface-design.md` |

## Two things in here are actively wrong, not merely old

**`dashboard_wireframe_panels.html` builds the panel-absence story on a T-cell example.** That does
not work on this data: CD3D, CD3E, CD3G, CD4, CD8A and CD8B are all *on* the 280-gene panel, so
nothing about T cells demonstrates the rule. The real panel-absence spine is **c2 Stromal**, carried
by on-panel LUM (glm 18.00) and POSTN (15.80), with COL1A1, COL1A2, DCN, VIM, FAP and THY1 off-panel.
`HACKATHON_STRATEGY.md` repeats the same mistaken example. Do not rebuild any demo, figure or cold
open from either file.

**`ARCHITECTURE_v4.md` §5.4 states a rule the code does not and cannot honour** — "reference edges may
never move a confidence band". In practice `agent/verdict.py` selects the driving marker by
canonicity, and the driver sets the band, so curated knowledge necessarily moves it. v5 restates the
rule in a form that is both true and enforceable: *reference data determines which question is asked
of the measurement; the measurement alone determines the answer.*

## What did not move

`CLAUDE.md` and `README.md` stay at the repo root — the first is the live operating contract loaded
every session, the second is the repository's front door. `docs/note-capture-design.md` stays current:
it remains the spec of record for how the lab's knowledge is captured, and folds into v5's claim store.
