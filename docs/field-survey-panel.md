# Do published LLM annotation tools tell the model what was measured?

**No. Seven of seven verifiable tools, read at source: none serialises the measured
gene universe into the string sent to the API.**

Surveyed 2026-09-22 by reading each tool's prompt-construction code — the function
that assembles the API payload — rather than the paper's description of it. Papers
routinely describe an idealised prompt.

This matters because of the ablation in `evals/panel_absence/`: told nothing about the
panel, models made a false or unwarranted claim about the assay in **94.8%** of calls;
told the panel, **1.5%**. If no tool supplies the panel, the bare arm is their
operating condition on targeted-panel data, and that number describes the field rather
than the models.

## What each tool actually sends

| Tool | Gene universe? | Prompt contains | Source |
|---|---|---|---|
| **GPTCelltype** (Nat Methods 21:1462, 2024) | **No** | tissue name + top-10 positive-logFC **gene names only** | `R/gptcelltype.R` L60-77 |
| **CASSIA** (Nat Commun 2026) | **No** | species + tissue + ranked markers + optional free-text `additional_info` (default `None`) | `CASSIA_python/.../main_function_code.py` L147-159 |
| **mLLMCelltype** (Commun Biol 2026) | **No** | species/tissue + top-10 gene names per cluster; R and Python templates match | `R/R/prompt_templates.R` L268-289 |
| **NanoCellAnnotator** (bioRxiv 2026, spatial) | **No to the model** | functional programs + supporting genes + candidate taxonomy. Panel is the ORA background only | `Phase2/functionality/functionality_new_idea.py` L294-300 |
| **scExtract** (Genome Biol 26:174, 2025) | **No** | article text + top-10 DEGs | `src/scextract/utils/prompts.py` L131 (branch `master`) |
| **GPTBioInsightor** (Bioinform Adv 2026) | **No** | tissue string + top-15 DEGs + pathways | `gptbioinsightor/prompt/_celltype.py` L53-54 |
| **scChat** | **No** | top-25 **bare gene symbols**, no statistics, + unfiltered marker tree | `scchatbot/cell_types/annotation_pipeline.py` L279-307 |
| **CytoAnalyst** | **UNVERIFIED** | closed source; both repos are issue trackers with only a README | — |

GPTCelltype's prompt in full, since it is the field's canonical one and the whole
package is a single function:

```r
paste0('Identify cell types of ', tissuename, ' cells using the following markers
separately for each\n row. Only provide the cell type name. Do not show numbers
before the name.\n Some can be a mixture of multiple cell types.\n',
paste(input[id], collapse = '\n'))
```

`input` is `tapply(...)` over `avg_log2FC > 0` rows, sliced `[1:topgenenumber]`,
default 10. Nothing else is ever sent.

## The sharpest finding

**NanoCellAnnotator is the one spatial, panel-aware tool — and it still does not tell
the model.** Its ORA script takes the panel as a *required* argument:

```python
ap.add_argument("--universe", required=True,
                help="CSV with one column listing universe/panel genes")
```

with the docstring *"universe_genes restricts associations to your panel/universe"*.
The panel is loaded, hashed into a set, and used as `M` in `hypergeom.sf(k-1, M, K, n)`.
It then never reaches the model: `all_genes480` appears in no prompt in the repo,
across the v4 prompt, the v5 prompt, and every ablation arm including the
maximal-evidence one. The paper is motivated by *"sparse gene panels"*, and the
constraint it applies is on the **output label space** (PanglaoDB/CellMarker), not the
input gene universe.

So the information is loaded, in memory, one variable away — and nobody passed it. That
is the signature of a blind spot rather than an oversight.

## Negative evidence is not passed either

No tool sends a tested-and-not-enriched list. Four partial mechanisms exist and all
conflate *not enriched* with *not measured*:

- **CASSIA** returns `"the following genes are not in the differential expression
  list"` — literally accurate, but DE-absence, not measurement-absence.
- **scExtract** asks the model about genes "not expressed in cluster X", while
  `if gene not in adata.var_names: continue` silently drops unmeasured ones.
- **scChat** computes the missing set, prints it to stdout, then interpolates the
  **unfiltered** marker tree into the prompt anyway.
- **GPTBioInsightor** goes the other way, instructing twice *"do not use the lack of
  classic markers as the basis for your reasoning"* — justified by top-N truncation,
  not by assay limits, and it suppresses legitimate negative evidence along with the
  spurious kind.

## No tool warns about targeted panels

Every README and docs tree was grepped for panel / targeted / universe / background /
"not measured" / Xenium / CosMx / MERFISH. Every hit was UI chrome ("right panel"),
CSS (`background-color`), or unrelated prose.

## Two qualifications to keep in any write-up

1. **An escape hatch exists but is undocumented and not the default.** CASSIA's
   `additional_info` and GPTBioInsightor's `background` are free-text; a user *could*
   paste a panel into either. Neither defaults to it, neither documents it, and
   GPTBioInsightor's field is named for tissue context rather than gene sets.
2. **These tools are safe-by-accident on whole-transcriptome data**, where a missing
   canonical marker usually *is* biologically absent. None has a mechanism that
   survives a targeted panel — which is precisely the regime spatial work is moving
   into.

## Replication notes

- scExtract's default branch is **`master`**; `main` is a stale pre-refactor tree with
  different paths.
- NanoCellAnnotator's README says "ISMB 2026 (submitted)" while the bioRxiv DOI
  resolves to the same paper; cite both.
- CytoAnalyst should be reported as *implementation not public*, never as a negative
  data point.
