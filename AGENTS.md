# AGENTS.md

Canonical, tool-neutral guidance for AI agents working in this repository.

`CLAUDE.md` is a **symlink to this file**, so Claude Code and AGENTS.md-aware tools
(Copilot/VS Code, Codex, opencode, Cursor, Zed, …) read the same content. **Edit this
file, not the symlink.** Convention + per-client file map:
`~/.agents/knowledge/agent-docs-convention.md` (Agent Stack).

## Project overview

Pilot experiments for **"Phylogenies of Word Sense"** (PhyloSen / CaMeMo, ERC).
Pipeline: extract target-word concordances from the **GreLa** Latin corpus (DuckDB) →
contextual token/phrase embeddings (LaBERT-WSD, PhilBerta, Qwen) → LLM sense
classification → dimensionality reduction / projection → (sense) phylogenies.

Case-study lexemes: **passio, pathos, seminarium** (+ mercurius, labyrinthus).
Author: Vojtěch Kaše (CCS-ZCU). License: CC-BY-SA 4.0.

## Setup

```bash
cd ~/notebooks/phylosen_pilot
source phylosen_venv/bin/activate     # Python 3.12 venv
```

Run notebooks via Jupyter/editor **from `scripts/`** (they use `../data/...` paths).

- `requirements.txt` — base deps (pandas, scikit-learn, gensim, networkx, matplotlib, seaborn, sddk).
- Already installed in the venv: `torch`, `transformers`, `umap-learn`, `biopython`,
  `iplotx`, `nbformat`, `duckdb`, `tempun`.

## Layout

- `scripts/` — numbered pipeline notebooks + reusable `*.py` helpers
- `data/` — reference data, `embedding_classifier_results.json`, Newick trees (`*.dnd`)
- `data/large_files/` — large Parquet corpora & samples (git-ignored)
- `figures/` — output figures

## Pipeline notebooks (`scripts/`)

1. `1_grela-extraction.ipynb` — target spans + KWIC from GreLa DuckDB.
2. `2_generate-samples.ipynb` — generate concordance samples.
3. `3_llm-classification.ipynb` — LLM labels via `llmprompt_*.py`.
4. `4_contextual-embeddings.ipynb` — embeddings for classified samples and full corpora.
5. `5_embeddings-classifier.ipynb` — per-word MLP benchmarks, out-of-fold confusion
   matrices, and saved scaler/MLP pipelines (`*_wsd_philberta_l11_mdeberta_l9_concat.joblib`).
   Current inputs are `*_sample2000_classified_embeddings.parquet`; annotation model
   is `qwen3.8:27b`. Its introductory Markdown still describes older data/labels.
6. `6_embedding-classification-applied.ipynb` — inference only on full Parquets,
   batches of 1,024; preserves columns/order, appends prediction IDs and sense names,
   writes `*_embeddigings_full_classified.parquet` using a temporary file then replace.
7. `7_projections.ipynb` — sample/full context maps, diagnostics, separate LDA views,
   composition, century trajectories, ERC A/B/C figures, and shared within-lexeme LDA
   experiments. Shared basis defaults to sample fitting; final shared trajectories
   use the sample and century centroids (minimum 3). Other lineage views cover both variants.
8. `8_embedding-denoising.ipynb` — current shared-space experiment: sample-only,
   label-free PC/coordinate ablation, work-held-out diagnostics, MLP signal retention,
   shared PCA and optional UMAP, and century/bi-century trajectories.
   Uses `embedding_denoising.py`; outputs to `figures/embedding-denoising/`.

- `6_projections.ipynb` and `_6_projections_BACKED-UP.ipynb` — earlier projection work;
  do not confuse `6_projections.ipynb` with the current inference notebook 6.
- `_python-phylogenies.ipynb` — manual Newick tree drawing (Bio.Phylo / iplotx).
  **Tree construction from embeddings is not implemented.**
- `wordnets.ipynb`, `pie-charts-helper.ipynb` — ancillary analyses.

## Helper modules (canonical locations)

> Note: some older docs swap the first two; the list below reflects the actual code.

- `embedding_denoising.py` — `SharedFilterBank`, `FilterSpec`, `century_centroids()`,
  `cross_word_distances()`; label-free filters fitted on pooled training samples with
  equal word weights. `century_centroids(..., bin_years=100)` also supports positive
  multiples of 100 (e.g. 200); averages usages, not century means. High variance is
  a nuisance hypothesis, not proof that a direction should be removed.
- `grela_helpers.py` — `make_connection()`, `create_target_spans_table()`,
  `kwic_from_spans()` (DuckDB at `/srv/data/grela/grela_v0.7.duckdb`)
- `classification_helpers.py` — `extract_contextual_embedding()`,
  `batch_extract_embeddings()`, `create_mlp_classifier()`, `evaluate_classifier()`,
  `prepare_classification_data()` (model-agnostic: XLM-R/LaBERT, PhilBerta, Qwen)
- `embedding_helpers.py` — KWIC→subword→hidden-state pipeline: `process_kwic_tokens()`,
  `augment_with_subwords_span()`, `encode_trunc()`, `hidden_phrase_embedding()`,
  `embed_emlap_instance()`
- `projections_helpers.py` — `REDUCTIONS`, `get_2d()`, `produce_hover()`, metrics
  (`silhouette_from_labels`, `davies_bouldin_from_labels`, …), `create_label_palette()`,
  `plot_single_2d_projection()`, `plot_projection_grid()`, `create_3d_temporal_projection()`,
  `compute_century_trajectories_data()`, `plot_3d_century_trajectories()`,
  `plot_3d_lda_temporal()`. **Also contains a DUPLICATE `LDAProjection` class.**
- `lda_projections.py` — **canonical** `LDAProjection`, `compute_lda_projection()`,
  `evaluate_lda_vs_unsupervised()`, `compute_shared_lda_projection()` (pooled LDA),
  `compute_within_lexeme_shared_projection()` (combined local discriminants),
  plus hybrid-dim selection: `compute_top_dims()`,
  `compute_hybrid_dims()`, `HybridDimSelection`, `select_dims()`, `save_hybrid_dims()`
- `llmprompt_{passio,pathos,seminarium}.py` — `LABEL_MAP` (sense id→name) +
  few-shot prompt templates (`SYSTEM`, `FEWSHOT`)

## Conventions & gotchas

- **Run notebooks from `scripts/`** — paths are `../data/...`.
- **Label column is `class_<model>`**, where `<model>` is a notebook variable
  (e.g. `qwen3.8:27b`, `llama3.3`). Don't hardcode; read the current value.
- **`random_date` is derived**, not a raw Parquet column (computed with `tempun` from
  `not_before`/`not_after`).
- **After editing a helper `.py`, reload it in the notebook** — `from x import y` binds
  names, not the module:
  ```python
  import importlib, <module>; importlib.reload(<module>)
  ```
- **Duplicate `LDAProjection`** in `lda_projections.py` (canonical) and
  `projections_helpers.py`. Check which one a notebook imported before changing it.
- **Per-dimension "hybrid" selection does not work here** (measured 2026-10): the sense
  signal in `emb_philberta_l11` is distributed across ~400+ dims. Selecting the top-k
  dims by marginal importance (LDA loadings / F-test / mutual info / RF) and refitting
  LDA collapses separation to ≈ random (passio silhouette 0.46 → ≈ −0.06 for k=18;
  random-18 ≈ −0.10). Prefer shrinkage LDA (`solver='eigen'`), PCA→LDA, or learned
  projections. See `data/hybrid_dims_passio_pathos.json` for the selection produced.
- `add_lda_cells.py` and `reorganize_notebook.py` rewrite an older projection notebook
  via `nbformat`; inspect their target paths before running. Preserve unrelated notebook
  cells and outputs when making focused edits.
- Large data (`data/large_files/`), `phylosen_venv/`, `CLAUDE.md`, `MEMORY.md`, and
  `.claude/` are git-ignored. `AGENTS.md` **is** committed.

## Data and artifact provenance

- `data/large_files/*_sample2000_classified_embeddings.parquet` — current LLM-labelled
  embedding samples; sense 0 is uncertain and omitted in the analyses.
- Saved classifier input order is **PhilBERTa L11, then mDeBERTa L9** (1,536 dims).
  Joblib artifacts contain the fitted `StandardScaler`/MLP pipeline, `embedding_cols`,
  `merge_method`, `label_column`, training `label_map`, and CV accuracy/macro-F1.
- Full inputs retain the spelling `*_embeddigings_full.parquet`; notebook 6 produces
  `*_embeddigings_full_classified.parquet`. Labels are
  `class_<annotation-model>_embedding_pred`, names in its `_sense` companion.
- `data/embedding_classifier_results.json` is an **older sample1000 benchmark**,
  not the current sample2000 results; use notebook 5's outputs and saved artifacts.
- `figures/embedding-denoising/heldout-diagnostics.csv`, `heldout-mlp.csv`, and
  `sample-split-and-dates.csv` record notebook 8's exploratory split, metrics and dates.
  Full classifier predictions and sample LLM labels are different evidence sources.
- `data/simple.dnd`, `data/labyrinthus_manual.dnd` — Newick trees.
- `data/hybrid_dims_passio_pathos.json` — earlier selected hybrid LDA dimensions.

## Verified results and current direction (reviewed 2026-10-11)

- Notebook 5: ReLU MLP `(512, 128)`, feature scaling inside each CV fold, 5-fold
  row-level evaluation. Concatenation accuracy / macro-F1: **passio .8315 / .7321**;
  **pathos .6841 / .6833**. Signal supports sense prediction; it does not establish
  cross-word alignment or that 2-D will retain it.
- Notebook 6's saved run classified **100,361 passio** and **16,280 pathos** rows.
- Separate LDA plots are clear but independently fitted axes cannot be overlaid.
  Notebook 7 combines 5 passio + 3 pathos discriminants, then uses within-word
  covariance for the 2-D display; current saved output retains **.410** of that variance
  (earlier session run .453). No cross-word sense matching is supplied.
- Notebook 8 fits filters/display axes on a fixed work-held-out training split with
  equal word weights. The sample includes 1,911 passio + 1,833 pathos labelled rows;
  missing dates affect temporal plots only. The following **concat** results are one
  exploratory split, not directly comparable to notebook 5's CV:

  | Filter | 2-D kNN macro-F1 passio / pathos | MLP macro-F1 passio / pathos |
  | --- | --- | --- |
  | raw | .275 / .290 | .640 / .659 |
  | center + L2 | .261 / .272 | .620 / .662 |
  | drop PC1 | .227 / .324 | not evaluated |
  | drop PCs 1–2 | .348 / .396 | .626 / .669 |
  | drop PCs 1–5 | .181 / .248 | .623 / .673 |

- PC1 accounts for .116 of concat training variance. Dropping it reduces held-out
  linear word-identity balanced accuracy from .999 to .685; this is a diagnostic,
  **not proof** that removed variation was semantically irrelevant.
- **User-selected visual focus is `drop_pc_1`**, not the quantitative winner.
  Transform: common training-fitted mean → subtract PC1 → L2-normalize each usage →
  common 2-D PCA → orthogonal rotation toward the raw display for comparison.
  Rotation preserves distances. Sense labels guide evaluation, not this transformation.
- User wants raw semantic similarity to emerge while preserving differences between
  words: similar emotion/suffering senses are **not identical**. Do not impose paired
  sense alignment, collapse inventories, or treat word identity as inherently nuisance.
  Removing a few directions differs from keeping a few original coordinates.
- Current focused figure: labelled sample, **200-year bins**, **minimum 5 usages** per
  word/sense/bin; squares = passio, diamonds = pathos; distinct per-word/sense colours.
  Average projected usages equally; time at bin midpoints; connect adjacent valid bins
  only, leaving gaps open. These are usage trajectories, not inferred genealogies.
- Controls: `TEMPORAL_BIN_YEARS` in comparison cell (100 or 200); `MIN_CENTURY` in setup
  applies to the entire bin. Focused cell has `FOCUS_ELEV/AZIM`, `FOCUS_XLIM/YLIM`, and
  automatic 2% padding. Rerun plotting cells for these changes; no classifier refit.
  Keep legends close and grouped by word; use subplot grid + `tight_layout`; **no title
  or footer captions inside the focused image**. Axis labels remain.
  Export stem: `figures/embedding-denoising/concat-pca-drop_pc_1-200yr-centroids`
  (`.png`, `.pdf`, `.csv`).
- Notebook 8's default path was executed successfully; UMAP remains opt-in and was
  not run in this session. Repeat splits/seeds before claiming filtering improves semantics.
- Known fresh-kernel pitfalls observed during review: notebook 5 references `LABEL_COL`
  without defining it in saved cells (intended `f"class_{model}"`); notebook 7 contains
  a stray `specs[""]` cell. These were documented, **not repaired**, in this quick review.

## Agent knowledge policy

Repo rules live here (`AGENTS.md`, committed). Client-specific config stays thin and
local: `CLAUDE.md` is a symlink to this file; `.claude/` is local. Reusable
cross-project skills/knowledge belong to the Agent Stack (`~/.agents/skills`,
`~/.agents/knowledge`), not this repo.

## Repository housekeeping

IDE artifacts (`.idea/`, `*.iml`) and Jupyter checkpoints (`.ipynb_checkpoints/`)
are not tracked. Bulk data and model directories (`large_files/`, `large_data/`)
are never committed; keep large inputs and outputs there. See the
[organizer](https://github.com/kasev/organizer) workspace for the shared workflow.
