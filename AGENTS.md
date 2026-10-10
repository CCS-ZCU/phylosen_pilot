# AGENTS.md

Canonical, tool-neutral guidance for AI agents working in this repository.

`CLAUDE.md` is a **symlink to this file**, so Claude Code and AGENTS.md-aware tools
(Copilot/VS Code, Codex, opencode, Cursor, Zed, …) read the same content. **Edit this
file, not the symlink.** Policy (see `.agent-square-src/README.md`): repository rules
live in `AGENTS.md`; keep client-specific adapters thin.

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
- `.agent-square-src/` — Agent Stack policy (do not edit)

## Pipeline notebooks (`scripts/`)

1. `1_grela-extraction.ipynb` — target spans + KWIC from GreLa DuckDB
2. `2_contextual-embeddings.ipynb` — contextual embeddings (uses `classification_helpers.py`)
3. `3_llm-classification.ipynb` — LLM sense classification (uses `llmprompt_*.py`)
4. `4_projections.ipynb` — LDA/PCA/UMAP/t-SNE projections, temporal & century-trajectory
   views. **~5 MB** (huge embedded base64 images) — handle diffs with care.
5. `5_embeddings-classifier.ipynb` — MLP classifier on embeddings; writes
   `data/embedding_classifier_results.json`
- `_python-phylogenies.ipynb` — tree **drawing** (manual Newick → Bio.Phylo / iplotx).
  **Tree construction from embeddings is not implemented yet.**
- `wordnets.ipynb`, `pie-charts-helper.ipynb` — ancillary analyses

## Helper modules (canonical locations)

> Note: some older docs swap the first two; the list below reflects the actual code.

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
  `evaluate_lda_vs_unsupervised()`, plus hybrid-dim selection: `compute_top_dims()`,
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
- `add_lda_cells.py` and `reorganize_notebook.py` **programmatically rewrite
  `4_projections.ipynb`** via `nbformat` — don't run them casually.
- Large data (`data/large_files/`), `phylosen_venv/`, `CLAUDE.md`, `MEMORY.md`, and
  `.claude/` are git-ignored. `AGENTS.md` **is** committed.

## Data

- `data/large_files/*_sample1000_classified.parquet` — sampled instances with embeddings
  + `class_<model>` labels
- `data/large_files/*.parquet`, `phylosen_instances/` — fuller corpora
- `data/embedding_classifier_results.json` — CV accuracy/F1 per embedding (passio, pathos, merged)
- `data/simple.dnd`, `data/labyrinthus_manual.dnd` — Newick trees
- `data/hybrid_dims_passio_pathos.json` — selected hybrid LDA dims

## Agent knowledge policy

Repo rules live here (`AGENTS.md`, committed). Client-specific config stays thin and
local: `CLAUDE.md` is a symlink to this file; `.claude/` is local. Reusable
cross-project skills/knowledge belong to the Agent Stack (`~/.agents/skills`,
`~/.agents/knowledge`), not this repo.
