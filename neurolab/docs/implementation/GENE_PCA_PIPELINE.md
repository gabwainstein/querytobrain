# Gene Expression PCA Pipeline — Implementation

This implements the pipeline in [gene_expression_pca_plan.md](gene_expression_pca_plan.md). Outputs live under `neurolab/data/gene_pca/`.

## Quick start

From repo root:

```bash
# Prerequisites: build combined atlas, install abagen + sklearn + pandas
python neurolab/scripts/build_combined_atlas.py
pip install abagen nilearn scikit-learn pandas  # optional: gseapy for Phase 3

# Run full pipeline (Phases 1–4)
python neurolab/scripts/run_gene_pca_pipeline.py

# Or run phases separately
python neurolab/scripts/run_gene_pca_phase1.py   # abagen expression, filter, scale
# Phase 1: --separate-cortex-subcortex (default) standardizes cortex (360) and subcortex (32) separately
#   because they have different expression distributions; avoids one dominating PCA
python neurolab/scripts/run_gene_pca_phase2.py    # full + receptor PCA
# Phase 2: default --variance 0.90 (90%% cumulative variance, standard). Use --variance 0 --n-full 15 for fixed counts.
python neurolab/scripts/run_gene_pca_phase3.py   # GO/cell-type/receptor labels, pc_registry
python neurolab/scripts/run_gene_pca_phase4.py   # drug-to-PC (requires PDSP Ki CSV)
```

For Phase 4, download PDSP Ki and place `KiDatabase.csv` in `neurolab/data/pdsp_ki/`:

```bash
python neurolab/scripts/download_pdsp_ki.py
```

## Outputs (in `neurolab/data/gene_pca/`)

| Phase | Files |
|-------|------|
| 1 | `expression_scaled.npy`, `gene_names.json`, `parcel_labels.json`, `receptor_gene_names.json`, `expression_scaler.pkl` |
| 2 | `pc_scores_full.npy`, `gene_loadings_full.npy`, `explained_variance.npy`, `pca_full_model.pkl`, and receptor PCA outputs |
| 3 | `celltype_loadings_per_pc.json`, `receptor_loadings_per_pc.json`, `pc_registry.json`, optional `go_enrichment_per_pc.json` |
| 4 | `drug_pc_coordinates.npy`, `drug_spatial_maps.npy`, `drug_names.json`, `drug_similarity_pc.npy` |

## Phase 5–6 (MLP and reports)

- **Phase 5 (gated PC+residual model):** The plan’s `GenePCBrainModel` uses `pc_scores_full.npy` as a fixed basis. To integrate: load `data/gene_pca/pc_scores_full.npy` and `pca_full_model.pkl` in the trainer and add the gated dual-head architecture from the plan; optionally train with drug spatial maps from Phase 4.
- **Phase 6 (drug enrichment reports):** Use `pc_registry.json`, `drug_pc_coordinates`, `receptor_loadings_per_pc.json`, and the trained model to build the query → PC decomposition → similar drugs → receptor involvement report described in the plan.

## Receptor gene list (v2) and knowledge base

- **`neurolab/docs/implementation/receptor_gene_list_v2.csv`** — **Canonical source.** Curated list of 250 receptor/transporter/channel genes with gene_symbol, gene_name, system, category, notes. Rows with category=EXCLUDE are omitted. Used by Gene PCA Phase 1 (default `--receptor-list`), receptor atlas, and `neurolab.receptor_kb`.
- **`neurolab/data/receptor_gene_names_v2.json`** — Generated from CSV via `build_receptor_data_from_csv.py` (for backward compatibility).
- **`neurolab/data/receptor_knowledge_base.json`** — Generated from CSV; metadata per gene for training. Use `build_abagen_cache.py --receptor-kb` and `neurolab.receptor_kb` (`load_receptor_genes`, `get_gene_descriptions`).

The current trainer already has a **gene head** that uses PCA on abagen terms in the expanded cache (`gene_pca.pkl`, `gene_loadings.npz`). The pipeline here provides a **shared full-genome PCA** basis and drug space for the plan’s gated model and reporting.
