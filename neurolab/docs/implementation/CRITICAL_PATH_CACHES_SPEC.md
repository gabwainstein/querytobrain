# Critical Path Caches — Detailed Specification

This document specifies the critical-path caches identified in the master plan gap analysis: **NeuroQuery full vocabulary + maps**, **PDSP Ki cache**, **Gene PCA basis cache**, and **Training embeddings cache**. These are prerequisites for the cognitive decoder, drug prediction engine, PC-constrained Generalizer, and zero-API-call training.

---

## 0. NeuroQuery Full Vocabulary + Maps Cache

### 0.1 Purpose

The decoder cache must contain **all** NeuroQuery vocabulary terms and their associated brain maps. This is the foundation for:
- Cognitive term → brain map lookup at inference
- Training the text-to-brain embedding (Generalizer + Memorizer)
- Ontology expansion (derived maps for related terms)

### 0.2 Current State

| Component | Status | Location |
|-----------|--------|----------|
| Full vocabulary list | ✅ Exists | `neuroquery_data/neuroquery_model/vocabulary.csv` (~7,548 terms) |
| NeuroQuery model (package) | ✅ Fetched | `neuroquery` fetches model; `model.vectorizer.get_feature_names()` gives vocab |
| Decoder cache builder | ✅ Exists | `build_term_maps_cache.py` |
| **Default caps terms** | ⚠️ Problem | Standalone run defaults to `--max-terms 5000` (variance cap) |
| **Vocabulary source mismatch** | ⚠️ Possible | `build_term_maps_cache` uses package model; `build_neuroquery_cache` uses local `vocabulary.csv` |

### 0.3 Requirements

1. **Fetch full vocabulary:** Use the complete NeuroQuery vocabulary — either from `model.vectorizer.get_feature_names()` (package) or from `neuroquery_data/neuroquery_model/vocabulary.csv` (canonical local list). The local CSV has 7,548 terms with document frequencies.
2. **Build maps for all terms:** No variance cap. Use `--max-terms 0` so every term that produces a valid map is kept.
3. **Canonical vocabulary:** Prefer `vocabulary.csv` when available so the pipeline uses a reproducible, versioned term list.

### 0.4 Target Output

**Directory:** `neurolab/data/decoder_cache/`

| File | Description |
|------|-------------|
| `term_maps.npz` | (N, 392) — parcellated maps for N terms |
| `term_maps_z.npz` | Z-scored rows (for fast Pearson in decode) |
| `term_vocab.pkl` | list of N strings — full vocabulary (or subset that produced valid maps) |

**Expected:** N ≈ 7,000–7,500 (some terms may fail: zero map, parcellation mismatch) — **not** 5,000.

### 0.5 Build Commands

```bash
# Full vocabulary + maps (no cap) — REQUIRED for production
python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0 --n-jobs 30

# Alternative: use local vocabulary.csv (build_neuroquery_cache)
python neurolab/scripts/build_neuroquery_cache.py --output-dir neurolab/data/decoder_cache --max-terms 0 --neuroquery-data-dir neuroquery_data/neuroquery_model
```

**rebuild_all_caches.py** already uses `--max-terms 0` in full mode. Ensure you never run with `--quick` in production (quick mode caps at 500).

### 0.6 Verification

```bash
python -c "
import pickle
import numpy as np
vocab = pickle.load(open('neurolab/data/decoder_cache/term_vocab.pkl', 'rb'))
data = np.load('neurolab/data/decoder_cache/term_maps.npz')
n = data['term_maps'].shape[0]
assert n == len(vocab)
assert n >= 6000, f'Expected 6000+ terms, got {n}'
print('OK:', n, 'terms x 392 parcels')
"
```

**Alert:** If `n < 6000`, the cache was likely built with a cap. Re-run with `--max-terms 0`.

---

## 1. PDSP Ki Cache

### 1.1 Purpose

The drug prediction engine depends on PDSP Ki binding affinities. This cache provides:
- **Compound → receptor affinity matrix** for lookup at inference
- **Compound → 392-parcel spatial map** via projection through gene expression PCA
- **Drug similarity in PC space** for enrichment reports

### 1.2 Current State

| Component | Status | Location |
|-----------|--------|----------|
| PDSP download | ✅ Exists | `download_pdsp_ki.py` → `neurolab/data/pdsp_ki/KiDatabase.csv` |
| Drug-to-PC projection | ✅ Exists | `run_gene_pca_phase4.py` → `gene_pca/drug_spatial_maps.npy`, `drug_pc_coordinates.npy`, `drug_names.json` |
| Standalone PDSP cache | ❌ Missing | No `pdsp_profiles.npz` or dedicated cache dir |
| Integration with inference | ⚠️ Partial | `text_to_brain.py` looks for `gene_pca/drug_spatial_maps.npy` but PDSP pathway not fully wired |

### 1.3 Target Output Structure

**Directory:** `neurolab/data/pdsp_cache/`

| File | Shape/Format | Description |
|------|--------------|--------------|
| `pdsp_profiles.npz` | `profiles`: (n_compounds, n_receptors), `compound_names`: list, `receptor_genes`: list | Raw binding affinity weights (1/Ki, normalized per compound). Sparse-friendly: store only non-zero entries if desired. |
| `pdsp_pc_projections.npz` | `projections`: (n_compounds, 392), `compound_names`: list | Spatial maps in Glasser+Tian parcel space. Reconstruct via `pc_scores @ (gene_loadings @ gene_weight_vector)`. |
| `pdsp_pc_coordinates.npy` | (n_compounds, 15) | PC-space coordinates for drug similarity (cosine, UMAP). |
| `compound_names.json` | list of str | Canonical compound names (PDSP "Ligand Name"). |
| `receptor_gene_index.json` | dict gene → column_idx | Maps gene symbol to column in profiles matrix. |
| `metadata.json` | `n_compounds`, `n_receptors`, `n_parcels`, `gene_pca_dir`, `pdsp_csv_path` | Provenance and dimensions. |

### 1.4 Build Script Specification

**Script:** `neurolab/scripts/build_pdsp_cache.py`

**Prerequisites:**
1. `neurolab/data/pdsp_ki/KiDatabase.csv` (from `download_pdsp_ki.py`)
2. `neurolab/data/gene_pca/` with Phase 1–2 outputs: `expression_scaled.npy`, `gene_names.json`, `gene_loadings_full.npy`, `pc_scores_full.npy`

**Algorithm:**
1. Load PDSP CSV; detect columns: Ligand Name, Receptor, Ki (nM) [or Ki Value]
2. Filter to human species if column exists
3. Map PDSP receptor names → gene symbols via `RECEPTOR_TO_GENE` (extend from `run_gene_pca_phase4.py`)
4. Build compound × gene affinity matrix: for each (compound, receptor), `weight = 1 / geometric_mean(Ki)`; normalize per compound (L1 or sum to 1)
5. Align genes to `gene_names.json` from gene_pca (only genes in both PDSP and abagen)
6. For each compound: `gene_weight_vector` = affinity row; `pc_coords = gene_loadings @ gene_weight_vector`; `spatial_map = pc_scores @ pc_coords`
7. Save all outputs to `pdsp_cache/`

**CLI:**
```bash
python neurolab/scripts/build_pdsp_cache.py --output-dir neurolab/data/pdsp_cache
python neurolab/scripts/build_pdsp_cache.py --pdsp-csv neurolab/data/pdsp_ki/KiDatabase.csv --gene-pca-dir neurolab/data/gene_pca --output-dir neurolab/data/pdsp_cache
```

**Options:**
- `--min-targets 2` — Skip compounds with fewer than N mapped targets
- `--max-compounds 0` — Cap (0 = all)
- `--receptor-subset` — Optional: only these genes (e.g. HTR2A, DRD2, SLC6A4)

### 1.5 Integration Points

- **Inference (PDSP pathway):** When query is drug-like, lookup `compound_names` → get `pdsp_pc_projections` row → return as additional context map
- **Enrichment report:** Receptor engagement from `pdsp_profiles`; similar drugs from `pdsp_pc_coordinates` cosine similarity
- **Training:** Optionally add PDSP compound spatial maps to `merged_sources` as synthetic (term, map) pairs for drug queries

---

## 2. Gene PCA Basis Cache

### 2.1 Purpose

The Generalizer's PC-constrained head uses a **fixed biological basis** to reconstruct brain maps from 15-dimensional PC coefficients. This cache stores:
- **PC scores** (392 × 15): spatial pattern per PC
- **Gene loadings** (n_genes × 15): gene weights per PC
- **Biological labels** per PC for enrichment reports

### 2.2 Current State

| Component | Status | Location |
|-----------|--------|----------|
| Full gene PCA pipeline | ✅ Exists | `run_gene_pca_phase1.py`–`phase4.py` → `neurolab/data/gene_pca/` |
| Abagen-based PCA in merged_sources | ✅ Exists | `build_expanded_term_maps.py --gene-pca-variance 0.95` → `gene_pca.pkl`, `gene_loadings.npz` in merged_sources/ |
| Dedicated basis cache | ⚠️ Fragmented | gene_pca/ has full-genome PCA; merged_sources has abagen-term-only PCA. Trainer uses merged_sources PCA for gene head. Generalizer (master plan) expects full-genome basis. |
| Parcellation | ✅ 392 | Phase 1 uses `get_combined_atlas_path()` → Glasser+Tian |

### 2.3 Target Output Structure

**Directory:** `neurolab/data/gene_pca/` (existing; standardize layout)

| File | Shape/Format | Description |
|------|--------------|--------------|
| `pc_scores.npy` | (392, 15) | Spatial patterns. Column i = PC i+1. Fixed basis for `map = pc_scores @ pc_coefs`. |
| `gene_loadings.npy` | (15, n_genes) | Gene weights per PC. Row i = PC i+1. For drug projection: `pc_coefs = gene_loadings @ gene_weight_vector`. |
| `gene_names.json` | list of str | Gene symbols in loadings column order. |
| `explained_variance.npy` | (15,) | Variance explained per PC. |
| `pc_registry.json` | `{ "PC1": { "label": "...", "go_terms": [...], "receptor_loadings": {...} }, ... }` | Biological interpretation per PC. From Phase 3. |
| `n_components` | int (15) | Number of PCs. |
| `n_parcels` | int (392) | Parcel count. |

**Receptor-only PCA (optional, for pharmacological specificity):**
| File | Shape | Description |
|------|-------|--------------|
| `receptor_pc_scores.npy` | (392, 10) | Receptor-only spatial basis. |
| `receptor_gene_loadings.npy` | (10, n_receptor_genes) | Receptor gene weights. |

### 2.4 Standardization Requirements

1. **Rename for clarity:** `pc_scores_full.npy` → `pc_scores.npy` (or keep both; document that `pc_scores.npy` is the canonical 15-PC basis)
2. **Ensure 392 parcels:** Phase 1 uses `get_n_parcels()` and `get_combined_atlas_path()`; verify output shape
3. **Canonical path:** Generalizer and PDSP cache should load from `neurolab/data/gene_pca/` by default
4. **pc_registry.json:** Must exist for enrichment (biological labels). Phase 3 produces it.

### 2.5 Build Pipeline (Existing)

```bash
# Prerequisites: combined_atlas_392.nii.gz, abagen
python neurolab/scripts/run_gene_pca_phase1.py   # expression_scaled, gene_names
python neurolab/scripts/run_gene_pca_phase2.py   # pc_scores_full, gene_loadings_full (n_full=15)
python neurolab/scripts/run_gene_pca_phase3.py   # pc_registry.json
# Phase 4 for drugs (see PDSP cache)
```

**Add a convenience script** `build_gene_pca_basis.py` that:
1. Runs phases 1–3 (or checks outputs exist)
2. Copies/renames to canonical names: `pc_scores.npy`, `gene_loadings.npy`
3. Writes `metadata.json` with n_parcels, n_components, n_genes

### 2.6 Integration Points

- **Generalizer:** Load `pc_scores.npy` as `nn.Parameter` or buffer (non-trainable). PC head predicts 15-dim; `map = pc_scores @ pc_coefs`
- **PDSP cache:** Uses `gene_loadings.npy`, `gene_names.json`, `pc_scores.npy` for drug→spatial projection
- **Enrichment report:** `pc_registry.json` labels each PC for radar chart

---

## 3. Training Embeddings Cache

### 3.1 Purpose

All training terms must have **pre-computed embeddings** so that:
- **Zero API calls during training** — no OpenAI requests in the training loop
- **Deterministic training** — same embedding every epoch
- **Cost control** — one-time embed cost (~$0.26 for ~60K terms at text-embedding-3-large)
- **Fast iteration** — change MLP architecture without re-embedding

### 3.2 Current State

| Component | Status | Location |
|-----------|--------|----------|
| Trainer encoder options | ✅ OpenAI, sentence-transformers, tfidf | `train_text_to_brain_embedding.py` |
| On-the-fly embedding | ✅ Default | Trainer calls `encode(terms)` per batch; OpenAI = API call per batch |
| Pre-computed embeddings | ❌ Missing | No cache; no `--use-cached-embeddings` |
| Ontology label embeddings | ⚠️ Partial | `--kg-context-mode semantic` embeds ontology labels; cached per model dir. Not training-term embeddings. |

### 3.3 Target Output Structure

**Directory:** `neurolab/data/embeddings/`

| File | Shape/Format | Description |
|------|--------------|--------------|
| `all_training_embeddings.npy` | (N, 1536) | Pre-computed vectors. Row i = embedding of `embedding_vocab[i]`. Float32. |
| `embedding_vocab.pkl` | list of N str | Term labels in same order as rows. Must match `term_vocab.pkl` from cache (or superset). |
| `embedding_metadata.json` | `model`, `dim`, `n_terms`, `cache_dir`, `created` | Provenance. |
| `embedding_config.json` | `model`, `dimensions`, `batch_size` | For reproducibility. |

**Optional (for KG-augmented training):**
| File | Description |
|------|-------------|
| `augmented_embeddings.npy` | (N, 1536) — embeddings of KG-augmented text per term |
| `augmented_vocab.pkl` | Same as embedding_vocab (augmented text used for each) |

### 3.4 Build Script Specification

**Script:** `neurolab/scripts/build_training_embeddings.py`

**Prerequisites:**
1. Training cache with `term_maps.npz`, `term_vocab.pkl` (e.g. `merged_sources/` or `decoder_cache_expanded/`)
2. `OPENAI_API_KEY` for text-embedding-3-large (or 3-small)

**Algorithm:**
1. Load `term_vocab.pkl` from cache
2. Optionally filter: `--max-terms 0` (all) or cap
3. Batch embed: 2048 terms per request (OpenAI limit); 100–500 for safety
4. Save `all_training_embeddings.npy`, `embedding_vocab.pkl`
5. If `--augment-with-kg`: for each term, fetch ontology neighbors, format augmented text, embed, save to `augmented_embeddings.npy`

**CLI:**
```bash
python neurolab/scripts/build_training_embeddings.py --cache-dir neurolab/data/merged_sources --output-dir neurolab/data/embeddings
python neurolab/scripts/build_training_embeddings.py --cache-dir neurolab/data/merged_sources --encoder openai --model text-embedding-3-large --dimensions 1536 --batch-size 500 --output-dir neurolab/data/embeddings
```

**Options:**
- `--encoder openai` (default) | `sentence-transformers`
- `--model text-embedding-3-large` | `text-embedding-3-small` | `NeuML/pubmedbert-base-embeddings`
- `--dimensions 1536` — Matryoshka truncation for 3-large (halve storage)
- `--batch-size 500` — Terms per API request
- `--max-terms 0` — Cap (0 = all)
- `--augment-with-kg` — Build augmented embeddings (requires ontology dir)
- `--ontology-dir` — For KG augmentation

### 3.5 Trainer Integration

**New flags for `train_text_to_brain_embedding.py`:**
- `--use-cached-embeddings` — Load from `--embeddings-dir` instead of encoding on-the-fly
- `--embeddings-dir neurolab/data/embeddings` — Path to embeddings cache

**Logic:**
1. If `--use-cached-embeddings`:
   - Load `all_training_embeddings.npy`, `embedding_vocab.pkl`
   - Align with cache: ensure `embedding_vocab` matches `term_vocab` (or subset). Index mapping: for each term in cache, find row in embedding_vocab
   - Skip encoder; use pre-loaded matrix for training
2. Else: current behavior (encode on-the-fly)

**Validation:** If cache has 50K terms and embeddings have 48K, warn and use intersection (or fail if strict).

### 3.6 Cost Estimate

- text-embedding-3-large: $0.13/1M tokens. ~20–50K terms × ~20 tokens/term ≈ 400K–1M tokens → **~$0.05–$0.13**
- text-embedding-3-small: $0.02/1M tokens → **~$0.01**
- Batched at 500/request: ~100–200 API calls. ~5–10 minutes total.

---

## 5. Build Order and Dependencies

```
0. build_term_maps_cache.py --max-terms 0  (or rebuild_all_caches full mode)
   → decoder_cache/ with FULL NeuroQuery vocabulary + maps (~7K terms)

1. download_pdsp_ki.py          → pdsp_ki/KiDatabase.csv
2. run_gene_pca_phase1.py       → gene_pca/expression_scaled, gene_names
3. run_gene_pca_phase2.py       → gene_pca/pc_scores, gene_loadings
4. run_gene_pca_phase3.py       → gene_pca/pc_registry.json
5. build_pdsp_cache.py          → pdsp_cache/ (depends on 1, 2, 3)
6. rebuild_all_caches.py        → merged_sources/ (includes decoder_cache from step 0)
7. build_training_embeddings.py → embeddings/ (depends on 6)
8. train_text_to_brain_embedding.py --use-cached-embeddings --embeddings-dir embeddings/
```

---

## 5. Verification

**PDSP cache:**
```bash
python -c "
import numpy as np
import json
d = 'neurolab/data/pdsp_cache'
proj = np.load(f'{d}/pdsp_pc_projections.npz')
names = json.load(open(f'{d}/compound_names.json'))
assert proj['projections'].shape[0] == len(names)
assert proj['projections'].shape[1] == 392
print('OK:', len(names), 'compounds x 392 parcels')
"
```

**Gene PCA basis:**
```bash
python -c "
import numpy as np
pc = np.load('neurolab/data/gene_pca/pc_scores_full.npy')  # or pc_scores.npy
assert pc.shape[0] == 392, pc.shape[1] == 15
print('OK: pc_scores', pc.shape)
"
```

**Training embeddings:**
```bash
python -c "
import numpy as np
import pickle
emb = np.load('neurolab/data/embeddings/all_training_embeddings.npy')
vocab = pickle.load(open('neurolab/data/embeddings/embedding_vocab.pkl', 'rb'))
assert emb.shape[0] == len(vocab)
assert emb.shape[1] in (1536, 3072)
print('OK:', len(vocab), 'terms x', emb.shape[1], 'dim')
"
```

---

## 6. Summary Table

| Cache | Script | Output Dir | Key Outputs | Depends On |
|-------|--------|------------|-------------|------------|
| **NeuroQuery full vocab + maps** | `build_term_maps_cache.py --max-terms 0` | `decoder_cache/` | term_maps.npz, term_vocab.pkl (~7K terms) | NeuroQuery model, atlas |
| PDSP Ki | `build_pdsp_cache.py` (new) | `pdsp_cache/` | pdsp_profiles.npz, pdsp_pc_projections.npz, compound_names.json | PDSP CSV, gene_pca |
| Gene PCA basis | `run_gene_pca_phase1-3` (+ optional `build_gene_pca_basis.py`) | `gene_pca/` | pc_scores.npy, gene_loadings.npy, pc_registry.json | abagen, atlas |
| Training embeddings | `build_training_embeddings.py` (new) | `embeddings/` | all_training_embeddings.npy, embedding_vocab.pkl | merged_sources, OpenAI API |
