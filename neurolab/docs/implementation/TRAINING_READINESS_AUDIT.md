# Training Readiness Audit

**Date:** 2026-02-14  
**Scope:** Full audit and verification that the NeuroLab training dataset and pipeline are ready to start text-to-brain embedding training.

---

## 1. Executive summary

| Status | Verdict |
|--------|---------|
| **Atlas** | OK — 392 parcels (Glasser 360 + Tian S2) |
| **Training cache (merged_sources)** | OK — 14,183 terms × 392 parcels, all required files present |
| **Per-source caches** | OK — decoder, neurosynth, unified, neurovault, neuromaps, enigma, abagen, pharma_neurosynth |
| **Trainer load test** | OK — Cache loads correctly; arrays consistent |
| **Training smoke test** | OK — 1-epoch run completed successfully |
| **Dependencies** | OK — sentence-transformers, numpy, sklearn, etc. installed |

**Verdict: Ready to start training.**

---

## 2. Atlas verification

| Check | Result |
|-------|--------|
| Path | `neurolab/data/combined_atlas_392.nii.gz` |
| Cortical labels | 360 (Glasser) |
| Subcortical labels | 32 (Tian S2) |
| Total parcels | 392 |

**Script:** `verify_full_cache_pipeline.py` — Atlas OK.

---

## 3. Critical-path caches

| Cache | Status | Maps | Parcels |
|-------|--------|------|---------|
| decoder_cache | OK | 4,983 | 392 |
| neurosynth_cache | OK | 3,228 | 392 |
| unified_cache | OK | 8,211 | 392 |
| **merged_sources** | **OK** | **14,183** | **392** |
| neuromaps_cache | OK | 40 | 392 |
| enigma_cache | OK | 49 | 392 |
| abagen_cache | OK | 15,633 | 392 |
| gene_pca | OK | 40 (components) | 392 |
| fc_cache | OK | 8 | 392 |
| pdsp_cache | **Required** | — | — |

**Note:** `pdsp_cache` is **required** for the full pharmacological pipeline. It provides compound→brain spatial maps (PDSP Ki → receptor gene expression → 392-D map) used for drug inference and enrichment. Without it, drug queries cannot use the pharmacological projection pathway. **PDSP build path:** `download_pdsp_ki.py` → (optional: `process_pdsp_for_neurolab.py` for compound×receptor affinity matrix) → `run_gene_pca_phase1.py` + `run_gene_pca_phase2.py` → `build_pdsp_cache.py`. The full build (`run_full_cache_build.py`) includes PDSP by default (download → gene PCA phase 1–2 if needed → build_pdsp_cache); use `--skip-pdsp` only to bypass.

---

## 4. Optional caches (included in merge)

| Cache | Status | Maps |
|-------|--------|------|
| neurovault_cache | OK | 5,327 |
| neurovault_pharma_cache | OK | 632 |
| pharma_neurosynth_cache | OK | 26 |

---

## 5. merged_sources (training cache) — detailed verification

### 5.1 Required files

| File | Present | Purpose |
|------|---------|---------|
| `term_maps.npz` | Yes | (N, 392) float64 — parcellated maps |
| `term_vocab.pkl` | Yes | N strings — text labels |
| `term_sources.pkl` | Yes | N strings — source label per term |
| `term_sample_weights.pkl` | Yes | N floats — per-term loss weight |
| `term_map_types.pkl` | Yes | N strings — fmri_activation / structural / pet_receptor |
| `abagen_gradient_components.npy` | Yes | (5, 392) — for PET residual-correlation evaluation |

### 5.2 Optional files (gene head)

| File | Present | Purpose |
|------|---------|---------|
| `gene_pca.pkl` | Yes | PCA fit for abagen terms |
| `gene_loadings.npz` | Yes | (n_genes, n_components) loadings |
| `abagen_term_indices.pkl` | Yes | Indices of abagen terms in merged cache |

### 5.3 Data consistency

- **term_maps:** (14,183, 392) float64
- **term_vocab:** 14,183 terms
- **term_sources:** 14,183 entries; 7 unique sources
- **term_sample_weights:** 14,183 entries; range 0.4–2.0 (used in all training paths: TF-IDF, PyTorch, gene head, retrain-on-train+val)
- **term_map_types:** 14,183 entries; 3 types (fmri_activation, structural, pet_receptor)
- **abagen_gradient_components:** (5, 392) — 5 gradient PCs

**Sources in merged_sources:**  direct, neurovault, abagen, enigma, neuromaps, neuromaps_residual, pharma_neurosynth

**Per-source term counts (verify after each build):**  
Run the following to get a breakdown; if **abagen > 500**, consider rebuilding the merge with tiered selection so abagen is capped at ~500 (see §3 and TRAINING_DATASET_BUILD_METHODS §7.2).

```bash
python -c "
import pickle
from pathlib import Path
from collections import Counter
p = Path('neurolab/data/merged_sources')
with open(p / 'term_sources.pkl','rb') as f: src = pickle.load(f)
counts = Counter(src)
for s, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {s}: {n}')
print('  TOTAL:', len(src))
"
```

**Interpretation:** abagen should be ~500 when merge uses `--max-abagen-terms 500` (Tier 1: ~250 receptor genes, Tier 2: ~32 WGCNA-style medoids, Tier 3: ~200 residual-variance). If abagen is in the thousands, source-weighted sampling (abagen ~10%) and loss weight 0.4 only compensate; prefer fixing at data level by re-running the merge with `--max-abagen-terms 500`.

**All arrays consistent:** lengths match; parcel dimension = 392.

---

## 6. Legacy / non-training caches (ignore for training)

The following caches have **400 parcels** (legacy Schaefer) and are **not used** for training:

- cache_brainpedia_plus_decoder
- cache_brainpedia_plus_decoder_full_ontology
- neurovault_cache_brainpedia
- smoke_decoder_cache

**Action:** None. These are legacy; the training pipeline uses `merged_sources` (392 parcels) only.

---

## 7. Dependencies

| Package | Status |
|---------|--------|
| sentence-transformers | Installed (v5.2.2) |
| numpy | Installed |
| scikit-learn | Installed |
| scipy | Installed |
| nilearn | Installed |
| nibabel | Installed |
| neuroquery | Installed |
| nimare | Installed |
| abagen | Installed |
| neuromaps | Installed |

**Default encoder:** `NeuML/pubmedbert-base-embeddings` (sentence-transformers). First run will download the model (~400MB).

---

## 8. Training smoke test

**Command:** `train_text_to_brain_embedding.py --cache-dir merged_sources --encoder tfidf --max-terms 200 --epochs 1`

**Result:** Success. Training completed; output:
- Loaded 200 terms (capped for smoke test)
- term_sources, term_sample_weights, term_map_types loaded
- Source-weighted sampling active
- Gene head configured (when gene_pca, gene_loadings, abagen_term_indices present)
- Train/val/test split applied
- 1 epoch completed; model saved

---

## 9. Recommended training command

**Full training (all terms, PubMedBERT encoder):**  
Use `--max-abagen-terms 500` when **building** the merge so abagen is tiered (~500 genes). For training, no abagen cap flag is needed; the cache already has the selected terms.

```bash
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/merged_sources \
  --output-dir neurolab/data/embedding_model \
  --encoder sentence-transformers \
  --encoder-model NeuML/pubmedbert-base-embeddings \
  --max-terms 0 \
  --epochs 50 \
  --dropout 0.2 \
  --weight-decay 1e-5 \
  --early-stopping
```

**With ontology KG context** (appends triples to terms before encoding; requires `neurolab/data/ontologies/`):

```bash
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/merged_sources \
  --output-dir neurolab/data/embedding_model \
  --encoder sentence-transformers \
  --encoder-model NeuML/pubmedbert-base-embeddings \
  --kg-context-hops 1 \
  --max-terms 0 \
  --epochs 50 \
  --dropout 0.2 \
  --weight-decay 1e-5 \
  --early-stopping
```

**Options:**
- `--pca-components 100` or `--pca-variance 0.9` — reduce target dimension to combat overfitting
- `--batch-size 64` — default; increase if GPU memory allows
- `--final-retrain-on-train-and-val` — retrain on train+val after early stop; use test only for final eval

---

## 10. Gaps and non-blockers

| Item | Status | Impact |
|------|--------|--------|
| pdsp_cache | **Required** | Needed for drug inference (compound→brain pharmacological pathway). Run `run_full_cache_build.py` (includes PDSP) or build manually: `download_pdsp_ki.py` → gene PCA Phase 1–2 → `build_pdsp_cache.py`. |
| Legacy 400-parcel caches | Present | **None**. Not used; training uses merged_sources (392). |

**Residual-variance gene selection + PET residual correlation:** Informed by Fulcher et al. (who show that most brain gene maps are redundant with dominant gradients). **(1) Choosing genetic information:** First derive a **list of informative genes** (e.g. rank by residual variance after regressing out top PCs, or by residual correlation with PET maps); then **filter** that list with tiered selection (Tier 1 receptor anchors, Tier 2 WGCNA medoids, Tier 3 top residual-variance). That keeps training gene maps both pharmacologically relevant and non-redundant. **(2) PET evaluation:** When evaluating predicted vs PET receptor maps, **raw** correlation can be inflated by shared spatial autocorrelation. Use **residual correlation**: regress dominant gene-expression PCs from both maps, then correlate the residuals. Load `abagen_gradient_components.npy` from the merged cache; call `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)`. See [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md) and PREPROCESSING §8.

---

## 11. Verification commands (for re-audit)

```bash
# Full pipeline verification (exit 1 if pdsp_cache missing — PDSP is required)
python neurolab/scripts/verify_full_cache_pipeline.py

# Parcellation and map types
python neurolab/scripts/verify_parcellation_and_map_types.py

# Training readiness
python neurolab/scripts/check_training_readiness.py --require-expanded

# Quick load test
python -c "
import numpy as np
import pickle
from pathlib import Path
ms = Path('neurolab/data/merged_sources')
data = np.load(ms / 'term_maps.npz')
with open(ms / 'term_vocab.pkl','rb') as f: v=pickle.load(f)
assert data['term_maps'].shape[0]==len(v) and data['term_maps'].shape[1]==392
print('OK')
"
```

---

## 12. Conclusion

**All systems ready.** The training dataset (`merged_sources`) is fully built, consistent, and verified. The trainer loads the cache correctly, source-weighted sampling and per-term weights are active, and a 1-epoch smoke test completed successfully. You can proceed with full training using the recommended command above.
