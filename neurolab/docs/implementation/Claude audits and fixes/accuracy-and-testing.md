# Accuracy and Testing

This document defines **how we measure accuracy** and **the process to run tests** for NeuroLab pipelines. Every pipeline that reports performance should align with these definitions and be reproducible using the steps below.

**Atlas:** All evaluation is in **392-D** parcel space (Glasser 360 cortex + Tian S2 32 subcortex). Legacy references to 400-D / Schaefer are obsolete.

---

## 1. How we define accuracy

Accuracy is **context-dependent** by component. We use the following definitions consistently.

### 1.1 Text-to-brain embedding (primary model)

**What we measure:** How well the model predicts a parcellated brain map (392-D) for a given text term.

**Primary metric:** **Mean Pearson correlation** — For each (term, ground-truth map) pair, compute the Pearson correlation between the predicted 392-D vector and the ground-truth 392-D vector; average over terms. Reported as Train mean correlation, Val mean correlation, Test mean correlation.

**Secondary metric:** **MSE** — Mean squared error between predicted and ground-truth maps. Reported as Val MSE, Test MSE.

**Interpretation:**

- **Test mean correlation** = unbiased estimate of generalization to unseen terms (held-out test set). This is the main number for model comparison.
- **Train mean correlation** = fit quality. The gap (train − test) indicates overfitting; should be small.
- All correlations and MSE are computed in **392-D parcel space** (after inverse PCA if PCA was used during training).

### 1.2 Per-source evaluation (multi-source training set)

The training set includes multiple data sources (direct/NeuroQuery, NeuroSynth, NeuroVault, neuromaps PET, ENIGMA structural, abagen gene expression, pharma). Report test correlation **broken down by source** to identify where the model excels or fails:

```
Overall test correlation: X.XX
  direct:            X.XX (N terms)
  neurovault:        X.XX (N terms)
  neuromaps:         X.XX (N terms)
  abagen:            X.XX (N terms)
  enigma:            X.XX (N terms)
  neuromaps_residual: X.XX (N terms)
  pharma_neurosynth: X.XX (N terms)
  neurovault_pharma: X.XX (N terms)
```

Low correlation on neuromaps or abagen relative to direct/neurovault may indicate the model is underfitting pharmacological/genetic targets (consider increasing their loss weight or sampling fraction).

### 1.3 Per-map-type evaluation (type-conditioned MLP)

The model uses type-conditioned inputs (fmri_activation, structural, pet_receptor). Report test correlation per map type:

```
fmri_activation:  X.XX (N terms)
structural:       X.XX (N terms)
pet_receptor:     X.XX (N terms)
```

### 1.4 Residual correlation (Fulcher-style — pharmacological specificity)

**Purpose:** Raw correlation between predicted and PET receptor maps can be inflated by shared spatial autocorrelation (both follow dominant cortical gradients). Residual correlation measures whether the model learned **pharmacologically specific** patterns beyond these gradients.

**Method:** For each (predicted, PET ground-truth) pair:
1. Load `abagen_gradient_components.npy` (K × 392, from merged_sources).
2. Project out the first K gene-expression PCs from both prediction and ground truth.
3. Compute Pearson correlation between the residuals.

**Implementation:** `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)`

**Interpretation:** High raw correlation + low residual correlation = model learned the gradient shortcut (predicts cortical hierarchy for everything pharmacological). High residual correlation = model learned receptor-specific spatial patterns. Report both raw and residual correlation for all PET/receptor test terms.

**Reference:** Fulcher et al. (Nat Commun 2021) — overcoming false-positive gene-category enrichment in spatial transcriptomics.

### 1.5 Gene head evaluation (pharmacological pathway)

When the gene head is active (predicts abagen terms via PCA loadings), report:
- **Gene head test correlation**: mean correlation of gene head predictions (after PCA inverse transform) vs ground-truth gene expression maps, on held-out abagen terms.
- **Gene head reconstruction error**: MSE in PC space (before inverse transform).

### 1.6 Cognitive decoder (term maps cache)

**What we measure:** How well an input parcellated map matches cached term maps.

- **Metric:** For a given map, correlate with each cached term map and return top-N terms by correlation. No single accuracy number; quality assessed by whether top terms are semantically plausible.
- **Verification:** `verify_decoder.py` loads the cache and runs decode on sample maps.

### 1.7 Biological / receptor enrichment

**What we measure:** Spatial correlation between a parcellated map and receptor (or other biological) density maps.

- **Metric:** Pearson r per receptor between input map and each receptor map. Top biological hits (receptor name, system, r, p).
- **Caveat:** Use real receptor data (neuromaps cache or Hansen PET atlas), not placeholder data.

### 1.8 Scope guardrail

- **Metric:** Score = max cosine similarity of query embedding to training-term embeddings. in_scope = score ≥ threshold.

---

## 2. Train / val / test and evaluation protocol

### 2.1 Data split

- **Source:** Merged training cache (`merged_sources/term_maps.npz`, `term_vocab.pkl`, `term_sources.pkl`).
- **Split:** Random split (fixed seed; `--seed 42`). Default 80% train / 10% val / 10% test. Saved as `split_info.pkl`.
- **Stratification:** When source-weighted sampling is active, the split is random across all sources. Each source appears in train/val/test proportionally. Verify that rare sources (neuromaps, enigma) have test examples.

### 2.2 Optional: final model on train+val

If `--final-retrain-on-train-and-val` is set, retrain on train ∪ val, save that model, report Test correlation from the retrained model. Test remains unbiased.

### 2.3 Reporting

Every training run prints:
- **Train recovery** — mean correlation on training terms.
- **Test generalization** — mean correlation on held-out test terms.
- **Summary:** Train = X.XX, Test = Y.YY, gap = X − Y.
- **Per-source** and **per-map-type** breakdowns (§1.2, §1.3).
- **Residual correlation** for PET test terms (§1.4).

---

## 3. Process to run tests (reproducible)

### 3.1 Build training data

See [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) for full build. Summary:

```bash
# Build atlas + per-source caches + merge
python neurolab/scripts/run_full_cache_build.py
# Or: step-by-step (see BUILD_METHODS §12)
```

### 3.2 Train text-to-brain model

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
  --early-stopping \
  --final-retrain-on-train-and-val
```

**Output:** Printed Train, Val, Test correlation (overall + per-source + per-map-type). Saved model, split_info, guardrail embeddings.

**Encoder comparison:** PubMedBERT (`NeuML/pubmedbert-base-embeddings`) is recommended. Previous results on the old 400-D decoder-only cache showed test ~0.64 with PubMedBERT vs ~0.55 with TF-IDF/SciBERT/MiniLM. Expected baselines for the new 392-D multi-source pipeline have not yet been established; record them after the first full training run.

### 3.3 Evaluate pharmacological specificity

After training, evaluate PET predictions with residual correlation:

```python
from neurolab.evaluation_utils import residual_correlation
import numpy as np

gradient_pcs = np.load("neurolab/data/merged_sources/abagen_gradient_components.npy")  # (K, 392)
# For each PET test term:
r_raw = np.corrcoef(pred, target)[0, 1]
r_residual = residual_correlation(pred, target, gradient_pcs)
```

### 3.4 Verify pipelines (smoke tests)

- **Decoder:** `python neurolab/scripts/verify_decoder.py`
- **Unified:** `python neurolab/scripts/verify_unified.py`
- **Embedding:** `python neurolab/scripts/verify_embedding.py`
- **Full pipeline:** `python neurolab/scripts/verify_full_cache_pipeline.py`

### 3.5 Diagnose generalization

```bash
python neurolab/scripts/diagnose_generalization_ceiling.py --cache-dir neurolab/data/merged_sources
```

Reports best-map ceiling (per test term, best correlation achievable by any train map) and nearest-neighbor baseline.

---

## 4. Expected baselines (update after first training run)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall test correlation | TBD | PubMedBERT on merged_sources (~14K terms) |
| direct test correlation | TBD | NeuroQuery/NeuroSynth terms |
| neurovault test correlation | TBD | Task fMRI terms |
| neuromaps test correlation (raw) | TBD | PET receptor maps |
| neuromaps test correlation (residual) | TBD | Fulcher-corrected |
| abagen test correlation | TBD | Gene expression maps |
| Train-test gap | TBD | Should be < 0.15 |

**Historical (legacy 400-D decoder-only):** PubMedBERT test ~0.64, TF-IDF test ~0.55. These are not directly comparable to the new pipeline.

---

## 5. Implicit rule for new pipelines

Whenever we add a new pipeline or model that reports performance:

1. **Define accuracy**: state metric, unit, and what "good" means.
2. **Define evaluation protocol**: data split, held-out test, leakage prevention.
3. **Document exact commands** to reproduce (as in §3).
4. **Cross-link** this doc from the pipeline's README.

---

## 6. References

- **Training dataset build methods:** [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md)
- **Preprocessing and training pipeline:** [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md)
- **Abagen tiered gene selection / Fulcher:** [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md)
- **Training readiness audit:** [TRAINING_READINESS_AUDIT.md](TRAINING_READINESS_AUDIT.md)
