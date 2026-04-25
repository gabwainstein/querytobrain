# Pipeline Changes Audit: Good Training → Abagen Improvements → Pharma Improvements

**Date:** 2026-02-21  
**Purpose:** Trace what changed between the "good" training run and the current state, after abagen improvements and neurovault pharma improvements. Identify root causes of any performance regression.

---

## Executive Summary

| Change | Before | After | Impact |
|-------|--------|-------|--------|
| **Pharma source** | `neurovault_pharma_data` (keyword search) | `neurovault_curated_data` + schema ([SCHEMA_INDEX.md](SCHEMA_INDEX.md#pharma-collection-schema), 22 collections) | **621 → 153 terms (~75% drop)** |
| **Pharma labels** | Improved only | Improved + schema relabel ([SCHEMA_INDEX.md](SCHEMA_INDEX.md#pharma-collection-schema)) (prefer improved) | Schema prefixes hurt encoder; merge prefers improved |
| **Abagen** | 505 terms (tiered selection) | 617 terms (500 + additional cache + 3 gradient PCs) | Slight increase |
| **Pipeline** | `run_full_cache_build` or `rebuild_all_caches` | `run_full_cache_build` only (curated path) | Pharma path diverged |

**Root cause of regression:** The pharma "improvement" replaced the legacy keyword-search pharma (623 terms) with a curated subset (22 collections → 153 terms). This **~75% reduction in pharma training data** is the primary driver of any performance drop. Schema relabeling (drug=X|control=Y prefixes) was tried but hurt encoder embeddings; the merge now prefers `neurovault_pharma_cache_improved` over `relabeled`.

---

## 1. Timeline of Changes

### 1.1 Good Training (before improvements)

**Evidence:** CACHE_AUDIT_2026-02-23, TRAINING_DATA_METHODOLOGY_AUDIT, TRAINING_SET_GAP_AUDIT

| Source | Terms | Data path | Build |
|--------|------:|-----------|-------|
| neurovault_pharma | **621** | neurovault_pharma_data | download_neurovault_pharma → build_neurovault_cache |
| abagen | **505** | abagen_cache | build_abagen_cache + tiered selection (max 500) |
| neurovault | 3,958 | neurovault_curated_data | build_neurovault_cache |
| direct | 8,211 | unified_cache | decoder + neurosynth merge |
| neuromaps | 31 | neuromaps_data | build_neuromaps_cache |
| enigma | 52 | enigmatoolbox | build_enigma_cache |
| pharma_neurosynth | 10 | neurosynth_data | build_pharma_neurosynth_cache |

**Pharma path:** `download_neurovault_pharma.py` searched NeuroVault for keywords (drug, ketamine, LSD, pain, opioid, cannabis, etc.) → `neurovault_pharma_data` → `build_neurovault_cache` → `neurovault_pharma_cache` (623 terms). Some collections were "polluted" (e.g. 1257 NeuroSynth Encoding) but the total was large.

### 1.2 Abagen Improvements (before pharma changes)

**What changed:**
- Added `abagen_cache_receptor_residual_selected_denoised` as `--additional-abagen-cache-dir` (181 terms)
- Tiered selection: Tier 1 receptor, Tier 2 cluster medoids (32), Tier 3 residual_variance
- 3 gradient PCs (`--abagen-add-gradient-pcs 3`)
- 95% PCA denoising (`--abagen-pca-variance 0.95`)
- Gene label enrichment (gene_info.json, receptor KB)

**Result:** 505 → 617 abagen terms (500 from main + ~117 from additional after dedup, + 3 gradient PCs). Abagen improved; no regression from this step.

### 1.3 Pharma Improvements (when regression occurred)

**What changed:**
- **Source switch:** `neurovault_pharma_data` (keyword search) → `neurovault_curated_data` + `neurovault_pharma_schema.json` (22 curated collections)
- **Rationale:** Legacy keyword search returned polluted collections (e.g. 1257). Curated list ensures only true drug/placebo/task-on-drug collections.
- **Pipeline:** `run_full_cache_build.py` now **never** uses `neurovault_pharma_data`. It uses `neurovault_curated_data` + `_pharma_collection_ids()` from schema.
- **Build:** `build_neurovault_cache --data-dir neurovault_curated_data --collections <22 ids>` → `neurovault_pharma_cache` (153 terms)
- **Post-build:** improve_neurovault_labels → `neurovault_pharma_cache_improved`; relabel_pharma_terms → `neurovault_pharma_cache_relabeled`
- **Merge preference:** `improved` > `relabeled` > base ([schema prefixes](SCHEMA_INDEX.md#pharma-collection-schema) hurt encoder; see run_full_cache_build line 259–262)

**Result:** 621 → 153 pharma terms in merged_sources. **~75% reduction.**

---

## 2. Pipeline Divergence

### 2.1 run_full_cache_build.py (current primary path)

- **Pharma:** Uses `neurovault_curated_data` + [schema collections](SCHEMA_INDEX.md#pharma-collection-schema) only. **Ignores neurovault_pharma_data.**
- **Merge:** Prefers `neurovault_pharma_cache_improved` over `relabeled` ([schema prefixes](SCHEMA_INDEX.md#pharma-collection-schema) hurt encoder).

### 2.2 rebuild_all_caches.py (legacy path)

- **Pharma:** Uses `neurovault_pharma_data` when it exists. Runs `download_neurovault_pharma` flow.
- **Merge:** Uses whatever neurovault_pharma_cache exists (improved/relabeled/base).

So: **run_full_cache_build** = curated pharma only. **rebuild_all_caches** = legacy pharma when neurovault_pharma_data exists.

---

## 3. Curated vs Legacy Pharma

| Aspect | Legacy (neurovault_pharma_data) | Curated (schema, see SCHEMA_INDEX.md#pharma-collection-schema) |
|--------|--------------------------------|------------------|
| **Source** | download_neurovault_pharma.py (keyword search) | download_neurovault_curated.py --all (includes PHARMA) |
| **Collections** | Many (keyword matches; some polluted) | 22 (explicit list in schema (SCHEMA_INDEX.md#pharma-collection-schema)) |
| **Terms** | 623 | 153 |
| **Pollution** | Yes (e.g. 1257) | No (hand-picked) |
| **Excluded** | — | 3264 (control task battery), 2508 (cannabis meta) |

**Schema collections (22):** 1083, 12212, 17403, 8306, 3902, 4040, 4041, 9246, 5488, 3666, 3808, 13312, 13665, 4414, 12992, 9206, 9244, 20308, 3713, 1501, 1186, 19291.

---

## 4. Current Training Report (2026-02-24)

From `train_report_history.csv`:

| Source | train_r | test_r | n |
|--------|---------|--------|---|
| direct | 0.84 | 0.66 | 7390 |
| neurovault | 0.48 | 0.27 | 2389 |
| **neurovault_pharma** | **-0.002** | **0.04** | **138** |
| **abagen** | **0.04** | **0.02** | **617** |
| enigma | -0.02 | -0.02 | 47 |

Pharma and abagen both show very low train/test correlation. Abagen at 617 terms (improved) but still weak. Pharma at 138 terms (curated) with near-zero correlation.

---

## 5. Recommendations

### 5.1 Restore pharma volume (if regression is the priority)

**Option A — Hybrid:** Keep curated quality but add more collections. Expand `neurovault_pharma_schema.json` and `download_neurovault_curated.py` PHARMA list with additional drug/placebo collections from NeuroVault. Audit for pollution before adding.

**Option B — Legacy fallback:** Add a `--use-legacy-pharma` flag to `run_full_cache_build` that, when `neurovault_pharma_data` exists, builds pharma from it instead of curated. Merge would get 623 terms again. Document that legacy may include polluted collections.

**Option C — Union:** Build both curated and legacy pharma caches; merge both into merged_sources (with dedup by normalized label). Curated for quality, legacy for volume.

### 5.2 Schema labels

- **Current:** Merge prefers `improved` (no schema prefix) because `drug=X|control=Y` hurt encoder.
- **If relabeled is desired:** Consider shorter prefixes or encoder fine-tuning on schema-style labels. Or keep improved for training.

### 5.3 Abagen

- Abagen pipeline is sound. 617 terms from tiered selection + additional cache + gradient PCs.
- For more abagen: `--max-abagen-terms 0` adds all 15,638 (run merge with this for abagen-heavy training).

### 5.4 Reproducibility

- Document which pipeline produced the "good" training: `run_full_cache_build` vs `rebuild_all_caches`, and whether `neurovault_pharma_data` existed.
- If good run used legacy pharma, Option B or C above restores that data.

---

## 6. Verification Commands

```bash
# Check current pharma term count
python -c "
import pickle
from pathlib import Path
p = Path('neurolab/data/merged_sources')
with open(p/'term_sources.pkl','rb') as f: src=pickle.load(f)
from collections import Counter
print('neurovault_pharma:', Counter(src).get('neurovault_pharma', 0))
print('abagen:', Counter(src).get('abagen', 0))
"

# Check which pharma cache merge uses
ls -la neurolab/data/neurovault_pharma_cache*/
# run_full_cache_build prefers: improved > relabeled > base

# Restore legacy pharma (if neurovault_pharma_data exists)
python neurolab/scripts/download_neurovault_pharma.py --output-dir neurolab/data/neurovault_pharma_data
# Then use rebuild_all_caches OR add --use-legacy-pharma to run_full_cache_build
```

---

## 7. Summary

| Finding | Detail |
|---------|--------|
| **Pharma drop** | 621 → 153 terms from switching to curated-only (22 collections) |
| **Abagen** | 505 → 617 terms; improved, not regressed |
| **Schema labels** | Hurt encoder; merge prefers improved (no prefix) |
| **Pipeline split** | run_full_cache_build uses curated only; rebuild_all_caches uses legacy when neurovault_pharma_data exists |
| **Fix** | Restore pharma volume via hybrid, legacy fallback, or union |
