# Train Recovery Diagnostic — 2026-02-23

Diagnostic checklist for collections/sources with poor train recovery. Run:
```bash
python neurolab/scripts/diagnose_collection_failures.py
```

---

## Summary of Findings

### Bad five collections (426, 437, 507, 555, 2508)

| Collection | n | Train corr | [A] Embed collapse | [B] Target degeneracy | [C] Map type |
|------------|---|------------|--------------------|------------------------|--------------|
| 426 | 6 | -0.14 | 0.87 (OK) | OK | activation |
| 437 | 20 | -0.01 | 0.77 (OK) | OK | subnetwork |
| 507 | 4 | -0.10 | 0.87 (OK) | OK | activation |
| 555 | 10 | 0.01 | 0.86 (OK) | OK | P-map (meta) |
| 2508 | 3 | 0.00 | 0.78 (OK) | OK | **ROI/mask** |

**Key finding:** None of 426/437/507/555 show severe embedding collapse (>0.95) or target degeneracy. The most likely culprit is **small n + high embedding similarity (0.77–0.87)** — the model struggles to separate them. **2508** is clearly ROI/mask (ROI_ACC, ROI_DLPFC, ROI_Striatum) → **wrong supervision type** for regression.

### ENIGMA sign test

- `corr(pred, target)`: 0.01 | `corr(pred, -target)`: -0.01  
- Sign flip does **not** help. ENIGMA failure is not primarily a sign mismatch.

---

## Labels by collection (for reference)

### 426 — Theory of mind
- `fMRI: false belief question`
- `fMRI: false picture question`
- `fMRI: false picture story`
- `fMRI: false belief question - false picture question`
- `fMRI: false belief - false picture`
- `fMRI: false belief story - false picture story`

### 437 — Consensus subnetworks
- `fMRI: subnetwork NT01   Default Mode` … NT12  
- `fMRI: subnetwork AS01` … AS12 (opaque labels)

### 507 — Main/control experiment
- `fMRI: main plus control experiment`
- `fMRI: main experiment`
- `fMRI: control experiment`
- `fMRI: main minus control experiment`

### 555 — Meta-analysis (P-maps)
- `fMRI: participants with overweight/obesity > lean individuals`
- `fMRI: participants with substance addictions > control individuals`
- … (paired > / < contrasts)

### 2508 — Cannabis ROI
- `fMRI: ROI_ACC`
- `fMRI: ROI_DLPFC`
- `fMRI: ROI_Striatum`

---

## Recommended fixes (by priority)

### 1. 2508 — Exclude from regression

ROI/mask maps are **not** activation regression targets. Either:
- Add to `EXCLUDE_FROM_REGRESSION` list and serve via retrieval, or
- Train a separate overlap/BCE head for mask-like targets.

### 2. Improve labels for 426, 437, 507, 555

- **426**: Labels are already good; consider adding collection context (e.g. “Theory of mind: false belief question”).
- **437**: AS01–AS12 are opaque; add network names from metadata if available.
- **507**: Add task/collection context (e.g. “Main experiment: …”).
- **555**: Labels already have direction; verify P-map transform and QC.

### 3. Collection policy switch

- **activation/stat map** → regression training
- **P-map** → -log10(p) + regression (verify it’s truly p)
- **ROI/mask/connectivity** → exclude from regression or separate loss

### 4. ENIGMA

- Add direction to labels (e.g. “cortical thickness reduction (patients < controls)”).
- Or use abs(effect) targets and record direction separately.

### 5. neurovault_pharma / pharma_neurosynth

- Richer labels (e.g. “ketamine (NMDA antagonist) fMRI: drug > placebo”).
- Type-conditioned input is already in place; ensure labels carry enough context.

---

## Scripts

- `neurolab/scripts/diagnose_collection_failures.py` — run full diagnostic
- `neurolab/scripts/report_train_correlation_by_collection.py` — train corr by source/collection
