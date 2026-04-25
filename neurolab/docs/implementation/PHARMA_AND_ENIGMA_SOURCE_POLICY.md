# Pharma and ENIGMA Source Policy — Train Recovery Fixes

Diagnosis and fixes for pharma_neurosynth (~0.07), enigma (-0.03), neurovault_pharma (~0) train recovery.

---

## 1. pharma_neurosynth — Why It Fails

**Current terms (10):** dose, double blind, challenge, inhibitor, metabolism, positron emission, drugs, ampa, dependence, dopaminergic

These are **methodological / pharmacology keywords**, not drug contrasts:
- Broad, appear across many unrelated papers
- No specific compound (ketamine, caffeine, etc.)
- Meta-analytic maps become diffuse / inconsistent
- Near-zero train corr is expected

### Fixes

**A. Exclude generic terms from supervision**

Add `--exclude-generic-terms` to `build_pharma_neurosynth_cache.py`:
- Filter out: dose, double blind, challenge, inhibitor, metabolism, positron emission, drugs, dependence, dopaminergic, ampa (when used as generic)
- Keep only compound names and tight classes: ketamine, caffeine, SSRI, etc.

**B. Rebuild with compound-only vocabulary**

Use `--pharma-terms-json` with `high_confidence_terms` (compound names only). Or:
- `--terms ketamine caffeine nicotine ...` (explicit list)
- Higher `--min-studies` for generic words; lower for rare compounds

**C. Reclassify as "pharma-methods"**

If keeping generic terms: use for retrieval/keyword expansion only, not regression training. Set `source_weight = 0` or exclude from training.

---

## 2. enigma — Why It Goes Negative

**Problems:**
- **Type mismatch:** Structural maps (CT/SA/SubVol) vs fMRI activation semantics
- **Sign/direction ambiguity:** "schizophrenia cortical thickness" — patients < controls or controls < patients?
- **Scale mismatch:** CT vs SA vs SubVol have different distributions

### Fixes (implemented)

**A. Make direction explicit in labels**

```text
"schizophrenia cortical thickness (patients - controls)"
```

ENIGMA convention: positive Cohen's d = patients have larger values. Added to `build_enigma_cache.py` via `--add-direction`.

**B. Type-conditioning**

Ensure `term_map_types.pkl` has `enigma` → `structural`. Already in place. Verify type one-hot is concatenated to MLP input.

**C. Normalize structural targets**

Already done: `zscore_cortex_subcortex_separately` per map.

---

## 3. neurovault_pharma — Why It's ~0

**Root cause:** Drug names are not in the labels.
- Terms are cognitive constructs + `regparam`: "anhedonia regparam", "reason regparam"
- The "pharma" is in study design (drug, dose, condition), not in the text
- Many maps share identical/similar text → embeddings collapse → head can't disambiguate

### Fixes

**A. Inject pharma context into labels**

Template: `drug: X | condition: Y | regparam: Z`

Where to get it:
- `--pharma-add-drug`: extract drug from collection name (already exists; needs `collections_meta` with descriptive names)
- Image metadata: `contrast_definition`, `description`, `name` — parse for drug/condition
- Collection-level description

**B. Expand PHARMA_DRUGS and collection metadata**

- Ensure `download_neurovault_pharma` fetches `collections_meta` with full names
- Add more drugs to `PHARMA_DRUGS` in `build_neurovault_cache.py`
- Prepend drug when available: `"ketamine | anhedonia regparam"` instead of `"anhedonia regparam (Encoding with Cogni)"`

**C. If drug context unavailable**

Reclassify: not "pharma supervision." Put in generic NeuroVault pool or exclude from training.

---

## 4. One Decision Rule

> If the map's meaning depends on hidden metadata (drug, dose, condition) and that metadata is not present in the text → don't use it as supervision until you add the metadata to the label.

---

## 5. Implementation Status

| Fix | Script | Status |
|-----|--------|--------|
| ENIGMA direction | build_enigma_cache.py | `--add-direction` (default on): appends " (patients - controls)" |
| pharma_neurosynth generic exclude | build_pharma_neurosynth_cache.py | `--exclude-generic-terms`: filters dose, double blind, etc. |
| neurovault_pharma drug prepend | build_neurovault_cache.py | `--pharma-add-drug`: now "drug: label" (drug first) |

### Rebuild commands

```bash
# ENIGMA with direction (default)
python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache

# pharma_neurosynth without generic terms
python neurolab/scripts/build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache --exclude-generic-terms

# neurovault_pharma (drug from collection name; requires collections_meta in manifest)
python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_pharma_data --output-dir neurolab/data/neurovault_pharma_cache --from-downloads --pharma-add-drug --cluster-by-description
```
