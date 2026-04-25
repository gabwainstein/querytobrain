# Cache and Term Quality Audit — 2026-02-23

**Context:** Post-rebuild audit of NeuroLab caches and term quality, comparing against TRAINING_SET_GAP_AUDIT and TERM_INFORMATIVENESS_AUDIT.

---

## 1. Cache Status Summary

| Cache | Maps | Parcels | Status |
|-------|------:|--------:|--------|
| decoder_cache (NeuroQuery) | 4,983 | 427 | OK |
| neurosynth_cache | 3,228 | 427 | OK |
| unified_cache | 8,211 | 427 | OK |
| **merged_sources (training)** | **13,419** | **427** | **OK** |
| neurovault_cache | 4,338 | 427 | OK |
| neurovault_pharma_cache | 623 | 427 | OK |
| neuromaps_cache | 40 | 427 | OK |
| enigma_cache | 52 | 427 | OK |
| abagen_cache | 15,633 | 427 | OK |
| pharma_neurosynth_cache | 26 | 427 | OK |
| pdsp_cache | 1,780 | 392 | MISMATCH (expected 427) |
| gene_pca | 40 | 392 | MISMATCH (expected 427) |
| fc_cache | 8 | 392 | MISMATCH (expected 427) |

**Atlas:** combined_atlas_427.nii.gz (Glasser 360 + Tian 32 + BFB/Hyp) — OK

---

## 2. Term Quality (merged_sources)

### Data quality (TRAINING_SET_GAP_AUDIT checks)

| Check | Result |
|-------|--------|
| NaN rows | **0** |
| Near-zero rows (max &lt; 1e-10) | **0** |
| Broken/placeholder labels (verify_term_labels) | **None** |

### Source breakdown

| Source | Count |
|--------|------:|
| direct (NQ+NS merged) | 8,211 |
| neurovault | 3,958 |
| neurovault_pharma | 621 |
| abagen | 505 |
| enigma | 52 |
| neuromaps | 31 |
| neuromaps_residual | 31 |
| pharma_neurosynth | 10 |

### Term informativeness (TERM_INFORMATIVENESS_AUDIT)

| Issue | Audit recommendation | Current status |
|-------|----------------------|----------------|
| `[colN]` prefix | Strip — meaningless to encoders | **Fixed** — 0 terms with prefix |
| WM atlas `tstatA`/`tstatB` | Strip to cognitive term | **Fixed** — 0 terms |
| `neurovault_image_N` placeholders | Discard or enrich | **Fixed** — 0 terms |
| `fMRI: test` / `test2` junk | Filter | **Fixed** — 0 terms |
| `trm_` opaque (Cognitive Atlas task IDs) | Enrich from API or keep with note | **10 remaining** (low priority) |
| Movie IDs (col20820) | Relabel as "HCP movie-watching tICA component N" | Not checked in this audit |

---

## 3. NeuroVault Label Improvements

**improve_neurovault_labels.py** ran after NeuroVault build (per rebuild pipeline). Verified on neurovault_cache:

- `[colN]` prefix: **0** (stripped)
- WM `tstatA`/`tstatB`: **0** (stripped)
- `neurovault_image_` placeholders: **0**
- Sample terms: `incorrect response (task)`, `overt verb generation`, `finger` — readable cognitive labels

---

## 4. Gaps and Recommendations

### Resolved (from prior audits)

- **ENIGMA NaN:** 0 NaN rows in merged_sources (sanitization in build_enigma_cache applied or data clean)
- **neurovault_pharma near-zero:** 0 near-zero rows (QC or filtering effective)
- **NeuroVault label cleanup:** `[colN]`, tstat, placeholders — all applied

### Remaining gaps

1. **Decoder cache:** 4,983 terms (&lt; ~7.5K full vocab). Rebuild with `--max-terms 0` and `--n-jobs 1` for reliability if full vocab desired.
2. **pdsp_cache, gene_pca, fc_cache:** 392 parcels vs 427. Rebuild with current atlas if needed for drug-inference or gene-head training.
3. **10 trm_ opaque terms:** Cognitive Atlas task IDs; maps valid, labels opaque. Optional: enrich via Cognitive Atlas API.
4. **decoder_cache_expanded:** Not built (optional ontology path).

### Non-blocking

- Legacy caches (cache_brainpedia_plus_decoder, etc.) at 400 parcels — not used for primary training path.
- PermissionError on nilearn joblib (NeuroSynth) — cache valid; consider `--n-jobs 1` for NeuroSynth on Windows if recurring.

---

## 5. Training Readiness

**Ready for training.** Primary path uses merged_sources (13,419 terms × 427 parcels). No NaN, no near-zero maps, no broken labels. Term quality improvements from TERM_INFORMATIVENESS_AUDIT are applied.

**Next step:**
```bash
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/merged_sources \
  --output-dir neurolab/data/embedding_model \
  --encoder sentence-transformers \
  --encoder-model NeuML/pubmedbert-base-embeddings \
  --max-terms 0 --epochs 50 --dropout 0.2 --weight-decay 1e-5
```
