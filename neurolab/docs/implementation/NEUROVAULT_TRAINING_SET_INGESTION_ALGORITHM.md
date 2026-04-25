# NeuroVault Training Set Ingestion Algorithm

A comprehensive specification for building a consistent, high-quality training set from NeuroVault and related sources. The core problem: **raw collections come at three aggregation levels** (subject, group, meta-analytic), and naively mixing them biases training toward collections with more individual maps.

---

## 1. The Three-Level Ingestion Pipeline

### Level 1: Already group-level (~95 collections)

**Action:** Use as-is. One map = one training sample. No processing.

**Examples:** HCP 457, Cognitive Atlas 1274, Kragel 3324, all Tier 2 meta-analyses, NARPS, atlases, connectivity, pharma.

**Reference:** See [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md) for the full list.

---

### Level 2: Subject-level maps (~25 collections)

**Action:** Average by contrast before adding to the cache.

**Grouping key:** `collection_id` + `contrast_definition` (or `cognitive_paradigm_cogatlas_id` if available, or parsed filename as fallback).

**Algorithm:**
```
for each collection in AVERAGE_FIRST:
    for each unique contrast in collection:
        maps = all subject maps for that contrast
        if len(maps) >= min_subjects:  # e.g. 3 (or 2 for heterogeneous like BrainPedia)
            avg_map = maps.mean(axis=0)
            training_set.append(avg_map, label=contrast_name)
```

**High-priority collections:** 1952, 6618, 2138, 4343, 16284

**Medium-priority:** 426, 445, 507, 2503, 4804, 504, 13042, 13705, 2108, 4683, 3887, 1516, 13474, 20510, 11646, 437, 12992, 19012, 6825, 1620

**Special handling:**
- **UCLA LA5C (4343):** Average healthy and clinical groups separately (130 healthy vs 142 clinical).
- **NeuroVault metadata:** `contrast_definition` is inconsistently populated. For IBC (6618/2138) it's clean; for BrainPedia (1952) or UCLA LA5C (4343), parse filenames or use `cognitive_paradigm_cogatlas_id`.

---

### Level 3: Meta-analytic maps

**Action:** Use as-is. Already super-averaged across studies.

**Note:** Higher SNR, less noise. Consider weighting them slightly higher in training (e.g. 2× loss weight) so they aren't diluted by noisier subject-averaged maps.

---

## 2. The 11-Stage Pipeline

### Stage 0 — Classification

Every collection gets tagged with:
- **Aggregation level:** `meta-analytic` | `group` | `subject-level`
- **Domain:** `cognitive` | `pharma` | `clinical` | `structural` | `connectivity` | `reference`

This drives all downstream decisions (averaging, weighting, sampling).

---

### Stage 1 — Parcellate

Raw NIfTI → 392-D vectors via Glasser+Tian atlas.

- Standard `nanmean` within each parcel mask.
- Output: `(n_images, 392)` float array.

---

### Stage 2 — QC (Quality Control)

Five rejection criteria:

| Criterion | Threshold | Reason |
|-----------|-----------|--------|
| All-zero | >95% zeros | Failed parcellation or empty mask |
| Too many NaN | >10% | Bad registration or missing data |
| Extreme values | \|T\| > 50 | Outlier or wrong map type |
| Constant map | std < 0.01 | No spatial variation |
| Wrong map type | p-value maps, atlases | Not activation maps |

---

### Stage 3 — Averaging

For the ~25 subject-level collections:

1. **Group** images by contrast key (priority: `contrast_definition` → `cognitive_paradigm_cogatlas_id` → parsed filename).
2. **Minimum subjects:** Require ≥3 per contrast (≥2 for heterogeneous collections like BrainPedia).
3. **Outlier removal:** If any map correlates <0.2 with the group mean, exclude it before averaging.
4. **Special:** UCLA LA5C — average healthy and clinical separately.

---

### Stage 4 — Normalization

Z-score each 392-D vector across parcels.

- Strips scale differences between T-maps, Z-maps, and betas.
- Model learns spatial patterns, not absolute magnitudes.

---

### Stage 5 — Labeling

Each map gets a text label for embedding.

**Priority order:**
1. CogAtlas concept name (if available)
2. Expanded `contrast_definition`
3. Collection name + parsed filename

**Abbreviation expansion:** WM→working memory, GT→greater than, etc., to improve embedding quality.

---

### Stage 6 — Deduplication

1. **Exact hash dedup:** Remove identical maps.
2. **Within-collection near-dedup:** If r > 0.95 between two maps in same collection, keep one.
3. **Cross-collection duplicates:** Keep (different labels = useful for training).

---

### Stage 7 — Stratified Sampling

**Problem:** Without stratification, training is dominated by NeuroVault task maps (~20K raw, ~2–3K after averaging). NeuroQuery (5–7K), NeuroSynth (3.4K), ENIGMA (49–100), pharma (632), neuromaps (40) get negligible gradient signal.

**Solution:** Source-stratified sampling. Each training batch draws equal proportions from each source:

| Source | Target % | Rationale |
|--------|----------|-----------|
| Cognitive (task) | 20% | Downsampled from ~88% |
| Pharmacological | 15% | Upsampled from ~3% |
| Clinical | 15% | Upsampled from ~0.2% |
| Meta-mixed | 10% | Meta-analytic + NeuroSynth |
| Emotion | 10% | Domain balance |
| Reference (receptor/atlas) | 5% | Gene expression, receptor maps |
| Other | 25% | Structural, connectivity, etc. |

**Loss weighting:**
- Meta-analytic maps: **2×** weight (higher SNR)
- Subject-averaged maps: **0.8×** weight
- Reference/atlas: **1.0×**

**Alternatives:**
- **Inverse-frequency weighting:** Weight each sample by `1/N_source`.
- **Two-stage training:** Train on full 20K+ for general topology, then fine-tune on curated pharma+receptor+disorder subset.

---

### Stages 8–10 — Embeddings, Save, Training Loop

- **Stage 8:** Pre-compute all text embeddings once (e.g. PubMedBERT).
- **Stage 9:** Save as NPZ (embeddings + maps + metadata).
- **Stage 10:** Weighted MSE training loop with stratified batch sampling.

---

## 3. Genetics / Gene Expression: Why Separate, Where It Goes

### Why gene expression was separated

Gene expression (abagen, 15,633 maps) is fundamentally different:

| NeuroVault map | Gene expression map |
|----------------|---------------------|
| "When people do 2-back, this is the activation pattern" | "Where gene GRIA1 is physically expressed in cortex" |
| Text label: "working memory" | Text label: "GRIA1 expression" — not a concept that clusters with "working memory" or "ketamine effects" |

**Problems:**
1. **Label mismatch:** "GRIA1 expression" doesn't align with cognitive/pharmacological concepts in embedding space.
2. **Scale:** 15,633 gene maps would dominate training, teaching "where genes are expressed" rather than "what brain patterns correspond to cognitive/pharmacological concepts."

### Where genetics goes: the pharmacological pathway

Genetics connects to brain activity through the **pharmacological pathway**:

```
Compound → receptor binding (PDSP Ki) → receptor gene expression (abagen) → brain map
```

**Concrete flow:**
1. **Gene PCA basis:** 250 receptor genes → 392-D expression maps → PCA → ~20–30 PCs (receptor space).
2. **Compound projection:** PDSP Ki profile → project through gene expression matrix → 392-D predicted brain map.
3. **Integration:** Fuse with Generalizer text prediction for compounds.

---

## 4. Integration Strategies for Gene Expression

### Strategy A: Gene maps as additional training samples (simplest, do first)

Add receptor gene maps and gene PCs to the training set with **rich text labels**.

| Vector | Label (example) |
|--------|-----------------|
| PC1 | "Cortical gradient of neurotransmitter receptor expression from sensory to association cortex" |
| PC2 | "Subcortical versus cortical monoamine receptor density gradient" |
| HTR2A | "Serotonin 5-HT2A receptor density across cortex, high in prefrontal and temporal association areas, primary target of psychedelics like psilocybin and LSD" |
| DRD2 | "Dopamine D2 receptor density, concentrated in striatum and limbic regions" |
| OPRM1 | "Mu-opioid receptor expression, thalamus and limbic regions" |

**Source:** Use `receptor_gene_list_v2.csv` and `receptor_knowledge_base.json` for labels. The `notes` and `gene_name` columns provide rich descriptions.

**Count:** 250 receptor genes + ~30 PCs ≈ 280 samples. Tag as `DomainTag.REFERENCE`, `AggLevel.ATLAS`. Stratified sampler gives reference maps **5%** of each batch.

**Implementation:** `build_abagen_cache.py --receptor-kb` already produces these. Add them to the expanded cache with `term_sources = "abagen"` (or a new `"reference"` source for receptor-only).

---

### Strategy B: Dual-pathway prediction with late fusion (recommended)

For compounds, fuse **text prediction** and **pharmacological pathway prediction**:

```
"piracetam enhances AMPA..." ──→ Generalizer (text → 392-D) ──→ predicted_text ──┐
                                                                                   ├──→ COMBINE → final 392-D
PDSP Ki profile [GRIA1: 500nM...] ──→ Pharma Pathway (Ki → gene PCA → 392-D) ──→ predicted_pharma ──┘
```

**Combination options:**
- Simple average: `final = 0.5 * predicted_text + 0.5 * predicted_pharma`
- Learned weighting: `final = α * predicted_text + (1-α) * predicted_pharma` (α trained)
- Gated: small network takes both predictions, outputs final map

**Ground truth:** NeuroVault pharma cache (632 maps), Luppi drug FC data.

**Prerequisites:** PDSP cache, gene PCA (Phase 1–2), `build_pdsp_cache.py`.

---

### Strategy C: Gene expression as learned embedding space (most ambitious)

Jointly learn text and gene pathways so they align:

- **Training objective:** For compound X with brain map Y:
  - Text pathway: `embed(mechanism_text_X) → Generalizer → predict Y`
  - Gene pathway: `Ki_profile_X × gene_expression_matrix → predict Y`
  - **Alignment loss:** Make text and gene predictions agree

Forces the model to learn that "5-HT2A agonist" (text) and "high Ki at HTR2A" (binding) produce the same brain map.

---

## 5. Recommended Implementation Order

1. **Immediate:** Strategy A — add 250 receptor genes + ~30 gene PCs with rich labels to training set. Tag as reference, 5% batch share.
2. **Next:** Implement Stages 0–7 (classification, parcellation, QC, averaging, normalization, labeling, dedup, stratified sampling) for NeuroVault.
3. **Then:** Build PDSP cache + gene PCA; implement Strategy B (dual-pathway fusion) for compounds.

---

## 6. Pseudocode: Averaging Step

```python
AVERAGE_FIRST = {
    1952, 6618, 2138, 4343, 16284,  # High priority
    426, 445, 507, 2503, 4804, 504, 13042, 13705,
    2108, 4683, 3887, 1516, 13474, 20510, 11646,
    437, 12992, 19012, 6825, 1620   # Medium priority
}

def ingest_collection(collection_id, maps):
    if collection_id in AVERAGE_FIRST:
        groups = defaultdict(list)
        for m in maps:
            key = (m.get('contrast_definition') or 
                   m.get('cognitive_paradigm_cogatlas_id') or 
                   m.get('name') or 
                   parse_contrast_from_filename(m['path']))
            groups[key].append(m['data'])  # 392-D vector
        
        result = {}
        for k, v in groups.items():
            if len(v) >= 3:  # min_subjects (2 for BrainPedia)
                arr = np.array(v)
                # Outlier removal: exclude maps with r < 0.2 to group mean
                mean_map = arr.mean(axis=0)
                corrs = [np.corrcoef(m, mean_map)[0,1] for m in arr]
                keep = [arr[i] for i in range(len(arr)) if corrs[i] >= 0.2]
                if len(keep) >= 2:
                    result[k] = np.mean(keep, axis=0)
        return result
    else:
        return {m['name']: m['data'] for m in maps}
```

---

## 7. References

- [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md) — Full collection list and average-first IDs
- [NeuroVault acquisition guide](NeuroVault%20acquisition%20guide%20for%20brain%20activation%20prediction%20training.md) — Tier structure and collection metadata
- [receptor_gene_list_v2.csv](receptor_gene_list_v2.csv) — Receptor gene labels (gene_name, system, notes)
- [BUILD_MAPS_AND_TRAINING_PIPELINE.md](BUILD_MAPS_AND_TRAINING_PIPELINE.md) — Abagen, gene PCA, sample weighting
