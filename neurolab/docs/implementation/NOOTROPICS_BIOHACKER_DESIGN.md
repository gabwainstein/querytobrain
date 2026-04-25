# Nootropics/Biohacker Product Design: Term Schema & Multi-Head Training

This document specifies the **term object schema**, **embedding strategy**, and **training recipe** for a single user-facing "text → brain map" interface that serves nootropics/biohacker queries while cleanly separating activation, structural, and biology supervision.

---

## 1. Product North Star

**One interface:** User asks a question → gets (1) a brain map and (2) evidence/provenance.

**Query modes users expect:**
- **Mechanism-first:** "What brain systems are most consistent with increasing noradrenergic tone?"
- **Compound-first:** "What would ibuprofen 600 mg likely modulate in emotion tasks?"
- **Goal-first:** "Improve sustained attention without anxiety — what circuits are implicated?"

**Output per query:**
- Predicted map + uncertainty (or confidence)
- Provenance: which sources contributed (NeuroVault vs meta-analyses vs connectivity)
- Mechanistic annotation: receptors/transporters → circuits (when possible)
- Comparable evidence: nearest-neighbor studies/maps and why they match

---

## 2. Term Object Schema (Structured Fields)

Each training term should carry structured metadata so dose/route/design can be used as **first-class conditioning**, not accidental text.

### 2.1 Core Fields (all terms)

| Field | Type | Description | Used for |
|-------|------|-------------|----------|
| `label` | str | Natural-language label (what gets embedded) | Encoder input |
| `map_type` | enum | `fmri_activation` \| `structural` \| `pet_receptor` \| `gene_expression` | Head routing |
| `source` | str | `direct` \| `neurovault` \| `neurovault_pharma` \| `enigma` \| `neuromaps` \| `receptor` \| `abagen` \| ... | Sample weighting, provenance |
| `map` | ndarray | (n_parcels,) parcellated vector | Regression target |

### 2.2 Pharmacology / Dose Fields (when applicable)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `drug` | str | Compound name (ground to ChEBI/standard ID when possible) | `ibuprofen`, `caffeine` |
| `dose_mg` | float \| list | Dose in mg (or list for multiple arms) | `600`, `[200, 600]` |
| `route` | str | Oral, IV, etc. | `oral` |
| `design` | str | Study design flags | `placebo-controlled`, `randomized`, `crossover` |
| `population` | str | Healthy vs clinical | `healthy`, `chronic pain` |
| `timing` | str | Acute vs chronic | `acute`, `chronic` |
| `task` | str | Task domain if fMRI | `emotion`, `pain`, `working memory` |

### 2.3 Provenance Fields

| Field | Type | Description |
|-------|------|-------------|
| `collection_id` | int | NeuroVault collection ID (when from NeuroVault) |
| `image_id` | int | NeuroVault image ID |
| `doi` | str | Publication DOI if available |
| `dataset` | str | Source dataset (e.g. `neurovault`, `neuroquery`, `enigma`) |

### 2.4 Synthesized Label (for embedding)

The `label` field is what the encoder sees. For pharmacology terms, synthesize from:

```
"{drug} {dose_mg} mg, {task}, {design}"
```

Example: `"ibuprofen 600 mg, emotion task, placebo-controlled"`

**Rule:** Always include collection metadata (title + description) when generating labels. The LLM relabel step should receive:
- `image_name`, `contrast_definition`
- `collection_name`, `collection_description`
- Parsed fields: `drug`, `dose`, `design`, `task`

---

## 3. Map Types and Head Routing

### 3.1 Map Type Enum (5 types for clean separation)

| Type | Sources | Target distribution | Loss | Head |
|------|---------|---------------------|------|------|
| `fmri_activation` | direct, neurovault, ontology, pharma_neurosynth | Activation contrasts | MSE | Head A |
| `structural` | enigma | CT/SA/SubVol effect maps | MSE or 1−corr | Head B |
| `pet_receptor` | neuromaps, receptor, neuromaps_residual, receptor_residual | PET/annotation | MSE | Head C |
| `gene_expression` | abagen | Gene PCA loadings → inverse | MSE | Head D (gene head) |
| `roi_mask` | (excluded from regression) | — | — | Retrieval only |

### 3.2 Current vs Target Architecture

**Current:** Single MLP with type one-hot concat; gene head for abagen only.

**Target:**
- **Head A (fMRI):** NQ, NS, NeuroVault, ontology, pharma_neurosynth
- **Head B (structural):** ENIGMA
- **Head C (biology):** neuromaps, receptor, residuals
- **Head D (gene):** abagen (existing gene head; predicts PC loadings)

**ROI/mask (2508, 437, etc.):** Excluded from regression; available for retrieval/provenance only.

**Router + fusion:** See [ROUTER_FUSION_DESIGN.md](ROUTER_FUSION_DESIGN.md) for the layer that combines head outputs into one consensus map with explainable breakdown (routing policy, return schema, evidence trace).

---

## 4. Embedding Strategy

### 4.1 What Gets Embedded

For each term, the encoder input is the **synthesized label** (possibly augmented):

- **fMRI:** `label` as-is (or LLM-relabeled with collection context)
- **Pharmacology:** `"{drug} {dose} mg, {task}, {design}"` — never raw `regparam`
- **Structural:** `"Structural: {disorder} cortical thickness"` (existing)
- **Biology:** `"Gene: {symbol} ({name}), {signaling}"` (enriched abagen) or neuromaps label

### 4.2 Dose Collections: Metadata-First Pipeline

1. **Store** in manifest/cache:
   - `collection_title`, `collection_description`
   - Parsed: `drug`, `dose_mg`, `route`, `design`, `population`, `timing`

2. **Generate** structured fields + short natural label:
   - `drug = ibuprofen` (ground to ChEBI when possible)
   - `dose_mg = 200 | 600`
   - `design = placebo-controlled, randomized`
   - `task = emotion-related activation`
   - Synthesized: `"ibuprofen 600 mg, emotion task, placebo-controlled"`

3. **LLM relabel** only after supplying full context:
   - Input: `image_label` + `collection_title` + `collection_description` + parsed fields
   - Output: Natural label for embedding

---

## 5. Training Recipe

### 5.1 Data Preparation

1. **Build caches** (unchanged): NQ, NS, NeuroVault, neuromaps, ENIGMA, abagen.
2. **Enrich NeuroVault** with dose metadata (already in `build_neurovault_cache`: `_is_dose_related`, `_extract_dose_from_text`).
3. **Relabel** with `relabel_terms_with_llm` using collection metadata (already passes `cname`, `cdesc`, `map_kind`, `dose_related`).
4. **Merge** with `build_expanded_term_maps`; ensure `term_map_types.pkl` has correct routing.

### 5.2 Head Separation (Implementation Steps)

1. **Add `term_map_types` values:** Ensure `structural` and `pet_receptor` are distinct; consider splitting `pet_receptor` into `pet_receptor` (neuromaps/receptor) and `gene_expression` (abagen) if gene head is separate.
2. **Create separate MLP heads:** One per map type (or one shared + type-conditioned as now, but with **separate loss terms** per type).
3. **Loss choice:**
   - fMRI, biology: MSE
   - Structural: MSE or `1 - Pearson_r` (correlation loss often fits effect maps better)
4. **Routing at train time:** Batch by map type; each head only sees its type.
5. **Routing at inference:** `_infer_map_type(query)` → select head. Extend keyword rules for dose/compound queries.

### 5.3 Small-n Collections

- **Keep** in cache and in regression training. Do not exclude small-n collections.
- **Retrieval:** Always include; user gets "nearest evidence" from all collections.

---

## 6. External Knowledge (for Biohacker Credibility)

| Resource | Use |
|----------|-----|
| **Cognitive Atlas** | Task/cognitive term normalization |
| **PDSP Ki** | Receptor affinity fingerprints (compound → targets) |
| **NeuroVault** | Unthresholded statistical maps |
| **NeuroQuery / NeuroSynth** | Broad term→map grounding (sparse pharmacology) |
| **OpenNeuro** | Curated pharmacology/psychedelic datasets |
| **abagen** | Gene expression (separate head) |
| **ENIGMA** | Structural (separate head) |

---

## 7. Minimal V1 for Biohackers

1. **Compound page:** Mechanism fingerprint (targets from PDSP) + predicted circuit map + nearest evidence maps.
2. **Dose-aware query:** Parse "200 mg / placebo / acute" into structured conditioning; pass to encoder.
3. **Evidence-first UI:** Show provenance (study/collection links), uncertainty.
4. **Guardrails:** "Hypothesis generation, not medical advice."

---

## 8. Implementation Checklist

- [ ] Extend `term_map_types` to support 4-way routing (fmri, structural, pet_receptor, gene_expression) if gene head is separate.
- [ ] Add separate structural head in `train_text_to_brain_embedding.py`; route ENIGMA to it.
- [ ] Add separate biology head; route neuromaps + receptor to it.
- [ ] Implement correlation loss option for structural head.
- [ ] Extend `_infer_map_type` for compound/dose queries ("ibuprofen", "600 mg", etc.).
- [ ] Add retrieval-first output: top-k nearest terms + sources to inference API.
- [ ] Store structured dose fields in NeuroVault cache/manifest for pharmacology collections.
