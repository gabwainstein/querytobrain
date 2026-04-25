# NeuroLab Platform: Master Architecture Document

## 1. DATASETS

### 1.1 Core Neuroimaging Training Data (Term → Brain Map Pairs)

| Dataset | Content | Format | License | Role |
|---------|---------|--------|---------|------|
| **NeuroSynth** | ~14,000 terms → coordinate-based meta-analytic maps | MNI volumetric | CC0 | Primary training: largest source of term-to-brain-map pairs |
| **NeuroQuery** | ~7,000 terms → encoding model maps (smoother than NeuroSynth) | MNI volumetric | CC0 | Primary training: higher quality per-map, better generalization |
| **NeuroVault** | ~100,000+ statistical maps from published studies | MNI volumetric | CC0 (most) | Training augmentation: real activation/contrast maps with metadata |
| **Neurosynth-Compose / NiMARE** | Custom meta-analyses from NeuroSynth coordinates | MNI volumetric | CC0 (derived) | Generate ~50+ pharmacological meta-analytic maps (drug terms) |

### 1.2 Disease & Disorder Maps

| Dataset | Content | Format | License | Role |
|---------|---------|--------|---------|------|
| **ENIGMA** | 100+ disorder-specific cortical thickness & subcortical volume effect-size maps | Summary statistics per parcel | BSD-3-Clause | Training: disease→brain associations (schizophrenia, MDD, bipolar, ADHD, epilepsy, etc.) |
| **NeuroVault clinical collections** | Disorder-specific contrast maps deposited by researchers | MNI volumetric | CC0 (most) | Training augmentation for clinical queries |

### 1.3 Pharmacological Neuroimaging Data

| Dataset | Content | Format | License | Role |
|---------|---------|--------|---------|------|
| **OpenNeuro ds006072** | Psilocybin precision functional mapping (7 subjects, crossover with methylphenidate) | Raw multi-echo fMRI, structural, diffusion | CC0 | Process to drug-vs-placebo contrast maps for training |
| **OpenNeuro ds003059** | LSD and psilocybin resting-state fMRI (Carhart-Harris) | Raw fMRI | CC0 | Process to drug-vs-placebo maps |
| **Dryad MDMA dataset** | Simultaneous rat PET/fMRI, baseline (n=30) + MDMA challenge (n=11) | NIfTI (anat, func, PET) | CC0 | Translational: MDMA brain engagement |
| **NeuroVault pharma collections** | Various drug-challenge contrast maps already processed | MNI volumetric | CC0 (most) | Direct training data for drug queries |
| **NiMARE batch meta-analyses** | ~50 drug terms meta-analyzed from NeuroSynth coordinates | MNI volumetric | CC0 (derived) | Pharmacological training maps for: psilocybin, LSD, DMT, ketamine, amphetamine, methylphenidate, cocaine, caffeine, nicotine, modafinil, alcohol, benzodiazepine, SSRI, fluoxetine, haloperidol, risperidone, clozapine, morphine, THC, MDMA, propofol, levodopa, etc. |

### 1.4 Drug Binding & Pharmacology Databases

| Dataset | Content | Size | License | Role |
|---------|---------|------|---------|------|
| **PDSP Ki Database** | Ki binding affinities (drug × receptor) | ~98K Ki values, ~7,500 compounds, ~738 targets | Public domain | Core: drug→receptor affinity matrix for spatial projection through gene expression |
| **ChEMBL** | IC50, Ki, EC50 activity values | 5.4M activity values, 8,200 targets | CC-BY-SA 3.0 | Extended drug-target coverage; supplements PDSP |
| **BindingDB** | Measured binding affinities (Ki, Kd, IC50) | 2.9M data points, 1.3M compounds, 9.3K targets | Public domain | Fallback for compounds missing from PDSP |
| **IUPHAR Guide to Pharmacology** | Expert-curated ligand-target relationships | Smaller but highest quality | CC-BY-SA 4.0 | Clean drug class → receptor subtype mapping |
| **PubChem BioAssay** | Massive assay data | Very large, noisy | Public domain | Last-resort fallback |
| **DrugBank** | Drug mechanisms, pathways, interactions | Comprehensive | CC-BY-NC | Free tier enrichment only (blocked for commercial) |

**Killer combination:** PDSP (public domain, CNS Ki values) + ChEMBL (CC-BY-SA, broad coverage) = comprehensive drug-to-receptor affinity matrix. Query drug → lookup binding profile → weight receptor gene expression maps by 1/Ki → predicted brain engagement map.

### 1.5 Gene Expression

| Dataset | Content | Format | License | Role |
|---------|---------|--------|---------|------|
| **Allen Human Brain Atlas (via abagen)** | ~15,000 genes × cortical parcels (6 donors) | Parcellated expression matrix | CC-BY-NC 3.0 | PCA decomposition → biological basis set for MLP output; drug-to-gene-to-brain projection pathway |
| **neuromaps genePC1** | Pre-computed first principal component of Allen gene expression | Surface map (fsaverage) | CC-BY-NC 3.0 | Validation of our PCA against published gradient |

### 1.6 Receptor & Neurotransmitter Atlases

| Dataset | Content | Format | License | Role |
|---------|---------|--------|---------|------|
| **Hansen et al. 2022 receptor atlas** | 19 receptor/transporter maps from PET studies | Parcellated (Schaefer; reparcellate to Glasser) | CC-BY-NC-SA 4.0 | Free-tier enrichment layer (not for commercial prediction core) |
| **neuromaps PET collection** | Individual PET maps (5-HT1a, 5-HT2a, D1, D2, DAT, SERT, etc.) | Surface/volume | Mixed (some CC0) | Some maps individually CC0 on NeuroVault — usable for commercial |
| **JuSpace PET maps** | PET-derived neurotransmitter maps for spatial correlation | Volume | Mixed per map | Check individual licenses; some usable |

### 1.7 Structural & White Matter Atlases

| Dataset | Content | Format | License | Role |
|---------|---------|--------|---------|------|
| **HCP S1200 cortical thickness** | Population-average cortical thickness map | Surface (fsLR 32k) | Open access (registration) | Structural context for enrichment reports |
| **HCP S1200 myelin (T1w/T2w)** | Intracortical myelin map | Surface (fsLR 32k) | Open access (registration) | Structural context; correlate with gene PCA |
| **JHU ICBM-DTI-81** | 48 white matter tract labels (81 subjects) | MNI volumetric | CC0 (NeuroVault) | White matter connectivity overlay for affected regions |
| **JHU tractography atlas** | 20 probabilistic tract structures (28 subjects) | MNI volumetric | CC0 (NeuroVault) | Same |
| **HCP DTI templates (FSL)** | FA, MD maps from 1,065 subjects | MNI volumetric | Open access | Highest quality DTI reference maps |
| **Fun With Tracts (FWT)** | 68 bundle atlas with automated dissection | MNI volumetric | Check GitHub | Most comprehensive open WM atlas |
| **BigBrain laminar thickness** | Cortical layer thickness gradients | Surface | CC-BY-NC-SA | Free tier only |

### 1.8 Additional neuromaps Annotations (~80+ maps)

Used for multimodal enrichment at inference time:

- **PET receptor maps**: 5-HT1a, 5-HT1b, 5-HT2a, 5-HT4, 5-HT6, D1, D2, DAT, SERT, NET, GABAa, NMDA, mGluR5, M1, VAChT, α4β2, MU, CB1, H3
- **MEG power maps**: delta, theta, alpha, beta, gamma band power
- **Metabolic maps**: CMRGlu, CMRO2, CBF, CBV
- **Functional gradients**: principal gradient (Margulies), functional connectivity gradients
- **Developmental maps**: evolutionary expansion, allometric scaling, cortical growth rates
- **Laminar/cytoarchitectonic**: BigBrain layer thicknesses, Von Economo types

---

## 2. ONTOLOGIES

### 2.1 Cognitive & Behavioral (Term Vocabulary)

| Ontology | Terms | Content | Role |
|----------|-------|---------|------|
| **Cognitive Atlas** | ~800 concepts + tasks | Cognitive constructs, conditions, tasks, contrasts | Core vocabulary for cognitive queries; maps concepts to NeuroSynth terms |
| **CogPO (Cognitive Paradigm Ontology)** | ~300 terms | Experimental paradigms, behavioral conditions | Links tasks/paradigms to cognitive constructs |
| **NBO (Neurobehavior Ontology)** | ~1,500 terms | Behavioral phenotypes, processes | Bridges behavioral descriptions to neural substrates |
| **MF (Mental Functioning Ontology)** | ~500 terms | Mental processes, cognitive functions | Complements CogAt with broader mental functioning |
| **MFOEM (MF Emotion Module)** | ~100 terms | Emotion categories and their cognitive/physiological relations | Emotional granularity (fear, disgust, sadness, anger, surprise) |
| **RDoC** | ~100 terms (manual) | 5 domains × 23 constructs × subconstructs | Bridges clinical constructs to neuroscience mechanisms; hand-curated edges to CogAt and MONDO |

### 2.2 Anatomical

| Ontology | Terms | Content | Role |
|----------|-------|---------|------|
| **UBERON** | ~15,000 terms | Cross-species anatomy | Maps anatomical terms to atlas parcels; formal brain region hierarchy |
| **NIFSTD (NIF Standard Ontology)** | ~30,000 terms | Neuroscience-specific vocabulary (brain regions, cell types, techniques) | Comprehensive neuroscience vocabulary |

### 2.3 Clinical & Disease

| Ontology | Terms | Content | Role |
|----------|-------|---------|------|
| **MONDO** | ~30,000 terms | Merged disease ontology (DOID + Orphanet + OMIM) | Disease hierarchy; cross-refs to HPO; connects to ENIGMA disease maps |
| **HPO (Human Phenotype Ontology)** | ~18,000 terms | Symptoms, phenotypes (hallucinations, anhedonia, tremor, seizure) | Bridges diseases to cognitive/behavioral terms; symptom-level queries |
| **DOID (Disease Ontology)** | ~12,000 terms | Disease classification hierarchy | Alternative to MONDO (use one, not both — MONDO preferred for broader coverage) |

### 2.4 Molecular & Pharmacological

| Ontology | Terms | Content | Role |
|----------|-------|---------|------|
| **ChEBI-lite (role subontology)** | ~5,000 terms | Drug class hierarchies (SSRI → antidepressant → psychotropic drug) | Pharmacological class structure; links to PDSP drug names |
| **GO (Gene Ontology)** | ~45,000 terms | Biological processes, molecular functions, cellular components | Labels gene expression PCA components; connects genes to biological functions |
| **IUPHAR nomenclature** | ~3,000 terms | Receptor/transporter/channel classification | Formal receptor naming and hierarchy |

### 2.5 Foundational (Structural Glue)

| Ontology | Terms | Content | Role |
|----------|-------|---------|------|
| **BFO (Basic Formal Ontology)** | ~35 terms | Upper-level ontology categories | Structural consistency across ontologies |
| **RO (Relations Ontology)** | ~400 terms | Standardized relationship types (part_of, has_function, etc.) | Consistent edge types in knowledge graph |
| **PATO** | ~2,600 terms | Phenotypic qualities (increased thickness, decreased volume, atrophied) | Structured descriptions of structural abnormalities |

### 2.6 Ontology Interconnection (Meta-Graph)

Ontologies are not merged but connected via bridge edges. **Critically, the ontology graph is used at training time for data augmentation and enrichment report generation — NOT at inference time for map retrieval.** At inference, the entire prediction pipeline is pure tensor operations (see Section 5).

**Tier 1 — Direct Bridges (highest impact):**
- Cognitive Atlas ↔ MONDO (cognitive constructs ↔ diseases)
- PDSP/ChEBI ↔ receptor genes ↔ gene expression PCA (drug mechanisms ↔ brain topology)
- MONDO ↔ HPO (diseases ↔ phenotypes/symptoms)

**Tier 2 — Enrichment Bridges:**
- Cognitive Atlas ↔ NeuroSynth terms (formal constructs ↔ meta-analytic vocabulary)
- ChEBI hierarchy ↔ PDSP drug names (drug classes ↔ binding profiles)
- GO ↔ gene expression PC loadings (biological processes ↔ spatial gradients)

**Tier 3 — Structural Bridges:**
- UBERON ↔ atlas parcels (anatomy terms ↔ Glasser/Tian regions)
- DisGeNET ↔ gene expression (disease-gene associations ↔ brain expression patterns)

**Bridge edge sources:**
1. Lexical matching (shared names/synonyms across ontologies)
2. Embedding similarity (cosine > 0.85 between cross-ontology term embeddings)
3. Curated cross-references (MONDO→HPO xrefs, ChEBI→DrugBank, etc.)

**Training-time uses of the meta-graph:**
1. **Data multiplication:** For each brain map, generate multiple training pairs from synonyms, related terms, parent/child concepts (see Section 5.2)
2. **Text augmentation:** Enrich training term text with KG triples before embedding, producing augmented embeddings that carry relational context
3. **Enrichment reports:** At inference, the graph is optionally traversed to populate the human-readable enrichment report (related diseases, phenotypes, drugs) — but this is a display feature, not part of the prediction pathway

---

## 3. EMBEDDING STRATEGY

### 3.1 Model Choice

**OpenAI text-embedding-3-large** at 1536 dimensions (Matryoshka truncation from native 3072).

Rationale: Best general-purpose embedding quality (MTEB 64.6), captures biomedical vocabulary well, Matryoshka property allows dimension reduction without retraining. 1536-dim balances quality with storage (vs full 3072).

### 3.2 Term Embedding Protocol

For each ontology term, embed **enriched text** (not just the label):

```
{label} | {synonym_1}, {synonym_2}, ... | {definition} | parent: {parent_label} | {key_relations}
```

Example:
```
working memory | short-term memory, WM | The cognitive system for temporarily 
holding and manipulating information | parent: executive function | measured_by: 
n-back task, Sternberg task | related_to: attention, cognitive control
```

### 3.3 Embedding Pipeline

```
Build time (once):
  1. Embed all ~60K ontology terms → label_embeddings.npz (~350MB)
  2. For each training term:
     a. Find top-k ontology neighbors via cosine sim against label_embeddings
     b. Format augmented text (term + KG triples from neighbors)
     c. Embed augmented text
     d. Embed all synonyms and close ontological relations
     e. Save everything → training_embeddings.npz (all variants per term)
  3. Train Generalizer on all embedding variants (zero API calls during training)
  4. Train Memorizer on all embedding variants (zero API calls during training)

Inference (per query):
  1. Embed user query (1 API call, ~$0.0001)
  2. Cosine sim against training_embeddings.npz (1 matmul, <1ms on GPU)
     → routing α + top-k explanation (free byproduct)
  3. Generalizer forward pass → predicted map + PC coefficients + gate
  4. Memorizer forward pass → memorized/interpolated map
  5. Blend: (1 - α) × generalizer + α × memorizer
  6. Done. No graph traversal, no retrieval, no dictionary lookups.
```

**Note:** The ontology graph is NOT used at inference for prediction. It is optionally
traversed for populating the enrichment report (display layer only). All ontological
knowledge enters the prediction pathway through training-time data multiplication
and embedding augmentation.

**Cost:** ~$0.26 to embed entire ontology. ~$0.0001 per inference query (single embed call). Batched at 2048 terms/request, full embedding takes ~5-10 minutes. All local compute at inference: <5ms.

---

## 4. GENE EXPRESSION PCA

### 4.1 Overview

Decompose Allen Human Brain Atlas gene expression (via abagen) into principal components that serve as:
- Biologically-labeled axes of cortical organization
- Pharmacological fingerprinting space for any drug
- Dimensionality-reduced output basis for the MLP
- Structural context for enrichment reports

### 4.2 Two Parallel PCAs

**Full-genome PCA:** ~15,000 genes × 392 parcels → **15 PCs** (~60% variance explained)
- Captures complete cortical organization (cell types, developmental gradients, metabolism)
- PC1: sensorimotor-to-transmodal hierarchy (~25% variance)
- PC2: anterior-posterior axis (~8% variance)
- PC3+: increasingly specific biological gradients
- Used as MLP output basis

**Receptor-only PCA:** ~300 receptor/transporter/channel genes × 392 parcels → **10 PCs**
- Captures purely neurotransmitter-related spatial gradients
- PC1_receptor: e.g., "dopaminergic vs serotonergic cortex"
- Used for pharmacological specificity analysis

### 4.3 Biological Labeling

Each PC is labeled via:
1. **Gene ontology enrichment** (gseapy/Enrichr) — top 300 genes per pole → GO terms
2. **Cell-type deconvolution** — marker gene loadings per PC (excitatory neurons, oligodendrocytes, astrocytes, microglia, etc.)
3. **Neurotransmitter receptor loadings** — which receptors drive each PC
4. **Correlation with known gradients** — myelin, thickness, evolutionary expansion, functional connectivity gradients

Result: a `pc_registry.json` with human-readable labels and biological context for each PC.

### 4.4 Drug-to-PC-Space Projection

```
Drug query → PDSP Ki lookup → receptor gene weights (1/Ki) →
  → project through gene loadings matrix → 15-dim PC coordinates →
  → reconstruct spatial brain map via PC scores
```

This gives every drug in PDSP (~7,500 compounds) a 15-dimensional pharmacological fingerprint grounded in brain transcriptomic architecture. Drug similarity = cosine similarity in PC space.

---

## 5. PIPELINE ARCHITECTURE

### 5.1 Atlas Choice

**Primary atlas:** Glasser 360 (cortex) + Tian S2 (~32 subcortical) = **~392 parcels**

**Glasser 360 (cortex):**
- Multimodally defined (T1w/T2w myelin, cortical thickness, fMRI, rfMRI)
- Anatomically meaningful labels throughout ("area 55b," "V1," "TE1a," "FEF")
- Cytoarchitectonic designations that appear extensively in neuroanatomy literature
- Respects true areal boundaries (vs arbitrary geometric subdivisions)
- DK-style interpretability with Schaefer-level resolution — no need for a separate reporting atlas

**Tian S2 (~32 subcortical parcels):**
- Distinguishes anterior/posterior hippocampus, caudate head/body/tail, thalamic nuclei
- Functional subdivisions align with distinct cognitive ontologies (sensory vs executive thalamus)
- Gold standard subcortical parcellation, pairs natively with Glasser cortical parcels

**Reporting:** Glasser labels are already anatomically meaningful, so no DK lookup table
needed. Peak regions reported directly as "V1," "FEF," "area 44" etc.

### 5.2 Training Data Assembly

**Step 1: Parcellate all brain maps to Glasser 360 + Tian S2 (~392 parcels)**

Sources and expected yields:
- NeuroSynth terms: ~14,000 maps
- NeuroQuery terms: ~7,000 maps
- NeuroVault collections: variable (hundreds to thousands of usable maps)
- ENIGMA summary statistics: ~100+ disorder maps
- NiMARE pharmacological meta-analyses: ~50 maps
- OpenNeuro drug challenges (processed): ~10-20 contrast maps
- PDSP × gene expression projections: ~7,500 drug spatial maps (synthetic)

Base training pairs: **~20,000-30,000 (text, brain_map) pairs**

**Step 2: Ontology-based data multiplication**

For each (term, brain_map) pair, the ontology graph generates additional valid training pairs.
This happens once at data preparation time. The graph is NOT used at inference.

```python
training_samples = []

for term, brain_map in base_training_maps.items():
    # Version A: raw embedding
    raw_emb = embed(term)                           # cached
    training_samples.append((raw_emb, brain_map))
    
    # Version B: KG-augmented embedding
    kg_context = get_kg_neighbors(term, ontology_graph, top_k=5)
    augmented_text = format_augmented(term, kg_context)
    aug_emb = embed(augmented_text)                 # cached
    training_samples.append((aug_emb, brain_map))
    
    # Version C: synonym embeddings (from ontology)
    for synonym in get_synonyms(term, ontology_graph):
        syn_emb = embed(synonym)                    # cached
        training_samples.append((syn_emb, brain_map))
    
    # Version D: closely related terms (parent/child, 1-hop)
    for related, relation in get_close_relations(term, ontology_graph):
        rel_emb = embed(related)                    # cached
        training_samples.append((rel_emb, brain_map))

# Effective training set: ~2-3x the base count = ~50,000-80,000 pairs
# ALL embeddings pre-computed and cached as numpy arrays
# Zero API calls during actual training
```

**Step 3: Embed all training terms**

For each training term:
1. Compute raw OpenAI embedding (1536-dim)
2. Find top-k ontology neighbors from label_embeddings.npz via cosine sim
3. Construct augmented text: `{term} | {KG context: neighbor labels + relation triples}`
4. Compute augmented embedding (1536-dim)
5. Extract all synonyms and close relations from ontology graph
6. Embed each synonym/relation
7. Cache everything — raw, augmented, synonyms — as `training_embeddings.npz`

**Step 4: Compute gene expression PCA outputs**

For each training brain map:
1. Project onto gene expression PC scores: `pc_coefficients = pc_scores.T @ brain_map` (15-dim)
2. Store as alternative training target (model can predict PC coefficients instead of raw 392-dim maps)

### 5.3 Model Architecture: Dual-Network (Generalizer + Memorizer)

The core insight: instead of retrieving cached maps and blending them post-hoc (which
forces linear combinations of maps that may interact nonlinearly), we use two networks
with complementary objectives, blended via a similarity-based routing signal.

```
                     ┌──────────────────────────────────────┐
                     │           USER QUERY                  │
                     │  "cognitive effects of sertraline"    │
                     └─────────────────┬────────────────────┘
                                       │
                                       ▼
                           ┌───────────────────┐
                           │   OpenAI Embed    │
                           │   (1536-dim)      │
                           └─────────┬─────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
         │  GENERALIZER │  │  MEMORIZER   │  │  COSINE SIM      │
         │  (Network 1) │  │  (Network 2) │  │  vs training     │
         │              │  │              │  │  embeddings       │
         │  Regularized │  │  Overfit     │  │  (20K×1536)      │
         │  Dropout,    │  │  No dropout, │  │  @ (1536,)       │
         │  weight decay│  │  no reg,     │  │  = (20K,)        │
         │  PC-constrain│  │  near-zero   │  │                  │
         │              │  │  train loss  │  │  → best_sim      │
         │  Learns:     │  │              │  │  → top_k terms   │
         │  • nonlinear │  │  Learns:     │  │    (explanation) │
         │    interact. │  │  • exact map │  │                  │
         │  • composit. │  │    reprod.   │  └────────┬─────────┘
         │  • novel     │  │  • smooth    │           │
         │    queries   │  │    interp.   │           │
         └──────┬───────┘  └──────┬───────┘           │
                │                 │                    │
                ▼                 ▼                    ▼
         ┌────────────────────────────────────────────────────┐
         │              SIMILARITY-BASED ROUTING              │
         │                                                    │
         │  α = sigmoid(steepness * (best_sim - midpoint))   │
         │                                                    │
         │  final_map = (1 - α) × gen_map + α × mem_map     │
         │                                                    │
         │  • best_sim ≈ 0.95 → α ≈ 0.98 (trust memorizer) │
         │  • best_sim ≈ 0.75 → α ≈ 0.50 (equal blend)     │
         │  • best_sim ≈ 0.50 → α ≈ 0.01 (trust generaliz.)│
         └────────────────────────┬───────────────────────────┘
                                  │
             ┌────────────────────┼─────────────────────┐
             ▼                    ▼                     ▼
  ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  PDSP Pathway  │   │  FINAL BRAIN MAP │   │  EXPLANATION     │
  │  (drug queries)│   │    (392-dim)     │   │  (from cosine    │
  │                │   │                  │   │   sim search)    │
  │  Drug → Ki →   │   │  → 3D visual.   │   │                  │
  │  gene expr →   │   │  → PC radar     │   │  "Closest terms: │
  │  PC-space map  │   │  → enrichment   │   │   SSRI (0.89),   │
  │  (additional   │   │    report       │   │   serotonin      │
  │   context)     │   │                  │   │   reuptake (0.84)│
  └────────────────┘   └──────────────────┘   │   depression     │
                                              │   (0.71)"        │
                                              └──────────────────┘
```

**Why two networks instead of cache retrieval (V1 → V2 decision):**

1. **Nonlinear interactions preserved.** V1 blended cached maps via weighted average —
   a linear operation that assumes "anhedonia in Parkinson's" ≈ 0.4×Parkinson's + 0.3×anhedonia.
   This is wrong: the intersection is a specific dopaminergic subnetwork, not an average.
   Both V2 networks are nonlinear function approximators, so even the memorizer's interpolation
   between nearby training embeddings goes through nonlinear transformations.

2. **No graph traversal at inference.** V1 required KG expansion, dictionary lookups,
   conditional logic, and weighted averaging — branching code that's hard to optimize and
   produces variable latency. V2 inference is: 1 embedding call, 1 matrix-vector multiply,
   2 forward passes, 1 scalar blend. Pure tensor ops, fully GPU-able, deterministic latency.

3. **Explainability preserved for free.** The cosine similarity computation needed for
   routing α also identifies the closest training terms — the same information V1's
   graph traversal provided, but as a byproduct of an operation you're already doing.

4. **Ontologies still contribute** — at training time, through data multiplication and
   embedding augmentation. Their relational knowledge is baked into the training distribution.
   The graph does its work during data prep, then you don't need it at inference.

### 5.4 Network Details

**Network 1: Generalizer (PC-constrained dual-head MLP)**

```python
class Generalizer(nn.Module):
    """
    Input:  text embedding (1536-dim)
    Output: brain activation map (392-dim) + PC coefficients (15-dim) + gate value
    
    PC Head: predicts 15 gene expression PC coefficients, 
             reconstructs map via fixed PC score basis (392 × 15)
    Residual Head: predicts direct 392-dim map correction
    Gate: learned α ∈ [0,1] weighting PC vs residual contribution
    
    Regularized: dropout (0.3), weight decay (1e-4), LayerNorm
    PC scores registered as non-trainable buffer (fixed biological basis)
    
    Training schedule:
      Phase A (epochs 1-50):   λ_residual=1.0 (force PC pathway only)
      Phase B (epochs 50-100): λ_residual=0.1 (allow residual gradually)
      Phase C (epochs 100+):   λ_residual=0.01 (fine-tune freely)
    
    Loss = MSE(predicted, target) + λ_residual * (1-gate).mean() + λ_smooth * ||pc_coefs||²
    """
    # Shared trunk: Linear(1536→512) → LN → GELU → Dropout → Linear(512→512) → LN → GELU → Dropout
    # PC head:      Linear(512→256) → GELU → Linear(256→15) → matmul with PC scores (392×15)
    # Residual head: Linear(512→392)
    # Gate:         Linear(512→1) → Sigmoid
```

**Network 2: Memorizer (overfit MLP)**

```python
class Memorizer(nn.Module):
    """
    Input:  text embedding (1536-dim)
    Output: brain activation map (392-dim)
    
    Deliberately overfitted to training data.
    NO dropout, NO weight decay, NO regularization.
    Trained for 500+ epochs until training loss ≈ 0.
    
    Purpose: perfect reproduction of known maps at training embeddings,
    smooth nonlinear interpolation between them for nearby queries.
    
    Small architecture is sufficient — it just needs to memorize, not generalize.
    ~8-10MB of weights replaces a ~32MB numpy map cache.
    """
    # Linear(1536→1024) → GELU → Linear(1024→1024) → GELU → Linear(1024→392)
```

**Inference (complete pipeline):**

```python
def inference(query: str):
    # Step 1: embed query (1 OpenAI API call)
    emb = embed(query)                              # (1536,)
    
    # Step 2: cosine similarity routing + explanation
    # training_embeddings: (N, 1536), pre-normalized, stays in GPU memory
    sims = training_embeddings @ emb                # (N,) — one matmul
    top_k = sims.topk(5)
    best_sim = top_k.values[0]
    
    # Soft routing: sigmoid with tunable steepness and midpoint
    alpha = sigmoid(20.0 * (best_sim - 0.75))
    # sim=0.95 → α=0.98 | sim=0.75 → α=0.50 | sim=0.55 → α=0.02
    
    # Step 3: dual network forward passes
    gen_map, pc_coefs, gate = generalizer(emb)      # (392,), (15,), scalar
    mem_map = memorizer(emb)                        # (392,)
    
    # Step 4: blend
    final_map = (1 - alpha) * gen_map + alpha * mem_map   # (392,)
    
    # Step 5: explanation (free from step 2)
    explanation = [(term_names[i], float(sims[i])) for i in top_k.indices]
    # "Closest terms: SSRI (0.89), serotonin reuptake (0.84), depression (0.71)..."
    
    # Step 6: PDSP pathway (drug queries only — independent, additive context)
    pdsp_map = None
    if is_drug_query(query):
        pdsp_map = pdsp_project(query)              # (392,) from Ki → gene expr → PC space
    
    return final_map, pc_coefs, gate, alpha, explanation, pdsp_map
```

**Total inference cost:**
- 1 OpenAI embedding API call (~$0.0001, ~100ms)
- 1 matrix-vector multiply for cosine similarity (~1ms on GPU)
- 2 small MLP forward passes (~0.5ms each on GPU)
- 1 scalar multiply and add (~negligible)
- **Total: ~100ms, dominated by the API call. <5ms for all local compute.**

### 5.5 Routing Hyperparameters

The sigmoid routing function `α = σ(steepness × (best_sim - midpoint))` has two
hyperparameters that control the generalizer/memorizer balance:

- **midpoint** (default: 0.75): cosine similarity at which the blend is 50/50.
  Lower = trust memorizer more often. Higher = trust generalizer more often.
  
- **steepness** (default: 20.0): how sharply the transition happens.
  Higher = more binary (memorizer OR generalizer). Lower = more gradual blending.

These are tuned on the held-out set by sweeping a grid and maximizing prediction
correlation. Expected to take ~5 minutes given that both networks are already trained
and inference is fast.

**Diagnostic:** Plot α distribution across the held-out set. If α is almost always
near 1.0, the memorizer is doing all the work (generalizer isn't needed, or the held-out
terms are too similar to training terms). If α is almost always near 0.0, the held-out
terms are all far from training data (memorizer isn't helping, or the threshold is too high).
Healthy distribution should show a spread, with a mode around 0.7-0.9 for well-covered
domains and a tail near 0.0-0.3 for novel queries.

### 5.6 PDSP Pharmacological Pathway (Drug Queries)

For queries identified as drug-related, an independent pathway provides additional context:

```
Drug name → PDSP Ki lookup → binding profile {receptor: 1/Ki} →
  → weight receptor gene expression vectors → spatial engagement map (392-dim) →
  → project into PC space → 15-dim pharmacological fingerprint
```

This is NOT blended into the final map prediction (both networks already handle drug
queries through the text embedding). Instead, it provides:
- Independent validation (does MLP prediction correlate with pharmacology-derived map?)
- Receptor engagement profile for the enrichment report
- Drug similarity in PC space (which other drugs have similar brain engagement?)
- UMAP visualization of pharmacological landscape

### 5.7 Enrichment Report Generation

The enrichment report is populated from multiple sources. The ontology graph CAN be
traversed here for display purposes (finding related diseases, phenotypes, etc.), but
this is a presentation layer — it does not affect the brain map prediction.

Sources for the report:
1. **Predicted map peaks** → DK/Glasser anatomical labels (via lookup table)
2. **PC coefficients** → pc_registry.json biological labels per dominant PC
3. **Cosine similarity top-k** → closest training terms (from routing computation)
4. **PDSP profile** → receptor engagement, similar drugs (drug queries only)
5. **neuromaps correlations** → spatial correlation of predicted map with receptor/metabolic/structural annotations
6. **Ontology context** → related diseases, phenotypes, cognitive constructs (optional graph traversal, display only)

---

## 6. INFERENCE OUTPUT STRUCTURE

For a query like "What brain regions are affected by sertraline?":

### 6.1 Predicted Brain Map
- 392-dim parcellated activation vector (Glasser 360 + Tian S2)
- Rendered as 3D brain visualization with color-coded activation
- Peak regions reported with DK/Glasser anatomical labels
- **Routing α** value: how much the prediction relies on memorized training data vs novel generalization

### 6.2 Nearest Training Terms (from routing computation)
- Top-5 closest training terms by cosine similarity (free byproduct of the routing α computation)
- e.g., "Closest terms: SSRI (0.89), serotonin reuptake inhibitor (0.84), fluoxetine (0.78), depression treatment (0.71), anxiolytic (0.67)"
- Provides an interpretable audit trail without requiring graph traversal

### 6.2 PC Decomposition (Radar Chart)
- 15 PC coefficients with biological labels on each axis
- e.g., "High on Sensorimotor-Transmodal (PC1), moderate Serotonergic gradient (PC_receptor_1)"
- Gate value indicating how much is explained by gene expression topology

### 6.3 Receptor Engagement Profile
- From PDSP Ki data: sertraline's binding profile (SERT >> NET > σ1 > D1 > α1)
- Each receptor's spatial contribution to the predicted map
- Comparison to other drugs in same class (other SSRIs)

### 6.4 Pharmacological Similarity
- Top 5-10 drugs with most similar brain engagement profiles in PC space
- UMAP visualization of drug similarity landscape
- Drug cluster membership (SSRI cluster, etc.)

### 6.5 Ontology-Derived Context (display layer only — does not affect prediction)
- **Related diseases:** Major depressive disorder, GAD, OCD, PTSD, panic disorder (from MONDO + drug indications)
- **Related phenotypes:** Anhedonia, anxiety, insomnia, decreased appetite (from HPO via MONDO bridge)
- **Related cognitive constructs:** Reward processing, emotional regulation, cognitive control (from CogAt via HPO bridge)
- **Receptor systems:** Serotonin (5-HT) primary, norepinephrine secondary (from PDSP + ChEBI hierarchy)

### 6.6 Structural Context
- neuromaps overlays: myelin, cortical thickness, metabolic maps for affected regions
- JHU/FWT white matter tracts connecting peak activation regions
- ENIGMA depression map correlation (how much does this drug's pattern align with the disease's structural signature?)

---

## 7. TESTING & VALIDATION

### 7.1 Held-Out Evaluation

- Reserve 20% of training maps (stratified by source and category)
- Metrics: Pearson correlation (predicted vs actual map), RMSE, spatial overlap of top-k% activated parcels
- Report per-category: cognitive terms, disease terms, drug terms, anatomy terms

### 7.2 Cross-Validation by Data Source

- Train on NeuroSynth → test on NeuroQuery (and vice versa)
- Train on meta-analytic maps → test on NeuroVault real activation maps
- Tests generalization across data generation methods

### 7.3 Biological Plausibility Checks

- Predicted maps should correlate with known neuromaps annotations (e.g., "dopamine" query should correlate with D1/D2 receptor maps)
- Gene PCA gate value: should be high for broad neurobiological queries, lower for very specific task queries
- Drug predictions: PDSP-derived maps vs NiMARE meta-analytic maps for same drug should agree

### 7.4 Ablation Studies

- Full model (generalizer + memorizer) vs generalizer only vs memorizer only
- Generalizer: PC-constrained vs unconstrained (no PC head, just residual)
- With ontology data multiplication vs without (raw terms only)
- With KG-augmented embeddings vs raw embeddings only
- Routing: sigmoid routing vs fixed alpha vs learned MLP-based routing
- Routing hyperparameters: sweep steepness (5-50) × midpoint (0.6-0.9)
- Memorizer depth: 2-layer vs 3-layer vs 4-layer
- Embedding: text-embedding-3-large vs text-embedding-3-small
- Parcellation: Glasser 360 + Tian S2 vs Schaefer 400 vs Schaefer 200

### 7.5 Comparison to Baseline

- **NeuroQuery encoder** (existing model): direct text→brain mapping
- **Luppi-style zero-shot**: LLM descriptions of brain regions → embedding similarity
- **Nearest-neighbor retrieval**: embed query, find closest training term, return its map
- **Memorizer-only**: overfit network without generalizer (tests whether interpolation alone suffices)
- **Generalizer-only**: PC-constrained MLP without memorizer (tests whether generalization alone suffices)
- Our full model (generalizer + memorizer + routing) should beat all, especially for cross-domain and pharmacological queries

---

## 8. LICENSING SUMMARY

### Commercially Safe (prediction core + paid features)

| Resource | License |
|----------|---------|
| NeuroSynth / NeuroQuery | CC0 |
| NeuroVault (most collections) | CC0 |
| NiMARE meta-analytic maps (derived) | CC0 |
| ENIGMA summary statistics | BSD-3-Clause |
| OpenNeuro datasets | CC0 |
| PDSP Ki Database | Public domain |
| ChEMBL | CC-BY-SA 3.0 |
| BindingDB | Public domain |
| IUPHAR | CC-BY-SA 4.0 |
| PubChem BioAssay | Public domain |
| JHU white matter atlases | CC0 (NeuroVault) |
| HCP structural data | Open access (registration + citation) |
| All ontologies (OBO Foundry) | CC-BY 4.0 or CC0 |
| OpenAI embeddings | Standard API terms |

### Non-Commercial Only (free tier enrichment)

| Resource | License | Workaround |
|----------|---------|------------|
| Hansen receptor atlas | CC-BY-NC-SA 4.0 | Reconstruct from individual CC0 PET maps on NeuroVault; or negotiate license |
| Allen gene expression (abagen) | CC-BY-NC 3.0 | PCA components are heavily transformed (15 numbers from 15K genes); model-as-derivative argument |
| BigBrain laminar data | CC-BY-NC-SA | Free tier only |
| DrugBank | CC-BY-NC | Free tier only |

### Architecture: Two-Tier Licensing

- **Layer 1 (Commercial Core):** Prediction engine trained exclusively on CC0/BSD/CC-BY data
- **Layer 2 (Free Enrichment):** NC-restricted data displayed as context/enrichment, always free
- **Layer 3 (Paid Services):** API rate limits, batch processing, priority compute, consulting

---

## 9. IMPLEMENTATION PRIORITY

| Priority | Task | Dependencies | Est. Time |
|----------|------|--------------|-----------|
| **P0** | Parcellate NeuroSynth + NeuroQuery to Glasser 360 + Tian S2 | nilearn, nibabel | 2-3 hours |
| **P0** | Download & embed all ontologies | OpenAI API | 1 day |
| **P0** | Build knowledge graph with bridge edges | networkx | 1 day |
| **P0** | Ontology data multiplication: generate all training pairs (raw + augmented + synonyms) | KG + OpenAI | 1 day |
| **P1** | Run abagen → gene expression PCA (full + receptor) | abagen, sklearn | 1 day |
| **P1** | Label PCs via GO enrichment + cell-type analysis | gseapy | 1 day |
| **P1** | Download PDSP, build drug binding profiles | pandas | 0.5 day |
| **P1** | Project all PDSP drugs into PC space | numpy | 0.5 day |
| **P2** | Process ENIGMA disease maps | ENIGMA toolbox | 1 day |
| **P2** | Run NiMARE batch meta-analyses for 50 drug terms | NiMARE | 0.5 day |
| **P2** | Process OpenNeuro pharma datasets to contrast maps | fMRIPrep | 2-3 days |
| **P3** | Train Generalizer (PC-constrained dual-head MLP, phased schedule) | PyTorch | 2-4 days |
| **P3** | Train Memorizer (overfit MLP, 500+ epochs, near-zero loss) | PyTorch | 1 day |
| **P3** | Tune routing hyperparameters (steepness, midpoint) on held-out set | numpy | 0.5 day |
| **P4** | Build enrichment report generator | custom | 1 day |
| **P4** | Integrate 3D visualization + radar chart + UMAP | NiiVue/React | 3-5 days |
| **P5** | Held-out evaluation + ablations | custom | 2 days |
| **P5** | Comparison to baselines (NeuroQuery, Luppi-style, nearest-neighbor) | custom | 2 days |

**Total estimated development time: ~3-4 weeks focused work**

---

## 10. KEY PYTHON DEPENDENCIES

```
# Core
numpy, pandas, scipy, scikit-learn, joblib, networkx

# Neuroimaging
nilearn, nibabel, neuromaps, abagen, NiMARE

# Gene enrichment
gseapy

# Deep learning
torch (PyTorch >= 2.0)

# Embeddings
openai (API client)

# Pharmacology
requests, chembl_webresource_client

# Visualization
matplotlib, seaborn, plotly, umap-learn

# Web framework (for API serving)
fastapi, uvicorn
```

---

## 11. READINESS FOR TRAINING AND NEXT PHASES

Before training the text-to-brain embedding (Generalizer + Memorizer) and running gene PCA / ontology expansion:

1. **Source data local:** Run `ensure_all_brain_map_data_local.py --download` (or `rebuild_all_caches.py --ensure-data`) so NeuroQuery, NeuroSynth, NeuroVault, neuromaps, ontologies, and atlas NIfTIs are present. For the full NeuroVault acquisition guide (~104 collections): `ensure_all_brain_map_data_local.py --download --use-curated-neurovault`.
2. **Parcellation complete:** Run `rebuild_all_caches.py --n-jobs 30` so all caches are parcellated to **Glasser+Tian (392)**. Any data in Schaefer or other atlases is reparcellated to Glasser+Tian. This produces decoder_cache, neurosynth_cache, unified_cache, neurovault/neuromaps/enigma/abagen caches (if data present), and **decoder_cache_expanded**.
3. **Check readiness:** Run `check_training_readiness.py` to verify atlas and cache exist with 392 parcels and to print next steps.
4. **Verify parcellation:** Run `verify_parcellation_and_map_types.py` to confirm all caches use 392 parcels and map types (fMRI, structural, PET).
5. **Train:** Use `merged_sources` first (NQ+NS+neuromaps+neurovault+enigma+abagen, no ontology): `train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources ...`. Use `decoder_cache_expanded` (with ontology) only when built with `--build-expanded`.
6. **Gene PCA (optional):** For PC-constrained Generalizer and gene-head training, ensure `build_expanded_term_maps.py` was run with `--gene-pca-variance 0.95` (rebuild does this when abagen cache exists), or run `run_gene_pca_phase1.py` through phase 4 and merge into expanded cache.

Pipeline parcellation is **Glasser 360 + Tian S2 = 392 parcels** only. Data in Schaefer or other atlases must be reparcellated to Glasser+Tian before use.
