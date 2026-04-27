# Build Maps, Cache, and Training Sets — Step-by-Step Pipeline

This document is a **thorough, step-by-step guide** to building all maps, caches, and training sets used by the NeuroLab enrichment and text-to-brain pipeline.

**Critical path caches (PDSP Ki, Gene PCA basis, Training embeddings):** See [CRITICAL_PATH_CACHES_SPEC.md](CRITICAL_PATH_CACHES_SPEC.md) for detailed specs. It is intended to be passed to another agent or used as a single reference for reproducing the full build.

**Run all commands from the querytobrain repository root** (the directory that contains `neurolab/`). Paths are given relative to that root unless noted.

> **Production note:** Several optional/experimental scripts referenced below were removed during the production cleanup (PDSP Ki, ChEMBL supplementary binding, Gene PCA phases, ACPI / OpenNeuro derivatives auxiliary downloaders, the standalone ontology graph cache builder, and the multi-source pharma wrapper). The core build pipeline at `run_full_cache_build.py` runs without them. To restore any of these for advanced experimentation, recover from git history (commit `c094235` and earlier).

---

## 0. One-command full build (with optional background Semantic Scholar)

**Script:** `neurolab/scripts/run_full_cache_build.py`

Runs the full pipeline: atlas → decoder → neurosynth → merge → neurovault → neuromaps → **expanded ENIGMA** (CT+SA+SubVol+splits) → pharma neurosynth → abagen → merged_sources. Optionally starts Semantic Scholar compound fetch in background.

```bash
python neurolab/scripts/run_full_cache_build.py
python neurolab/scripts/run_full_cache_build.py --start-semantic-scholar-background  # compound literature in background
python neurolab/scripts/run_full_cache_build.py --quick  # smaller caps for testing
```

---

## 1. Overview: What Gets Built and in What Order

| Step | What | Output | Used for |
|------|------|--------|----------|
| 0 | Combined atlas (cortical + subcortical + brainstem) | `neurolab/data/combined_atlas_427.nii.gz` | Single parcellation for all maps (**427 parcels**: Glasser 360 + Tian S2 + Brainstem + BFB/Hyp) |
| 0a | Ontologies (OBO/OWL) | `neurolab/data/ontologies/*.owl` | Ontology expansion (term→related terms) |
| 0b | Neuromaps raw data (optional) | `neurolab/data/neuromaps_data/` | Building neuromaps cache without re-downloading |
| 0c | NeuroVault curated (optional) | `neurolab/data/neurovault_curated_data/` or `neurovault_data/` (manifest.json + NIfTIs) | Task-contrast maps + descriptions for future cache |
| 1 | NeuroQuery term maps | `decoder_cache/` (term_maps.npz, term_vocab.pkl) | Cognitive decoder; base for expansion |
| 2 | NeuroSynth term maps | `neurosynth_cache/` (same format) | Merge with NQ for larger cognitive set |
| 3 | Merge NQ + NeuroSynth | `unified_cache/` (same format) | Single cognitive cache (NQ+NS) |
| 4 | Neuromaps annotations | `neuromaps_cache/` (annotation_maps.npz, annotation_labels.pkl) | Biological enrichment; optional merge into expanded set |
| 4b | ENIGMA disorder maps (optional) | `enigma_cache/` (term_maps.npz, term_vocab.pkl) | Structural (cortical thickness, subcortical volume) case-control maps |
| 4c | abagen gene expression (optional) | `abagen_cache/` (term_maps.npz, term_vocab.pkl) | Gene expression maps (receptor, cell-type, disease genes) |
| 0d | Open pharma data (optional) | PDSP Ki, OpenNeuro pharma, NeuroVault pharma (see Section 6c) | Drug–target affinities; raw pharma fMRI; pharma contrast maps |
| 5 | Ontology + optional neuromaps/receptor/ENIGMA/abagen expansion | `unified_cache_expanded/` or `decoder_cache_expanded/` (term_maps.npz, term_vocab.pkl) | **Training set** for text-to-brain embedding |
| 6 | Train text-to-brain model | `embedding_model/` (encoder + MLP head, split_info.pkl) | Inference: any text → parcel-D map |

**Parcellation (single source of truth):** All caches and models MUST use the atlas returned by `neurolab/parcellation.py:get_masker()`.
Current default is **427 parcels** (Glasser 360 + Tian S2 32 + Brainstem Navigator + BFB/Hyp), written to `neurolab/data/combined_atlas_427.nii.gz`.
`get_combined_atlas_path()` prefers 427 → 450 → 392; `get_n_parcels()` returns the actual count from the atlas file. Any mention of 400/414 parcels is **legacy**.

**One-shot option:** Steps 1–5 (and optionally 0b, 4) can be run in one go with `build_all_maps.py --expand` (see Section 7).

---

## 2. Prerequisites

- **Python:** 3.9+ (3.10+ recommended).
- **Dependencies:** Install from `neurolab/requirements-enrichment.txt` (or equivalent). Key packages: `neuroquery`, `nilearn`, `nibabel`, `numpy`, `scipy`, `obonet`, `networkx`, `rdflib`, `neuromaps`, `nimare` (for NeuroSynth), `sentence-transformers` (for training). On Windows, **setuptools &lt; 82** may be required for `neuromaps` (pkg_resources).
- **Data directory:** Default base is `neurolab/data/`. Create it if missing: `mkdir -p neurolab/data`.

---

## 3. Step 0: Build Combined Atlas (Cortical + Subcortical)

All map caches and inference use a single parcellation. The combined atlas is built once; `neurolab/parcellation.py` provides `get_combined_atlas_path()`, `get_n_parcels()`, and `get_masker()` so every build script and inference path uses the same atlas.

**Script:** `neurolab/scripts/build_combined_atlas.py`

**Output:** `neurolab/data/combined_atlas_427.nii.gz` (default: Glasser 360 + Tian S2 32 + Brainstem + BFB/Hyp = **427 parcels**)

**Commands (from repo root):**

```bash
# Build combined atlas (427 parcels: Glasser + Tian + Brainstem + BFB/Hyp) — matches run_full_cache_build
python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem+bfb+hyp --output neurolab/data/combined_atlas_427.nii.gz

# Minimal (392 parcels): Glasser + Tian only
python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian
```

**Fetch data:** Run this step first so the atlas components (Glasser, Tian) are downloaded. Then run Step 0a (ontologies), 0b (neuromaps), 0c (neurovault) as needed. No other step re-downloads the atlas.

---

## 4. Step 0a: Download Ontologies

Ontologies provide **term → related terms** (synonyms, parents, children) for expansion. All OBO/OWL files in the ontology directory are loaded and merged.

**Script:** `neurolab/scripts/download_ontologies.py`

**Default output:** `neurolab/data/ontologies/`

**Commands (from repo root):**

```bash
# Default: MF + UBERON (skip UBERON if you want to avoid large download)
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies

# Skip UBERON (smaller)
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --no-uberon

# Extra: BFO, RO, Cognitive Atlas, CogPO, NBO
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --extra

# Include Cognitive Atlas explicitly (recommended for cognitive terms)
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --cognitive-atlas --bfo --ro

# Clinical: MONDO, HPO, ChEBI lite, MFOEM, PATO (for ENIGMA disease maps, pharmacological fingerprints, symptom queries)
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --clinical
# Disease + phenotype only (skip ChEBI):
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --clinical --no-chebi
# Full ChEBI instead of lite (large):
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --clinical --chebi-full
# DOID instead of MONDO (use one or the other; DOID ~12K, MONDO ~30K):
python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --clinical --doid
```

**Result:** OWL/OBO files in `neurolab/data/ontologies/`. With `--clinical`: `mondo.owl`, `hp.owl`, `chebi_lite.obo` (drug roles/classes), `mfoem.owl`, `pato.owl`. MONDO and HPO have cross-references (`has_phenotype`); the unified graph gets disease→symptom edges so queries like "anhedonia" can traverse to "major depressive disorder" and pull ENIGMA depression context. Expansion and KG context load every file in this directory (including `.obo`).

---

## 4a. Step 0b: Download Neuromaps Raw Data (Optional but Recommended)

Neuromaps annotations (receptors, structure, etc.) are used for biological enrichment and can be merged into the expanded training set. Raw annotation files can live in the repo so the neuromaps cache build does not download at build time.

**Script:** `neurolab/scripts/download_neuromaps_data.py`

**Default output:** `neurolab/data/neuromaps_data/`

**Command:**

```bash
python neurolab/scripts/download_neuromaps_data.py
```

**Options:** `--output-dir`, `--space MNI152`, `--tags`, `--format`. Default fetches all MNI152 annotations. Run once; then `build_neuromaps_cache.py` uses this directory by default (no download during cache build).

**Note (Windows):** If you see `'str' object has no attribute 'mkdir'`, the script patches neuromaps’ internal fetch to use `Path`; ensure you have the latest version of the script. If you see `ModuleNotFoundError: pkg_resources`, install `setuptools<82`.

---

## 4b. Step 0c: Download NeuroVault Curated Data (Optional)

NeuroVault provides **task-contrast statistical maps** with free-text names and descriptions (e.g. "face vs baseline", "fearful faces > neutral"). Curated collections give real contrast maps + natural-language labels for text-to-brain training.

**Recommended: full acquisition guide** (Tiers 1–4, **126 collections**; after averaging, **~2,700–5,000 maps**):

```bash
python neurolab/scripts/download_neurovault_curated.py --all
```

**Output:** `neurolab/data/neurovault_curated_data/` (manifest.json + downloads/neurovault/). The **download** stores all **raw images** in those collections; run **`build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level`** to get the **curated training set** (~2–4K maps). `rebuild_all_caches.py` prefers this over `neurovault_data` when present.

**Options:** `--tiers 1 2 3`, `--include-pharma`, `--include-wm-atlas`, `--include-slugs`, `--max-images N` (cap for quick test).

**Alternative: legacy BrainPedia** (1952 only):

```bash
python neurolab/scripts/download_neurovault_data.py
```

**Output:** `neurolab/data/neurovault_data/` (images under `downloads/neurovault/`, manifest at `manifest.json`). Default collections: 1952 (BrainPedia), 4337 (HCP). Options: `--collections 1952 4337 ...`, `--mode download_new|overwrite|offline`, `--timeout 60`.

**Result:** NIfTI images per collection and a `manifest.json` with paths, `name`, `contrast_definition`, `task`, `collection_id` per image.

**Where the terms (labels) come from:** Each NeuroVault image has metadata from the API. We use as the text label for that map (in order of preference): **contrast_definition** (natural language, e.g. "Watch a face image and respond to 0 or 2-back working memory task versus fixation"), else **task** (e.g. "task007_face_vs_baseline"), else **name** (e.g. "HCP_task007_sub041"). So the "terms" are already in the downloaded metadata; the cache builder below just picks one per image and writes them into `term_vocab.pkl`.

**Build (term, map) cache:** Run `build_neurovault_cache.py` to parcellate each image to the pipeline atlas (427 by default) and save `term_maps.npz` + `term_vocab.pkl` (same format as decoder cache). For **curated** data use **`--average-subject-level`** so subject-level collections are averaged by contrast (~2–4K maps). Point at the data dir you used:

```bash
python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level
# or for legacy/bulk:
python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_data --output-dir neurolab/data/neurovault_cache
```

---

## 5. Step 1: Build NeuroQuery Decoder Cache (Cognitive Term Maps)

NeuroQuery provides **term → parcellated map** via a predictive model. This step precomputes maps for (a subset or all of) the NeuroQuery vocabulary and saves them so the decoder and expansion do not call the model at runtime.

**Script:** `neurolab/scripts/build_term_maps_cache.py`

**Output:** `neurolab/data/decoder_cache/term_maps.npz`, `term_vocab.pkl`

**Format:** `term_maps.npz` contains key `term_maps`, shape `(N, get_n_parcels())` (expected **427** with the default combined atlas). `term_vocab.pkl` is a list of N strings (term labels). Row `i` corresponds to `term_vocab[i]`.

**Commands:**

```bash
# Full vocabulary (0 = no cap; ~7.5k terms; can take 1–2+ hours)
python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0

# Quick test (500 terms)
python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 500
```

**Result:** `decoder_cache/` contains N cognitive terms and their parcel-D maps (combined atlas; **427** by default).

---

## 6. Step 2: Build NeuroSynth Cache (Optional)

NeuroSynth provides **term → brain map** from coordinate-based meta-analysis (e.g. MKDA). Maps are parcellated using `get_masker()` so they match the pipeline atlas (**427-D** by default).

**Script:** `neurolab/scripts/build_neurosynth_cache.py`

**Output:** `neurolab/data/neurosynth_cache/term_maps.npz`, `term_vocab.pkl` (same format as decoder cache)

**Commands:**

```bash
# All terms (~1.3k; can take a long time)
python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache --max-terms 0

# Quick test (50 terms)
python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache --max-terms 50
```

**Requires:** `nimare` (and NeuroSynth data fetched by nimare on first run).

---

## 6b. Step 2b: Pharmacological NeuroSynth maps (optional, CC0-derived)

Batch-generate meta-analytic brain maps for a curated pharmacological vocabulary (psychedelics, stimulants, antidepressants, antipsychotics, opioids, cannabis, nootropics, etc.) using the same NiMARE + NeuroSynth coordinate database. **Derived from CC0 data — commercially clean.** Same cache format as NeuroSynth; can be merged into the unified cache for training.

**Script:** `neurolab/scripts/build_pharma_neurosynth_cache.py`

**Output:** `neurolab/data/pharma_neurosynth_cache/term_maps.npz`, `term_vocab.pkl`

**Commands:**

```bash
# All built-in PHARMA_TERMS (reuses neurolab/data/neurosynth_data from Step 2)
python neurolab/scripts/build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache

# Only specific terms
python neurolab/scripts/build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache --terms ketamine caffeine psilocybin
```

**Requires:** Same as Step 2 (`nimare`, NeuroSynth data). Run Step 2 first (or share `--data-dir`) so the NeuroSynth dataset is available. Terms with fewer than `--min-studies` (default 5) are skipped.

**Merge with unified cache:** After building, run the merge script again: `--neuroquery-cache-dir neurolab/data/unified_cache --neurosynth-cache-dir neurolab/data/pharma_neurosynth_cache --output-dir neurolab/data/unified_with_pharma` so pharmacological terms are included in the training set and in query resolution.

---

## 6c. Download all open pharmacological data (optional)

Scripts to pull in **all open data** you listed: PDSP Ki, OpenNeuro pharma fMRI, NeuroVault pharmacological collections. No license filtering (get everything that is publicly available).

**One command (runs all three downloaders):**

```bash
python neurolab/scripts/download_all_pharma_data.py
```

**Individual scripts:**

| Script | What it does | Output |
|--------|----------------|--------|
| `download_pdsp_ki.py` | PDSP Ki database (NIMH); ~98K Ki values, drugs × targets | `neurolab/data/pdsp_ki/KiDatabase.csv` (or manual if server requires browser) |
| `download_openneuro_pharma.py` | OpenNeuro drug-challenge datasets (ds006072 psilocybin, ds003059 LSD/psilocybin, etc.) | `neurolab/data/openneuro_pharma/<dataset_id>/` (requires openneuro-py or datalad) |
| `download_neurovault_pharma.py` | Search NeuroVault for drug/pharmacological collections, download all images | `neurolab/data/neurovault_pharma_data/` (manifest.json + downloads/) |

- **PDSP Ki:** Use for drug–brain fingerprints (weight receptor maps by 1/Ki). If direct CSV URL fails, the script prints the manual download page.
- **OpenNeuro:** Raw fMRI; run `--list-only` for dataset IDs and datalad commands if you prefer not to install openneuro-py.
- **NeuroVault pharma:** Then run `build_neurovault_cache.py --data-dir neurolab/data/neurovault_pharma_data` to parcellate and build a (label, map) cache.

Pharmacological **meta-analysis maps** (NiMARE/NeuroSynth) are built by **Step 2b** (`build_pharma_neurosynth_cache.py`), not by these download scripts.

### 6d. Nootropics knowledge base (optional)

Verify and build knowledge graph from `nootropics_knowledge_base_expanded.csv`. Outputs triples and ontology-compatible format for embedding.

**Script:** `neurolab/scripts/build_nootropics_knowledge_graph.py`

**Output:** `neurolab/data/nootropics_kb/` — triples, ontology index, validation report, NetworkX graph.

```bash
# Build and copy to ontology dir (for embedding with other ontologies)
python neurolab/scripts/build_nootropics_knowledge_graph.py --copy-to-ontology-dir neurolab/data/ontologies

# Verify only
python neurolab/scripts/build_nootropics_knowledge_graph.py --verify-only
```

Place `nootropics_ontology_index.json` in `neurolab/data/ontologies/`; `load_ontology_index()` and `embed_ontology_labels.py` will include it.

### 6e. Nootropic/compound vocabulary + Semantic Scholar expansion (optional)

Fetch papers, citations, and facts for nootropic/compound terms via Semantic Scholar API. Expands triples for compounds not well-covered by NeuroSynth. **Rate-limited for use without API key** (5 s delay, retry on 429).

**Script:** `neurolab/scripts/fetch_semantic_scholar_compounds.py`

**Output:** `neurolab/data/compound_literature/compound_literature.json`, `compound_triples.json`

```bash
# All built-in nootropic/compound terms (5 s delay, no API key)
python neurolab/scripts/fetch_semantic_scholar_compounds.py --output-dir neurolab/data/compound_literature

# Run in background so you can keep working
python neurolab/scripts/fetch_semantic_scholar_compounds.py --background --resume

# Use PHARMA_TERMS from build_pharma_neurosynth_cache
python neurolab/scripts/fetch_semantic_scholar_compounds.py --from-pharma-terms --max-papers 10

# Custom compounds
python neurolab/scripts/fetch_semantic_scholar_compounds.py --compounds alpha-GPC bacopa lion\'s mane --max-papers 20
```

Set `S2_API_KEY` for higher rate limits. Requires `requests`.

### 6f. ChEMBL supplementary binding cache (optional)

Fill PDSP gaps for compounds without Ki values. Uses ChEMBL REST API.

**Script:** `neurolab/scripts/build_chembl_binding_cache.py`

**Output:** `neurolab/data/chembl_binding_cache/chembl_profiles.npz`, `compound_names.json`

```bash
python neurolab/scripts/build_chembl_binding_cache.py --output-dir neurolab/data/chembl_binding_cache
python neurolab/scripts/build_chembl_binding_cache.py --pdsp-cache neurolab/data/pdsp_cache  # supplement PDSP gaps
```

Requires `chembl-webresource-client`. License: CC-BY-SA 3.0.

### 6g. Luppi multi-drug FC maps (optional)

Luppi et al. 2023 (Science Advances): 10 drugs, pharmacologically induced FC. Resample to pipeline atlas via `get_masker()`.

**Script:** `neurolab/scripts/download_luppi_fc_maps.py`

```bash
python neurolab/scripts/download_luppi_fc_maps.py --list-only  # print Cambridge/Science Advances URLs
```

Data may require manual download from Cambridge repository or Science Advances supplementary.

### 6h. OpenNeuro fMRIPrep derivatives (optional)

Preprocessed pharma datasets from OpenNeuroDerivatives (avoids running fMRIPrep on raw data).

**Script:** `neurolab/scripts/download_openneuro_derivatives.py`

```bash
python neurolab/scripts/download_openneuro_derivatives.py --list-only
python neurolab/scripts/download_openneuro_derivatives.py --output-dir neurolab/data/openneuro_derivatives
```

Requires datalad or git. If derivative repos do not exist for ds006072/ds003059, use `download_openneuro_pharma.py` for raw data.

### 6i. ACPI resting-state (optional)

Addiction Connectome Preprocessed Initiative: 192 subjects, cannabis + cocaine, 8 pipelines.

**Script:** `neurolab/scripts/download_acpi_resting_state.py`

```bash
python neurolab/scripts/download_acpi_resting_state.py --list-only
```

NITRC may require registration; script prints manual acquisition steps.

### 6j. Ontology graph cache (optional)

Build queryable meta-graph with bridge edges (CogAt ↔ MONDO ↔ HPO ↔ ChEBI) for training data multiplication.

**Script:** `neurolab/scripts/build_ontology_graph_cache.py`

**Output:** `neurolab/data/ontology_graph_cache/ontology_meta_graph.pkl`, `metadata.json`

```bash
python neurolab/scripts/download_ontologies.py --clinical  # prerequisite
python neurolab/scripts/build_ontology_graph_cache.py --ontology-dir neurolab/data/ontologies
```

---

## 7. Step 3: Merge NeuroQuery + NeuroSynth → Unified Cognitive Cache

Merges the two cognitive caches into one vocabulary and one set of maps. For terms that appear in both, one source is chosen (`--prefer neurosynth` or `neuroquery`).

**Script:** `neurolab/scripts/merge_neuroquery_neurosynth_cache.py`

**Output:** `neurolab/data/unified_cache/term_maps.npz`, `term_vocab.pkl`

**Command:**

```bash
python neurolab/scripts/merge_neuroquery_neurosynth_cache.py \
  --neuroquery-cache-dir neurolab/data/decoder_cache \
  --neurosynth-cache-dir neurolab/data/neurosynth_cache \
  --output-dir neurolab/data/unified_cache \
  --prefer neurosynth
```

**Result:** Single cognitive cache (NQ+NS) used as the base for expansion and for query/decoder when you want the combined set.

---

## 8. Step 4: Build Neuromaps Annotation Cache (Biological Maps)

Fetches neuromaps annotations (e.g. receptors, structure), parcellates them to the pipeline atlas via `get_masker()` (**427-D** by default), and saves a matrix of maps plus labels. Used for biological enrichment and optionally merged into the expanded training set.

**Script:** `neurolab/scripts/build_neuromaps_cache.py`

**Output:** `neurolab/data/neuromaps_cache/annotation_maps.npz` (key `matrix`, shape `(M, get_n_parcels())`), `annotation_labels.pkl` (list of M strings)

**Default data source:** `neurolab/data/neuromaps_data/` (from Step 0b). If that directory exists and contains fetched files, no download happens. Otherwise the script downloads via neuromaps.

**Commands:**

```bash
# All MNI152 annotations (uses neurolab/data/neuromaps_data by default)
python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache

# Limit to 30 for quick test
python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache --max-annot 30
```

**Note (Windows):** Paths with backslashes (e.g. `H:\...`) can cause issues; the script resolves NIfTI paths from `data_dir/annotations/source/desc/` so parcellation works with local data.

---

## 8b. Step 4b: ENIGMA disorder cache (optional, expanded)

ENIGMA Toolbox provides case-control effect size maps (cortical thickness, surface area, subcortical volume) for many disorders. **Expanded** per [Complete-audit-of-ENIGMA-consortium-datasets](Complete-audit-of-ENIGMA-consortium-datasets-for-brain-mapping-expansion.md): CT + SA + SubVol + age splits (ADHD adult/pediatric, MDD adult/adolescent, OCD adult/pediatric) + epilepsy subtypes (LTLE, RTLE, GGE, all) + asymmetry. ~50+ maps from toolbox.

**Script:** `neurolab/scripts/build_enigma_cache.py`

**Requirements:** `pip install enigmatoolbox`. For proper DK→pipeline resampling, provide a precomputed mapping file (`--dk-to-schaefer path/to/mapping.npy`, shape n_dk × get_n_parcels()) or use the replicate fallback (no spatial mapping).

```bash
python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache
# With a DK→pipeline mapping file (build once, e.g. from surface overlap):
python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache --dk-to-schaefer neurolab/data/dk_to_pipeline_atlas.npy
```

**Output:** `term_maps.npz` (N, get_n_parcels()), `term_vocab.pkl` (e.g. "schizophrenia cortical thickness", "major depression cortical thickness"). Merge with `build_expanded_term_maps.py --enigma-cache-dir neurolab/data/enigma_cache --save-term-sources`.

---

## 8c. Step 4c: abagen gene expression cache (optional)

Allen Human Brain Atlas gene expression parcellated to the pipeline atlas via abagen. Adds receptor genes (HTR2A, DRD2, …), cell-type markers (PVALB, SST), and disease-relevant genes (BDNF, SLC6A4).

**Script:** `neurolab/scripts/build_abagen_cache.py`

**Requirements:** `pip install abagen nilearn`. abagen downloads AHBA data on first run.

```bash
python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache
# Custom gene list:
python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --genes HTR2A DRD2 SLC6A4 BDNF
```

**Output:** `term_maps.npz` (N, n_parcels), `term_vocab.pkl` (e.g. "serotonin 2A receptor gene expression (HTR2A)"). Merge with `build_expanded_term_maps.py --abagen-cache-dir neurolab/data/abagen_cache --save-term-sources`.

### 8c.1 Allen Institute (AHBA): thousands of genes and overweighting

The Allen Human Brain Atlas (AHBA) provides **thousands** of genes. Including all of them in the training set can **overweight** the loss toward gene-expression maps and hurt generalization for cognitive/task queries.

**Two ways to avoid overweighting:**

1. **Sample weighting (already in place):** Training uses `term_sources.pkl` and applies a lower **sample weight** to abagen (default **0.4**) than to direct NQ/NS (1.0) or neurovault (0.8). So each abagen sample contributes less to the loss. You can lower the abagen weight further in `train_text_to_brain_embedding.py` (`SAMPLE_WEIGHT_BY_SOURCE["abagen"]`) if needed.

2. **Split / cap the number of gene maps:**
   - **When building abagen cache:** Use a **curated list** (default ~20 genes) or use **all genes** with a cap:
     - `--all-genes --max-genes 500` → build cache with 500 randomly sampled genes (reproducible with `--seed`).
   - **When merging into expanded cache:** Use `--max-abagen-terms N` so at most N abagen terms are added. Selection is **tiered** (literature-aligned): Tier 1 = receptor genes (~250), Tier 2 = WGCNA-style cluster medoids (default 32), Tier 3 = **residual-variance** (default; informed by Fulcher et al.) or cluster medoids to fill remaining slots. See [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md).

**Recommendation:** Use **both**: (a) cap abagen when building or merging (e.g. 300–500 genes) so the training set isn’t dominated by gene maps, and (b) use source-weighted sampling so abagen is ~10% of batches. For tiered selection: `--max-abagen-terms 500` (and optionally `--abagen-n-clusters 32`) in build_expanded_term_maps.

### 8c.2 Train genes separately or together?

**Current approach (together):** One text-to-brain model trained on cognitive + ontology + neuromaps + receptor + abagen. The MLP is **type-conditioned** (input = concat(embedding, type_one_hot)) so it learns “when type = pet_receptor, produce gene-like map.” At inference, map type is inferred from the query (e.g. “receptor”, “gene expression” → pet_receptor).

**When training together is fine:**

- You want **one model** for all queries (cognitive, structural, gene); no routing logic.
- You **cap** gene maps (e.g. 300–500) and **downweight** them (sample weight 0.4) so they don’t dominate.
- Gene labels are **augmented** with descriptive text (e.g. “serotonin 2A receptor gene expression (HTR2A)”) so the encoder has enough signal.
- You care about **cross-modal** queries (e.g. “dopamine” might mean task or receptor); a shared encoder can learn a joint space.

**When training genes separately is better:**

- You have **thousands** of gene maps and don’t want to dilute or overweight the cognitive model even with caps/weights.
- Gene queries are **keyword-like** (e.g. “HTR2A”, “DRD2”); a **retrieval path** (match query to gene label, return stored map) may work better than a learned encoder–head for genes.
- You want to **tune** architecture or loss for genes independently (e.g. different head size, or correlation loss for genes).
- You observe that **cognitive performance drops** when you add many gene samples (even with weighting); then a separate gene model (or retrieval) avoids that trade-off.

**Practical split option:**

- **Gene retrieval:** Don’t train genes in the main model. At inference: if the query matches an abagen (or receptor) label (e.g. by similarity or exact match), return that **stored map**; otherwise use the **cognitive model**. This is already supported by `term_to_map.py` (neuromaps-by-name path). You can add an abagen-by-name path: load `abagen_cache` term_vocab + term_maps; match query to labels; return the map if score &gt; threshold.
- **Separate gene model:** Train a second model on **abagen_cache only** (and optionally receptor/neuromaps). At inference: if the query looks like a gene/receptor query (keywords), call the gene model; else call the cognitive model. Two checkpoints, one routing rule.

**Recommendation:** Start with **together** (type-conditioned, cap + weight). If cognitive test metrics drop when you add genes, or gene retrieval quality is poor, add **gene retrieval** (abagen-by-name) or a **separate gene model** and route at inference.

### 8c.3 Alternative: train genes on PC maps (gene-only PCA)

You can keep genes in the **same** model but train them in a **reduced space**: fit PCA on **abagen maps only** (e.g. top 95% variance), store the **gene loadings** in the cache, and add a **gene head** that predicts those PC loadings; at inference, inverse transform to parcel space.

**Steps:**

1. **Build expanded cache with gene PCA:** When merging abagen, pass `--gene-pca-variance 0.95`. The script fits PCA on the abagen rows only, saves `gene_pca.pkl`, `gene_loadings.npz`, and `abagen_term_indices.pkl` in the expanded cache dir.
2. **Train:** Run `train_text_to_brain_embedding.py` on that cache. The trainer loads the gene PCA and build a **second head** (gene head) with output dim = number of PC components. For abagen samples it trains on the loadings (gene head); for others on the usual get_n_parcels()-D or global PC targets (main head). It saves `gene_head_weights.pt` and a copy of `gene_pca.pkl` in the model dir.
3. **Inference:** `TextToBrainEmbedding` loads the gene head and gene PCA when present. For queries inferred as **pet_receptor** (e.g. “receptor”, “gene expression”), it uses the gene head and `gene_pca.inverse_transform()` to get the get_n_parcels()-D map; for other types it uses the main head (and global PCA inverse if applicable).

**Why use it:** Gene maps often live in a lower-dimensional subspace; predicting K loadings (e.g. K ≈ 50–100 for 95% variance) can be easier than get_n_parcels()-D and can reduce overfitting. The main head stays get_n_parcels()-D (or global PC) for cognitive/structural terms.

**Options:** `--gene-pca-variance 0.95` (or 0.9) in `build_expanded_term_maps.py`. No extra trainer flags; the trainer enables the gene head automatically when it finds `gene_pca.pkl` and `gene_loadings.npz` in the cache.

---

## 9. Step 5: Build Expanded Cache (Training Set: Cognitive + Ontology + Optional Neuromaps/Receptor)

This step produces the **full (label, map) training set** used to train the text-to-brain embedding. It:

1. Starts from a **cognitive cache** (decoder_cache or unified_cache).
2. **Forward expansion:** For each ontology label not already in the cache, finds related cache terms via the ontology and assigns a **derived map** = weighted average of those cache maps (weights from relation type: synonym 0.95, child 0.85, parent 0.8; see `RELATION_WEIGHTS` in `ontology_expansion.py`).
3. **Reverse expansion:** For ontology labels that appear only as “related” of cache terms (e.g. parent “executive function” of cache term “working memory”), assigns a derived map = weighted average of all cache terms that point to them (not a single child’s map).
4. Optionally **merges neuromaps**, **receptor**, **ENIGMA** disorder maps, and **abagen** gene expression into the same term list and map matrix.

**When a term already has a direct map** (e.g. “executive function” in NQ/NS cache), that map is always used; expansion only **adds** terms, it never overwrites cache terms.

**Script:** `neurolab/scripts/build_expanded_term_maps.py`

**Input:** A cognitive cache dir (e.g. `unified_cache` or `decoder_cache`) with `term_maps.npz` and `term_vocab.pkl`. Ontology dir (e.g. `neurolab/data/ontologies`). Optionally neuromaps cache dir, receptor path, ENIGMA cache dir, and/or abagen cache dir.

**Output:** A directory (e.g. `unified_cache_expanded` or `decoder_cache_expanded`) containing `term_maps.npz` and `term_vocab.pkl` with **expanded** term list and maps (same format as cognitive cache).

**Commands:**

```bash
# Expand from unified cache (NQ+NS) + ontologies only
python neurolab/scripts/build_expanded_term_maps.py \
  --cache-dir neurolab/data/unified_cache \
  --ontology-dir neurolab/data/ontologies \
  --output-dir neurolab/data/unified_cache_expanded

# Expand and include neuromaps + receptor + ENIGMA + abagen
python neurolab/scripts/build_expanded_term_maps.py \
  --cache-dir neurolab/data/unified_cache \
  --ontology-dir neurolab/data/ontologies \
  --output-dir neurolab/data/unified_cache_expanded \
  --neuromaps-cache-dir neurolab/data/neuromaps_cache \
  --receptor-path path/to/receptor_data_parcellated_to_pipeline.csv \
  --enigma-cache-dir neurolab/data/enigma_cache \
  --abagen-cache-dir neurolab/data/abagen_cache \
  --save-term-sources
```

**Options:**

- `--min-cache-matches` (default **2**): add an ontology term only if it expands to at least this many cache terms. Using 1 lets a single match inherit a map verbatim, which can be misleading.
- `--min-pairwise-correlation` (e.g. `0.3`): if set, skip an ontology term when the mean pairwise correlation among its contributing cache maps is below this threshold (avoids meaningless averages when maps are too dissimilar).
- `--save-term-sources`: write `term_sources.pkl` (values: `direct` | `ontology` | `neuromaps` | `receptor` | `enigma` | `abagen`) for sample weighting in training. `build_all_maps.py --expand` enables this by default.
- `--enigma-cache-dir`: merge ENIGMA disorder maps (from `build_enigma_cache.py`); source = `enigma`, map_type = structural.
- `--abagen-cache-dir`: merge abagen gene expression maps (from `build_abagen_cache.py`); source = `abagen`, map_type = pet_receptor.
- `--receptor-path`: Receptor matrices must match `get_n_parcels()`; do not pass a Schaefer-400 matrix unless you explicitly run the legacy pipeline.
- `--max-abagen-terms` (default **0** = all): when merging abagen, add at most this many terms using **tiered selection** (receptor + cluster medoids). Use to avoid overweighting and redundancy (see §8c.1 and [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md)).
- `--abagen-n-clusters` (default **32**): number of Tier-2 co-expression clusters when max-abagen-terms is set (WGCNA-style; Hawrylycz et al.).
- `--gradient-pc-label-style distinct`: optimized labels for gradient PCs (PC1 brain_context phrasing, PC2/3 maximally different). Used by `run_full_cache_build` for merged_sources.
- `--abagen-gene-info`, `--abagen-pca-variance 0.95`, `--truncate-to-392`: gene label enrichment, denoising, and 392-parcel output. Run `download_gene_info.py` first for gene_info.json.

**Output files:** `term_maps.npz`, `term_vocab.pkl`, and optionally `term_sources.pkl`.

**Result:** One (label, map) set: cognitive + ontology-derived + (optionally) neuromaps + receptor + ENIGMA disorder + abagen gene expression. This directory is the **training set** for the next step.

---

## 9b. Step 5b: Validate expanded maps (before training)

**Never train on data you haven't visually inspected.** Run this after building the expanded cache and before spending hours on training.

**1. Get data-driven relation weights**

```bash
python neurolab/scripts/ontology_brain_correlation.py \
  --cache-dir neurolab/data/unified_cache \
  --ontology-dir neurolab/data/ontologies \
  --output-weights neurolab/data/relation_weights.json
```

**2. Rebuild expansion with data-driven weights**

```bash
python neurolab/scripts/build_expanded_term_maps.py \
  --cache-dir neurolab/data/unified_cache \
  --ontology-dir neurolab/data/ontologies \
  --output-dir neurolab/data/unified_cache_expanded \
  --relation-weights-file neurolab/data/relation_weights.json \
  --min-cache-matches 2 \
  --min-pairwise-correlation 0.3 \
  --save-term-sources
```

**3. Visual sanity check**

```bash
python neurolab/scripts/inspect_expanded_maps.py \
  --cache-dir neurolab/data/unified_cache_expanded \
  --n-random 20 \
  --output-dir neurolab/data/map_inspections
```

Review the PNGs and quality flags. If many maps are flagged (washed-out average, nearly flat, or high correlation with global mean), tighten `--min-pairwise-correlation` or raise `--min-cache-matches` to 3 before training.

---

## 10. Step 6: Train Text-to-Brain Embedding Model

Trains a model that maps **text → parcellated map** (shape `get_n_parcels()`, default **427**): encoder (TF-IDF, sentence-transformers, or OpenAI) + MLP regression head. Training data = the expanded cache (term_vocab + term_maps).

**Script:** `neurolab/scripts/train_text_to_brain_embedding.py`

**Input:** A cache directory with `term_maps.npz` and `term_vocab.pkl` (e.g. `unified_cache_expanded`).

**Output:** `--output-dir` (e.g. `neurolab/data/embedding_model/`): saved encoder + MLP weights, `split_info.pkl` (train/val/test indices), and metadata.

**Commands:**

```bash
# Train on expanded cache with PubMedBERT (recommended for best test correlation)
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/unified_cache_expanded \
  --output-dir neurolab/data/embedding_model \
  --encoder sentence-transformers \
  --encoder-model NeuML/pubmedbert-base-embeddings \
  --head-hidden 1024 \
  --dropout 0.2 \
  --weight-decay 1e-5 \
  --pca-components 100 \
  --early-stopping

# Quick test (smaller cache, fewer epochs)
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/decoder_cache_expanded \
  --output-dir neurolab/data/embedding_model \
  --encoder sentence-transformers \
  --encoder-model NeuML/pubmedbert-base-embeddings \
  --max-terms 1000 \
  --epochs 10
```

**Important options:** `--cache-dir` (training set), `--encoder` (tfidf | sentence-transformers | openai), `--encoder-model`, `--head-hidden`, `--head-hidden2` (optional second hidden layer, e.g. 256, for semantic vs spatial nonlinearity), `--dropout`, `--weight-decay`, `--pca-components`, `--pca-variance` (e.g. 0.9 to capture ≥90% variance), `--early-stopping`, `--final-retrain-on-train-and-val`.

**KG context (embedding ontology triples into text):** Use `--kg-context-hops 1` (or `2`) and `--ontology-dir neurolab/data/ontologies` to append ontology relations to each term before encoding (e.g. "working memory | parent: executive function | child: n-back"). **1 hop** = direct relations only (parent, child, synonym); **2 hops** = also relations of those neighbors. This gives the encoder hierarchy context; the same KG context is applied at inference when the model was trained with it. Default is 0 (no KG context).

**Cognitive tasks ontology:** For task-related queries (e.g. "2-back task", "image doing working memory"), include **Cognitive Atlas** (and optionally **CogPO**) in `--ontology-dir`. Download with: `python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --cognitive-atlas --bfo --ro`. At inference, the pipeline finds every ontology concept that appears in the query (e.g. "2-back", "working memory") and appends their triples so the encoder sees one connected block: e.g. "image doing a 2-back task | parent: working memory, n-back | child: ... | 2hop working memory -> parent: executive function".

**Semantic KG context (embedding + cosine similarity):** Use `--kg-context-mode semantic` (and optionally `--kg-semantic-top-k 5`) so that at **training and inference** the text (term or query) is embedded with the same encoder, compared to precomputed embeddings of ontology labels, and the **top-k closest nodes** by cosine similarity supply the triples. The description is then augmented with those triples and the augmented text is encoded for the MLP. This pulls in related concepts (e.g. "n-back", "working memory") even when the exact words don't appear. Label embeddings are built once (ontology labels embedded with the same encoder) and cached in the model dir per encoder and ontology dir.

**Full flow (recommended for image/term → map):** (1) Embed all ontology labels (and optionally restrict with `--embed-ontology-sources` to control cost). (2) For each term/text: embed it, compute cosine similarity to ontology nodes, take the closest nodes (top-k; optional sim_floor). (3) Expand the description with ontology triples from those nodes (parent, child, synonym, etc.). (4) Encode this augmented description and train the MLP on (embedding, map). Composite descriptions (e.g. "attentional task on disease population") naturally pull in nodes from multiple ontologies if you embed cognitive + clinical sources; top-k and sim_floor already control how many nodes and how strict the match. Multiple close nodes are handled by taking up to top-k nodes and merging their triples (ranked by similarity × relation weight); parameters `--kg-semantic-top-k`, `--kg-sim-floor`, `--kg-max-triples` tune this.

**Restricting which ontologies are embedded (recommended with clinical ontologies):** With `--clinical` you get MONDO, HPO, ChEBI lite, MFOEM, PATO (~60K+ labels); embedding all of them is expensive. Use **`--embed-ontology-sources`** to embed only a subset (e.g. cognitive only). Labels from other files remain in the graph for **substring matching** and expansion; only semantic retrieval is limited. Example: `--embed-ontology-sources cogat.v2.owl mf.owl nbo.owl CogPOver1.owl` so "working memory", "n-back", etc. are embedded for cosine search, while MONDO/HPO/ChEBI terms are still found when they appear in the query (e.g. "schizophrenia", "anhedonia"). This is saved in config and used at inference.

**Ontology embedding strategy (best for retrieval):** Use **text-embedding-3-large** for best informal-to-formal matching. By default, each ontology label is embedded as **rich text** (label + synonyms + "Type of: …" + "Related: …") via `build_embedding_text_for_label`, so the embedding captures ontological neighborhood; disable with `kg_embed_rich_text: false` in config to embed labels only. With `embed_ontology_sources` set to cognitive-only files, cost to embed ~2K–10K terms once is small. Cache is keyed by model + ontology dir + rich flag + embed_sources hash; pre-normalized vectors give fast cosine retrieval at inference. Optional: use the API's `dimensions` option (e.g. `dimensions=1536` for text-embedding-3-large, which defaults to 3072) to halve storage and speed retrieval (Matryoshka). Use the same dimension consistently for training and inference.

**Natural-language KG style (depth 1, token budget):** Use `--kg-context-style natural` with `--kg-sim-floor 0.4` and `--kg-max-triples 15` so that triples are formatted as natural language ("n-back is a type of working memory. n-back measures executive function.") and **prepended** to the query. Only **depth-1** triples from the selected nodes; triples are ranked by (query_sim × relation_weight) and capped at `max_triples` to avoid diluting the embedding. With OpenAI embeddings, a sim_floor of 0.4 keeps genuinely related nodes; tune top_k (3–7), sim_floor (0.3–0.5), and max_triples (10–20) on validation queries.

**Sample weights:** If the cache dir contains `term_sources.pkl` (from `build_expanded_term_maps.py --save-term-sources`), training applies per-sample weights: 1.0 direct (NQ/NS), 0.6 ontology-derived, 0.4 neuromaps/receptor, so lower-quality derived maps contribute less to the loss.

**PCA:** When using `--pca-components`, the script logs explained variance; if it is below 90%, consider `--pca-variance 0.9` or a higher component count.

**Result:** A saved model that can map arbitrary text to a parcel-D map (used by query pipeline, predict_map, etc.).

**Compression use case (term → brain map, maximize fidelity):** If the goal is to **compress** (term, map) data so that any term can be turned into a brain map at inference (and you care more about fitting the data than avoiding overfitting), use:

- **No PCA** — predict full parcel-D directly so no variance is lost (`--pca-components` and `--pca-variance` omitted or 0).
- **Large MLP** — e.g. `--head-hidden 1024 --head-hidden2 512` for capacity.
- **Light regularization** — e.g. `--dropout 0.1 --weight-decay 1e-6` so the model can fit the training set well.
- **All data** — `--max-terms 0` and the largest cache you have (decoder + NeuroVault merged, or expanded).
- **Strong encoder** — PubMedBERT or OpenAI `text-embedding-3-large` for good term representation.

Example (NeuroVault, full parcel-D, no PCA):

```bash
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/neurovault_cache \
  --output-dir neurolab/data/embedding_model_neurovault \
  --encoder sentence-transformers --encoder-model NeuML/pubmedbert-base-embeddings \
  --max-terms 0 --head-hidden 1024 --head-hidden2 512 \
  --epochs 50 --dropout 0.1 --weight-decay 1e-6 --batch-size 64
```

(Omit `--pca-variance` and `--pca-components` so the model predicts parcel-D directly. At inference: term → encoder → MLP → parcellated map.)

---

## 11. One-Shot: Build All Maps + Expansion

Steps 1–5 (and optionally 0b and 4) can be run in one command so that you get the full cognitive pipeline plus expansion without running each script manually.

**Script:** `neurolab/scripts/build_all_maps.py`

**Commands:**

```bash
# Full build + expansion (no caps; can take many hours)
python neurolab/scripts/build_all_maps.py --expand

# Quick test (500 NQ, 50 NS, 30 neuromaps) + expansion
python neurolab/scripts/build_all_maps.py --quick --expand

# Skip neuromaps (cognitive only)
python neurolab/scripts/build_all_maps.py --quick --skip-neuromaps --expand
```

**What it does (in order):**

1. Build NeuroQuery → `decoder_cache`
2. Build NeuroSynth → `neurosynth_cache`
3. Merge → `unified_cache`
4. Build neuromaps cache (uses `neurolab/data/neuromaps_data` by default) → `neuromaps_cache`
5. If `--expand`: run `build_expanded_term_maps.py` with `unified_cache`, `neurolab/data/ontologies`, and (if present) `neuromaps_cache` and `--receptor-path` → `unified_cache_expanded`

**Optional:** `--receptor-path path/to/hansen.csv` so the expanded set includes receptor atlas maps.

**Result:** You get `decoder_cache`, `neurosynth_cache`, `unified_cache`, `neuromaps_cache`, and `unified_cache_expanded` (if `--expand`). Then run Step 6 manually to train the text-to-brain model on `unified_cache_expanded`.

---

## 12. File Formats Reference

| Path | Format | Description |
|------|--------|-------------|
| `*/term_maps.npz` | `data["term_maps"]` → (N, get_n_parcels()) float64 | One parcel-D map per term (default **427**) |
| `*/term_vocab.pkl` | list of N strings | Term labels; index i = row i of term_maps |
| `neuromaps_cache/annotation_maps.npz` | `data["matrix"]` → (M, get_n_parcels()) | One parcel-D map per annotation (default **427**) |
| `neuromaps_cache/annotation_labels.pkl` | list of M strings | Annotation labels |
| `*_expanded/term_sources.pkl` | list of N strings: `direct` \| `ontology` \| `neuromaps` \| `receptor` | Source of each term (for sample weighting) |
| `*_expanded/term_map_types.pkl` | list of N strings: `fmri_activation` \| `structural` \| `pet_receptor` | Map type per term (for type-conditioned MLP; derived from term_sources when present) |
| `embedding_model/` | Various (encoder config, MLP weights, split_info.pkl) | Trained text→brain model |

**Cache metadata (recommended):** To prevent silent dimension mismatches, each cache directory should include a `metadata.json` with: `n_parcels` (must equal `get_n_parcels()`), `atlas_path` and/or `atlas_id`, `created_utc`, `git_commit`, `script_name`, `script_args`, and `source` (neuroquery, neurosynth, neurovault, neuromaps, enigma, abagen, ontology, etc.). Then `verify_parcellation_and_map_types.py` can validate caches without inferring intent from filenames.

---

## 12b. Cortical vs subcortical: current setup

**Current setup:** The pipeline uses **Glasser 360 + Tian S2 + Brainstem + BFB/Hyp** = **427 parcels**. The combined atlas (`combined_atlas_427.nii.gz`) provides cortical, subcortical, brainstem, and basal forebrain/hypothalamus coverage. All caches and models use `get_n_parcels()` (427 when that atlas exists) via `neurolab/parcellation.py:get_masker()`.

(Legacy Schaefer-only discussion moved to a separate legacy atlases note.)

**Do you need to refetch all the data?** No. The **MNI (or native) volumetric maps or raw data are stored locally** where applicable; you only need to **re-run the cache builders** with the new atlas (re-parcellate). You do **not** need to re-download NeuroVault images, neuromaps annotations, NeuroSynth database, AHBA (abagen), or the NeuroQuery model. Summary:

| Source | What's stored locally | When adding subcortical |
|--------|----------------------|--------------------------|
| **NeuroVault** | NIfTI images in `downloads/neurovault/` | **No refetch.** Re-run `build_neurovault_cache.py` with the new (combined) atlas; it will load the existing NIfTIs and re-parcellate. |
| **Neuromaps** | Annotations in `neuromaps_data/` (from `download_neuromaps_data.py`) | **No refetch.** Re-run `build_neuromaps_cache.py` with the new atlas; it will use the existing annotation NIfTIs and re-parcellate. |
| **NeuroSynth** | Database (coordinates, metadata, features) in `neurosynth_data/` | **No refetch.** Re-run `build_neurosynth_cache.py` with the new atlas; it will re-run meta-analysis (MKDA) and parcellate with the new atlas. Volumetric maps are not cached, so this step re-computes them. |
| **NeuroQuery** | Only the NeuroQuery model (fetched once); **no volumetric maps** are saved | **No refetch** of the model. Re-run `build_term_maps_cache.py` (or `build_neuroquery_cache.py`) with the new atlas; it will re-call the model per term and parcellate. Slow (1–2 h) but no new downloads. |
| **ENIGMA** | Nothing volumetric; summary stats come from enigmatoolbox | Re-run `build_enigma_cache.py` with a new DK+subcortical → combined-atlas mapping. |
| **abagen** | AHBA microarray data cached by abagen (e.g. in abagen data_dir) | **No refetch** if already cached. Re-run `build_abagen_cache.py` with the new atlas; abagen will re-assign samples to parcels. |
| **Receptor (Hansen)** | Usually a CSV/matrix parcellated to pipeline | Receptor matrix must match `get_n_parcels()`; may require re-deriving from volumetric receptor maps if available. |

So: **no refetch** of the underlying data. Re-run the **build** scripts (decoder, neurosynth, neurovault, neuromaps, enigma, abagen) with the new combined atlas and N_PARCELS; then re-run ontology expansion and training.

### 12.1 Parcellation and map-type audit

To verify that **all** map caches use the pipeline atlas (`get_n_parcels()` = 427 by default) and to see which **map types** you have (fMRI, structural, PET), run:

```bash
python neurolab/scripts/verify_parcellation_and_map_types.py
```

Optional: `--strict` exits with code 1 if any cache does not match `get_n_parcels()`.

**What it reports:**

- **Parcellation:** For each cache under `neurolab/data/`, it loads `term_maps.npz` (or `annotation_maps.npz` for neuromaps) and checks that the second dimension equals `get_n_parcels()`. Caches built with a different atlas will need a rebuild.
- **Map types:** Each cache is classified as **fmri** (task/activation), **structural** (e.g. ENIGMA thickness/volume), or **pet** (receptor, gene expression, neuromaps annotations). The summary counts how many caches you have per type.

Use this script after any atlas change to ensure all maps are parcellated correctly and to confirm fMRI, structural, and PET coverage where intended.

### 12.2 One-command rebuild

To rebuild **all** caches to the pipeline atlas (427 parcels by default) and include fMRI, structural (ENIGMA), and PET (neuromaps, abagen) where available:

```bash
python neurolab/scripts/rebuild_all_caches.py
```

- **Quick test:** `--quick` caps decoder (500 terms), neurosynth (100), neuromaps (20) so the run finishes in under an hour.
- **Skip steps:** e.g. `--skip-decoder --skip-neurosynth` to only rebuild neuromaps/neurovault/enigma/abagen and expand.
- **Optional structural:** ENIGMA cache requires `enigmatoolbox` (may need install from source on some platforms). Use `--skip-enigma` if unavailable.
- **Optional PET (gene):** abagen cache requires `abagen`; some pandas versions break abagen (e.g. 2.2+). Use `--skip-abagen` if it fails.

After the script finishes, run `verify_parcellation_and_map_types.py` to confirm parcellation and map-type coverage.

---

## 13. Ontology Expansion: Relation Weights and Direct Maps

- **Direct map wins:** If a term (e.g. “executive function”) already exists in the cognitive cache, we **always use that map**. Expansion only adds new terms; it never overwrites.
- **Relation type → weight:** When blending related cache maps for an ontology term, we use `RELATION_WEIGHTS` in `ontology_expansion.py`: self 1.0, synonym 0.95, child 0.85, parent 0.8. So “executive function” (if added via ontology) gets a **weighted average** of related cache terms (e.g. working memory, attention), not a copy of one.
- **Validation:** Run `ontology_brain_correlation.py --cache-dir ... --ontology-dir ...`. It reports mean brain-map correlation **by direction**: `parent_of` (label is parent of related), `child_of` (label is child of related). Use `--output-weights` to export weights; the JSON includes directional keys and mapped `parent`/`child` for expansion.
- **Data-driven weights:** Run `ontology_brain_correlation.py --output-weights path/to/weights.json`, then `build_expanded_term_maps.py --relation-weights-file path/to/weights.json` to use observed mean r per relation type as blend weights. **Caveat:** When you run correlation on NQ/NS cache, you measure correlation of *model outputs*, not real brain activations; NQ uses semantic smoothing so related terms get correlated maps by construction. Measured r values are upper bounds; if you later use real meta-analytic maps (e.g. NeuroSynth MKDA), optimal weights will likely be lower.
- **Directional scaling:** Parent's map used to approximate a child is too broad → downweight (e.g. 0.7); child's map used to approximate a parent is a subset → slight downweight (0.9). See `DIRECTION_SCALE` in `ontology_expansion.py`. The ontology index now stores `(term, weight, relation_type)` so expansion and validation can use direction.
- **Graph-theoretic (implemented):** Use `--use-graph-distance` in `build_expanded_term_maps.py` to set weight = γ^path_length (shortest path in each OBO graph; min path across ontologies that contain both terms). OBO only; OWL ontologies are skipped for path length. `--graph-distance-gamma` (default 0.8) controls decay. When merging multiple ontologies there are no cross-ontology edges, so path is computed per-ontology and the minimum path length is used.

---

## 14. Risks and Recommendations

- **Ontology expansion risk:** Derived maps (weighted averages of related cache maps) can be blurry centroids that do not match true meta-analytic patterns (e.g. sharp ACC/dlPFC for “executive function”). **Recommendation:** Get the pipeline working end-to-end **without** expansion first; validate base model quality (train/test R², qualitative glass-brain checks). Then add expansion and compare metrics; if test R² drops, derived maps may be hurting more than helping.
- **Quality filters:** Use `--min-cache-matches 2` (or 3) and optionally `--min-pairwise-correlation 0.3` so that ontology terms backed by a single or very dissimilar set of maps are not added.
- **Merge strategy (NQ vs NeuroSynth):** NeuroSynth’s MKDA maps are coordinate-count based; NeuroQuery uses full-abstract text regression. For single-word/high-N terms (e.g. “amygdala”, “motor”), NeuroSynth is often better; for multi-word/compositional terms, NeuroQuery’s semantic smoothing is better. The merge default is **--prefer neuroquery** (smoother maps, better for MSE training). A future improvement is to prefer NeuroSynth for single-word anatomical terms and NeuroQuery for multi-word.
- **Neuromaps/receptor in training:** Short labels (e.g. "5HT2A") give the encoder little to work with; use them only at inference or augment with descriptive text.
- **Training:** Saves `training_history.pkl` and `split_info["test_term_correlations"]` for diagnostics. `ontology_brain_correlation.py` now reports per-relation-type correlation to tune RELATION_WEIGHTS.
- **Data augmentation:** Each term appears once; augmenting with case/suffix/abbreviation variants (e.g. "working memory", "Working Memory", "working memory task", "WM") mapping to the same map would increase effective dataset size and is recommended for the sentence-transformer path.
- **Future data sources:** (1) **NeuroVault** (e.g. via nimare): task-contrast maps with free-text descriptions (e.g. “fearful faces > neutral”); curated collections (e.g. IBC, NARPS) give real contrast maps + natural language. (2) **Cognitive Atlas task→concept:** e.g. “Stroop task → cognitive control, selective attention, inhibition” can generate extra (text, map) pairs by pairing Stroop’s map with multiple CogAt labels.
- **Held-out evaluation:** Beyond train/val/test metrics, run the model on 20 well-known queries (e.g. “primary visual cortex”, “language comprehension”, “reward”, “fear”) and inspect glass-brain outputs; optionally correlate predictions with NeuroSynth association-test maps for the same terms as external validation.

---

## 14b. KG-regularized contrastive loss (optional architecture)

**Current approach (ontology expansion):** We add ontology terms by assigning each a **derived map** = weighted average of related cache maps, then train the model with MSE(predicted_map, target_map). That treats the average as **ground truth** — a fabricated target the model must reproduce. For terms where KG proximity and brain-map similarity align (e.g. "working memory" → "executive function", dlPFC/ACC overlap), this works; where they don't (e.g. "amygdala" → "fear" — one strong KG edge but fear is a distributed circuit), the hard average can be misleading.

**Why KG distance ≠ embedding distance:** NLP embeddings (e.g. PubMedBERT) encode **distributional semantics** — similar context → similar vectors. "Dopamine" and "serotonin" are close in embedding space because papers discuss them similarly, but their brain maps differ (striatal vs broadly cortical). The **knowledge graph** encodes **relational/hierarchical** structure: sibling/parent-child, expressed_in, modulates. That topology can capture information distributional semantics misses; it does **not** always predict brain-map similarity, so it should be used as a **soft** constraint, not as the sole source of targets.

**KG-regularized auxiliary loss (recommended direction):** Use the KG **alongside** MSE, not as a replacement:

- **Main loss:** MSE(predicted_map, target_map) on terms that have direct or high-quality targets (NQ/NS, optionally ontology-derived with high pairwise correlation).
- **Aux loss (contrastive):** If term A and term B are **close in the KG** (e.g. synonym, parent-child), penalize the model when their **predicted brain maps** are very different. If A and B are **far in the KG**, do not constrain them. So: KG distance should correlate with allowed brain-map distance — neighbors in the KG should have similar predicted maps; distant terms can differ.

Concretely, this can be implemented as a **triplet/contrastive** setup: for each training term, sample a **KG neighbor** (positive) and a **KG-distant** term (negative); add a margin loss that pulls predicted maps of neighbors closer and pushes predicted maps of distant terms apart. **RELATION_WEIGHTS** (synonym 0.95, child 0.85, parent 0.8) can define the **margin targets** (e.g. desired similarity for positive pairs).

**Why this is stronger than expansion-only:** Expansion creates a single average map per ontology term and forces the model to match it. A graph-regularized loss instead says *these terms should be similar in brain space, but I don't know the exact map* — a **soft constraint** that lets the model learn the real mapping while respecting relational structure, without fabricating targets for every ontology term.

**Caveat — validate first:** This only helps if **KG distance and brain-map similarity are actually correlated**. That is what `ontology_brain_correlation.py` is for: compare mean brain-map correlation for ontology-related pairs vs random pairs. If the correlation is **weak (r < 0.2)**, the KG is likely adding noise; if **moderate (r > 0.3)**, using it as a regularizer is justified. Run that script before investing in a KG-regularized training path.

---

## 14b2. Multiple map types (fMRI, structural, PET): re-route vs type-conditioned weights

When you load different map types (fMRI activation, structural thickness/volume, PET receptor, etc.) into the same cache, the model must answer: *if the user asks for "activity" give an fMRI-style map; if they ask for "structure" or "thickness" give a structural map; if they ask for "receptor" or "binding" give a PET-style map.*

**Two main strategies:**

1. **Re-routing** — Detect map type from the query (keywords or a small classifier) and route to type-specific models or caches.  
   - **Pros:** Clear separation; each model is trained only on one map type.  
   - **Cons:** Need separate models or caches per type; boundary cases ("dlPFC" without "activity"/"structure") require a default or blend.

2. **Type-conditioned single model (weights give importance)** — One encoder + one MLP, but the **map type** is an extra input so the same weights produce different map styles.  
   - **Training:** Each sample has a `map_type` (e.g. `fmri_activation` | `structural` | `pet_receptor`). Input to the MLP = `concat(embedding, type_one_hot)` (or type embedding). The MLP learns: "when type=fmri, produce activation-like map; when type=structural, produce thickness-like map."  
   - **Inference:** Infer map type from the query (e.g. "activity", "activation", "task", "BOLD" → fmri; "thickness", "volume", "atrophy", "structure" → structural; "receptor", "binding", "PET", "dopamine" → pet). Feed that type (one-hot or embedding) together with the query embedding into the MLP. The **weights** of the model then give a lot of importance to the right type because the first layer effectively gates by type.  
   - **Pros:** Single model; no hard routing; type is learned end-to-end.  
   - **Cons:** Need to tag training data with `map_type` and implement type inference at query time.

**Practical recommendation:** Use **type-conditioned single head**: add `term_map_types.pkl` (or derive from `term_sources`: direct/neurovault/ontology → fmri; enigma → structural; neuromaps/receptor → pet). MLP input dim = encoder_dim + num_types. At inference, run a lightweight type classifier (keyword rules or a small classifier on the query) to get a type (or soft type distribution), then feed `concat(query_embedding, type_one_hot)` into the MLP. So "activity" gives high weight to the fmri type dimension and the model outputs an activation-like map; "structure" gives high weight to structural and the model outputs a thickness-like map — **the weights give a lot of importance to the type** via the extra input dimensions. No need to re-route to different models; one forward pass with the right type input.

**Optional: sample weights by type** — When training, you can upweight samples whose type matches the dominant type in the batch (or use type-balanced sampling) so the model sees enough of each type. That way the "weights" (sample weights) also give more importance to the type you care about when the data is imbalanced.

---

## 14c. Using the triad (embedding + KG + interaction) to improve MLP training

**Finding (from `regress_brain_r_on_hop_and_embedding.py`):** Brain-map correlation between term pairs is explained by **both** hierarchy distance (hop) and embedding distance (1 − cos_sim), **plus their interaction**. So: word embedding, KG position, and how they interact together predict brain-map similarity. We can use this **triad** to improve the term→brain MLP training.

**Ways to use it in training:**

1. **Pairwise auxiliary loss (regression-predicted target r)**  
   - **Idea:** For a sample of pairs (i, j) of training terms, we have hop(i,j) and emb_dist(i,j). The regression gives a **target brain correlation**:  
     `r_target = intercept + coef_hop*hop + coef_emb*emb_dist + coef_int*hop*emb_dist`  
     (load coefficients from `brain_r_hop_embedding_regression.json`).  
   - **Predicted:** From the model, get predicted maps for i and j; compute `pred_r_ij = Pearson(pred_map_i, pred_map_j)`.  
   - **Aux loss:** `L_pair = mean over pairs of (pred_r_ij - r_target)^2`.  
   - **Total loss:** `(1 - λ)*MSE(pred_map, target_map) + λ*L_pair` with small λ (e.g. 0.1–0.2).  
   - **Effect:** The model is encouraged to match the observed structure: close (hop + embedding) → higher predicted r; far (especially both) → lower predicted r.

2. **Contrastive / margin loss (KG + embedding)**  
   - **Idea:** Same as §14b (neighbors in KG → similar maps) but **also** use embedding distance.  
   - Sample **positive** pairs: low hop and low emb_dist (e.g. hop ≤ 2 and emb_dist below median).  
   - Sample **negative** pairs: high hop or high emb_dist.  
   - Loss: pull `pred_r` high for positives, push `pred_r` low for negatives (e.g. margin loss or target r from regression).  
   - **Effect:** Directly encodes “close in both KG and embedding → similar maps; far in either → can differ.”

3. **Sample weighting by triad consistency**  
   - For each training term, compare its predicted map to predicted maps of its KG neighbors. If the model predicts very different maps for terms that are close in KG+embedding, upweight that term’s MSE so the model pays more attention to fixing that inconsistency.  
   - **Effect:** Softer than (1)–(2); focuses gradient on terms that violate the triad.

4. **Implementation requirements**  
   - **Ontology:** Load graph and compute hop for training-term pairs (same as `regress_brain_r_on_hop_and_embedding.py`); use `--ontology-dir` and training term list.  
   - **Embeddings:** We already have `X_train` (text embeddings) for training terms; `emb_dist(i,j) = 1 - cos_sim(X_train[i], X_train[j])`.  
   - **Regression coefficients:** Load from `--kg-regression-json` (output of `regress_brain_r_on_hop_and_embedding.py`).  
   - **Pair sampling:** In each batch, either (a) sample pairs within the batch and compute hop from a precomputed (train × train) hop matrix, or (b) add a separate pairwise pass that samples pairs across the training set and adds L_pair.  
   - **Recommendation:** Start with (1) and λ small (e.g. 0.1); monitor train/val/test correlation. If val improves without hurting test, the triad regularizer is helping.

**When to use:** After you have run `regress_brain_r_on_hop_and_embedding.py` and confirmed that hop + emb_dist + interaction explain meaningful variance (R² ~ 0.04 at pair level is modest but the interaction is strong). Use ontology terms that appear in the graph (same 827-term set as the regression) so hop is defined.

**Composite distance/similarity (single scalar combining hop + embedding):** The regression-predicted brain r is your **composite similarity**: higher = more similar (higher expected brain-map correlation). So:
- **Similarity:** `r_predicted = intercept + coef_hop*hop + coef_emb*emb_dist + coef_int*hop*emb_dist` (use as target in pairwise loss; see above).
- **Distance:** `composite_distance = 1 - r_predicted` (higher = more distant; use for ranking or sampling).

Reusable helpers: `neurolab/scripts/composite_distance_utils.py` exposes `load_regression_coefficients()`, `predicted_r(hop, emb_dist, coef)`, and `composite_distance()`. To build the full (n×n) composite similarity matrix and optionally plot brain r vs composite: run `compute_composite_matrix.py --kg-regression-json ... --ontology-dir ... --encoder sentence-transformers --out-dir neurolab/data --plot`. That saves `composite_similarity.npy` and `terms_composite.pkl` and validates with Pearson(brain_r, composite_similarity). The triad pairwise loss in training (Section 14c) uses this same composite similarity as the target for each pair.

---

## 15. Summary Checklist for Another Agent

1. **Ontologies:** Run `download_ontologies.py --output-dir neurolab/data/ontologies` (and optionally `--extra` or `--cognitive-atlas --bfo --ro`).
2. **Neuromaps data (optional):** Run `download_neuromaps_data.py` so raw annotations live in `neurolab/data/neuromaps_data/`.
3. **Cognitive caches:** Run `build_term_maps_cache.py` (and optionally `build_neurosynth_cache.py` + `merge_neuroquery_neurosynth_cache.py`) or use `build_all_maps.py` without `--expand` to get `decoder_cache` and `unified_cache`.
4. **Neuromaps cache:** Run `build_neuromaps_cache.py` (uses `neuromaps_data/` by default).
5. **Expanded training set:** Run `build_expanded_term_maps.py` with `--cache-dir unified_cache`, `--ontology-dir neurolab/data/ontologies`, `--output-dir neurolab/data/unified_cache_expanded`, and optionally `--neuromaps-cache-dir`, `--receptor-path`, `--min-cache-matches 2`, `--min-pairwise-correlation 0.3`, `--save-term-sources`.
6. **Train model:** Run `train_text_to_brain_embedding.py` with `--cache-dir neurolab/data/unified_cache_expanded` and desired encoder/regularization options (e.g. `--head-hidden2 256` for a deeper MLP; training will use `term_sources.pkl` for sample weights if present).

Alternatively, run `build_all_maps.py --expand` (with or without `--quick`) to perform steps 1–5 in one go, then run step 6.

All paths above are relative to the **querytobrain repository root**. Scripts live under `neurolab/scripts/`; data lives under `neurolab/data/` by default.
