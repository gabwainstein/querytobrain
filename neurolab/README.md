# NeuroLab

**NeuroLab** is a neuroscience map analysis and text-to-brain modeling layer for retrieval, enrichment, comparison, and open-vocabulary brain-map prediction. It supports cognitive, receptor, gene, and multimodal workflows in a neutral research-oriented codebase.

## Documentation (start here)

All NeuroLab product, architecture, and process docs live under **docs/** in a documentation-first structure:

- **Docs map (navigation):** [docs/README.md](docs/README.md)
- **Product truth:** [docs/product/README.md](docs/product/README.md)
- **Architecture baseline:** [docs/architecture/architecture.md](docs/architecture/architecture.md)
- **Implementation plan:** [docs/implementation/README.md](docs/implementation/README.md)
- **Process (how we work):** [docs/process/README.md](docs/process/README.md)

## Legacy references

These remain available as historical context only and are not part of the default publish surface for this personal repo:

- Archived integration and implementation notes remain under `docs/external-specs/` if needed for historical reconstruction.

## Repo structure (neurolab)

```
neurolab/
├── README.md           # This file
└── docs/               # Documentation spine (product → architecture → implementation → process)
    ├── README.md       # Docs map
    ├── product/        # Product truth
    ├── architecture/   # Baseline + ADRs
    ├── implementation/ # Slices, runbooks, specs
    └── process/        # DoR, DoD, documentation rules, issue types
```

Code for NeuroLab lives in this repository and follows the contracts defined in `docs/architecture/interfaces.md`.

## Quick start (enrichment pipeline)

From **querytobrain** repo root:

```bash
# Python 3.9+
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r neurolab/requirements-enrichment.txt
python neurolab/scripts/verify_environment.py
```

This runs Phase 0 (imports) and Phase 1 (fetch NeuroQuery model + Schaefer 400). First run may download ~500 MB–1 GB.

**Phase 2 (build local decoder cache):**
```bash
python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 5000
```
First run: 1–2 hours. Use `--max-terms 100` for a quick test. Writes **local** files only (no API): `neurolab/data/decoder_cache/term_maps.npz`, `term_vocab.pkl`. That folder is the “decoder cache” used by all local scripts.

**Build the whole connected system (recommended):**
```bash
pip install -r neurolab/requirements-enrichment.txt   # includes nimare, neuromaps
python neurolab/scripts/build_all_maps.py --quick
```
Builds NeuroQuery → decoder_cache, NeuroSynth → neurosynth_cache, **merges them → unified_cache**, and neuromaps → neuromaps_cache. This is the **full hybrid** (NQ+NS cognitive + receptor + neuromaps biological). Use `--quick` for a test; omit for full build (many hours). **query.py** and all **verify_*** scripts default to unified_cache + neuromaps_cache when present.

**Phase 3 (verify decoder):**
```bash
python neurolab/scripts/verify_decoder.py
```
Loads `CognitiveDecoder` from cache and runs decode tests.

**Phase 4+5 (receptor + unified):**
```bash
python neurolab/scripts/verify_unified.py
```
Runs `ReceptorEnrichment` (placeholder data if no Hansen CSV/NPZ) and `UnifiedEnrichment.enrich()`. **Note:** Receptor data is **PET-based (Hansen atlas)**, not Allen gene expression. Without a real Hansen CSV, biological/receptor correlations (r) are low because placeholder data is random; use `receptor_data_scale400.csv` from [netneurolab/hansen_receptors](https://github.com/netneurolab/hansen_receptors) (results/) or neuromaps cache for meaningful receptor enrichment.

**NeuroSynth (NiMARE) — optional cognitive decoder:**
```bash
pip install nimare
python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache --max-terms 100
```
Output format matches the NeuroQuery cache (`term_maps.npz`, `term_vocab.pkl`). Use with `CognitiveDecoder(cache_dir="neurolab/data/neurosynth_cache")` or `query.py --cache-dir neurolab/data/neurosynth_cache`. First run downloads NeuroSynth; building many terms takes hours (MKDA per term).

**Neuromaps — biological annotations (includes Hansen receptor atlas):**
```bash
pip install neuromaps
python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache --max-annot 30
```
Neuromaps provides the Hansen neurotransmitter receptor atlas (and other annotations). Pass `neuromaps_cache_dir="neurolab/data/neuromaps_cache"` to `UnifiedEnrichment`; enrichment will include receptor data (via neuromaps) plus other biological maps. The standalone `ReceptorEnrichment` with a Hansen CSV/NPZ path is optional (e.g. if you have pre-parcellated files).

**Expandable term space + predict maps not in DB (local only):**
```bash
python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir neurolab/data/embedding_model
python neurolab/scripts/query.py "noradrenergic modulation" --use-embedding-model neurolab/data/embedding_model
python neurolab/scripts/predict_map.py "alpha-GPC and choline" --embedding-model neurolab/data/embedding_model --save-nifti out.nii.gz
```
The embedding model **predicts** the parcellated brain map for any text (including novel compounds not in NeuroQuery/NeuroSynth). Use `predict_map.py` to get the predicted (400,) or export to 3D NIfTI. See [expandable-term-space-embeddings.md](docs/implementation/expandable-term-space-embeddings.md). Verify: `python neurolab/scripts/verify_embedding.py`. **How we define accuracy and run tests:** [accuracy-and-testing.md](docs/implementation/accuracy-and-testing.md).

**Pipeline-oriented training wrapper:**
```bash
python neurolab/scripts/train_neurolab_model.py --pipeline fmri_cbma
python neurolab/scripts/train_neurolab_model.py --pipeline gene_receptor
python neurolab/scripts/train_neurolab_model.py --pipeline multimodal
```
Use this wrapper when you want the MLP training path selected by data regime rather than manually assembling a long flag list.

**Query (run a term locally):**
```bash
python neurolab/scripts/query.py "attention"
python neurolab/scripts/query.py "serotonin" --top-n 20
```
Uses local decoder cache (default: unified_cache if present); no API. When **similarity to cache terms is low**, **ontology fallback** is used if ontologies are present (`neurolab/data/ontologies/`): run `download_ontologies.py --output-dir neurolab/data/ontologies` or `setup_production.py`. Use `--no-ontology` to disable. See [TESTING_RUNBOOK](docs/implementation/TESTING_RUNBOOK.md).

**Local testing (no API):**
```bash
python neurolab/scripts/verify_environment.py    # Phase 0+1
python neurolab/scripts/verify_decoder.py        # Phase 3 (needs cache)
python neurolab/scripts/verify_unified.py        # Phase 4+5
python neurolab/scripts/test_enrichment_e2e.py   # E2E: text -> map -> enrich
python neurolab/scripts/verify_embedding.py      # Train small embedding + query with --use-embedding-model
```
See [docs/implementation/enrichment-pipeline-build-plan.md](docs/implementation/enrichment-pipeline-build-plan.md).

**Compare encoders (pick best for your data):**
```bash
python neurolab/scripts/compare_encoders.py --cache-dir neurolab/data/decoder_cache --max-terms 500
```
Trains with tfidf, MiniLM, and mpnet on the same subset and prints val MSE and mean correlation; use the encoder with highest correlation for your final run.

**Custom prompt → map (how many steps?):**
- **One-time:** 2 steps — (1) build decoder cache, (2) train embedding model. See [custom-prompt-to-map.md](docs/implementation/custom-prompt-to-map.md).
- **Per prompt:** 1 step — `python neurolab/scripts/predict_map.py "YOUR CUSTOM PROMPT" [--save-nifti out.nii.gz]`. So the system gives you a map from any custom prompt after the one-time setup.

**Suggested next (local only):**
1. Build full decoder cache: `build_term_maps_cache.py --max-terms 0` (or `build_all_maps.py` for NeuroQuery + NeuroSynth + neuromaps).
2. Run `compare_encoders.py` (or train with `--encoder-model all-mpnet-base-v2` / a biomedical model) and pick an encoder.
3. Train final embedding: `train_text_to_brain_embedding.py --encoder sentence-transformers --encoder-model <best> --max-terms 0 --epochs 30`.
4. Use `predict_map.py "any prompt"` or `query.py --use-embedding-model "any prompt"` for open-vocabulary map and decode.

---

## Quick context

- **Architecture:** NeuroLab can run as a standalone neuroscience module or integrate into a larger agent/tooling stack. See [docs/architecture/architecture.md](docs/architecture/architecture.md).
- **Evidence Tiers:** Every map is labeled A (IBMA/image), B (CBMA/coords), or C (predictive/hypothesis). See canonical spec §3.
- **Sprint plan:** Canonical Sprints 0–6 are in the [v0.3 spec §10](../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md#10-sprint-plan-mvp-first).
- **Related tools:** We use NeuroQuery; [Text2Brain](https://github.com/ngohgia/text2brain) and [Chat2Brain](https://arxiv.org/pdf/2309.05021) (both open-source) are alternative text→map systems. See [docs/implementation/related-text-to-brain-tools.md](docs/implementation/related-text-to-brain-tools.md) for comparison and integration options.