# NeuroLab Testing Runbook

Run all commands from **querytobrain** repo root. Python 3.9+, venv recommended.

---

## Whole connected system (default)

The pipeline is **one connected system**: **NeuroQuery + NeuroSynth (cognitive, merged) + receptor + neuromaps (biological)**. You build it once; all scripts use it by default.

1. **build_all_maps.py** builds: NeuroQuery → `decoder_cache/`, NeuroSynth → `neurosynth_cache/`, **merge → `unified_cache/`** (one cognitive cache), and neuromaps → `neuromaps_cache/`.
2. **query.py**, **verify_decoder.py**, **verify_unified.py**, **test_enrichment_e2e.py** default to: cognitive = **unified_cache** (if present), else decoder_cache; biological = **neuromaps_cache** (if present) + receptor (placeholder or Hansen).
3. So one build gives you the full hybrid; no need to pass `--cache-dir` or `--neuromaps-cache-dir` unless you override.

---

## How the pipeline works (in one go)

1. **Cognitive:** One decoder cache of term → (400,) — either **unified_cache** (NQ+NS merged) or **decoder_cache** (NQ-only). CognitiveDecoder loads it and returns top terms for a (400,) map.
2. **Biological:** ReceptorEnrichment (Hansen or placeholder) + NeuromapsEnrichment (`neuromaps_cache/`). Same (400,) map is correlated with receptor and neuromaps annotations.
3. **UnifiedEnrichment** runs cognitive decode + biological and returns a **summary**.
4. **Expandable term space** (optional): train text → (400,) on whichever cognitive cache you use; then any phrase can be decoded/enriched.

Everything stays in **parcellated 400-D**.

---

## Test order and dependencies

| Order | Script | Needs | What it checks |
|-------|--------|-------|----------------|
| 1 | `verify_environment.py` | pip install -r neurolab/requirements-enrichment.txt | Phase 0+1: imports, NeuroQuery, Schaefer 400 |
| 2 | **`build_all_maps.py --quick`** | Phase 1 + nimare | **Build whole system:** NQ + NS + merge → unified_cache + neuromaps_cache |
| 3 | `verify_decoder.py` | Step 2 | Uses unified_cache (or decoder_cache); Phase 3 decode tests |
| 4 | `verify_unified.py` | Step 2 | Uses unified_cache + neuromaps_cache; Phase 4+5 receptor + unified |
| 5 | `test_enrichment_e2e.py` | Step 2 | E2E: text → map → enrich (cognitive + biological + summary) |
| 6 | `verify_embedding.py` | Step 2 | Small embedding model; optional guardrail |
| 7 | `query.py "attention"` | Step 2 | Full query; defaults to unified_cache + neuromaps_cache |

**Optional (accuracy / PCA):** `train_text_to_brain_embedding.py` (e.g. `--cache-dir neurolab/data/unified_cache` for full vocab), then `compare_pca_components.py`. See [accuracy-and-testing.md](accuracy-and-testing.md).

---

## Commands (copy-paste from repo root)

**One-time setup**
```bash
pip install -r neurolab/requirements-enrichment.txt
```

**1. Environment (no cache needed)**
```bash
python neurolab/scripts/verify_environment.py
```
Expect: "All checks passed. Ready for Phase 2."

**2. Neuromaps data (repo-local, optional one-time fetch)**  
Raw neuromaps annotations live in **`neurolab/data/neuromaps_data/`**. If you already have that data in the repo, `build_neuromaps_cache` uses it with no download. If not, either run once:
```bash
python neurolab/scripts/download_neuromaps_data.py
```
(or `neuromaps_fetch.py fetch-all --output-dir neurolab/data/neuromaps_data --space MNI152`). Then build uses this dir by default.

**3. Build the whole connected system (one command)**  
Builds NeuroQuery, NeuroSynth, **merges them into unified_cache**, and builds neuromaps (from `neuromaps_data/`). All verify/query scripts then use these caches by default.
```bash
python neurolab/scripts/build_all_maps.py --quick
```
Expect: `decoder_cache/`, `neurosynth_cache/`, **`unified_cache/`** (use this as default cognitive cache), `neuromaps_cache/`. Takes a few minutes to ~tens of minutes depending on caps.

**Minimal fallback (NQ-only, no NS/neuromaps):**  
`python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 100` — then scripts use decoder_cache (unified_cache absent).

**4. Decoder**
```bash
python neurolab/scripts/verify_decoder.py
```
Expect: "Phase 3 passed. CognitiveDecoder is ready."

**5. Unified (receptor + cognitive)**
```bash
python neurolab/scripts/verify_unified.py
```
Expect: "Phase 4+5 passed."

**6. E2E**
```bash
python neurolab/scripts/test_enrichment_e2e.py
```
Expect: "Local E2E tests passed."

The E2E test covers: (1) NeuroQuery term → map → enrich, (2) random vector → enrich, (3) **cache + ontology** term → map (no embedding), (4) **neuromaps-by-name** (if neuromaps_cache present), (5) **combined term → map** (neuromaps first, then cognitive). Tests 4–5 are skipped if neuromaps_cache is missing.

**7. Embedding (trains small model, then verifies)**
```bash
python neurolab/scripts/verify_embedding.py
```
Expect: builds cache if missing, trains small embedding, runs prediction.

**8. Query (manual)**  
Uses unified_cache + neuromaps_cache by default when present. When not using the embedding model, the map is built from **cache-term weights** (TF-IDF over terms); if **max similarity to cache terms is below a threshold** (default 0.15), **ontology fallback** is used: related terms from ontologies (mf.owl, uberon.owl, etc.) that are in the cache are used to build the map. Ontologies are optional: put OBO/OWL files in `neurolab/data/ontologies/` (e.g. run `python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies`) or use `setup_production.py` which downloads them. Disable with `--no-ontology`.
```bash
python neurolab/scripts/query.py "attention"
python neurolab/scripts/query.py "prosopagnosia"   # low similarity → ontology fallback if ontologies present
```
Expect: printed summary, top cognitive terms, top biological hits; "Used ontology fallback" when applicable.

---

## Parameter sweep (threshold optimization)

After E2E passes, you can optimize **similarity_threshold (A)** and **similarity_threshold_ontology (B)** by comparing cache+ontology maps to NeuroQuery reference maps over a set of test terms.

```bash
python neurolab/scripts/parameter_sweep_thresholds.py
```

- **Default:** Grid A = B = [0.05, 0.10, 0.15, 0.20, 0.25]; terms = attention, memory, working memory, pain, emotion, language, vision, executive function. Reports mean Pearson r (cache+ontology map vs NeuroQuery map) per (A, B) and **best (A, B)**.
- **Custom grid:** `--grid 0.1 0.15 0.2` (and optionally `--grid-b 0.1 0.15`).
- **Custom terms:** `--terms "attention" "memory" "pain"`.
- **No ontology in sweep:** `--no-ontology`.
- **Save results:** `--output neurolab/data/results_sweep.csv`.

Use the reported best (A, B) with `query.py --similarity-threshold A --similarity-threshold-ontology B`.

---

## Quick vs full

| Goal | Quick | Full |
|------|--------|------|
| Build connected system | `build_all_maps.py --quick` (NQ 500, NS 50, neuromaps 30 + merge) | `build_all_maps.py` (no caps; many hours) |
| E2E / verify / query | Same; all use unified_cache + neuromaps by default | Same |
| Embedding training | `verify_embedding.py` or `--max-terms 500 --epochs 10` | `--cache-dir neurolab/data/unified_cache --max-terms 0 --epochs 40` + PCA |
| Accuracy numbers | Not representative | Full cache + [accuracy-and-testing.md](accuracy-and-testing.md) §3 |

---

## Ontology (optional, for low-similarity fallback)

When the query term has **low similarity** to cache terms (e.g. rare or OOV terms), the system can use **ontologies** to find **related** cache terms and build the map from those. Default threshold: max similarity &lt; 0.15 → try ontology.

- **Get ontologies (default: MF + UBERON):**  
  `python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies`  
  Or run `setup_production.py` (it downloads ontologies in step 1).
- **More ontologies:**  
  `python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies --extra`  
  adds **BFO**, **RO**, and **Cognitive Atlas** (cogat.v2.owl). Add **NIFSTD** (neuroscience) with `--nifstd`. Individual flags: `--bfo`, `--ro`, `--cognitive-atlas`, `--nifstd`.
- **query.py:** Uses ontology dir default `neurolab/data/ontologies` when present. Disable with `--no-ontology`. Adjust threshold with `--similarity-threshold 0.2`.

---

## If something fails

- **verify_environment.py fails:** Install deps (`pip install -r neurolab/requirements-enrichment.txt`). NeuroQuery first run downloads ~500MB–1GB.
- **verify_decoder.py fails:** Run step 2 first (`build_all_maps.py --quick` or at least `build_term_maps_cache.py --max-terms 10`).
- **verify_unified.py / test_enrichment_e2e.py:** Need step 2 (unified_cache or decoder_cache). Receptor uses placeholder if no Hansen file; neuromaps used automatically if neuromaps_cache exists.
- **verify_embedding.py:** Needs cache; builds a small model (may take a few minutes).

See also [enrichment-pipeline-build-plan.md](enrichment-pipeline-build-plan.md) (phases 0–6) and [accuracy-and-testing.md](accuracy-and-testing.md) (metrics and reproducibility).
