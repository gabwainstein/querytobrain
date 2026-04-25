# Ontology distance to brain map similarity: implementation note

This note links the **graph_distance_correlation** script and the compass research report to a three-tier implementation blueprint (IC-based, graph embeddings, hybrid) and validation with spatial null models.

## Reference report

**Predicting brain map similarity from ontological distance: a methods guide**  
`compass_artifact_wf-86e07ff5-ba6f-46d5-aef5-d0772b6c69f1_text_markdown.md` (same directory)

Summary from the report:

- **Why hop count collapses at d≥2:** Edges near the ontology root span huge semantic gaps; hierarchy distance (parent_child hops) and synonym-as-own-category improve but do not fix root-level uniformity.
- **Information content (IC) fixes it:** Lin similarity with **Sánchez intrinsic IC** weights each edge by the information it carries; recommended over raw hop count and over corpus-based IC for neuroscience ontologies.
- **Cross-ontology:** IC requires a shared hierarchy. For Cognitive Atlas ↔ UBERON ↔ GO, **graph embeddings** (OWL2Vec*, mOWL) are the principled option.
- **Brain map side:** Pearson is appropriate; **spatial null models** (BrainSMASH, neuromaps) are essential for valid inference; **Mantel test** (ontology distance matrix vs brain correlation matrix, with spatial permutations) for overall relationship.

## Current implementation: `graph_distance_correlation.py`

**Location:** `neurolab/scripts/graph_distance_correlation.py`

**What it does:**

1. **Hierarchy distance**  
   Distance = number of **parent_child** (is_a) edges along the path. Synonym-only paths → d_hierarchy = 0 (own category). Reports mean brain-map r, percentiles, and path type (synonym / all_parent_child / mixed) per d_hierarchy.

2. **Optional Lin + Sánchez IC** (`--use-ic`)  
   Builds an NXOntology DAG from the same unified graph (parent_child only), then computes **Lin similarity** with `ic_metric="intrinsic_ic_sanchez"` (via **nxontology**). Reports Pearson and Spearman correlation between brain-map r and Lin similarity for all pairs with a shared hierarchy.

**Usage:**

```bash
# Hierarchy distance only
python neurolab/scripts/graph_distance_correlation.py \
  --cache-dir neurolab/data/decoder_cache \
  --ontology-dir neurolab/data/ontologies \
  --plot neurolab/data/distance_decay.png

# Add Lin (Sanchez IC) vs r
python neurolab/scripts/graph_distance_correlation.py \
  --cache-dir neurolab/data/decoder_cache \
  --ontology-dir neurolab/data/ontologies \
  --use-ic

# Add Mantel test (ontology vs brain similarity, permutation null)
python neurolab/scripts/graph_distance_correlation.py \
  --cache-dir neurolab/data/decoder_cache \
  --ontology-dir neurolab/data/ontologies \
  --mantel --mantel-perms 1000
```

**Dependencies for IC:** `pip install nxontology`

## Three-tier blueprint (from compass report)

| Tier | Approach | Tools | When to use |
|------|-----------|--------|-------------|
| **1** | Lin + Sánchez intrinsic IC | **nxontology** (`pip install nxontology`) | Single-ontology or merged DAG; interpretable; implemented via `--use-ic`. |
| **2** | Graph embeddings | **OWL2Vec*** via **mowl-borg** (`pip install mowl-borg`) | Cross-ontology terms (no shared ancestor); 200-d embeddings, cosine similarity. |
| **3** | Hybrid | Linear combo of Lin + embedding similarity, or IC as edge weights in OWL2Vec* | If Tier 1 and 2 capture different variance. |

**Brain map comparison and validation:**

- **Similarity:** Full Pearson (and optionally Spearman) correlation matrix between term maps.
- **Testing ontology–brain relationship:** **Mantel test** with **BrainSMASH** (or neuromaps) spatial surrogates so null preserves spatial autocorrelation. Report Mantel r and spatial-null p-value.

## Validation: Mantel test and spatial nulls

**Implemented:** `--mantel` runs a **Mantel test** between ontology similarity (1/(1+d_hierarchy)) and brain-map r across the same pairs. Null: **permutation** (shuffle ontology similarity across pairs). Reports Mantel r and one-tailed p. This tests whether ontology similarity predicts brain similarity beyond chance.

- **Problem:** Parcellated brain maps have spatial autocorrelation; **permutation null** is conservative for the ontology–brain link (we permute ontology, not brain). For stricter control, use a **spatial null** (see below).
- **Spatial null (recommended next):** Generate surrogate brain maps (BrainSMASH or neuromaps) that preserve spatial autocorrelation; compute brain correlation matrix from surrogates; recompute Mantel(ontology, brain_surrogate) to get null distribution. Then p = proportion of null Mantel ≥ observed.
- **Frameworks:**  
  - **BrainSMASH:** variogram-matched surrogates; needs 400×400 geodesic distance matrix for Schaefer.  
  - **neuromaps:** `neuromaps.nulls` with `parcellation` for parcellated data; `compare_images(..., nulls=...)` for pairwise p-values.

**Python stack (from report):**  
`neuromaps`, `brainsmash`, `nxontology`, `mowl-borg` (optional Tier 2).

## Summary

- **Script:** `graph_distance_correlation.py` implements hierarchy distance and optional **Lin + Sánchez IC** (`--use-ic`) for ontology–brain r analysis.
- **Compass report:** Justifies IC over hop count, cross-ontology embeddings, and spatial nulls; no prior work explicitly correlates Resnik/Lin/Jiang–Conrath with brain map similarity.
- **Next steps:** (1) Run with `--use-ic` and compare predictive strength of Lin vs hierarchy distance; (2) add Mantel test with BrainSMASH/neuromaps for inference; (3) add Tier 2 (OWL2Vec*) for cross-ontology terms if needed.
