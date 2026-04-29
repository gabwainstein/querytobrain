# KG-to-Brain GNN — Validation Report (2026-04-29)

This report validates the KG-to-brain GNN against two field-standard
benchmarks: agreement with the NeuroQuery decoder on common cognitive
terms (within-domain), and statistical significance against
spatial-autocorrelation-preserving nulls on held-out PET receptor
density maps (cross-modality, generalization).

Model evaluated: `kg_brain_gnn_model` (v3, GraphConv, 100 epochs, pruned
graph; commit `3aec149`).

---

## 1. Agreement with NeuroQuery decoder (19 cognitive terms)

**Question:** does the GNN produce maps that match the field-standard
NeuroQuery decoder for well-known cognitive concepts?

**Method:** for 19 cognitive terms (working memory, fear, amygdala,
reward, etc.) present in both the GNN supervision vocab and the
NeuroQuery decoder cache (`neurolab/data/decoder_cache/`), compute:
- Pearson correlation between GNN predicted parcel vector and
  NeuroQuery decoder's parcel vector.
- Jaccard overlap of the top-10 most-activated parcels.

**Result (mean across 19 terms):**

| Metric | Value |
|---|---:|
| Pearson r | **0.934** |
| Jaccard top-10 | 0.640 |

**Per-term breakdown** (excerpt):

| Term | Pearson r | Jaccard@10 |
|---|---:|---:|
| amygdala | 0.962 | 1.000 |
| fear | 0.970 | 1.000 |
| visual perception | 0.971 | 0.429 |
| motor | 0.968 | 0.538 |
| pain | 0.953 | 0.818 |
| reward | 0.955 | 0.818 |
| executive function | 0.913 | 0.429 |
| working memory | 0.883 | 0.538 |
| default mode | 0.854 | 0.667 |

**Caveat:** these terms are present in the GNN's training set (the
NeuroQuery decoder cache is one of the input sources to merged_sources).
This validates that the GNN faithfully reproduces field-standard maps on
its supervision distribution — not that it generalizes to unseen terms.
For the latter, see Section 2.

---

## 2. Spin test on held-out PET maps (cross-modality generalization)

**Question:** are the GNN's predictions on data the model never saw
during training statistically significant beyond what spatial
autocorrelation alone would produce?

**Method:** the model is trained with the entire `neuromaps` source
(31 PET receptor density maps: 5-HT1A, D1/D2, GABA-A, NMDA, VAChT, etc.)
held out as the test bucket. Training data is 99% fMRI activation maps,
so this is a true **cross-modality** generalization test.

For each of the 31 held-out PET maps:
1. Compute actual Pearson r between GNN prediction and ground truth.
2. Generate 1000 spatial-autocorrelation-preserving surrogate maps of
   the ground truth using **brainsmash** (variogram-matching
   methodology, Burt et al. 2020). Distance matrix = pairwise Euclidean
   between MNI parcel centroids.
3. Compute correlation of the GNN prediction against each surrogate.
4. p_spin = fraction of surrogate correlations whose absolute value
   meets or exceeds the actual r.

**Result (n=31):**

| Metric | Value |
|---|---:|
| Mean Pearson r | **0.616** |
| Median Pearson r | 0.623 |
| Null distribution mean (across surrogates) | ≈ 0.0 |
| Null distribution std | 0.13–0.20 per term |
| p_spin < 0.05 | **31/31 (100%)** |
| p_spin < 0.01 | 29/31 (94%) |
| p_spin < 0.001 | 27/31 (87%) |

Per-term examples (excerpt):

| Term | Pearson r | p_spin |
|---|---:|---:|
| PET: CUMI-101 binding to 5-HT1A | 0.714 | < 0.001 |
| PET: FEOBV binding to VAChT | 0.709 | < 0.001 |
| PET: DASB binding to 5-HTT | 0.667 | < 0.001 |
| PET: O-methylreboxetine binding to NET | 0.656 | < 0.001 |
| PET: LY-2795050 binding to D4 | 0.654 | < 0.001 |
| PET: Raclopride binding to D2 | 0.518 | < 0.001 |
| PET: AZ10419369 binding to 5-HT1B | 0.570 | < 0.001 |
| PET: SB207145 binding to 5-HT4 | 0.583 | < 0.001 |

**Interpretation:** at the parcel level, the GNN's PET density predictions
are 4–5σ above the spatial null for typical terms. Holding out an entire
data source (cross-modality) the model never saw and still recovering
significant spatial alignment is the central evidence that the GNN is
learning real graph relationships — not memorizing — and that those
relationships transfer across map types via the gene-expression and
ontology edges.

---

## 3. What this validates and what it does not

**Validates:**
- GNN matches NeuroQuery decoder on canonical cognitive terms (r ≈ 0.93)
- GNN's parcellated predictions for held-out PET density maps are
  statistically significant against spatial-autocorrelation nulls
  (100% p<0.05, 87% p<0.001) at n=31
- Anatomy reasoning checks (separately documented in
  `KG_TO_BRAIN_GNN.md`): subcortical-targeted queries correctly hit
  basal ganglia / amygdala / NAcc; cortical queries stay cortical

**Does not yet validate:**
- Comparison against expert-curated brain atlases (e.g. Hansen et al.
  2023 on the same PET tracers used in train) — would require atlas-
  aligned versions of those maps and is a follow-up.
- Inter-rater agreement on novel-query anatomy — would require
  multiple domain experts blind-rating predictions for novel queries.
- Performance against existing tools (NeuroSynth meta-analytic CBMA,
  NiMARE) on a fixed third-party benchmark — would require a curated
  external test list with published gold-standard maps.
- Surface-level (vertex-space) accuracy — current model operates at
  392-parcel resolution, which masks sub-parcel structure.

**Methodological notes:**
- Spin test uses brainsmash variogram-matching surrogates; this is the
  recommended parcel-space null for testing spatial map correlations
  (preserves the empirical spatial autocorrelation of the ground-truth
  map while randomizing alignment with the test map).
- Distance matrix is Euclidean between MNI parcel centroids (Glasser+
  Tian S2, 392 parcels). A surface-geodesic distance matrix would be
  more rigorous for cortical parcels but requires parcel→vertex
  projection that we have not implemented.
- 1000 permutations gives p resolution down to 0.001; tighter
  significance bands would require more permutations.

---

## 4. Reproducibility

```bash
# Validation 1 (NeuroQuery decoder agreement)
python -c "see scripts/validation snippet in commit message"

# Validation 2 (brainsmash variogram null)
pip install brainsmash
# ~50 min for 31 terms × 1000 perms on CPU
```
