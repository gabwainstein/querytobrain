# Term Map Normalization Strategy

Cortex (360 Glasser parcels) and subcortex (32 Tian S2 parcels) have fundamentally different signal properties in most modalities. The choice of **global** vs **cortex/subcortex separate** normalization depends on the data type and what the model needs to learn.

## Core Principle

| Data type | Normalization | Reason |
|-----------|---------------|--------|
| **fMRI activation** (NeuroSynth, NeuroQuery, NeuroVault, pharma) | **Global** (all 392 parcels) | fMRI BOLD effect sizes are in the same units (t-stats, z-stats) across cortex and subcortex. A "dopamine" map with high striatum and low cortex encodes a real cross-compartment pattern the model should learn. Global z-score preserves this. |
| **Gene expression** (abagen) | **Separate** cortex/subcortex | Different absolute scales, cell-type compositions, AHBA sampling density. A gene "high in thalamus, low in cortex" is a real pattern; global z-score would let cortical mean dominate and distort subcortical signal. |
| **Receptor maps** (receptor_reference, neuromaps PET) | **Separate** | Same as gene expression — PET binding has different absolute scales by tissue. |
| **Structural** (ENIGMA) | **Separate** | Cortical thickness vs subcortical volume are different measures entirely. |

## Implementation

### Cache builders (each normalizes at build time)

| Source | Script | Normalization |
|--------|--------|---------------|
| NeuroQuery decoder | `build_term_maps_cache.py` | Global (`stats.zscore`, axis=1) |
| NeuroSynth | `build_neurosynth_cache.py` | Global (`zscore_maps`) |
| Pharma NeuroSynth | `build_pharma_neurosynth_cache.py` | Global (`zscore_maps`) |
| NeuroVault | `build_neurovault_cache.py` | Global (`zscore_maps`) |
| abagen | `build_abagen_cache.py` | Cortex/subcortex separate |
| receptor_reference | `build_receptor_reference_cache.py` | Cortex/subcortex separate |
| neuromaps | `build_neuromaps_cache.py` | Cortex/subcortex separate |
| ENIGMA | `build_enigma_cache.py` | Cortex/subcortex separate |

### build_expanded_term_maps (merge step)

**No re-normalization.** Each source contributes maps already normalized appropriately. Re-applying cortex/subcortex normalization to the merged set would incorrectly re-normalize fMRI maps and destroy cross-compartment signal (e.g. a "reward" map with strong striatum and weak cortex would get subcortical parcels re-scaled to mean=0 within subcortex, losing the informative pattern).

The `--zscore-renormalize` flag exists only for backwards compatibility; it is deprecated and may distort fMRI patterns.

## Gene PCA pathway

If gene PCA is fit on abagen data (separately normalized), the pharmacological pathway (PDSP → gene PCA → brain map) produces maps in the separately-normalized space. That is correct as long as the gene head output is used as a separate prediction pathway rather than directly compared to fMRI maps. The dual-network architecture (Generalizer + Memorizer) absorbs source-specific scaling in the embedding space.
