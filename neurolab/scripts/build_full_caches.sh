#!/bin/bash
# Build all caches with FULL sizes (no caps).
# Decoder ~7K terms, NeuroSynth ~3.4K, neuromaps all, ENIGMA all disorders.
# Run from repo root. Decoder + NeuroSynth take 30-90 min each.

set -e
cd "$(dirname "$0")/../.."
DATA=neurolab/data

echo "=== Full cache build (no --quick) ==="

# 1. Decoder: full NeuroQuery vocab (~7K). Use n_jobs=1 for reliability.
python neurolab/scripts/build_term_maps_cache.py --cache-dir $DATA/decoder_cache --max-terms 0 --n-jobs 1

# 2. NeuroSynth: full vocab (~3.4K)
python neurolab/scripts/build_neurosynth_cache.py --cache-dir $DATA/neurosynth_cache --max-terms 0

# 3. Merge NQ+NS
python neurolab/scripts/merge_neuroquery_neurosynth_cache.py \
  --neuroquery-cache-dir $DATA/decoder_cache \
  --neurosynth-cache-dir $DATA/neurosynth_cache \
  --output-dir $DATA/unified_cache --prefer neuroquery

# 4. Neuromaps: all MNI152 annotations
python neurolab/scripts/build_neuromaps_cache.py --cache-dir $DATA/neuromaps_cache --max-annot 0

# 5. ENIGMA: all disorders (expanded)
python neurolab/scripts/build_enigma_cache.py --output-dir $DATA/enigma_cache

# 6. Merged sources (training set)
python neurolab/scripts/build_expanded_term_maps.py \
  --cache-dir $DATA/unified_cache --output-dir $DATA/merged_sources \
  --no-ontology --save-term-sources \
  --neuromaps-cache-dir $DATA/neuromaps_cache \
  --enigma-cache-dir $DATA/enigma_cache \
  --abagen-cache-dir $DATA/abagen_cache --max-abagen-terms 2000 --abagen-add-gradient-pcs 3 --gene-pca-variance 0.95

echo "Done. Verify: python neurolab/scripts/verify_full_cache_pipeline.py"
