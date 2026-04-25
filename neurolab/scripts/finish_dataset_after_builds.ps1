# Run AFTER decoder + neurosynth full builds complete.
# Decoder: --max-terms 0 (~7K). NeuroSynth: --max-terms 0 (~3.2K).
# Merges NQ+NS -> unified_cache, then builds merged_sources (training set).
# Run from repo root.
# Remove --neurovault-cache-dir / --pharma-neurosynth-cache-dir if those caches don't exist.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $MyInvocation.MyCommand.Path)\..\..
$DATA = "neurolab\data"

Write-Host "=== Finishing dataset: merge + merged_sources ==="

# 1. Merge NQ+NS -> unified_cache
python neurolab/scripts/merge_neuroquery_neurosynth_cache.py `
  --neuroquery-cache-dir $DATA/decoder_cache `
  --neurosynth-cache-dir $DATA/neurosynth_cache `
  --output-dir $DATA/unified_cache --prefer neuroquery

# 2. Merged sources (training set: NQ+NS + neuromaps + enigma + abagen + neurovault + pharma)
python neurolab/scripts/build_expanded_term_maps.py `
  --cache-dir $DATA/unified_cache --output-dir $DATA/merged_sources `
  --no-ontology --save-term-sources `
  --neuromaps-cache-dir $DATA/neuromaps_cache `
  --enigma-cache-dir $DATA/enigma_cache `
  --abagen-cache-dir $DATA/abagen_cache --max-abagen-terms 2000 --gene-pca-variance 0.95 `
  --neurovault-cache-dir $DATA/neurovault_cache `
  --pharma-neurosynth-cache-dir $DATA/pharma_neurosynth_cache

Write-Host "Done. Verify: python neurolab/scripts/verify_full_cache_pipeline.py"
