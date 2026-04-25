# Build all caches with FULL sizes (no caps).
# Decoder ~7K terms, NeuroSynth ~3.4K, neuromaps all, ENIGMA all disorders.
# Run from repo root. Decoder + NeuroSynth take 30-90 min each.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $MyInvocation.MyCommand.Path)\..\..
$DATA = "neurolab\data"

Write-Host "=== Full cache build (no --quick) ==="

# 1. Decoder: full NeuroQuery vocab (~7K). Use n_jobs=1 for reliability.
python neurolab/scripts/build_term_maps_cache.py --cache-dir $DATA/decoder_cache --max-terms 0 --n-jobs 1

# 2. NeuroSynth: full vocab (~3.4K)
python neurolab/scripts/build_neurosynth_cache.py --cache-dir $DATA/neurosynth_cache --max-terms 0

# 3. Merge NQ+NS
python neurolab/scripts/merge_neuroquery_neurosynth_cache.py `
  --neuroquery-cache-dir $DATA/decoder_cache `
  --neurosynth-cache-dir $DATA/neurosynth_cache `
  --output-dir $DATA/unified_cache --prefer neuroquery

# 4. Neuromaps: all MNI152 annotations
python neurolab/scripts/build_neuromaps_cache.py --cache-dir $DATA/neuromaps_cache --max-annot 0

# 5. ENIGMA: all disorders (expanded)
python neurolab/scripts/build_enigma_cache.py --output-dir $DATA/enigma_cache

# 5b. NeuroVault: curated only (all tiers 1–4, ~2.7–5K maps per acquisition guide, NOT bulk ~20K)
# Run download first if neurovault_curated_data/manifest.json missing:
#   python neurolab/scripts/download_neurovault_curated.py --all
if (Test-Path "$DATA/neurovault_curated_data/manifest.json") {
  python neurolab/scripts/build_neurovault_cache.py --data-dir $DATA/neurovault_curated_data --output-dir $DATA/neurovault_cache --from-downloads
} elseif (Test-Path "$DATA/neurovault_data/manifest.json") {
  python neurolab/scripts/build_neurovault_cache.py --data-dir $DATA/neurovault_data --output-dir $DATA/neurovault_cache --from-downloads
}

# 6. Merged sources (training set)
$mergeArgs = @(
  "--cache-dir", "$DATA/unified_cache", "--output-dir", "$DATA/merged_sources",
  "--no-ontology", "--save-term-sources",
  "--neuromaps-cache-dir", "$DATA/neuromaps_cache",
  "--enigma-cache-dir", "$DATA/enigma_cache",
  "--abagen-cache-dir", "$DATA/abagen_cache", "--max-abagen-terms", "2000", "--abagen-add-gradient-pcs", "3", "--gene-pca-variance", "0.95"
)
if (Test-Path "$DATA/neurovault_cache/term_maps.npz") { $mergeArgs += "--neurovault-cache-dir", "$DATA/neurovault_cache" }
python neurolab/scripts/build_expanded_term_maps.py @mergeArgs

Write-Host "Done. Verify: python neurolab/scripts/verify_full_cache_pipeline.py"
