#!/bin/bash
# Train text-to-brain embedding in WSL with GPU.
# Usage: bash neurolab/scripts/train_wsl_gpu.sh [epochs]
# Default: 100 epochs, sentence-transformers (PubMedBERT), CUDA.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${1:-100}"

# Use project venv (create if missing)
if [ ! -d .venv ]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# Install training deps if needed
pip install -q sentence-transformers torch 'numpy>=1.21' 'scipy>=1.7' 'pandas>=1.3' 'scikit-learn>=1.0' 2>/dev/null || true
pip install -q -r neurolab/requirements-enrichment.txt 2>/dev/null || true

echo "Training: $EPOCHS epochs, device=cuda, cache=merged_sources"
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/merged_sources \
  --output-dir neurolab/data/embedding_model \
  --epochs "$EPOCHS" \
  --device cuda
