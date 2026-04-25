#!/bin/bash
# Train memorizer (text-to-brain) on WSL with CUDA and OpenAI embeddings.
# Requires: OPENAI_API_KEY in .env or environment, PyTorch with CUDA in WSL.
#
# From repo root (e.g. /mnt/h/querytobrain or ~/querytobrain):
#   bash neurolab/scripts/train_memorizer_wsl.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present (for OPENAI_API_KEY)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python3 neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/merged_sources \
  --output-dir neurolab/data/embedding_model \
  --encoder openai \
  --encoder-model text-embedding-3-large \
  --device cuda \
  --epochs 100
