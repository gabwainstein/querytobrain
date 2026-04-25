#!/usr/bin/env python3
"""
Predict brain map for text that is not in NeuroQuery/NeuroSynth (local only).

Uses the trained text->brain embedding to output a parcellated (400,) map and
optionally a 3D NIfTI file for visualization.

  python neurolab/scripts/predict_map.py "alpha-GPC and acetylcholine" --embedding-model neurolab/data/embedding_model
  python neurolab/scripts/predict_map.py "novel compound" --embedding-model neurolab/data/embedding_model --save-nifti predicted.nii.gz
"""
import argparse
import os
import sys

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

N_PARCELS = 400


def main():
    parser = argparse.ArgumentParser(description="Predict brain map for text (local, no API)")
    parser.add_argument("text", help="Phrase to predict map for (e.g. novel compound or long description)")
    parser.add_argument(
        "--embedding-model",
        default=os.path.join(repo_root, "neurolab", "data", "embedding_model"),
        help="Path to trained embedding model dir",
    )
    parser.add_argument(
        "--save-nifti",
        metavar="PATH",
        default=None,
        help="If set, save predicted map as 3D NIfTI (e.g. for viewer)",
    )
    args = parser.parse_args()

    model_dir = args.embedding_model if os.path.isabs(args.embedding_model) else os.path.join(repo_root, args.embedding_model)
    if not os.path.exists(os.path.join(model_dir, "config.pkl")):
        print("Embedding model not found. Train first:", file=sys.stderr)
        print("  python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir", model_dir, file=sys.stderr)
        sys.exit(1)

    from neurolab.enrichment.text_to_brain import TextToBrainEmbedding

    emb = TextToBrainEmbedding(model_dir)
    text = args.text.strip()
    print(f"Predicting map for: \"{text}\"")
    parcellated = emb.predict_map(text)
    assert parcellated.shape == (N_PARCELS,), parcellated.shape
    print(f"  Parcellated map shape: {parcellated.shape}")
    print(f"  Min: {parcellated.min():.3f}, Max: {parcellated.max():.3f}, Mean: {parcellated.mean():.3f}")

    if args.save_nifti:
        out_path = args.save_nifti
        if not os.path.isabs(out_path):
            out_path = os.path.join(repo_root, out_path)
        emb.predict_map_to_nifti(text, out_path, n_parcels=N_PARCELS)
        print(f"  Saved 3D NIfTI: {out_path}")

    print("Done. Use this parcellated map with CognitiveDecoder.decode() or UnifiedEnrichment.enrich().")


if __name__ == "__main__":
    main()
