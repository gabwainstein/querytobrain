#!/usr/bin/env python3
"""
Local verification: train a small text->brain embedding and run embed + decode.

No API. Run from querytobrain root:
  python neurolab/scripts/verify_embedding.py

Requires decoder cache (Phase 2). Trains with --max-terms 200 and --epochs 5 for speed.
"""
import os
import sys
import subprocess

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

CACHE_DIR = os.path.join(repo_root, "neurolab", "data", "decoder_cache")
EMBEDDING_DIR = os.path.join(repo_root, "neurolab", "data", "embedding_model")
N_PARCELS = 400


def main():
    if not os.path.exists(os.path.join(CACHE_DIR, "term_maps.npz")):
        print("Decoder cache missing. Build first:", file=sys.stderr)
        print("  python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 200", file=sys.stderr)
        sys.exit(1)

    print("Step 1: Train text->brain embedding (small, local)...")
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "neurolab", "scripts", "train_text_to_brain_embedding.py"),
            "--cache-dir", CACHE_DIR,
            "--output-dir", EMBEDDING_DIR,
            "--encoder", "tfidf",
            "--max-terms", "200",
            "--epochs", "5",
        ],
        cwd=repo_root,
        timeout=300,
    )
    if r.returncode != 0:
        print("Training failed.", file=sys.stderr)
        sys.exit(1)

    print("\nStep 2: Load embedding and decode...")
    from neurolab.enrichment.text_to_brain import TextToBrainEmbedding
    from neurolab.enrichment.cognitive_decoder import CognitiveDecoder

    emb = TextToBrainEmbedding(EMBEDDING_DIR)
    vec = emb.embed("attention and working memory")
    assert vec.shape == (N_PARCELS,), vec.shape
    print(f"  embed('attention and working memory') -> shape {vec.shape}")

    decoder = CognitiveDecoder(cache_dir=CACHE_DIR)
    out = decoder.decode(vec, top_n=5)
    print(f"  decode(embed(...)) top 5: {[t[0] for t in out['top_terms']]}")

    print("\nStep 3: Query script with --use-embedding-model...")
    r2 = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "neurolab", "scripts", "query.py"),
            "noradrenergic modulation",
            "--use-embedding-model", EMBEDDING_DIR,
            "--top-n", "5",
        ],
        cwd=repo_root,
        timeout=30,
    )
    if r2.returncode != 0:
        print("Query with embedding failed.", file=sys.stderr)
        sys.exit(1)

    print("\nVerify embedding: OK. Local only, no API.")


if __name__ == "__main__":
    main()
