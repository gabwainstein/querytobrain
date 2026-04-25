#!/usr/bin/env python3
"""
Compare text encoders for the text->brain embedding (local only).

Trains with a few encoder options on the same data subset and reports
validation MSE and mean correlation so you can pick the best for your goal.

  python neurolab/scripts/compare_encoders.py --cache-dir neurolab/data/decoder_cache --max-terms 500
"""
import argparse
import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Compare encoders for text->brain embedding")
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache", help="Decoder cache path")
    parser.add_argument("--max-terms", type=int, default=500, help="Use this many terms (same for all; 0 = all)")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per run")
    parser.add_argument("--out-dir", default="neurolab/data/embedding_compare", help="Parent dir for each run's output")
    args = parser.parse_args()

    cache = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(repo_root, args.cache_dir)
    out_parent = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)
    os.makedirs(out_parent, exist_ok=True)

    configs = [
        ("tfidf", ["--encoder", "tfidf"]),
        ("sentence-transformers (MiniLM)", ["--encoder", "sentence-transformers", "--encoder-model", "all-MiniLM-L6-v2"]),
        ("sentence-transformers (mpnet)", ["--encoder", "sentence-transformers", "--encoder-model", "all-mpnet-base-v2"]),
    ]

    results = []
    for name, extra in configs:
        out_dir = os.path.join(out_parent, name.replace(" ", "_").replace("(", "").replace(")", ""))
        cmd = [
            sys.executable,
            os.path.join(repo_root, "neurolab", "scripts", "train_text_to_brain_embedding.py"),
            "--cache-dir", cache,
            "--output-dir", out_dir,
            "--max-terms", str(args.max_terms),
            "--epochs", str(args.epochs),
        ] + extra
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        r = subprocess.run(cmd, cwd=repo_root)
        if r.returncode != 0:
            print(f"  FAILED: {name}", file=sys.stderr)
            results.append((name, None, None))
            continue
        # Parse last printed "Val MSE: X.XXXX, mean correlation: Y.YYYY" from stdout (we don't capture it here; user sees it)
        # So we just record that we ran; for a proper comparison we'd need train script to write metrics to a file
        results.append((name, "see above", "see above"))

    print("\n" + "="*60)
    print("Compare the 'Val MSE' and 'mean correlation' lines above for each encoder.")
    print("Higher mean correlation = better predicted map match on held-out terms.")
    print("Then train your final model with the best encoder, e.g.:")
    print("  python neurolab/scripts/train_text_to_brain_embedding.py --encoder sentence-transformers --encoder-model all-mpnet-base-v2 --max-terms 0 --epochs 30")


if __name__ == "__main__":
    main()
