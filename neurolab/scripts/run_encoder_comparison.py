#!/usr/bin/env python3
"""
Compare text encoders for text-to-brain: run training with several
sentence-transformers models and report test generalization.

Goal: find a better text encoder to improve text-to-map alignment (diagnostics
showed best-map ceiling ~0.97 but our model ~0.55; better encoder may raise test).

Usage (from repo root):
  python neurolab/scripts/run_encoder_comparison.py
  python neurolab/scripts/run_encoder_comparison.py --quick   # fewer epochs
  python neurolab/scripts/run_encoder_comparison.py --encoders all-MiniLM-L6-v2 all-mpnet-base-v2

Output: table of encoder -> train_corr, test_corr, gap; sorted by test_corr (best first).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

DATA_DIR = os.path.join(repo_root, "neurolab", "data")
CACHE_DIR = os.path.join(DATA_DIR, "decoder_cache")
SCRIPT = os.path.join(repo_root, "neurolab", "scripts", "train_text_to_brain_embedding.py")

# Encoders to compare: (display_name, model_id for sentence-transformers)
DEFAULT_ENCODERS = [
    ("MiniLM-L6-v2 (default)", "all-MiniLM-L6-v2"),
    ("mpnet-base-v2 (larger)", "all-mpnet-base-v2"),
    ("PubMedBERT embeddings", "NeuML/pubmedbert-base-embeddings"),
    ("BioClinical ModernBERT", "NeuML/bioclinical-modernbert-base-embeddings"),
]


def run_one(
    name: str,
    model_id: str,
    output_suffix: str,
    epochs: int,
    pca: int,
    early_stop: bool,
    timeout: int,
) -> dict:
    """Run train_text_to_brain_embedding.py with given encoder; return train_corr, test_corr, gap, error."""
    output_dir = os.path.join(DATA_DIR, f"embedding_encoder_{output_suffix}")
    cmd = [
        sys.executable,
        SCRIPT,
        "--cache-dir", CACHE_DIR,
        "--output-dir", output_dir,
        "--encoder", "sentence-transformers",
        "--encoder-model", model_id,
        "--max-terms", "0",
        "--val-frac", "0.1",
        "--test-frac", "0.1",
        "--dropout", "0.2",
        "--weight-decay", "1e-5",
        "--epochs", str(epochs),
        "--batch-size", "64",
        "--lr", "1e-3",
        "--pca-components", str(pca),
    ]
    if early_stop:
        cmd += ["--early-stopping", "--patience", "10"]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return {"name": name, "train_corr": None, "test_corr": None, "gap": None, "error": "timeout"}
    except Exception as e:
        return {"name": name, "train_corr": None, "test_corr": None, "gap": None, "error": str(e)}

    train_corr = None
    test_corr = None
    for line in out.splitlines():
        m = re.search(r"Train recovery \(mean corr[^)]*\):\s*([\d.]+)", line)
        if m:
            train_corr = float(m.group(1))
        m = re.search(r"Test generalization \(mean corr[^)]*\):\s*([\d.]+)", line)
        if m:
            test_corr = float(m.group(1))
        m = re.search(r"Summary:\s*Train recovery\s*=\s*([\d.]+).*Test generalization\s*=\s*([\d.]+)", line)
        if m:
            train_corr = float(m.group(1))
            test_corr = float(m.group(2))
    if train_corr is None:
        m = re.search(r"Train (?:recovery|mean correlation)[^:]*:\s*([\d.]+)", out)
        if m:
            train_corr = float(m.group(1))
    if test_corr is None:
        m = re.search(r"Test (?:generalization|mean correlation)[^:]*:\s*([\d.]+)", out)
        if m:
            test_corr = float(m.group(1))
    gap = (train_corr - test_corr) if (train_corr is not None and test_corr is not None) else None
    err = None
    if result.returncode != 0:
        err = f"exit {result.returncode}"
    return {"name": name, "train_corr": train_corr, "test_corr": test_corr, "gap": gap, "error": err}


def main():
    parser = argparse.ArgumentParser(description="Compare text encoders for text-to-brain")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs (30), no BioClinical (slower model)")
    parser.add_argument("--encoders", type=str, nargs="+", default=None,
                        help="Encoder model IDs to try (default: MiniLM, mpnet, PubMedBERT, BioClinical)")
    parser.add_argument("--pca", type=int, default=100, help="PCA components (0 = no PCA)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per run (seconds)")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(CACHE_DIR, "term_maps.npz")):
        print("Cache missing. Build first: python neurolab/scripts/build_term_maps_cache.py --cache-dir ...", file=sys.stderr)
        sys.exit(1)

    epochs = 30 if args.quick else 50
    early_stop = not args.no_early_stopping

    if args.encoders:
        encoders = [(e, e) for e in args.encoders]
    else:
        encoders = DEFAULT_ENCODERS
        if args.quick:
            encoders = [e for e in encoders if "bioclinical" not in e[1].lower()]

    results = []
    for name, model_id in encoders:
        suffix = model_id.replace("/", "_").replace("-", "_")[:40]
        print(f"Running: {name} ({model_id}) ...", flush=True)
        r = run_one(name, model_id, suffix, epochs, args.pca, early_stop, args.timeout)
        results.append(r)
        if r.get("error"):
            print(f"  Error: {r['error']}", flush=True)
        else:
            print(f"  -> train={r['train_corr']:.4f}, test={r['test_corr']:.4f}, gap={r['gap']:.4f}", flush=True)

    valid = [r for r in results if r.get("test_corr") is not None]
    valid.sort(key=lambda x: -x["test_corr"])

    print("\n" + "=" * 80)
    print("Encoder (model)                    | Train recovery | Test generalization |  Gap")
    print("=" * 80)
    for r in results:
        name = (r["name"][:36] + "..") if len(r["name"]) > 38 else r["name"].ljust(38)
        tr = f"{r['train_corr']:.4f}" if r["train_corr"] is not None else "  —  "
        te = f"{r['test_corr']:.4f}" if r["test_corr"] is not None else "  —  "
        gap = f"{r['gap']:.4f}" if r["gap"] is not None else "  —  "
        err = f" ({r['error']})" if r.get("error") else ""
        print(f" {name} |     {tr}      |        {te}         | {gap}{err}")
    print("=" * 80)
    if valid:
        best = valid[0]
        print(f"Best test generalization: {best['name']} (test={best['test_corr']:.4f})")
    print("(Goal: maximize Test generalization for better text-to-map alignment.)")


if __name__ == "__main__":
    main()
