#!/usr/bin/env python3
"""
Run multiple text-to-brain training configs (PCA, early stopping, encoder, etc.)
and compare **test generalization** (primary) and train recovery.

Goal: find configs that **increase test** (generalization to unseen terms), not just reduce train.

Usage (from repo root):
  python neurolab/scripts/run_embedding_experiments.py
  python neurolab/scripts/run_embedding_experiments.py --quick   # fewer epochs, smaller grid
  python neurolab/scripts/run_embedding_experiments.py --pca 0 80 100 --early-stopping

Output: table of config -> train_corr, test_corr, gap; sorted by test_corr (best first).
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


def run_one(
    name: str,
    output_suffix: str,
    extra_args: list[str],
    timeout: int = 600,
) -> dict:
    """Run train_text_to_brain_embedding.py with given extra_args; return train_corr, test_corr, gap, error."""
    output_dir = os.path.join(DATA_DIR, f"embedding_experiment_{output_suffix}")
    cmd = [
        sys.executable,
        SCRIPT,
        "--cache-dir", CACHE_DIR,
        "--output-dir", output_dir,
        "--encoder", "sentence-transformers",
        "--encoder-model", "allenai/scibert_scivocab_uncased",
        "--max-terms", "0",
        "--val-frac", "0.1",
        "--test-frac", "0.1",
        "--dropout", "0.2",
        "--weight-decay", "1e-5",
        "--epochs", "60",
        "--batch-size", "64",
        "--lr", "1e-3",
    ] + extra_args
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

    # Parse: "Train recovery (mean corr ...): X.XX" and "Test generalization (...): X.XX" or "Summary: ..."
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
    parser = argparse.ArgumentParser(description="Run embedding experiments (PCA, early stopping, etc.) and compare test generalization")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs (30), smaller grid (PCA 0 100 only)")
    parser.add_argument("--pca", type=int, nargs="+", default=[0, 80, 100], help="PCA component values to try (0 = no PCA; default includes 100)")
    parser.add_argument("--early-stopping", action="store_true", help="Include runs with --early-stopping")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per run (seconds)")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(CACHE_DIR, "term_maps.npz")):
        print("Cache missing. Build first: python neurolab/scripts/build_all_maps.py --quick", file=sys.stderr)
        sys.exit(1)

    epochs = 30 if args.quick else 60
    pca_values = [0, 100] if args.quick else args.pca

    experiments = []
    # 1) Baseline: no PCA
    experiments.append(("no PCA", "nopca", ["--pca-components", "0", "--epochs", str(epochs)]))
    # 2) PCA only (various)
    for pc in pca_values:
        if pc == 0:
            continue
        experiments.append((f"PCA {pc}", f"pca{pc}", ["--pca-components", str(pc), "--epochs", str(epochs)]))
    # 3) Early stopping (no PCA and PCA 100)
    experiments.append(("no PCA + early stop", "nopca_es", ["--pca-components", "0", "--early-stopping", "--patience", "10", "--epochs", str(epochs)]))
    experiments.append(("PCA 100 + early stop", "pca100_es", ["--pca-components", "100", "--early-stopping", "--patience", "10", "--epochs", str(epochs)]))

    results = []
    for name, suffix, extra in experiments:
        print(f"Running: {name} ...", flush=True)
        r = run_one(name, suffix, extra, timeout=args.timeout)
        results.append(r)
        if r.get("error"):
            print(f"  Warning: {r['error']}", flush=True)
        else:
            print(f"  -> train={r['train_corr']:.4f}, test={r['test_corr']:.4f}, gap={r['gap']:.4f}", flush=True)

    # Sort by test_corr descending (best generalization first)
    valid = [r for r in results if r.get("test_corr") is not None]
    valid.sort(key=lambda x: -x["test_corr"])

    print("\n" + "=" * 78)
    print("Config                        | Train recovery | Test generalization |  Gap")
    print("=" * 78)
    for r in results:
        name = r["name"][:30].ljust(30)
        tr = f"{r['train_corr']:.4f}" if r["train_corr"] is not None else "  —  "
        te = f"{r['test_corr']:.4f}" if r["test_corr"] is not None else "  —  "
        gap = f"{r['gap']:.4f}" if r["gap"] is not None else "  —  "
        err = f" ({r['error']})" if r.get("error") else ""
        print(f" {name} |     {tr}      |        {te}         | {gap}{err}")
    print("=" * 78)
    if valid:
        best = valid[0]
        print(f"Best test generalization: {best['name']} (test={best['test_corr']:.4f})")
    print("(Goal: maximize Test generalization; gap = train - test.)")


if __name__ == "__main__":
    main()
