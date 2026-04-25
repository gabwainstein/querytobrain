#!/usr/bin/env python3
"""
Run text-to-brain training with different --pca-components and compare train vs test correlation.

Usage (from repo root):
  python neurolab/scripts/compare_pca_components.py
  python neurolab/scripts/compare_pca_components.py --pca-values 0 50 80 100 150 200

Prints a table: PC components | Train (or Train+val) corr | Test corr | Gap (train - test).
"""
import argparse
import os
import re
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def run_training(pca_components: int, output_suffix: str, base_args: list) -> dict:
    """Run train_text_to_brain_embedding.py; return dict with train_corr, test_corr, gap."""
    output_dir = os.path.join(repo_root, "neurolab", "data", f"embedding_model_pca{pca_components}")
    cmd = [
        sys.executable,
        os.path.join(repo_root, "neurolab", "scripts", "train_text_to_brain_embedding.py"),
        "--cache-dir", os.path.join(repo_root, "neurolab", "data", "decoder_cache"),
        "--output-dir", output_dir,
        "--encoder", "sentence-transformers",
        "--encoder-model", "all-mpnet-base-v2",
        "--max-terms", "0",
        "--epochs", "40",
        "--lr", "5e-4",
        "--head-hidden", "1024",
        "--dropout", "0.2",
        "--weight-decay", "1e-5",
        "--pca-components", str(pca_components),
        "--final-retrain-on-train-and-val",
    ] + base_args
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return {"train_corr": None, "test_corr": None, "gap": None, "error": "timeout"}
    except Exception as e:
        return {"train_corr": None, "test_corr": None, "gap": None, "error": str(e)}

    # Parse: "Train mean correlation: 0.XXXX" or "Train+val mean correlation (final model): 0.XXXX"
    train_corr = None
    for line in out.splitlines():
        m = re.search(r"(?:Train(?:\+val)? mean correlation(?: \(final model\))?:)\s*([\d.]+)", line)
        if m:
            train_corr = float(m.group(1))
            break
    if train_corr is None:
        m = re.search(r"Train mean correlation:\s*([\d.]+)", out)
        if m:
            train_corr = float(m.group(1))

    test_corr = None
    m = re.search(r"Test mean correlation:\s*([\d.]+)", out)
    if m:
        test_corr = float(m.group(1))

    gap = (train_corr - test_corr) if (train_corr is not None and test_corr is not None) else None
    return {"train_corr": train_corr, "test_corr": test_corr, "gap": gap, "error": result.returncode if result.returncode != 0 else None}


def main():
    parser = argparse.ArgumentParser(description="Compare train vs test correlation across PCA component counts")
    parser.add_argument(
        "--pca-values",
        type=int,
        nargs="+",
        default=[0, 50, 80, 100, 150],
        help="List of --pca-components to try (0 = no PCA)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per run (lower for faster comparison)")
    args = parser.parse_args()

    base_args = ["--epochs", str(args.epochs)]
    rows = []
    for pc in args.pca_values:
        print(f"Running pca_components={pc} ...", flush=True)
        r = run_training(pc, f"pca{pc}", base_args)
        rows.append((pc, r))
        if r.get("error"):
            print(f"  Warning: {r['error']}", flush=True)

    # Table
    print("\n" + "=" * 72)
    print("PCA components  |  Train (or Train+val) corr  |  Test corr  |  Gap (train - test)")
    print("=" * 72)
    for pc, r in rows:
        tr = f"{r['train_corr']:.4f}" if r["train_corr"] is not None else " — "
        te = f"{r['test_corr']:.4f}" if r["test_corr"] is not None else " — "
        gap = f"{r['gap']:.4f}" if r["gap"] is not None else " — "
        print(f"     {pc:3d}         |         {tr:>8}              |    {te:>6}   |      {gap:>6}")
    print("=" * 72)
    print("(Gap = overfitting; smaller gap with similar/better test = better generalization.)")


if __name__ == "__main__":
    main()
