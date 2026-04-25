#!/usr/bin/env python3
"""
Gene Expression PCA Pipeline — Phase 2: PCA decomposition.

Loads Phase 1 outputs from data/gene_pca/, runs full-genome PCA and receptor PCA.
Default: 90% cumulative variance (standard rule of thumb; ~40 full, ~21 receptor).

See neurolab/docs/implementation/gene_expression_pca_plan.md.

Usage (from repo root):
  python neurolab/scripts/run_gene_pca_phase2.py
  python neurolab/scripts/run_gene_pca_phase2.py --variance 0.90 --receptor-variance 0.90
  python neurolab/scripts/run_gene_pca_phase2.py --variance 0 --n-full 15 --n-receptor 10  # fixed counts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gene PCA Phase 2: full + receptor PCA")
    parser.add_argument("--output-dir", type=Path, default=None, help="Input/output dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--n-full", type=int, default=None, help="Number of full-genome PC components (overridden by --variance)")
    parser.add_argument("--n-receptor", type=int, default=None, help="Number of receptor PCA components (overridden by --receptor-variance)")
    parser.add_argument("--variance", type=float, default=0.90, help="Cumulative variance threshold for full-genome (e.g. 0.90 = 90%%). Standard: 90-95%%. Set 0 to use --n-full instead.")
    parser.add_argument("--receptor-variance", type=float, default=0.90, help="Cumulative variance threshold for receptor PCA. Set 0 to use --n-receptor instead.")
    parser.add_argument("--seed", type=int, default=42, help="Random state for PCA")
    args = parser.parse_args()

    try:
        from sklearn.decomposition import PCA
        import joblib
    except ImportError:
        print("Phase 2 requires scikit-learn and joblib.", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (repo_root / "neurolab" / "data" / "gene_pca")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    # Load Phase 1
    expr_path = out_dir / "expression_scaled.npy"
    if not expr_path.exists():
        print(f"Run Phase 1 first. Missing: {expr_path}", file=sys.stderr)
        return 1
    expression_scaled = np.load(expr_path)
    with open(out_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    with open(out_dir / "parcel_labels.json") as f:
        parcel_labels = json.load(f)
    with open(out_dir / "receptor_gene_names.json") as f:
        receptor_gene_names = json.load(f)

    n_parcels, n_genes = expression_scaled.shape
    if len(gene_names) != n_genes:
        print("Mismatch: gene_names length vs expression columns.", file=sys.stderr)
        return 1

    # Receptor mask
    receptor_set = set(receptor_gene_names)
    receptor_idx = np.array([i for i, g in enumerate(gene_names) if g in receptor_set])
    expression_receptors = expression_scaled[:, receptor_idx]

    # Full-genome PCA
    max_full = min(n_parcels - 1, n_genes - 1)
    if args.variance and args.variance > 0:
        pca_temp = PCA(n_components=min(100, max_full), random_state=args.seed)
        pca_temp.fit(expression_scaled)
        cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
        n_full = min(int(np.searchsorted(cumvar, args.variance)) + 1, max_full)
        n_full = max(2, n_full)
        print(f"Full-genome: {args.variance*100:.0f}% variance -> {n_full} components")
    else:
        n_full = min(args.n_full or 15, max_full)
    pca_full = PCA(n_components=n_full, random_state=args.seed)
    pc_scores_full = pca_full.fit_transform(expression_scaled)
    gene_loadings_full = pca_full.components_
    explained_variance = pca_full.explained_variance_ratio_

    print("Full-genome PCA:")
    cumvar = np.cumsum(explained_variance)
    for i in range(min(10, n_full)):
        print(f"  PC{i+1}: {explained_variance[i]:.4f} (cumulative: {cumvar[i]:.4f})")
    if n_full > 10:
        print(f"  ... PC{n_full}: {explained_variance[-1]:.4f} (cumulative: {cumvar[-1]:.4f})")

    np.save(out_dir / "pc_scores_full.npy", pc_scores_full.astype(np.float32))
    np.save(out_dir / "gene_loadings_full.npy", gene_loadings_full.astype(np.float32))
    np.save(out_dir / "explained_variance.npy", explained_variance.astype(np.float32))
    joblib.dump(pca_full, out_dir / "pca_full_model.pkl")

    # Receptor PCA
    max_rep = min(expression_receptors.shape[1] - 1, n_parcels - 1)
    if args.receptor_variance and args.receptor_variance > 0:
        pca_temp = PCA(n_components=min(50, max_rep), random_state=args.seed)
        pca_temp.fit(expression_receptors)
        cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
        n_rep = min(int(np.searchsorted(cumvar, args.receptor_variance)) + 1, max_rep)
        n_rep = max(2, n_rep)
        print(f"Receptor: {args.receptor_variance*100:.0f}% variance -> {n_rep} components")
    else:
        n_rep = min(args.n_receptor or 10, max_rep)
    if n_rep < 2:
        print("Too few receptor genes for PCA; skipping receptor PCA.", file=sys.stderr)
    else:
        pca_receptor = PCA(n_components=n_rep, random_state=args.seed)
        receptor_pc_scores = pca_receptor.fit_transform(expression_receptors)
        receptor_gene_loadings = pca_receptor.components_
        np.save(out_dir / "receptor_pc_scores.npy", receptor_pc_scores.astype(np.float32))
        np.save(out_dir / "receptor_gene_loadings.npy", receptor_gene_loadings.astype(np.float32))
        joblib.dump(pca_receptor, out_dir / "pca_receptor_model.pkl")
        print(f"Receptor PCA: {n_rep} components, {expression_receptors.shape[1]} genes")

    print(f"Phase 2 done. Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
