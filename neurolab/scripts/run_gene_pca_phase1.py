#!/usr/bin/env python3
"""
Gene Expression PCA Pipeline — Phase 1: Data acquisition and preprocessing.

Fetches AHBA gene expression via abagen on the pipeline atlas (Glasser+Tian, 392 parcels).
applies variance and optional receptor filtering, standardizes, and saves
expression matrix and metadata to data/gene_pca/ for Phase 2.

See neurolab/docs/implementation/gene_expression_pca_plan.md.

Usage (from repo root):
  python neurolab/scripts/run_gene_pca_phase1.py
  python neurolab/scripts/run_gene_pca_phase1.py --output-dir neurolab/data/gene_pca --variance-percentile 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neurolab.parcellation import (
    get_combined_atlas_path,
    get_n_parcels,
    N_CORTICAL_GLASSER,
    N_PARCELS_392,
)

# Receptor/transporter/channel gene prefixes for pharmacology-focused PCA (plan §1.2)
RECEPTOR_GENE_FAMILIES = [
    "HTR", "DRD", "GABA", "GRI", "GRM", "GRIA", "GRIK", "GRIN", "CHRN", "CHRM",
    "OPRM", "OPRD", "OPRK", "ADRA", "ADRB", "HRH", "SLC6A", "CNR",
    "CACNA", "SCN", "KCNA", "KCNJ",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Gene PCA Phase 1: fetch expression, filter, standardize")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--data-dir", default=None, help="abagen data dir for AHBA; default abagen cache")
    parser.add_argument("--variance-percentile", type=float, default=10.0, help="Drop genes below this variance percentile (default 10)")
    parser.add_argument("--no-variance-filter", action="store_true", help="Skip variance filtering")
    parser.add_argument(
        "--separate-cortex-subcortex",
        action="store_true",
        default=True,
        help="Standardize cortex (parcels 1-360) and subcortex (361-392) separately (default: True). "
        "Cortex and subcortex have different expression distributions; separate scaling avoids one dominating PCA.",
    )
    parser.add_argument("--no-separate-cortex-subcortex", action="store_false", dest="separate_cortex_subcortex")
    parser.add_argument(
        "--receptor-list",
        type=Path,
        default=None,
        help="Curated receptor gene list (CSV or JSON). Default: neurolab/docs/implementation/receptor_gene_list_v2.csv. "
        "Receptor subset = intersection with expressed genes. Use prefix-based selection if file not found.",
    )
    args = parser.parse_args()

    try:
        import pandas as _pd
        # Patch pandas for abagen compatibility (pandas 2.0+ removed append, 3.0 removed set_axis inplace)
        if not hasattr(_pd.DataFrame, "_append_patched"):
            def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
                return _pd.concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
            _pd.DataFrame.append = _df_append
            _pd.DataFrame._append_patched = True
        _orig_set_axis = _pd.DataFrame.set_axis
        def _patched_set_axis(self, labels, axis=0, **kwargs):
            kwargs.pop("inplace", None)
            return _orig_set_axis(self, labels, axis=axis, **kwargs)
        _pd.DataFrame.set_axis = _patched_set_axis
        import abagen
        import nibabel as nib
    except ImportError as e:
        print("Phase 1 requires abagen and nibabel: pip install abagen nilearn", file=sys.stderr)
        return 1

    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Phase 1 requires sklearn: pip install scikit-learn", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (repo_root / "neurolab" / "data" / "gene_pca")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_parcels = get_n_parcels()
    atlas_path = get_combined_atlas_path()
    if not atlas_path.exists():
        print("Pipeline requires Glasser+Tian (392). Build with: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian --output neurolab/data/combined_atlas_392.nii.gz", file=sys.stderr)
        return 1
    atlas_img = nib.load(str(atlas_path))

    kwargs = {
        "atlas": atlas_img,
        "ibf_threshold": 0.5,
        "probe_selection": "diff_stability",
        "donor_probes": "aggregate",
        "lr_mirror": "bidirectional",
        "missing": "interpolate",
        "tolerance": 2,
        "sample_norm": "srs",
        "gene_norm": "srs",
        "norm_matched": True,
        "return_donors": False,
        "verbose": 1,
    }
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir

    print("Fetching gene expression via abagen (may download AHBA on first run)...")
    expression = abagen.get_expression_data(**kwargs)
    if expression is None or expression.empty:
        print("abagen returned no data.", file=sys.stderr)
        return 1

    # Align rows to parcel order 1..n_parcels
    parcel_ids = np.arange(1, n_parcels + 1)
    expression = expression.reindex(parcel_ids).fillna(0.0)
    expression = expression.loc[:, expression.notna().all(axis=0)]
    if expression.shape[1] == 0:
        print("No genes left after reindex.", file=sys.stderr)
        return 1

    gene_names = expression.columns.tolist()
    parcel_labels = expression.index.astype(int).tolist()

    # Variance filter: drop bottom percentile
    if not args.no_variance_filter and args.variance_percentile > 0:
        gene_var = expression.var(axis=0)
        thresh = np.percentile(gene_var, args.variance_percentile)
        high_var = gene_var >= thresh
        expression = expression.loc[:, high_var]
        gene_names = expression.columns.tolist()
        print(f"Variance filter: kept {len(gene_names)} genes (dropped below {args.variance_percentile}th percentile)")

    # Receptor subset (for Phase 2 receptor PCA). Default: receptor_gene_list_v2.csv (curated).
    from neurolab.receptor_kb import load_receptor_genes
    receptor_list_path = args.receptor_list or (repo_root / "neurolab" / "docs" / "implementation" / "receptor_gene_list_v2.csv")
    if receptor_list_path.exists():
        receptor_canonical = set(load_receptor_genes(receptor_list_path))
        receptor_gene_names = [g for g in gene_names if g in receptor_canonical]
        print(f"Receptor/transporter genes: {len(receptor_gene_names)} (from {receptor_list_path.name})")
    else:
        receptor_mask = np.array([
            any(g.startswith(prefix) for prefix in RECEPTOR_GENE_FAMILIES)
            for g in gene_names
        ])
        receptor_gene_names = [g for g, m in zip(gene_names, receptor_mask) if m]
        print(f"Receptor/transporter genes: {len(receptor_gene_names)} (prefix-based, v2 not found)")

    # Standardize: each gene mean=0, std=1 across parcels
    expr = expression.values
    if args.separate_cortex_subcortex and n_parcels == N_PARCELS_392:
        # Cortex and subcortex have different expression distributions (AHBA sampling, cell types).
        # Standardize each separately so neither dominates PCA.
        n_cort = N_CORTICAL_GLASSER
        expr_cort = expr[:n_cort]
        expr_sub = expr[n_cort:n_parcels]
        scaler_cort = StandardScaler()
        scaler_sub = StandardScaler()
        expr_cort_scaled = scaler_cort.fit_transform(expr_cort)
        expr_sub_scaled = scaler_sub.fit_transform(expr_sub)
        expression_scaled = np.vstack([expr_cort_scaled, expr_sub_scaled]).astype(np.float32)
        import joblib
        joblib.dump({"cortex": scaler_cort, "subcortex": scaler_sub}, out_dir / "expression_scaler.pkl")
        print("Standardized cortex and subcortex separately (360 + 32 parcels)")
    else:
        scaler = StandardScaler()
        expression_scaled = scaler.fit_transform(expr).astype(np.float32)
        import joblib
        joblib.dump(scaler, out_dir / "expression_scaler.pkl")

    # Save
    np.save(out_dir / "expression_scaled.npy", expression_scaled)
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f, indent=0)
    with open(out_dir / "parcel_labels.json", "w") as f:
        json.dump(parcel_labels, f)
    with open(out_dir / "receptor_gene_names.json", "w") as f:
        json.dump(receptor_gene_names, f)

    print(f"Phase 1 done. Shape: {expression_scaled.shape[0]} parcels x {expression_scaled.shape[1]} genes -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
