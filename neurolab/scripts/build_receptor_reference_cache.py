#!/usr/bin/env python3
"""
Build receptor reference cache: 250 receptor genes + receptor PCs with rich labels.

Strategy A from NEUROVAULT_TRAINING_SET_INGESTION_ALGORITHM.md: add receptor geography
to the training set so the Generalizer learns where receptors are expressed. Tag as
"reference" source (5% batch share in stratified sampling).

Prerequisites:
  - abagen (for gene expression)
  - neurolab/data/gene_pca/ with receptor_pc_scores.npy (from run_gene_pca_phase2.py)

Usage (from repo root):
  python neurolab/scripts/build_receptor_reference_cache.py
  python neurolab/scripts/build_receptor_reference_cache.py --output-dir neurolab/data/receptor_reference_cache --skip-pcs
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neurolab.parcellation import get_n_parcels, zscore_cortex_subcortex_separately
from neurolab.receptor_kb import load_receptor_genes, get_rich_gene_descriptions

# Descriptive labels for receptor PCs (generic; can be refined from loadings analysis)
RECEPTOR_PC_LABELS = [
    "Principal component 1 of receptor gene expression: cortical gradient from sensory to association cortex",
    "Principal component 2 of receptor gene expression: subcortical versus cortical monoamine receptor density gradient",
    "Principal component 3 of receptor gene expression: serotonergic versus dopaminergic receptor expression dominance",
    "Principal component 4 of receptor gene expression: glutamatergic receptor distribution gradient",
    "Principal component 5 of receptor gene expression: GABAergic receptor expression pattern",
    "Principal component 6 of receptor gene expression: cholinergic receptor density gradient",
    "Principal component 7 of receptor gene expression: opioid receptor expression pattern",
    "Principal component 8 of receptor gene expression: adrenergic receptor distribution",
    "Principal component 9 of receptor gene expression: ion channel expression gradient",
    "Principal component 10 of receptor gene expression: transporter expression pattern",
]
# Extend with generic labels if more PCs
for i in range(11, 35):
    RECEPTOR_PC_LABELS.append(f"Principal component {i} of receptor gene expression: neurotransmitter system spatial gradient")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build receptor reference cache (genes + PCs, rich labels)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: neurolab/data/receptor_reference_cache)")
    parser.add_argument("--gene-pca-dir", type=Path, default=None, help="Gene PCA dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--skip-pcs", action="store_true", help="Skip receptor PCs; only add gene maps")
    args = parser.parse_args()

    try:
        import pandas as _pd
        if not hasattr(_pd.DataFrame, "_append_patched"):
            def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
                return _pd.concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
            _pd.DataFrame.append = _df_append
            _pd.DataFrame._append_patched = True
        if not hasattr(_pd.DataFrame, "_set_axis_patched"):
            _orig_set_axis = _pd.DataFrame.set_axis
            def _patched_set_axis(self, labels, axis=0, **kwargs):
                kwargs.pop("inplace", None)
                return _orig_set_axis(self, labels, axis=axis, **kwargs)
            _pd.DataFrame.set_axis = _patched_set_axis
            _pd.DataFrame._set_axis_patched = True
        import abagen
        import nibabel as nib
    except ImportError as e:
        print(f"Requires abagen, nibabel, pandas: {e}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (repo_root / "neurolab" / "data" / "receptor_reference_cache")
    gene_pca_dir = args.gene_pca_dir or (repo_root / "neurolab" / "data" / "gene_pca")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    if not gene_pca_dir.is_absolute():
        gene_pca_dir = repo_root / gene_pca_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    n_parcels = get_n_parcels()

    # 1. Receptor genes with rich labels
    genes = load_receptor_genes()
    rich_desc = get_rich_gene_descriptions()
    from neurolab.parcellation import get_combined_atlas_path
    atlas_path = get_combined_atlas_path()
    if not atlas_path.exists():
        print(f"Atlas not found: {atlas_path}. Run build_combined_atlas.py first.", file=sys.stderr)
        return 1
    atlas_img = nib.load(str(atlas_path))

    print("Fetching gene expression via abagen...")
    expression = abagen.get_expression_data(atlas=atlas_img, missing="centroids", verbose=0)
    if expression is None or expression.empty:
        print("abagen returned no data.", file=sys.stderr)
        return 1

    parcel_ids = np.arange(1, n_parcels + 1)
    expression = expression.reindex(parcel_ids).fillna(0.0)

    terms = []
    maps_list = []
    for gene in genes:
        if gene not in expression.columns:
            continue
        vec = np.asarray(expression[gene].values, dtype=np.float64)
        if vec.size != n_parcels:
            vec = vec[:n_parcels] if vec.size >= n_parcels else np.pad(vec, (0, n_parcels - vec.size), constant_values=0.0)
        vec = zscore_cortex_subcortex_separately(vec)
        label = rich_desc.get(gene, f"{gene} gene expression across cortex")
        terms.append(f"{label} from Allen Human Brain Atlas ({gene})")
        maps_list.append(vec.ravel())

    n_genes = len(terms)
    print(f"Added {n_genes} receptor genes with rich labels")

    # 2. Receptor PCs (from gene PCA Phase 2)
    if not args.skip_pcs:
        pc_path = gene_pca_dir / "receptor_pc_scores.npy"
        if pc_path.exists():
            pc_scores = np.load(pc_path)  # (392, n_components)
            if pc_scores.shape[0] == n_parcels:
                n_pcs = min(pc_scores.shape[1], len(RECEPTOR_PC_LABELS))
                for i in range(n_pcs):
                    vec = pc_scores[:, i].astype(np.float64)
                    vec = zscore_cortex_subcortex_separately(vec)
                    base = RECEPTOR_PC_LABELS[i] if i < len(RECEPTOR_PC_LABELS) else f"Receptor PC{i+1} spatial gradient"
                    label = f"{base} (Allen Human Brain Atlas)"
                    terms.append(label)
                    maps_list.append(vec.ravel())
                print(f"Added {n_pcs} receptor PCs")
            else:
                print(f"Receptor PC shape mismatch (expected {n_parcels} parcels); skipping PCs.", file=sys.stderr)
        else:
            print(f"Receptor PC scores not found at {pc_path}; run run_gene_pca_phase2.py first. Skipping PCs.", file=sys.stderr)

    if not terms:
        print("No terms to save.", file=sys.stderr)
        return 1

    term_maps = np.stack(maps_list, axis=0).astype(np.float64)
    # Already z-scored cortex/subcortex separately per-vec above.
    # Receptor/gene expression has different scales in cortex vs subcortex.
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)

    # Save term_sources for "reference" tagging
    term_sources = ["reference"] * len(terms)
    with open(out_dir / "term_sources.pkl", "wb") as f:
        pickle.dump(term_sources, f)

    print(f"Saved {len(terms)} terms x {n_parcels} parcels -> {out_dir}")
    print("Merge with: build_expanded_term_maps.py --receptor-reference-cache-dir", str(out_dir), "--save-term-sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
