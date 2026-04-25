#!/usr/bin/env python3
"""
Build a (term, map) cache from Allen Human Brain Atlas (AHBA) gene expression via abagen.

Parcellates AHBA microarray expression data to the pipeline atlas (Glasser+Tian, 392 parcels) using abagen,
then saves one (label, map) per gene. Labels are enriched to reduce text underdetermination:
"SYMBOL (full_name), signaling context" (e.g. "HTR2A (serotonin 2A receptor), serotonin signaling")
so the encoder has semantics to latch onto instead of bare symbols like "ADRB2".

**Requirements:** pip install abagen nilearn. abagen will download AHBA data on first run.

**Output:** term_maps.npz (N, n_parcels), term_vocab.pkl. Use with
build_expanded_term_maps.py --abagen-cache-dir neurolab/data/abagen_cache --save-term-sources
(optionally --max-abagen-terms 500 to avoid overweighting the expanded cache).

Usage (from repo root):
  python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache
  python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --genes HTR2A DRD2 SLC6A4
  python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --all-genes --max-genes 500
  python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --receptor-kb  # enriched labels from receptor KB
  python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --receptor-kb --regress-gradient-pcs 5  # Residual maps + gradient PCs + abagen_gradient_components.npy for residual_correlation eval (informed by Fulcher et al.)
  python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache_receptor_residual_selected --receptor-kb --select-by-residual-variance 5  # Select receptors by residual variance (Fulcher), output raw maps of selected
  python neurolab/scripts/build_abagen_cache.py ... --select-by-residual-variance 5 --pca-variance 0.95  # + denoise: project onto PCs explaining 95%% variance
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from neurolab.parcellation import get_combined_atlas_path, get_n_parcels, zscore_cortex_subcortex_separately

# Curated genes: receptor, cell-type, disease-relevant. Expand as needed.
DEFAULT_GENES = [
    "HTR2A", "HTR1A", "SLC6A4",   # serotonin
    "DRD1", "DRD2", "SLC6A3",     # dopamine
    "GRIN2A", "GRIN2B", "SLC17A7", # glutamate
    "GABRA1", "GAD1", "GAD2",     # GABA
    "PVALB", "SST", "VIP",        # interneuron markers
    "BDNF", "SLC6A4",             # depression-related
    "DISC1", "NRG1",              # schizophrenia-related
    "COMT", "MAOA",               # metabolism
]

# Enriched labels: "SYMBOL (full_name), signaling context" — reduces text underdetermination
# so the encoder has semantics to latch onto (e.g. "HTR2A (serotonin 2A receptor), serotonin signaling").
BUILTIN_ENRICHED = {
    "HTR2A": "HTR2A (serotonin 2A receptor), serotonin signaling",
    "HTR1A": "HTR1A (serotonin 1A receptor), serotonin signaling",
    "SLC6A4": "SLC6A4 (serotonin transporter), serotonin signaling",
    "DRD1": "DRD1 (dopamine D1 receptor), dopamine signaling",
    "DRD2": "DRD2 (dopamine D2 receptor), dopamine signaling",
    "SLC6A3": "SLC6A3 (dopamine transporter), dopamine signaling",
    "GRIN2A": "GRIN2A (NMDA receptor subunit), glutamate signaling",
    "GRIN2B": "GRIN2B (NMDA receptor subunit), glutamate signaling",
    "SLC17A7": "SLC17A7 (vesicular glutamate transporter), glutamate signaling",
    "GABRA1": "GABRA1 (GABA-A receptor subunit), GABA signaling",
    "GAD1": "GAD1 (glutamate decarboxylase 1), GABA synthesis",
    "GAD2": "GAD2 (glutamate decarboxylase 2), GABA synthesis",
    "PVALB": "PVALB (parvalbumin), interneuron marker",
    "SST": "SST (somatostatin), interneuron marker",
    "VIP": "VIP (vasoactive intestinal peptide), interneuron marker",
    "BDNF": "BDNF (brain-derived neurotrophic factor), neuroplasticity",
    "DISC1": "DISC1 (disrupted in schizophrenia 1), neurodevelopment",
    "NRG1": "NRG1 (neuregulin 1), neurodevelopment",
    "COMT": "COMT (catechol-O-methyltransferase), catecholamine metabolism",
    "MAOA": "MAOA (monoamine oxidase A), monoamine metabolism",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build abagen gene expression (term, map) cache.")
    parser.add_argument("--output-dir", default="neurolab/data/abagen_cache", help="Output dir for term_maps.npz, term_vocab.pkl")
    parser.add_argument("--genes", nargs="*", default=None, help="Gene symbols (default: curated list; ignored if --all-genes or --receptor-kb)")
    parser.add_argument("--receptor-kb", action="store_true", help="Use receptor_gene_names_v2.json + receptor_knowledge_base.json for genes and descriptions (for training)")
    parser.add_argument("--all-genes", action="store_true", help="Use all genes from AHBA (thousands). Use with --max-genes to cap.")
    parser.add_argument("--data-dir", default=None, help="abagen data dir (AHBA download); default abagen cache")
    parser.add_argument("--max-genes", type=int, default=0, help="Max genes to include (0 = no cap). With --all-genes, random sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for gene sampling when --all-genes and --max-genes")
    parser.add_argument("--zscore-separate", action="store_true", default=True, help="Z-score cortex and subcortex separately (default: True)")
    parser.add_argument("--no-zscore-separate", action="store_false", dest="zscore_separate")
    parser.add_argument("--regress-gradient-pcs", type=int, default=0, help="Regress first K PCs from each gene map. Fit PCA on full AHBA matrix; output residual maps. 0=off.")
    parser.add_argument("--select-by-residual-variance", type=int, default=0, help="With --receptor-kb: fit PCA on full matrix, rank receptors by residual variance (Fulcher), select top --top-receptors. Output RAW maps of selected. 0=off.")
    parser.add_argument("--top-receptors", type=int, default=0, help="With --select-by-residual-variance: max receptors to keep (0=all).")
    parser.add_argument("--pca-variance", type=float, default=0.0, help="With --select-by-residual-variance: denoise by projecting onto PCs explaining this variance (e.g. 0.95). 0=off.")
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
        # Patch abagen for pandas 2.2+ (groupby axis=1 removed)
        import abagen.probes_ as _probes

        def _patched_groupby(microarray, annotation):
            import abagen.io as io
            sid = io.read_annotation(annotation)["structure_id"]
            df = io.read_microarray(microarray)
            return df.T.groupby(sid).mean().T

        _probes._groupby_structure_id = _patched_groupby
    except ImportError:
        print("Install abagen: pip install abagen", file=sys.stderr)
        return 1
    try:
        from nilearn import datasets as nilearn_datasets
        import nibabel as nib
    except ImportError:
        print("Install nilearn: pip install nilearn", file=sys.stderr)
        return 1

    n_parcels = get_n_parcels()
    atlas_path = get_combined_atlas_path()
    if not atlas_path.exists():
        print("Pipeline requires Glasser+Tian (392). Build with: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian --output neurolab/data/combined_atlas_392.nii.gz", file=sys.stderr)
        return 1
    atlas_img = nib.load(str(atlas_path))

    if args.all_genes:
        genes = None
        # Load receptor KB labels for overlap (genes in both AHBA and receptor list get enriched labels)
        try:
            from neurolab.receptor_kb import get_enriched_gene_labels
            gene_labels = get_enriched_gene_labels()
        except Exception:
            gene_labels = {}
    elif args.receptor_kb:
        from neurolab.receptor_kb import load_receptor_genes, get_enriched_gene_labels
        genes = load_receptor_genes()
        gene_labels = get_enriched_gene_labels()
        if args.max_genes and len(genes) > args.max_genes:
            genes = genes[: args.max_genes]
        print(f"Using receptor KB: {len(genes)} genes with enriched labels (SYMBOL (name), signaling)")
    else:
        genes = args.genes or DEFAULT_GENES
        gene_labels = BUILTIN_ENRICHED
        if args.max_genes and len(genes) > args.max_genes:
            genes = genes[: args.max_genes]

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(repo_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # abagen expects atlas as NIfTI or (left, right) GIFTIs
    kwargs = {"atlas": atlas_img, "missing": "centroids", "verbose": 0}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir

    print("Running abagen (may download AHBA data on first run)...")
    expression = abagen.get_expression_data(**kwargs)
    # expression: DataFrame index = parcel IDs (1..n_parcels), columns = gene symbols
    if expression is None or expression.empty:
        print("abagen returned no data.", file=sys.stderr)
        return 1

    # Parcel order: abagen index = atlas parcel IDs (1..n_parcels); ensure n_parcels rows in order
    parcel_ids = np.arange(1, n_parcels + 1)
    expression = expression.reindex(parcel_ids).fillna(0.0)

    select_by_residual = getattr(args, "select_by_residual_variance", 0) or 0
    if select_by_residual and args.receptor_kb:
        from neurolab.receptor_kb import load_receptor_genes, get_enriched_gene_labels
        gene_labels = get_enriched_gene_labels()
        receptor_set = set(g.upper() for g in load_receptor_genes())
        genes = [g for g in expression.columns if g.upper() in receptor_set]
        print(f"Receptor genes in AHBA: {len(genes)} (will select by residual variance)")
    elif args.all_genes:
        genes = list(expression.columns)
        if args.max_genes and len(genes) > args.max_genes:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(genes), size=args.max_genes, replace=False)
            genes = [genes[i] for i in np.sort(idx)]
        print(f"Using {len(genes)} genes from AHBA" + (f" (random sample of {args.max_genes})" if args.max_genes and len(genes) == args.max_genes else ""))

    # Build full gene x parcel matrix (for PCA when regressing gradient PCs)
    n_regress = getattr(args, "regress_gradient_pcs", 0) or 0
    full_maps = []
    full_genes = []
    for g in expression.columns:
        vec = np.asarray(expression[g].values, dtype=np.float64)
        if vec.size != n_parcels:
            vec = vec[:n_parcels] if vec.size >= n_parcels else np.pad(vec, (0, n_parcels - vec.size), constant_values=0.0)
        if args.zscore_separate:
            vec = zscore_cortex_subcortex_separately(vec)
        full_maps.append(vec.ravel())
        full_genes.append(g)
    full_maps = np.stack(full_maps, axis=0).astype(np.float64)
    full_gene_to_idx = {g: i for i, g in enumerate(full_genes)}

    # Fit PCA on full matrix when regressing gradient PCs or selecting by residual variance
    pca = None
    gradient_components = None
    n_comp = max(n_regress, select_by_residual)
    if n_comp > 0 and full_maps.shape[0] > n_comp:
        from sklearn.decomposition import PCA
        n_comp = min(n_comp, full_maps.shape[0] - 1, full_maps.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(full_maps)
        gradient_components = pca.components_.astype(np.float64)  # (K, n_parcels) for residual_correlation
        if n_regress > 0:
            print(f"Fitted PCA on {full_maps.shape[0]} genes; regressing first {n_comp} PCs from each map")
        if select_by_residual:
            print(f"Fitted PCA on {full_maps.shape[0]} genes; selecting receptors by residual variance (first {n_comp} PCs)")

    # Select receptors by residual variance (Fulcher): rank by var(residual), keep top N, output RAW maps
    if select_by_residual and pca is not None and genes:
        residual_variances = []
        for gene in genes:
            if gene not in full_gene_to_idx:
                continue
            vec = full_maps[full_gene_to_idx[gene]].copy()
            reconstructed = pca.inverse_transform(pca.transform(vec.reshape(1, -1)))[0]
            residual = vec - reconstructed
            residual_variances.append((gene, float(np.var(residual))))
        residual_variances.sort(key=lambda x: x[1], reverse=True)
        top_n = args.top_receptors or len(residual_variances)
        genes = [g for g, _ in residual_variances[:top_n]]
        print(f"Selected top {len(genes)} receptors by residual variance (raw maps for training)")

    terms = []
    maps_list = []
    # Add gradient PC maps as synthetic terms (dominant spatial axes)
    if gradient_components is not None and n_regress > 0:
        for i in range(gradient_components.shape[0]):
            terms.append(f"Gene expression gradient PC{i + 1}")
            maps_list.append(gradient_components[i])
    use_residual_for_output = n_regress > 0 and not select_by_residual
    for gene in genes:
        if gene not in full_gene_to_idx:
            print(f"  Gene {gene} not in abagen output; skipping.", file=sys.stderr)
            continue
        vec = full_maps[full_gene_to_idx[gene]].copy()
        if use_residual_for_output and pca is not None:
            # Residual = map - projection onto first K PCs
            reconstructed = pca.inverse_transform(pca.transform(vec.reshape(1, -1)))[0]
            vec = vec - reconstructed
        # Enriched label: "SYMBOL (full_name), signaling" — reduces text underdetermination for encoder
        label = gene_labels.get(gene) or f"{gene} gene expression from Allen Human Brain Atlas ({gene})"
        terms.append(label)
        maps_list.append(vec)

    if not terms:
        print("No genes found in abagen output.", file=sys.stderr)
        return 1

    term_maps = np.stack(maps_list, axis=0).astype(np.float64)
    assert term_maps.shape[1] == n_parcels

    # Denoise: project onto PCs explaining pca_variance (e.g. 95%) — cleans noise
    pca_var = getattr(args, "pca_variance", 0.0) or 0.0
    if pca_var > 0 and select_by_residual and term_maps.shape[0] > 1:
        from sklearn.decomposition import PCA
        n_max = min(term_maps.shape[0] - 1, term_maps.shape[1])
        pca_fit = PCA(n_components=n_max, random_state=42)
        pca_fit.fit(term_maps)
        cumvar = np.cumsum(pca_fit.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, pca_var)) + 1
        k = min(max(1, k), len(cumvar))
        pca_denoise = PCA(n_components=k, random_state=42)
        pca_denoise.fit(term_maps)
        term_maps = pca_denoise.inverse_transform(pca_denoise.transform(term_maps)).astype(np.float64)
        print(f"Denoised: projected onto {k} PCs explaining {100 * cumvar[k - 1]:.1f}% variance")
    # Already z-scored cortex/subcortex separately per-vec above (--zscore-separate default).
    # Gene expression has different scales in cortex vs subcortex (AHBA sampling, cell types).
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    if gradient_components is not None:
        np.save(out_dir / "abagen_gradient_components.npy", gradient_components)
        print(f"Saved abagen_gradient_components.npy for residual correlation evaluation")
    print(f"Saved {len(terms)} gene expression maps x {n_parcels} parcels -> {out_dir}")
    print("Merge with: build_expanded_term_maps.py --abagen-cache-dir", str(out_dir), "--save-term-sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
