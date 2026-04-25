#!/usr/bin/env python3
"""
Gene Expression PCA Pipeline — Phase 3: Biological labeling of PCs.

Loads Phase 2 outputs, runs GO/cell-type/receptor enrichment per PC,
and writes pc_registry.json (human-readable PC labels). gseapy is optional.

See neurolab/docs/implementation/gene_expression_pca_plan.md.

Usage (from repo root):
  python neurolab/scripts/run_gene_pca_phase3.py
  python neurolab/scripts/run_gene_pca_phase3.py --output-dir neurolab/data/gene_pca --skip-go
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

# Cell-type marker genes (plan §3.2)
CELL_TYPE_MARKERS = {
    "excitatory_neurons": ["SLC17A7", "CAMK2A", "NRGN", "SATB2", "TBR1", "SLC17A6"],
    "inhibitory_neurons": ["GAD1", "GAD2", "SLC32A1", "PVALB", "SST", "VIP", "RELN"],
    "astrocytes": ["GFAP", "AQP4", "ALDH1L1", "GJA1", "SLC1A2", "SLC1A3"],
    "oligodendrocytes": ["MBP", "MOG", "PLP1", "OLIG1", "OLIG2", "CNP"],
    "microglia": ["CX3CR1", "P2RY12", "TMEM119", "AIF1", "CSF1R", "CD68"],
    "endothelial": ["CLDN5", "FLT1", "VWF", "PECAM1", "CDH5"],
    "OPC": ["PDGFRA", "CSPG4", "GPR17", "OLIG2"],
}

# Neurotransmitter/receptor genes for loading summary (plan §3.3)
NEUROTRANSMITTER_GENES = {
    "5HT1a": "HTR1A", "5HT1b": "HTR1B", "5HT2a": "HTR2A", "5HT4": "HTR4", "5HT6": "HTR6",
    "D1": "DRD1", "D2": "DRD2",
    "GABAa": "GABRA1",
    "NMDA": "GRIN1", "mGluR5": "GRM5",
    "VAChT": "SLC18A3", "M1": "CHRM1", "a4b2": "CHRNA4",
    "NET": "SLC6A2", "MU": "OPRM1", "CB1": "CNR1", "H3": "HRH3",
    "DAT": "SLC6A3", "SERT": "SLC6A4",
}


def compute_celltype_loadings(gene_loadings: np.ndarray, gene_names: list[str]) -> dict:
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    out = {}
    for pc_idx in range(gene_loadings.shape[0]):
        loadings = gene_loadings[pc_idx, :]
        ct_scores = {}
        for cell_type, markers in CELL_TYPE_MARKERS.items():
            present = [gene_to_idx[m] for m in markers if m in gene_to_idx]
            ct_scores[cell_type] = float(np.mean(loadings[present])) if present else 0.0
        out[f"PC{pc_idx + 1}"] = ct_scores
    return out


def compute_receptor_loadings(gene_loadings: np.ndarray, gene_names: list[str]) -> dict:
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    out = {}
    for pc_idx in range(gene_loadings.shape[0]):
        loadings = gene_loadings[pc_idx, :]
        rec_scores = {}
        for rec_name, gene in NEUROTRANSMITTER_GENES.items():
            if gene in gene_to_idx:
                rec_scores[rec_name] = float(loadings[gene_to_idx[gene]])
        out[f"PC{pc_idx + 1}"] = rec_scores
    return out


def run_go_enrichment(gene_loadings: np.ndarray, gene_names: list[str], n_top: int = 300) -> dict:
    """Run gseapy enrichr for top/negative genes per PC. Returns {pc_idx: {'positive': df, 'negative': df}} or empty."""
    try:
        import gseapy as gp
        import pandas as pd
    except ImportError:
        return {}

    libraries = [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "KEGG_2021_Human",
    ]
    results = {}
    for pc_idx in range(gene_loadings.shape[0]):
        loadings = gene_loadings[pc_idx, :]
        pos_idx = np.argsort(loadings)[-n_top:]
        neg_idx = np.argsort(loadings)[:n_top]
        pos_genes = [gene_names[i] for i in pos_idx]
        neg_genes = [gene_names[i] for i in neg_idx]
        res = {}
        for pole, genes in [("positive", pos_genes), ("negative", neg_genes)]:
            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=libraries,
                    organism="human",
                    outdir=None,
                    no_plot=True,
                )
                df = enr.results
                if df is not None and "Adjusted P-value" in df.columns:
                    sig = df[df["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value").head(10)
                    res[pole] = sig[["Term", "Adjusted P-value", "Overlap"]].to_dict(orient="records")
                else:
                    res[pole] = []
            except Exception:
                res[pole] = []
        results[pc_idx] = res
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Gene PCA Phase 3: biological labeling")
    parser.add_argument("--output-dir", type=Path, default=None, help="Input/output dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--skip-go", action="store_true", help="Skip GO enrichment (no gseapy or slow)")
    args = parser.parse_args()

    out_dir = args.output_dir or (repo_root / "neurolab" / "data" / "gene_pca")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    gene_loadings_path = out_dir / "gene_loadings_full.npy"
    if not gene_loadings_path.exists():
        print(f"Run Phase 2 first. Missing: {gene_loadings_path}", file=sys.stderr)
        return 1

    gene_loadings = np.load(gene_loadings_path)
    with open(out_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    explained_var = np.load(out_dir / "explained_variance.npy")

    # Cell-type and receptor loadings (no extra deps)
    celltype_loadings = compute_celltype_loadings(gene_loadings, gene_names)
    receptor_loadings = compute_receptor_loadings(gene_loadings, gene_names)

    with open(out_dir / "celltype_loadings_per_pc.json", "w") as f:
        json.dump(celltype_loadings, f, indent=2)
    with open(out_dir / "receptor_loadings_per_pc.json", "w") as f:
        json.dump(receptor_loadings, f, indent=2)

    go_results = {}
    if not args.skip_go:
        go_results = run_go_enrichment(gene_loadings, gene_names, n_top=200)
        if go_results:
            with open(out_dir / "go_enrichment_per_pc.json", "w") as f:
                json.dump(go_results, f, indent=2)

    # Build PC registry (template; can be hand-curated later)
    n_pcs = gene_loadings.shape[0]
    pc_registry = {}
    for i in range(n_pcs):
        pc_registry[f"PC{i + 1}"] = {
            "variance_explained": float(explained_var[i]),
            "short_label": f"PC{i + 1}",
            "positive_pole": {"cell_types": list(CELL_TYPE_MARKERS.keys()), "receptors_high": []},
            "negative_pole": {"cell_types": [], "receptors_high": []},
        }
        if i in go_results and go_results[i].get("positive"):
            pc_registry[f"PC{i + 1}"]["positive_pole"]["enriched_go"] = [
                r.get("Term", "") for r in go_results[i]["positive"][:5]
            ]
        if i in go_results and go_results[i].get("negative"):
            pc_registry[f"PC{i + 1}"]["negative_pole"]["enriched_go"] = [
                r.get("Term", "") for r in go_results[i]["negative"][:5]
            ]
        # Top receptors by loading for this PC
        rec_loading = receptor_loadings.get(f"PC{i + 1}", {})
        sorted_rec = sorted(rec_loading.items(), key=lambda x: abs(x[1]), reverse=True)
        pc_registry[f"PC{i + 1}"]["positive_pole"]["receptors_high"] = [r[0] for r in sorted_rec[:5]]
    with open(out_dir / "pc_registry.json", "w") as f:
        json.dump(pc_registry, f, indent=2)

    print(f"Phase 3 done. Saved celltype_loadings_per_pc.json, receptor_loadings_per_pc.json, pc_registry.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
