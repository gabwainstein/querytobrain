#!/usr/bin/env python3
"""
Build PDSP Ki cache: compound → receptor affinity profiles and compound → 392-parcel spatial maps.

Parses PDSP KiDatabase.csv, maps receptors to abagen genes, projects through gene PCA,
and saves to pdsp_cache/ for inference and enrichment. See CRITICAL_PATH_CACHES_SPEC.md.

Prerequisites:
  - neurolab/data/pdsp_ki/KiDatabase.csv (from download_pdsp_ki.py)
  - neurolab/data/gene_pca/ with Phase 1–2: expression_scaled.npy, gene_names.json,
    gene_loadings_full.npy, pc_scores_full.npy

Usage (from repo root):
  python neurolab/scripts/build_pdsp_cache.py
  python neurolab/scripts/build_pdsp_cache.py --output-dir neurolab/data/pdsp_cache --min-targets 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# PDSP receptor name -> gene symbol (extended from run_gene_pca_phase4)
RECEPTOR_TO_GENE = {
    "5-HT1A": "HTR1A", "5-HT1B": "HTR1B", "5-HT2A": "HTR2A", "5-HT2C": "HTR2C",
    "5-HT4": "HTR4", "5-HT6": "HTR6", "5-HT7": "HTR7",
    "D1": "DRD1", "D2": "DRD2", "D3": "DRD3", "D4": "DRD4", "D5": "DRD5",
    "DAT": "SLC6A3", "SERT": "SLC6A4", "NET": "SLC6A2",
    "GABA-A": "GABRA1", "GABAB": "GABBR1", "NMDA": "GRIN1", "mGluR5": "GRM5",
    "M1": "CHRM1", "M2": "CHRM2", "alpha4beta2": "CHRNA4", "MU": "OPRM1",
    "CB1": "CNR1", "CB2": "CNR2", "H3": "HRH3", "VAChT": "SLC18A3",
    "alpha1": "ADRA1A", "alpha2": "ADRA2A", "beta1": "ADRB1", "beta2": "ADRB2",
}


def _normalize_receptor_to_gene(name: str) -> str | None:
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name in RECEPTOR_TO_GENE:
        return RECEPTOR_TO_GENE[name]
    if re.match(r"^[A-Z0-9\-]+$", name) and len(name) >= 2:
        return name
    return None


def _build_drug_profile(drug_name: str, pdsp_df, gene_names_set: set, drug_col: str, target_col: str, ki_col: str, species_col: str | None) -> dict | None:
    """Build drug binding profile: {gene: affinity_weight}. Weight = 1/Ki, normalized."""
    subset = pdsp_df[pdsp_df[drug_col].astype(str).str.strip().str.lower() == drug_name.strip().lower()]
    if subset.empty:
        return None
    profile = {}
    for _, row in subset.iterrows():
        gene = _normalize_receptor_to_gene(str(row.get(target_col, "")))
        if gene is None or gene not in gene_names_set:
            continue
        try:
            ki = float(row[ki_col])
        except (TypeError, ValueError):
            continue
        if ki <= 0 or np.isnan(ki):
            continue
        if gene not in profile:
            profile[gene] = []
        profile[gene].append(ki)
    if not profile:
        return None
    weights = {}
    for gene, ki_list in profile.items():
        log_ki = np.log(np.array(ki_list, dtype=float))
        mean_ki = np.exp(np.nanmean(log_ki))
        weights[gene] = 1.0 / mean_ki
    total = sum(weights.values())
    weights = {g: w / total for g, w in weights.items()}
    return weights


def _drug_to_spatial(drug_weights: dict, gene_names: list[str], gene_loadings: np.ndarray, pc_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project drug profile into PC coords and spatial map. Returns (pc_coords, spatial_map)."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_weight_vector = np.zeros(len(gene_names), dtype=np.float32)
    for gene, w in drug_weights.items():
        if gene in gene_to_idx:
            gene_weight_vector[gene_to_idx[gene]] = w
    pc_coords = gene_loadings @ gene_weight_vector
    spatial_map = pc_scores @ pc_coords
    return pc_coords.astype(np.float32), spatial_map.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PDSP Ki cache (compound profiles + spatial maps)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: neurolab/data/pdsp_cache)")
    parser.add_argument("--pdsp-csv", type=Path, default=None, help="PDSP Ki CSV (default: neurolab/data/pdsp_ki/KiDatabase.csv)")
    parser.add_argument("--gene-pca-dir", type=Path, default=None, help="Gene PCA dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--min-targets", type=int, default=2, help="Minimum targets per compound (default 2)")
    parser.add_argument("--max-compounds", type=int, default=0, help="Cap compounds (0 = all)")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Requires pandas: pip install pandas", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir or repo_root / "neurolab" / "data" / "pdsp_cache")
    gene_pca_dir = Path(args.gene_pca_dir or repo_root / "neurolab" / "data" / "gene_pca")
    pdsp_path = Path(args.pdsp_csv or repo_root / "neurolab" / "data" / "pdsp_ki" / "KiDatabase.csv")

    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    if not gene_pca_dir.is_absolute():
        gene_pca_dir = repo_root / gene_pca_dir
    if not pdsp_path.is_absolute():
        pdsp_path = repo_root / pdsp_path

    for f in ("expression_scaled.npy", "gene_names.json", "gene_loadings_full.npy", "pc_scores_full.npy"):
        if not (gene_pca_dir / f).exists():
            print(f"Run gene PCA Phase 1–2 first. Missing: {gene_pca_dir / f}", file=sys.stderr)
            return 1

    if not pdsp_path.exists():
        print(f"PDSP CSV not found: {pdsp_path}. Run download_pdsp_ki.py", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    expression_scaled = np.load(Path(gene_pca_dir) / "expression_scaled.npy")
    with open(Path(gene_pca_dir) / "gene_names.json") as f:
        gene_names = json.load(f)
    gene_loadings = np.load(Path(gene_pca_dir) / "gene_loadings_full.npy")
    pc_scores = np.load(Path(gene_pca_dir) / "pc_scores_full.npy")

    n_parcels = pc_scores.shape[0]
    n_pcs = gene_loadings.shape[0]
    gene_names_set = set(gene_names)

    pdsp = pd.read_csv(pdsp_path, low_memory=False, nrows=200000)
    pdsp.columns = pdsp.columns.str.strip()
    ki_col = "Ki (nM)" if "Ki (nM)" in pdsp.columns else "Ki Value" if "Ki Value" in pdsp.columns else "ki Val" if "ki Val" in pdsp.columns else next((c for c in pdsp.columns if "ki" in c.lower() and "note" not in c.lower()), None)
    if ki_col is None:
        print("No Ki column in CSV.", file=sys.stderr)
        return 1
    drug_col = "Ligand Name" if "Ligand Name" in pdsp.columns else "Ligand"
    target_col = "Receptor" if "Receptor" in pdsp.columns else "Target" if "Target" in pdsp.columns else "Name"
    species_col = "Species" if "Species" in pdsp.columns else "species" if "species" in pdsp.columns else None
    if species_col and "human" in pdsp[species_col].astype(str).str.lower().unique():
        pdsp = pdsp[pdsp[species_col].astype(str).str.lower() == "human"]

    drugs = pd.unique(pdsp[drug_col].dropna().astype(str).str.strip())
    drug_profiles = {}
    drug_pc_coords = {}
    drug_spatial_maps = {}

    for drug in drugs:
        if not drug or drug == "nan":
            continue
        weights = _build_drug_profile(drug, pdsp, gene_names_set, drug_col, target_col, ki_col, species_col)
        if weights is None or len(weights) < args.min_targets:
            continue
        pc_coords, spatial_map = _drug_to_spatial(weights, gene_names, gene_loadings, pc_scores)
        drug_profiles[drug] = weights
        drug_pc_coords[drug] = pc_coords
        drug_spatial_maps[drug] = spatial_map

    if not drug_profiles:
        print("No compounds with valid binding profiles.", file=sys.stderr)
        return 1

    compound_names = sorted(drug_profiles.keys())
    if args.max_compounds and len(compound_names) > args.max_compounds:
        compound_names = compound_names[: args.max_compounds]

    receptor_genes = sorted(set().union(*(p.keys() for p in drug_profiles.values())))
    receptor_gene_index = {g: i for i, g in enumerate(receptor_genes)}
    n_receptors = len(receptor_genes)

    profiles_matrix = np.zeros((len(compound_names), n_receptors), dtype=np.float32)
    for i, name in enumerate(compound_names):
        for gene, w in drug_profiles[name].items():
            if gene in receptor_gene_index:
                profiles_matrix[i, receptor_gene_index[gene]] = w

    projections = np.array([drug_spatial_maps[n] for n in compound_names], dtype=np.float32)
    pc_coordinates = np.array([drug_pc_coords[n] for n in compound_names], dtype=np.float32)

    np.savez_compressed(
        out_dir / "pdsp_profiles.npz",
        profiles=profiles_matrix,
        compound_names=compound_names,
        receptor_genes=receptor_genes,
    )
    np.savez_compressed(
        out_dir / "pdsp_pc_projections.npz",
        projections=projections,
        compound_names=compound_names,
    )
    np.save(out_dir / "pdsp_pc_coordinates.npy", pc_coordinates)
    with open(out_dir / "compound_names.json", "w") as f:
        json.dump(compound_names, f, indent=0)
    with open(out_dir / "receptor_gene_index.json", "w") as f:
        json.dump(receptor_gene_index, f, indent=0)
    metadata = {
        "n_compounds": len(compound_names),
        "n_receptors": n_receptors,
        "n_parcels": n_parcels,
        "n_pcs": n_pcs,
        "gene_pca_dir": str(gene_pca_dir),
        "pdsp_csv_path": str(pdsp_path),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    from sklearn.metrics.pairwise import cosine_similarity
    norms = np.linalg.norm(pc_coordinates, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    similarity = cosine_similarity(pc_coordinates / norms).astype(np.float32)
    np.save(out_dir / "drug_similarity_pc.npy", similarity)

    print(f"PDSP cache: {len(compound_names)} compounds x {n_receptors} receptors x {n_parcels} parcels -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
