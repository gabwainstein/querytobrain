#!/usr/bin/env python3
"""
Gene Expression PCA Pipeline — Phase 4: Drug-to-PC projection.

Loads Phase 1–2 outputs and PDSP Ki (or drug-target CSV). Builds per-drug
binding profiles, projects into PC space, and saves drug coordinates and
spatial maps. Optional: drug similarity in PC space.

See neurolab/docs/implementation/gene_expression_pca_plan.md.

Usage (from repo root):
  python neurolab/scripts/run_gene_pca_phase4.py
  python neurolab/scripts/run_gene_pca_phase4.py --pdsp-csv neurolab/data/pdsp_ki/KiDatabase.csv
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

# Common PDSP receptor name -> gene symbol (for Ki DB "Receptor" column)
RECEPTOR_TO_GENE = {
    "5-HT1A": "HTR1A", "5-HT1B": "HTR1B", "5-HT2A": "HTR2A", "5-HT2C": "HTR2C",
    "D1": "DRD1", "D2": "DRD2", "DAT": "SLC6A3", "SERT": "SLC6A4", "NET": "SLC6A2",
    "GABA-A": "GABRA1", "NMDA": "GRIN1", "mGluR5": "GRM5",
    "M1": "CHRM1", "alpha4beta2": "CHRNA4", "MU": "OPRM1", "CB1": "CNR1",
    "H3": "HRH3", "VAChT": "SLC18A3",
}


def _normalize_receptor_to_gene(name: str) -> str | None:
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name in RECEPTOR_TO_GENE:
        return RECEPTOR_TO_GENE[name]
    # Already gene-like (e.g. HTR2A)
    if re.match(r"^[A-Z0-9\-]+$", name) and len(name) >= 2:
        return name
    return None


def build_drug_profile(
    drug_name: str,
    pdsp_df,
    gene_names_set: set,
    drug_col: str = "Ligand Name",
    target_col: str = "Receptor",
    ki_col: str = "Ki (nM)",
    species_col: str | None = "Species",
) -> dict | None:
    """Build drug binding profile: {gene: affinity_weight}. Weight = 1/Ki, normalized."""
    import pandas as pd
    dc = drug_col if drug_col in pdsp_df.columns else next((c for c in pdsp_df.columns if "ligand" in c.lower() or "drug" in c.lower()), None)
    tc = target_col if target_col in pdsp_df.columns else next((c for c in pdsp_df.columns if "receptor" in c.lower() or "target" in c.lower() or "gene" in c.lower()), None)
    kc = ki_col if ki_col in pdsp_df.columns else next((c for c in pdsp_df.columns if "ki" in c.lower()), None)
    if not dc or not tc or not kc:
        return None
    subset = pdsp_df[pdsp_df[dc].astype(str).str.strip().str.lower() == drug_name.strip().lower()]
    if subset.empty:
        return None
    profile = {}
    for _, row in subset.iterrows():
        gene = _normalize_receptor_to_gene(str(row.get(tc, "")))
        if gene is None or gene not in gene_names_set:
            continue
        try:
            ki = float(row[kc])
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


def drug_to_pc_coordinates(
    drug_weights: dict,
    gene_names: list[str],
    gene_loadings: np.ndarray,
    pc_scores: np.ndarray,
    expression_scaled: np.ndarray,
) -> dict:
    """Project drug profile into PC space (Approach B: gene loadings @ gene_weight_vector)."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_weight_vector = np.zeros(len(gene_names))
    for gene, w in drug_weights.items():
        if gene in gene_to_idx:
            gene_weight_vector[gene_to_idx[gene]] = w
    pc_coords = gene_loadings @ gene_weight_vector
    spatial_profile = np.zeros(expression_scaled.shape[0])
    total_w = 0.0
    for gene, w in drug_weights.items():
        if gene in gene_to_idx:
            spatial_profile += w * expression_scaled[:, gene_to_idx[gene]]
            total_w += w
    if total_w > 0:
        spatial_profile /= total_w
    spatial_from_pcs = pc_scores @ pc_coords
    return {
        "pc_coordinates": pc_coords,
        "spatial_profile_raw": spatial_profile,
        "spatial_profile_pc": spatial_from_pcs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gene PCA Phase 4: drug-to-PC projection")
    parser.add_argument("--output-dir", type=Path, default=None, help="Input/output dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--pdsp-csv", type=Path, default=None, help="PDSP Ki CSV (default: neurolab/data/pdsp_ki/KiDatabase.csv)")
    parser.add_argument("--min-targets", type=int, default=2, help="Minimum targets per drug to include")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Phase 4 requires pandas.", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (repo_root / "neurolab" / "data" / "gene_pca")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    for f in ("expression_scaled.npy", "gene_names.json", "gene_loadings_full.npy", "pc_scores_full.npy"):
        if not (out_dir / f).exists():
            print(f"Run Phase 1 and 2 first. Missing: {out_dir / f}", file=sys.stderr)
            return 1

    expression_scaled = np.load(out_dir / "expression_scaled.npy")
    with open(out_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    gene_loadings = np.load(out_dir / "gene_loadings_full.npy")
    pc_scores = np.load(out_dir / "pc_scores_full.npy")

    pdsp_path = args.pdsp_csv or (repo_root / "neurolab" / "data" / "pdsp_ki" / "KiDatabase.csv")
    if not pdsp_path.is_absolute():
        pdsp_path = repo_root / pdsp_path
    if not pdsp_path.exists():
        print(f"PDSP CSV not found: {pdsp_path}. Run download_pdsp_ki.py or set --pdsp-csv.", file=sys.stderr)
        return 1

    # PDSP columns: Ligand Name, Receptor, Ki (nM), Species (or Ki Value)
    pdsp = pd.read_csv(pdsp_path, low_memory=False, nrows=200000)
    ki_col = "Ki (nM)" if "Ki (nM)" in pdsp.columns else "Ki Value"
    if ki_col not in pdsp.columns:
        ki_col = [c for c in pdsp.columns if "ki" in c.lower()][0] if any("ki" in c.lower() for c in pdsp.columns) else None
    if ki_col is None:
        print("No Ki column found in CSV.", file=sys.stderr)
        return 1
    drug_col = "Ligand Name" if "Ligand Name" in pdsp.columns else "Ligand"
    target_col = "Receptor" if "Receptor" in pdsp.columns else "Target"
    species_col = "Species" if "Species" in pdsp.columns else None
    if species_col and "human" in pdsp[species_col].astype(str).str.lower().unique():
        pdsp = pdsp[pdsp[species_col].astype(str).str.lower() == "human"]

    gene_names_set = set(gene_names)
    drug_profiles = {}
    drug_pc_coords = {}
    drug_spatial_maps = {}

    drugs = pd.unique(pdsp[drug_col].dropna().astype(str).str.strip())
    for drug in drugs:
        if not drug or drug == "nan":
            continue
        weights = build_drug_profile(drug, pdsp, gene_names_set, drug_col=drug_col, target_col=target_col, ki_col=ki_col, species_col=species_col or "")
        if weights is None or len(weights) < args.min_targets:
            continue
        res = drug_to_pc_coordinates(weights, gene_names, gene_loadings, pc_scores, expression_scaled)
        drug_profiles[drug] = weights
        drug_pc_coords[drug] = res["pc_coordinates"]
        drug_spatial_maps[drug] = res["spatial_profile_pc"]

    if not drug_pc_coords:
        print("No drugs with valid binding profiles in gene set. Check PDSP columns and RECEPTOR_TO_GENE mapping.", file=sys.stderr)
        return 1

    drug_names_sorted = sorted(drug_pc_coords.keys())
    drug_pc_matrix = np.array([drug_pc_coords[d] for d in drug_names_sorted])
    drug_spatial_matrix = np.array([drug_spatial_maps[d] for d in drug_names_sorted])

    np.save(out_dir / "drug_pc_coordinates.npy", drug_pc_matrix.astype(np.float32))
    np.save(out_dir / "drug_spatial_maps.npy", drug_spatial_matrix.astype(np.float32))
    with open(out_dir / "drug_names.json", "w") as f:
        json.dump(drug_names_sorted, f)

    # Optional: pairwise similarity in PC space
    from sklearn.metrics.pairwise import cosine_similarity
    norms = np.linalg.norm(drug_pc_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    drug_similarity = cosine_similarity(drug_pc_matrix / norms)
    np.save(out_dir / "drug_similarity_pc.npy", drug_similarity.astype(np.float32))

    print(f"Phase 4 done. {len(drug_names_sorted)} drugs -> PC space and spatial maps. Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
