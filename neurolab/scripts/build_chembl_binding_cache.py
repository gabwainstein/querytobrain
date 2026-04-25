#!/usr/bin/env python3
"""
Build ChEMBL supplementary binding cache for compounds missing from PDSP Ki.
Fills gaps: compounds without Ki values in PDSP get IC50/Ki from ChEMBL.
Output: chembl_binding_cache/ with compound × target affinity matrix (aligned to PDSP receptor genes).

Usage:
  python neurolab/scripts/build_chembl_binding_cache.py --output-dir neurolab/data/chembl_binding_cache
  python neurolab/scripts/build_chembl_binding_cache.py --pdsp-cache neurolab/data/pdsp_cache  # supplement PDSP gaps

Requires: chembl-webresource-client (pip install chembl-webresource-client)
License: ChEMBL data CC-BY-SA 3.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    new_client = None

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Map ChEMBL target names/symbols to gene symbols used in gene_pca (align with PDSP).
CHEMBL_TARGET_TO_GENE = {
    "5-HT1A": "HTR1A",
    "5-HT1B": "HTR1B",
    "5-HT2A": "HTR2A",
    "5-HT2B": "HTR2B",
    "5-HT2C": "HTR2C",
    "D1": "DRD1",
    "D2": "DRD2",
    "D3": "DRD3",
    "D4": "DRD4",
    "D5": "DRD5",
    "DAT": "SLC6A3",
    "SERT": "SLC6A4",
    "NET": "SLC6A2",
    "GABA-A": "GABRA1",  # simplified
    "NMDA": "GRIN1",
    "M1": "CHRM1",
    "M2": "CHRM2",
    "alpha1": "ADRA1A",
    "alpha2": "ADRA2A",
    "beta1": "ADRB1",
    "beta2": "ADRB2",
    "CB1": "CNR1",
    "CB2": "CNR2",
    "H1": "HRH1",
    "H3": "HRH3",
    "MU": "OPRM1",
    "delta": "OPRD1",
    "kappa": "OPRK1",
}


def _normalize_activity_type(atype: str) -> str | None:
    """Prefer Ki, then IC50, then Kd."""
    a = (atype or "").upper()
    if "KI" in a or "K_I" in a or a == "KI":
        return "Ki"
    if "IC50" in a or "IC_50" in a or a == "IC50":
        return "IC50"
    if "KD" in a or "K_D" in a or a == "KD":
        return "Kd"
    return None


def _pchembl_to_affinity(pchembl: float | None) -> float | None:
    """pChEMBL = -log10(affinity_nM) => affinity_nM = 10^(-pChEMBL)."""
    if pchembl is None:
        return None
    try:
        return 10 ** (-float(pchembl))
    except (TypeError, ValueError):
        return None


def _query_chembl_for_compound(molecule_client, activity_client, compound_name: str, limit: int = 500) -> list[dict]:
    """Query ChEMBL for activities of a compound (by name or synonym)."""
    if new_client is None:
        return []
    try:
        mols = molecule_client.search(compound_name)
        if not mols:
            return []
        results = []
        for mol in mols[:5]:  # top 5 molecule matches
            mol_chembl_id = mol.get("molecule_chembl_id")
            if not mol_chembl_id:
                continue
            # Filter by standard_type (Ki, IC50, Kd)
            for stype in ("Ki", "IC50", "Kd"):
                try:
                    acts = activity_client.filter(
                        molecule_chembl_id=mol_chembl_id,
                        standard_type=stype,
                    ).only(
                        "target_chembl_id",
                        "target_pref_name",
                        "standard_type",
                        "standard_value",
                        "standard_units",
                        "pchembl_value",
                    )
                    for a in list(acts)[:limit]:
                        results.append(dict(a))
                except Exception:
                    continue
        return results
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ChEMBL supplementary binding cache for PDSP gaps."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: neurolab/data/chembl_binding_cache)",
    )
    parser.add_argument(
        "--pdsp-cache",
        type=Path,
        default=None,
        help="PDSP cache dir to find compounds missing binding data",
    )
    parser.add_argument(
        "--compounds",
        nargs="*",
        default=None,
        help="Explicit compound list (default: query PDSP compound_names for gaps)",
    )
    parser.add_argument(
        "--max-compounds",
        type=int,
        default=100,
        help="Max compounds to query (default 100)",
    )
    args = parser.parse_args()

    if new_client is None:
        print("Install chembl-webresource-client: pip install chembl-webresource-client", file=sys.stderr)
        return 1

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "chembl_binding_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    compounds_to_query = []
    if args.compounds:
        compounds_to_query = args.compounds[: args.max_compounds]
    elif args.pdsp_cache:
        pdsp_path = Path(args.pdsp_cache) / "compound_names.json"
        if pdsp_path.exists():
            compounds_to_query = json.load(open(pdsp_path))
            compounds_to_query = compounds_to_query[: args.max_compounds]
        else:
            print(f"PDSP compound_names.json not found at {pdsp_path}", file=sys.stderr)
    else:
        # Default: small curated list of nootropics/compounds often missing from PDSP
        compounds_to_query = [
            "piracetam", "aniracetam", "alpha-GPC", "citicoline", "bacopa",
            "ashwagandha", "L-theanine", "N-acetyl cysteine", "noopept",
            "vinpocetine", "sulbutiamine", "idebenone", "creatine",
        ][: args.max_compounds]

    molecule_client = new_client.molecule
    activity_client = new_client.activity

    compound_profiles = {}
    target_to_gene = {}

    for i, name in enumerate(compounds_to_query):
        print(f"[{i+1}/{len(compounds_to_query)}] Querying ChEMBL for {name!r} ...")
        acts = _query_chembl_for_compound(molecule_client, activity_client, name)
        if not acts:
            continue
        for a in acts:
            tname = (a.get("target_pref_name") or "").upper()
            gene = None
            for k, g in CHEMBL_TARGET_TO_GENE.items():
                if k.upper() in tname or g in tname:
                    gene = g
                    break
            if not gene:
                continue
            target_to_gene[tname] = gene
            atype = _normalize_activity_type(a.get("standard_type") or a.get("type", ""))
            if not atype:
                continue
            val = a.get("standard_value") or a.get("pchembl_value")
            if val is None:
                continue
            try:
                if isinstance(val, (int, float)):
                    affinity_nm = float(val)
                else:
                    affinity_nm = float(val)
                if a.get("standard_units", "").upper() in ("UM", "uM", "µM"):
                    affinity_nm *= 1000
            except (TypeError, ValueError):
                pchembl = a.get("pchembl_value")
                affinity_nm = _pchembl_to_affinity(pchembl)
            if affinity_nm is None or affinity_nm <= 0:
                continue
            if name not in compound_profiles:
                compound_profiles[name] = {}
            existing = compound_profiles[name].get(gene)
            if existing is None or affinity_nm < existing:
                compound_profiles[name][gene] = affinity_nm

    if not compound_profiles:
        print("No ChEMBL binding data retrieved.", file=sys.stderr)
        return 1

    all_genes = sorted(set(g for prof in compound_profiles.values() for g in prof))
    compound_names = sorted(compound_profiles.keys())
    n_compounds = len(compound_names)
    n_targets = len(all_genes)
    gene_idx = {g: i for i, g in enumerate(all_genes)}

    import numpy as np
    profiles = np.zeros((n_compounds, n_targets), dtype=np.float32)
    for i, name in enumerate(compound_names):
        for gene, ki in compound_profiles[name].items():
            j = gene_idx.get(gene)
            if j is not None:
                profiles[i, j] = 1.0 / ki  # weight = 1/Ki

    np.savez_compressed(
        out_dir / "chembl_profiles.npz",
        profiles=profiles,
        compound_names=compound_names,
        receptor_genes=all_genes,
    )
    with open(out_dir / "compound_names.json", "w") as f:
        json.dump(compound_names, f, indent=0)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "n_compounds": n_compounds,
            "n_targets": n_targets,
            "source": "ChEMBL",
            "license": "CC-BY-SA 3.0",
        }, f, indent=2)

    print(f"ChEMBL cache: {n_compounds} compounds x {n_targets} targets -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
