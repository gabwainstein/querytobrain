#!/usr/bin/env python3
"""
Process PDSP Ki Database CSV into a NeuroLab-ready receptor affinity matrix.

Input:  Raw PDSP CSV (from download_pdsp_ki.py or manual download)
Output:
  - pdsp_affinity_matrix.csv   : compound × receptor matrix of pKi values
  - pdsp_receptor_mapping.json : PDSP receptor names → gene symbols (for abagen mapping)
  - pdsp_compound_profiles.json: per-compound receptor profiles (when --compound-list given)

The affinity matrix feeds into the pharmacological pathway:
  PDSP (pKi vector) → gene_loadings → PC coords → spatial map prediction

Usage:
    python neurolab/scripts/process_pdsp_for_neurolab.py \
        --input neurolab/data/pdsp_ki/KiDatabase.csv \
        --output-dir neurolab/data/pdsp/ \
        [--compound-list neurolab/data/nootropics_knowledge_base.json] \
        [--receptor-genes neurolab/data/receptor_gene_names_v2.json] \
        [--species human] \
        [--min-ki-per-compound 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default paths
_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
_DEFAULT_INPUT = _repo_root / "neurolab" / "data" / "pdsp_ki" / "KiDatabase.csv"
_DEFAULT_OUTPUT = _repo_root / "neurolab" / "data" / "pdsp"

# ── PDSP receptor name → gene symbol mapping ────────────────────────────────
# Maps receptor names as they appear in PDSP to HUGO gene symbols
# that match abagen expression data.

PDSP_TO_GENE = {
    # Serotonin (5-HT)
    "5-ht1a": "HTR1A", "5-ht1b": "HTR1B", "5-ht1d": "HTR1D",
    "5-ht1e": "HTR1E", "5-ht1f": "HTR1F",
    "5-ht2a": "HTR2A", "5-ht2b": "HTR2B", "5-ht2c": "HTR2C",
    "5-ht3": "HTR3A", "5-ht3a": "HTR3A",
    "5-ht4": "HTR4", "5-ht5a": "HTR5A",
    "5-ht6": "HTR6", "5-ht7": "HTR7",
    "sert": "SLC6A4", "5-ht transporter": "SLC6A4",
    # Dopamine
    "d1": "DRD1", "d2": "DRD2", "d3": "DRD3", "d4": "DRD4", "d5": "DRD5",
    "d2l": "DRD2", "d2s": "DRD2", "d4.4": "DRD4",
    "dat": "SLC6A3", "dopamine transporter": "SLC6A3",
    # Adrenergic
    "alpha1a": "ADRA1A", "alpha1b": "ADRA1B", "alpha1d": "ADRA1D",
    "alpha2a": "ADRA2A", "alpha2b": "ADRA2B", "alpha2c": "ADRA2C",
    "beta1": "ADRB1", "beta2": "ADRB2", "beta3": "ADRB3",
    "net": "SLC6A2", "norepinephrine transporter": "SLC6A2",
    # Muscarinic
    "m1": "CHRM1", "m2": "CHRM2", "m3": "CHRM3", "m4": "CHRM4", "m5": "CHRM5",
    # Nicotinic
    "alpha4beta2": "CHRNA4", "alpha7": "CHRNA7", "alpha3beta4": "CHRNA3",
    # Histamine
    "h1": "HRH1", "h2": "HRH2", "h3": "HRH3", "h4": "HRH4",
    # Opioid
    "mor": "OPRM1", "mu": "OPRM1", "mu opioid": "OPRM1",
    "dor": "OPRD1", "delta": "OPRD1", "delta opioid": "OPRD1",
    "kor": "OPRK1", "kappa": "OPRK1", "kappa opioid": "OPRK1",
    "nop": "OPRL1", "orl1": "OPRL1",
    # Cannabinoid
    "cb1": "CNR1", "cb2": "CNR2",
    # GABA
    "gabaa": "GABRA1", "bz": "GABRA1", "benzodiazepine": "GABRA1",
    "gabab": "GABBR1",
    # Glutamate
    "nmda": "GRIN1", "pcp site": "GRIN1",
    "ampa": "GRIA1", "kainate": "GRIK1",
    "mglur1": "GRM1", "mglur2": "GRM2", "mglur3": "GRM3",
    "mglur4": "GRM4", "mglur5": "GRM5",
    # Sigma
    "sigma 1": "SIGMAR1", "sigma1": "SIGMAR1", "sigma-1": "SIGMAR1",
    "sigma 2": "TMEM97", "sigma2": "TMEM97", "sigma-2": "TMEM97",
    # Adenosine
    "a1": "ADORA1", "a2a": "ADORA2A", "a2b": "ADORA2B", "a3": "ADORA3",
    # Melatonin
    "mt1": "MTNR1A", "mt2": "MTNR1B", "ml1a": "MTNR1A", "ml1b": "MTNR1B",
    # Trace amine
    "taar1": "TAAR1",
    # Ion channels
    "herg": "KCNH2", "nav1.7": "SCN9A",
    # Transporters
    "vmat2": "SLC18A2", "vmat1": "SLC18A1",
    "gat1": "SLC6A1",
    # Enzymes
    "mao-a": "MAOA", "mao a": "MAOA", "maoa": "MAOA",
    "mao-b": "MAOB", "mao b": "MAOB", "maob": "MAOB",
    "comt": "COMT", "ache": "ACHE",
    # Neuropeptide
    "nk1": "TACR1", "nk2": "TACR2", "nk3": "TACR3",
    "ox1": "HCRTR1", "ox2": "HCRTR2",
    "v1a": "AVPR1A", "v1b": "AVPR1B", "v2": "AVPR2",
    "oxytocin": "OXTR",
    "npy y1": "NPY1R", "npy y2": "NPY2R",
    "cck-a": "CCKAR", "cck-b": "CCKBR", "ccka": "CCKAR", "cckb": "CCKBR",
}


def normalize_receptor_name(name: str) -> str:
    """Normalize PDSP receptor name for matching."""
    name = name.strip().lower()
    name = re.sub(r"^(human|rat|mouse|guinea pig|hamster|bovine|porcine)\s+", "", name)
    name = re.sub(r"\s*\(.*?\)\s*", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_ki_value(ki_str: str, ki_note: str = "") -> Optional[float]:
    """
    Parse Ki value string to float (nM). Handle '>', '<', '~', ranges.
    Returns None if unparseable.
    """
    if not ki_str or ki_str.strip() in ("", "NA", "N/A", "-", "ND"):
        return None

    s = ki_str.strip()
    s = re.sub(r"^[<>~≈≥≤]+\s*", "", s)

    range_match = re.match(r"([\d.]+)\s*[-–]\s*([\d.]+)", s)
    if range_match:
        try:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            if lo > 0 and hi > 0:
                return math.sqrt(lo * hi)
        except ValueError:
            pass

    s = s.replace(",", "")
    try:
        val = float(s)
        if val > 0:
            return val
    except ValueError:
        pass

    return None


def ki_to_pki(ki_nm: float) -> float:
    """Convert Ki (nM) to pKi = -log10(Ki in M)."""
    if ki_nm <= 0:
        return 0.0
    return -math.log10(ki_nm * 1e-9)


def load_pdsp_csv(path: Path, species_filter: str = "human") -> List[dict]:
    """Load and filter PDSP CSV."""
    records = []

    with open(path, "r", errors="replace") as f:
        first_line = f.readline()
        f.seek(0)

        if "\t" in first_line and "," not in first_line:
            reader = csv.DictReader(f, delimiter="\t")
        else:
            reader = csv.DictReader(f)

        raw_fields = list(reader.fieldnames or [])
        fields_stripped = [fn.strip() for fn in raw_fields]
        field_map = {fs: raw for fs, raw in zip(fields_stripped, raw_fields)}
        print(f"  CSV fields: {fields_stripped[:15]}...")

        def find(candidates: List[str]) -> Optional[str]:
            for c in candidates:
                for fn in fields_stripped:
                    if fn.lower() == c.lower():
                        return field_map[fn]
            return None

        col_receptor = find(["receptor", "name", "Receptor"])
        col_ligand = find(["ligand_name", "ligandname", "Ligand name", "ligand"])
        col_ki = find(["ki_value", "kival", "Ki value", "ki val", "Ki (nM)", "ki"])
        col_kinote = find(["ki_note", "kinote", "Ki note"])
        col_species = find(["species", "Species"])
        col_source = find(["source", "Source"])
        col_smiles = find(["smiles", "SMILES"])
        col_cas = find(["cas", "CAS"])
        col_ligand_id = find(["ligand_id", "ligandid", "Ligand ID"])

        if not col_receptor or not col_ligand or not col_ki:
            print(f"  ERROR: Could not find required columns. Found: {fields}")
            return []

        for row in reader:
            if species_filter and col_species:
                sp = (row.get(col_species) or "").strip().lower()
                if species_filter.lower() not in sp and sp not in ("", "human", "homo sapiens"):
                    if sp:
                        continue

            receptor = (row.get(col_receptor) or "").strip()
            ligand = (row.get(col_ligand) or "").strip()
            ki_str = (row.get(col_ki) or "").strip()
            ki_note = (row.get(col_kinote) or "").strip() if col_kinote else ""

            ki_val = parse_ki_value(ki_str, ki_note)
            if ki_val is None or not receptor or not ligand:
                continue

            records.append({
                "receptor": receptor,
                "receptor_norm": normalize_receptor_name(receptor),
                "ligand": ligand,
                "ligand_lower": ligand.strip().lower(),
                "ki_nm": ki_val,
                "pki": ki_to_pki(ki_val),
                "smiles": (row.get(col_smiles) or "").strip() if col_smiles else "",
                "cas": (row.get(col_cas) or "").strip() if col_cas else "",
                "ligand_id": (row.get(col_ligand_id) or "").strip() if col_ligand_id else "",
                "source": (row.get(col_source) or "").strip() if col_source else "",
            })

    print(f"  Loaded {len(records)} valid Ki records (species={species_filter or 'all'})")
    return records


def build_affinity_matrix(
    records: List[dict],
    receptor_genes: Optional[Dict[str, str]] = None,
    min_ki_per_compound: int = 3,
) -> Tuple[dict, list, list]:
    """
    Build compound × receptor affinity matrix.
    For each (ligand, receptor) pair with multiple Ki values, take geometric mean.
    Returns: (matrix, compounds, receptors)
    """
    receptor_map = dict(PDSP_TO_GENE)
    if receptor_genes:
        receptor_map.update(receptor_genes)

    ki_values: Dict[tuple, List[float]] = defaultdict(list)
    unmapped_receptors: set = set()

    for rec in records:
        gene = receptor_map.get(rec["receptor_norm"])
        if not gene:
            rn = rec["receptor_norm"]
            for key, val in receptor_map.items():
                if key in rn or rn in key:
                    gene = val
                    break

        if not gene:
            unmapped_receptors.add(rec["receptor"])
            continue

        ki_values[(rec["ligand_lower"], gene)].append(rec["ki_nm"])

    if unmapped_receptors:
        print(f"  {len(unmapped_receptors)} unmapped receptor names (showing top 20):")
        for r in sorted(unmapped_receptors)[:20]:
            print(f"    - {r}")

    matrix = {}
    compound_receptors: Dict[str, set] = defaultdict(set)

    for (ligand, gene), kis in ki_values.items():
        geo_mean_ki = math.exp(sum(math.log(k) for k in kis) / len(kis))
        pki = ki_to_pki(geo_mean_ki)
        matrix[(ligand, gene)] = pki
        compound_receptors[ligand].add(gene)

    compounds = sorted([c for c, rs in compound_receptors.items() if len(rs) >= min_ki_per_compound])
    receptors = sorted(set(gene for (_, gene) in matrix.keys()))

    print(f"  Affinity matrix: {len(compounds)} compounds x {len(receptors)} receptors")
    print(f"  (filtered from {len(compound_receptors)} total compounds, min {min_ki_per_compound} receptors)")

    return matrix, compounds, receptors


def main() -> int:
    parser = argparse.ArgumentParser(description="Process PDSP Ki for NeuroLab")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Path to PDSP Ki CSV")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--compound-list", type=Path, help="JSON with compound names (e.g. nootropics)")
    parser.add_argument("--receptor-genes", type=Path, help="JSON with receptor gene symbols")
    parser.add_argument("--species", default="human", help="Species filter (human, rat, or empty for all)")
    parser.add_argument("--min-ki-per-compound", type=int, default=3, help="Min receptor measurements per compound")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PDSP Ki -> NeuroLab Affinity Matrix")
    print("=" * 60)

    receptor_genes = None
    if args.receptor_genes and args.receptor_genes.exists():
        with open(args.receptor_genes) as f:
            receptor_genes = json.load(f)
        print(f"  Loaded {len(receptor_genes)} receptor gene symbols")

    print(f"\nLoading PDSP Ki data from: {args.input}")
    records = load_pdsp_csv(args.input, species_filter=args.species)
    if not records:
        print("ERROR: No valid records loaded")
        return 1

    print("\nBuilding affinity matrix...")
    matrix, compounds, receptors = build_affinity_matrix(
        records,
        receptor_genes=receptor_genes,
        min_ki_per_compound=args.min_ki_per_compound,
    )

    matrix_path = args.output_dir / "pdsp_affinity_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["compound"] + receptors)
        for compound in compounds:
            row = [compound]
            for receptor in receptors:
                pki = matrix.get((compound, receptor), 0.0)
                row.append(f"{pki:.3f}" if pki > 0 else "0")
            writer.writerow(row)
    print(f"\n  Wrote: {matrix_path}")

    mapping_path = args.output_dir / "pdsp_receptor_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(PDSP_TO_GENE, f, indent=2, sort_keys=True)
    print(f"  Wrote: {mapping_path}")

    if args.compound_list and args.compound_list.exists():
        print(f"\nExtracting profiles for compounds in: {args.compound_list}")
        with open(args.compound_list) as f:
            kb = json.load(f)

        if isinstance(kb, list):
            target_names = {c.get("name", "").lower(): c.get("name", "") for c in kb if isinstance(c, dict)}
            target_aliases = {}
            for c in kb:
                if isinstance(c, dict):
                    for alias in c.get("aliases", []):
                        target_aliases[alias.lower()] = c.get("name", "").lower()
        elif isinstance(kb, dict):
            target_names = {k.lower(): k for k in kb.keys()}
            target_aliases = {}
        else:
            target_names = {}
            target_aliases = {}

        profiles = {}
        matched = 0
        for compound in compounds:
            canonical = target_names.get(compound) or target_names.get(target_aliases.get(compound, ""))
            if canonical:
                profile = {}
                for receptor in receptors:
                    pki = matrix.get((compound, receptor), 0.0)
                    if pki > 0:
                        profile[receptor] = round(pki, 3)
                if profile:
                    profiles[canonical] = profile
                    matched += 1

        profiles_path = args.output_dir / "pdsp_compound_profiles.json"
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, sort_keys=True)
        print(f"  Matched {matched}/{len(target_names)} target compounds")
        print(f"  Wrote: {profiles_path}")

    n_nonzero = sum(1 for v in matrix.values() if v > 0)
    density = n_nonzero / (len(compounds) * len(receptors)) * 100 if compounds and receptors else 0
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total Ki records: {len(records)}")
    print(f"  Compounds (>= {args.min_ki_per_compound} receptors): {len(compounds)}")
    print(f"  Receptors (gene symbols): {len(receptors)}")
    print(f"  Non-zero pKi values: {n_nonzero}")
    print(f"  Matrix density: {density:.1f}%")

    compound_counts = defaultdict(int)
    for (c, _), v in matrix.items():
        if v > 0 and c in compounds:
            compound_counts[c] += 1
    print("\n  Top 10 most-profiled compounds:")
    for c, n in sorted(compound_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {c}: {n} receptors")

    receptor_counts = defaultdict(int)
    for (_, r), v in matrix.items():
        if v > 0:
            receptor_counts[r] += 1
    print("\n  Top 10 most-targeted receptors:")
    for r, n in sorted(receptor_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {r}: {n} compounds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
