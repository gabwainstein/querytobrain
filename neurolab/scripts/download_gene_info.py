#!/usr/bin/env python3
"""
Download HGNC gene symbol -> name mapping for enriching abagen term labels.
Saves to neurolab/data/gene_info.json for use by build_expanded_term_maps.py.

Usage:
  python neurolab/scripts/download_gene_info.py
  python neurolab/scripts/download_gene_info.py --output neurolab/data/gene_info.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = repo_root / "neurolab" / "data" / "gene_info.json"
# HGNC complete set (symbol, name). EBI FTP or Google Cloud.
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download HGNC gene symbol->name for abagen re-labeling")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    args = ap.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        req = urllib.request.Request(HGNC_URL, headers={"User-Agent": "NeuroLab/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    except Exception as e:
        print(f"Failed to fetch HGNC: {e}", file=sys.stderr)
        return 1

    if not lines:
        print("Empty HGNC response", file=sys.stderr)
        return 1

    header = lines[0].split("\t")
    try:
        sym_idx = header.index("symbol")
        name_idx = header.index("name")
    except ValueError:
        print("HGNC format changed; expected symbol and name columns", file=sys.stderr)
        return 1
    locus_idx = header.index("locus_group") if "locus_group" in header else -1

    gene_info = {}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) > max(sym_idx, name_idx):
            sym = parts[sym_idx].strip()
            name = parts[name_idx].strip()
            locus = parts[locus_idx].strip() if locus_idx >= 0 and len(parts) > locus_idx else ""
            if sym and name:
                entry = {"name": name}
                if locus:
                    entry["locus_group"] = locus
                gene_info[sym] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gene_info, f, indent=0, sort_keys=True)

    print(f"Saved {len(gene_info)} gene symbol->name mappings -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
