#!/usr/bin/env python3
"""
Build receptor_gene_names_v2.json and receptor_knowledge_base.json from the canonical CSV.

Canonical source: neurolab/docs/implementation/receptor_gene_list_v2.csv

Usage (from repo root):
  python neurolab/scripts/build_receptor_data_from_csv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from neurolab.receptor_kb import load_receptor_kb

CSV_PATH = repo_root / "neurolab" / "docs" / "implementation" / "receptor_gene_list_v2.csv"
DATA_DIR = repo_root / "neurolab" / "data"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        return 1

    kb = load_receptor_kb(CSV_PATH)
    genes = kb["genes"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "receptor_gene_names_v2.json", "w") as f:
        json.dump(genes, f, indent=2)

    with open(DATA_DIR / "receptor_knowledge_base.json", "w") as f:
        json.dump(kb, f, indent=2)

    print(f"Built from {CSV_PATH.name}: {len(genes)} genes -> {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
