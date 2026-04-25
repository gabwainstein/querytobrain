#!/usr/bin/env python3
"""
Apply curated pharma relabeling schema to NeuroVault pharma cache.

Uses neurolab/data/neurovault_pharma_schema.json to prepend canonical
drug/control/measure prefixes so labels stop collapsing into generic task semantics.
See NEUROVAULT_PHARMA_AUDIT.md and the schema for the full spec.

Usage:
  python neurolab/scripts/relabel_pharma_terms.py --cache-dir neurolab/data/neurovault_pharma_cache
  python neurolab/scripts/relabel_pharma_terms.py --cache-dir neurolab/data/neurovault_pharma_cache --output-dir neurolab/data/neurovault_pharma_cache_relabeled
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "neurolab" / "data" / "neurovault_pharma_schema.json"


def _make_labels_unique(labels: list[str]) -> list[str]:
    """Append _1, _2, ... to duplicates."""
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 0
            out.append(lab)
        else:
            seen[lab] += 1
            out.append(f"{lab}_{seen[lab]}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply curated pharma relabeling schema to NeuroVault pharma cache"
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "neurolab" / "data" / "neurovault_pharma_cache"),
        help="Input pharma cache directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: overwrite cache-dir)",
    )
    parser.add_argument(
        "--schema",
        default=str(SCHEMA_PATH),
        help="Path to neurovault_pharma_schema.json",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Skip applying prefixes (dry run / copy only)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = REPO_ROOT / schema_path

    if not (cache_dir / "term_maps.npz").exists():
        print(f"Cache not found: {cache_dir}", file=sys.stderr)
        return 1

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    prefix_map = schema.get("label_prefix_by_collection") or {}
    prefix_map = {int(k): v for k, v in prefix_map.items()}

    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    cids = pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb"))
    if (cache_dir / "term_sample_weights.pkl").exists():
        weights = pickle.load(open(cache_dir / "term_sample_weights.pkl", "rb"))
    else:
        weights = [1.0] * len(terms)

    n_before = len(terms)
    print(f"Loaded {n_before} terms from {cache_dir}")

    new_terms = []
    for lab, cid in zip(terms, cids):
        if not args.no_prefix and cid in prefix_map:
            prefix = prefix_map[cid]
            if not lab.startswith(prefix):
                lab = prefix + lab
        new_terms.append(lab)

    terms = _make_labels_unique(new_terms)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "term_maps.npz", term_maps=term_maps)
    with open(output_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    with open(output_dir / "term_collection_ids.pkl", "wb") as f:
        pickle.dump(cids, f)
    with open(output_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump(weights, f)

    print(f"Saved {len(terms)} terms to {output_dir}")
    print("\nSample relabeled terms:")
    for t in terms[:12]:
        print(f"  {t[:95]}{'...' if len(t) > 95 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
