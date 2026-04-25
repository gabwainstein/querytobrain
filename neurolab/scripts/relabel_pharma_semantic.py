#!/usr/bin/env python3
"""
Convert pharma cache labels to natural-language semantic descriptions.

Uses semantic_label_by_collection and abbreviation_expansions from
neurovault_pharma_schema.json. Produces labels optimized for general-purpose
embeddings (e.g. OpenAI) that respond better to natural language than
technical key-value pairs.

Usage:
  python neurolab/scripts/relabel_pharma_semantic.py --cache-dir neurolab/data/neurovault_pharma_cache_improved --output-dir neurolab/data/neurovault_pharma_cache_semantic
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "neurolab" / "data" / "neurovault_pharma_schema.json"


def _expand_abbreviations(text: str, expansions: dict[str, str]) -> str:
    """Replace abbreviations with full phrases (case-sensitive, whole-word)."""
    out = text
    for abbr, full in sorted(expansions.items(), key=lambda x: -len(x[0])):
        # Whole-word match
        out = re.sub(rf"\b{re.escape(abbr)}\b", full, out)
    return out


def _extract_contrast(label: str) -> str:
    """Extract the contrast/measure part from improved-style label."""
    # Strip " | drug=X | placebo-controlled" suffix
    if " | " in label:
        label = label.split(" | ")[0].strip()
    # Strip leading "drug: " or "LSD: " etc.
    if ": " in label:
        parts = label.split(": ", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return label


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
        description="Convert pharma labels to natural-language semantic descriptions"
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "neurolab" / "data" / "neurovault_pharma_cache_improved"),
        help="Input pharma cache (improved or relabeled)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "neurolab" / "data" / "neurovault_pharma_cache_semantic"),
        help="Output directory",
    )
    parser.add_argument(
        "--schema",
        default=str(SCHEMA_PATH),
        help="Path to neurovault_pharma_schema.json",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    output_dir = Path(args.output_dir)
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
    semantic_map = schema.get("semantic_label_by_collection") or {}
    semantic_map = {int(k): v for k, v in semantic_map.items()}
    expansions = schema.get("abbreviation_expansions") or {}

    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    cids = pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb"))
    if (cache_dir / "term_sample_weights.pkl").exists():
        weights = pickle.load(open(cache_dir / "term_sample_weights.pkl", "rb"))
    else:
        weights = [1.0] * len(terms)

    new_terms = []
    for lab, cid in zip(terms, cids):
        base = semantic_map.get(cid, "Pharmacological fMRI contrast")
        contrast = _extract_contrast(lab)
        contrast_expanded = _expand_abbreviations(contrast, expansions)
        if contrast_expanded and contrast_expanded != contrast:
            new_lab = f"{base}: {contrast_expanded}"
        elif contrast:
            new_lab = f"{base}: {contrast}"
        else:
            new_lab = base
        new_terms.append(new_lab)

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
    print("\nSample semantic labels:")
    for t in terms[:15]:
        print(f"  {t[:95]}{'...' if len(t) > 95 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
