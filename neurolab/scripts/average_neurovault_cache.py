#!/usr/bin/env python3
"""
Average AVERAGE_FIRST collections in an existing NeuroVault cache.

Uses already parcellated and normalized maps — no re-parcellation from raw images.

**Logic:** Extract only the cognitive condition from labels (strip subject ID, image ID,
IAPS, etc.). Group by condition, average within each, then output contrasts of averages
(e.g. all positive -> avg_pos; all neutral -> avg_neu; contrast = avg_pos - avg_neu).

Collections 503 (PINES), 504 (Pain), 16284 (IAPS valence) use condition-based contrasts.
Others: group by exact label, average when 2+ maps.

By default (--min-subjects 1) keeps all groups. Use --min-subjects 3 to drop small groups.

Usage:
  python neurolab/scripts/average_neurovault_cache.py --cache-dir neurolab/data/neurovault_cache
  python neurolab/scripts/average_neurovault_cache.py --min-subjects 1  # preserve all (default)
  python neurolab/scripts/average_neurovault_cache.py --min-subjects 3  # drop small groups
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from neurolab.neurovault_ingestion import (
    AVERAGE_FIRST,
    IAPS_NEGATIVE_IDS,
    IAPS_NEUTRAL_IDS,
    MIN_SUBJECTS_HETEROGENEOUS,
    MIN_SUBJECTS_PER_CONTRAST,
    OUTLIER_CORR_THRESHOLD,
    get_sample_weight,
)


def get_cognitive_condition(label: str, collection_id: int) -> str | None:
    """Extract cognitive condition only. Strips subject ID, image ID, IAPS, etc.
    Handles both space and underscore variants (NeuroVault name vs filename/path stem)."""
    if not label:
        return None
    # Strip [col123] prefix if present (from improve_neurovault_labels or strip_col_prefix)
    label = re.sub(r"^\[col\d+\]\s*", "", str(label)).strip()
    # 503 (PINES): "IAPS Subject 102 Image 9921" or "IAPS_Subject_102_Image_9921" -> valence from image ID
    if collection_id == 503:
        m = re.search(r"IAPS[_ ]Subject[_ ]\d+[_ ]Image[_ ](\d+)", label, re.I)
        if m:
            img_id = int(m.group(1))
            if img_id in IAPS_NEGATIVE_IDS:
                return "negative"
            if img_id in IAPS_NEUTRAL_IDS:
                return "neutral"
        # Also handle pre-grouped "IAPS Image 9921" from ingest
        m2 = re.search(r"IAPS[_ ]Image[_ ](\d+)", label, re.I)
        if m2:
            img_id = int(m2.group(1))
            if img_id in IAPS_NEGATIVE_IDS:
                return "negative"
            if img_id in IAPS_NEUTRAL_IDS:
                return "neutral"
        return None
    # 504 (Pain): "Pain Subject 10 High" or "Pain_Subject_10_High" -> high, low, medium
    if collection_id == 504:
        m = re.search(r"Pain[_ ]Subject[_ ]\d+[_ ](High|Low|Medium)", label, re.I)
        if m:
            return m.group(1).lower()
        return None
    # 16284 (IAPS valence): "sub001 positive" or "sub001_positive" -> positive, negative, neutral
    if collection_id == 16284:
        m = re.search(r"sub\d+[_ ](positive|negative|neutral)", label, re.I)
        if m:
            return m.group(1).lower()
        return None
    # 3324 (Kragel): "Study01Subject01" -> "Study 1" (average 15 subjects per study)
    if collection_id == 3324:
        m = re.search(r"Study(\d+)Subject\d+", label, re.I)
        if m:
            return f"Study {int(m.group(1))}"
        return None
    # 16266 (Emotion regulation): "PIP Subject0001 LookNeg Beta" or "AHAB Subject0001 LookNeg Beta" -> "LookNeg" (condition)
    if collection_id == 16266:
        m = re.search(r"(?:PIP|AHAB)[_ ]Subject\d+[_ ](\w+)", label, re.I)
        if m:
            return m.group(1)
        return None
    return None


def get_contrast_key_from_label(label: str, collection_id: int) -> str:
    """Grouping key: cognitive condition if available, else label."""
    cond = get_cognitive_condition(label, collection_id)
    if cond is not None:
        return cond
    return label or f"collection_{collection_id}_unknown"


def _average_with_outliers(arr_list: list[np.ndarray], min_n: int) -> np.ndarray | None:
    """Average maps with outlier removal. Returns None if too few."""
    if len(arr_list) < min_n:
        return None
    arr = np.array(arr_list)
    if len(arr) == 1:
        return arr[0].astype(np.float64)
    mean_map = arr.mean(axis=0)
    keep = []
    for j in range(len(arr)):
        r = np.corrcoef(arr[j], mean_map)[0, 1] if arr[j].std() > 1e-8 else 1.0
        if not np.isnan(r) and r >= OUTLIER_CORR_THRESHOLD:
            keep.append(arr[j])
    if len(keep) >= max(2, min_n - 1) if min_n > 1 else len(keep) >= 1:
        return np.mean(keep, axis=0).astype(np.float64)
    return mean_map.astype(np.float64)


def _condition_contrasts(
    groups: dict[str, list[np.ndarray]],
    contrasts: list[tuple[str, str]],
    min_n: int,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """Average each condition, output contrasts of averages (A - B)."""
    cond_avgs = {}
    for cond, arr_list in groups.items():
        avg = _average_with_outliers(arr_list, min_n)
        if avg is not None:
            cond_avgs[cond] = avg
    out = {}
    for a, b in contrasts:
        if a in cond_avgs and b in cond_avgs:
            out[f"{prefix}{a} vs {b}"] = (cond_avgs[a] - cond_avgs[b]).astype(np.float64)
    return out


# Collections where we group by cognitive condition, average, then output contrasts
# (contrasts, prefix, optional label override per contrast)
CONDITION_CONTRASTS: dict[int, tuple[list[tuple[str, str]], str, dict | None]] = {
    503: ([("negative", "neutral")], "IAPS ", {"IAPS negative vs neutral": "IAPS negative vs neutral (PINES)"}),
    504: ([("high", "low"), ("high", "medium"), ("medium", "low")], "pain intensity ", None),
    16284: ([("positive", "neutral"), ("negative", "neutral")], "emotional valence ", None),
}


def process_condition_collection(
    groups: dict[str, list[np.ndarray]], collection_id: int, min_n: int
) -> dict[str, np.ndarray] | None:
    """If collection has condition-based contrasts, return them; else None."""
    if collection_id not in CONDITION_CONTRASTS:
        return None
    contrasts, prefix, label_override = CONDITION_CONTRASTS[collection_id]
    result = _condition_contrasts(groups, contrasts, min_n, prefix)
    if label_override:
        result = {label_override.get(k, k): v for k, v in result.items()}
    return result


def average_groups(
    groups: dict[str, list[np.ndarray]], min_n: int
) -> dict[str, np.ndarray]:
    """Average maps per group. min_n=1 keeps all; min_n>=2 drops groups with fewer maps."""
    out = {}
    for key, arr_list in groups.items():
        if len(arr_list) < min_n:
            continue
        arr = np.array(arr_list)
        if len(arr) == 1:
            out[key] = arr[0].astype(np.float64)
            continue
        mean_map = arr.mean(axis=0)
        keep = []
        for j in range(len(arr)):
            r = np.corrcoef(arr[j], mean_map)[0, 1] if arr[j].std() > 1e-8 else 1.0
            if not np.isnan(r) and r >= OUTLIER_CORR_THRESHOLD:
                keep.append(arr[j])
        if len(keep) >= max(2, min_n - 1) if min_n > 1 else len(keep) >= 1:
            out[key] = np.mean(keep, axis=0).astype(np.float64)
        else:
            out[key] = mean_map.astype(np.float64)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average AVERAGE_FIRST collections in existing NeuroVault cache (no re-parcellation)"
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "neurolab" / "data" / "neurovault_cache"),
        help="Input cache directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: overwrite cache-dir)",
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=1,
        help="Min maps per contrast to keep (1=preserve all, 3=drop small groups). Default 1.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    terms = list(pickle.load(open(cache_dir / "term_vocab.pkl", "rb")))
    cids = list(pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb")))
    if (cache_dir / "term_sample_weights.pkl").exists():
        weights = list(pickle.load(open(cache_dir / "term_sample_weights.pkl", "rb")))
    else:
        weights = [1.0] * len(terms)

    n_before = len(terms)
    print(f"Loaded {n_before} terms from {cache_dir}")

    # Split: AVERAGE_FIRST vs keep-as-is
    avg_indices = [i for i in range(n_before) if cids[i] in AVERAGE_FIRST]
    keep_indices = [i for i in range(n_before) if cids[i] not in AVERAGE_FIRST]

    new_terms = [terms[i] for i in keep_indices]
    new_maps = [term_maps[i] for i in keep_indices]
    new_cids = [cids[i] for i in keep_indices]
    new_weights = [weights[i] for i in keep_indices]

    # Group AVERAGE_FIRST by (collection_id, contrast_key)
    by_collection: dict[int, dict[str, list[tuple[int, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i in avg_indices:
        cid = cids[i]
        key = get_contrast_key_from_label(terms[i], cid)
        by_collection[cid][key].append((i, term_maps[i].astype(np.float64)))

    # Average per collection
    min_n = args.min_subjects
    for cid in sorted(by_collection.keys()):
        groups = {
            k: [m for _, m in v]
            for k, v in by_collection[cid].items()
        }
        cond_result = process_condition_collection(groups, cid, min_n)
        if cond_result is not None and cond_result:
            averaged = cond_result
        else:
            averaged = average_groups(groups, min_n)

        w = get_sample_weight(cid, was_averaged=True)
        for label, vec in averaged.items():
            new_terms.append(label)
            new_maps.append(vec)
            new_cids.append(cid)
            new_weights.append(w)

    n_after = len(new_terms)
    term_maps_out = np.array(new_maps, dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "term_maps.npz", term_maps=term_maps_out)
    with open(output_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(new_terms, f)
    with open(output_dir / "term_collection_ids.pkl", "wb") as f:
        pickle.dump(new_cids, f)
    with open(output_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump(new_weights, f)
    # Provenance: collections from averaged path get was_averaged=True
    collection_was_averaged = {cid: True for cid in by_collection.keys()}
    for cid in set(cids[i] for i in keep_indices):
        collection_was_averaged.setdefault(cid, False)
    with open(output_dir / "collection_provenance.json", "w") as f:
        json.dump({str(k): {"was_averaged": v} for k, v in sorted(collection_was_averaged.items())}, f, indent=2)

    print(f"Saved {n_after} terms to {output_dir}")
    print(f"  Kept {len(keep_indices)} as-is, averaged {n_before - len(keep_indices)} -> {n_after - len(keep_indices)}")

    # Sample
    col503 = [(t, c) for t, c in zip(new_terms, new_cids) if c == 503]
    if col503:
        print(f"  Collection 503 (PINES): {len(col503)} term(s): {[t for t, _ in col503]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
