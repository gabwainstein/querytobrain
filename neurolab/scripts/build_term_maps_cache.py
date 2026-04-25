#!/usr/bin/env python3
"""
Phase 2: Build cognitive term maps cache for the decoder.

Generates a parcellated brain map for each NeuroQuery term, saves term_maps.npz
and term_vocab.pkl. First run: 1–2 hours (depends on vocabulary size). Subsequent
decode loads use the cache (< 1 s).

Usage (from querytobrain repo root):
  python neurolab/scripts/build_term_maps_cache.py
  python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0   # full vocab (default)
  python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 5000   # cap for quick test

Exit 0 = cache built and validated. Exit 1 = failure.
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import stats

# Ensure neurolab is importable (repo root = parent of neurolab)
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from neurolab.parcellation import get_masker, get_n_parcels, resample_to_atlas

from neuroquery import fetch_neuroquery_model, NeuroQueryModel

# Same as addendum: exclude generic/non-cognitive terms
EXCLUDE_TERMS = {
    "participants", "studies", "results", "analysis", "data",
    "significant", "compared", "group", "effects", "task",
    "conditions", "trials", "research", "findings", "methods",
    "study", "reported", "brain", "regions", "cortex",
    "activation", "activity", "increased", "decreased",
    "greater", "response", "left", "right", "bilateral",
    "anterior", "posterior", "dorsal", "ventral",
    "patients", "controls", "healthy", "clinical",
}


def main():
    parser = argparse.ArgumentParser(description="Build cognitive decoder term maps cache")
    parser.add_argument(
        "--cache-dir",
        default="neurolab/data/decoder_cache",
        help="Directory for term_maps.npz and term_vocab.pkl",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=0,
        help="Max number of terms to keep (by variance). 0 = no cap (full vocabulary, ~7.5K; recommended for training).",
    )
    parser.add_argument(
        "--min-term-length",
        type=int,
        default=3,
        help="Minimum term length (chars)",
    )
    parser.add_argument(
        "--parcellation",
        default="glasser_tian",
        choices=["glasser_tian"],
        help="Parcellation atlas",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (default 1; >1 may fail due to masker thread-safety — use 1 for reliable full rebuild)")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "term_maps.npz")
    vocab_path = os.path.join(cache_dir, "term_vocab.pkl")

    print("Phase 2: Building cognitive term maps cache")
    print(f"  cache_dir = {cache_dir}")
    print(f"  max_terms = {args.max_terms or 'all'}")

    # Load model and parcellation
    print("\n[1/4] Loading NeuroQuery model...")
    t0 = time.time()
    model_path = fetch_neuroquery_model()
    model = NeuroQueryModel.from_data_dir(model_path)
    full_vocab = list(model.vectorizer.get_feature_names())
    print(f"  Vocabulary size: {len(full_vocab)} ({time.time() - t0:.1f}s)")

    print("\n[2/4] Loading parcellation (combined cortical+subcortical when available)...")
    masker = get_masker()
    n_parcels = get_n_parcels()
    masker.fit()
    print(f"  Parcels: {n_parcels}")

    # Filter vocabulary
    filtered_vocab = [
        term for term in full_vocab
        if (
            len(term) >= args.min_term_length
            and term.lower() not in EXCLUDE_TERMS
            and not term.isnumeric()
        )
    ]
    print(f"  Filtered: {len(full_vocab)} -> {len(filtered_vocab)} terms")

    # Build term maps (optionally in parallel)
    n_jobs = max(1, getattr(args, "n_jobs", 1))
    print(f"\n[3/4] Generating and parcellating brain maps (n_jobs={n_jobs})...")

    _masker_lock = threading.Lock() if n_jobs > 1 else None

    def _process_one(term):
        try:
            result = model.transform([term])
            brain_map = result["brain_map"][0]
            if hasattr(brain_map, "get_fdata"):
                brain_img = brain_map
            elif isinstance(brain_map, (str, os.PathLike)):
                brain_img = nib.load(brain_map)
            else:
                brain_img = nib.Nifti1Image(np.asarray(brain_map), np.eye(4))
            brain_img = resample_to_atlas(brain_img)
            if _masker_lock:
                with _masker_lock:
                    parcellated = masker.transform(brain_img).ravel()
            else:
                parcellated = masker.transform(brain_img).ravel()
            # NiftiLabelsMasker may drop labels (e.g. 283) when resampling atlas to data; pad to n_parcels
            if parcellated.shape[0] == n_parcels - 1:
                parcellated = np.concatenate([parcellated[:282], [0.0], parcellated[282:]])  # label 283 missing
            if parcellated.shape[0] != n_parcels or np.sum(np.abs(parcellated)) < 1e-6:
                return (term, None)
            return (term, parcellated.astype(np.float64))
        except Exception:
            return (term, None)

    term_maps_list = []
    valid_vocab = []
    t_start = time.time()
    if n_jobs <= 1:
        for i, term in enumerate(filtered_vocab):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  {i + 1}/{len(filtered_vocab)} ({time.time() - t_start:.0f}s elapsed)")
            _, arr = _process_one(term)
            if arr is not None:
                term_maps_list.append(arr)
                valid_vocab.append(term)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_process_one, term): term for term in filtered_vocab}
            for fut in as_completed(futures):
                term, arr = fut.result()
                if arr is not None:
                    term_maps_list.append(arr)
                    valid_vocab.append(term)
                done += 1
                if done % 500 == 0 or done == len(filtered_vocab):
                    print(f"  {done}/{len(filtered_vocab)} ({time.time() - t_start:.0f}s elapsed)")
    n_skip = len(filtered_vocab) - len(valid_vocab)

    if not term_maps_list:
        print("  FAIL: No term maps produced.", file=sys.stderr)
        sys.exit(1)

    term_maps = np.array(term_maps_list)
    vocabulary = valid_vocab
    print(f"  Built {term_maps.shape[0]} term maps (skipped {n_skip})")

    # Optional: cap by variance
    if args.max_terms and len(vocabulary) > args.max_terms:
        variances = np.var(term_maps, axis=1)
        top_idx = np.argsort(variances)[-args.max_terms:]
        term_maps = term_maps[top_idx]
        vocabulary = [vocabulary[i] for i in top_idx]
        print(f"  Capped to top {args.max_terms} by variance")

    # Z-score each map across parcels (standardization for training consistency)
    term_maps = np.nan_to_num(stats.zscore(term_maps, axis=1), nan=0.0, posinf=0.0, neginf=0.0)

    # Save (term_maps is now standardized)
    print("\n[4/4] Saving cache (z-scored)...")
    np.savez_compressed(cache_path, term_maps=term_maps)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocabulary, f)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  {cache_path} ({size_mb:.1f} MB)")
    print(f"  {vocab_path}")

    # Validate
    print("\nValidating...")
    data = np.load(cache_path)
    tm = data["term_maps"]
    with open(vocab_path, "rb") as f:
        voc = pickle.load(f)
    assert tm.shape[1] == n_parcels, f"shape[1] {tm.shape[1]} != {n_parcels}"
    assert len(voc) == tm.shape[0], "vocab len != term_maps shape[0]"
    bad = np.any(np.isnan(tm), axis=1) | (np.sum(np.abs(tm), axis=1) < 1e-10)
    assert not np.all(bad), "All rows are NaN or zero"
    print(f"  OK: {tm.shape[0]} terms × {tm.shape[1]} parcels")
    print(f"  Total time: {time.time() - t_start:.0f}s")
    print("\nPhase 2 done. Ready for Phase 3 (CognitiveDecoder class).")


if __name__ == "__main__":
    main()
