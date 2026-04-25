#!/usr/bin/env python3
"""
Build decoder cache from NeuroSynth: (term, map) via NiMARE.
Downloads NeuroSynth data, converts to Dataset, runs association meta-analysis per term,
parcellates to pipeline atlas (Glasser+Tian, 392). Same format as NeuroQuery cache for merge.

Usage:
  python build_neurosynth_cache.py --output-dir data/neurosynth_cache [--max-terms 0] [--data-dir data]
  --max-terms 0 = all terms; set e.g. 100 for a quick test.
  --n-jobs 1 recommended (parallel can yield spurious failures; NiMARE/nilearn not thread-safe).
  --debug to print skip reasons for first terms (forces n_jobs=1).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _get_masker():
    from neurolab.parcellation import get_masker
    masker = get_masker(memory="nilearn_cache", verbose=0)
    masker.fit()
    return masker


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NeuroSynth (term, map) cache, Glasser+Tian 392.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir for term_maps.npz, term_vocab.pkl")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Same as --output-dir (for build_all_maps.py)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dir for NiMARE download (default: data/neurosynth_data)")
    parser.add_argument("--max-terms", type=int, default=0, help="Max terms (0 = all)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for term loop (e.g. 30)")
    parser.add_argument("--min-studies", type=int, default=5, help="Min studies per term (default 5)")
    parser.add_argument("--label-threshold", type=float, default=0.001, help="TF-IDF threshold for term presence (default 0.001)")
    parser.add_argument("--debug", action="store_true", help="Print failure reason for each skipped term (forces n_jobs=1)")
    args = parser.parse_args()
    out_dir = args.output_dir or args.cache_dir
    if not out_dir:
        parser.error("Provide --output-dir or --cache-dir")

    try:
        from nimare.extract import fetch_neurosynth
        from nimare.io import convert_neurosynth_to_dataset
        from nimare.meta.cbma.mkda import MKDADensity
    except ImportError:
        print("Install nimare: pip install nimare", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir or _scripts.parent / "data" / "neurosynth_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching NeuroSynth data...")
    files = fetch_neurosynth(
        data_dir=str(data_dir),
        version="7",
        overwrite=False,
        source="abstract",
        vocab="terms",
    )
    neurosynth_db = files[0]
    print("Converting to NiMARE Dataset...")
    dset = convert_neurosynth_to_dataset(
        coordinates_file=neurosynth_db["coordinates"],
        metadata_file=neurosynth_db["metadata"],
        annotations_files=neurosynth_db["features"],
    )

    if not hasattr(dset, "annotations") or dset.annotations is None:
        print("Dataset has no annotations.", file=sys.stderr)
        return 1
    skip = {"id", "study_id", "contrast_id", "pmid", "doi"}
    term_cols = [c for c in dset.annotations.columns if c not in skip and not c.startswith("_")]
    if args.max_terms > 0:
        term_cols = term_cols[: args.max_terms]
    n_terms = len(term_cols)
    print(f"Running meta-analysis for {n_terms} terms...")

    from neurolab.parcellation import get_n_parcels
    n_parcels = get_n_parcels()
    n_jobs = 1 if args.debug else max(1, getattr(args, "n_jobs", 1))
    if n_jobs > 1:
        print("WARNING: n_jobs>1 can yield spurious failures (NiMARE/nilearn not thread-safe). Use n_jobs=1 for reliability.", file=sys.stderr)
    min_studies = getattr(args, "min_studies", 5)
    label_threshold = getattr(args, "label_threshold", 0.001)
    # Serial mode: reuse estimator and masker. Parallel: create fresh per-call (NiMARE/nilearn not thread-safe)
    estimator = MKDADensity(kernel__r=6, null_method="approximate") if n_jobs <= 1 else None
    masker = _get_masker() if n_jobs <= 1 else None

    def _process_one(term, debug=False):
        try:
            ids = dset.annotations[dset.annotations[term] > label_threshold]["id"].tolist()
            if len(ids) < min_studies:
                return (None, "too_few_studies" if debug else None)
            sub = dset.slice(ids)
            est = estimator if estimator is not None else MKDADensity(kernel__r=6, null_method="approximate")
            result = est.fit(sub)
            img = result.get_map("z", return_type="image")
            if img is None:
                return (None, "mkda_no_map" if debug else None)
            from neurolab.parcellation import resample_to_atlas
            img = resample_to_atlas(img)
            m = masker if masker is not None else _get_masker()
            parcel_vals = m.transform(img).ravel()
            if parcel_vals.size != n_parcels:
                return (None, f"parcel_mismatch_{parcel_vals.size}" if debug else None)
            return ((term.replace("_", " "), parcel_vals.astype(np.float32)), None)
        except Exception as e:
            return (None, f"error:{e!r}" if debug else None)

    maps_list = []
    success_terms = []
    if n_jobs <= 1:
        for i, term in enumerate(term_cols):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  {i + 1}/{n_terms} {term!r}")
            out, reason = _process_one(term, debug=args.debug)
            if out is not None:
                success_terms.append(out[0])
                maps_list.append(out[1])
            elif args.debug and (i < 20 or reason != "too_few_studies"):
                print(f"    SKIP {term!r}: {reason}")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_process_one, term, False): term for term in term_cols}
            for fut in as_completed(futures):
                out, _ = fut.result()
                if out is not None:
                    success_terms.append(out[0])
                    maps_list.append(out[1])
                done += 1
                if done % 50 == 0 or done == n_terms:
                    print(f"  {done}/{n_terms} terms done")

    if not maps_list:
        print("No maps produced.", file=sys.stderr)
        return 1

    term_maps = np.stack(maps_list)
    # Z-score each map across parcels (standardization for training consistency)
    from neurolab.neurovault_ingestion import zscore_maps
    term_maps = zscore_maps(term_maps, axis=1)
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(success_terms, f)
    print(f"Wrote {out_dir}: {len(success_terms)} terms, shape {term_maps.shape}")
    if len(success_terms) < n_terms:
        print(f"Skipped {n_terms - len(success_terms)} terms (too few studies or error).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
