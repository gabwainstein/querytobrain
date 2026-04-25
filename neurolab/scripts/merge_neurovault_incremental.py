#!/usr/bin/env python3
"""
Incrementally add/update NeuroVault collections without re-parcellating everything.

Builds only the specified collections (with P-map transform, min_subjects, ROI fixes),
then merges them into the existing neurovault_cache (replacing terms from those collections).

Usage:
  python neurolab/scripts/merge_neurovault_incremental.py
  python neurolab/scripts/merge_neurovault_incremental.py --collections 426 437 507 555 2485 3434

Then run improve_neurovault_labels and rebuild merged_sources.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
_data = _repo_root / "neurolab" / "data"
_scripts = _repo_root / "neurolab" / "scripts"

# Collections we fixed: min_subjects (426,507), P-map (555), ROI (2485,3434)
# 2508 excluded: ROI masks (ACC/DLPFC/Striatum), not activation maps
# 437 excluded: autism subnetworks (graph-derived), not activation contrasts
EXCLUDE_FROM_CACHE_COLLECTION_IDS = {2508, 437}
DEFAULT_COLLECTIONS = [426, 507, 555, 2485, 3434]


def main() -> int:
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    ap = argparse.ArgumentParser(description="Incrementally add fixed NeuroVault collections to cache")
    ap.add_argument("--collections", type=int, nargs="+", default=DEFAULT_COLLECTIONS,
                    help=f"Collection IDs to rebuild (default: {DEFAULT_COLLECTIONS})")
    ap.add_argument("--cache-dir", type=Path, default=_data / "neurovault_cache",
                    help="Existing NeuroVault cache to merge into")
    ap.add_argument("--data-dir", type=Path, default=_data / "neurovault_curated_data",
                    help="NeuroVault curated data (manifest + downloads)")
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel parcellation jobs")
    ap.add_argument("--no-fetch-metadata", action="store_true", help="Skip API fetch for missing labels")
    args = ap.parse_args()

    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = _repo_root / cache_dir
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = _repo_root / data_dir

    if not (cache_dir / "term_maps.npz").exists():
        print(f"Existing cache not found: {cache_dir / 'term_maps.npz'}")
        print("Run full build first: python neurolab/scripts/build_neurovault_cache.py ... --average-subject-level")
        return 1
    if not (data_dir / "manifest.json").exists():
        print(f"Manifest not found: {data_dir / 'manifest.json'}")
        return 1

    add_dir = cache_dir.parent / "neurovault_cache_additions"
    add_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build only the fixed collections
    coll_args = [str(c) for c in args.collections]
    build_args = [
        sys.executable, str(_scripts / "build_neurovault_cache.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(add_dir),
        "--average-subject-level",
        "--collections", *coll_args,
        "--n-jobs", str(args.n_jobs),
    ]
    if args.no_fetch_metadata:
        build_args += ["--no-fetch-missing-metadata"]
    print(f"\n{'='*60}\n  Building collections {args.collections} only (~{len(coll_args)} collections)\n  {' '.join(build_args)}\n{'='*60}")
    r = subprocess.run(build_args, cwd=str(_repo_root))
    if r.returncode != 0:
        print("Build failed.")
        return 1

    if not (add_dir / "term_maps.npz").exists():
        print("No output from build (all collections may have failed).")
        return 1

    # 2. Load existing cache
    existing = np.load(cache_dir / "term_maps.npz")
    term_maps = existing["term_maps"].copy()
    terms = list(pickle.load(open(cache_dir / "term_vocab.pkl", "rb")))
    collection_ids = list(pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb")))
    weights = list(pickle.load(open(cache_dir / "term_sample_weights.pkl", "rb")))
    prov = {}
    if (cache_dir / "collection_provenance.json").exists():
        prov = json.loads((cache_dir / "collection_provenance.json").read_text())

    coll_set = set(args.collections) | EXCLUDE_FROM_CACHE_COLLECTION_IDS
    n_parcels = term_maps.shape[1]

    # 3. Remove terms from the collections we're replacing (incl. excluded ROI/mask collections)
    removed = sum(1 for c in collection_ids if c in coll_set)
    keep = [i for i in range(len(terms)) if collection_ids[i] not in coll_set]
    terms = [terms[i] for i in keep]
    term_maps = term_maps[keep]
    collection_ids = [collection_ids[i] for i in keep]
    weights = [weights[i] for i in keep]
    excl = coll_set & EXCLUDE_FROM_CACHE_COLLECTION_IDS
    if excl:
        print(f"Removed {removed} terms from collections {list(coll_set)} (incl. excluded {excl})")
    else:
        print(f"Removed {removed} terms from collections {args.collections}")

    # 4. Merge: append new terms
    new_maps = np.load(add_dir / "term_maps.npz")["term_maps"]
    new_terms = pickle.load(open(add_dir / "term_vocab.pkl", "rb"))
    new_cids = pickle.load(open(add_dir / "term_collection_ids.pkl", "rb"))
    new_weights = pickle.load(open(add_dir / "term_sample_weights.pkl", "rb"))
    if new_maps.shape[1] != n_parcels:
        print(f"Parcel mismatch: existing {n_parcels}, new {new_maps.shape[1]}. Re-run full build.")
        return 1

    terms.extend(new_terms)
    term_maps = np.vstack([term_maps, new_maps.astype(term_maps.dtype)])
    collection_ids.extend(new_cids)
    weights.extend(new_weights)
    from neurolab.neurovault_ingestion import AVERAGE_FIRST
    for c in set(new_cids):
        prov[str(c)] = {"was_averaged": c in AVERAGE_FIRST}

    # 5. Save
    np.savez_compressed(cache_dir / "term_maps.npz", term_maps=term_maps)
    with open(cache_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    with open(cache_dir / "term_collection_ids.pkl", "wb") as f:
        pickle.dump(collection_ids, f)
    with open(cache_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump(weights, f)
    with open(cache_dir / "collection_provenance.json", "w") as f:
        json.dump(dict(sorted(prov.items(), key=lambda x: (int(x[0]) if x[0].isdigit() else 0, x[0]))), f, indent=2)

    # Merge collection_metadata (name, description) for LLM relabeling
    coll_meta = {}
    if (cache_dir / "collection_metadata.json").exists():
        coll_meta = json.loads((cache_dir / "collection_metadata.json").read_text())
    if (add_dir / "collection_metadata.json").exists():
        coll_meta.update(json.loads((add_dir / "collection_metadata.json").read_text()))
    if coll_meta:
        with open(cache_dir / "collection_metadata.json", "w") as f:
            json.dump(dict(sorted(coll_meta.items(), key=lambda x: (int(x[0]) if x[0].isdigit() else 0, x[0]))), f, indent=2)

    print(f"Merged: {len(terms)} terms total (+{len(new_terms)} from collections {args.collections})")
    print(f"Saved to {cache_dir}")

    # 6. Improve labels
    imp_args = [sys.executable, str(_scripts / "improve_neurovault_labels.py"), "--cache-dir", str(cache_dir)]
    if args.no_fetch_metadata:
        imp_args += ["--no-fetch-metadata"]
    print(f"\n{'='*60}\n  Improving labels\n{'='*60}")
    subprocess.run(imp_args, cwd=str(_repo_root))

    # 7. Rebuild merged_sources
    if (_data / "unified_cache" / "term_maps.npz").exists():
        merge_args = [
            sys.executable, str(_scripts / "build_expanded_term_maps.py"),
            "--cache-dir", str(_data / "unified_cache"),
            "--output-dir", str(_data / "merged_sources"),
            "--neurovault-cache-dir", str(cache_dir),
            "--no-ontology", "--save-term-sources",
        ]
        if (_data / "neuromaps_cache" / "annotation_maps.npz").exists():
            merge_args += ["--neuromaps-cache-dir", str(_data / "neuromaps_cache")]
        if (_data / "neurovault_pharma_cache" / "term_maps.npz").exists():
            merge_args += ["--neurovault-pharma-cache-dir", str(_data / "neurovault_pharma_cache")]
        if (_data / "pharma_neurosynth_cache" / "term_maps.npz").exists():
            merge_args += ["--pharma-neurosynth-cache-dir", str(_data / "pharma_neurosynth_cache")]
        if (_data / "enigma_cache" / "term_maps.npz").exists():
            merge_args += ["--enigma-cache-dir", str(_data / "enigma_cache")]
        if (_data / "abagen_cache" / "term_maps.npz").exists():
            merge_args += ["--abagen-cache-dir", str(_data / "abagen_cache"), "--max-abagen-terms", "500",
                          "--abagen-add-gradient-pcs", "3", "--add-pet-residuals"]
        print(f"\n{'='*60}\n  Rebuilding merged_sources\n{'='*60}")
        subprocess.run(merge_args, cwd=str(_repo_root))

    # Cleanup temp
    if add_dir.exists():
        shutil.rmtree(add_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
