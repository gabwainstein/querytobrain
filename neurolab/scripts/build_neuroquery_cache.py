#!/usr/bin/env python3
"""
Build decoder cache from NeuroQuery: (term, map) for each term in the model vocabulary.
Maps are parcellated to pipeline atlas (Glasser+Tian, 392). Uses neuroquery + nilearn.

Usage:
  python build_neuroquery_cache.py --output-dir data/neuroquery_cache [--max-terms 0]
  --max-terms 0 = all ~7547 terms; set e.g. 500 for a quick test.
"""

from __future__ import annotations

import argparse
import pickle
import sys
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
    parser = argparse.ArgumentParser(description="Build NeuroQuery (term, map) cache, Glasser+Tian 392.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir for term_maps.npz, term_vocab.pkl")
    parser.add_argument("--max-terms", type=int, default=0, help="Max terms to build (0 = all)")
    parser.add_argument("--neuroquery-data-dir", type=Path, default=None, help="NeuroQuery model dir (default: fetch)")
    args = parser.parse_args()

    try:
        from neuroquery import fetch_neuroquery_model, NeuroQueryModel
    except ImportError:
        print("Install neuroquery: pip install neuroquery", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.neuroquery_data_dir and Path(args.neuroquery_data_dir).exists():
        model_dir = str(args.neuroquery_data_dir)
    else:
        model_dir = fetch_neuroquery_model()
    encoder = NeuroQueryModel.from_data_dir(model_dir)

    vocab_path = Path(model_dir) / "vocabulary.csv"
    if not vocab_path.exists():
        print(f"Vocabulary not found: {vocab_path}", file=sys.stderr)
        return 1
    import pandas as pd
    vocab_df = pd.read_csv(vocab_path, header=None)
    terms = vocab_df.iloc[:, 0].astype(str).str.strip().tolist()
    if args.max_terms > 0:
        terms = terms[: args.max_terms]
    n_terms = len(terms)
    print(f"Building cache for {n_terms} terms...")

    from neurolab.parcellation import get_n_parcels
    masker = _get_masker()
    n_parcels = get_n_parcels()
    maps_list = []
    success_terms = []
    for i, term in enumerate(terms):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i + 1}/{n_terms} {term!r}")
        try:
            result = encoder(term)
            brain_map = result.get("brain_map")
            if brain_map is None:
                continue
            if isinstance(brain_map, (list, tuple)) and brain_map:
                brain_map = brain_map[0]
            from neurolab.parcellation import resample_to_atlas
            import nibabel as nib
            if hasattr(brain_map, "get_fdata"):
                brain_img = brain_map
            elif isinstance(brain_map, (str, Path)) and Path(brain_map).exists():
                brain_img = nib.load(brain_map)
            else:
                brain_img = nib.Nifti1Image(np.asarray(brain_map), np.eye(4))
            brain_img = resample_to_atlas(brain_img)
            parcel_vals = masker.transform(brain_img).ravel()
            if parcel_vals.size != n_parcels:
                continue
            maps_list.append(parcel_vals.astype(np.float32))
            success_terms.append(term)
        except Exception as e:
            if (i + 1) <= 5:
                print(f"    Skip {term!r}: {e}", file=sys.stderr)
            continue

    if not maps_list:
        print("No maps produced.", file=sys.stderr)
        return 1

    term_maps = np.stack(maps_list)
    assert len(success_terms) == term_maps.shape[0], (len(success_terms), term_maps.shape)

    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(success_terms, f)
    print(f"Wrote {out_dir}: {len(success_terms)} terms, shape {term_maps.shape}")
    if len(success_terms) < n_terms:
        print(f"Skipped {n_terms - len(success_terms)} terms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
