#!/usr/bin/env python3
"""
Build pharmacological brain maps from NeuroSynth via NiMARE (CC0-derived).
Runs meta-analysis for a curated list of drug/compound terms, parcellates to
pipeline atlas (Glasser+Tian, 392). Same cache format as NeuroSynth/NeuroQuery
for merge. Commercially clean: derived from CC0 coordinate data.

Usage:
  python build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache
  python build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache --data-dir neurolab/data/neurosynth_data
  python build_pharma_neurosynth_cache.py --output-dir ... --terms ketamine caffeine  # only these (plus defaults)

Requires: nimare, neurolab.parcellation (same as build_neurosynth_cache.py).
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Generic methodological terms — weak supervision (no specific compound); exclude when --exclude-generic-terms
GENERIC_PHARMA_TERMS = frozenset({
    "dose", "double blind", "challenge", "inhibitor", "metabolism", "positron emission",
    "drugs", "dependence", "dopaminergic", "ampa", "drug", "pharmacolog", "receptor",
})

# Default pharmacological vocabulary (fallback when JSON not used)
DEFAULT_PHARMA_TERMS = [
    # Psychedelics
    "psilocybin",
    "LSD",
    "DMT",
    "ayahuasca",
    "mescaline",
    # Dissociatives
    "ketamine",
    "PCP",
    "nitrous oxide",
    # Stimulants
    "amphetamine",
    "methylphenidate",
    "cocaine",
    "caffeine",
    "nicotine",
    "modafinil",
    # Depressants
    "alcohol",
    "benzodiazepine",
    "barbiturate",
    # Antidepressants
    "SSRI",
    "fluoxetine",
    "escitalopram",
    "venlafaxine",
    "bupropion",
    # Antipsychotics
    "haloperidol",
    "risperidone",
    "clozapine",
    "olanzapine",
    "quetiapine",
    # Opioids
    "morphine",
    "fentanyl",
    "naloxone",
    "buprenorphine",
    # Cannabis
    "THC",
    "cannabidiol",
    "cannabis",
    # Nootropics / cognitive enhancers
    "piracetam",
    "aniracetam",
    "alpha-GPC",
    "citicoline",
    "acetyl-L-carnitine",
    "huperzine A",
    "donepezil",
    "memantine",
    "galantamine",
    "bacopa monnieri",
    "ashwagandha",
    "rhodiola rosea",
    "lion's mane",
    "ginkgo biloba",
    "L-theanine",
    "N-acetyl cysteine",
    "modafinil",
    "noopept",
    "sulbutiamine",
    "vinpocetine",
    # Others
    "MDMA",
    "propofol",
    "levodopa",
]


def _normalize_for_match(s: str) -> str:
    return re.sub(r"[_\s]+", "", s.lower())


def _get_masker():
    from neurolab.parcellation import get_masker
    masker = get_masker(memory="nilearn_cache", verbose=0)
    masker.fit()
    return masker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build pharmacological (term, map) cache from NeuroSynth NiMARE meta-analyses."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for term_maps.npz, term_vocab.pkl (e.g. neurolab/data/pharma_neurosynth_cache)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Alias for --output-dir (for build_all_maps.py compatibility).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="NiMARE/NeuroSynth data dir (default: neurolab/data/neurosynth_data). Reuses same data as build_neurosynth_cache.",
    )
    parser.add_argument(
        "--terms",
        nargs="*",
        default=None,
        help="Override: only run these terms (default: use built-in PHARMA_TERMS).",
    )
    parser.add_argument(
        "--min-studies",
        type=int,
        default=3,
        help="Skip term if number of studies with non-zero weight < this (default: 3).",
    )
    parser.add_argument(
        "--all-drug-columns",
        action="store_true",
        help="Include ALL NeuroSynth columns matching drug/pharma keywords (not just PHARMA_TERMS).",
    )
    parser.add_argument(
        "--pharma-terms-json",
        type=Path,
        default=None,
        help="Path to neurosynth_pharma_terms.json. If not set, uses neurolab/docs/implementation/neurosynth_pharma_terms.json when it exists.",
    )
    parser.add_argument(
        "--pharma-terms-key",
        choices=("all_terms_sorted", "high_confidence_terms"),
        default="high_confidence_terms",
        help="Key in JSON: all_terms_sorted (194) or high_confidence_terms (~80). Default: high_confidence_terms.",
    )
    parser.add_argument(
        "--exclude-generic-terms",
        action="store_true",
        default=False,
        help="Exclude generic methodological terms (dose, double blind, challenge, etc.) — keep compound names only.",
    )
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

    # Drug-related keywords for --all-drug-columns (NeuroSynth column substring match)
    DRUG_KEYWORDS = [
        "drug", "drugs", "alcohol", "caffeine", "nicotine", "cocaine", "amphetamine",
        "ketamine", "cannabis", "morphine", "methylphenidate", "modafinil", "psilocybin",
        "LSD", "THC", "cannabidiol", "SSRI", "fluoxetine", "bupropion", "haloperidol",
        "risperidone", "clozapine", "donepezil", "memantine", "levodopa", "propofol",
        "MDMA", "benzodiazepine", "opioid", "pharmacolog", "dopamine", "dopaminergic",
        "serotonin", "receptor",
    ]

    if args.terms is not None:
        terms_to_run = [t.strip() for t in args.terms if t.strip()]
    else:
        if args.pharma_terms_json:
            json_path = Path(args.pharma_terms_json).resolve()
        else:
            json_path = _repo_root / "neurolab" / "docs" / "implementation" / "neurosynth_pharma_terms.json"
        if json_path.exists():
            import json
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            terms_to_run = data.get(args.pharma_terms_key) or data.get("all_terms_sorted") or []
            terms_to_run = [str(t).strip() for t in terms_to_run if t]
            print(f"Loaded {len(terms_to_run)} terms from {json_path.name} ({args.pharma_terms_key})")
        else:
            terms_to_run = DEFAULT_PHARMA_TERMS
            print(f"Pharma terms JSON not found: {json_path}; using built-in DEFAULT_PHARMA_TERMS ({len(terms_to_run)} terms).")
    terms_to_run = [t.strip() for t in terms_to_run if t.strip()]

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
    all_cols = [c for c in dset.annotations.columns if c not in skip and not c.startswith("_")]
    # Match requested terms to NeuroSynth columns: any column whose normalized name contains
    # the normalized term (e.g. "caffeine" -> "terms_abstract_tfidf__caffeine").
    normalized_cols = {c: _normalize_for_match(c) for c in all_cols}
    matched = []
    if args.all_drug_columns:
        # Include ALL columns matching drug keywords
        for col in all_cols:
            ncol = normalized_cols[col]
            for kw in DRUG_KEYWORDS:
                if _normalize_for_match(kw) in ncol:
                    base_name = col.split("__")[-1] if "__" in col else col
                    matched.append((base_name.replace("_", " "), col))
                    break
    else:
        for term in terms_to_run:
            nterm = _normalize_for_match(term)
            for col, ncol in normalized_cols.items():
                if nterm in ncol or ncol in nterm:
                    matched.append((term, col))
                    break
            else:
                for col in all_cols:
                    if col.replace("_", " ").lower() == term.lower():
                        matched.append((term, col))
                        break
                else:
                    print(f"  No NeuroSynth column for {term!r}; skipping.")
    if not matched:
        print("No terms matched NeuroSynth vocabulary. Check PHARMA_TERMS or --terms.", file=sys.stderr)
        return 1

    # Deduplicate by column (keep first term label for that column)
    seen_col = set()
    unique = []
    for label, col in matched:
        if col not in seen_col:
            seen_col.add(col)
            if args.exclude_generic_terms:
                lab_norm = label.replace("_", " ").lower()
                if any(g in lab_norm for g in GENERIC_PHARMA_TERMS):
                    continue  # skip generic terms
            unique.append((label, col))
    n_terms = len(unique)
    print(f"Running meta-analysis for {n_terms} pharmacological terms...")

    from neurolab.parcellation import get_n_parcels
    masker = _get_masker()
    n_parcels = get_n_parcels()
    estimator = MKDADensity(kernel__r=6, null_method="approximate")
    min_studies = args.min_studies

    maps_list = []
    success_terms = []
    for i, (label, col) in enumerate(unique):
        print(f"  {i + 1}/{n_terms} {label!r} (column {col!r})")
        try:
            ids = dset.annotations[dset.annotations[col] > 0.001]["id"].tolist()
            if len(ids) < min_studies:
                print(f"    Skip: only {len(ids)} studies (need >= {min_studies})")
                continue
            sub = dset.slice(ids)
            result = estimator.fit(sub)
            img = result.get_map("z", return_type="image")
            if img is None:
                print(f"    Skip: no z map")
                continue
            from neurolab.parcellation import resample_to_atlas
            img = resample_to_atlas(img)
            parcel_vals = masker.transform(img).ravel()
            if parcel_vals.size != n_parcels:
                print(f"    Skip: parcellation size {parcel_vals.size} != {n_parcels}")
                continue
            maps_list.append(parcel_vals.astype(np.float32))
            success_terms.append(label.replace("_", " ") if label == col else label)
        except Exception as e:
            print(f"    Skip: {e}", file=sys.stderr)
            continue

    if not maps_list:
        print("No maps produced. Try --min-studies 3 or check NeuroSynth vocabulary.", file=sys.stderr)
        return 1

    term_maps = np.stack(maps_list)
    # Z-score each map across parcels (standardization for training consistency)
    from neurolab.neurovault_ingestion import zscore_maps
    term_maps = zscore_maps(term_maps, axis=1)
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(success_terms, f)
    # Higher sample weight for pharma terms (1.2) so they get more importance in training
    PHARMA_SAMPLE_WEIGHT = 1.2
    with open(out_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump([PHARMA_SAMPLE_WEIGHT] * len(success_terms), f)
    print(f"Wrote {out_dir}: {len(success_terms)} terms, shape {term_maps.shape} (sample weight {PHARMA_SAMPLE_WEIGHT})")
    if len(success_terms) < n_terms:
        print(f"Skipped {n_terms - len(success_terms)} terms (too few studies or error).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
