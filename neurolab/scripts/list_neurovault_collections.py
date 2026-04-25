#!/usr/bin/env python3
"""List NeuroVault collections by preprocessing status."""
import json
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def main():
    from neurolab.neurovault_ingestion import AVERAGE_FIRST

    manifest = REPO_ROOT / "neurolab/data/neurovault_curated_data/manifest.json"
    cache_dir = REPO_ROOT / "neurolab/data/neurovault_cache"

    if not manifest.exists():
        print("Manifest not found")
        return
    m = json.load(open(manifest))
    images = m.get("images", m)
    curated = set(i.get("collection_id") for i in images) if isinstance(images, list) else set()

    in_cache = {}
    if (cache_dir / "term_collection_ids.pkl").exists():
        cids = pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb"))
        from collections import Counter
        in_cache = dict(Counter(cids))

    names = {
        503: "PINES/IAPS", 504: "Pain NPS", 16284: "IAPS valence",
        426: "False belief ToM", 445: "Why/How ToM", 507: "Consensus decision",
        1952: "BrainPedia", 6618: "IBC 2nd", 2138: "IBC 1st", 4343: "UCLA LA5C",
        2503: "Social Bayesian", 4804: "Fusiform-network", 13042: "Language/WM epilepsy",
        13705: "Tier3 language", 2108: "Tier3", 4683: "Tier3", 3887: "Tier3",
        1516: "Tier3", 13474: "Tensorial ICA MDD", 20510: "Lesions psychosis",
        11646: "ASD emotional egocentricity", 437: "Autism subnetworks",
        12992: "Nicotine abstinence", 19012: "Incentive-boosted inhibitory",
        6825: "Schizophrenia deformation", 1620: "Depression resting-state",
    }

    need_prep = sorted(curated & AVERAGE_FIRST)
    contrast_ready = sorted(curated - AVERAGE_FIRST)

    print("=" * 70)
    print("NEED PREPROCESSING (AVERAGE_FIRST — subject-level, average by contrast)")
    print("=" * 70)
    for cid in need_prep:
        nimg = sum(1 for i in images if i.get("collection_id") == cid)
        nc = in_cache.get(cid, 0)
        status = "DONE (in cache)" if nc else "MISSING — rebuild cache, then run average_neurovault_cache"
        print(f"  {cid:5} | {names.get(cid, '?'):30} | {nimg:5} images | {nc:4} terms | {status}")

    print()
    print("=" * 70)
    print("CONTRAST-READY (group-level, use as-is)")
    print("=" * 70)
    print(f"  {len(contrast_ready)} collections, {sum(in_cache.get(c, 0) for c in contrast_ready)} total terms in cache")
    for cid in contrast_ready[:20]:
        nc = in_cache.get(cid, 0)
        print(f"  {cid:5} | {nc:4} terms")
    if len(contrast_ready) > 20:
        print(f"  ... and {len(contrast_ready) - 20} more")

if __name__ == "__main__":
    main()
