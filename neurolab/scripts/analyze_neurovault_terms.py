#!/usr/bin/env python3
"""
Deep analysis of NeuroVault cache term labels. Identifies:
- Atlas/structural collections to exclude
- Collections with poor contrast definitions
- Term quality heuristics for improvement
"""
import pickle
import re
import json
import urllib.request
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
NV_CACHE = REPO_ROOT / "neurolab" / "data" / "neurovault_cache"
API_BASE = "https://neurovault.org/api"
HEADERS = {"Accept": "application/json", "User-Agent": "NeuroLab/1.0"}

# Collections that are atlas/structural, NOT fMRI task contrasts (exclude from main cache)
ATLAS_COLLECTION_IDS = {262, 264, 1625, 6074, 9357}  # Harvard-Oxford, JHU DTI, Brainnetome, brainstem, APOE

def fetch_collection_metadata(cid: int, timeout: float = 15.0) -> dict:
    """Fetch collection name/description from NeuroVault API."""
    try:
        req = urllib.request.Request(f"{API_BASE}/collections/{cid}/", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}

def strip_col_prefix(label: str) -> str:
    """Remove [col123] prefix from term."""
    m = re.match(r"^\[col\d+\]\s*", label)
    return label[m.end() :] if m else label

def is_atlas_like(label: str) -> bool:
    """True if label describes atlas/structural mask, not fMRI contrast."""
    lower = strip_col_prefix(label).lower()
    return any(
        p in lower
        for p in [
            "harvardoxford",
            "jhu icbm",
            "maxprob",
            "dwi ",
            "fa 1mm",
            "fa 2mm",
            "tracts maxprob",
            "thr0",
            "thr25",
            "thr50",
        ]
    )

def is_poor_definition(raw: str) -> bool:
    """Heuristic: contrast definition is study-specific jargon, not cognitive concept."""
    raw = strip_col_prefix(raw).strip().lower()
    if len(raw) < 12:
        return True
    if "island" in raw and ("south" in raw or "north" in raw or "east" in raw or "west" in raw):
        return True  # IBC location jargon
    if "tstata" in raw or "tstatb" in raw:
        return True
    if raw.startswith("neurovault") or "collection " in raw:
        return True
    return False

def main():
    terms = pickle.load(open(NV_CACHE / "term_vocab.pkl", "rb"))
    cids = pickle.load(open(NV_CACHE / "term_collection_ids.pkl", "rb"))

    by_col = defaultdict(list)
    for t, c in zip(terms, cids):
        by_col[c].append(t)

    print("=== NeuroVault term label analysis ===\n")
    print(f"Total: {len(terms)} terms, {len(by_col)} collections\n")

    # Atlas collections
    atlas_terms = [(t, c) for t, c in zip(terms, cids) if c in ATLAS_COLLECTION_IDS]
    atlas_by_col = defaultdict(int)
    for _, c in atlas_terms:
        atlas_by_col[c] += 1
    print("Atlas/structural collections (EXCLUDE):")
    for cid in sorted(ATLAS_COLLECTION_IDS & set(by_col.keys())):
        meta = fetch_collection_metadata(cid)
        name = meta.get("name", "?")
        n = atlas_by_col.get(cid, len(by_col[cid]))
        print(f"  col{cid}: {n} terms - {name}")
    print()

    # Poor definition analysis per collection
    print("Collections with >40% poor definitions (study-specific jargon):")
    poor_cols = []
    for cid in sorted(by_col.keys()):
        if cid in ATLAS_COLLECTION_IDS:
            continue
        terms_c = by_col[cid]
        raw = [strip_col_prefix(t) for t in terms_c]
        n_poor = sum(1 for t in raw if is_poor_definition(t))
        if n_poor > len(terms_c) * 0.4:
            ex = terms_c[0][:70] if terms_c else ""
            poor_cols.append((cid, len(terms_c), n_poor, ex))
    for cid, n, npoor, ex in sorted(poor_cols, key=lambda x: -x[2] / max(1, x[1]))[:25]:
        meta = fetch_collection_metadata(cid)
        name = meta.get("name", "?")[:50]
        print(f"  col{cid}: {npoor}/{n} poor | {name}")
        print(f"    e.g. \"{ex}...\"")
    print()

    # Sample improved labels (strip prefix)
    print("=== Sample: raw vs stripped (no [colN]) ===\n")
    for t in terms[100:110]:
        stripped = strip_col_prefix(t)
        print(f"  raw:    {t[:80]}...")
        print(f"  strip:  {stripped[:80]}...")
        print()

if __name__ == "__main__":
    main()
