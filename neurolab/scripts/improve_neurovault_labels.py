#!/usr/bin/env python3
"""
Improve NeuroVault cache term labels for text-to-brain training.

- Excludes atlas/structural collections (262 Harvard-Oxford, 264 JHU DTI)
- Strips non-informative [colN] prefix from terms
- Improves WM atlas labels: "acoustic tstatA" -> "acoustic"
- For Figure references, prepends collection name when available (NeuroVault API)
- Resolves trm_ Cognitive Atlas task IDs to human-readable names (e.g. trm_4f2453ce33f16 -> social judgment task)
- Optional: expand common abbreviations (--expand-abbreviations): MDD, ASD, PTSD, ADHD, etc.

Usage:
  python neurolab/scripts/improve_neurovault_labels.py
  python neurolab/scripts/improve_neurovault_labels.py --cache-dir neurolab/data/neurovault_cache --output-dir neurolab/data/neurovault_cache_improved
  python neurolab/scripts/improve_neurovault_labels.py --no-fetch-metadata  # Skip API calls
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
API_BASE = "https://neurovault.org/api"
COGNITIVE_ATLAS_JSON = "https://www.cognitiveatlas.org/task/json"
HEADERS = {"Accept": "application/json", "User-Agent": "NeuroLab/1.0"}

# Atlas/structural collections — NOT fMRI task contrasts; exclude from cache
# 262 Harvard-Oxford, 264 JHU DTI, 1625 Brainnetome, 6074 brainstem tract, 9357 APOE structural
ATLAS_COLLECTION_IDS = {262, 264, 1625, 6074, 9357}

# WM atlas collections (7756-7761): strip " tstatA", " tstatB" to get clean cognitive terms
WM_ATLAS_COLLECTION_IDS = {7756, 7757, 7758, 7759, 7760, 7761}

# Clinical/neuro abbreviations to expand (whole-word match only)
ABBREVIATION_EXPANSIONS: list[tuple[str, str]] = [
    (r"\bMDD\b", "major depressive disorder"),
    (r"\bASD\b", "autism spectrum disorder"),
    (r"\bPTSD\b", "post-traumatic stress disorder"),
    (r"\bADHD\b", "attention-deficit/hyperactivity disorder"),
    (r"\bGAD\b", "generalized anxiety disorder"),
    (r"\bDLPFC\b", "dorsolateral prefrontal cortex"),
    (r"\bDMN\b", "default mode network"),
    (r"\bACC\b", "anterior cingulate cortex"),
]


def fetch_collection_metadata(cid: int, timeout: float = 15.0) -> dict:
    """Fetch collection name from NeuroVault API."""
    try:
        req = urllib.request.Request(f"{API_BASE}/collections/{cid}/", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def strip_col_prefix(label: str) -> str:
    """Remove [col123] prefix."""
    m = re.match(r"^\[col\d+\]\s*", label)
    return label[m.end() :] if m else label


def replace_regparam(label: str) -> str:
    """Replace opaque 'regparam' suffix with descriptive 'parametric contrast' (pharmacological GLM betas)."""
    return re.sub(r"\bregparam\b", "parametric contrast", label, flags=re.I)


def strip_source_from_label(label: str) -> str:
    """Remove NeuroVault and other source identifiers. Keep the term content only."""
    s = label.strip()
    s = re.sub(r"^NeuroVault fMRI task\s+", "", s, flags=re.I)
    s = re.sub(r"^NeuroVault fMRI collection\s+\d+\s+image\s+\d+\s*", "", s, flags=re.I)
    s = re.sub(r"^NeuroVault\s+", "", s, flags=re.I)
    return s.strip()


def improve_wm_atlas_label(raw: str) -> str:
    """'acoustic tstatA' -> 'acoustic'; 'decision making tstatB' -> 'decision making'."""
    raw = strip_col_prefix(raw)
    for suffix in (" tstatA", " tstatB", " tstat"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)].strip()
    return raw


def is_figure_reference(raw: str) -> bool:
    """True if contrast is a figure/table reference rather than cognitive concept."""
    raw = strip_col_prefix(raw).strip()
    if re.match(r"^(Figure|Fig\.?)\s*\d+", raw, re.I):
        return True
    if re.match(r"^[A-Za-z]+\s*\d+[A-Za-z]?$", raw) and len(raw) < 25:  # e.g. "k=5_MAG-1"
        return True
    return False


def expand_abbreviations(label: str) -> str:
    """Replace whole-word abbreviations with full terms."""
    for pattern, replacement in ABBREVIATION_EXPANSIONS:
        label = re.sub(pattern, replacement, label)
    return label


def resolve_trm_id(trm_id: str, timeout: float = 10.0) -> str | None:
    """Fetch Cognitive Atlas task name for trm_XXXX ID. Returns None on failure."""
    try:
        url = f"{COGNITIVE_ATLAS_JSON}/{trm_id}/"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        name = (data.get("name") or "").strip()
        return name if name else None
    except Exception:
        return None


def resolve_trm_in_label(label: str, cache: dict[str, str], timeout: float = 10.0) -> str:
    """Replace trm_XXXX Cognitive Atlas IDs in label with human-readable task names."""
    pattern = re.compile(r"trm_[a-f0-9]{12,}")
    for m in pattern.finditer(label):
        trm_id = m.group(0)
        if trm_id not in cache:
            cache[trm_id] = resolve_trm_id(trm_id, timeout) or trm_id
            time.sleep(0.2)  # rate limit
        name = cache[trm_id]
        label = label.replace(trm_id, name)
    # Clean up redundant "NeuroVault fMRI task" prefix when we now have a real name
    label = re.sub(r"fMRI:\s*NeuroVault fMRI task\s+", "fMRI: ", label)
    label = re.sub(r"^NeuroVault fMRI task\s+", "", label)
    return label.strip()


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
        description="Improve NeuroVault cache term labels (exclude atlas, strip prefix, clean WM atlas)"
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
        "--exclude-atlas",
        action="store_true",
        default=True,
        help="Exclude atlas collections 262, 264 (default: True)",
    )
    parser.add_argument(
        "--no-exclude-atlas",
        action="store_true",
        help="Do not exclude atlas collections",
    )
    parser.add_argument(
        "--strip-prefix",
        action="store_true",
        default=True,
        help="Strip [colN] prefix from terms (default: True)",
    )
    parser.add_argument(
        "--no-strip-prefix",
        action="store_true",
        help="Keep [colN] prefix",
    )
    parser.add_argument(
        "--improve-wm-atlas",
        action="store_true",
        default=True,
        help="Clean WM atlas labels: 'acoustic tstatA' -> 'acoustic' (default: True)",
    )
    parser.add_argument(
        "--improve-figure-labels",
        action="store_true",
        default=True,
        help="Prepend collection name for Figure references (default: True)",
    )
    parser.add_argument(
        "--no-fetch-metadata",
        action="store_true",
        help="Skip NeuroVault API calls (no collection names for Figure labels)",
    )
    parser.add_argument(
        "--expand-abbreviations",
        action="store_true",
        help="Expand clinical/neuro abbreviations (MDD→major depressive disorder, ASD→autism spectrum disorder, etc.)",
    )
    parser.add_argument(
        "--no-resolve-trm",
        action="store_true",
        help="Skip resolving trm_ Cognitive Atlas IDs to human-readable task names",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    exclude_atlas = args.exclude_atlas and not args.no_exclude_atlas
    strip_prefix = args.strip_prefix and not args.no_strip_prefix
    improve_wm = args.improve_wm_atlas
    improve_fig = args.improve_figure_labels and not args.no_fetch_metadata
    expand_abbrev = args.expand_abbreviations
    resolve_trm = not args.no_resolve_trm

    # Load cache
    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    cids = pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb"))
    if (cache_dir / "term_sample_weights.pkl").exists():
        weights = pickle.load(open(cache_dir / "term_sample_weights.pkl", "rb"))
    else:
        weights = [1.0] * len(terms)

    n_before = len(terms)
    print(f"Loaded {n_before} terms from {cache_dir}")

    # Filter: exclude atlas collections
    if exclude_atlas:
        keep = [i for i in range(len(terms)) if cids[i] not in ATLAS_COLLECTION_IDS]
        n_excluded = n_before - len(keep)
        if n_excluded:
            print(f"Excluding {n_excluded} terms from atlas collections {ATLAS_COLLECTION_IDS}")
        term_maps = term_maps[keep]
        terms = [terms[i] for i in keep]
        cids = [cids[i] for i in keep]
        weights = [weights[i] for i in keep]

    # Improve labels
    collection_names: dict[int, str] = {}
    trm_cache: dict[str, str] = {}
    new_terms = []
    for i, (lab, cid) in enumerate(zip(terms, cids)):
        raw = lab

        # Strip [colN] prefix
        if strip_prefix:
            raw = strip_col_prefix(raw)

        # WM atlas: remove tstatA/tstatB
        if improve_wm and cid in WM_ATLAS_COLLECTION_IDS:
            raw = improve_wm_atlas_label(lab)

        # Figure references: prepend collection name
        if improve_fig and is_figure_reference(raw):
            if cid not in collection_names:
                meta = fetch_collection_metadata(cid)
                collection_names[cid] = (meta.get("name") or f"Collection {cid}")[:60]
                time.sleep(0.15)  # rate limit
            cname = collection_names[cid]
            raw = f"{cname}: {raw}"

        # Expand abbreviations (MDD→major depressive disorder, ASD→autism spectrum disorder, etc.)
        if expand_abbrev:
            raw = expand_abbreviations(raw)

        # Replace trm_XXXX Cognitive Atlas IDs with human-readable task names
        if resolve_trm and "trm_" in raw:
            raw = resolve_trm_in_label(raw, trm_cache)

        # Replace regparam with parametric contrast (pharmacological parametric regressors)
        raw = replace_regparam(raw)

        # Remove NeuroVault and other source identifiers (keep term content only)
        raw = strip_source_from_label(raw)

        new_terms.append(raw)

    terms = new_terms

    # Ensure uniqueness (e.g. "acoustic" from tstatA and tstatB)
    terms = _make_labels_unique(terms)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "term_maps.npz", term_maps=term_maps)
    with open(output_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    with open(output_dir / "term_collection_ids.pkl", "wb") as f:
        pickle.dump(cids, f)
    with open(output_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump(weights, f)

    print(f"Saved {len(terms)} terms to {output_dir}")
    if exclude_atlas and n_before > len(terms):
        print(f"  (removed {n_before - len(terms)} atlas terms)")

    # Show samples
    print("\nSample improved terms:")
    for t in terms[:15]:
        print(f"  {t[:85]}{'...' if len(t) > 85 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
