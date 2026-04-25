#!/usr/bin/env python3
"""Report how the NeuroVault curated cache is formed: collections, raw images, output terms."""
import json
import pickle
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "neurolab" / "data"

NAMES = {
    503: "PINES/IAPS", 504: "Pain NPS", 16284: "IAPS valence", 1952: "BrainPedia",
    6618: "IBC 2nd", 2138: "IBC 1st", 4343: "UCLA LA5C", 426: "False belief ToM",
    445: "Why/How ToM", 507: "Consensus decision", 2503: "Social Bayesian",
    4804: "Fusiform-network", 13042: "Language/WM epilepsy", 13705: "Tier3 language",
    1516: "Tier3", 2108: "Tier3", 4683: "Tier3", 3887: "Tier3", 13474: "Tensorial ICA MDD",
    20510: "Lesions psychosis", 11646: "ASD emotional egocentricity", 437: "Autism subnetworks",
    12992: "Nicotine abstinence", 19012: "Incentive-boosted inhibitory", 6825: "Schizophrenia deformation",
    1620: "Depression resting-state", 457: "HCP", 1274: "Cognitive Atlas", 3324: "Pain+Cog+Emotion",
    20820: "HCP-YA network", 262: "Harvard-Oxford (excl)", 264: "JHU DTI (excl)",
    15030: "Tier3", 7760: "WM atlas", 7758: "WM atlas", 7759: "WM atlas",
    7756: "WM atlas", 7761: "WM atlas", 15274: "Tier3", 4146: "Tier3",
    16266: "Tier3", 6237: "Tier3", 3822: "Meta", 13665: "Pharma",
}

AVERAGE_FIRST = {
    1952, 6618, 2138, 4343, 16284, 3324, 426, 445, 507, 2503, 4804, 504, 13042, 13705,
    2108, 4683, 3887, 1516, 13474, 20510, 11646, 437, 12992, 19012, 6825, 1620, 503,
}


def main() -> None:
    manifest_path = BASE / "neurovault_curated_data" / "manifest.json"
    cache_dir = BASE / "neurovault_cache"

    if not manifest_path.exists():
        print("Manifest not found:", manifest_path)
        return
    if not (cache_dir / "term_vocab.pkl").exists():
        print("Cache not found:", cache_dir)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images", [])

    raw_counts: dict[int, int] = defaultdict(int)
    for img in images:
        cid = img.get("collection_id")
        if cid:
            raw_counts[cid] += 1

    vocab = pickle.loads((cache_dir / "term_vocab.pkl").read_bytes())
    cids = pickle.loads((cache_dir / "term_collection_ids.pkl").read_bytes())
    cache_counts: dict[int, int] = defaultdict(int)
    for cid in cids:
        cache_counts[cid] += 1

    cids_with_out = [c for c in cache_counts if cache_counts[c] > 0]
    rows = []
    for cid in sorted(cids_with_out, key=lambda c: -cache_counts[c]):
        raw = raw_counts.get(cid, 0)
        out = cache_counts[cid]
        name = NAMES.get(cid, f"col{cid}")
        avg = "Yes" if cid in AVERAGE_FIRST else "No"
        ratio = f"{raw} → {out}" if raw else f"→ {out}"
        rows.append((cid, name, raw, out, avg, ratio))

    total_raw_manifest = sum(raw_counts.values())
    total_raw_in_cache = sum(raw_counts.get(c, 0) for c in cids_with_out)
    total_out = sum(cache_counts[c] for c in cids_with_out)

    out_lines = [
        "# NeuroVault Curated Cache Formation",
        "",
        "**Source:** `neurovault_curated_data` (manifest + downloads)",
        "**Output:** `neurovault_cache` (term_maps.npz, term_vocab.pkl)",
        "",
        "Pipeline: `build_neurovault_cache.py --average-subject-level` (parcellate → average AVERAGE_FIRST → QC → z-score)",
        "",
        "## Collections in cache (by output terms, desc)",
        "",
        "| ID | Name | Raw images | Output terms | Averaged? | Ratio |",
        "|----|------|:----------:|:------------:|:---------:|-------|",
    ]
    for cid, name, raw, out, avg, ratio in rows:
        out_lines.append(f"| {cid} | {name} | {raw} | {out} | {avg} | {ratio} |")

    out_lines.extend([
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total raw images (manifest) | {total_raw_manifest:,} |",
        f"| Raw images from collections in cache | {total_raw_in_cache:,} |",
        f"| Total output terms in cache | {total_out:,} |",
        f"| Collections with output | {len(cids_with_out)} |",
        "",
        "## Excluded from cache",
        "",
        "- **262, 264:** Atlas/structural (Harvard-Oxford, JHU DTI) — excluded by `--exclude-atlas-collections`",
        "- **Collections with 0 output:** Failed QC, parcellation, or dropped by averaging (min_subjects)",
        "",
    ])

    out_path = REPO / "neurolab" / "docs" / "implementation" / "NEUROVAULT_CURATED_CACHE_FORMATION.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
