"""
NeuroVault training set ingestion: classification, averaging, QC, labeling.

Implements the pipeline from NEUROVAULT_TRAINING_SET_INGESTION_ALGORITHM.md.
Use ingest_collection() for per-collection processing; run build_neurovault_curated_cache.py
for the full pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

# Collections that need subject-level averaging (from neurovault_collections_averaging_guide.md)
# Moved to use-as-is (≤7 images, produced 0 when averaged — each contrast had n=1, min_subjects dropped):
# 19012, 6825, 20510, 1620, 2503, 13474, 13705, 3887
AVERAGE_FIRST = {
    1952, 6618, 2138, 4343, 16284,  # High priority
    426, 445, 507, 4804, 504, 13042, 13705,
    2108, 4683, 1516, 11646, 437, 12992,  # Medium priority
    503,  # PINES/IAPS: 182 subjects × 30 images → average by IAPS image
    3324,  # Kragel 2018: 15 subjects × 18 studies → average by study (Study01Subject01 -> Study 1)
    16266,  # Emotion regulation: PIP Subject0001 LookNeg Beta -> LookNeg (condition)
}

# Meta-analytic collections (IBMA, ALE, SDM, etc.) — higher SNR, weight 2× in training
META_ANALYTIC_COLLECTIONS = {
    18197, 844, 833, 830, 825, 839, 1425, 1432, 1501, 2462, 3884, 5070, 5377,
    5943, 6262, 7793, 8448, 11343, 20036, 555, 3822, 15965,
}

# UCLA LA5C: average healthy and clinical separately (requires metadata)
COLLECTION_4343_GROUP_BY = "cognitive_paradigm_cogatlas_id"  # or contrast_definition

# Collection 503 (PINES/IAPS): IAPS image IDs by valence (NeuroVault collection description)
IAPS_NEGATIVE_IDS = {2053, 3051, 3102, 3120, 3350, 3500, 3550, 6831, 9040, 9050, 9252, 9300, 9400, 9810, 9921}
IAPS_NEUTRAL_IDS = {5720, 5800, 7000, 7006, 7010, 7040, 7060, 7090, 7100, 7130, 7150, 7217, 7490, 7500, 9210}

MIN_SUBJECTS_PER_CONTRAST = 3
MIN_SUBJECTS_HETEROGENEOUS = 2  # BrainPedia, etc.
# Collections with 1–2 images per contrast (group-level or consensus maps): still average (output that map)
MIN_SUBJECTS_OVERRIDE = {426: 1, 437: 1, 507: 1}  # 426: 8 contrasts×1; 437: 26 subnetworks×1; 507: 4 contrasts, some n=1–2
OUTLIER_CORR_THRESHOLD = 0.2

# Per-sample loss weights by aggregation level (NEUROVAULT_TRAINING_SET_INGESTION_ALGORITHM.md)
WEIGHT_META_ANALYTIC = 2.0
WEIGHT_GROUP = 1.0
WEIGHT_SUBJECT_AVERAGED = 0.8


def get_sample_weight(collection_id: int, was_averaged: bool) -> float:
    """Return loss weight for a map from this collection. Meta 2×, group 1×, subject-averaged 0.8×."""
    if was_averaged:
        return WEIGHT_SUBJECT_AVERAGED
    if collection_id in META_ANALYTIC_COLLECTIONS:
        return WEIGHT_META_ANALYTIC
    return WEIGHT_GROUP


def get_contrast_key(entry: dict, collection_id: int) -> str:
    """Extract grouping key for averaging. Priority: contrast_definition, name (for 504/16284), cognitive_paradigm_cogatlas_id, path."""
    import re
    contrast = (entry.get("contrast_definition") or "").strip()
    name = (entry.get("name") or "").strip()
    cogatlas = (entry.get("cognitive_paradigm_cogatlas_id") or "").strip()
    # For 504/16284, name has condition (Pain Subject 10 High, sub001 positive) — prefer over cogatlas which groups all
    if collection_id in (504, 16284) and name:
        key = name
    else:
        key = contrast or cogatlas or name
    path = entry.get("path") or entry.get("file_path") or ""
    if not key and path:
        from pathlib import Path
        key = Path(path).stem.replace(".nii", "").replace(".gz", "")

    # Collection 503 (PINES/IAPS): "IAPS Subject 102 Image 9921" -> "IAPS Image 9921" (average across subjects)
    if collection_id == 503 and key:
        m = re.search(r"IAPS[_ ]Subject[_ ]\d+[_ ]Image[_ ](\d+)", key, re.I)
        if m:
            return f"IAPS Image {m.group(1)}"
    # Collection 504 (Pain NPS): "Pain Subject 10 High" or "Pain_Subject_10_High" -> "high" (group by intensity)
    if collection_id == 504 and key:
        m = re.search(r"Pain[_ ]Subject[_ ]\d+[_ ](High|Low|Medium)", key, re.I)
        if m:
            return m.group(1).lower()
    # Collection 16284 (IAPS valence): "sub001 positive" or "sub001_positive" -> "positive" (group by valence)
    if collection_id == 16284 and key:
        m = re.search(r"sub\d+[_ ](positive|negative|neutral)", key, re.I)
        if m:
            return m.group(1).lower()
    # Collection 3324 (Kragel): "Study01Subject01" -> "Study 1" (average 15 subjects per study)
    if collection_id == 3324 and key:
        m = re.search(r"Study(\d+)Subject\d+", key, re.I)
        if m:
            return f"Study {int(m.group(1))}"
    # Collection 16266 (Emotion regulation): "PIP Subject0001 LookNeg Beta" or "AHAB Subject0001 LookNeg Beta" -> "LookNeg"
    if collection_id == 16266 and key:
        m = re.search(r"(?:PIP|AHAB)[_ ]Subject\d+[_ ](\w+)", key, re.I)
        if m:
            return m.group(1)

    if key:
        return key
    return f"collection_{collection_id}_unknown"


def ingest_collection(
    collection_id: int,
    maps: list[dict],
    min_subjects: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Process one collection: average subject-level or use as-is.

    maps: list of {"data": N-D array (parcels), "contrast_definition": str, ...}
    Returns: {label: N-D map}. When averaging produces 0 terms, falls back to use-as-is.
    """
    if collection_id not in AVERAGE_FIRST:
        # Use as-is: one map per entry
        out = {}
        for i, m in enumerate(maps):
            data = m.get("data")
            if data is None or not isinstance(data, np.ndarray):
                continue
            label = (
                (m.get("contrast_definition") or "").strip()
                or (m.get("name") or f"collection_{collection_id}_map_{i}")
            )
            if not label:
                label = f"collection_{collection_id}_map_{i}"
            out[label] = np.asarray(data, dtype=np.float64)
        return out

    # Average by contrast
    min_n = min_subjects or MIN_SUBJECTS_OVERRIDE.get(
        collection_id,
        MIN_SUBJECTS_HETEROGENEOUS if collection_id == 1952 else MIN_SUBJECTS_PER_CONTRAST,
    )
    groups: dict[str, list[np.ndarray]] = defaultdict(list)

    for m in maps:
        data = m.get("data")
        if data is None or not isinstance(data, np.ndarray):
            continue
        key = get_contrast_key(m, collection_id)
        groups[key].append(np.asarray(data, dtype=np.float64))

    # Collection 503 (PINES): output single "negative vs neutral" contrast instead of 30 per-image maps
    if collection_id == 503:
        import re
        per_image: dict[str, np.ndarray] = {}
        for key, arr_list in groups.items():
            if len(arr_list) < min_n:
                continue
            arr = np.array(arr_list)
            mean_map = arr.mean(axis=0)
            keep = []
            for j in range(len(arr)):
                r = np.corrcoef(arr[j], mean_map)[0, 1] if arr[j].std() > 1e-8 else 1.0
                if not np.isnan(r) and r >= OUTLIER_CORR_THRESHOLD:
                    keep.append(arr[j])
            if len(keep) >= max(2, min_n - 1):
                per_image[key] = np.mean(keep, axis=0).astype(np.float64)
            elif len(arr_list) >= min_n:
                per_image[key] = mean_map.astype(np.float64)
        neg_maps = []
        neu_maps = []
        for key, m in per_image.items():
            match = re.search(r"IAPS Image (\d+)", key, re.I)
            if match:
                img_id = int(match.group(1))
                if img_id in IAPS_NEGATIVE_IDS:
                    neg_maps.append(m)
                elif img_id in IAPS_NEUTRAL_IDS:
                    neu_maps.append(m)
        if len(neg_maps) >= 5 and len(neu_maps) >= 5:
            neg_mean = np.mean(neg_maps, axis=0).astype(np.float64)
            neu_mean = np.mean(neu_maps, axis=0).astype(np.float64)
            return {"IAPS negative vs neutral (PINES)": (neg_mean - neu_mean).astype(np.float64)}
        # Fallback: if we can't form the contrast, return per-image (original behavior)
        return {k: v for k, v in per_image.items()}

    # Collection 504 (Pain NPS): output contrasts (high vs low, high vs medium, medium vs low)
    if collection_id == 504:
        cond_avgs: dict[str, np.ndarray] = {}
        for key, arr_list in groups.items():
            if len(arr_list) < min_n:
                continue
            arr = np.array(arr_list)
            mean_map = arr.mean(axis=0)
            keep = []
            for j in range(len(arr)):
                r = np.corrcoef(arr[j], mean_map)[0, 1] if arr[j].std() > 1e-8 else 1.0
                if not np.isnan(r) and r >= OUTLIER_CORR_THRESHOLD:
                    keep.append(arr[j])
            if len(keep) >= max(2, min_n - 1):
                cond_avgs[key] = np.mean(keep, axis=0).astype(np.float64)
            elif len(arr_list) >= min_n:
                cond_avgs[key] = mean_map.astype(np.float64)
        pain_out = {}
        for a, b in [("high", "low"), ("high", "medium"), ("medium", "low")]:
            if a in cond_avgs and b in cond_avgs:
                pain_out[f"pain intensity {a} vs {b}"] = (cond_avgs[a] - cond_avgs[b]).astype(np.float64)
        if pain_out:
            return pain_out

    # Collection 16284 (IAPS valence): output contrasts (positive vs neutral, negative vs neutral)
    if collection_id == 16284:
        cond_avgs = {}
        for key, arr_list in groups.items():
            if len(arr_list) < min_n:
                continue
            arr = np.array(arr_list)
            mean_map = arr.mean(axis=0)
            keep = []
            for j in range(len(arr)):
                r = np.corrcoef(arr[j], mean_map)[0, 1] if arr[j].std() > 1e-8 else 1.0
                if not np.isnan(r) and r >= OUTLIER_CORR_THRESHOLD:
                    keep.append(arr[j])
            if len(keep) >= max(2, min_n - 1):
                cond_avgs[key] = np.mean(keep, axis=0).astype(np.float64)
            elif len(arr_list) >= min_n:
                cond_avgs[key] = mean_map.astype(np.float64)
        valence_out = {}
        for a, b in [("positive", "neutral"), ("negative", "neutral")]:
            if a in cond_avgs and b in cond_avgs:
                valence_out[f"emotional valence {a} vs {b}"] = (cond_avgs[a] - cond_avgs[b]).astype(np.float64)
        if valence_out:
            return valence_out

    out = {}
    for key, arr_list in groups.items():
        if len(arr_list) < min_n:
            continue
        arr = np.array(arr_list)
        mean_map = arr.mean(axis=0)
        # Outlier removal: exclude maps with r < OUTLIER_CORR_THRESHOLD to group mean
        keep = []
        for j in range(len(arr)):
            r = np.corrcoef(arr[j], mean_map)[0, 1] if arr[j].std() > 1e-8 else 1.0
            if not np.isnan(r) and r >= OUTLIER_CORR_THRESHOLD:
                keep.append(arr[j])
        if len(keep) >= max(2, min_n - 1):
            out[key] = np.mean(keep, axis=0).astype(np.float64)
        elif len(arr_list) >= min_n:
            out[key] = mean_map.astype(np.float64)

    # Fallback: when averaging produces 0 (all contrasts dropped by min_subjects), use each map as-is
    if not out:
        for i, m in enumerate(maps):
            data = m.get("data")
            if data is None or not isinstance(data, np.ndarray):
                continue
            label = (
                (m.get("contrast_definition") or "").strip()
                or (m.get("name") or f"collection_{collection_id}_map_{i}")
            )
            if not label:
                label = f"collection_{collection_id}_map_{i}"
            out[label] = np.asarray(data, dtype=np.float64)
    return out


def qc_filter(maps: np.ndarray, n_parcels: int = 427) -> np.ndarray:
    """
    Stage 2 QC: reject bad maps. Returns boolean mask (True = keep).
    """
    if maps.ndim != 2 or maps.shape[1] != n_parcels:
        return np.zeros(maps.shape[0], dtype=bool)

    keep = np.ones(maps.shape[0], dtype=bool)
    for i in range(maps.shape[0]):
        m = maps[i]
        if np.sum(np.abs(m)) < 1e-10:
            keep[i] = False  # all-zero (catches maps that become 0 after parcellation)
            continue
        zeros = np.sum(np.abs(m) < 1e-10) / len(m)
        if zeros > 0.95:
            keep[i] = False
        nans = np.sum(np.isnan(m)) / len(m)
        if nans > 0.1:
            keep[i] = False
        if np.nanmax(np.abs(m)) > 50:
            keep[i] = False
        if np.nanstd(m) < 0.01:
            keep[i] = False
    return keep


def zscore_maps(maps: np.ndarray, axis: int = 1) -> np.ndarray:
    """Stage 4: Z-score each map across parcels."""
    m = np.nanmean(maps, axis=axis, keepdims=True)
    s = np.nanstd(maps, axis=axis, keepdims=True)
    s = np.where(s < 1e-10, 1.0, s)
    return ((maps - m) / s).astype(np.float32)
