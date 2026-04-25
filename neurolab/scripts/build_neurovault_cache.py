#!/usr/bin/env python3
"""
Build a (term, map) cache from NeuroVault data: parcellate each image to the pipeline
atlas (Glasser+Tian, 392/427/450), resample if needed, average subject-level collections, QC, z-score.

**Curation and preprocessing (see neurovault_collections_averaging_guide.md):**
- **Atlas:** Glasser 360 + Tian S2 (+ optional brainstem/BFB/Hyp). Every image is resampled to this atlas
  via resample_to_atlas() before parcellation (correct alignment regardless of source MNI/template).
- **Group averages:** Use --average-subject-level so subject-level collections (1952, 6618, 2138,
  4343, 16284, etc.) are averaged by contrast within collection; group/meta-analytic collections
  are used as-is.
- **QC:** Bad maps (all-zero, high-NaN, extreme, constant) are rejected; --no-qc to skip.
- **Z-score:** Global across parcels (fMRI: preserves cross-compartment pattern); --no-zscore to skip.
- **Sample weights:** term_sample_weights.pkl (meta 2×, group 1×, subject-averaged 0.8×).

**Data source:** Curated: download_neurovault_curated.py --all → neurolab/data/neurovault_curated_data.
Legacy: download_neurovault_data.py → neurolab/data/neurovault_data. Use --collections to restrict.

**Terms:** contrast_definition from NeuroVault; missing ones can be fetched from API (--no-fetch-missing-metadata to disable).
Duplicate labels get _1, _2, etc. Use --cluster-by-description to average by exact description.

**Input:** Data dir with manifest.json and downloads/neurovault/; or --from-downloads to scan disk (no manifest).

  python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level
  python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --from-downloads   # when manifest not yet written
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(_repo_root, "neurolab", "data", "neurovault_data")
DEFAULT_OUTPUT_DIR = os.path.join(_repo_root, "neurolab", "data", "neurovault_cache")

# 0 = use contrast_definition verbatim (no truncation). Set >0 to cap length (e.g. 512 for sentence-transformers).
MAX_CONTRAST_CHARS = 0
NEUROVAULT_IMAGE_API = "https://neurovault.org/api/images/{id}/"

# Atlas/structural collections — not fMRI task contrasts; exclude by default
# 262 Harvard-Oxford, 264 JHU DTI, 1625 Brainnetome, 6074 brainstem tract atlas, 9357 APOE structural
ATLAS_COLLECTION_IDS = {262, 264, 1625, 6074, 9357}
# WM atlas (7756-7761): "acoustic tstatA" -> "acoustic" for cleaner labels
WM_ATLAS_COLLECTION_IDS = {7756, 7757, 7758, 7759, 7760, 7761}

# Known P-map collections (NeuroVault map_type "P map"); transform to -log10(p) before parcellation
# 555: reward/addiction/obesity meta-analysis; 2508: cannabis meta-analysis (DOI 10.1177/0269881117744995)
P_MAP_COLLECTION_IDS = {555, 2508}

# ROI/mask collections: parcellation with mean yields zeros; retry with strategy='sum' for overlap counts
ROI_MASK_COLLECTION_IDS = {2485, 3434}

# Exclude from cache: ROI/cluster masks and subnetwork maps (wrong supervision for regression).
# 2508: cannabis meta-analysis — ROI_ACC, ROI_DLPFC, ROI_Striatum (cluster-derived masks).
# 437: autism functional subnetworks — graph-derived consistency maps (0–1), not activation contrasts.
EXCLUDE_FROM_CACHE_COLLECTION_IDS = {2508, 437}

# Map kind by collection (activation=default; p_map=unsigned significance; roi_mask=overlap)
MAP_KIND_P_MAP = P_MAP_COLLECTION_IDS  # 555
MAP_KIND_SUBNETWORK = {437}
MAP_KIND_ROI_MASK = ROI_MASK_COLLECTION_IDS | EXCLUDE_FROM_CACHE_COLLECTION_IDS

# Dose-related patterns (collection name/description)
_DOSE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|ug|µg|mcg|ml)\b",
    r"\bdose\b",
    r"\bplacebo\b",
    r"\bdouble[-\s]?blind\b",
    r"\brandomi[sz]ed\b",
    r"\bcrossover\b",
]

# Known drugs for extraction from pharmacological collection names (--pharma-add-drug)
# Order matters: first match wins. Prefer specific compounds over classes.
PHARMA_DRUGS = [
    "ketamine", "psilocybin", "LSD", "DMT", "ayahuasca", "mescaline",
    "caffeine", "nicotine", "cocaine", "amphetamine", "methylphenidate", "modafinil",
    "alcohol", "cannabis", "THC", "cannabidiol", "MDMA",
    "fluoxetine", "escitalopram", "bupropion", "haloperidol", "risperidone", "clozapine",
    "morphine", "fentanyl", "buprenorphine", "naloxone",
    "ibuprofen", "oxytocin", "topiramate",
    "opioid", "antidepressant", "antipsychotic", "SSRI", "benzodiazepine",
]


def _is_dose_related(name: str, description: str) -> bool:
    """True if collection metadata indicates dose arms (mg, placebo, etc.)."""
    text = f"{(name or '')} {(description or '')}".lower()
    return any(re.search(p, text) for p in _DOSE_PATTERNS)


def _sanitize_label_for_unsigned_maps(label: str) -> str:
    """For P-maps: remove direction (A - B -> A vs B) so label matches unsigned significance."""
    if not label or not isinstance(label, str):
        return label
    s = label.strip()
    # A minus B / A - B -> A vs B
    s = re.sub(r"\s+minus\s+", " vs ", s, flags=re.I)
    s = re.sub(r"\s+>\s+", " vs ", s, flags=re.I)
    s = re.sub(r"\s+<\s+", " vs ", s, flags=re.I)
    s = re.sub(r"\s+-\s+", " vs ", s, count=1)  # first hyphen only (contrast)
    if " vs " in s and "significance" not in s.lower():
        s = f"{s} (significance)"
    return s.strip()


def _extract_drug_from_collection_name(name: str) -> str:
    """Extract drug from collection name for pharmacological labels. Returns first match or truncated name."""
    if not name or not isinstance(name, str):
        return ""
    n = name.strip()
    n_lower = n.lower()
    for drug in PHARMA_DRUGS:
        if drug.lower() in n_lower:
            return drug
    return n[:30].strip() if len(n) > 30 else n


def _extract_dose_from_text(text: str) -> str | None:
    """Extract dose string from name/description, e.g. '200 mg' or 'placebo, 200 mg, 600 mg'."""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|g|ug|µg|mcg|ml)\b)", text, re.I)
    return m.group(1).strip() if m else None


# NeuroVault returns 403 for default Python User-Agent; use a browser-like one
_FETCH_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (compatible; NeuroVaultCache/1.0)"}


def _fetch_image_metadata(image_id: int, timeout: float = 15.0) -> dict:
    """Fetch image metadata from NeuroVault API (contrast_definition, task, name, map_type, etc.)."""
    try:
        req = urllib.request.Request(
            NEUROVAULT_IMAGE_API.format(id=image_id),
            headers=_FETCH_HEADERS,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_p_map(entry: dict) -> bool:
    """True if image is a P map (p-values); transform to -log10(p) before parcellation."""
    mt = (entry.get("map_type") or "").strip()
    if mt and "p" in mt.lower() and "map" in mt.lower():
        return True
    return entry.get("collection_id") in P_MAP_COLLECTION_IDS


def _entry_has_good_label(entry: dict) -> bool:
    """True if entry has a non-empty contrast_definition (we use only the definition)."""
    c = (entry.get("contrast_definition") or "").strip()
    return bool(c)


def _get_label(entry: dict, index: int, max_contrast_chars: int = MAX_CONTRAST_CHARS) -> str:
    """Use contrast_definition as label; fallback to name/task or descriptive NeuroVault collection+image ID."""
    contrast = (entry.get("contrast_definition") or "").strip()
    if contrast:
        if max_contrast_chars and max_contrast_chars > 0:
            return contrast[:max_contrast_chars]
        return contrast
    # Fallbacks when contrast_definition missing
    name = (entry.get("name") or "").strip()
    if name and len(name) > 3:
        return name[:200] if max_contrast_chars <= 0 else name[:min(200, max_contrast_chars)]
    task = (entry.get("cognitive_paradigm_cogatlas_id") or entry.get("task") or "").strip()
    if task and len(task) > 2:
        return task[:200]  # no source prefix; modality (fMRI) added at merge
    cid = entry.get("collection_id")
    iid = entry.get("image_id")
    if cid is not None and iid is not None:
        return f"collection {cid} image {iid}"  # no source prefix
    return f"neurovault_image_{index}"


def _improve_wm_atlas_label(lab: str) -> str:
    """'acoustic tstatA' -> 'acoustic' for WM atlas collections."""
    for suffix in (" tstatA", " tstatB", " tstat"):
        if lab.endswith(suffix):
            return lab[: -len(suffix)].strip()
    return lab


def _make_labels_unique(labels: list[str]) -> list[str]:
    """Append _1, _2, ... to duplicates so every term is unique."""
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
        description="Build (term, map) cache from NeuroVault data using metadata as terms"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory with manifest.json and downloads/neurovault/ (from download_neurovault_data.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for term_maps.npz and term_vocab.pkl",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=0,
        help="Max number of maps to process (0 = all in manifest or all on disk)",
    )
    parser.add_argument(
        "--from-downloads",
        action="store_true",
        help="If manifest.json missing, scan downloads/neurovault/collection_*/*.nii.gz and build cache (labels = collection_image_id)",
    )
    parser.add_argument(
        "--max-contrast-chars",
        type=int,
        default=MAX_CONTRAST_CHARS,
        help="Max characters for contrast_definition label; 0 = verbatim (default). Set e.g. 512 to truncate.",
    )
    parser.add_argument(
        "--fetch-missing-metadata",
        action="store_true",
        default=True,
        help="When contrast_definition is missing, fetch per-image metadata from NeuroVault API (default: True). Use --no-fetch-missing-metadata to disable.",
    )
    parser.add_argument(
        "--no-fetch-missing-metadata",
        action="store_false",
        dest="fetch_missing_metadata",
        help="Do not fetch missing contrast_definition from API; images without it are skipped.",
    )
    parser.add_argument(
        "--cluster-by-description",
        action="store_true",
        help="Group images by exact contrast_definition and average their maps; output one (term, map) per unique description (no _1, _2). More homogeneous cache.",
    )
    parser.add_argument(
        "--average-subject-level",
        action="store_true",
        help="For subject-level collections (1952, 6618, 2138, 4343, 16284, etc.), average by contrast within collection. Use-as-is for group-level collections. See neurovault_collections_averaging_guide.md.",
    )
    parser.add_argument(
        "--collections",
        type=int,
        nargs="*",
        default=None,
        help="If set, only process images from these NeuroVault collection IDs (e.g. --collections 1952 for BrainPedia/IBC only). Default: all in manifest.",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip Stage 2 QC filter (reject all-zero, high-NaN, extreme, constant maps). Default: apply QC.",
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Skip Stage 4 Z-score normalization across parcels. Default: z-score each map.",
    )
    parser.add_argument(
        "--exclude-atlas-collections",
        action="store_true",
        default=True,
        help="Exclude atlas collections 262 (Harvard-Oxford), 264 (JHU DTI) — not fMRI contrasts (default: True).",
    )
    parser.add_argument(
        "--no-exclude-atlas-collections",
        action="store_false",
        dest="exclude_atlas_collections",
        help="Include atlas collections.",
    )
    parser.add_argument(
        "--strip-col-prefix",
        action="store_true",
        default=True,
        help="Omit [colN] prefix from terms; collection_id stored in term_collection_ids.pkl (default: True).",
    )
    parser.add_argument(
        "--no-strip-col-prefix",
        action="store_false",
        dest="strip_col_prefix",
        help="Add [colN] prefix to terms (legacy behavior).",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for parcellation (default 1)")
    parser.add_argument(
        "--pharma-add-drug",
        action="store_true",
        help="For pharmacological caches: append drug from collection name to each label (e.g. 'reason (ketamine)'). Requires manifest with collections_meta.",
    )
    args = parser.parse_args()

    try:
        from nilearn.maskers import NiftiLabelsMasker
        import nibabel as nib
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from neurolab.parcellation import get_masker, get_n_parcels, resample_to_atlas
    except ImportError as e:
        print(f"Install nilearn and nibabel: {e}", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(_repo_root) / args.data_dir
    downloads_dir = data_dir / "downloads" / "neurovault"
    manifest_path = data_dir / "manifest.json"

    collection_drug: dict[int, str] = {}
    collection_meta: dict[int, dict] = {}  # cid -> {name, description} for dose/metadata
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images") or []
        for c in data.get("collections_meta") or []:
            cid = c.get("id")
            if cid is not None:
                collection_meta[int(cid)] = {"name": c.get("name") or "", "description": c.get("description") or ""}
        if getattr(args, "pharma_add_drug", False):
            for c in data.get("collections_meta") or []:
                cid = c.get("id")
                name = c.get("name") or ""
                if cid is not None and name:
                    drug = _extract_drug_from_collection_name(name)
                    if drug:
                        collection_drug[int(cid)] = drug
            if collection_drug:
                print(f"Pharma-add-drug: loaded drug/context for {len(collection_drug)} collections", flush=True)
        if getattr(args, "collections", None) is not None and args.collections:
            images = [e for e in images if e.get("collection_id") in args.collections]
            print(f"Filtered to collections {args.collections}: {len(images)} images", flush=True)
        if args.max_maps > 0:
            images = images[: args.max_maps]
        n_total = len(images)
        from_manifest = True
    elif args.from_downloads and downloads_dir.exists():
        # Build minimal entry list from files on disk (download in progress, no manifest yet)
        images = []
        for cdir in sorted(downloads_dir.iterdir()):
            if not cdir.is_dir() or not cdir.name.startswith("collection_"):
                continue
            try:
                cid = int(cdir.name.replace("collection_", ""))
            except ValueError:
                continue
            for f in cdir.glob("image_*.*"):
                stem = f.stem
                if f.suffix == ".gz" and stem.endswith(".nii"):
                    stem = Path(stem).stem  # image_12345.nii
                if stem.startswith("image_"):
                    try:
                        iid = int(stem.replace("image_", "").split(".")[0])
                    except ValueError:
                        continue
                    images.append({"path": str(f), "collection_id": cid, "image_id": iid})
        if getattr(args, "collections", None) is not None and args.collections:
            images = [e for e in images if e.get("collection_id") in args.collections]
            print(f"Filtered to collections {args.collections}: {len(images)} images", flush=True)
        if args.max_maps > 0:
            images = images[: args.max_maps]
        n_total = len(images)
        from_manifest = False
        print(f"Using {n_total} images from downloads (no manifest; labels will be collection_image_id)")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for c in data.get("collections_meta") or []:
                cid = c.get("id")
                if cid is not None:
                    collection_meta[int(cid)] = {"name": c.get("name") or "", "description": c.get("description") or ""}
            if getattr(args, "pharma_add_drug", False):
                for c in data.get("collections_meta") or []:
                    cid = c.get("id")
                    name = c.get("name") or ""
                    if cid is not None and name:
                        drug = _extract_drug_from_collection_name(name)
                        if drug:
                            collection_drug[int(cid)] = drug
                if collection_drug:
                    print(f"Pharma-add-drug: loaded drug/context for {len(collection_drug)} collections", flush=True)
    else:
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        print("Run download_neurovault_data.py first, or use --from-downloads to use existing files.", file=sys.stderr)
        return 1

    # Exclude ROI/mask collections (wrong supervision for regression)
    n_excluded = sum(1 for e in images if e.get("collection_id") in EXCLUDE_FROM_CACHE_COLLECTION_IDS)
    if n_excluded:
        images = [e for e in images if e.get("collection_id") not in EXCLUDE_FROM_CACHE_COLLECTION_IDS]
        print(f"Excluded {n_excluded} images from collections {EXCLUDE_FROM_CACHE_COLLECTION_IDS} (ROI/mask, not activation)", flush=True)

    if not images:
        print("No images to process.", file=sys.stderr)
        return 1

    n_with_contrast = sum(1 for e in images if _entry_has_good_label(e))
    if args.fetch_missing_metadata:
        print("Will fetch missing contrast_definition from NeuroVault API when needed (rate-limited).")
    print(f"Manifest: {n_total} images, {n_with_contrast} already have contrast_definition (rest will be fetched from API).", flush=True)

    n_parcels = get_n_parcels()
    masker = get_masker(memory="nilearn_cache", verbose=0)
    masker.fit()
    has_roi_collections = any(e.get("collection_id") in ROI_MASK_COLLECTION_IDS for e in images)
    masker_sum = None
    if has_roi_collections:
        masker_sum = get_masker(memory="nilearn_cache", verbose=0, strategy="sum")
        masker_sum.fit()

    terms: list[str] = []
    maps_list: list[np.ndarray] = []
    collection_ids: list[int] = []  # per-term collection_id for sample weighting
    failed = 0
    n_fetched = 0
    n_truncated = 0
    max_raw_len = 0
    progress_interval = 200

    # When --average-subject-level, collect (entry, vec) first, then process per-collection
    use_average_subject = getattr(args, "average_subject_level", False)
    if use_average_subject:
        from collections import defaultdict
        from neurolab.neurovault_ingestion import ingest_collection, AVERAGE_FIRST
        collected: dict[int, list[dict]] = defaultdict(list)

    n_jobs = max(1, getattr(args, "n_jobs", 1))
    _masker_lock = threading.Lock() if n_jobs > 1 else None

    def _parcellate_one(item):
        i, entry, path = item
        cid = entry.get("collection_id")
        try:
            with (_masker_lock or _dummy_lock):
                img = nib.load(path)
                if _is_p_map(entry):
                    data = img.get_fdata()
                    data = -np.log10(np.clip(data.astype(np.float64), 1e-10, 1.0))
                    img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
                img = resample_to_atlas(img)
                vec = masker.transform(img).ravel().astype(np.float64)
                # ROI/mask collections: mean can yield zeros; retry with sum for overlap counts
                if (
                    masker_sum is not None
                    and cid in ROI_MASK_COLLECTION_IDS
                    and vec.shape[0] == n_parcels
                    and np.isfinite(vec).all()
                    and np.abs(vec).max() < 1e-10
                ):
                    vec = masker_sum.transform(img).ravel().astype(np.float64)
                    # Normalize sum (voxel counts) to ~[0,1] so QC passes
                    vmax = np.max(vec) + 1e-10
                    if vmax > 1e-10:
                        vec = vec / vmax
            if vec.shape[0] != n_parcels or not np.isfinite(vec).all():
                return (i, entry, path, None)
            return (i, entry, path, vec)
        except Exception:
            return (i, entry, path, None)

    class _DummyLock:
        def __enter__(self):
            pass
        def __exit__(self, *a):
            pass
    _dummy_lock = _DummyLock()

    work_items: list[tuple[int, dict, str]] = []
    for i, entry in enumerate(images):
        path = entry.get("path")
        if not path or not os.path.isfile(path):
            cid = entry.get("collection_id")
            iid = entry.get("image_id")
            if cid is not None and iid is not None:
                cand = data_dir / "downloads" / "neurovault" / f"collection_{cid}" / f"image_{iid}.nii.gz"
                if cand.exists():
                    path = str(cand)
                else:
                    for ext in (".nii.gz", "_resampled.nii.gz", ".nii"):
                        c = data_dir / "downloads" / "neurovault" / f"collection_{cid}" / f"image_{iid}{ext}"
                        if c.exists():
                            path = str(c)
                            break
            if not path or not os.path.isfile(path):
                failed += 1
                continue
        if args.fetch_missing_metadata and entry.get("image_id"):
            meta = _fetch_image_metadata(entry["image_id"])
            if not _entry_has_good_label(entry) and meta.get("contrast_definition"):
                entry["contrast_definition"] = meta["contrast_definition"]
            if not entry.get("name") and meta.get("name"):
                entry["name"] = meta["name"]
            if not entry.get("cognitive_paradigm_cogatlas_id") and meta.get("cognitive_paradigm_cogatlas_id"):
                entry["cognitive_paradigm_cogatlas_id"] = meta["cognitive_paradigm_cogatlas_id"]
            if meta.get("map_type"):
                entry["map_type"] = meta["map_type"]
            if meta:
                n_fetched += 1
            time.sleep(0.2)
        raw = (entry.get("contrast_definition") or "").strip()
        if args.max_contrast_chars and len(raw) > args.max_contrast_chars:
            n_truncated += 1
        if len(raw) > max_raw_len:
            max_raw_len = len(raw)
        work_items.append((i, entry, path))

    results: list[tuple[int, dict, str, np.ndarray | None]] = []
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for r in ex.map(_parcellate_one, work_items):
                results.append(r)
                if len(results) % 50 == 0 or len(results) == 1:
                    ok_count = sum(1 for x in results if x[3] is not None)
                    print(f"  {ok_count}/{len(work_items)} parcellated", flush=True)
    else:
        for item in work_items:
            results.append(_parcellate_one(item))
            if len(results) % 50 == 0 or len(results) == 1:
                ok_count = sum(1 for x in results if x[3] is not None)
                print(f"  {ok_count}/{len(work_items)} parcellated", flush=True)

    for i, entry, path, vec in sorted(results, key=lambda x: x[0]):
        if vec is None:
            failed += 1
            continue
        label = _get_label(entry, i, args.max_contrast_chars)
        cid = entry.get("collection_id", 0)
        if getattr(args, "exclude_atlas_collections", True) and cid in ATLAS_COLLECTION_IDS:
            continue
        if use_average_subject:
            collected[cid].append({
                "data": vec,
                "contrast_definition": entry.get("contrast_definition"),
                "cognitive_paradigm_cogatlas_id": entry.get("cognitive_paradigm_cogatlas_id"),
                "name": entry.get("name"),
                "path": path,
            })
        else:
            if cid in MAP_KIND_P_MAP:
                label = _sanitize_label_for_unsigned_maps(label)
            if cid in WM_ATLAS_COLLECTION_IDS:
                label = _improve_wm_atlas_label(label)
            if getattr(args, "pharma_add_drug", False) and cid in collection_drug:
                drug = collection_drug[cid]
                label = f"{drug}: {label}"
            meta = collection_meta.get(cid) or {}
            cname, cdesc = meta.get("name") or "", meta.get("description") or ""
            if _is_dose_related(cname, cdesc):
                dose_str = _extract_dose_from_text(f"{cname} {cdesc}")
                drug_meta = collection_drug.get(cid) or _extract_drug_from_collection_name(cname)
                parts = [label]
                if drug_meta:
                    parts.append(f"drug={drug_meta}")
                if dose_str:
                    parts.append(f"dose={dose_str}")
                if "placebo" in (cname + " " + cdesc).lower():
                    parts.append("placebo-controlled")
                if len(parts) > 1:
                    label = " | ".join(parts)
            terms.append(label)
            maps_list.append(vec)
            collection_ids.append(cid)

    if use_average_subject and collected:
        from neurolab.neurovault_ingestion import qc_filter, ingest_collection, AVERAGE_FIRST, get_sample_weight
        for cid in sorted(collected.keys()):
            maps_raw = collected[cid]
            if not maps_raw:
                continue
            # Stage 2 QC: filter bad maps before ingestion
            if not getattr(args, "no_qc", False):
                arr = np.stack([m["data"] for m in maps_raw], axis=0)
                keep = qc_filter(arr, n_parcels)
                maps_raw = [m for i, m in enumerate(maps_raw) if keep[i]]
            if not maps_raw:
                continue
            result = ingest_collection(cid, maps_raw)
            was_averaged = cid in AVERAGE_FIRST
            w = get_sample_weight(cid, was_averaged)
            for lab, vec in result.items():
                if cid in MAP_KIND_P_MAP:
                    lab = _sanitize_label_for_unsigned_maps(lab)
                if cid in WM_ATLAS_COLLECTION_IDS:
                    lab = _improve_wm_atlas_label(lab)
                if getattr(args, "pharma_add_drug", False) and cid in collection_drug:
                    drug = collection_drug[cid]
                    lab = f"{drug}: {lab}"
                meta = collection_meta.get(cid) or {}
                cname, cdesc = meta.get("name") or "", meta.get("description") or ""
                if _is_dose_related(cname, cdesc):
                    dose_str = _extract_dose_from_text(f"{cname} {cdesc}")
                    drug_meta = collection_drug.get(cid) or _extract_drug_from_collection_name(cname)
                    parts = [lab]
                    if drug_meta:
                        parts.append(f"drug={drug_meta}")
                    if dose_str:
                        parts.append(f"dose={dose_str}")
                    if "placebo" in (cname + " " + cdesc).lower():
                        parts.append("placebo-controlled")
                    if len(parts) > 1:
                        lab = " | ".join(parts)
                label_out = f"[col{cid}] {lab}" if not getattr(args, "strip_col_prefix", True) else lab
                terms.append(label_out)
                maps_list.append(vec)
                collection_ids.append(cid)
        print(f"Average-subject-level: {len(terms)} terms from {sum(len(v) for v in collected.values())} images across {len(collected)} collections.", flush=True)
    elif not terms:
        print("No maps could be parcellated.", file=sys.stderr)
        return 1

    # Stage 2 QC for non-average path (average path does QC per-collection above)
    if not use_average_subject and not getattr(args, "no_qc", False):
        from neurolab.neurovault_ingestion import qc_filter
        arr = np.stack(maps_list, axis=0)
        keep = qc_filter(arr, n_parcels)
        n_before = len(terms)
        terms = [t for i, t in enumerate(terms) if keep[i]]
        maps_list = [m for i, m in enumerate(maps_list) if keep[i]]
        collection_ids = [c for i, c in enumerate(collection_ids) if keep[i]]
        if len(terms) < n_before:
            print(f"QC filter: kept {len(terms)}/{n_before} maps (rejected {n_before - len(terms)}).", flush=True)
        if not terms:
            print("No maps passed QC.", file=sys.stderr)
            return 1

    if not use_average_subject and getattr(args, "cluster_by_description", False):
        # Group by exact contrast_definition; one term per unique description, map = mean of maps in group
        from collections import defaultdict
        groups: dict[str, list[tuple[np.ndarray, int]]] = defaultdict(list)
        for lab, vec, cid in zip(terms, maps_list, collection_ids):
            groups[lab].append((vec, cid))
        terms = []
        maps_list = []
        collection_ids = []
        for lab in sorted(groups.keys()):
            vecs = [x[0] for x in groups[lab]]
            cids = [x[1] for x in groups[lab]]
            terms.append(lab)
            maps_list.append(np.mean(vecs, axis=0).astype(np.float64))
            collection_ids.append(cids[0])
        term_maps = np.stack(maps_list, axis=0)
        n_images = sum(len(v) for v in groups.values())
        print(f"Clustered by description: {len(terms)} unique terms (from {n_images} images averaged).", flush=True)
    else:
        terms = _make_labels_unique(terms)
        term_maps = np.stack(maps_list, axis=0)

    # Ensure collection_ids matches terms (for non-average without cluster, we have it from loop)
    if len(collection_ids) != len(terms):
        collection_ids = [0] * len(terms)  # fallback when lengths diverge
    assert term_maps.shape[0] == len(terms) and term_maps.shape[1] == n_parcels

    # Stage 4: Z-score each map across parcels (unless --no-zscore)
    if not getattr(args, "no_zscore", False):
        from neurolab.neurovault_ingestion import zscore_maps
        term_maps = zscore_maps(term_maps, axis=1)
        print("Applied Z-score normalization across parcels.", flush=True)

    # Per-term sample weights for training (meta 2×, group 1×, subject-averaged 0.8×)
    from neurolab.neurovault_ingestion import AVERAGE_FIRST, get_sample_weight
    term_sample_weights = np.array([
        get_sample_weight(cid, cid in AVERAGE_FIRST) for cid in collection_ids
    ], dtype=np.float32)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(_repo_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    with open(out_dir / "term_collection_ids.pkl", "wb") as f:
        pickle.dump(collection_ids, f)
    with open(out_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump(term_sample_weights.tolist(), f)
    # Provenance: was each collection averaged at build time? (not inferred from labels)
    collection_was_averaged = {
        cid: (use_average_subject and cid in AVERAGE_FIRST)
        for cid in set(collection_ids)
    }
    with open(out_dir / "collection_provenance.json", "w") as f:
        json.dump({str(k): {"was_averaged": v} for k, v in sorted(collection_was_averaged.items())}, f, indent=2)

    # Collection metadata (name, description, map_kind, dose_related) for LLM relabeling
    if manifest_path.exists() or collection_meta:
        try:
            if not collection_meta and manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                for c in mdata.get("collections_meta") or []:
                    cid = c.get("id")
                    if cid is not None:
                        collection_meta[int(cid)] = {"name": c.get("name") or "", "description": c.get("description") or ""}
            collection_metadata = {}
            for cid, meta in collection_meta.items():
                cname = meta.get("name") or ""
                cdesc = (meta.get("description") or "")[:2000]
                if cid in MAP_KIND_P_MAP:
                    map_kind = "p_map"
                elif cid in MAP_KIND_SUBNETWORK or cid in EXCLUDE_FROM_CACHE_COLLECTION_IDS:
                    map_kind = "subnetwork" if cid in MAP_KIND_SUBNETWORK else "roi_mask"
                elif cid in MAP_KIND_ROI_MASK:
                    map_kind = "roi_mask"
                else:
                    map_kind = "activation"
                collection_metadata[str(cid)] = {
                    "name": cname,
                    "description": cdesc,
                    "map_kind": map_kind,
                    "dose_related": _is_dose_related(cname, meta.get("description") or ""),
                }
            if collection_metadata:
                with open(out_dir / "collection_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(collection_metadata, f, indent=2)
        except Exception:
            pass
    print(f"Saved {len(terms)} terms x {n_parcels} parcels -> {out_dir}", flush=True)
    if failed:
        print(f"  ({failed} images skipped or failed)", flush=True)
    if n_truncated:
        print(f"  ({n_truncated} contrast_definitions truncated to {args.max_contrast_chars} chars; max raw length seen: {max_raw_len})", flush=True)
    elif max_raw_len > 0:
        if args.max_contrast_chars and args.max_contrast_chars > 0:
            print(f"  (contrast_definition max length seen: {max_raw_len} chars; none truncated)", flush=True)
        else:
            print(f"  (contrast_definition used verbatim; max length seen: {max_raw_len} chars)", flush=True)
    print("Terms come from contrast_definition only; images without it were skipped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
