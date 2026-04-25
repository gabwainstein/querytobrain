#!/usr/bin/env python3
"""
Build neuromaps annotation cache: fetch annotations (e.g. receptors, metabolism),
parcellate to pipeline atlas (Glasser+Tian, 392), save matrix + labels for biological enrichment.

Uses all MNI152 volumetric annotations (format='volume'); excludes EEG/MEG (surface-only).
If that dir exists and contains fetched files, no download. Otherwise neuromaps downloads
into it. To pre-fetch all needed data into the repo, run first:
  python neurolab/scripts/download_neuromaps_data.py

Run from querytobrain root:
  python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache
  python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache --tags receptors --max-annot 50
"""
import argparse
import os
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_NEUROMAPS_DATA_DIR = os.path.join(_repo_root, "neurolab", "data", "neuromaps_data")

# Map neuromaps source (study) -> map type for labels. User wants type (PET, Gene, etc.), not author.
_SOURCE_TO_MAP_TYPE = {
    "abagen": "Gene",
    "hcps1200": "Structural",
    "neurosynth": "Cognitive",
    "satterthwaite2014": "Perfusion",
}

# Map technical desc -> human-readable label (full name + receptor/target). From neuromaps listofmaps.
_DESC_TO_LABEL = {
    "abp688": "ABP688 binding to mGluR5 (glutamate receptor)",
    "altanserin": "Altanserin binding to 5-HT2A (serotonin receptor)",
    "az10419369": "AZ10419369 binding to 5-HT1B (serotonin receptor)",
    "carfentanil": "Carfentanil binding to mu-opioid receptor",
    "cimbi36": "CIMBI-36 binding to 5-HT2A (serotonin receptor)",
    "cogpc1": "Principal component 1 of cognitive gradient (NeuroSynth meta-analysis)",
    "cumi101": "CUMI-101 binding to 5-HT1A (serotonin receptor)",
    "dasb": "DASB binding to serotonin transporter (5-HTT)",
    "fallypride": "Fallypride binding to D2/D3 (dopamine receptor)",
    "feobv": "FEOBV binding to VAChT (vesicular acetylcholine transporter)",
    "fepe2i": "FE-PE2I binding to dopamine transporter (DAT)",
    "flb457": "FLB-457 binding to D2/D3 (dopamine receptor)",
    "flubatine": "Flubatine binding to alpha4beta2 (nicotinic acetylcholine receptor)",
    "flumazenil": "Flumazenil binding to GABA-A receptor",
    "fmpepd2": "FMPEP-d2 binding to cannabinoid CB1 receptor",
    "fpcit": "FP-CIT binding to dopamine transporter (DAT)",
    "gsk189254": "GSK-189254 binding to H3 (histamine receptor)",
    "gsk215083": "GSK-215083 binding to 5-HT2A (serotonin receptor)",
    "lsn3172176": "LSN-3172176 binding to M1 (muscarinic acetylcholine receptor)",
    "ly2795050": "LY-2795050 binding to D4 (dopamine receptor)",
    "madam": "MADAM binding to serotonin transporter (5-HTT)",
    "meancbf": "Mean cerebral blood flow (perfusion)",
    "methylreboxetine": "Methylreboxetine binding to NET (norepinephrine transporter)",
    "mrb": "O-methylreboxetine (MRB) binding to NET (norepinephrine transporter)",
    "omar": "OMAR binding to CB1 (cannabinoid receptor)",
    "p943": "P943 binding to 5-HT1A (serotonin receptor)",
    "raclopride": "Raclopride binding to D2 (dopamine receptor)",
    "sb207145": "SB207145 binding to 5-HT4 (serotonin receptor)",
    "sch23390": "SCH23390 binding to D1 (dopamine receptor)",
    "ucbj": "UCB-J binding to SV2A (synaptic vesicle protein, synapse marker)",
    "way100635": "WAY-100635 binding to 5-HT1A (serotonin receptor)",
    "cmrglc": "Glucose metabolism",
    "genepc1": "Principal component 1 of gene expression (Allen Human Brain Atlas)",
}

try:
    from neuromaps.datasets import available_annotations, fetch_annotation
except ImportError:
    print("Install neuromaps: pip install neuromaps", file=sys.stderr)
    sys.exit(1)

try:
    from nilearn import datasets as nilearn_datasets
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib
except ImportError as e:
    print(f"Install nilearn and nibabel: {e}", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Build neuromaps annotation cache (parcellated)")
    parser.add_argument(
        "--cache-dir",
        default="neurolab/data/neuromaps_cache",
        help="Output directory for annotation_maps.npz and annotation_labels.pkl",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Filter by tags (e.g. receptors). Empty = all MNI152 annotations.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for parcellation (default 1)")
    parser.add_argument(
        "--space",
        default="MNI152",
        help="Coordinate space (MNI152 for volumetric parcellation)",
    )
    parser.add_argument(
        "--max-annot",
        type=int,
        default=0,
        help="Max annotations to include (0 = all matching). Use 20-50 for quick test.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Raw neuromaps annotation data directory (default: {os.path.basename(DEFAULT_NEUROMAPS_DATA_DIR)} in repo)",
    )
    parser.add_argument(
        "--zscore-separate",
        action="store_true",
        default=True,
        help="Z-score cortex and subcortex separately for PET/receptor maps (default: True)",
    )
    parser.add_argument("--no-zscore-separate", action="store_false", dest="zscore_separate")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    out_npz = os.path.join(cache_dir, "annotation_maps.npz")
    out_pkl = os.path.join(cache_dir, "annotation_labels.pkl")

    kwargs = {"space": args.space, "format": "volume"}
    if args.tags:
        kwargs["tags"] = args.tags
    # format='volume' excludes EEG/MEG (surface-only in fsLR/fsaverage)
    print("Listing neuromaps annotations...", kwargs)
    available = list(available_annotations(**kwargs))
    # All MNI152 volumetric (1mm, 2mm, 3mm) — resample_to_atlas handles any resolution
    available = [a for a in available if len(a) >= 4 and "mm" in str(a[3])]
    if not available:
        available = list(available_annotations(**kwargs))
    if args.max_annot and len(available) > args.max_annot:
        available = available[: args.max_annot]
    print(f"  Fetching {len(available)} annotations...")

    data_dir = args.data_dir
    if not data_dir:
        data_dir = DEFAULT_NEUROMAPS_DATA_DIR
    elif not os.path.isabs(data_dir):
        data_dir = os.path.join(_repo_root, data_dir)
    data_dir_path = Path(data_dir).resolve()
    data_dir_path.mkdir(parents=True, exist_ok=True)
    # Use forward slashes so Windows paths like H:\b... don't get \b interpreted as backspace
    fetch_kw = {"data_dir": str(data_dir_path.as_posix())}
    print(f"  Data dir: {data_dir_path}")

    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from neurolab.parcellation import get_masker, get_n_parcels, resample_to_atlas, zscore_cortex_subcortex_separately
    n_parcels = get_n_parcels()
    masker = get_masker()
    masker.fit()
    n_jobs = max(1, getattr(args, "n_jobs", 1))
    masker_lock = threading.Lock() if n_jobs > 1 else None

    def _process_ann(ann):
        try:
            kw = {"return_single": False, **fetch_kw}
            if len(ann) >= 1:
                kw["source"] = ann[0]
            if len(ann) >= 2:
                kw["desc"] = ann[1]
            if len(ann) >= 3:
                kw["space"] = ann[2]
            if len(ann) >= 4:
                if "mm" in str(ann[3]) or ann[3] in ("1mm", "2mm"):
                    kw["res"] = ann[3]
                else:
                    kw["den"] = ann[3]
            result = fetch_annotation(**kw)
            if isinstance(result, dict):
                keys = list(result.keys())
                paths = result[keys[0]] if keys else []
            else:
                paths = result if isinstance(result, list) else [result]
            if not paths:
                return None
            sub = data_dir_path / "annotations" / ann[0] / ann[1]
            nii = list(sub.rglob("*.nii*")) if sub.exists() else []
            if not nii:
                path = paths[0] if isinstance(paths[0], str) else paths[0][0]
                path = str(Path(path).resolve()) if path and Path(path).exists() else ""
            else:
                path = str(nii[0].resolve())
            if not path or not Path(path).exists():
                return None
            with (masker_lock or _noop_lock):
                img = nib.load(path)
                img = resample_to_atlas(img)
                parcellated = masker.transform(img).ravel()
            if parcellated.shape[0] != n_parcels:
                return None
            if args.zscore_separate:
                parcellated = zscore_cortex_subcortex_separately(parcellated)
            desc = str(ann[1]) if len(ann) >= 2 else str(ann)
            human_desc = _DESC_TO_LABEL.get(desc, desc)
            map_type = "PET" if args.tags and "receptors" in args.tags else _SOURCE_TO_MAP_TYPE.get(ann[0] if len(ann) >= 1 else "", "PET")
            return (parcellated.astype(np.float64), f"{map_type}: {human_desc}")
        except Exception as e:
            print(f"  Skip {ann}: {e}")
            return None

    class _NoopLock:
        def __enter__(self):
            pass
        def __exit__(self, *a):
            pass
    _noop_lock = _NoopLock()

    maps_list = []
    labels_list = []
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for out in ex.map(_process_ann, available):
                if out is not None:
                    maps_list.append(out[0])
                    labels_list.append(out[1])
    else:
        for ann in available:
            out = _process_ann(ann)
            if out is not None:
                maps_list.append(out[0])
                labels_list.append(out[1])

    if not maps_list:
        print("No annotations parcellated. Try without --tags or check neuromaps docs.", file=sys.stderr)
        sys.exit(1)

    matrix = np.array(maps_list)
    # Already z-scored cortex/subcortex separately per-annotation above (--zscore-separate default).
    # PET/receptor maps have different scales in cortex vs subcortex.
    np.savez_compressed(out_npz, matrix=matrix)
    with open(out_pkl, "wb") as f:
        pickle.dump(labels_list, f)
    print(f"Saved {matrix.shape[0]} x {n_parcels} to {out_npz}")
    print(f"Labels: {out_pkl}")
    print("Use NeuromapsEnrichment(cache_dir=<this cache_dir>) for enrichment.")


if __name__ == "__main__":
    main()
