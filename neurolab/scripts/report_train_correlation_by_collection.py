#!/usr/bin/env python3
"""
Report train recovery and test generalization by source, NeuroVault collection,
abagen gene, and ENIGMA disorder.
Uses split_info.train_term_correlations and test_term_correlations (no model load needed).

Usage:
  python neurolab/scripts/report_train_correlation_by_collection.py
  python neurolab/scripts/report_train_correlation_by_collection.py --model-dir neurolab/data/embedding_model_openai_100ep_relabeled
"""
import argparse
import csv
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
_data = repo / "neurolab" / "data"


def _gene_from_abagen_term(term: str) -> str | None:
    """Extract gene symbol from abagen term. Handles both formats:
    - 'Gene: HTR2A ... (HTR2A)' -> 'HTR2A'
    - 'Gene: HTR2A (serotonin 2A receptor), serotonin signaling' -> 'HTR2A' (first token after Gene:)
    """
    if not term or "Gene:" not in term:
        return None
    rest = term.replace("Gene:", "").strip()
    if not rest:
        return None
    # First token is typically the symbol (e.g. HTR2A, DRD2)
    first = rest.split()[0] if rest.split() else ""
    if first and first[0].isupper() and first.isalnum():
        return first
    m = re.search(r"\(([A-Z0-9][A-Za-z0-9_-]*)\)\s*$", term)
    return m.group(1) if m else None


def _abagen_subcollection(term: str, receptor_kb_genes: set) -> str:
    """Return abagen subcollection: receptors, PCs, or others."""
    if not term or "Gene:" not in term:
        return "others"
    # Gene expression gradients: various label styles (current, hybrid, standard, brain_context, distinctive, short, dominant, distinct)
    t = term.replace("Gene:", "").strip()
    if re.search(r"Gene expression gradient|Principal cortical gradient|Cortical gene expression gradient|Dominant gene expression axis", t, re.I):
        return "PCs"
    if re.search(r"^(Sensorimotor-association|Metabolic and oxidative|Developmental and synaptic|Cognitive metabolism|Developmental plasticity) (gradient|axis)", t, re.I):
        return "PCs"
    if re.search(r"Primary cortex gradient|Metabolic gradient:|Developmental gradient:", t, re.I):
        return "PCs"
    gene = _gene_from_abagen_term(term)
    if gene and receptor_kb_genes and gene.upper() in receptor_kb_genes:
        return "receptors"
    return "others"


def _disorder_from_enigma_term(term: str) -> str | None:
    """Extract disorder from ENIGMA term, e.g. 'Structural: schizophrenia cortical thickness' -> 'schizophrenia'."""
    if not term or "Structural:" not in term:
        return None
    s = term.replace("Structural:", "").strip()
    for sep in (" cortical ", " subcortical ", " surface area "):
        if sep in s.lower():
            return s.split(sep)[0].strip()
    return s.split()[0] if s else None


def _build_term_to_collection() -> tuple[dict, dict]:
    """Map merged_sources term -> collection_id and term -> (cid, is_pharma).
    Returns (term_to_cid, term_to_cid_pharma) where term_to_cid_pharma[term] = (cid, True|False).
    Prefers relabeled/improved pharma cache when present (merge uses those).
    """
    term_to_cid = {}
    term_to_cid_pharma = {}
    # NeuroVault task: prefer relabeled > improved > base
    nv_candidates = ["neurovault_cache_relabeled", "neurovault_cache_improved", "neurovault_cache"]
    for cand in nv_candidates:
        cache = _data / cand
        if (cache / "term_vocab.pkl").exists() and (cache / "term_collection_ids.pkl").exists():
            nv_terms = pickle.load(open(cache / "term_vocab.pkl", "rb"))
            nv_cids = pickle.load(open(cache / "term_collection_ids.pkl", "rb"))
            for t, cid in zip(nv_terms, nv_cids):
                key = ("fMRI: " + t.strip()).strip()
                if key not in term_to_cid:
                    term_to_cid[key] = cid
                    term_to_cid_pharma[key] = (cid, False)
            break
    # NeuroVault pharma: prefer relabeled > improved > base
    pharma_candidates = ["neurovault_pharma_cache_relabeled", "neurovault_pharma_cache_improved", "neurovault_pharma_cache"]
    for cand in pharma_candidates:
        cache = _data / cand
        if (cache / "term_vocab.pkl").exists() and (cache / "term_collection_ids.pkl").exists():
            nv_terms = pickle.load(open(cache / "term_vocab.pkl", "rb"))
            nv_cids = pickle.load(open(cache / "term_collection_ids.pkl", "rb"))
            for t, cid in zip(nv_terms, nv_cids):
                key = ("fMRI: " + t.strip()).strip()
                if key not in term_to_cid:
                    term_to_cid[key] = cid
                    term_to_cid_pharma[key] = (cid, True)
            break
    return term_to_cid, term_to_cid_pharma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="neurolab/data/embedding_model", help="Model dir with split_info.pkl")
    ap.add_argument("--cache-dir", default="neurolab/data/merged_sources", help="Cache with term_vocab, term_sources")
    ap.add_argument("--compact", action="store_true", help="Print only one list: collection/source, train_r, test_r, n")
    ap.add_argument("--csv", default=None, metavar="PATH", help="Append results to CSV (columns: timestamp, run, source_or_collection, train_r, test_r, n)")
    args = ap.parse_args()

    model_dir = repo / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    cache_dir = repo / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)

    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    term_sources = pickle.load(open(cache_dir / "term_sources.pkl", "rb"))
    term_to_idx = {t: i for i, t in enumerate(terms)}

    split_info = pickle.load(open(model_dir / "split_info.pkl", "rb"))
    train_term_corrs = split_info.get("train_term_correlations") or []
    test_term_corrs = split_info.get("test_term_correlations") or []

    term_to_cid, term_to_cid_pharma = _build_term_to_collection()
    nv_sources = {"neurovault", "neurovault_pharma"}

    receptor_kb_genes = set()
    try:
        import sys
        sys.path.insert(0, str(repo))
        from neurolab.receptor_kb import load_receptor_genes
        receptor_kb_genes = set(g.upper() for g in load_receptor_genes())
    except Exception:
        pass

    def _by_source_and_subgroup(corr_list, label):
        corrs_by_source = defaultdict(list)
        corrs_by_collection = defaultdict(list)
        corrs_by_abagen_gene = defaultdict(list)
        corrs_by_abagen_receptor_kb = []
        corrs_by_abagen_other = []
        corrs_by_abagen_subcollection = defaultdict(list)  # receptors, PCs, others
        corrs_by_source_subcollection = defaultdict(list)  # "direct", "neurovault", "abagen/receptors", etc.
        corrs_by_enigma_disorder = defaultdict(list)
        for term, r in corr_list:
            if r is None or not np.isfinite(r):
                continue
            idx = term_to_idx.get(term)
            src = term_sources[idx] if idx is not None and idx < len(term_sources) else "unknown"
            corrs_by_source[src].append(r)
            # Collection/subcollection for unified view (abagen split; others as single bucket)
            if src == "abagen":
                sub = _abagen_subcollection(term, receptor_kb_genes)
                corrs_by_abagen_subcollection[sub].append(r)
                corrs_by_source_subcollection[f"abagen/{sub}"].append(r)
            elif src in nv_sources:
                corrs_by_source_subcollection["neurovault"].append(r)
            elif src == "enigma":
                corrs_by_source_subcollection["enigma"].append(r)
            elif src == "neuromaps":
                corrs_by_source_subcollection["neuromaps"].append(r)
            else:
                corrs_by_source_subcollection[src].append(r)
            if src in nv_sources and term_to_cid_pharma:
                info = term_to_cid_pharma.get(term)
                if info is not None:
                    cid, is_pharma = info
                    key = (cid, "pharma" if is_pharma else "task")
                    corrs_by_collection[key].append(r)
            if src == "abagen":
                gene = _gene_from_abagen_term(term)
                if gene:
                    corrs_by_abagen_gene[gene].append(r)
                    if receptor_kb_genes and gene.upper() in receptor_kb_genes:
                        corrs_by_abagen_receptor_kb.append(r)
                    else:
                        corrs_by_abagen_other.append(r)
                else:
                    corrs_by_abagen_gene["(gradient/other)"].append(r)
                    corrs_by_abagen_other.append(r)  # gradient PCs etc. are not receptor KB
            if src == "enigma":
                disorder = _disorder_from_enigma_term(term)
                if disorder:
                    corrs_by_enigma_disorder[disorder].append(r)

        # Collection and subcollection summary (abagen receptors, PCs, others; neurovault by col; etc.)
        print(f"\n{label} by COLLECTION / SUBCOLLECTION:")
        print("-" * 50)
        for key in sorted(corrs_by_source_subcollection.keys()):
            corrs = corrs_by_source_subcollection[key]
            if len(corrs) >= 1:
                print(f"  {key}: {np.mean(corrs):.4f} (n={len(corrs)})")
        print(f"\n{label} by SOURCE:")
        print("-" * 50)
        for src in sorted(corrs_by_source.keys()):
            corrs = corrs_by_source[src]
            print(f"  {src}: {np.mean(corrs):.4f} (n={len(corrs)})")
        if corrs_by_abagen_receptor_kb or corrs_by_abagen_other:
            print(f"\n{label} by ABAGEN / RECEPTOR KB:")
            print("-" * 50)
            if corrs_by_abagen_receptor_kb:
                print(f"  abagen (receptor KB): {np.mean(corrs_by_abagen_receptor_kb):.4f} (n={len(corrs_by_abagen_receptor_kb)})")
            if corrs_by_abagen_other:
                print(f"  abagen (other genes): {np.mean(corrs_by_abagen_other):.4f} (n={len(corrs_by_abagen_other)})")
        _MIN_N_FOR_SUBGROUP = 5  # Only report subgroups with n>=5; singletons are high-variance noise
        if corrs_by_collection:
            print(f"\n{label} by NEUROVAULT COLLECTION (n>={_MIN_N_FOR_SUBGROUP}):")
            print("-" * 50)
            subgroups = [(k, corrs) for k, corrs in sorted(corrs_by_collection.items()) if len(corrs) >= _MIN_N_FOR_SUBGROUP]
            singletons = [(k, len(corrs)) for k, corrs in sorted(corrs_by_collection.items()) if len(corrs) < _MIN_N_FOR_SUBGROUP]
            for (cid, col_type), corrs in subgroups:
                print(f"  Collection {cid} ({col_type}): {np.mean(corrs):.4f} (n={len(corrs)})")
            if singletons:
                all_singleton_corrs = [r for k, corrs in corrs_by_collection.items() if len(corrs) < _MIN_N_FOR_SUBGROUP for r in corrs]
                print(f"  (singletons: {len(singletons)} collections, {sum(n for _, n in singletons)} terms; median r={np.median(all_singleton_corrs):.4f}, IQR=[{np.percentile(all_singleton_corrs, 25):.4f}, {np.percentile(all_singleton_corrs, 75):.4f}])")
        if corrs_by_abagen_gene:
            print(f"\n{label} by ABAGEN GENE (n>={_MIN_N_FOR_SUBGROUP}):")
            print("-" * 50)
            subgroups = [(g, corrs) for g, corrs in sorted(corrs_by_abagen_gene.items()) if len(corrs) >= _MIN_N_FOR_SUBGROUP]
            for gene, corrs in subgroups:
                print(f"  {gene}: {np.mean(corrs):.4f} (n={len(corrs)})")
            singletons = [(g, len(corrs)) for g, corrs in corrs_by_abagen_gene.items() if len(corrs) < _MIN_N_FOR_SUBGROUP]
            if singletons:
                all_singleton_corrs = [r for g, corrs in corrs_by_abagen_gene.items() if len(corrs) < _MIN_N_FOR_SUBGROUP for r in corrs]
                print(f"  (singletons: {len(singletons)} genes, {sum(n for _, n in singletons)} terms; median r={np.median(all_singleton_corrs):.4f}, IQR=[{np.percentile(all_singleton_corrs, 25):.4f}, {np.percentile(all_singleton_corrs, 75):.4f}])")
        if corrs_by_enigma_disorder:
            print(f"\n{label} by ENIGMA DISORDER (n>={_MIN_N_FOR_SUBGROUP}):")
            print("-" * 50)
            subgroups = [(d, corrs) for d, corrs in sorted(corrs_by_enigma_disorder.items()) if len(corrs) >= _MIN_N_FOR_SUBGROUP]
            for disorder, corrs in subgroups:
                print(f"  {disorder}: {np.mean(corrs):.4f} (n={len(corrs)})")
            singletons = [(d, len(corrs)) for d, corrs in corrs_by_enigma_disorder.items() if len(corrs) < _MIN_N_FOR_SUBGROUP]
            if singletons:
                all_singleton_corrs = [r for d, corrs in corrs_by_enigma_disorder.items() if len(corrs) < _MIN_N_FOR_SUBGROUP for r in corrs]
                print(f"  (singletons: {len(singletons)} disorders, {sum(n for _, n in singletons)} terms; median r={np.median(all_singleton_corrs):.4f}, IQR=[{np.percentile(all_singleton_corrs, 25):.4f}, {np.percentile(all_singleton_corrs, 75):.4f}])")

    if args.compact:
        # One compact list: source/collection, train_r, test_r, n
        corrs_by_source_train = defaultdict(list)
        corrs_by_source_test = defaultdict(list)
        corrs_by_collection_train = defaultdict(list)
        corrs_by_collection_test = defaultdict(list)
        for term, r in (train_term_corrs or []):
            if r is None or not np.isfinite(r):
                continue
            idx = term_to_idx.get(term)
            src = term_sources[idx] if idx is not None and idx < len(term_sources) else "unknown"
            corrs_by_source_train[src].append(r)
            if src in nv_sources and term_to_cid_pharma:
                info = term_to_cid_pharma.get(term)
                if info is not None:
                    cid, is_pharma = info
                    key = f"col{cid}" + ("_pharma" if is_pharma else "")
                    corrs_by_collection_train[key].append(r)
        for term, r in (test_term_corrs or []):
            if r is None or not np.isfinite(r):
                continue
            idx = term_to_idx.get(term)
            src = term_sources[idx] if idx is not None and idx < len(term_sources) else "unknown"
            corrs_by_source_test[src].append(r)
            if src in nv_sources and term_to_cid_pharma:
                info = term_to_cid_pharma.get(term)
                if info is not None:
                    cid, is_pharma = info
                    key = f"col{cid}" + ("_pharma" if is_pharma else "")
                    corrs_by_collection_test[key].append(r)
        all_keys = sorted(set(corrs_by_source_train) | set(corrs_by_source_test) | set(corrs_by_collection_train) | set(corrs_by_collection_test))
        # Prefer collection-level when available; else source
        by_key = {}
        for k in all_keys:
            if k.startswith("col"):
                tr = np.mean(corrs_by_collection_train[k]) if corrs_by_collection_train[k] else np.nan
                te = np.mean(corrs_by_collection_test[k]) if corrs_by_collection_test[k] else np.nan
                n = len(corrs_by_collection_train[k]) + len(corrs_by_collection_test[k])
            else:
                tr = np.mean(corrs_by_source_train[k]) if corrs_by_source_train[k] else np.nan
                te = np.mean(corrs_by_source_test[k]) if corrs_by_source_test[k] else np.nan
                n = len(corrs_by_source_train[k]) + len(corrs_by_source_test[k])
            by_key[k] = (tr, te, n)
        # Sort: sources first, then collections by ID
        sources = [k for k in all_keys if not k.startswith("col")]
        colls = [k for k in all_keys if k.startswith("col")]
        def _cid(x):
            s = x.replace("col", "").replace("_pharma", "")
            return int(s) if s.isdigit() else 0
        colls.sort(key=lambda x: (_cid(x), x))
        for k in sources + colls:
            tr, te, n = by_key[k]
            print(f"{k}: train_r={tr:.4f} test_r={te:.4f} n={n}")

        if args.csv:
            csv_path = Path(args.csv)
            if not csv_path.is_absolute():
                csv_path = repo / csv_path
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            run_id = model_dir.name if isinstance(model_dir, Path) else str(model_dir)
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestamp", "run", "source_or_collection", "train_r", "test_r", "n"])
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                for k in sources + colls:
                    tr, te, n = by_key[k]
                    w.writerow([ts, run_id, k, f"{tr:.4f}" if np.isfinite(tr) else "", f"{te:.4f}" if np.isfinite(te) else "", n])
            print(f"Appended to {csv_path}")
        return

    print(f"Model: {model_dir}")
    _by_source_and_subgroup(train_term_corrs, "Train recovery")
    _by_source_and_subgroup(test_term_corrs, "Test generalization")


if __name__ == "__main__":
    main()
