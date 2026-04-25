#!/usr/bin/env python3
"""
Analyze memorizer recovery (MSE, correlation) per source/category.
Reports per-dataset breakdown for merged_sources.
"""
import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
MAP_TYPES = ["fmri_activation", "structural", "pet_receptor"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="neurolab/data/embedding_model_memorizer_openai")
    parser.add_argument("--cache-dir", default="neurolab/data/merged_sources")
    args = parser.parse_args()

    model_dir = repo_root / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    cache_dir = repo_root / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)

    # Load cache
    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    with open(cache_dir / "term_vocab.pkl", "rb") as f:
        terms = pickle.load(f)
    with open(cache_dir / "term_sources.pkl", "rb") as f:
        term_sources = pickle.load(f)
    with open(cache_dir / "term_map_types.pkl", "rb") as f:
        term_map_types = pickle.load(f)

    n = len(terms)
    n_parcels = term_maps.shape[1]

    # Load model and run predictions
    cfg = pickle.load(open(model_dir / "config.pkl", "rb"))
    all_emb = np.load(model_dir / "training_embeddings.npy").astype(np.float32)
    type_indices = [MAP_TYPES.index(t) if t in MAP_TYPES else 0 for t in term_map_types]
    type_oh = np.eye(len(MAP_TYPES), dtype=np.float32)[type_indices]
    X = np.hstack([all_emb, type_oh]).astype(np.float32)
    dim = X.shape[1]

    import torch
    import torch.nn as nn
    head_hidden = cfg["head_hidden"]
    head_hidden2 = cfg.get("head_hidden2", 0)
    n_out = n_parcels
    use_gene = cfg.get("use_gene_head", False) and (model_dir / "gene_pca.pkl").exists()

    model = nn.Sequential(
        nn.Linear(dim, head_hidden), nn.ReLU(),
        nn.Linear(head_hidden, head_hidden2), nn.ReLU(),
        nn.Linear(head_hidden2, n_out)
    ) if head_hidden2 else nn.Sequential(
        nn.Linear(dim, head_hidden), nn.ReLU(),
        nn.Linear(head_hidden, n_out)
    )
    model.load_state_dict(torch.load(model_dir / "head_weights.pt", map_location="cpu"))
    model.eval()

    with open(cache_dir / "abagen_term_indices.pkl", "rb") as f:
        abagen_idx = set(pickle.load(f))
    abagen_set = {terms[i] for i in abagen_idx}

    gene_pca = None
    gene_head = None
    if use_gene:
        gene_pca = pickle.load(open(model_dir / "gene_pca.pkl", "rb"))
        gsd = torch.load(model_dir / "gene_head_weights.pt", map_location="cpu")
        gene_head = nn.Sequential(
            nn.Linear(dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden2), nn.ReLU(),
            nn.Linear(head_hidden2, gsd["4.weight"].shape[0])
        )
        gene_head.load_state_dict(gsd)
        gene_head.eval()

    with torch.no_grad():
        pred_main = model(torch.from_numpy(X)).numpy()
        if use_gene and gene_head is not None:
            pred_gene = gene_head(torch.from_numpy(X)).numpy()
            pred_400 = np.zeros((n, n_parcels), dtype=np.float32)
            for i in range(n):
                if terms[i] in abagen_set:
                    pred_400[i] = gene_pca.inverse_transform(pred_gene[i : i + 1]).ravel()
                else:
                    pred_400[i] = pred_main[i]
        else:
            pred_400 = pred_main

    targets = term_maps.astype(np.float64)
    pred_400 = pred_400.astype(np.float64)

    # Per-source and per-map-type metrics
    by_source = defaultdict(lambda: {"mse": [], "corr": []})
    by_map_type = defaultdict(lambda: {"mse": [], "corr": []})
    by_source_map = defaultdict(lambda: {"mse": [], "corr": []})

    for i in range(n):
        src = term_sources[i] if i < len(term_sources) else "unknown"
        mtype = term_map_types[i] if i < len(term_map_types) else "unknown"
        mse = float(np.mean((pred_400[i] - targets[i]) ** 2))
        r = np.corrcoef(pred_400[i], targets[i])[0, 1]
        r = float(r) if np.isfinite(r) else 0.0
        by_source[src]["mse"].append(mse)
        by_source[src]["corr"].append(r)
        by_map_type[mtype]["mse"].append(mse)
        by_map_type[mtype]["corr"].append(r)
        by_source_map[(src, mtype)]["mse"].append(mse)
        by_source_map[(src, mtype)]["corr"].append(r)

    def _print_table(title, data, key_label):
        print(f"\n{title}")
        print("=" * 70)
        print(f"{key_label:<25} {'N':>6} {'MSE':>8} {'RMSE':>8} {'VarExpl':>8} {'Mean r':>8}")
        print("-" * 70)
        for k in sorted(data.keys()):
            d = data[k]
            n_s = len(d["mse"])
            mse_mean = np.mean(d["mse"])
            rmse = np.sqrt(mse_mean)
            var_expl = max(0, 1 - mse_mean) * 100
            corr_mean = np.mean(d["corr"])
            print(f"{str(k):<25} {n_s:>6} {mse_mean:>8.4f} {rmse:>8.4f} {var_expl:>7.1f}% {corr_mean:>8.4f}")
        all_mse = [x for d in data.values() for x in d["mse"]]
        all_corr = [x for d in data.values() for x in d["corr"]]
        print("-" * 70)
        print(f"{'OVERALL':<25} {len(all_mse):>6} {np.mean(all_mse):>8.4f} {np.sqrt(np.mean(all_mse)):>8.4f} {(1-np.mean(all_mse))*100:>7.1f}% {np.mean(all_corr):>8.4f}")
        print("=" * 70)

    # Print tables
    _print_table("Memorizer recovery by source/collection", by_source, "Source")
    _print_table("Memorizer recovery by map type (category)", by_map_type, "Map type")

    # Source x map type cross-tabulation
    print("\nMemorizer recovery by source × map type")
    print("=" * 80)
    print(f"{'Source':<20} {'Map type':<20} {'N':>6} {'MSE':>8} {'RMSE':>8} {'VarExpl':>8} {'Mean r':>8}")
    print("-" * 80)
    for (src, mtype) in sorted(by_source_map.keys()):
        d = by_source_map[(src, mtype)]
        n_s = len(d["mse"])
        mse_mean = np.mean(d["mse"])
        rmse = np.sqrt(mse_mean)
        var_expl = max(0, 1 - mse_mean) * 100
        corr_mean = np.mean(d["corr"])
        print(f"{src:<20} {mtype:<20} {n_s:>6} {mse_mean:>8.4f} {rmse:>8.4f} {var_expl:>7.1f}% {corr_mean:>8.4f}")
    print("=" * 80)
    print("\n(VarExpl = 1 - MSE as % variance explained for z-scored maps)")


if __name__ == "__main__":
    main()
