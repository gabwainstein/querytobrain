#!/usr/bin/env python3
"""
Rank terms by how well the memorizer captures them (MSE, correlation).
Outputs a CSV sorted by recovery quality (best first).
"""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
MAP_TYPES = ["fmri_activation", "structural", "pet_receptor"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="neurolab/data/embedding_model_memorizer_openai")
    parser.add_argument("--cache-dir", default="neurolab/data/merged_sources")
    parser.add_argument("--output", default="neurolab/data/merged_sources_terms_ranked.csv")
    parser.add_argument("--sort-by", choices=["mse", "corr"], default="corr",
                        help="Sort by: mse (low=best) or corr (high=best)")
    args = parser.parse_args()

    model_dir = repo_root / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    cache_dir = repo_root / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    out_path = repo_root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    # Load cache
    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    term_sources = pickle.load(open(cache_dir / "term_sources.pkl", "rb"))
    term_map_types = pickle.load(open(cache_dir / "term_map_types.pkl", "rb"))

    n = len(terms)
    n_parcels = term_maps.shape[1]

    # Load model
    cfg = pickle.load(open(model_dir / "config.pkl", "rb"))
    all_emb = np.load(model_dir / "training_embeddings.npy").astype(np.float32)
    type_indices = [MAP_TYPES.index(t) if t in MAP_TYPES else 0 for t in term_map_types]
    type_oh = np.eye(len(MAP_TYPES), dtype=np.float32)[type_indices]
    X = np.hstack([all_emb, type_oh]).astype(np.float32)

    import torch
    import torch.nn as nn
    head_hidden = cfg["head_hidden"]
    head_hidden2 = cfg.get("head_hidden2", 0)
    use_gene = cfg.get("use_gene_head", False) and (model_dir / "gene_pca.pkl").exists()

    model = nn.Sequential(
        nn.Linear(X.shape[1], head_hidden), nn.ReLU(),
        nn.Linear(head_hidden, head_hidden2), nn.ReLU(),
        nn.Linear(head_hidden2, n_parcels)
    ) if head_hidden2 else nn.Sequential(
        nn.Linear(X.shape[1], head_hidden), nn.ReLU(),
        nn.Linear(head_hidden, n_parcels)
    )
    model.load_state_dict(torch.load(model_dir / "head_weights.pt", map_location="cpu"))
    model.eval()

    abagen_set = set()
    if (cache_dir / "abagen_term_indices.pkl").exists():
        abagen_idx = pickle.load(open(cache_dir / "abagen_term_indices.pkl", "rb"))
        abagen_set = {terms[i] for i in abagen_idx}

    gene_pca = None
    gene_head = None
    if use_gene:
        gene_pca = pickle.load(open(model_dir / "gene_pca.pkl", "rb"))
        gsd = torch.load(model_dir / "gene_head_weights.pt", map_location="cpu")
        gene_head = nn.Sequential(
            nn.Linear(X.shape[1], head_hidden), nn.ReLU(),
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

    # Per-term metrics
    rows = []
    for i in range(n):
        mse = float(np.mean((pred_400[i] - targets[i]) ** 2))
        r = np.corrcoef(pred_400[i], targets[i])[0, 1]
        r = float(r) if np.isfinite(r) else 0.0
        var_explained = max(0, 1 - mse) * 100
        rows.append({
            "index": i,
            "term": terms[i],
            "source": term_sources[i] if i < len(term_sources) else "",
            "map_type": term_map_types[i] if i < len(term_map_types) else "",
            "mse": round(mse, 6),
            "correlation": round(r, 6),
            "var_explained_pct": round(var_explained, 2),
        })

    # Sort: best first
    if args.sort_by == "corr":
        rows.sort(key=lambda x: -x["correlation"])
    else:
        rows.sort(key=lambda x: x["mse"])

    # Add rank
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank

    # Reorder columns
    fieldnames = ["rank", "index", "term", "source", "map_type", "mse", "correlation", "var_explained_pct"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} terms ranked by {args.sort_by} to {out_path}")
    print(f"  Best:  r={rows[0]['correlation']:.4f}, MSE={rows[0]['mse']:.4f} — {rows[0]['term'][:60]}...")
    print(f"  Worst: r={rows[-1]['correlation']:.4f}, MSE={rows[-1]['mse']:.4f} — {rows[-1]['term'][:60]}...")


if __name__ == "__main__":
    main()
