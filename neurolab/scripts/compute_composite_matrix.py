#!/usr/bin/env python3
"""
Compute the composite similarity matrix (regression-predicted brain r) from hop + embedding distance.

Uses coefficients from brain_r_hop_embedding_regression.json. For each pair (i,j):
  composite_similarity[i,j] = predicted_r(hop[i,j], emb_dist[i,j], coef)

Saves the matrix and optionally plots decay: brain r vs composite similarity (and vs hop-only, emb-only).

Usage (from repo root):
  python neurolab/scripts/compute_composite_matrix.py \
    --kg-regression-json neurolab/data/brain_r_hop_embedding_regression.json \
    --ontology-dir neurolab/data/ontologies \
    --encoder sentence-transformers \
    --cache-dir neurolab/data/decoder_cache \
    --out-dir neurolab/data \
    --plot
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite_distance_utils import load_regression_coefficients, predicted_r  # noqa: E402
from graph_distance_correlation import _normalize, build_unified_graph  # noqa: E402
from ontology_term_dendrogram import build_hierarchy_distance_matrix  # noqa: E402
from term_r_per_embedding_distance import (  # noqa: E402
    load_embeddings_from_dir,
    encode_with_sentence_transformers,
)


def main():
    parser = argparse.ArgumentParser(description="Compute composite similarity matrix (hop + emb_dist + interaction)")
    parser.add_argument("--kg-regression-json", required=True, help="Path to brain_r_hop_embedding_regression.json")
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache", help="term_maps.npz + term_vocab.pkl")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies", help="Ontology dir for hop matrix")
    parser.add_argument("--embedding-dir", default=None, help="training_embeddings.npy dir; else use --encoder")
    parser.add_argument("--encoder", choices=("sentence-transformers",), default=None)
    parser.add_argument("--encoder-model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--max-hops", type=int, default=19, help="Exclude catch-all hop (use 0..max_hops only)")
    parser.add_argument("--out-dir", default="neurolab/data", help="Save composite_similarity.npy and terms_composite.pkl here")
    parser.add_argument("--plot", action="store_true", help="Plot brain r vs composite similarity (and hop-only, emb-only)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir) if not os.path.isabs(args.cache_dir) else repo_root / args.cache_dir
    ontology_dir = Path(args.ontology_dir) if not os.path.isabs(args.ontology_dir) else repo_root / args.ontology_dir
    out_dir = Path(args.out_dir) if not os.path.isabs(args.out_dir) else repo_root / args.out_dir
    reg_path = Path(args.kg_regression_json) if os.path.isabs(args.kg_regression_json) else repo_root / args.kg_regression_json
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else repo_root / "neurolab/data/embedding_model"
    if args.embedding_dir and not os.path.isabs(args.embedding_dir):
        embedding_dir = repo_root / args.embedding_dir

    if not reg_path.exists():
        print("Regression JSON not found:", reg_path, file=sys.stderr)
        sys.exit(1)
    coef = load_regression_coefficients(reg_path)

    # Same data loading as regress_brain_r_on_hop_and_embedding.py
    npz_path = cache_dir / "term_maps.npz"
    pkl_path = cache_dir / "term_vocab.pkl"
    if not npz_path.exists() or not pkl_path.exists():
        print("Cache not found.", file=sys.stderr)
        sys.exit(1)
    data = np.load(npz_path)
    term_maps_full = np.asarray(data["term_maps"])
    with open(pkl_path, "rb") as f:
        term_vocab = list(pickle.load(f))
    norm_to_idx = {_normalize(t): i for i, t in enumerate(term_vocab) if _normalize(t)}

    print("Building ontology graph...")
    G = build_unified_graph(str(ontology_dir))
    graph_nodes = set(G.nodes())
    cache_in_graph = {k: idx for k, idx in norm_to_idx.items() if k in graph_nodes}
    terms = list(cache_in_graph.keys())
    n_terms = len(terms)
    print(f"Terms in graph: {n_terms}")

    if not args.encoder and (Path(embedding_dir) / "training_embeddings.npy").exists():
        print("Loading embeddings from", embedding_dir, "...")
        embeddings = load_embeddings_from_dir(embedding_dir, terms)
    else:
        embeddings = None
    if embeddings is None and args.encoder == "sentence-transformers":
        print("Encoding with", args.encoder_model, "...")
        embeddings = encode_with_sentence_transformers(terms, args.encoder_model)
    if embeddings is None:
        print("No embeddings. Use --embedding-dir or --encoder sentence-transformers.", file=sys.stderr)
        sys.exit(1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    cos_sim = embeddings @ embeddings.T
    emb_dist = 1.0 - cos_sim
    np.fill_diagonal(emb_dist, 0.0)

    term_indices = [cache_in_graph[t] for t in terms]
    term_maps = term_maps_full[term_indices]
    brain_r = np.corrcoef(term_maps)
    np.fill_diagonal(brain_r, np.nan)

    matrix_cutoff = args.max_hops + 1
    print(f"Computing hop matrix (cutoff={matrix_cutoff})...")
    D = build_hierarchy_distance_matrix(G, terms, cutoff=matrix_cutoff)

    # Composite similarity matrix: r_predicted = intercept + hop*coef_hop + emb*coef_emb + hop*emb*coef_int
    composite = (
        coef["intercept"]
        + coef["hop"] * D
        + coef["emb_dist"] * emb_dist
        + coef["hop_emb_dist"] * (D * emb_dist)
    ).astype(np.float64)
    composite[D > args.max_hops] = np.nan
    np.fill_diagonal(composite, np.nan)

    os.makedirs(out_dir, exist_ok=True)
    np.save(out_dir / "composite_similarity.npy", composite)
    with open(out_dir / "terms_composite.pkl", "wb") as f:
        pickle.dump(terms, f)
    print("Saved:", out_dir / "composite_similarity.npy", out_dir / "terms_composite.pkl")

    # Correlation: brain r vs composite similarity (upper triangle, finite)
    iu, ju = np.triu_indices(n_terms, k=1)
    br_flat = brain_r[iu, ju]
    co_flat = composite[iu, ju]
    mask = np.isfinite(br_flat) & np.isfinite(co_flat) & (D[iu, ju] <= args.max_hops)
    br_flat = br_flat[mask]
    co_flat = co_flat[mask]
    r_brain_composite = np.corrcoef(br_flat, co_flat)[0, 1]
    print(f"Pearson(brain_r, composite_similarity) = {r_brain_composite:.4f}  (n_pairs={len(br_flat)})")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skip plot", file=sys.stderr)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(co_flat, br_flat, alpha=0.06, s=2, color="#1565C0", rasterized=True)
            ax.set_xlabel("Composite similarity (predicted r from hop + emb_dist + interaction)", fontsize=11)
            ax.set_ylabel("Brain-map Pearson r (actual)", fontsize=11)
            ax.set_title("Brain r vs composite similarity (%d pairs); r=%.3f" % (len(br_flat), r_brain_composite), fontsize=12)
            ax.axhline(np.nanmean(brain_r), color="#F44336", linestyle=":", linewidth=1, label="Mean brain r")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = out_dir / "brain_r_vs_composite_similarity.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print("Plot saved:", plot_path)


if __name__ == "__main__":
    main()
