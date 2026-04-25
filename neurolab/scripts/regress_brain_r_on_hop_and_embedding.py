#!/usr/bin/env python3
"""
Regress brain-map Pearson r on **both** hierarchy (hop) and embedding distance,
plus their interaction:  r ~ 1 + hop + emb_dist + hop * emb_dist.

Uses pair-level data (one row per term pair i<j): hop(i,j), emb_dist(i,j), brain_r(i,j).
Reports OLS coefficients, R², and p-values so we can see main effects and interaction.

Usage (from repo root):
  python neurolab/scripts/regress_brain_r_on_hop_and_embedding.py --embedding-dir neurolab/data/embedding_model
  python neurolab/scripts/regress_brain_r_on_hop_and_embedding.py --encoder sentence-transformers --encoder-model NeuML/pubmedbert-base-embeddings
  python neurolab/scripts/regress_brain_r_on_hop_and_embedding.py --max-hops 15 --output-json neurolab/data/brain_r_hop_embedding_regression.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_distance_correlation import _normalize, build_unified_graph  # noqa: E402
from ontology_term_dendrogram import build_hierarchy_distance_matrix  # noqa: E402
from term_r_per_embedding_distance import (  # noqa: E402
    load_embeddings_from_dir,
    encode_with_sentence_transformers,
)


def main():
    parser = argparse.ArgumentParser(
        description="OLS: brain r ~ hop + emb_dist + hop*emb_dist (pair-level)"
    )
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache",
                        help="Cache with term_maps.npz + term_vocab.pkl")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies",
                        help="Ontology dir (same term set as hop/embedding analyses)")
    parser.add_argument("--embedding-dir", default=None,
                        help="Dir with training_embeddings.npy + training_terms.pkl")
    parser.add_argument("--encoder", choices=("sentence-transformers",), default=None,
                        help="Encode terms on the fly")
    parser.add_argument("--encoder-model", default="NeuML/pubmedbert-base-embeddings",
                        help="Model when --encoder sentence-transformers")
    parser.add_argument("--max-hops", type=int, default=19,
                        help="Max hierarchy distance; pairs with hop > this excluded (default 19; use 19 to exclude catch-all hop=20)")
    parser.add_argument("--output-json", default=None,
                        help="Write coefficients and R² to JSON")
    parser.add_argument("--scatter-out", default=None,
                        help="Save hop vs embedding distance scatter plot (e.g. neurolab/data/hop_vs_embedding_distance.png)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir) if not os.path.isabs(args.cache_dir) else Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    ontology_dir = Path(args.ontology_dir) if not os.path.isabs(args.ontology_dir) else repo_root / args.ontology_dir
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else (repo_root / "neurolab/data/embedding_model")
    if args.embedding_dir and not os.path.isabs(args.embedding_dir):
        embedding_dir = repo_root / args.embedding_dir

    # Load cache and terms in graph (same as term_r_per_embedding_distance)
    npz_path = cache_dir / "term_maps.npz"
    pkl_path = cache_dir / "term_vocab.pkl"
    if not npz_path.exists() or not pkl_path.exists():
        print("Cache not found.", file=sys.stderr)
        sys.exit(1)
    data = np.load(npz_path)
    term_maps_full = np.asarray(data["term_maps"])
    with open(pkl_path, "rb") as f:
        term_vocab = list(pickle.load(f))
    norm_to_idx = {}
    for i, t in enumerate(term_vocab):
        k = _normalize(t)
        if k:
            norm_to_idx[k] = i

    print("Building ontology graph...")
    G = build_unified_graph(str(ontology_dir))
    graph_nodes = set(G.nodes())
    cache_in_graph = {k: idx for k, idx in norm_to_idx.items() if k in graph_nodes}
    terms = list(cache_in_graph.keys())
    n_terms = len(terms)
    print(f"Terms in graph: {n_terms}")

    # Embeddings
    embeddings = None
    if (args.embedding_dir or (embedding_dir / "training_embeddings.npy").exists()) and not args.encoder:
        print("Loading embeddings from", embedding_dir, "...")
        embeddings = load_embeddings_from_dir(embedding_dir, terms)
    if embeddings is None and args.encoder == "sentence-transformers":
        print("Encoding terms with", args.encoder_model, "...")
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

    # Brain r and hop
    term_indices = [cache_in_graph[t] for t in terms]
    term_maps = term_maps_full[term_indices]
    brain_r = np.corrcoef(term_maps)
    np.fill_diagonal(brain_r, np.nan)

    # Build with cutoff = max_hops+1 so hop 0..max_hops are real; hop == max_hops+1 is catch-all (excluded)
    matrix_cutoff = args.max_hops + 1
    print(f"Computing hierarchy distance matrix (cutoff={matrix_cutoff}; excluding catch-all hop={matrix_cutoff})...")
    D = build_hierarchy_distance_matrix(G, terms, cutoff=matrix_cutoff)

    # Pair-level data (upper triangle only)
    iu, ju = np.triu_indices(n_terms, k=1)
    hop_flat = D[iu, ju]
    emb_flat = emb_dist[iu, ju]
    r_flat = brain_r[iu, ju]

    mask = np.isfinite(r_flat)
    # Optionally restrict to hop <= max_hops (matrix already uses cutoff; unreachable = cutoff)
    mask = mask & (hop_flat <= args.max_hops)
    hop_flat = hop_flat[mask]
    emb_flat = emb_flat[mask]
    r_flat = r_flat[mask]
    n_pairs = int(np.sum(mask))
    print(f"Pairs (i<j, finite r, hop<={args.max_hops}): {n_pairs}")

    # Correlation between the two regressors (are hop and emb_dist related?)
    r_hop_emb = np.corrcoef(hop_flat, emb_flat)[0, 1]
    print(f"Pearson r(hop, emb_dist) = {r_hop_emb:.4f}  (regressors correlated -> main effects conditional)")

    # Design matrix: intercept, hop, emb_dist, hop * emb_dist
    X = np.column_stack([
        np.ones(n_pairs),
        hop_flat,
        emb_flat,
        hop_flat * emb_flat,
    ])
    y = r_flat

    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels required: pip install statsmodels", file=sys.stderr)
        sys.exit(1)

    model = sm.OLS(y, X).fit()
    print("\n" + "=" * 60)
    print("OLS: brain_r ~ 1 + hop + emb_dist + hop:emb_dist")
    print("=" * 60)
    print(model.summary())

    # Extract key numbers for JSON
    coef_names = ["intercept", "hop", "emb_dist", "hop_emb_dist"]
    coefs = {name: float(model.params[i]) for i, name in enumerate(coef_names)}
    pvalues = {name: float(model.pvalues[i]) for i, name in enumerate(coef_names)}
    r_sq = float(model.rsquared)
    r_sq_adj = float(model.rsquared_adj)

    print("\nCoefficients (interpretation):")
    print("  intercept:     baseline r when hop=0, emb_dist=0")
    print("  hop:           main effect of hierarchy distance (r per hop)")
    print("  emb_dist:      main effect of embedding distance (r per unit 1-cos_sim)")
    print("  hop_emb_dist:  interaction (does effect of emb_dist depend on hop?)")
    print(f"\n  R² = {r_sq:.4f},  R²_adj = {r_sq_adj:.4f}")

    if args.output_json:
        out_path = Path(args.output_json) if not os.path.isabs(args.output_json) else Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        os.makedirs(out_path.parent, exist_ok=True)
        payload = {
            "n_pairs": n_pairs,
            "max_hops": args.max_hops,
            "corr_hop_emb_dist": float(r_hop_emb),
            "coefficients": coefs,
            "pvalues": pvalues,
            "rsquared": r_sq,
            "rsquared_adj": r_sq_adj,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print("Wrote:", out_path)

    # Compare to models with only one predictor (nested)
    print("\n" + "-" * 60)
    print("Nested models (for comparison):")
    # 1) r ~ 1 + hop
    X_hop = np.column_stack([np.ones(n_pairs), hop_flat])
    m_hop = sm.OLS(y, X_hop).fit()
    print(f"  R²(hop only)           = {m_hop.rsquared:.4f}")
    # 2) r ~ 1 + emb_dist
    X_emb = np.column_stack([np.ones(n_pairs), emb_flat])
    m_emb = sm.OLS(y, X_emb).fit()
    print(f"  R²(emb_dist only)      = {m_emb.rsquared:.4f}")
    # 3) r ~ 1 + hop + emb_dist (no interaction)
    X_both = np.column_stack([np.ones(n_pairs), hop_flat, emb_flat])
    m_both = sm.OLS(y, X_both).fit()
    print(f"  R²(hop + emb, no int.) = {m_both.rsquared:.4f}")
    print(f"  R²(full + interaction) = {r_sq:.4f}")

    # Scatter: hop vs embedding distance (one point per pair)
    scatter_path = args.scatter_out
    if scatter_path is not None:
        scatter_path = Path(scatter_path) if not os.path.isabs(scatter_path) else Path(scatter_path)
        if not scatter_path.is_absolute():
            scatter_path = repo_root / scatter_path
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(hop_flat, emb_flat, alpha=0.08, s=1, color="#1565C0", rasterized=True)
            ax.set_xlabel("Hierarchy distance (hop; 0 = synonym, parent_child steps)", fontsize=11)
            ax.set_ylabel("Embedding distance (1 - cos_sim; 0 = same, 2 = opposite)", fontsize=11)
            ax.set_title("Hop vs embedding distance (%d pairs); r = %.3f" % (n_pairs, r_hop_emb), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, args.max_hops + 0.5)
            ax.set_ylim(-0.05, 2.05)
            plt.tight_layout()
            os.makedirs(scatter_path.parent, exist_ok=True)
            plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close()
            print("Scatter saved:", scatter_path)
        except Exception as e:
            print("Scatter plot failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
