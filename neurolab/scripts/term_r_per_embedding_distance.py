#!/usr/bin/env python3
"""
Repeat the term r-per-hop analysis using **embedding-space distance** instead of
ontology hop: for each term, mean brain-map Pearson r with all other terms at each
distance bin in embedding space. One line per term; fit 1/(d+c).

Requires either:
  --embedding-dir  with training_embeddings.npy + training_terms.pkl (from train_text_to_brain_embedding),
  or  --encoder sentence-transformers  (and optional --encoder-model) to encode terms on the fly.

Usage (from repo root):
  python neurolab/scripts/term_r_per_embedding_distance.py --embedding-dir neurolab/data/embedding_model
  python neurolab/scripts/term_r_per_embedding_distance.py --encoder sentence-transformers --encoder-model NeuML/pubmedbert-base-embeddings
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_distance_correlation import _normalize, build_unified_graph  # noqa: E402


def load_embeddings_from_dir(embedding_dir: Path, terms: list[str]) -> np.ndarray | None:
    """Load training_embeddings.npy and training_terms.pkl; return (n_terms, dim) for our terms, or None."""
    emb_path = embedding_dir / "training_embeddings.npy"
    terms_path = embedding_dir / "training_terms.pkl"
    if not emb_path.exists() or not terms_path.exists():
        return None
    emb_all = np.load(emb_path).astype(np.float64)
    with open(terms_path, "rb") as f:
        training_terms = list(pickle.load(f))
    training_norm = {_normalize(t): i for i, t in enumerate(training_terms)}
    n = len(terms)
    dim = emb_all.shape[1]
    out = np.full((n, dim), np.nan)
    for i, t in enumerate(terms):
        idx = training_norm.get(_normalize(t))
        if idx is not None:
            out[i] = emb_all[idx]
    # Require all terms to have embeddings (otherwise use --encoder)
    if np.any(~np.isfinite(out)):
        return None
    return out


def encode_with_sentence_transformers(terms: list[str], model_name: str) -> np.ndarray | None:
    """Encode terms with sentence-transformers; return (n, dim) or None."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(terms, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float64)
    except Exception as e:
        print("sentence-transformers encode failed:", e, file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Mean brain-map r per term at each embedding-distance bin (like term_r_per_hop but embedding space)"
    )
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache",
                        help="Cache with term_maps.npz + term_vocab.pkl")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies",
                        help="Ontology dir (to get same 827 terms as hop analysis)")
    parser.add_argument("--embedding-dir", default=None,
                        help="Dir with training_embeddings.npy + training_terms.pkl (from train_text_to_brain_embedding)")
    parser.add_argument("--encoder", choices=("sentence-transformers",), default=None,
                        help="Encode terms on the fly (e.g. sentence-transformers)")
    parser.add_argument("--encoder-model", default="NeuML/pubmedbert-base-embeddings",
                        help="Model name when --encoder sentence-transformers")
    parser.add_argument("--out", default="neurolab/data/term_r_per_embedding_distance.png",
                        help="Output path for plot")
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Number of embedding-distance bins (default 20)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir) if not os.path.isabs(args.cache_dir) else Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    ontology_dir = Path(args.ontology_dir) if not os.path.isabs(args.ontology_dir) else repo_root / args.ontology_dir
    out_path = Path(args.out) if not os.path.isabs(args.out) else repo_root / args.out
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else (repo_root / "neurolab/data/embedding_model")
    if args.embedding_dir and not os.path.isabs(args.embedding_dir):
        embedding_dir = repo_root / args.embedding_dir
    # Load cache and get terms in ontology (same 827 as hop analysis)
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

    print("Building ontology graph (to get same term set as hop analysis)...")
    G = build_unified_graph(str(ontology_dir))
    graph_nodes = set(G.nodes())
    cache_in_graph = {k: idx for k, idx in norm_to_idx.items() if k in graph_nodes}
    terms = list(cache_in_graph.keys())
    n_terms = len(terms)
    print(f"Terms in graph: {n_terms}")

    # Term embeddings (n_terms, dim)
    embeddings = None
    if (args.embedding_dir or (embedding_dir / "training_embeddings.npy").exists()) and not args.encoder:
        print("Loading embeddings from", embedding_dir, "...")
        embeddings = load_embeddings_from_dir(embedding_dir, terms)
        if embeddings is None:
            print("Could not load from embedding dir (missing terms or files).", file=sys.stderr)
    if embeddings is None and args.encoder == "sentence-transformers":
        print("Encoding terms with", args.encoder_model, "...")
        embeddings = encode_with_sentence_transformers(terms, args.encoder_model)
    if embeddings is None:
        print("No embeddings. Use --embedding-dir with training_embeddings.npy + training_terms.pkl, or --encoder sentence-transformers.", file=sys.stderr)
        sys.exit(1)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Pairwise distance in embedding space: 1 - cos_sim (0 = same, 2 = opposite)
    cos_sim = embeddings @ embeddings.T
    np.fill_diagonal(cos_sim, np.nan)
    emb_dist = 1.0 - cos_sim
    np.fill_diagonal(emb_dist, 0.0)

    # Brain-map correlation matrix
    term_indices = [cache_in_graph[t] for t in terms]
    term_maps = term_maps_full[term_indices]
    brain_r = np.corrcoef(term_maps)
    np.fill_diagonal(brain_r, np.nan)

    # Bin embedding distances (use 0 to 2 for 1-cos_sim)
    n_bins = args.n_bins
    bin_edges = np.linspace(0, 2.0, n_bins + 1)
    bin_edges[-1] += 1e-9
    mean_r_per_term_per_bin = np.full((n_terms, n_bins), np.nan)
    n_per_term_per_bin = np.zeros((n_terms, n_bins), dtype=int)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for i in range(n_terms):
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            j_mask = (emb_dist[i, :] >= lo) & (emb_dist[i, :] < hi) & (np.arange(n_terms) != i)
            if not np.any(j_mask):
                continue
            vals = brain_r[i, j_mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                mean_r_per_term_per_bin[i, b] = np.mean(vals)
            n_per_term_per_bin[i, b] = int(np.sum(j_mask))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = bin_centers
    for i in range(n_terms):
        y = mean_r_per_term_per_bin[i, :]
        valid = np.isfinite(y)
        if np.any(valid):
            ax.plot(x[valid], y[valid], color="#1565C0", alpha=0.25, linewidth=0.8)
    mean_over_terms = np.nanmean(mean_r_per_term_per_bin, axis=0)
    ax.plot(x, mean_over_terms, color="#0D47A1", linewidth=3, label="Mean over terms")
    median_over_terms = np.nanmedian(mean_r_per_term_per_bin, axis=0)
    ax.plot(x, median_over_terms, color="#1976D2", linewidth=2, linestyle="--", label="Median over terms")

    # Fit 1/(d+c) on mean curve (skip first bin if d≈0)
    from scipy.optimize import curve_fit
    def inv_model(d, a, c):
        return a / (d + c)
    popt_inv = None
    x_curve_fit = None
    valid_mean = np.isfinite(mean_over_terms)
    if np.sum(valid_mean) >= 5:
        x_fit = x[valid_mean]
        y_fit = mean_over_terms[valid_mean]
        try:
            popt_inv, _ = curve_fit(
                inv_model, x_fit, y_fit,
                p0=[0.5, 0.5], bounds=([0.01, 0.01], [2.0, 10.0]),
            )
            pred_mean = inv_model(x_fit, *popt_inv)
            ss_res = np.sum((y_fit - pred_mean) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2_mean = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            x_curve_fit = np.linspace(0.05, 2, 200)
            ax.plot(x_curve_fit, inv_model(x_curve_fit, *popt_inv), color="#2E7D32", linewidth=2,
                    label=r"Exploratory fit $r=a/(d+c)$: R²=%.3f" % r2_mean)
            print("\nFit r = a/(d+c) on mean curve: a=%.4f, c=%.4f, R²=%.4f" % (*popt_inv, r2_mean))

            # R² and F-test on actual data (all term-bin pairs)
            d_flat = []
            r_flat = []
            for b in range(n_bins):
                for i in range(n_terms):
                    r_val = mean_r_per_term_per_bin[i, b]
                    if np.isfinite(r_val):
                        d_flat.append(bin_centers[b])
                        r_flat.append(r_val)
            d_flat = np.array(d_flat)
            r_flat = np.array(r_flat)
            if len(r_flat) >= 10:
                pred_flat = inv_model(d_flat, *popt_inv)
                ss_res_flat = np.sum((r_flat - pred_flat) ** 2)
                ss_tot_flat = np.sum((r_flat - np.mean(r_flat)) ** 2)
                r2_actual = 1.0 - (ss_res_flat / ss_tot_flat) if ss_tot_flat > 0 else 0.0
                rmse = np.sqrt(np.mean((r_flat - pred_flat) ** 2))
                n_flat = len(r_flat)
                print("  R² on actual data (all term-bin pairs, n=%d): %.4f" % (n_flat, r2_actual))
                print("  RMSE on actual data: %.4f" % rmse)
                df1, df2 = 2, n_flat - 3
                ss_exp = ss_tot_flat - ss_res_flat
                F = (ss_exp / df1) / (ss_res_flat / df2) if ss_res_flat > 0 else 0.0
                from scipy import stats as sp_stats
                p_value = 1.0 - sp_stats.f.cdf(F, df1, df2)
                print("  F-test (curve vs constant): F(2, %d)=%.1f, p=%.2e" % (df2, F, p_value))
        except Exception as e:
            print("  Fit failed:", e)

    ax.axhline(np.nanmean(brain_r), color="#F44336", linestyle=":", linewidth=1.5, label="Overall mean r (all pairs)")
    ax.set_xlabel("Embedding distance (1 - cos_sim; 0 = same, 2 = opposite)", fontsize=11)
    ax.set_ylabel("Mean brain-map Pearson r (with all terms in this distance bin)", fontsize=11)
    ax.set_title("One line per term: mean r vs embedding-distance bin (%d bins)" % n_bins, fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 2.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Plot saved:", out_path)
    plt.close()

    # Scatter: one point per pair (embedding distance vs brain r), mean trend overlaid
    iu, ju = np.triu_indices(n_terms, k=1)
    x_scatter = emb_dist[iu, ju]
    y_scatter = brain_r[iu, ju]
    mask = np.isfinite(y_scatter) & np.isfinite(x_scatter)
    x_scatter = x_scatter[mask]
    y_scatter = y_scatter[mask]
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.scatter(x_scatter, y_scatter, alpha=0.12, s=2, color="#1565C0", rasterized=True, label="Term pairs (i<j)")
    ax2.plot(bin_centers, mean_over_terms, color="#0D47A1", linewidth=3, label="Mean trend (binned)")
    if popt_inv is not None and x_curve_fit is not None:
        ax2.plot(x_curve_fit, inv_model(x_curve_fit, *popt_inv), color="#2E7D32", linewidth=2,
                 label=r"Exploratory fit $r=a/(d+c)$")
    ax2.axhline(np.nanmean(brain_r), color="#F44336", linestyle=":", linewidth=1.5, label="Overall mean r")
    ax2.set_xlabel("Embedding distance (1 - cos_sim; 0 = same, 2 = opposite)", fontsize=11)
    ax2.set_ylabel("Brain-map Pearson r (pair)", fontsize=11)
    ax2.set_title("Scatter: embedding distance vs brain r (%d pairs), mean trend overlaid" % len(x_scatter), fontsize=12)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 2.05)
    plt.tight_layout()
    out_scatter = out_path.parent / (out_path.stem + "_scatter" + out_path.suffix)
    plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
    print("Scatter plot saved:", out_scatter)
    plt.close()

    print("\nPer-bin summary (embedding distance):")
    print("  bin_center   n_terms_with_data   mean_r   median_r")
    for b in range(n_bins):
        col = mean_r_per_term_per_bin[:, b]
        valid = np.isfinite(col)
        n_with = int(np.sum(valid))
        if n_with > 0:
            print("  %8.3f   %5d                 %.4f   %.4f" % (bin_centers[b], n_with, np.nanmean(col), np.nanmedian(col)))


if __name__ == "__main__":
    main()
