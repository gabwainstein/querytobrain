#!/usr/bin/env python3
"""
For each term (in the ontology), compute mean Pearson r with all other terms
at each hierarchy distance (hop) 0..20. Plot one line per term: x = hop, y = mean r.

Usage (from repo root):
  python neurolab/scripts/term_r_per_hop.py
  python neurolab/scripts/term_r_per_hop.py --out neurolab/data/term_r_per_hop.png --max-hops 20
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
from ontology_term_dendrogram import (  # noqa: E402
    build_hierarchy_distance_matrix,
)


def main():
    parser = argparse.ArgumentParser(
        description="Mean brain-map Pearson r per term at each ontology hop (0..max-hops)"
    )
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache",
                        help="Cache with term_maps.npz + term_vocab.pkl")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies",
                        help="Directory containing OBO/OWL ontology files")
    parser.add_argument("--out", default="neurolab/data/term_r_per_hop.png",
                        help="Output path for plot")
    parser.add_argument("--max-hops", type=int, default=20,
                        help="Max hierarchy distance (hop) to include (default 20)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir) if not os.path.isabs(args.cache_dir) else Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    ontology_dir = Path(args.ontology_dir) if not os.path.isabs(args.ontology_dir) else repo_root / args.ontology_dir
    out_path = Path(args.out) if not os.path.isabs(args.out) else repo_root / args.out

    # Load cache
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

    # Build graph and get terms in graph
    print("Building ontology graph...")
    G = build_unified_graph(str(ontology_dir))
    graph_nodes = set(G.nodes())
    cache_in_graph = {k: idx for k, idx in norm_to_idx.items() if k in graph_nodes}
    terms = list(cache_in_graph.keys())
    n_terms = len(terms)
    print(f"Terms in graph: {n_terms}")

    term_indices = [cache_in_graph[t] for t in terms]
    term_maps = term_maps_full[term_indices]
    brain_r = np.corrcoef(term_maps)
    np.fill_diagonal(brain_r, np.nan)

    # Hierarchy distance matrix (parent_child steps; synonym = 0)
    print(f"Computing hierarchy distance matrix (cutoff={args.max_hops})...")
    D = build_hierarchy_distance_matrix(G, terms, cutoff=args.max_hops)

    # For each term i and each hop d in 0..max_hops: mean r of term i with all j where D[i,j]==d, j!=i
    max_h = args.max_hops
    mean_r_per_term_per_hop = np.full((n_terms, max_h + 1), np.nan)
    n_per_term_per_hop = np.zeros((n_terms, max_h + 1), dtype=int)

    for i in range(n_terms):
        for d in range(max_h + 1):
            j_mask = (D[i, :] == d) & (np.arange(n_terms) != i)
            if not np.any(j_mask):
                continue
            vals = brain_r[i, j_mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                mean_r_per_term_per_hop[i, d] = np.mean(vals)
            n_per_term_per_hop[i, d] = int(np.sum(j_mask))

    # Plot: one line per term, x = 0..max_hops, y = mean r
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(max_h + 1, dtype=float)
    for i in range(n_terms):
        y = mean_r_per_term_per_hop[i, :]
        valid = np.isfinite(y)
        if np.any(valid):
            ax.plot(x[valid], y[valid], color="#1565C0", alpha=0.25, linewidth=0.8)
    # Mean across terms (at each hop)
    mean_over_terms = np.nanmean(mean_r_per_term_per_hop, axis=0)
    ax.plot(x, mean_over_terms, color="#0D47A1", linewidth=3, label="Mean over terms")
    # Median
    median_over_terms = np.nanmedian(mean_r_per_term_per_hop, axis=0)
    ax.plot(x, median_over_terms, color="#1976D2", linewidth=2, linestyle="--", label="Median over terms")

    # Fit 1/(d+c) and exponential; report R² and plot fits
    from scipy.optimize import curve_fit
    # Fit on d>=1 (skip d=0 synonym regime)
    x_fit = np.arange(1, max_h + 1, dtype=float)
    y_fit = mean_over_terms[1 : max_h + 1]
    valid = np.isfinite(y_fit)
    if np.sum(valid) >= 5:
        def inv_model(d, a, c):
            return a / (d + c)

        def exp_model(d, a, k):
            return a * np.exp(-d / k)

        try:
            popt_inv, _ = curve_fit(
                inv_model, x_fit[valid], y_fit[valid],
                p0=[0.5, 1.0], bounds=([0.01, 0.01], [2.0, 50.0]),
            )
            pred_inv = inv_model(x_fit, *popt_inv)
            ss_res_inv = np.nansum((y_fit - pred_inv) ** 2)
            ss_tot = np.nansum((y_fit - np.nanmean(y_fit)) ** 2)
            r2_inv = 1.0 - (ss_res_inv / ss_tot) if ss_tot > 0 else 0.0
            x_curve = np.linspace(0.5, max_h, 200)
            ax.plot(x_curve, inv_model(x_curve, *popt_inv), color="#2E7D32", linewidth=2,
                    label=r"Exploratory fit $r=a/(d+c)$: R²=%.3f" % r2_inv)
            print("\nFit r = a/(d+c) (d>=1): a=%.4f, c=%.4f" % (*popt_inv,))
            print("  R² on mean curve (one value per hop): %.4f" % r2_inv)
            # R² on actual data: all (term, hop) pairs with d>=1 and finite r
            d_flat = []
            r_flat = []
            for d in range(1, max_h + 1):
                for i in range(n_terms):
                    r_val = mean_r_per_term_per_hop[i, d]
                    if np.isfinite(r_val):
                        d_flat.append(d)
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
                print("  R² on actual data (all term-hop pairs, d>=1, n=%d): %.4f" % (n_flat, r2_actual))
                print("  RMSE on actual data: %.4f" % rmse)
                # F-test: model (2 params) vs null (constant mean). F = (explained/df1) / (residual/df2)
                df1, df2 = 2, n_flat - 3
                ss_exp = ss_tot_flat - ss_res_flat
                F = (ss_exp / df1) / (ss_res_flat / df2) if ss_res_flat > 0 else 0.0
                from scipy import stats as sp_stats
                p_value = 1.0 - sp_stats.f.cdf(F, df1, df2)
                print("  F-test (curve vs constant): F(2, %d)=%.1f, p=%.2e (significant: p<0.05)" % (df2, F, p_value))
        except Exception as e:
            print("\n1/(d+c) fit failed:", e)
        try:
            popt_exp, _ = curve_fit(
                exp_model, x_fit[valid], y_fit[valid],
                p0=[0.3, 5.0], bounds=([0.01, 0.5], [2.0, 50.0]),
            )
            pred_exp = exp_model(x_fit, *popt_exp)
            ss_res_exp = np.nansum((y_fit - pred_exp) ** 2)
            r2_exp = 1.0 - (ss_res_exp / ss_tot) if ss_tot > 0 else 0.0
            x_curve = np.linspace(0.5, max_h, 200)
            ax.plot(x_curve, exp_model(x_curve, *popt_exp), color="#F9A825", linewidth=1.5, linestyle="--",
                    label=r"Exploratory fit $r=a\,e^{-d/k}$: R²=%.3f" % r2_exp)
            print("Fit r = a*exp(-d/k) (d>=1): a=%.4f, k=%.4f, R²=%.4f" % (*popt_exp, r2_exp))
        except Exception as e:
            print("Exponential fit failed:", e)

    ax.axhline(np.nanmean(brain_r), color="#F44336", linestyle=":", linewidth=1.5, label="Overall mean r (all pairs)")
    ax.set_xlabel("Hierarchy distance (hop; 0 = same concept / synonym)", fontsize=11)
    ax.set_ylabel("Mean brain-map Pearson r (with all terms at this hop)", fontsize=11)
    ax.set_title("One line per term: mean r with neighbours at each hop (0..%d)" % max_h, fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max_h + 0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()

    # Print summary: at each hop, how many terms have at least one neighbour, and mean/median of mean r
    print("\nPer-hop summary (mean over terms of 'mean r at this hop'):")
    print("  hop   n_terms_with_data   mean_r   median_r   mean_n_neighbours")
    for d in range(max_h + 1):
        col = mean_r_per_term_per_hop[:, d]
        valid = np.isfinite(col)
        n_with = int(np.sum(valid))
        if n_with == 0:
            print(f"  {d:3d}   {n_with:5d}                 -         -         -")
        else:
            print(f"  {d:3d}   {n_with:5d}                 {np.nanmean(col):.4f}   {np.nanmedian(col):.4f}   {np.mean(n_per_term_per_hop[:, d]):.1f}")


if __name__ == "__main__":
    main()
