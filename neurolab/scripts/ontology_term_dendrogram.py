#!/usr/bin/env python3
"""
Build a dendrogram of decoder cache terms that appear in the ontology,
using hierarchy distance (number of parent_child steps) as the distance.
Ties to activation: heatmap of brain-map correlation reordered by dendrogram,
and mean within-cluster brain r at selected cut heights.

Usage (from repo root):
  python neurolab/scripts/ontology_term_dendrogram.py
  python neurolab/scripts/ontology_term_dendrogram.py --out neurolab/data/term_dendrogram.png --max-leaves 80
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Reuse graph building and normalization from graph_distance_correlation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_distance_correlation import (  # noqa: E402
    _normalize,
    build_unified_graph,
)


def hierarchy_distance_on_path(G, path):
    """Count parent_child edges along a path. path = list of nodes."""
    import networkx as nx
    n_parent_child = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rel = G.edges.get((u, v), G.edges.get((v, u), {})).get("relation", "other")
        if rel == "parent_child":
            n_parent_child += 1
    return n_parent_child


def build_hierarchy_distance_matrix(G, terms: list[str], cutoff: int = 20) -> np.ndarray:
    """
    terms: list of node names (normalized) that are in G.
    Returns symmetric matrix D[i,j] = number of parent_child steps on shortest path (synonym=0).
    Unreachable or beyond cutoff -> D[i,j] = cutoff.
    """
    import networkx as nx
    n = len(terms)
    term_to_idx = {t: i for i, t in enumerate(terms)}
    D = np.full((n, n), cutoff, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for i, src in enumerate(terms):
        if (i + 1) % 100 == 0 and n > 150:
            print(f"  Computing paths from term {i+1}/{n}...")
        try:
            paths = nx.single_source_shortest_path(G, src, cutoff=cutoff)
        except (nx.NetworkXError, KeyError):
            continue
        for tgt, path in paths.items():
            if tgt not in term_to_idx:
                continue
            j = term_to_idx[tgt]
            if i >= j:
                continue
            d = hierarchy_distance_on_path(G, path)
            D[i, j] = d
            D[j, i] = d
    return D


def main():
    parser = argparse.ArgumentParser(
        description="Dendrogram of cache terms by ontology hierarchy distance (divergence steps)"
    )
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache",
                        help="Cache with term_maps.npz + term_vocab.pkl")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies",
                        help="Directory containing OBO/OWL ontology files")
    parser.add_argument("--out", default="neurolab/data/term_dendrogram.png",
                        help="Output path for dendrogram figure")
    parser.add_argument("--max-leaves", type=int, default=80,
                        help="Max leaf labels to show (truncate to avoid clutter; default 80)")
    parser.add_argument("--cutoff", type=int, default=20,
                        help="Max hierarchy steps for distance (default 20)")
    parser.add_argument("--linkage", default="average",
                        choices=("average", "single", "complete", "ward"),
                        help="Linkage method (default average)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir) if os.path.isabs(args.cache_dir) else repo_root / args.cache_dir
    ontology_dir = Path(args.ontology_dir) if os.path.isabs(args.ontology_dir) else repo_root / args.ontology_dir
    out_path = Path(args.out) if os.path.isabs(args.out) else repo_root / args.out

    # Load cache (vocab + brain maps for activation tie-in)
    npz_path = cache_dir / "term_maps.npz"
    pkl_path = cache_dir / "term_vocab.pkl"
    if not npz_path.exists() or not pkl_path.exists():
        print("Cache not found. Build decoder cache first.", file=sys.stderr)
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
    print("Building unified ontology graph...")
    G = build_unified_graph(str(ontology_dir))
    if G.number_of_nodes() < 10:
        print("Ontology graph too small.", file=sys.stderr)
        sys.exit(1)
    graph_nodes = set(G.nodes())
    cache_in_graph = {k: idx for k, idx in norm_to_idx.items() if k in graph_nodes}
    terms = list(cache_in_graph.keys())
    n_terms = len(terms)
    print(f"Cache terms in graph: {n_terms} / {len(term_vocab)}")

    if n_terms < 3:
        print("Need at least 3 terms in graph for dendrogram.", file=sys.stderr)
        sys.exit(1)

    # Brain maps for the 827 terms (for activation heatmap and within-cluster r)
    term_indices = [cache_in_graph[t] for t in terms]
    term_maps = term_maps_full[term_indices]  # (n_terms, n_parcels)
    print("Computing brain-map correlation matrix (activation)...")
    brain_r = np.corrcoef(term_maps)  # (n_terms, n_terms)
    np.fill_diagonal(brain_r, np.nan)  # avoid diagonal in means

    # Distance matrix (hierarchy steps)
    print(f"Computing pairwise hierarchy distance (cutoff={args.cutoff})...")
    D = build_hierarchy_distance_matrix(G, terms, cutoff=args.cutoff)

    # Condensed form for linkage (upper triangle, row-major)
    from scipy.spatial.distance import squareform
    condensed = squareform(D, checks=False)

    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
    Z = linkage(condensed, method=args.linkage)

    # Cophenetic (dendrogram) distance: height at which each pair merges (same order as condensed)
    coph_corr, coph_condensed = cophenet(Z, condensed)
    coph_matrix = squareform(coph_condensed)  # (n_terms, n_terms)
    print(f"  Cophenetic correlation (dendrogram vs input distances): {coph_corr:.4f}")

    # Cluster counts at different distance thresholds (hierarchy steps)
    print("\nClusters at different cut heights (hierarchy-distance threshold):")
    print("  (cut dendrogram at this height to get N clusters; height = divergence steps)")
    for thresh in [1, 2, 3, 4, 5, 6, 8, 10]:
        labels = fcluster(Z, t=thresh, criterion="distance")
        n_clusters = len(set(labels))
        print(f"    height <= {thresh}:  {n_clusters} clusters")
    # Exactly k clusters: cut just above the merge that leaves k clusters
    print("  (requested number of clusters):")
    heights = np.sort(Z[:, 2])
    for k in [5, 10, 20, 50]:
        if k >= n_terms or k < 1:
            continue
        # After (n_terms - k) merges we have k clusters; merge index = n_terms - k - 1
        merge_idx = n_terms - k - 1
        if merge_idx < 0:
            continue
        t_cut = heights[merge_idx] + 1e-6  # just above that merge
        if merge_idx + 1 < len(heights):
            t_cut = (heights[merge_idx] + heights[merge_idx + 1]) / 2.0
        labels = fcluster(Z, t=t_cut, criterion="distance")
        n_clusters = len(set(labels))
        print(f"    k={k}:  {n_clusters} clusters (cut at height ~{heights[merge_idx]:.2f})")

    # Within-cluster mean brain r (activation coherence) at k=10 and k=20
    print("\nMean within-cluster brain-map r (activation coherence) at k=10 and k=20:")
    for k in [10, 20]:
        if k >= n_terms or k < 1:
            continue
        merge_idx = n_terms - k - 1
        if merge_idx < 0:
            continue
        t_cut = (heights[merge_idx] + heights[merge_idx + 1]) / 2.0 if merge_idx + 1 < len(heights) else heights[merge_idx] + 0.01
        labels = fcluster(Z, t=t_cut, criterion="distance")
        unique_labels = sorted(set(labels))
        for cid in unique_labels[:15]:  # first 15 clusters
            mask = labels == cid
            n_c = int(np.sum(mask))
            if n_c < 2:
                continue
            sub_r = brain_r[np.ix_(mask, mask)]
            mean_r = np.nanmean(sub_r)
            print(f"  k={k} cluster {cid}: n={n_c}, mean within-cluster r={mean_r:.3f}")
        if len(unique_labels) > 15:
            print(f"  ... and {len(unique_labels) - 15} more clusters")
        # Overall mean within-cluster r (average across clusters)
        within_means = []
        for cid in unique_labels:
            mask = labels == cid
            if np.sum(mask) < 2:
                continue
            sub_r = brain_r[np.ix_(mask, mask)]
            within_means.append(np.nanmean(sub_r))
        print(f"  k={k} overall mean within-cluster r = {np.mean(within_means):.3f}")

    # Term-level pattern: mean brain r at "close" vs "far" dendrogram distance (property of terms)
    print("\nTerm pattern clusters (mean brain r with close vs far ontology neighbours):")
    close_thresh, far_thresh = 2.0, 5.0  # dendrogram distance
    mean_r_close = np.full(n_terms, np.nan)
    mean_r_far = np.full(n_terms, np.nan)
    for i in range(n_terms):
        close_j = (coph_matrix[i, :] <= close_thresh) & (np.arange(n_terms) != i)
        far_j = coph_matrix[i, :] > far_thresh
        if np.any(close_j):
            mean_r_close[i] = np.nanmean(brain_r[i, close_j])
        if np.any(far_j):
            mean_r_far[i] = np.nanmean(brain_r[i, far_j])
    # Replace nan with global mean for clustering
    mean_r_close_safe = np.where(np.isfinite(mean_r_close), mean_r_close, np.nanmean(mean_r_close))
    mean_r_far_safe = np.where(np.isfinite(mean_r_far), mean_r_far, np.nanmean(mean_r_far))
    features = np.column_stack([mean_r_close_safe, mean_r_far_safe])
    from scipy.cluster.vq import kmeans2
    k_pattern = 3
    rng = np.random.default_rng(42)
    centroids, pattern_labels = kmeans2(features, k_pattern, minit="points", seed=rng)
    far_rank = np.argsort(centroids[:, 1])[::-1]  # cluster with highest mean_r_far first
    print(f"  Close = dendrogram dist <= {close_thresh}; Far = dendrogram dist > {far_thresh}")
    for idx in range(k_pattern):
        cid = far_rank[idx]
        mask = (pattern_labels == cid)
        n_c = int(np.sum(mask))
        mc = centroids[cid, 0]
        mf = centroids[cid, 1]
        label = "stay_high" if mf > 0.12 else ("drop" if mc > 0.2 else "low_overall")
        print(f"  Pattern cluster '{label}' (n={n_c}): mean_r_close={mc:.3f}, mean_r_far={mf:.3f}")
        if n_c <= 25:
            names = [term_vocab[cache_in_graph[terms[i]]][:40] for i in np.where(mask)[0]]
            for nm in names:
                print(f"    - {nm}")
        else:
            top_far = np.where(mask)[0][np.argsort(mean_r_far_safe[mask])[-5:][::-1]]
            print(f"    Top 5 by mean_r_far: {[term_vocab[cache_in_graph[terms[i]]][:35] for i in top_far]}")

    # Dendrogram plot + activation heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping dendrogram plot.", file=sys.stderr)
        return

    # Plot: Pearson r vs dendrogram distance (not raw hierarchy steps)
    brain_upper = brain_r[np.triu_indices(n_terms, k=1)]  # same order as condensed / coph_condensed
    ok = np.isfinite(brain_upper)
    if np.sum(ok) >= 50:
        plot_dir = os.path.dirname(out_path)
        coph_vs_r_path = os.path.join(plot_dir, "pearson_vs_dendrogram_distance.png")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(coph_condensed[ok], brain_upper[ok], alpha=0.2, s=8, c="#1565C0", edgecolors="none")
        ax2.axhline(np.nanmean(brain_r), color="#F44336", linestyle="--", linewidth=1.5, label="Overall mean r")
        ax2.set_xlabel("Dendrogram distance (height at which the two terms merge)", fontsize=11)
        ax2.set_ylabel("Brain-map Pearson r", fontsize=11)
        ax2.set_title("Brain-map Pearson r vs dendrogram distance (ontology clustering)", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(coph_vs_r_path, dpi=150, bbox_inches="tight")
        print(f"  Pearson vs dendrogram distance saved: {coph_vs_r_path}")
        plt.close()

    # Leaf order from dendrogram (for heatmap reordering)
    dnd = dendrogram(Z, no_plot=True)
    leaf_order = dnd["leaves"]  # indices 0..n_terms-1 in dendrogram left-to-right order
    brain_r_ordered = brain_r[np.ix_(leaf_order, leaf_order)].copy()
    np.fill_diagonal(brain_r_ordered, 0)  # for display

    # Leaf labels: display name from term_vocab (first 30 chars)
    leaf_labels = [term_vocab[cache_in_graph[t]][:30] for t in terms] if n_terms <= args.max_leaves else None
    if n_terms > args.max_leaves:
        print(f"  (showing {n_terms} leaves without labels; use --max-leaves {n_terms} to show names)")

    fig = plt.figure(figsize=(14, 12))
    ax_dend = fig.add_axes([0.05, 0.55, 0.9, 0.4])  # top: dendrogram
    ax_heat = fig.add_axes([0.05, 0.05, 0.75, 0.45])  # bottom: heatmap
    cax = fig.add_axes([0.82, 0.05, 0.02, 0.45])  # colorbar

    dendrogram(
        Z,
        labels=leaf_labels,
        ax=ax_dend,
        leaf_rotation=90,
        leaf_font_size=6,
        show_leaf_counts=(n_terms <= args.max_leaves),
        color_threshold=0.7 * max(Z[:, 2]) if len(Z) else 0,
    )
    ax_dend.set_xlabel("")
    ax_dend.set_ylabel("Hierarchy distance (steps diverged)", fontsize=11)
    ax_dend.set_title("Ontology dendrogram (top) and brain-map correlation reordered by it (bottom)", fontsize=12)

    im = ax_heat.imshow(brain_r_ordered, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.8)
    ax_heat.set_xlabel("Term index (dendrogram order)", fontsize=10)
    ax_heat.set_ylabel("Term index (dendrogram order)", fontsize=10)
    plt.colorbar(im, cax=cax, label="Brain-map Pearson r")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nDendrogram + activation heatmap saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
