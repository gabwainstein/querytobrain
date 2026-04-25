#!/usr/bin/env python3
"""
Diagnose why test generalization is stuck around ~0.55.

Answers:
1. **Best-map ceiling**: For each test term, what's the best correlation achievable
   by *any* training-term map? If this is only ~0.6, test maps are not well
   approximated by any single train map (data ceiling). If it's ~0.9, the limit
   is in our model.
2. **Nearest-neighbor (text) baseline**: For each test term, use the *nearest*
   training term (by TF-IDF cosine similarity) and that train term's map as the
   prediction. Mean test correlation = how well "retrieve by text" does. If ~0.55,
   our learned model is doing about as well as retrieval.
3. **Per-term distribution**: Min/max/percentiles of per-term test correlations
   (for best-map and NN baselines). Tells you if a few hard terms drag the mean down.
4. **Seed stability** (optional): Run training with several seeds; mean ± std test
   correlation. If always ~0.55 ± 0.01, the ceiling is stable.

Usage (from repo root):
  python neurolab/scripts/diagnose_generalization_ceiling.py --cache-dir neurolab/data/decoder_cache
  python neurolab/scripts/diagnose_generalization_ceiling.py --cache-dir neurolab/data/decoder_cache --split-info neurolab/data/embedding_model/split_info.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

N_PARCELS = 392  # Pipeline uses Glasser+Tian


def _same_split_as_training(terms, term_maps, seed, test_frac, val_frac, term_sources=None):
    """Replicate train/val/test split from train_text_to_brain_embedding.py.
    When term_sources present, uses stratified split (sources <50 terms get >=20% in test)."""
    n = len(terms)
    rng = np.random.default_rng(seed)
    if term_sources is not None and len(term_sources) == n:
        from collections import defaultdict
        import math
        idx_by_source = defaultdict(list)
        for i in range(n):
            idx_by_source[term_sources[i]].append(i)
        test_idx_list, val_idx_list, train_idx_list = [], [], []
        for src, indices in sorted(idx_by_source.items()):
            n_s = len(indices)
            perm_s = rng.permutation(n_s)
            idx_arr = np.array(indices)[perm_s]
            if n_s < 50:
                n_test_s = max(1, math.ceil(n_s * 0.2))
                n_val_s = max(0, min(n_s - n_test_s, math.ceil(n_s * val_frac)))
            else:
                n_test_s = max(0, int(n_s * test_frac))
                n_val_s = max(0, min(n_s - n_test_s, int(n_s * val_frac)))
            test_idx_list.extend(idx_arr[:n_test_s].tolist())
            val_idx_list.extend(idx_arr[n_test_s : n_test_s + n_val_s].tolist())
            train_idx_list.extend(idx_arr[n_test_s + n_val_s :].tolist())
        return (
            np.array(train_idx_list, dtype=int),
            np.array(val_idx_list, dtype=int),
            np.array(test_idx_list, dtype=int),
        )
    perm = rng.permutation(n)
    n_test = max(0, int(n * test_frac))
    n_val = max(1, int(n * val_frac)) if n - n_test > 0 else 0
    test_idx = perm[:n_test] if n_test else np.array([], dtype=int)
    val_idx = perm[n_test : n_test + n_val] if n_val else np.array([], dtype=int)
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Diagnose why test generalization ~0.55")
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache", help="Decoder cache (term_maps.npz, term_vocab.pkl)")
    parser.add_argument("--split-info", default="", help="Optional: path to split_info.pkl from a training run (use same split as that run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-terms", type=int, default=0)
    parser.add_argument("--seed-sweep", action="store_true", help="Recompute diagnostics for seeds 42,43,44 to see if ceiling depends on split")
    parser.add_argument("--no-stratified-split", action="store_true", help="Use random split instead of stratified (when term_sources present)")
    args = parser.parse_args()

    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(repo_root, args.cache_dir)
    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        print("Cache not found. Build first: python neurolab/scripts/build_term_maps_cache.py --cache-dir ...", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)
    term_maps = data["term_maps"]
    with open(pkl_path, "rb") as f:
        terms = pickle.load(f)
    assert len(terms) == term_maps.shape[0]
    term_sources = None
    src_pkl = os.path.join(cache_dir, "term_sources.pkl")
    if os.path.exists(src_pkl) and not getattr(args, "no_stratified_split", False):
        with open(src_pkl, "rb") as f:
            term_sources = pickle.load(f)
        if len(term_sources) != len(terms):
            term_sources = None
        elif term_sources:
            print("Using stratified split (term_sources present)")
    if args.max_terms and len(terms) > args.max_terms:
        idx = np.random.default_rng(42).choice(len(terms), args.max_terms, replace=False)
        term_maps = term_maps[idx]
        terms = [terms[i] for i in idx]
        if term_sources is not None:
            term_sources = [term_sources[i] for i in idx]
    n = len(terms)

    if args.split_info:
        split_path = args.split_info if os.path.isabs(args.split_info) else os.path.join(repo_root, args.split_info)
        with open(split_path, "rb") as f:
            info = pickle.load(f)
        train_terms = info["train_terms"]
        test_terms = info["test_terms"]
        term_to_idx = {t: i for i, t in enumerate(terms)}
        train_idx = np.array([term_to_idx[t] for t in train_terms if t in term_to_idx])
        test_idx = np.array([term_to_idx[t] for t in test_terms if t in term_to_idx])
        if len(train_idx) != len(train_terms) or len(test_idx) != len(test_terms):
            print("Warning: split_info terms not all in cache; using indices that matched.", file=sys.stderr)
    else:
        train_idx, _, test_idx = _same_split_as_training(
            terms, term_maps, args.seed, args.test_frac, args.val_frac, term_sources
        )

    train_maps = term_maps[train_idx]
    test_maps = term_maps[test_idx]
    n_train, n_test = train_maps.shape[0], test_maps.shape[0]
    print(f"Data: {n} terms; train={n_train}, test={n_test}")
    if n_test == 0:
        print("No test set; increase --test-frac or use a cache with more terms.", file=sys.stderr)
        sys.exit(1)

    # ----- 1. Best-map ceiling: for each test term, max correlation with any train map -----
    # (n_test, n_train) correlations: test_maps @ train_maps.T (after center); faster: compute per test term
    best_corrs = []
    for i in range(n_test):
        y = test_maps[i]
        y = y - y.mean()
        yn = np.sqrt(np.sum(y ** 2))
        if yn < 1e-10:
            best_corrs.append(0.0)
            continue
        corrs = (train_maps - train_maps.mean(axis=1, keepdims=True)) @ y
        train_norms = np.sqrt(np.sum((train_maps - train_maps.mean(axis=1, keepdims=True)) ** 2, axis=1))
        train_norms = np.maximum(train_norms, 1e-10)
        corrs = corrs / (train_norms * yn)
        best_corrs.append(float(np.max(corrs)))
    best_corrs = np.array(best_corrs)
    mean_best = float(np.mean(best_corrs))
    print("\n--- 1. Best-map ceiling (per test term: max corr with any train map) ---")
    print(f"  Mean: {mean_best:.4f}")
    print(f"  Min / 25% / 50% / 75% / Max: {np.min(best_corrs):.4f} / {np.percentile(best_corrs, 25):.4f} / {np.percentile(best_corrs, 50):.4f} / {np.percentile(best_corrs, 75):.4f} / {np.max(best_corrs):.4f}")
    if mean_best < 0.65:
        print("  => Test maps are not well approximated by any single train map; ceiling is largely in the *data* (term->map diversity or noise).")
    else:
        print("  => There is headroom: some train map often matches test well; model or retrieval could in principle do better.")

    # ----- 2. Nearest-neighbor (text) baseline: TF-IDF, predict = nearest train map -----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    train_terms_list = [terms[j] for j in train_idx]
    test_terms_list = [terms[j] for j in test_idx]
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
    X_train = vectorizer.fit_transform(train_terms_list)
    X_test = vectorizer.transform(test_terms_list)
    sim = cosine_similarity(X_test, X_train)  # (n_test, n_train)
    nn_idx = np.argmax(sim, axis=1)
    nn_pred = train_maps[nn_idx]
    nn_corrs = []
    for i in range(n_test):
        r = np.corrcoef(nn_pred[i], test_maps[i])[0, 1]
        nn_corrs.append(r if np.isfinite(r) else 0.0)
    nn_corrs = np.array(nn_corrs)
    mean_nn = float(np.mean(nn_corrs))
    print("\n--- 2. Nearest-neighbor (text) baseline (predict = map of nearest train term by TF-IDF) ---")
    print(f"  Mean test correlation: {mean_nn:.4f}")
    print(f"  Min / 25% / 50% / 75% / Max: {np.min(nn_corrs):.4f} / {np.percentile(nn_corrs, 25):.4f} / {np.percentile(nn_corrs, 50):.4f} / {np.percentile(nn_corrs, 75):.4f} / {np.max(nn_corrs):.4f}")
    if abs(mean_nn - 0.55) < 0.03:
        print("  => NN baseline ~0.55: our learned model is doing about as well as 'retrieve by text similarity'.")
    elif mean_nn < 0.45:
        print("  => NN baseline is below our model (~0.55): the regression head adds value over raw text retrieval; gap to best-map is text-to-map alignment.")

    # ----- 3. Summary interpretation -----
    print("\n--- 3. Interpretation ---")
    if mean_best < 0.65 and abs(mean_nn - 0.55) < 0.05:
        print("  The ~0.55 test ceiling is consistent with:")
        print("  - Test-term maps are not well matched by any single training map (best-map ceiling is low).")
        print("  - Retrieval by text similarity (NN) already achieves ~0.55; learning does not add much.")
        print("  To improve: more/better data (more terms, or maps that are more consistent across similar terms).")
    elif mean_best > 0.75:
        print("  There is headroom: best-map ceiling is high. The limit is in *text-to-map alignment*:")
        print("  - The right map exists in the training set, but the model (and NN by text) cannot identify it well.")
        print("  To improve: better text encoder, or retrieval over train maps (e.g. embed query, find nearest train map) instead of regression.")
    else:
        print("  Best-map ceiling and NN baseline suggest a mix of data limit and model limit; try more data or different encoders.")

    # ----- 4. Optional: seed sweep (does ceiling depend on which terms are test?) -----
    if args.seed_sweep:
        print("\n--- 4. Seed sweep (best-map & NN baseline for seeds 42, 43, 44) ---")
        best_means, nn_means = [], []
        for s in [42, 43, 44]:
            ti, _, te = _same_split_as_training(
                terms, term_maps, s, args.test_frac, args.val_frac, term_sources
            )
            tm, test_m = term_maps[ti], term_maps[te]
            n_test = test_m.shape[0]
            best_c = []
            for i in range(n_test):
                y = test_m[i] - test_m[i].mean()
                yn = max(np.sqrt(np.sum(y ** 2)), 1e-10)
                c = (tm - tm.mean(axis=1, keepdims=True)) @ y / (np.maximum(np.sqrt(np.sum((tm - tm.mean(axis=1, keepdims=True)) ** 2, axis=1)), 1e-10) * yn)
                best_c.append(float(np.max(c)))
            best_means.append(np.mean(best_c))
            tr_terms = [terms[j] for j in ti]
            te_terms = [terms[j] for j in te]
            vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
            Xt = vec.fit_transform(tr_terms)
            Xe = vec.transform(te_terms)
            sim = cosine_similarity(Xe, Xt)
            nn_i = np.argmax(sim, axis=1)
            nn_c = [np.corrcoef(tm[nn_i[i]], test_m[i])[0, 1] for i in range(n_test)]
            nn_c = [x if np.isfinite(x) else 0.0 for x in nn_c]
            nn_means.append(np.mean(nn_c))
        print(f"  Best-map ceiling: {np.mean(best_means):.4f} ± {np.std(best_means):.4f}")
        print(f"  NN baseline:      {np.mean(nn_means):.4f} ± {np.std(nn_means):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
