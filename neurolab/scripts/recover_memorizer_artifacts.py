#!/usr/bin/env python3
"""
Reconstruct missing memorizer artifacts (training_embeddings, training_terms, split_info)
from cached data. Run when training crashed before save completed.
"""
import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
MAP_TYPES = ["fmri_activation", "structural", "pet_receptor"]
SOURCE_TO_MAP_TYPE = {
    "direct": "fmri_activation", "neurovault": "fmri_activation", "ontology": "fmri_activation",
    "neurovault_pharma": "fmri_activation", "pharma_neurosynth": "fmri_activation",
    "neuromaps": "pet_receptor", "receptor": "pet_receptor", "neuromaps_residual": "pet_receptor",
    "receptor_residual": "pet_receptor", "structural": "structural", "enigma": "structural",
    "abagen": "pet_receptor", "reference": "pet_receptor",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="neurolab/data/embedding_model_memorizer_openai")
    parser.add_argument("--cache-dir", default="neurolab/data/merged_sources")
    parser.add_argument("--embeddings-dir", default="neurolab/data/embeddings_openai_large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--no-stratified-split", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    cache_dir = repo_root / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    emb_dir = repo_root / args.embeddings_dir if not Path(args.embeddings_dir).is_absolute() else Path(args.embeddings_dir)

    # Load cache
    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    with open(cache_dir / "term_vocab.pkl", "rb") as f:
        terms = pickle.load(f)
    term_sources = None
    if (cache_dir / "term_sources.pkl").exists():
        with open(cache_dir / "term_sources.pkl", "rb") as f:
            term_sources = pickle.load(f)

    n = len(terms)
    n_parcels = term_maps.shape[1]

    # Replicate split logic from train_text_to_brain_embedding.py
    rng = np.random.default_rng(args.seed)
    if term_sources is not None and not args.no_stratified_split:
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
                n_val_s = max(0, min(n_s - n_test_s, math.ceil(n_s * args.val_frac)))
            else:
                n_test_s = max(0, int(n_s * args.test_frac))
                n_val_s = max(0, min(n_s - n_test_s, int(n_s * args.val_frac)))
            n_train_s = n_s - n_test_s - n_val_s
            test_idx_list.extend(idx_arr[:n_test_s].tolist())
            val_idx_list.extend(idx_arr[n_test_s : n_test_s + n_val_s].tolist())
            train_idx_list.extend(idx_arr[n_test_s + n_val_s :].tolist())
        train_idx = np.array(train_idx_list, dtype=int)
        val_idx = np.array(val_idx_list, dtype=int)
        test_idx = np.array(test_idx_list, dtype=int)
    else:
        perm = rng.permutation(n)
        n_test = max(0, int(n * args.test_frac))
        n_val = max(1, int(n * args.val_frac)) if n - n_test > 0 else 0
        test_idx = perm[:n_test] if n_test else np.array([], dtype=int)
        val_idx = perm[n_test : n_test + n_val] if n_val else np.array([], dtype=int)
        train_idx = perm[n_test + n_val :]

    train_terms = [terms[i] for i in train_idx]
    val_terms = [terms[i] for i in val_idx]
    test_terms = [terms[i] for i in test_idx]
    print(f"Split: train={len(train_terms)}, val={len(val_terms)}, test={len(test_terms)}")

    # Load embeddings
    emb_loaded = np.load(emb_dir / "all_training_embeddings.npy").astype(np.float32)
    with open(emb_dir / "embedding_vocab.pkl", "rb") as f:
        emb_vocab = pickle.load(f)
    emb_vocab = list(emb_vocab)
    vocab_to_idx = {t: i for i, t in enumerate(emb_vocab)}
    missing = [t for t in terms if t not in vocab_to_idx]
    if missing:
        raise SystemExit(f"Terms not in embedding cache: {len(missing)} missing")
    emb_rows = np.array([vocab_to_idx[t] for t in terms])
    all_emb = emb_loaded[emb_rows]
    print(f"Embeddings: {all_emb.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "training_embeddings.npy", all_emb.astype(np.float64))
    with open(out_dir / "training_terms.pkl", "wb") as f:
        pickle.dump(terms, f)
    split_info = {"train_terms": train_terms, "val_terms": val_terms, "test_terms": test_terms, "seed": args.seed}
    with open(out_dir / "split_info.pkl", "wb") as f:
        pickle.dump(split_info, f)
    with open(out_dir / "training_history.pkl", "wb") as f:
        pickle.dump([], f)  # placeholder; epoch data not recoverable
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
