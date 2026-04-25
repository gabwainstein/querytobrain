#!/usr/bin/env python3
"""
Diagnostic checklist for collections with poor train recovery.
Tests: (A) embedding collapse, (B) target degeneracy, (C) sign mismatch.
"""
import pickle
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
cache_dir = repo / "neurolab" / "data" / "merged_sources"
model_dir = repo / "neurolab" / "data" / "embedding_model_openai"
nv_cache = repo / "neurolab" / "data" / "neurovault_cache"

BAD_COLLECTIONS = [426, 437, 507, 555, 2508]


def cosine_sim(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def main():
    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    sources = pickle.load(open(cache_dir / "term_sources.pkl", "rb"))
    term_maps = np.load(cache_dir / "term_maps.npz")["term_maps"]
    term_to_idx = {t: i for i, t in enumerate(terms)}

    # NeuroVault term -> collection
    nv_terms = pickle.load(open(nv_cache / "term_vocab.pkl", "rb"))
    nv_cids = pickle.load(open(nv_cache / "term_collection_ids.pkl", "rb"))
    nv_term_to_cid = {}
    for t, cid in zip(nv_terms, nv_cids):
        key = ("fMRI: " + t.strip()).strip()
        if key not in nv_term_to_cid:
            nv_term_to_cid[key] = cid

    # Load embeddings (OpenAI model; 1:1 with term_vocab order)
    emb_arr = np.load(cache_dir / "openai_embeddings_text-embedding-3-large.npz")
    all_emb = emb_arr["embeddings"]
    term_to_emb_idx = {t: i for i, t in enumerate(terms) if i < all_emb.shape[0]}

    # Raw maps from neurovault_cache (pre-merge, to check degeneracy before z-score)
    nv_maps = np.load(nv_cache / "term_maps.npz")["term_maps"]
    merged_term_to_nv_idx = {}
    for i, t in enumerate(nv_terms):
        key = ("fMRI: " + t.strip()).strip()
        if key not in merged_term_to_nv_idx:
            merged_term_to_nv_idx[key] = i

    print("=" * 70)
    print("DIAGNOSTIC CHECKLIST: Collections 426, 437, 507, 555, 2508")
    print("=" * 70)

    for cid in BAD_COLLECTIONS:
        labels = []
        idxs = []
        nv_idxs = []
        for t, s in zip(terms, sources):
            if s != "neurovault":
                continue
            if nv_term_to_cid.get(t) == cid:
                labels.append(t)
                idxs.append(term_to_idx[t])
                ni = merged_term_to_nv_idx.get(t)
                if ni is not None:
                    nv_idxs.append(ni)

        print(f"\n--- Collection {cid} (n={len(labels)}) ---")
        print("Labels:")
        for lbl in labels:
            print(f"  {repr(lbl)}")

        # A) Embedding collapse
        emb_idxs = [term_to_emb_idx.get(t) for t in labels]
        emb_idxs = [i for i in emb_idxs if i is not None]
        if len(emb_idxs) >= 2:
            embs = all_emb[emb_idxs].astype(np.float64)
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sims.append(cosine_sim(embs[i], embs[j]))
            med_sim = np.median(sims)
            print(f"\n  [A] Embedding collapse: median pairwise cosine sim = {med_sim:.4f}")
            if med_sim > 0.95:
                print(f"      FLAG: embeddings nearly identical (collapse)")
            else:
                print(f"      OK: embeddings distinguishable")
        else:
            print(f"\n  [A] Embedding collapse: n<2, skip")

        # B) Target degeneracy (raw neurovault maps + within-collection similarity)
        if nv_idxs:
            raw = nv_maps[nv_idxs].astype(np.float64)
            stds = np.std(raw, axis=1)
            pct_zero = 100 * np.mean(np.abs(raw) < 1e-8)
            # Within-collection map similarity (are maps redundant?)
            map_sims = []
            for i in range(len(raw)):
                for j in range(i + 1, len(raw)):
                    map_sims.append(cosine_sim(raw[i], raw[j]))
            med_map_sim = np.median(map_sims) if map_sims else 0
            print(f"\n  [B] Target degeneracy (raw NV cache):")
            print(f"      std per map: mean={np.mean(stds):.4f} min={np.min(stds):.4f}")
            print(f"      % near-zero: {pct_zero:.1f}%")
            print(f"      median pairwise map corr: {med_map_sim:.4f}")
            if np.mean(stds) < 0.02:
                print(f"      FLAG: maps near-constant")
            elif med_map_sim > 0.98:
                print(f"      FLAG: maps nearly identical (redundant)")
            else:
                print(f"      OK")
        else:
            # Use merged_sources maps (z-scored)
            maps = term_maps[idxs].astype(np.float64)
            map_sims = []
            for i in range(len(maps)):
                for j in range(i + 1, len(maps)):
                    map_sims.append(cosine_sim(maps[i], maps[j]))
            med_map_sim = np.median(map_sims) if map_sims else 0
            print(f"\n  [B] Target degeneracy (merged, z-scored):")
            print(f"      median pairwise map corr: {med_map_sim:.4f}")
            if med_map_sim > 0.98:
                print(f"      FLAG: maps nearly identical")
            else:
                print(f"      OK")

        # C) Map type note for 2508 (ROI)
        if cid == 2508:
            print(f"\n  [C] Map type: ROI/mask (ROI_ACC, ROI_DLPFC, ROI_Striatum)")
            print(f"      FLAG: wrong supervision type for regression; use overlap/BCE or exclude")

    # ENIGMA sign-flip test (load model, compute pred, compare corr(pred,target) vs corr(pred,-target))
    print("\n" + "=" * 70)
    print("ENIGMA SIGN TEST")
    print("=" * 70)
    enigma_idx = [i for i, s in enumerate(sources) if s == "enigma"]
    if enigma_idx:
        try:
            import torch
            cfg = pickle.load(open(model_dir / "config.pkl", "rb"))
            MAP_TYPES = ["fmri_activation", "structural", "pet_receptor"]
            term_map_types = pickle.load(open(cache_dir / "term_map_types.pkl", "rb"))
            type_oh = np.eye(3, dtype=np.float32)[[MAP_TYPES.index(term_map_types[i]) if term_map_types[i] in MAP_TYPES else 0 for i in range(len(terms))]]
            X = np.hstack([all_emb.astype(np.float32), type_oh])
            h2 = cfg.get("head_hidden2", 0)
            if h2 > 0:
                model = torch.nn.Sequential(
                    torch.nn.Linear(X.shape[1], cfg["head_hidden"]), torch.nn.ReLU(),
                    torch.nn.Linear(cfg["head_hidden"], h2), torch.nn.ReLU(),
                    torch.nn.Linear(h2, term_maps.shape[1])
                )
            else:
                model = torch.nn.Sequential(
                    torch.nn.Linear(X.shape[1], cfg["head_hidden"]), torch.nn.ReLU(),
                    torch.nn.Linear(cfg["head_hidden"], term_maps.shape[1])
                )
            model.load_state_dict(torch.load(model_dir / "head_weights.pt", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X)).numpy()
            targets = term_maps[enigma_idx].astype(np.float64)
            pred_ep = pred[enigma_idx].astype(np.float64)
            r_pos = np.mean([np.corrcoef(pred_ep[i], targets[i])[0, 1] for i in range(len(enigma_idx)) if np.isfinite(targets[i]).all()])
            r_neg = np.mean([np.corrcoef(pred_ep[i], -targets[i])[0, 1] for i in range(len(enigma_idx)) if np.isfinite(targets[i]).all()])
            print(f"  corr(pred, target):   {r_pos:.4f}")
            print(f"  corr(pred, -target):  {r_neg:.4f}")
            if r_neg > r_pos + 0.1:
                print(f"  FLAG: sign flip improves corr -> add direction to labels or use abs(target)")
            else:
                print(f"  (sign flip does not clearly help)")
        except Exception as e:
            print(f"  (sign test skipped: {e})")


if __name__ == "__main__":
    main()
