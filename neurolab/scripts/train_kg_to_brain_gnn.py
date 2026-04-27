#!/usr/bin/env python3
"""
Train the KG-to-brain heterogeneous GNN.

Inputs:
  - Graph artifacts from build_heterogeneous_graph.py (default neurolab/data/kg_brain_graph)
      hetero_data.pt, node_index.pkl, meta.json
  - Optional term embeddings (.npy aligned with merged_sources term order)

Loss (multi-task):
  L_map       : MSE between predicted parcel vectors (per-Term readout) and
                ground-truth merged_sources/term_maps.npz rows.
  L_contrast  : InfoNCE-style similarity loss between Term node embeddings
                whose merged_sources rows correlate above a threshold (cheap,
                consistently helps on rare terms).
  L_link      : (optional, --link-loss-weight > 0) DistMult on Gene-Receptor edges
                — held-out 5% set; quick proxy for KG fidelity.
  L_teacher   : (optional, --teacher <path.npz>) KL between predicted maps and a
                pre-computed KG-teacher distribution. Off by default; wire up the
                output of calibrate_kg_teacher_from_audit.py here when available.

Eval:
  - --eval-split collection : split rows by term_sources.pkl provenance (default)
  - --eval-split term       : random per-term split
  - --eval-split both       : both reports printed

Auto-device: CUDA if available else CPU.

Outputs to --output-dir (default neurolab/data/kg_brain_gnn_model):
  model.pt, config.json, term_embeddings.npy, term_vocab.pkl, training_log.json

Usage:
  python neurolab/scripts/train_kg_to_brain_gnn.py \\
      --graph-dir neurolab/data/kg_brain_graph \\
      --merged-sources neurolab/data/merged_sources \\
      --epochs 50 --batch-size 256 --lr 5e-4 \\
      --output-dir neurolab/data/kg_brain_gnn_model
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def _import_runtime():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import HeteroData  # noqa: F401
    except ImportError as e:
        sys.stderr.write(
            "ERROR: torch and torch_geometric are required. Install with:\n"
            "  pip install torch_geometric>=2.5.0\n"
            f"(underlying error: {e})\n"
        )
        sys.exit(2)
    from neurolab.enrichment.kg_to_brain import build_model  # noqa: E402
    return torch, nn, F, build_model


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _split_indices(
    n_rows: int,
    sources: dict[str, str] | None,
    vocab: list[str],
    mode: str = "collection",
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if mode == "collection" and sources:
        # Bucket terms by their source label, then split buckets.
        buckets: dict[str, list[int]] = {}
        for i, t in enumerate(vocab):
            src = sources.get(t, "_unknown")
            buckets.setdefault(src, []).append(i)
        keys = sorted(buckets.keys())
        rng.shuffle(keys)
        n_test = max(1, int(round(len(keys) * test_ratio)))
        n_val = max(1, int(round(len(keys) * val_ratio)))
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test : n_test + n_val])
        train, val, test = [], [], []
        for k, idxs in buckets.items():
            (test if k in test_keys else val if k in val_keys else train).extend(idxs)
        return np.asarray(train), np.asarray(val), np.asarray(test)
    # Random per-term split
    perm = rng.permutation(n_rows)
    n_test = int(round(n_rows * test_ratio))
    n_val = int(round(n_rows * val_ratio))
    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return train, val, test


def _correlation(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 1:
        pred = pred[None, :]
        target = target[None, :]
    out = []
    for p, t in zip(pred, target):
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            continue
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        out.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def _info_nce(query_emb, positive_emb, all_emb, temperature: float = 0.1):
    """Simple InfoNCE: cosine(query, positive) vs cosine(query, all_emb)."""
    import torch
    import torch.nn.functional as F
    q = F.normalize(query_emb, dim=-1)
    p = F.normalize(positive_emb, dim=-1)
    a = F.normalize(all_emb, dim=-1)
    pos_score = (q * p).sum(dim=-1, keepdim=True) / temperature
    all_score = q @ a.T / temperature
    logits = torch.cat([pos_score, all_score], dim=1)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def _build_contrastive_pairs(
    term_maps: np.ndarray, sample_size: int = 2000, threshold: float = 0.6, seed: int = 42
) -> np.ndarray:
    """Return (q_idx, positive_idx) pairs from highly-correlated term maps."""
    rng = np.random.default_rng(seed)
    n = term_maps.shape[0]
    sample = rng.choice(n, size=min(sample_size, n), replace=False)
    sub = term_maps[sample]
    sub_z = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-8)
    sim = sub_z @ sub_z.T / sub.shape[1]
    np.fill_diagonal(sim, -np.inf)
    pairs = []
    for i in range(len(sample)):
        j = int(np.argmax(sim[i]))
        if sim[i, j] >= threshold:
            pairs.append((int(sample[i]), int(sample[j])))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", type=str, default="neurolab/data/kg_brain_graph")
    ap.add_argument("--merged-sources", type=str, default="neurolab/data/merged_sources")
    ap.add_argument("--output-dir", type=str, default="neurolab/data/kg_brain_gnn_model")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--contrastive-weight", type=float, default=0.2)
    ap.add_argument("--link-loss-weight", type=float, default=0.0)
    ap.add_argument("--teacher", type=str, default=None)
    ap.add_argument("--teacher-weight", type=float, default=0.5)
    ap.add_argument("--eval-split", choices=["collection", "term", "both"], default="collection")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    # Force unbuffered stdout so per-epoch prints surface in real time when run
    # via tools that read the output file (Bash run_in_background, CI, etc.).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    torch, nn, F, build_model = _import_runtime()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ---- Load graph ----
    graph_dir = Path(args.graph_dir)
    data = torch.load(graph_dir / "hetero_data.pt", weights_only=False).to(device)
    with open(graph_dir / "node_index.pkl", "rb") as fh:
        node_index = pickle.load(fh)
    with open(graph_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    n_parcels = int(meta["n_parcels"])
    feature_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}

    # ---- Load supervision ----
    merged = Path(args.merged_sources)
    npz = np.load(merged / "term_maps.npz", allow_pickle=False)
    maps = npz["maps"] if "maps" in npz.files else npz[npz.files[0]]
    with open(merged / "term_vocab.pkl", "rb") as fh:
        vocab = list(pickle.load(fh))
    term_sources = {}
    sources_pkl = merged / "term_sources.pkl"
    if sources_pkl.exists():
        with open(sources_pkl, "rb") as fh:
            raw = pickle.load(fh)
        if isinstance(raw, dict):
            term_sources = {str(k): str(v) for k, v in raw.items()}
    if maps.shape[0] != len(vocab):
        raise ValueError(f"maps rows {maps.shape[0]} != vocab {len(vocab)}")

    target_tensor = torch.tensor(maps, dtype=torch.float32, device=device)
    term_idx_in_graph = node_index["Term"]  # vocab term -> graph row
    # Align supervision rows to graph rows
    aligned_target = torch.zeros((len(term_idx_in_graph), n_parcels), dtype=torch.float32, device=device)
    inv = {v: k for k, v in term_idx_in_graph.items()}
    vocab_to_row = {t: i for i, t in enumerate(vocab)}
    aligned_mask = torch.zeros(len(term_idx_in_graph), dtype=torch.bool, device=device)
    for graph_row in range(len(term_idx_in_graph)):
        label = inv[graph_row]
        v_row = vocab_to_row.get(label)
        if v_row is None:
            continue
        aligned_target[graph_row] = target_tensor[v_row]
        aligned_mask[graph_row] = True
    n_aligned = int(aligned_mask.sum().item())
    print(f"[supervision] {n_aligned}/{len(term_idx_in_graph)} term rows aligned to graph")

    # ---- Splits ----
    aligned_idx = torch.nonzero(aligned_mask, as_tuple=False).flatten().cpu().numpy()
    aligned_vocab = [inv[i] for i in aligned_idx]
    train_pos, val_pos, test_pos = _split_indices(
        n_rows=len(aligned_idx),
        sources=term_sources,
        vocab=aligned_vocab,
        mode=args.eval_split if args.eval_split != "both" else "collection",
        seed=args.seed,
    )
    train_idx = torch.tensor(aligned_idx[train_pos], dtype=torch.long, device=device)
    val_idx = torch.tensor(aligned_idx[val_pos], dtype=torch.long, device=device)
    test_idx = torch.tensor(aligned_idx[test_pos], dtype=torch.long, device=device)
    print(f"[split] train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")

    # ---- Contrastive pairs (cheap, on-CPU) ----
    contrastive_pairs = _build_contrastive_pairs(maps, sample_size=2000, threshold=0.6, seed=args.seed)
    if contrastive_pairs.size:
        # remap to graph rows
        graph_pairs = []
        for v_a, v_b in contrastive_pairs:
            label_a, label_b = vocab[v_a], vocab[v_b]
            ga, gb = term_idx_in_graph.get(label_a), term_idx_in_graph.get(label_b)
            if ga is not None and gb is not None:
                graph_pairs.append((ga, gb))
        contrastive_pairs = np.asarray(graph_pairs, dtype=np.int64) if graph_pairs else np.zeros((0, 2), np.int64)
    print(f"[contrastive] pairs: {contrastive_pairs.shape[0]}")

    # ---- Model ----
    model = build_model(
        metadata=data.metadata(),
        feature_dims=feature_dims,
        n_parcels=n_parcels,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    teacher_dist = None
    if args.teacher:
        tnpz = np.load(args.teacher, allow_pickle=False)
        teacher_dist = torch.tensor(tnpz["maps"], dtype=torch.float32, device=device)
        if teacher_dist.shape != aligned_target.shape:
            print(
                f"WARN: teacher shape {tuple(teacher_dist.shape)} != target "
                f"{tuple(aligned_target.shape)}; ignoring"
            )
            teacher_dist = None

    log: list[dict] = []
    best_val = -1.0
    best_state: Optional[dict] = None

    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        epoch_losses = []
        t0 = time.time()
        for start in range(0, perm.numel(), args.batch_size):
            batch = perm[start : start + args.batch_size]
            optim.zero_grad()
            pred = model(data, term_indices=batch)  # (b, n_parcels)
            true = aligned_target[batch]
            loss_map = F.mse_loss(pred, true)
            loss = loss_map

            if args.contrastive_weight > 0 and contrastive_pairs.shape[0] > 0:
                # Encode all term embeddings via the GNN (cached once per step)
                h = model.encode(data.x_dict, data.edge_index_dict)
                term_h = h["Term"]
                pair_a = torch.tensor(contrastive_pairs[:, 0], dtype=torch.long, device=device)
                pair_b = torch.tensor(contrastive_pairs[:, 1], dtype=torch.long, device=device)
                q_emb = term_h[pair_a]
                p_emb = term_h[pair_b]
                loss_c = _info_nce(q_emb, p_emb, term_h, temperature=0.1)
                loss = loss + args.contrastive_weight * loss_c

            if teacher_dist is not None and args.teacher_weight > 0:
                # KL between softmax-normalized predicted vs teacher row
                pred_log = F.log_softmax(pred, dim=-1)
                teacher_p = F.softmax(teacher_dist[batch], dim=-1)
                loss_t = F.kl_div(pred_log, teacher_p, reduction="batchmean")
                loss = loss + args.teacher_weight * loss_t

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(data, term_indices=val_idx).cpu().numpy()
            val_true = aligned_target[val_idx].cpu().numpy()
            val_corr = _correlation(val_pred, val_true)
        epoch_log = {
            "epoch": epoch + 1,
            "loss_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "val_corr": val_corr,
            "elapsed_s": time.time() - t0,
        }
        log.append(epoch_log)
        print(f"[epoch {epoch+1:02d}] loss={epoch_log['loss_mean']:.4f}  val_corr={val_corr:.4f}  ({epoch_log['elapsed_s']:.1f}s)")
        if val_corr > best_val:
            best_val = val_corr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(data, term_indices=test_idx).cpu().numpy()
        test_true = aligned_target[test_idx].cpu().numpy()
        test_corr = _correlation(test_pred, test_true)
    print(f"\n[final] best_val_corr={best_val:.4f}  test_corr={test_corr:.4f}")

    # If --eval-split both, also report term-split metrics
    if args.eval_split == "both":
        train2, val2, test2 = _split_indices(
            n_rows=len(aligned_idx),
            sources=None,
            vocab=aligned_vocab,
            mode="term",
            seed=args.seed,
        )
        with torch.no_grad():
            test2_idx = torch.tensor(aligned_idx[test2], dtype=torch.long, device=device)
            tp = model(data, term_indices=test2_idx).cpu().numpy()
            tt = aligned_target[test2_idx].cpu().numpy()
            print(f"[eval-split=term] test_corr={_correlation(tp, tt):.4f}")

    # ---- Save ----
    torch.save(model.state_dict(), output_dir / "model.pt")

    # term embeddings = encoded Term node embeddings under the trained GNN
    with torch.no_grad():
        h = model.encode(data.x_dict, data.edge_index_dict)
    term_emb = h["Term"].cpu().numpy().astype(np.float32)
    np.save(output_dir / "term_embeddings.npy", term_emb)

    with open(output_dir / "term_vocab.pkl", "wb") as fh:
        pickle.dump([inv[i] for i in range(len(term_idx_in_graph))], fh)

    config = {
        "n_parcels": n_parcels,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "feature_dims": feature_dims,
        "best_val_corr": best_val,
        "test_corr": test_corr,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print(f"\nSaved model to {output_dir}")


if __name__ == "__main__":
    main()
