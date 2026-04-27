#!/usr/bin/env python3
"""
Smoke test for the KG-to-brain GNN stack.

Builds a tiny synthetic HeteroData graph (in-memory, no disk I/O), runs a
forward + backward pass, and verifies output shape & gradient flow. Intended
as a CI/dev check independent of any cache files.

Exit code: 0 on success, 2 on missing deps, 1 on functional failure.
"""
from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def main():
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as e:
        sys.stderr.write(
            f"ERROR: torch_geometric not installed ({e}). Install:\n"
            "  pip install torch_geometric>=2.5.0\n"
        )
        sys.exit(2)

    from neurolab.enrichment.kg_to_brain import build_model

    n_terms, n_genes, n_receptors, n_concepts, n_parcels = 8, 6, 4, 5, 16
    feat_dim = 12

    data = HeteroData()
    data["Term"].x = torch.randn(n_terms, feat_dim)
    data["Gene"].x = torch.randn(n_genes, feat_dim)
    data["Receptor"].x = torch.randn(n_receptors, feat_dim)
    data["OntologyConcept"].x = torch.randn(n_concepts, feat_dim)
    data["Region"].x = torch.eye(n_parcels)

    def _ei(src_n, dst_n, k=4):
        return torch.stack([
            torch.randint(0, src_n, (k,)),
            torch.randint(0, dst_n, (k,)),
        ])

    edges = [
        ("Term", "activates", "Region", n_terms, n_parcels),
        ("Gene", "expressedIn", "Region", n_genes, n_parcels),
        ("Receptor", "densityIn", "Region", n_receptors, n_parcels),
        ("Gene", "encodes", "Receptor", n_genes, n_receptors),
        ("OntologyConcept", "relatedTo", "OntologyConcept", n_concepts, n_concepts),
    ]
    for src_t, rel, dst_t, sn, dn in edges:
        ei = _ei(sn, dn, k=4)
        data[(src_t, rel, dst_t)].edge_index = ei
        data[(dst_t, f"rev_{rel}", src_t)].edge_index = torch.stack([ei[1], ei[0]])

    feature_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
    model = build_model(
        metadata=data.metadata(),
        feature_dims=feature_dims,
        n_parcels=n_parcels,
        hidden_dim=24,
        out_dim=16,
        num_layers=2,
        dropout=0.0,
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randn(n_terms, n_parcels)
    pred = model(data)
    if pred.shape != (n_terms, n_parcels):
        print(f"FAIL: forward shape {tuple(pred.shape)} != ({n_terms}, {n_parcels})")
        sys.exit(1)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    grad_total = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    if grad_total <= 0:
        print("FAIL: zero total gradient")
        sys.exit(1)
    optim.step()

    # Single-term inference path
    pred_one = model(data, term_indices=torch.tensor([0]))
    if pred_one.shape != (1, n_parcels):
        print(f"FAIL: single-term shape {tuple(pred_one.shape)} != (1, {n_parcels})")
        sys.exit(1)

    print("OK: forward/backward pass succeeded")
    print(f"  pred shape         = {tuple(pred.shape)}")
    print(f"  loss               = {float(loss.detach()):.4f}")
    print(f"  total |grad|       = {grad_total:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
