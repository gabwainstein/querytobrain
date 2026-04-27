"""
KG-to-brain GNN: predict parcellated brain maps from a heterogeneous knowledge
graph using a relational GCN over (Term, Gene, Receptor, OntologyConcept, Region)
nodes.

Design notes (kept short — see docs/implementation/KG_TO_BRAIN_GNN.md when authored):
  - 3 RGCN layers over a HeteroData graph built by scripts/build_heterogeneous_graph.py.
  - Each node type has its own input projection to a shared hidden dim.
  - Region readout = dot product between the propagated query embedding and
    learned Region node embeddings (392 parcels by default).
  - Inference path mirrors enrichment.text_to_brain.TextToBrainEmbedding so the
    rest of the pipeline (UnifiedEnrichment, query.py) can swap predictors.

Heavy deps (torch_geometric) are imported lazily so the module can be imported
in environments where the GNN is not used.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _pyg_modules():
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    return HeteroData, HeteroConv, SAGEConv


def build_model(
    metadata: tuple,
    feature_dims: dict[str, int],
    n_parcels: int,
    hidden_dim: int = 256,
    out_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.2,
):
    """Return a ready-to-train heterogeneous GNN.

    Args:
        metadata: HeteroData.metadata() tuple (node_types, edge_types).
        feature_dims: {node_type: input_feature_dim}.
        n_parcels: number of regions in the readout space (e.g. 392).
        hidden_dim, out_dim: GNN hidden / output sizes.
        num_layers: number of message-passing layers.
        dropout: dropout applied between layers.
    """
    torch, nn, F = _torch_modules()
    _, HeteroConv, SAGEConv = _pyg_modules()

    node_types, edge_types = metadata

    class KGBrainGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.ModuleDict({
                nt: nn.Linear(feature_dims[nt], hidden_dim) for nt in node_types
            })
            # Region embedding: a learnable table on top of the one-hot input
            # so the readout has a low-rank, trainable per-parcel vector.
            self.region_embedding = nn.Embedding(n_parcels, out_dim)

            self.convs = nn.ModuleList()
            for layer in range(num_layers):
                in_d = hidden_dim
                out_d = out_dim if layer == num_layers - 1 else hidden_dim
                # One SAGEConv per edge type inside HeteroConv gives per-relation
                # weights — PyG-canonical heterogeneous message passing.
                # (-1, -1) lets SAGEConv lazily infer src/dst feature dims, so we
                # don't need to track per-node-type sizes for bipartite edges.
                conv = HeteroConv(
                    {et: SAGEConv((-1, -1), out_d) for et in edge_types},
                    aggr="sum",
                )
                self.convs.append(conv)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(out_dim)

        def encode(self, x_dict, edge_index_dict):
            h = {nt: self.input_proj[nt](x_dict[nt]) for nt in x_dict}
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index_dict)
                if i != len(self.convs) - 1:
                    h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
            h = {k: self.norm(v) for k, v in h.items()}
            return h

        def predict_map_from_embedding(self, query_emb):
            """query_emb: (batch, out_dim). Returns (batch, n_parcels)."""
            region_h = self.region_embedding.weight  # (n_parcels, out_dim)
            return query_emb @ region_h.T

        def forward(self, data, term_indices=None):
            """If term_indices is given, returns predicted maps for those Term rows.

            Args:
                data: HeteroData with x_dict and edge_index_dict.
                term_indices: LongTensor of Term row indices to read out for.
            """
            h = self.encode(data.x_dict, data.edge_index_dict)
            term_h = h["Term"]
            if term_indices is None:
                return self.predict_map_from_embedding(term_h)
            sel = term_h[term_indices]
            return self.predict_map_from_embedding(sel)

    return KGBrainGNN()


# --------------------------------------------------------------------------- #
# Inference wrapper                                                            #
# --------------------------------------------------------------------------- #


class KGToBrainPredictor:
    """Inference adapter that mirrors TextToBrainEmbedding.predict_map().

    Loads:
      - `<model_dir>/model.pt`         (state_dict)
      - `<model_dir>/config.json`      (hidden_dim, out_dim, n_parcels, num_layers, dropout, feature_dims)
      - `<model_dir>/term_embeddings.npy`  (n_terms, d_term)  -- aligned with vocab
      - `<model_dir>/term_vocab.pkl`
    Plus the graph artifacts from build_heterogeneous_graph.py:
      - `<graph_dir>/hetero_data.pt`
      - `<graph_dir>/node_index.pkl`
    """

    def __init__(
        self,
        model_dir: str,
        graph_dir: str,
        device: Optional[str] = None,
    ):
        torch, nn, F = _torch_modules()
        HeteroData, HeteroConv, RGCNConv = _pyg_modules()

        self.model_dir = Path(model_dir)
        self.graph_dir = Path(graph_dir)

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._torch = torch

        with open(self.model_dir / "config.json", "r", encoding="utf-8") as fh:
            self.config = json.load(fh)
        with open(self.model_dir / "term_vocab.pkl", "rb") as fh:
            self.vocab = list(pickle.load(fh))
        self._vocab_index = {t: i for i, t in enumerate(self.vocab)}

        self.term_embeddings = np.load(self.model_dir / "term_embeddings.npy").astype(np.float32)
        self.data = torch.load(self.graph_dir / "hetero_data.pt", weights_only=False)
        self.data = self.data.to(self.device)

        feature_dims = {nt: int(self.data[nt].x.shape[1]) for nt in self.data.node_types}
        self.model = build_model(
            metadata=self.data.metadata(),
            feature_dims=feature_dims,
            n_parcels=int(self.config["n_parcels"]),
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            out_dim=int(self.config.get("out_dim", 256)),
            num_layers=int(self.config.get("num_layers", 3)),
            dropout=float(self.config.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir / "model.pt", map_location=self.device))
        self.model.eval()

    def _query_to_term_index(self, text: str) -> int:
        """Resolve a free-text query to the nearest known Term row by cosine."""
        text = (text or "").strip()
        if text in self._vocab_index:
            return self._vocab_index[text]
        toks = [t for t in text.lower().split() if t]
        if not toks:
            return 0
        # Bag-of-token average using the same hash family as the graph builder.
        d = self.term_embeddings.shape[1]
        q = np.zeros(d, dtype=np.float32)
        for tok in toks:
            h = hash(tok) % (2**31 - 1)
            q += np.random.default_rng(seed=h).standard_normal(d).astype(np.float32)
        q /= max(len(toks), 1)
        q /= np.linalg.norm(q) + 1e-8
        sims = self.term_embeddings @ q
        return int(np.argmax(sims))

    def predict_map(self, text: str) -> np.ndarray:
        torch = self._torch
        idx = self._query_to_term_index(text)
        with torch.no_grad():
            t_idx = torch.tensor([idx], dtype=torch.long, device=self.device)
            pred = self.model(self.data, term_indices=t_idx)
        return pred.detach().cpu().numpy().ravel()

    @property
    def n_parcels(self) -> int:
        return int(self.config["n_parcels"])


__all__ = ["KGToBrainPredictor", "build_model"]
