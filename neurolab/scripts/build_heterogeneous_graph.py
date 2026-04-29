#!/usr/bin/env python3
"""
Build a heterogeneous knowledge graph for the KG-to-brain GNN.

Node types:
  - Term            (NeuroQuery / NeuroSynth / merged_sources vocabulary)
  - Region          (n_parcels parcels of the combined atlas; default 392)
  - Gene            (data/gene_info.json)
  - Receptor        (receptor_knowledge_base.json)
  - Compound        (nootropics KG / ChEMBL bridges, optional)
  - OntologyConcept (UBERON / MF / MONDO / HPO / ChEBI / Cognitive Atlas)

Edge types (each with a reverse for message passing):
  - (Term,            activates,      Region)    [supervision; from merged_sources/term_maps.npz]
  - (Gene,            expressedIn,    Region)    [from abagen_cache]
  - (Receptor,        densityIn,      Region)    [from neuromaps_cache]
  - (Term,            mentions,       Compound)  [from KG bridges, optional]
  - (Compound,        binds,          Receptor)  [from KG bridges, optional]
  - (Gene,            encodes,        Receptor)  [from receptor KB]
  - (Term,            relatedTo,      OntologyConcept)
  - (OntologyConcept, relatedTo,      OntologyConcept) [meta-graph internal edges]

Outputs to --output-dir (default neurolab/data/kg_brain_graph):
  - hetero_data.pt   (torch_geometric HeteroData; saved with torch.save)
  - node_index.pkl   ({node_type: {label: row_index}})
  - meta.json        (counts, atlas info, build provenance)

Usage:
  python neurolab/scripts/build_heterogeneous_graph.py \\
      --merged-sources neurolab/data/merged_sources \\
      --abagen-cache neurolab/data/abagen_cache \\
      --neuromaps-cache neurolab/data/neuromaps_cache \\
      --receptor-kb neurolab/data/receptor_knowledge_base.json \\
      --gene-info neurolab/data/gene_info.json \\
      --ontology-dir neurolab/data/ontologies \\
      --output-dir neurolab/data/kg_brain_graph
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def _try_import_pyg():
    try:
        import torch
        from torch_geometric.data import HeteroData
        return torch, HeteroData
    except ImportError as e:
        sys.stderr.write(
            "ERROR: torch_geometric is required. Install with:\n"
            "  pip install torch_geometric>=2.5.0\n"
            f"(underlying error: {e})\n"
        )
        sys.exit(2)


def _load_term_maps(merged_sources: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Load term vocabulary, term->parcel map matrix, and term->source provenance."""
    term_maps_npz = merged_sources / "term_maps.npz"
    term_vocab_pkl = merged_sources / "term_vocab.pkl"
    term_sources_pkl = merged_sources / "term_sources.pkl"
    if not term_maps_npz.exists() or not term_vocab_pkl.exists():
        raise FileNotFoundError(
            f"Missing merged_sources artifacts under {merged_sources}: "
            f"need term_maps.npz and term_vocab.pkl"
        )
    npz = np.load(term_maps_npz, allow_pickle=False)
    maps = npz["maps"] if "maps" in npz.files else npz[npz.files[0]]
    with open(term_vocab_pkl, "rb") as fh:
        vocab = pickle.load(fh)
    sources: dict[str, str] = {}
    if term_sources_pkl.exists():
        with open(term_sources_pkl, "rb") as fh:
            raw = pickle.load(fh)
        if isinstance(raw, dict):
            sources = {str(k): str(v) for k, v in raw.items()}
    if maps.shape[0] != len(vocab):
        raise ValueError(
            f"term_maps rows ({maps.shape[0]}) != vocab size ({len(vocab)})"
        )
    return list(vocab), np.asarray(maps, dtype=np.float32), sources


def _load_gene_info(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_receptor_kb(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Gene symbols matching the HGNC ALL-CAPS convention: 3+ chars, uppercase, may
# contain digits / dashes. The token boundary kills accidental matches inside
# longer mixed-case identifiers.
_GENE_TOKEN_RE = __import__("re").compile(r"\b[A-Z][A-Z0-9]{2,}\b")

# Common fMRI / methodology abbreviations that look like gene symbols but
# are not (or whose gene namesakes are vanishingly rare in cognitive contexts).
# Exclude these from Term -> Gene mention extraction.
_GENE_MENTION_STOPWORDS = frozenset({
    # Imaging / methods
    "MRI", "FMRI", "PET", "MEG", "EEG", "BOLD", "ROI", "FOV", "MNI",
    "DTI", "DWI", "QSM", "ASL", "FA", "MD", "AD", "RD", "GFA",
    "ICA", "PCA", "GLM", "ANOVA", "ANCOVA", "RFX", "MFX", "MOCO",
    "TFCE", "SPM", "FSL", "AFNI", "MELODIC", "FEAT", "FIX", "FRACE",
    "BIDS", "NIFTI", "DICOM", "CIFTI", "GIFTI",
    # Generic neuroscience abbreviations
    "PSD", "LFP", "ERP", "MUA", "SUA", "VOI",
    # Common-word collisions
    "SET", "CIT", "TEC", "MAG", "CAP", "NET", "ART", "POS", "PRO",
    "USP", "NTS", "KEY", "TOP", "END", "MAP", "TAR", "CON",
})


_GENE_SYMBOL_RE = __import__("re").compile(r"\(([A-Z0-9][A-Z0-9\-\.]*)\)\s*$")


def _parse_gene_symbol(label: str) -> str | None:
    """Extract gene symbol from abagen vocab strings like
    'A1BG gene expression from Allen Human Brain Atlas (A1BG)'."""
    m = _GENE_SYMBOL_RE.search(str(label or ""))
    return m.group(1) if m else None


def _load_abagen_region_by_gene(abagen_cache: Path) -> tuple[list[str], np.ndarray] | None:
    """Load gene-by-region expression matrix from the abagen cache.

    Returns (gene_symbols, matrix [n_genes, n_parcels]) or None. Handles the
    actual layout (term_maps.npz with key 'term_maps' + term_vocab.pkl listing
    'GENE gene expression from Allen Human Brain Atlas (GENE)' strings).
    """
    if not abagen_cache.exists():
        return None
    npz_path = abagen_cache / "term_maps.npz"
    vocab_path = abagen_cache / "term_vocab.pkl"
    if not npz_path.exists() or not vocab_path.exists():
        return None
    try:
        npz = np.load(npz_path, allow_pickle=False)
        if "term_maps" in npz.files:
            matrix = np.asarray(npz["term_maps"], dtype=np.float32)
        elif "matrix" in npz.files:
            matrix = np.asarray(npz["matrix"], dtype=np.float32)
        else:
            return None
        with open(vocab_path, "rb") as fh:
            vocab = list(pickle.load(fh))
        if matrix.shape[0] != len(vocab):
            return None
        symbols = [_parse_gene_symbol(v) for v in vocab]
        keep = [(s, i) for i, s in enumerate(symbols) if s]
        if not keep:
            return None
        kept_symbols = [s for s, _ in keep]
        kept_idx = [i for _, i in keep]
        return kept_symbols, matrix[kept_idx]
    except Exception as exc:
        sys.stderr.write(f"WARN: abagen loader failed: {exc}\n")
        return None


_RECEPTOR_AFTER_BINDING_RE = __import__("re").compile(r"binding to\s+([^\s\(]+)", __import__("re").IGNORECASE)


def _parse_receptor_token(label: str) -> str | None:
    """Extract receptor/transporter symbol from neuromaps labels like
    'PET: FEOBV binding to VAChT (vesicular acetylcholine transporter)'."""
    m = _RECEPTOR_AFTER_BINDING_RE.search(str(label or ""))
    return m.group(1) if m else None


def _load_neuromaps_receptor_density(neuromaps_cache: Path) -> tuple[list[str], np.ndarray] | None:
    """Load receptor-by-region density from neuromaps annotation cache.

    Returns (receptor_symbols, matrix [n_receptors, n_parcels]) or None. Handles
    the actual layout (annotation_maps.npz with key 'matrix' + annotation_labels.pkl
    listing 'PET: <tracer> binding to <RECEPTOR> (...)' strings). Multiple tracers
    for the same receptor are averaged.
    """
    if not neuromaps_cache.exists():
        return None
    npz_path = neuromaps_cache / "annotation_maps.npz"
    labels_path = neuromaps_cache / "annotation_labels.pkl"
    if not npz_path.exists() or not labels_path.exists():
        return None
    try:
        npz = np.load(npz_path, allow_pickle=False)
        if "matrix" in npz.files:
            matrix = np.asarray(npz["matrix"], dtype=np.float32)
        else:
            return None
        with open(labels_path, "rb") as fh:
            labels = list(pickle.load(fh))
        if matrix.shape[0] != len(labels):
            return None
        receptors = [_parse_receptor_token(lab) for lab in labels]
        # Average rows that map to the same receptor token.
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(receptors):
            if r:
                groups.setdefault(r, []).append(i)
        if not groups:
            return None
        kept_symbols = sorted(groups.keys())
        agg = np.stack([matrix[groups[s]].mean(axis=0) for s in kept_symbols])
        return kept_symbols, agg.astype(np.float32)
    except Exception as exc:
        sys.stderr.write(f"WARN: neuromaps loader failed: {exc}\n")
        return None


def _build_term_features(terms: list[str], embed_dim: int = 128) -> np.ndarray:
    """Build deterministic, encoder-free term feature vectors as a baseline.

    Uses a hashing trick over normalized whitespace tokens. The trainer can
    overwrite these with sentence-transformer / OpenAI embeddings via
    --term-embeddings <path.npy>.
    """
    rng = np.random.default_rng(seed=0)
    n = len(terms)
    feats = np.zeros((n, embed_dim), dtype=np.float32)
    for i, t in enumerate(terms):
        toks = [tok for tok in str(t).lower().split() if tok]
        if not toks:
            feats[i] = rng.standard_normal(embed_dim).astype(np.float32) * 0.01
            continue
        for tok in toks:
            h = hash(tok) % (2**31 - 1)
            local = np.random.default_rng(seed=h).standard_normal(embed_dim).astype(np.float32)
            feats[i] += local
        feats[i] /= max(len(toks), 1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats /= norms
    return feats


def _maybe_load_external_embeddings(path: str | None, n_rows: int, name: str) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        sys.stderr.write(f"WARN: --{name}-embeddings {p} not found; falling back to hash features\n")
        return None
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        sys.stderr.write(
            f"WARN: --{name}-embeddings shape {arr.shape} does not match {n_rows} rows; ignoring\n"
        )
        return None
    return arr.astype(np.float32)


def _index(items: list[str]) -> dict[str, int]:
    return {it: i for i, it in enumerate(items)}


def _supervision_edges(
    term_idx: dict[str, int],
    term_maps: np.ndarray,
    n_parcels: int,
    threshold_quantile: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse Term -> Region edge set from the term_maps supervision matrix.

    For each term, edges to regions with values >= per-term `threshold_quantile`
    are kept. The continuous map is the regression target (used in the trainer);
    these discrete edges are used for message passing during encoding.
    """
    if term_maps.shape[1] != n_parcels:
        raise ValueError(
            f"term_maps n_parcels={term_maps.shape[1]} mismatches expected {n_parcels}"
        )
    src, dst = [], []
    for t_label, t_i in term_idx.items():
        row = term_maps[t_i]
        if not np.isfinite(row).any():
            continue
        thr = np.quantile(row, threshold_quantile)
        keep = np.where(row >= thr)[0]
        if keep.size == 0:
            continue
        src.extend([t_i] * len(keep))
        dst.extend(keep.tolist())
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def _term_gene_mentions(
    terms: list[str], gene_set: set[str]
) -> tuple[list[int], list[int], int]:
    """Text-mine gene-symbol mentions in term strings.

    Returns (term_row_indices, gene_symbols_aligned, total_unique_genes_hit).
    Caller is responsible for resolving gene symbols to graph rows.
    """
    src: list[int] = []
    gene_hits: list[str] = []
    seen_genes: set[str] = set()
    for i, t in enumerate(terms):
        toks = set(_GENE_TOKEN_RE.findall(t))
        # Filter: must be in gene_set, not a stopword, length >=3
        matched = {tok for tok in toks if tok in gene_set and tok not in _GENE_MENTION_STOPWORDS}
        for g in matched:
            src.append(i)
            gene_hits.append(g)
            seen_genes.add(g)
    return src, gene_hits, len(seen_genes)


def _ontology_edges(ontology_dir: Path, terms: list[str]) -> dict[str, Any] | None:
    """Best-effort load of the ontology meta-graph. Returns None if unavailable."""
    if not ontology_dir.exists():
        return None
    try:
        from ontology_expansion import load_ontology_index  # type: ignore
        from ontology_meta_graph import build_meta_graph  # type: ignore
    except Exception as exc:
        sys.stderr.write(f"WARN: ontology loaders unavailable ({exc}); skipping ontology edges\n")
        return None
    try:
        idx = load_ontology_index(str(ontology_dir))
        graph = build_meta_graph(idx)
    except Exception as exc:
        sys.stderr.write(f"WARN: building ontology meta-graph failed ({exc}); skipping\n")
        return None
    return {"graph": graph, "index": idx}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-sources", type=str, default="neurolab/data/merged_sources")
    ap.add_argument("--abagen-cache", type=str, default="neurolab/data/abagen_cache")
    ap.add_argument("--neuromaps-cache", type=str, default="neurolab/data/neuromaps_cache")
    ap.add_argument("--receptor-kb", type=str, default="neurolab/data/receptor_knowledge_base.json")
    ap.add_argument("--gene-info", type=str, default="neurolab/data/gene_info.json")
    ap.add_argument("--ontology-dir", type=str, default="neurolab/data/ontologies")
    ap.add_argument("--n-parcels", type=int, default=392)
    ap.add_argument("--feature-dim", type=int, default=128)
    ap.add_argument("--term-embeddings", type=str, default=None,
                    help="Optional .npy of shape (n_terms, d) aligned with merged_sources term order")
    ap.add_argument("--gene-embeddings", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="neurolab/data/kg_brain_graph")
    ap.add_argument("--quantile-threshold", type=float, default=0.9,
                    help="Per-term quantile cutoff for Term->Region message-passing edges")
    ap.add_argument("--no-term-gene-mentions", action="store_true",
                    help="Skip the Term->Gene mention edge extraction. Use to reproduce the "
                         "graph schema from before that edge type was introduced (e.g. to load "
                         "checkpoints trained on the older schema).")
    ap.add_argument("--exclude-ontology", action="store_true",
                    help="Exclude the OntologyConcept node type and its edges entirely. The "
                         "current schema has no Term<->OntologyConcept edges, so these nodes "
                         "contribute nothing to Term predictions while burning message-passing "
                         "compute (~371k nodes, ~900k edges). Requires retraining.")
    args = ap.parse_args()

    torch, HeteroData = _try_import_pyg()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Terms + supervision matrix
    terms, term_maps, term_sources = _load_term_maps(Path(args.merged_sources))
    print(f"[terms] {len(terms)} terms; map matrix {term_maps.shape}")

    # 2. Genes
    gene_info = _load_gene_info(Path(args.gene_info))
    genes = sorted(gene_info.keys())
    print(f"[genes] {len(genes)}")

    # 3. Receptors — use receptor tokens parsed from the neuromaps cache
    # (e.g. "5-HT1B", "D2", "VAChT"). The receptor_knowledge_base.json is a
    # curated list of *gene* symbols and is used below to flag genes.
    receptor_kb = _load_receptor_kb(Path(args.receptor_kb))
    receptor_genes_set = set(receptor_kb.get("genes") or [])
    nmaps_preview = _load_neuromaps_receptor_density(Path(args.neuromaps_cache))
    receptors = sorted(nmaps_preview[0]) if nmaps_preview is not None else []
    print(f"[receptors] {len(receptors)}  (receptor-gene flag set: {len(receptor_genes_set)})")

    # 4. Optional ontology meta-graph
    onto_concepts: list[str] = []
    onto_edges_src: list[int] = []
    onto_edges_dst: list[int] = []
    if args.exclude_ontology:
        print(f"[ontology] EXCLUDED (--exclude-ontology)")
        onto = None
    else:
        onto = _ontology_edges(Path(args.ontology_dir), terms)
    if onto is not None:
        try:
            G = onto["graph"]
            onto_concepts = sorted(list(G.nodes()))
            label_to_idx = _index(onto_concepts)
            for u, v in G.edges():
                if u in label_to_idx and v in label_to_idx:
                    onto_edges_src.append(label_to_idx[u])
                    onto_edges_dst.append(label_to_idx[v])
        except Exception as exc:
            sys.stderr.write(f"WARN: ontology graph extraction failed ({exc})\n")
            onto_concepts = []
    print(f"[ontology] {len(onto_concepts)} concepts; {len(onto_edges_src)} edges")

    # 5. Region nodes
    n_parcels = int(args.n_parcels)
    print(f"[regions] {n_parcels}")

    # ---------- Build features ----------

    _ext_term = _maybe_load_external_embeddings(args.term_embeddings, len(terms), "term")
    term_feats = _ext_term if _ext_term is not None else _build_term_features(terms, args.feature_dim)
    _ext_gene = _maybe_load_external_embeddings(args.gene_embeddings, len(genes), "gene")
    gene_feats = _ext_gene if _ext_gene is not None else _build_term_features(genes, args.feature_dim)
    receptor_feats = _build_term_features(receptors, args.feature_dim) if receptors else np.zeros((0, args.feature_dim), dtype=np.float32)
    onto_feats = _build_term_features(onto_concepts, args.feature_dim) if onto_concepts else np.zeros((0, args.feature_dim), dtype=np.float32)
    region_feats = np.eye(n_parcels, dtype=np.float32)  # learnable embedding will be added on top in the model

    # ---------- Indices ----------
    term_idx = _index(terms)
    gene_idx = _index(genes)
    receptor_idx = _index(receptors)
    onto_idx = _index(onto_concepts)

    # ---------- Edges ----------
    sup_src, sup_dst = _supervision_edges(term_idx, term_maps, n_parcels, args.quantile_threshold)
    print(f"[edges] Term->Region (activates): {sup_src.size}")

    # Gene -> Region from abagen
    abagen = _load_abagen_region_by_gene(Path(args.abagen_cache))
    g2r_src: list[int] = []
    g2r_dst: list[int] = []
    if abagen is not None:
        a_genes, a_matrix = abagen
        if a_matrix.shape[1] == n_parcels:
            for i, g in enumerate(a_genes):
                if g not in gene_idx:
                    continue
                row = a_matrix[i]
                if not np.isfinite(row).any():
                    continue
                thr = np.quantile(row, 0.9)
                keep = np.where(row >= thr)[0]
                if keep.size == 0:
                    continue
                g2r_src.extend([gene_idx[g]] * len(keep))
                g2r_dst.extend(keep.tolist())
        else:
            sys.stderr.write(
                f"WARN: abagen matrix has {a_matrix.shape[1]} cols, expected {n_parcels}; skipping\n"
            )
    print(f"[edges] Gene->Region (expressedIn): {len(g2r_src)}")

    # Receptor -> Region from neuromaps
    nmaps = _load_neuromaps_receptor_density(Path(args.neuromaps_cache))
    rc2r_src: list[int] = []
    rc2r_dst: list[int] = []
    if nmaps is not None and receptors:
        n_labels, n_matrix = nmaps
        if n_matrix.shape[1] == n_parcels:
            for i, lab in enumerate(n_labels):
                if lab not in receptor_idx:
                    continue
                row = n_matrix[i]
                if not np.isfinite(row).any():
                    continue
                thr = np.quantile(row, 0.9)
                keep = np.where(row >= thr)[0]
                rc2r_src.extend([receptor_idx[lab]] * len(keep))
                rc2r_dst.extend(keep.tolist())
        else:
            sys.stderr.write(
                f"WARN: neuromaps matrix has {n_matrix.shape[1]} cols, expected {n_parcels}; skipping\n"
            )
    print(f"[edges] Receptor->Region (densityIn): {len(rc2r_src)}")

    # Gene -> Receptor edges via a small static name map (receptor token -> gene
    # symbols). Only the receptor tokens we actually see in neuromaps need an
    # entry; unmapped tokens contribute no Gene->Receptor edges (still wired
    # via Receptor->Region density edges).
    RECEPTOR_TO_GENES: dict[str, list[str]] = {
        "5-HT1A": ["HTR1A"], "5-HT1B": ["HTR1B"], "5-HT2A": ["HTR2A"],
        "5-HT4": ["HTR4"], "5-HT6": ["HTR6"],
        "D1": ["DRD1"], "D2": ["DRD2", "DRD3"],
        "DAT": ["SLC6A3"], "SERT": ["SLC6A4"], "NET": ["SLC6A2"],
        "VAChT": ["SLC18A3"], "VMAT": ["SLC18A2"],
        "M1": ["CHRM1"], "M2": ["CHRM2"], "M4": ["CHRM4"],
        "MOR": ["OPRM1"], "DOR": ["OPRD1"], "KOR": ["OPRK1"],
        "NMDA": ["GRIN1", "GRIN2A", "GRIN2B"],
        "GABAa": ["GABRA1", "GABRB2", "GABRG2"],
        "mGluR5": ["GRM5"], "CB1": ["CNR1"], "H3": ["HRH3"],
    }
    g2rc_src: list[int] = []
    g2rc_dst: list[int] = []
    for rname in receptors:
        rid = receptor_idx.get(rname)
        if rid is None:
            continue
        for g in RECEPTOR_TO_GENES.get(rname, []):
            if g in gene_idx:
                g2rc_src.append(gene_idx[g])
                g2rc_dst.append(rid)
    print(f"[edges] Gene->Receptor (encodes): {len(g2rc_src)}")

    # ---------- Assemble HeteroData ----------
    data = HeteroData()
    data["Term"].x = torch.tensor(term_feats)
    data["Gene"].x = torch.tensor(gene_feats)
    data["Receptor"].x = torch.tensor(receptor_feats)
    if not args.exclude_ontology:
        data["OntologyConcept"].x = torch.tensor(onto_feats)
    data["Region"].x = torch.tensor(region_feats)

    # Save the supervision target (continuous Term->Region maps) for the trainer.
    data["Term"].y_map = torch.tensor(term_maps)

    def _add_edge(src_type: str, rel: str, dst_type: str, src_idx: list[int] | np.ndarray, dst_idx: list[int] | np.ndarray):
        if len(src_idx) == 0:
            return
        ei = torch.tensor(np.stack([np.asarray(src_idx, dtype=np.int64), np.asarray(dst_idx, dtype=np.int64)]))
        data[(src_type, rel, dst_type)].edge_index = ei
        # reverse for message passing (typed)
        rev_rel = f"rev_{rel}"
        data[(dst_type, rev_rel, src_type)].edge_index = torch.stack([ei[1], ei[0]])

    # Term -> Gene mentions (text-mined from merged_sources term strings)
    if args.no_term_gene_mentions:
        t2g_src, t2g_dst = [], []
        print(f"[edges] Term->Gene (mentions): SKIPPED (--no-term-gene-mentions)")
    else:
        t2g_term_idx_raw, t2g_gene_symbols, _ = _term_gene_mentions(terms, set(gene_idx.keys()))
        t2g_src = [term_idx[terms[i]] for i in t2g_term_idx_raw]
        t2g_dst = [gene_idx[g] for g in t2g_gene_symbols]
        print(f"[edges] Term->Gene (mentions): {len(t2g_src)}")

    _add_edge("Term", "activates", "Region", sup_src, sup_dst)
    _add_edge("Gene", "expressedIn", "Region", g2r_src, g2r_dst)
    _add_edge("Receptor", "densityIn", "Region", rc2r_src, rc2r_dst)
    _add_edge("Gene", "encodes", "Receptor", g2rc_src, g2rc_dst)
    _add_edge("Term", "mentions", "Gene", t2g_src, t2g_dst)
    if not args.exclude_ontology:
        _add_edge("OntologyConcept", "relatedTo", "OntologyConcept", onto_edges_src, onto_edges_dst)

    # ---------- Save ----------
    pt_path = output_dir / "hetero_data.pt"
    torch.save(data, pt_path)

    node_index = {
        "Term": term_idx,
        "Gene": gene_idx,
        "Receptor": receptor_idx,
        "Region": {f"parcel_{i}": i for i in range(n_parcels)},
    }
    if not args.exclude_ontology:
        node_index["OntologyConcept"] = onto_idx
    with open(output_dir / "node_index.pkl", "wb") as fh:
        pickle.dump(node_index, fh)

    meta = {
        "n_parcels": n_parcels,
        "feature_dim": int(term_feats.shape[1]),
        "counts": {
            "Term": len(terms),
            "Gene": len(genes),
            "Receptor": len(receptors),
            "OntologyConcept": len(onto_concepts),
            "Region": n_parcels,
        },
        "edges": {
            "Term__activates__Region": int(sup_src.size),
            "Gene__expressedIn__Region": len(g2r_src),
            "Receptor__densityIn__Region": len(rc2r_src),
            "Gene__encodes__Receptor": len(g2rc_src),
            "Term__mentions__Gene": len(t2g_src),
            "OntologyConcept__relatedTo__OntologyConcept": len(onto_edges_src),
        },
        "term_sources_known": bool(term_sources),
        "quantile_threshold": float(args.quantile_threshold),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nSaved HeteroData to {pt_path}")
    print(f"Saved node_index to {output_dir / 'node_index.pkl'}")
    print(f"Saved meta to {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
