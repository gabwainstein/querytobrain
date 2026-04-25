#!/usr/bin/env python3
"""
Build an expanded text-to-map cache: decoder cache terms + ontology terms with
derived maps (weighted average of related cache-term maps).

This gives a **broader** (term, map) set for training the text-to-brain embedding:
more text labels (ontology terms) mapped into the same parcellated space (Glasser+Tian, 392-D), so the model
sees more varied text and can generalize better.

**Cache source (NeuroQuery vs NeuroSynth):**
  - Default --cache-dir is decoder_cache, which is built from **NeuroQuery only**
    (build_term_maps_cache.py).
  - To use **NeuroSynth too**: build NeuroSynth cache then merge:
      python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache --max-terms 0
      python neurolab/scripts/merge_neuroquery_neurosynth_cache.py \\
        --neuroquery-cache-dir neurolab/data/decoder_cache \\
        --neurosynth-cache-dir neurolab/data/neurosynth_cache \\
        --output-dir neurolab/data/unified_cache
    Then: --cache-dir neurolab/data/unified_cache (expansion uses NQ+NS terms).
  - Or run build_all_maps.py to build decoder_cache + neurosynth_cache + unified_cache in one go.

**Ontologies:** All OBO/OWL files in --ontology-dir are loaded (Cognitive Atlas, GO, UBERON, MF, BFO, RO, etc.).
  Added terms are reported per ontology file.
  To expand **most or all** ontology terms (not just those with 2+ cache matches), use **--min-cache-matches 1**.
  Forward: ontology label -> weighted avg of related cache maps. Reverse: ontology labels that are "related of" cache terms -> weighted avg of those cache maps.
  Use **--no-direction-scale** to skip parent/child downweighting; hierarchy only selects which terms contribute, and the MLP learns the rest from embeddings.

**Include everything (cognitive + biological + task-contrast + disease + gene):** Use --neuromaps-cache-dir,
  --neurovault-cache-dir (from build_neurovault_cache.py), --receptor-path, --enigma-cache-dir (from
  build_enigma_cache.py), and/or --abagen-cache-dir (from build_abagen_cache.py) to merge neuromaps,
  NeuroVault, receptor, ENIGMA disorder maps, and AHBA gene expression into the same (label, map) set.
  The output then contains NQ+NS + ontology + neuromaps + neurovault + receptor + enigma + abagen.

Usage (from repo root):
  python neurolab/scripts/build_expanded_term_maps.py --cache-dir neurolab/data/decoder_cache --ontology-dir neurolab/data/ontologies --output-dir neurolab/data/decoder_cache_expanded
  python neurolab/scripts/build_expanded_term_maps.py --cache-dir neurolab/data/unified_cache --ontology-dir neurolab/data/ontologies --output-dir neurolab/data/unified_cache_expanded
  python neurolab/scripts/build_expanded_term_maps.py --cache-dir neurolab/data/unified_cache --ontology-dir neurolab/data/ontologies --output-dir neurolab/data/full_cache --neuromaps-cache-dir neurolab/data/neuromaps_cache --receptor-path path/to/hansen_400.csv
  python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache_expanded ...
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np


def _is_poor_term(label: str) -> bool:
    """True if label is a generic placeholder with no descriptive content."""
    if not label or not isinstance(label, str):
        return True
    s = label.strip()
    if not s or len(s) < 4:
        return True
    # neurovault_image_N only — no contrast/task/collection info
    if re.match(r"^neurovault_image_\d+$", s):
        return True
    return False


_GRADIENT_PC_RE = re.compile(r"Gene expression gradient PC[1-3]\b", re.I)


def _is_gradient_pc_term(label: str) -> bool:
    """True if label is a gradient PC term; exclude from abagen merge (we add expanded ones explicitly)."""
    if not label or not isinstance(label, str):
        return False
    s = str(label).replace("Gene:", "").strip()
    return bool(_GRADIENT_PC_RE.search(s))


# Map type prefix per source so the model learns modality (fMRI, Structural, PET, Gene, etc.) as an important category.
# Neuromaps/receptor labels already include type; we add prefix only when missing.
# neurovault_pharma/pharma_neurosynth: pharma = from download_neurovault_pharma (searches drug/ketamine/LSD/etc.)
#   and pharma_neurosynth (curated drug meta-analyses). Labels are fMRI contrasts; prefix indicates drug-study origin.
_SOURCE_TO_LABEL_PREFIX = {
    "direct": "fMRI: ",
    "ontology": "fMRI: ",
    "neurovault": "fMRI: ",
    "neurovault_pharma": "fMRI: ",
    "pharma_neurosynth": "fMRI: ",
    "enigma": "Structural: ",
    "abagen": "Gene: ",
    "reference": "Gene: ",
    "receptor": "PET: ",
    "neuromaps": None,   # already has PET:/Cognitive:/Perfusion: in label
    "neuromaps_residual": None,
    "receptor_residual": None,
}
_KNOWN_TYPE_PREFIXES = ("fMRI:", "Pharmacological fMRI:", "PET:", "Structural:", "Gene:", "Cognitive:", "Perfusion:", "DTI:")

# Source identifiers to strip from labels — keep modality (fMRI, PET, etc.), never put source name on term
_SOURCE_STRIP_PATTERNS = [
    (r"^fMRI:\s*NeuroVault fMRI task\s+", "fMRI: "),
    (r"^fMRI:\s*NeuroVault\s+", "fMRI: "),
    (r"^NeuroVault fMRI task\s+", ""),
    (r"^NeuroVault fMRI collection\s+\d+\s+image\s+\d+\s*", ""),
    (r"^NeuroVault\s+", ""),
    (r"^NeuroSynth\s+", ""),
    (r"^NeuroQuery\s+", ""),
    (r"^neuromaps\s+", ""),
    (r"^ENIGMA\s+", ""),
    (r"\bNeuroSynth\s+", ""),   # e.g. "z-value voxel loadings NeuroSynth IC10"
    (r"\(\s*NeuroSynth\s+", "("),  # e.g. "(NeuroSynth meta-analysis)"
]


def _strip_source_from_label(label: str) -> str:
    """Remove source identifiers (NeuroVault, NeuroSynth, etc.) from label. Keep modality."""
    if not label or not isinstance(label, str):
        return label
    s = label.strip()
    for pattern, repl in _SOURCE_STRIP_PATTERNS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s.strip()


def _add_map_type_prefix(label: str, source: str) -> str:
    """Prepend map type (fMRI, PET, Gene, etc.) if not already present. Strip source names from term."""
    if not label:
        return label
    label = _strip_source_from_label(label)
    if not label:
        return label
    s = label.strip()
    if any(s.startswith(p) for p in _KNOWN_TYPE_PREFIXES):
        return label
    prefix = _SOURCE_TO_LABEL_PREFIX.get(source)
    if prefix:
        return f"{prefix}{label}"
    return label

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def main():
    parser = argparse.ArgumentParser(description="Build expanded (cache + ontology) term maps for broader text-to-map training")
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache", help="Decoder cache: NeuroQuery (decoder_cache) or NQ+NS merged (unified_cache)")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies", help="Ontology directory (OBO/OWL)")
    parser.add_argument("--output-dir", default="neurolab/data/decoder_cache_expanded", help="Save term_maps.npz and term_vocab.pkl here")
    parser.add_argument("--min-cache-matches", type=int, default=2, help="Add ontology term only if it expands to at least this many cache terms (default 2). Use 1 to expand most/all ontology terms (single match inherits that map).")
    parser.add_argument("--min-pairwise-correlation", type=float, default=None, help="If set (e.g. 0.3), skip ontology term when mean pairwise correlation of matched cache maps is below this (avoids meaningless averages)")
    parser.add_argument("--save-term-sources", action="store_true", help="Save term_sources.pkl (direct|ontology|neuromaps|receptor) for sample weighting in training")
    parser.add_argument("--neuromaps-cache-dir", default=None, help="Merge neuromaps annotations (annotation_maps.npz, annotation_labels.pkl) into the set")
    parser.add_argument("--neurovault-cache-dir", default=None, help="Merge NeuroVault task-contrast cache (term_maps.npz, term_vocab.pkl from build_neurovault_cache.py) into the set")
    parser.add_argument("--receptor-path", default=None, help="Merge receptor atlas (e.g. Hansen CSV/NPZ) maps into the set")
    parser.add_argument("--enigma-cache-dir", default=None, help="Merge ENIGMA disorder maps (term_maps.npz, term_vocab.pkl from build_enigma_cache.py) into the set")
    parser.add_argument("--abagen-cache-dir", default=None, help="Merge abagen gene expression (term_maps.npz, term_vocab.pkl from build_abagen_cache.py) into the set")
    parser.add_argument("--additional-abagen-cache-dir", default=None, help="Merge an additional gene cache (e.g. receptor residual-selected denoised) into the set. Same format as abagen; source=abagen.")
    parser.add_argument("--exclude-abagen", action="store_true", help="Do NOT add abagen to training set. Use abagen as retrieval/annotation only (Option B: gene maps via term_to_map abagen-by-name, not regression).")
    parser.add_argument("--receptor-reference-cache-dir", default=None, help="Merge receptor reference cache (250 genes + PCs with rich labels from build_receptor_reference_cache.py); tagged as 'reference' for 5%% batch share")
    parser.add_argument("--neurovault-pharma-cache-dir", default=None, help="Merge NeuroVault pharmacological cache (drug-related contrasts)")
    parser.add_argument("--pharma-neurosynth-cache-dir", default=None, help="Merge pharmacological NeuroSynth meta-analysis (ketamine, caffeine, etc.)")
    parser.add_argument("--max-abagen-terms", type=int, default=0, help="When merging abagen, add at most this many terms. 0 = add all. When set: tiered selection (receptor + cluster medoids + residual-variance or medoids). Recommend 300-500.")
    parser.add_argument("--abagen-n-clusters", type=int, default=32, help="When max-abagen-terms set: number of WGCNA-style co-expression clusters (Tier 2). Default 32 (Hawrylycz et al.).")
    parser.add_argument("--abagen-add-gradient-pcs", type=int, default=0, help="When merging abagen: add this many gene-expression gradient maps (PCs of full abagen matrix). Recommend 3 (PC1-3 robust; PC4-5 often less generalizable). 0 = off.")
    parser.add_argument("--gradient-pc-label-style", choices=("current", "hybrid", "standard", "brain_context", "distinctive", "short", "dominant", "distinct"), default="current", help="Gradient PC label format: current (semantic only), hybrid (PC1+semantic), standard (literature terms), brain_context, distinctive, short, dominant, distinct (maximally different phrasing per PC)")
    parser.add_argument("--abagen-tier3-method", choices=("medoids", "residual_variance"), default="residual_variance", help="Tier 3 selection: residual_variance = high variance after regressing top PCs; medoids = cluster medoids. Default residual_variance.")
    parser.add_argument("--receptor-top-percentile", type=float, default=50.0, help="When Tier 1 uses residual-variance ranking: keep top X%% of receptors by residual variance. Default 50 (top half). 100 = all receptors.")
    parser.add_argument("--add-pet-residuals", action="store_true", help="Add gradient-regressed PET/receptor maps as extra terms (label_residual, source neuromaps_residual/receptor_residual). Requires --abagen-add-gradient-pcs > 0 and neuromaps/receptor merged.")
    parser.add_argument("--abagen-gene-info", default="neurolab/data/gene_info.json", help="JSON mapping gene symbol -> name for enriching labels. Default neurolab/data/gene_info.json (run download_gene_info.py first). Receptor KB genes use get_enriched_gene_labels; others use this.")
    parser.add_argument("--no-abagen-enrich-labels", action="store_true", help="Do not re-label abagen terms; keep plain 'Gene: SYMBOL gene expression from Allen...' format.")
    parser.add_argument("--abagen-pca-variance", type=float, default=0.0, help="If >0 (e.g. 0.95), denoise main abagen maps by projecting onto PCs explaining this variance before adding. 0 = off.")
    parser.add_argument("--gene-pca-variance", type=float, default=0.0, help="If >0 (e.g. 0.95), fit PCA on abagen maps only, store gene loadings and PCA in cache; trainer can use a gene head that predicts PC loadings. 0 = off.")
    parser.add_argument("--relation-weights-file", default=None, help="JSON from ontology_brain_correlation.py --output-weights: use observed mean r per relation type as blend weights (data-driven)")
    parser.add_argument("--no-direction-scale", action="store_true", help="Do not apply parent/child direction scaling when blending; use only ontology relation weights. Lets the MLP learn hierarchy from embeddings.")
    parser.add_argument("--use-graph-distance", action="store_true", help="Use gamma^path_length weighting (per-ontology min path) instead of relation-type weights; requires OBO ontologies")
    parser.add_argument("--graph-distance-gamma", type=float, default=0.8, help="Decay factor for path-length weighting when --use-graph-distance (default 0.8)")
    parser.add_argument("--no-ontology", action="store_true", help="Skip ontology expansion; only merge base cache + neuromaps + neurovault + enigma + abagen (use for merged_sources)")
    parser.add_argument("--no-skip-poor-terms", action="store_true", help="Include maps with generic labels (e.g. neurovault_image_N). Default: skip them.")
    parser.add_argument("--truncate-to-392", action="store_true", help="Truncate 427-parcel maps to 392 (Glasser+Tian only). Use when upstream caches are 427 but training expects 392.")
    parser.add_argument("--zscore-renormalize", action="store_true", help="(Deprecated) Re-apply z-score to merged maps. Default: no re-normalization — each source is already normalized appropriately (global for fMRI, cortex/subcortex separate for gene/receptor/structural).")
    args = parser.parse_args()

    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(repo_root, args.cache_dir)
    ontology_dir = args.ontology_dir if os.path.isabs(args.ontology_dir) else os.path.join(repo_root, args.ontology_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        print("Decoder cache not found. Build first: python neurolab/scripts/build_term_maps_cache.py --cache-dir ...", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)
    term_maps = np.asarray(data["term_maps"])
    with open(pkl_path, "rb") as f:
        term_vocab = pickle.load(f)
    assert term_maps.shape[0] == len(term_vocab)
    n_parcels = term_maps.shape[1]
    if getattr(args, "truncate_to_392", False) and n_parcels > 392:
        term_maps = term_maps[:, :392].astype(term_maps.dtype)
        n_parcels = 392
        print("Truncated to 392 parcels (Glasser+Tian)")
    n_cache = len(term_vocab)
    vocab_to_idx = {t: i for i, t in enumerate(term_vocab)}
    print(f"Cache: {n_cache} terms x {n_parcels} parcels")

    if args.no_ontology:
        ontology_index = {"label_to_related": {}}
        print("Skipping ontology expansion (--no-ontology); merging sources only.")
    elif not os.path.isdir(ontology_dir):
        print("Ontology dir not found; saving cache-only (no ontology terms).", file=sys.stderr)
        ontology_index = {"label_to_related": {}}
    else:
        from ontology_expansion import load_ontology_index, expand_term

        relation_weights = None
        if getattr(args, "relation_weights_file", None):
            import json
            rw_path = args.relation_weights_file if os.path.isabs(args.relation_weights_file) else os.path.join(repo_root, args.relation_weights_file)
            if os.path.exists(rw_path):
                with open(rw_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                relation_weights = {k: float(v) for k, v in raw.items() if not k.startswith("_")}
                print(f"Relation weights (data-driven): {relation_weights}")
            else:
                print(f"Relation weights file not found: {rw_path}", file=sys.stderr)
        ontology_index = load_ontology_index(ontology_dir, relation_weights=relation_weights)
        label_to_related = ontology_index.get("label_to_related") or {}
        print(f"Ontology: {len(label_to_related)} labels")

    # Decoder set (normalized) to skip ontology labels that are already in cache
    from ontology_expansion import _normalize_term

    decoder_set = {_normalize_term(t) for t in term_vocab}
    skip_poor_terms = not getattr(args, "no_skip_poor_terms", False)

    SAMPLE_WEIGHT_BY_SOURCE = {"direct": 1.0, "neurovault": 0.8, "ontology": 0.6, "neuromaps": 1.0, "receptor": 1.0, "neuromaps_residual": 0.6, "receptor_residual": 0.6, "enigma": 0.5, "abagen": 0.4, "reference": 0.6, "neurovault_pharma": 1.2, "pharma_neurosynth": 1.2}
    new_terms = [_add_map_type_prefix(t, "direct") for t in term_vocab]
    new_maps = [term_maps[i] for i in range(n_cache)]
    term_sources = ["direct"] * n_cache if args.save_term_sources else None  # direct = from base cache (NQ/NS)
    term_sample_weights = [SAMPLE_WEIGHT_BY_SOURCE["direct"]] * n_cache if args.save_term_sources else None
    abagen_gradient_components_to_save = None  # set when --abagen-add-gradient-pcs > 0; saved for PET residual-correlation evaluation

    if ontology_index.get("label_to_related"):
        from ontology_expansion import expand_term, DIRECTION_SCALE

        added = 0
        added_by_source: dict[str, int] = {}
        label_to_source = ontology_index.get("label_to_source") or {}
        for label_norm in ontology_index["label_to_related"]:
            if label_norm in decoder_set:
                continue
            related = expand_term(
                label_norm,
                term_vocab,
                ontology_index,
                use_graph_distance=getattr(args, "use_graph_distance", False),
                gamma=getattr(args, "graph_distance_gamma", 0.8),
            )
            if len(related) < args.min_cache_matches:
                continue
            # Weighted average of cache maps (optional direction scaling: parent too broad → downweight)
            indices = []
            weights = []
            scale = 1.0 if getattr(args, "no_direction_scale", False) else None
            for item in related:
                t, w = item[0], item[1]
                rtype = item[2] if len(item) >= 3 else "other"
                adj = scale if scale is not None else DIRECTION_SCALE.get(rtype, 0.8)
                i = vocab_to_idx.get(t)
                if i is not None:
                    indices.append(i)
                    weights.append(w * adj)
            if not indices:
                continue
            # Optional: skip if matched cache maps are too dissimilar (average would be meaningless)
            if args.min_pairwise_correlation is not None and len(indices) >= 2:
                sub = term_maps[indices]
                pairs = 0
                corr_sum = 0.0
                for a in range(len(indices)):
                    for b in range(a + 1, len(indices)):
                        r = np.corrcoef(sub[a], sub[b])[0, 1]
                        if np.isfinite(r):
                            corr_sum += r
                            pairs += 1
                if pairs > 0:
                    mean_r = corr_sum / pairs
                    if mean_r < args.min_pairwise_correlation:
                        continue
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            map_derived = np.average(term_maps[indices], axis=0, weights=weights)
            new_terms.append(_add_map_type_prefix(label_norm, "ontology"))
            new_maps.append(map_derived)
            if term_sources is not None:
                term_sources.append("ontology")
                term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["ontology"])
            added += 1
            src = label_to_source.get(label_norm, "?")
            added_by_source[src] = added_by_source.get(src, 0) + 1
        print(f"Added {added} ontology terms with derived maps (label -> weighted avg of related cache maps)")
        if added_by_source:
            by_src = sorted(added_by_source.items(), key=lambda x: -x[1])
            for fname, count in by_src:
                print(f"  by ontology: {fname} -> {count}")

        # Reverse expansion: ontology labels that appear only as "related" (e.g. parent of a cache term)
        # but are not themselves keys in label_to_related. Embed as weighted average of all cache terms
        # that point to them (so "executive function" -> blend of working memory, attention, etc., not
        # just one child's map).
        # Build: for each (ontology_label, cache_term), weight from ontology (with direction scale)
        label_to_cache_weights = {}  # label_norm -> [(cache_term, adjusted_weight), ...]
        scale_rev = 1.0 if getattr(args, "no_direction_scale", False) else None
        for label_norm, related_list in ontology_index["label_to_related"].items():
            for item in related_list:
                rel_name = item[0]
                w = item[1]
                rtype = item[2] if len(item) >= 3 else "other"
                adj = scale_rev if scale_rev is not None else DIRECTION_SCALE.get(rtype, 0.8)
                key = _normalize_term(rel_name)
                if not key:
                    continue
                if key in decoder_set:
                    orig = next((t for t in term_vocab if _normalize_term(t) == key), rel_name)
                    label_to_cache_weights.setdefault(label_norm, []).append((orig, w * adj))
        added_rev = 0
        seen_labels = set(new_terms)
        for label_norm, cache_weight_list in label_to_cache_weights.items():
            if label_norm in seen_labels:
                continue
            # Skip if we already added this label in forward expansion (it's a key in label_to_related with >= min matches)
            if label_norm in ontology_index["label_to_related"]:
                rel = expand_term(
                    label_norm,
                    term_vocab,
                    ontology_index,
                    use_graph_distance=getattr(args, "use_graph_distance", False),
                    gamma=getattr(args, "graph_distance_gamma", 0.8),
                )
                if len(rel) >= args.min_cache_matches:
                    continue  # already added in forward pass
            indices = []
            weights = []
            for cache_term, w in cache_weight_list:
                i = vocab_to_idx.get(cache_term)
                if i is not None:
                    indices.append(i)
                    weights.append(w)
            if not indices:
                continue
            if args.min_pairwise_correlation is not None and len(indices) >= 2:
                sub = term_maps[indices]
                pairs = 0
                corr_sum = 0.0
                for a in range(len(indices)):
                    for b in range(a + 1, len(indices)):
                        r = np.corrcoef(sub[a], sub[b])[0, 1]
                        if np.isfinite(r):
                            corr_sum += r
                            pairs += 1
                if pairs > 0 and (corr_sum / pairs) < args.min_pairwise_correlation:
                    continue
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            map_derived = np.average(term_maps[indices], axis=0, weights=weights)
            seen_labels.add(label_norm)
            new_terms.append(_add_map_type_prefix(label_norm, "ontology"))
            new_maps.append(map_derived)
            if term_sources is not None:
                term_sources.append("ontology")
                term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["ontology"])
            added_rev += 1
        print(f"Added {added_rev} reverse labels (ontology labels as related of cache terms -> weighted avg of those cache maps)")
    else:
        print("No ontology index; output is cache-only.")

    truncate_392 = getattr(args, "truncate_to_392", False) and n_parcels == 392

    # Merge neuromaps annotations (biological labels -> maps) into the set
    seen = {_normalize_term(t) for t in new_terms}
    if args.neuromaps_cache_dir:
        nm_dir = args.neuromaps_cache_dir if os.path.isabs(args.neuromaps_cache_dir) else os.path.join(repo_root, args.neuromaps_cache_dir)
        nm_npz = os.path.join(nm_dir, "annotation_maps.npz")
        nm_pkl = os.path.join(nm_dir, "annotation_labels.pkl")
        if os.path.exists(nm_npz) and os.path.exists(nm_pkl):
            nm_data = np.load(nm_npz)
            nm_maps = np.asarray(nm_data["matrix"])
            if truncate_392 and nm_maps.shape[1] > 392:
                nm_maps = nm_maps[:, :392].astype(np.float64)
            with open(nm_pkl, "rb") as f:
                nm_labels = pickle.load(f)
            nm_labels = list(nm_labels) if isinstance(nm_labels, (list, tuple)) else list(nm_labels.keys())
            if nm_maps.shape[1] != n_parcels or len(nm_labels) != nm_maps.shape[0]:
                print("Neuromaps cache shape mismatch; skipping.", file=sys.stderr)
            else:
                added_nm = 0
                for label, row in zip(nm_labels, nm_maps):
                    prefixed = _add_map_type_prefix(label, "neuromaps")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        new_maps.append(row.astype(np.float64))
                        if term_sources is not None:
                            term_sources.append("neuromaps")
                            term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["neuromaps"])
                        added_nm += 1
                print(f"Added {added_nm} neuromaps annotations (biological labels -> maps)")
        else:
            print("Neuromaps cache not found at", nm_dir, "; skipping.", file=sys.stderr)

    # Merge NeuroVault task-contrast cache (contrast_definition/task/name -> map) into the set
    if args.neurovault_cache_dir:
        nv_dir = args.neurovault_cache_dir if os.path.isabs(args.neurovault_cache_dir) else os.path.join(repo_root, args.neurovault_cache_dir)
        nv_npz = os.path.join(nv_dir, "term_maps.npz")
        nv_pkl = os.path.join(nv_dir, "term_vocab.pkl")
        nv_weights_pkl = os.path.join(nv_dir, "term_sample_weights.pkl")
        nv_weights = None
        if os.path.exists(nv_weights_pkl):
            with open(nv_weights_pkl, "rb") as f:
                nv_weights = pickle.load(f)
        if os.path.exists(nv_npz) and os.path.exists(nv_pkl):
            nv_data = np.load(nv_npz)
            nv_maps = np.asarray(nv_data["term_maps"])
            if truncate_392 and nv_maps.shape[1] > 392:
                nv_maps = nv_maps[:, :392].astype(np.float64)
            with open(nv_pkl, "rb") as f:
                nv_terms = pickle.load(f)
            nv_terms = list(nv_terms)
            if nv_maps.shape[1] != n_parcels or len(nv_terms) != nv_maps.shape[0]:
                print("NeuroVault cache shape mismatch; skipping.", file=sys.stderr)
            elif nv_weights is not None and len(nv_weights) != len(nv_terms):
                nv_weights = None
            else:
                added_nv = 0
                for i, (label, row) in enumerate(zip(nv_terms, nv_maps)):
                    prefixed = _add_map_type_prefix(label, "neurovault")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        new_maps.append(row.astype(np.float64))
                        if term_sources is not None:
                            term_sources.append("neurovault")
                            w = nv_weights[i] if nv_weights is not None else SAMPLE_WEIGHT_BY_SOURCE["neurovault"]
                            term_sample_weights.append(float(w))
                        added_nv += 1
                print(f"Added {added_nv} NeuroVault task-contrast maps (contrast/task labels -> maps)")
        else:
            print("NeuroVault cache not found at", nv_dir, "; skipping.", file=sys.stderr)

    def _merge_term_cache(cache_dir, source_name):
        """Merge a (term_maps.npz, term_vocab.pkl) cache into new_terms/new_maps.
        Uses per-term weights from term_sample_weights.pkl when present."""
        if not cache_dir:
            return
        d = cache_dir if os.path.isabs(cache_dir) else os.path.join(repo_root, cache_dir)
        npz = os.path.join(d, "term_maps.npz")
        pkl = os.path.join(d, "term_vocab.pkl")
        weights_pkl = os.path.join(d, "term_sample_weights.pkl")
        if not os.path.exists(npz) or not os.path.exists(pkl):
            return
        data = np.load(npz)
        maps = np.asarray(data["term_maps"])
        if truncate_392 and maps.shape[1] > 392:
            maps = maps[:, :392].astype(np.float64)
        with open(pkl, "rb") as f:
            terms = pickle.load(f)
        terms = list(terms)
        weights = None
        if os.path.exists(weights_pkl):
            with open(weights_pkl, "rb") as f:
                weights = pickle.load(f)
            if weights is not None and len(weights) != len(terms):
                weights = None
        if maps.shape[1] != n_parcels or len(terms) != maps.shape[0]:
            print(f"{source_name} cache shape mismatch; skipping.", file=sys.stderr)
            return
        added = 0
        for i, (label, row) in enumerate(zip(terms, maps)):
            prefixed = _add_map_type_prefix(label, source_name)
            norm = _normalize_term(prefixed)
            if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                seen.add(norm)
                new_terms.append(prefixed)
                new_maps.append(row.astype(np.float64))
                if term_sources is not None:
                    term_sources.append(source_name)
                    w = weights[i] if weights is not None else SAMPLE_WEIGHT_BY_SOURCE.get(source_name, 0.5)
                    term_sample_weights.append(float(w))
                added += 1
        if added:
            print(f"Added {added} {source_name} maps")

    if args.neurovault_pharma_cache_dir:
        _merge_term_cache(args.neurovault_pharma_cache_dir, "neurovault_pharma")
    if args.pharma_neurosynth_cache_dir:
        _merge_term_cache(args.pharma_neurosynth_cache_dir, "pharma_neurosynth")

    # Merge receptor atlas (receptor name -> map) into the set
    if args.receptor_path:
        rec_path = args.receptor_path if os.path.isabs(args.receptor_path) else os.path.join(repo_root, args.receptor_path)
        if os.path.exists(rec_path):
            try:
                from neurolab.enrichment.receptor_enrichment import ReceptorEnrichment
                from neurolab.parcellation import zscore_cortex_subcortex_separately
                rec = ReceptorEnrichment(receptor_matrix_path=rec_path, n_parcels=n_parcels)
                added_rec = 0
                systems = getattr(rec, "receptor_systems", None) or [""] * len(rec.receptor_names)
                for name, system, row in zip(rec.receptor_names, systems, rec.matrix):
                    if system:
                        label = f"PET: {name} ({system} receptor)"
                    else:
                        label = f"PET: {name}"
                    prefixed = _add_map_type_prefix(label, "receptor")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        # Normalize cortex/subcortex separately (audit §2.9: receptor may be raw)
                        row_norm = zscore_cortex_subcortex_separately(row.astype(np.float64))
                        new_maps.append(row_norm)
                        if term_sources is not None:
                            term_sources.append("receptor")
                            term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["receptor"])
                        added_rec += 1
                print(f"Added {added_rec} receptor atlas maps")
            except Exception as e:
                print("Receptor load failed:", e, "; skipping.", file=sys.stderr)
        else:
            print("Receptor path not found:", rec_path, "; skipping.", file=sys.stderr)

    # Merge ENIGMA disorder cache (structural: cortical thickness, subcortical volume)
    if args.enigma_cache_dir:
        eg_dir = args.enigma_cache_dir if os.path.isabs(args.enigma_cache_dir) else os.path.join(repo_root, args.enigma_cache_dir)
        eg_npz = os.path.join(eg_dir, "term_maps.npz")
        eg_pkl = os.path.join(eg_dir, "term_vocab.pkl")
        if os.path.exists(eg_npz) and os.path.exists(eg_pkl):
            eg_data = np.load(eg_npz)
            eg_maps = np.asarray(eg_data["term_maps"])
            if truncate_392 and eg_maps.shape[1] > 392:
                eg_maps = eg_maps[:, :392].astype(np.float64)
            with open(eg_pkl, "rb") as f:
                eg_terms = pickle.load(f)
            eg_terms = list(eg_terms)
            if eg_maps.shape[1] == n_parcels and len(eg_terms) == eg_maps.shape[0]:
                added_eg = 0
                for label, row in zip(eg_terms, eg_maps):
                    prefixed = _add_map_type_prefix(label, "enigma")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        new_maps.append(row.astype(np.float64))
                        if term_sources is not None:
                            term_sources.append("enigma")
                            term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["enigma"])
                        added_eg += 1
                print(f"Added {added_eg} ENIGMA disorder maps (structural)")
            else:
                print("ENIGMA cache shape mismatch; skipping.", file=sys.stderr)
        else:
            print("ENIGMA cache not found at", eg_dir, "; skipping.", file=sys.stderr)

    # Merge abagen gene expression cache (Allen Institute AHBA). Tiered selection as in abagen_tiered_gene_selection.md:
    # Tier 1 = receptor genes (~250), Tier 2 = WGCNA-style cluster medoids (--abagen-n-clusters), Tier 3 = residual_variance (default) or medoids.
    # Gradient PCs (--abagen-add-gradient-pcs) and PET residuals (--add-pet-residuals) for training/eval.
    def _is_valid_gene_symbol(sym):
        """Gene symbols: 2-20 chars, alphanumeric + hyphen/underscore."""
        if not sym or len(sym) < 2 or len(sym) > 20:
            return False
        return sym.replace("-", "").replace("_", "").isalnum()

    def _gene_symbol_from_abagen_label(label):
        """Extract gene symbol from abagen term label. Handles:
        - 'Gene: SYMBOL (full name), signaling' -> SYMBOL
        - 'SYMBOL gene expression from Allen Human Brain Atlas (SYMBOL)' -> SYMBOL
        - 'SYMBOL (name), signaling' (receptor-enriched) -> SYMBOL
        Returns empty string for gradient PCs and other non-gene terms."""
        if not label or not isinstance(label, str):
            return ""
        s = label.strip()
        # Skip gradient PCs and non-gene terms
        if "gradient" in s.lower() or s.startswith("Gene expression gradient PC"):
            return ""
        # Format: "Gene: SYMBOL (full name), ..." — symbol is first token before "("
        if s.startswith("Gene: "):
            rest = s[6:].strip()
            if "(" in rest:
                sym = rest.split("(")[0].strip().upper()
            else:
                sym = (rest.split()[0].upper() if rest.split() else "")
            if _is_valid_gene_symbol(sym):
                return sym
        # Format: "SYMBOL gene expression from Allen..." — first token is symbol
        if " gene expression from Allen" in s or " gene expression" in s:
            sym = s.split()[0].upper() if s.split() else ""
            if _is_valid_gene_symbol(sym):
                return sym
        # Format: "SYMBOL (full name), signaling" - first token is symbol
        if "(" in s and "), " in s:
            sym = s.split("(")[0].strip().upper()
            if _is_valid_gene_symbol(sym):
                return sym
        # Fallback: last paren content if it looks like a symbol (e.g. "... (SYMBOL)")
        if "(" in s and ")" in s:
            inner = s[s.rfind("(") + 1 : s.rfind(")")].strip().upper()
            if _is_valid_gene_symbol(inner):
                return inner
        return ""

    def _cluster_medoids_spatial(maps, global_indices, n_clusters):
        """WGCNA-style: cluster genes by spatial correlation, return one medoid (hub) per cluster.
        maps: (n_genes, n_parcels); global_indices: list of n_genes indices. Returns list of global indices (medoids)."""
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, fcluster
        n = len(global_indices)
        if n <= 0 or n_clusters <= 0:
            return []
        if n_clusters >= n:
            return list(global_indices)
        # Correlation (n, n); use 1 - corr as distance
        condensed = pdist(maps, metric="correlation")
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=min(n_clusters, n), criterion="maxclust")
        corr = np.corrcoef(maps)
        medoids = []
        for c in sorted(set(labels)):
            inds = np.where(labels == c)[0]
            # Medoid: index (within this subset) that maximizes sum of correlation to cluster members
            i_local = inds[np.argmax([corr[i, inds].sum() for i in inds])]
            medoids.append(global_indices[i_local])
        return medoids

    if args.abagen_cache_dir and not getattr(args, "exclude_abagen", False):
        ab_dir = args.abagen_cache_dir if os.path.isabs(args.abagen_cache_dir) else os.path.join(repo_root, args.abagen_cache_dir)
        ab_npz = os.path.join(ab_dir, "term_maps.npz")
        ab_pkl = os.path.join(ab_dir, "term_vocab.pkl")
        if os.path.exists(ab_npz) and os.path.exists(ab_pkl):
            ab_data = np.load(ab_npz)
            ab_maps_full = np.asarray(ab_data["term_maps"])
            if truncate_392 and ab_maps_full.shape[1] > 392:
                ab_maps_full = ab_maps_full[:, :392].astype(np.float64)
            with open(ab_pkl, "rb") as f:
                ab_terms_full = pickle.load(f)
            ab_terms_full = list(ab_terms_full)
            if ab_maps_full.shape[1] == n_parcels and len(ab_terms_full) == ab_maps_full.shape[0]:
                max_ab = getattr(args, "max_abagen_terms", 0)
                n_clusters_t2 = getattr(args, "abagen_n_clusters", 32)
                n_gradient_pcs = getattr(args, "abagen_add_gradient_pcs", 0)
                tier3_method = getattr(args, "abagen_tier3_method", "residual_variance")
                receptor_top_pct = getattr(args, "receptor_top_percentile", 50.0)
                n_ab_total = len(ab_terms_full)
                ab_maps = ab_maps_full
                ab_terms = ab_terms_full
                # PCA on full matrix when: adding gradient maps and/or Tier3 = residual_variance
                n_pcs_for_residual = 5 if tier3_method == "residual_variance" else 0
                n_pcs_fit = max(n_gradient_pcs, n_pcs_for_residual)
                pca_ab = None
                if n_pcs_fit > 0 and ab_maps.shape[0] > n_pcs_fit:
                    from sklearn.decomposition import PCA
                    pca_ab = PCA(n_components=min(n_pcs_fit, ab_maps.shape[0] - 1, ab_maps.shape[1]), random_state=42)
                    pca_ab.fit(ab_maps)
                # Add gradient maps as synthetic terms (dominant spatial axes)
                style = getattr(args, "gradient_pc_label_style", "current") or "current"
                GRADIENT_PC_LABEL_STYLES = {
                    "current": {
                        1: "Gene expression gradient: sensorimotor-association, cortical hierarchy, myelin axis",
                        2: "Gene expression gradient: cognitive metabolism",
                        3: "Gene expression gradient: adolescent plasticity, synaptic-immune",
                    },
                    "hybrid": {
                        1: "Gene expression gradient PC1: sensorimotor-association, cortical hierarchy, myelin axis",
                        2: "Gene expression gradient PC2: cognitive metabolism",
                        3: "Gene expression gradient PC3: adolescent plasticity, synaptic-immune",
                    },
                    "standard": {
                        1: "Principal cortical gradient: sensorimotor to transmodal association",
                        2: "Secondary gradient: visual-somatomotor and metabolic",
                        3: "Tertiary gradient: developmental and neuroimmune",
                    },
                    "brain_context": {
                        1: "Cortical gene expression gradient: sensorimotor-association axis (primary organizational axis)",
                        2: "Brain-wide gene expression: cognitive metabolism axis",
                        3: "Cortical gene expression: developmental and synaptic-immune gradient",
                    },
                    "distinctive": {
                        1: "Sensorimotor-association gradient (primary cortical hierarchy)",
                        2: "Metabolic and oxidative gradient (cognitive demand)",
                        3: "Developmental and synaptic-immune gradient",
                    },
                    "short": {
                        1: "Sensorimotor-association axis",
                        2: "Cognitive metabolism axis",
                        3: "Developmental plasticity axis",
                    },
                    "dominant": {
                        1: "Dominant gene expression axis 1: sensorimotor-association, myelin",
                        2: "Dominant gene expression axis 2: cognitive metabolism",
                        3: "Dominant gene expression axis 3: adolescent plasticity, synaptic-immune",
                    },
                    "distinct": {
                        1: "Cortical gene expression gradient: sensorimotor-association axis (primary organizational axis)",
                        2: "Metabolic gradient: oxidative vs. synaptic pathways",
                        3: "Developmental gradient: adolescent plasticity and neuroimmune axis",
                    },
                }
                GRADIENT_PC_LABELS = GRADIENT_PC_LABEL_STYLES.get(style, GRADIENT_PC_LABEL_STYLES["current"])
                if n_gradient_pcs > 0 and pca_ab is not None and pca_ab.n_components_ >= 1:
                    n_add = min(n_gradient_pcs, pca_ab.n_components_)
                    for i in range(n_add):
                        label = GRADIENT_PC_LABELS.get(i + 1) or f"Gene expression gradient PC{i + 1}"
                        row = pca_ab.components_[i].astype(np.float64)
                        prefixed = _add_map_type_prefix(label, "abagen")
                        norm = _normalize_term(prefixed)
                        if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                            seen.add(norm)
                            new_terms.append(prefixed)
                            new_maps.append(row)
                            if term_sources is not None:
                                term_sources.append("abagen")
                                term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["abagen"])
                    print(f"Abagen: added {n_add} gene expression gradient maps (semantic labels)")
                    abagen_gradient_components_to_save = pca_ab.components_[:n_add].astype(np.float64)
                if max_ab and n_ab_total > max_ab:
                    receptor_set = set()
                    try:
                        from neurolab.receptor_kb import load_receptor_genes
                        receptor_set = set(g.upper() for g in load_receptor_genes())
                    except Exception:
                        pass
                    symbols = [_gene_symbol_from_abagen_label(t) for t in ab_terms]
                    gradient_pc_idx = set(i for i in range(n_ab_total) if _is_gradient_pc_term(ab_terms_full[i]))
                    receptor_idx = [i for i in range(n_ab_total) if symbols[i] in receptor_set and i not in gradient_pc_idx]
                    other_idx = [i for i in range(n_ab_total) if i not in receptor_idx and i not in gradient_pc_idx]
                    n_tier1 = min(len(receptor_idx), max_ab)
                    if tier3_method == "residual_variance" and receptor_top_pct < 100:
                        n_tier1_cap = max(1, int(len(receptor_idx) * receptor_top_pct / 100))
                        n_tier1 = min(n_tier1, n_tier1_cap)
                    # Tier 1: rank receptors by residual variance (Fulcher), take top n_tier1
                    if receptor_idx and n_tier1 > 0 and tier3_method == "residual_variance" and pca_ab is not None and n_pcs_for_residual > 0:
                        rec_reconstructed = pca_ab.inverse_transform(pca_ab.transform(ab_maps[receptor_idx]))
                        rec_residuals = ab_maps[receptor_idx] - rec_reconstructed
                        rec_residual_var = np.var(rec_residuals, axis=1)
                        rec_order = np.argsort(-rec_residual_var)[:n_tier1]
                        tier1_indices = [receptor_idx[j] for j in rec_order]
                    else:
                        tier1_indices = receptor_idx[:n_tier1]
                    tier2_indices = _cluster_medoids_spatial(
                        ab_maps[other_idx], other_idx, min(n_clusters_t2, len(other_idx))
                    ) if other_idx and n_clusters_t2 > 0 else []
                    remaining = [i for i in other_idx if i not in tier2_indices]
                    n_slots_left = max(0, max_ab - n_tier1 - len(tier2_indices))
                    if n_slots_left <= 0:
                        tier3_indices = []
                    elif len(remaining) <= n_slots_left:
                        tier3_indices = remaining
                    elif tier3_method == "residual_variance" and pca_ab is not None and n_pcs_for_residual > 0:
                        # Genes with high variance after regressing out top PCs add unique spatial info (informed by Fulcher et al.)
                        reconstructed = pca_ab.inverse_transform(pca_ab.transform(ab_maps[remaining]))
                        residuals = ab_maps[remaining] - reconstructed
                        residual_var = np.var(residuals, axis=1)
                        order = np.argsort(-residual_var)[:n_slots_left]
                        tier3_indices = [remaining[j] for j in order]
                    else:
                        tier3_indices = _cluster_medoids_spatial(
                            ab_maps[remaining], remaining, n_slots_left
                        )
                    idx = sorted(tier1_indices + tier2_indices + tier3_indices)
                    ab_terms = [ab_terms[i] for i in idx]
                    ab_maps = ab_maps[idx]
                    t3_label = "residual_variance" if tier3_method == "residual_variance" else "medoids"
                    t1_label = "receptor (residual-variance)" if tier1_indices and tier3_method == "residual_variance" and pca_ab is not None else "receptor"
                    print(f"Abagen: selected {len(idx)} gene terms (from {n_ab_total}): Tier1 {t1_label}={len(tier1_indices)}, Tier2 cluster medoids={len(tier2_indices)}, Tier3 {t3_label}={len(tier3_indices)}")
                # Re-label: enrich plain "Gene: SYMBOL gene expression from Allen..." with symbol (name), context
                gene_info = {}
                receptor_enriched = {}
                if not getattr(args, "no_abagen_enrich_labels", False):
                    gene_info_path = getattr(args, "abagen_gene_info", None) or ""
                    gi_path = gene_info_path if os.path.isabs(gene_info_path) else os.path.join(repo_root, gene_info_path) if gene_info_path else ""
                    if gi_path and os.path.exists(gi_path):
                        import json
                        with open(gi_path, "r", encoding="utf-8") as f:
                            gene_info = json.load(f)
                        print(f"Loaded gene_info from {gi_path} ({len(gene_info)} symbols)")
                    try:
                        from neurolab.receptor_kb import get_enriched_gene_labels
                        receptor_enriched = get_enriched_gene_labels()
                    except Exception:
                        pass
                if gene_info or receptor_enriched:
                    enriched = 0
                    for i, label in enumerate(ab_terms):
                        if "gene expression from Allen Human Brain Atlas" in (label or ""):
                            sym = _gene_symbol_from_abagen_label(label)
                            if sym:
                                new_label = receptor_enriched.get(sym) or receptor_enriched.get(sym.upper())
                                if new_label:
                                    new_label = f"{new_label}, gene expression"
                                elif gene_info:
                                    entry = gene_info.get(sym) or gene_info.get(sym.upper())
                                    if entry:
                                        name = entry.get("name", entry) if isinstance(entry, dict) else entry
                                        locus = entry.get("locus_group", "") if isinstance(entry, dict) else ""
                                        locus_map = {"protein-coding gene": "protein-coding", "gene with protein product": "protein-coding"}
                                        locus_short = locus_map.get(locus, locus) if locus else ""
                                        locus_str = f"{locus_short} " if locus_short else ""
                                        new_label = f"{sym} ({name}), {locus_str}gene expression"
                                if new_label:
                                    ab_terms[i] = new_label
                                    enriched += 1
                    if enriched:
                        print(f"Abagen: enriched {enriched} labels (receptor KB + gene_info, extensive semantic)")
                # Denoise: project onto PCs explaining abagen_pca_variance (e.g. 0.95)
                abagen_pca_var = getattr(args, "abagen_pca_variance", 0.0) or 0.0
                if abagen_pca_var > 0 and ab_maps.shape[0] > 1:
                    from sklearn.decomposition import PCA
                    n_max = min(ab_maps.shape[0] - 1, ab_maps.shape[1])
                    pca_fit = PCA(n_components=n_max, random_state=42)
                    pca_fit.fit(ab_maps)
                    cumvar = np.cumsum(pca_fit.explained_variance_ratio_)
                    k = int(np.searchsorted(cumvar, abagen_pca_var)) + 1
                    k = min(max(1, k), len(cumvar))
                    pca_denoise = PCA(n_components=k, random_state=42)
                    pca_denoise.fit(ab_maps)
                    ab_maps = pca_denoise.inverse_transform(pca_denoise.transform(ab_maps)).astype(np.float64)
                    print(f"Abagen: denoised {ab_maps.shape[0]} maps onto {k} PCs ({100 * cumvar[k - 1]:.1f}% variance)")
                added_ab = 0
                for label, row in zip(ab_terms, ab_maps):
                    if _is_gradient_pc_term(label):
                        continue  # skip short gradient PCs - we add expanded ones above
                    prefixed = _add_map_type_prefix(label, "abagen")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        new_maps.append(row.astype(np.float64))
                        if term_sources is not None:
                            term_sources.append("abagen")
                            term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["abagen"])
                        added_ab += 1
                print(f"Added {added_ab} abagen gene expression maps")
            else:
                print("abagen cache shape mismatch; skipping.", file=sys.stderr)
        else:
            print("abagen cache not found at", ab_dir, "; skipping.", file=sys.stderr)

    # Merge additional gene cache (e.g. receptor residual-selected denoised)
    if getattr(args, "additional_abagen_cache_dir", None):
        add_dir = args.additional_abagen_cache_dir if os.path.isabs(args.additional_abagen_cache_dir) else os.path.join(repo_root, args.additional_abagen_cache_dir)
        add_npz = os.path.join(add_dir, "term_maps.npz")
        add_pkl = os.path.join(add_dir, "term_vocab.pkl")
        if os.path.exists(add_npz) and os.path.exists(add_pkl):
            add_data = np.load(add_npz)
            add_maps = np.asarray(add_data["term_maps"])
            if truncate_392 and add_maps.shape[1] > 392:
                add_maps = add_maps[:, :392].astype(np.float64)
            with open(add_pkl, "rb") as f:
                add_terms = pickle.load(f)
            if add_maps.shape[1] == n_parcels and len(add_terms) == add_maps.shape[0]:
                added_add = 0
                _gene_info_add = {}
                _receptor_enriched_add = {}
                if not getattr(args, "no_abagen_enrich_labels", False):
                    gi_path = getattr(args, "abagen_gene_info", None) or ""
                    gi_path = gi_path if os.path.isabs(gi_path) else os.path.join(repo_root, gi_path) if gi_path else ""
                    if gi_path and os.path.exists(gi_path):
                        import json
                        with open(gi_path, "r", encoding="utf-8") as f:
                            _gene_info_add = json.load(f)
                    try:
                        from neurolab.receptor_kb import get_enriched_gene_labels
                        _receptor_enriched_add = get_enriched_gene_labels()
                    except Exception:
                        pass
                for label, row in zip(add_terms, add_maps):
                    if _is_gradient_pc_term(label):
                        continue  # skip short gradient PCs - we add expanded ones from main abagen
                    if "gene expression from Allen Human Brain Atlas" in (label or "") and (_gene_info_add or _receptor_enriched_add):
                        sym = _gene_symbol_from_abagen_label(label)
                        if sym:
                            new_label = _receptor_enriched_add.get(sym) or _receptor_enriched_add.get(sym.upper())
                            if new_label:
                                label = f"{new_label}, gene expression"
                            else:
                                entry = _gene_info_add.get(sym) or _gene_info_add.get(sym.upper())
                                if entry:
                                    name = entry.get("name", entry) if isinstance(entry, dict) else entry
                                    locus = entry.get("locus_group", "") if isinstance(entry, dict) else ""
                                    locus_map = {"protein-coding gene": "protein-coding", "gene with protein product": "protein-coding"}
                                    locus_short = locus_map.get(locus, locus) if locus else ""
                                    locus_str = f"{locus_short} " if locus_short else ""
                                    label = f"{sym} ({name}), {locus_str}gene expression"
                    prefixed = _add_map_type_prefix(label, "abagen")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        new_maps.append(row.astype(np.float64))
                        if term_sources is not None:
                            term_sources.append("abagen")
                            term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["abagen"])
                        added_add += 1
                print(f"Added {added_add} additional gene maps from {add_dir}")
            else:
                print(f"Additional abagen cache shape mismatch (expected {n_parcels} parcels); skipping.", file=sys.stderr)
        else:
            print("Additional abagen cache not found at", add_dir, "; skipping.", file=sys.stderr)

    # Merge receptor reference cache (250 genes + PCs, rich labels; Strategy A)
    if getattr(args, "receptor_reference_cache_dir", None):
        ref_dir = args.receptor_reference_cache_dir if os.path.isabs(args.receptor_reference_cache_dir) else os.path.join(repo_root, args.receptor_reference_cache_dir)
        ref_npz = os.path.join(ref_dir, "term_maps.npz")
        ref_pkl = os.path.join(ref_dir, "term_vocab.pkl")
        if os.path.exists(ref_npz) and os.path.exists(ref_pkl):
            ref_data = np.load(ref_npz)
            ref_maps = np.asarray(ref_data["term_maps"])
            if truncate_392 and ref_maps.shape[1] > 392:
                ref_maps = ref_maps[:, :392].astype(np.float64)
            with open(ref_pkl, "rb") as f:
                ref_terms = pickle.load(f)
            if ref_maps.shape[1] == n_parcels and len(ref_terms) == ref_maps.shape[0]:
                added_ref = 0
                for label, row in zip(ref_terms, ref_maps):
                    prefixed = _add_map_type_prefix(label, "reference")
                    norm = _normalize_term(prefixed)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label)):
                        seen.add(norm)
                        new_terms.append(prefixed)
                        new_maps.append(row.astype(np.float64))
                        if term_sources is not None:
                            term_sources.append("reference")
                            term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE["reference"])
                        added_ref += 1
                print(f"Added {added_ref} receptor reference maps (genes + PCs, rich labels)")
            else:
                print("Receptor reference cache shape mismatch; skipping.", file=sys.stderr)
        else:
            print("Receptor reference cache not found at", ref_dir, "; skipping.", file=sys.stderr)

    # PET/receptor residual maps: gradient-regressed versions as extra training targets (pharmacologically specific)
    if getattr(args, "add_pet_residuals", False) and abagen_gradient_components_to_save is not None and term_sources is not None:
        from neurolab.parcellation import zscore_cortex_subcortex_separately
        G = abagen_gradient_components_to_save.astype(np.float64)
        if G.ndim == 1:
            G = G.reshape(1, -1)
        if G.shape[1] == n_parcels:
            GGT = G @ G.T
            try:
                added_res = 0
                for i in range(len(new_terms)):
                    if term_sources[i] not in ("neuromaps", "receptor"):
                        continue
                    x = np.asarray(new_maps[i], dtype=np.float64).ravel()
                    if x.shape[0] != n_parcels:
                        continue
                    coef = np.linalg.solve(GGT, G @ x)
                    residual = (x - G.T @ coef).astype(np.float64)
                    # Re-z-score cortex/subcortex separately so residual scale matches other targets (avoids artificially low MSE)
                    residual = zscore_cortex_subcortex_separately(residual).astype(np.float64)
                    label_res = f"{new_terms[i]}_residual"
                    norm = _normalize_term(label_res)
                    if norm and norm not in seen and (not skip_poor_terms or not _is_poor_term(label_res)):
                        seen.add(norm)
                        new_terms.append(label_res)
                        new_maps.append(residual)
                        term_sources.append(term_sources[i] + "_residual")
                        term_sample_weights.append(SAMPLE_WEIGHT_BY_SOURCE.get(term_sources[i] + "_residual", 0.6))
                        added_res += 1
                if added_res:
                    print(f"Added {added_res} PET/receptor residual maps (gradient-regressed for specificity)")
            except np.linalg.LinAlgError:
                print("PET residuals: singular gradient matrix; skipping.", file=sys.stderr)

    new_maps = np.stack(new_maps, axis=0).astype(np.float64)
    assert new_maps.shape[0] == len(new_terms)
    assert new_maps.shape[1] == n_parcels
    if term_sources is not None:
        assert len(term_sources) == len(new_terms), "term_sources length mismatch"
    if term_sample_weights is not None:
        assert len(term_sample_weights) == len(new_terms), "term_sample_weights length mismatch"

    # Drop all-zero and near-zero maps (blocking: NaN correlations and undefined gradients)
    zero_mask = (np.abs(new_maps).sum(axis=1) == 0)
    peak = np.nanmax(np.abs(new_maps), axis=1)
    near_zero_mask = (peak < 0.01)  # FLAT: peak < 0.01
    drop_mask = zero_mask | near_zero_mask
    if drop_mask.any():
        keep = ~drop_mask
        dropped = [new_terms[i] for i in range(len(new_terms)) if drop_mask[i]]
        n_zero, n_near = int(zero_mask.sum()), int((near_zero_mask & ~zero_mask).sum())
        new_maps = new_maps[keep]
        new_terms = [t for t, k in zip(new_terms, keep) if k]
        if term_sources is not None:
            term_sources = [s for s, k in zip(term_sources, keep) if k]
        if term_sample_weights is not None:
            term_sample_weights = [w for w, k in zip(term_sample_weights, keep) if k]
        print(f"Dropped {drop_mask.sum()} map(s): {n_zero} all-zero, {n_near} near-zero (peak<0.01) — {dropped[:5]}{'...' if len(dropped) > 5 else ''}")

    # No re-normalization: each source is already normalized by its builder.
    # fMRI (decoder, neurosynth, pharma_neurosynth, neurovault, ontology): global z-score — cross-compartment pattern is informative.
    # Gene/receptor/structural (abagen, receptor_reference, neuromaps, enigma): cortex/subcortex separate — different scales by tissue.
    # Re-normalizing fMRI maps with cortex/subcortex separate would destroy cross-compartment signal (e.g. striatum vs cortex).
    if getattr(args, "zscore_renormalize", False):
        from neurolab.parcellation import zscore_cortex_subcortex_separately
        for i in range(new_maps.shape[0]):
            new_maps[i] = zscore_cortex_subcortex_separately(new_maps[i])
        print("Applied z-score re-normalization (cortex/subcortex separately) — deprecated; may distort fMRI cross-compartment patterns.")

    out_npz = os.path.join(output_dir, "term_maps.npz")
    out_pkl = os.path.join(output_dir, "term_vocab.pkl")
    # Same key as build_term_maps_cache so train_text_to_brain_embedding loads with data["term_maps"]
    np.savez_compressed(out_npz, term_maps=new_maps)
    with open(out_pkl, "wb") as f:
        pickle.dump(new_terms, f)
    if term_sources is not None:
        with open(os.path.join(output_dir, "term_sources.pkl"), "wb") as f:
            pickle.dump(term_sources, f)
        print("Saved term_sources.pkl for sample weighting (direct|ontology|neuromaps|neurovault|receptor|enigma|abagen)")
        if term_sample_weights is not None:
            with open(os.path.join(output_dir, "term_sample_weights.pkl"), "wb") as f:
                pickle.dump(term_sample_weights, f)
            print("Saved term_sample_weights.pkl for per-term loss weights (overrides source defaults when present)")
        SOURCE_TO_MAP_TYPE = {"direct": "fmri_activation", "neurovault": "fmri_activation", "ontology": "fmri_activation", "neurovault_pharma": "fmri_activation", "pharma_neurosynth": "fmri_activation", "neuromaps": "pet_receptor", "receptor": "pet_receptor", "neuromaps_residual": "pet_receptor", "receptor_residual": "pet_receptor", "structural": "structural", "enigma": "structural", "abagen": "pet_receptor", "reference": "pet_receptor"}
        term_map_types = [SOURCE_TO_MAP_TYPE.get(s, "fmri_activation") for s in term_sources]
        with open(os.path.join(output_dir, "term_map_types.pkl"), "wb") as f:
            pickle.dump(term_map_types, f)
        print("Saved term_map_types.pkl for type-conditioned MLP (fmri_activation|structural|pet_receptor)")
    if abagen_gradient_components_to_save is not None:
        np.save(os.path.join(output_dir, "abagen_gradient_components.npy"), abagen_gradient_components_to_save)
        print("Saved abagen_gradient_components.npy for PET residual-correlation evaluation")

    # Optional: PCA on abagen maps only; store loadings so trainer can use a gene head (predict PC loadings, inverse to 392-D at inference)
    gene_pca_variance = getattr(args, "gene_pca_variance", 0.0)
    if gene_pca_variance > 0 and term_sources is not None:
        abagen_indices = [i for i in range(len(new_terms)) if term_sources[i] == "abagen"]
        if len(abagen_indices) >= 2:
            from sklearn.decomposition import PCA
            gene_maps = new_maps[abagen_indices]
            n_abagen = gene_maps.shape[0]
            max_k = min(n_abagen - 1, n_parcels)
            if max_k >= 1:
                pca_full = PCA(n_components=max_k, random_state=42)
                pca_full.fit(gene_maps)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                k = int(np.searchsorted(cumvar, gene_pca_variance)) + 1
                k = min(max(1, k), max_k)
                gene_pca = PCA(n_components=k, random_state=42)
                gene_pca.fit(gene_maps)
                loadings = gene_pca.transform(gene_maps).astype(np.float32)
                with open(os.path.join(output_dir, "gene_pca.pkl"), "wb") as f:
                    pickle.dump(gene_pca, f)
                np.savez_compressed(os.path.join(output_dir, "gene_loadings.npz"), loadings=loadings)
                with open(os.path.join(output_dir, "abagen_term_indices.pkl"), "wb") as f:
                    pickle.dump(abagen_indices, f)
                var_expl = gene_pca.explained_variance_ratio_.sum()
                print(f"Gene PCA (variance {gene_pca_variance:.2f}): {k} components, {len(abagen_indices)} abagen terms (explained variance: {var_expl:.3f})")
                print("Saved gene_pca.pkl, gene_loadings.npz, abagen_term_indices.pkl for gene-head training")
            else:
                print("Too few abagen samples for PCA; skipping gene PCA.", file=sys.stderr)
        else:
            print("No abagen terms for gene PCA; skipping.", file=sys.stderr)

    print(f"Saved {len(new_terms)} terms x {n_parcels} parcels -> {output_dir}")
    print("Train with: python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir", output_dir, "...")


if __name__ == "__main__":
    main()
