#!/usr/bin/env python3
"""
Run a single query: text term -> parcellated map -> enrichment (cognitive + receptor + summary).

Query-ready entry point. Local only (local decoder cache, no API).
Run from querytobrain root:
  python neurolab/scripts/query.py "attention"
  python neurolab/scripts/query.py "noradrenergic modulation" --use-embedding-model neurolab/data/embedding_model
"""
import argparse
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)
_env = os.path.join(repo_root, ".env")
if os.path.isfile(_env):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

def _get_n_parcels():
    from neurolab.parcellation import get_n_parcels
    return get_n_parcels()


def get_parcellated_map_from_embedding(
    term: str,
    embedding_model_dir: str,
    cache_dir: str | None = None,
    with_retrieval: bool = False,
):
    """
    Use trained text->brain embedding (expandable term space). Local only.
    If with_retrieval=True, returns (map, retrieval_result) else returns map only.
    """
    from neurolab.enrichment.text_to_brain import TextToBrainEmbedding
    emb = TextToBrainEmbedding(embedding_model_dir, cache_dir=cache_dir)
    if with_retrieval:
        return emb.predict_map_with_retrieval(term, top_k=10)
    return emb.embed(term)


def get_parcellated_map_for_term(term: str):
    """NeuroQuery text -> pipeline atlas (Glasser+Tian 392) parcellated vector."""
    from neuroquery import fetch_neuroquery_model, NeuroQueryModel
    import numpy as np
    import nibabel as nib

    model_path = fetch_neuroquery_model()
    model = NeuroQueryModel.from_data_dir(model_path)
    result = model.transform([term])
    brain_map = result["brain_map"][0]

    if hasattr(brain_map, "get_fdata"):
        brain_img = brain_map
    elif isinstance(brain_map, (str, os.PathLike)):
        brain_img = nib.load(brain_map)
    else:
        brain_img = nib.Nifti1Image(np.asarray(brain_map), np.eye(4))

    from neurolab.parcellation import get_masker, resample_to_atlas
    brain_img = resample_to_atlas(brain_img)
    masker = get_masker(memory="nilearn_cache", verbose=0)
    masker.fit()
    parcellated = masker.transform(brain_img).ravel()
    return parcellated.astype(np.float64)


def _default_cognitive_cache():
    data_dir = os.path.join(repo_root, "neurolab", "data")
    unified = os.path.join(data_dir, "unified_cache")
    decoder = os.path.join(data_dir, "decoder_cache")
    if os.path.exists(os.path.join(unified, "term_maps.npz")):
        return unified
    return decoder


def _default_neuromaps_cache():
    nm = os.path.join(repo_root, "neurolab", "data", "neuromaps_cache")
    if os.path.exists(os.path.join(nm, "annotation_maps.npz")):
        return nm
    return None


def _default_ontology_dir():
    ont = os.path.join(repo_root, "neurolab", "data", "ontologies")
    if os.path.isdir(ont) and any(
        f.endswith((".obo", ".owl", ".rdf", ".ttl"))
        for f in os.listdir(ont)
    ):
        return ont
    return None


def get_parcellated_map_from_cache_and_ontology(
    term: str,
    cache_dir: str,
    ontology_dir: str | None,
    similarity_threshold: float,
    similarity_threshold_ontology: float | None,
    use_ontology: bool,
) -> tuple["np.ndarray | None", bool]:
    """
    Get (n_parcels,) map from cache-term weights; if max similarity < threshold and use_ontology,
    use ontology to expand to related cache terms (only if similarity to those terms >= threshold B).
    Returns (map, used_ontology).
    """
    import numpy as np
    import pickle

    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        return None, False

    data = np.load(npz_path)
    key = "term_maps" if "term_maps" in data else data.files[0]
    term_maps = np.asarray(data[key], dtype=np.float64)
    with open(pkl_path, "rb") as f:
        term_vocab = pickle.load(f)
    if isinstance(term_vocab, dict):
        term_vocab = list(term_vocab.keys()) if term_vocab else []
    term_vocab = list(term_vocab)
    if term_maps.shape[0] != len(term_vocab):
        return None, False

    ontology_index = None
    if use_ontology and ontology_dir and os.path.isdir(ontology_dir):
        scripts_dir = os.path.join(repo_root, "neurolab", "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        try:
            from ontology_expansion import load_ontology_index
            ontology_index = load_ontology_index(ontology_dir)
            if not (ontology_index.get("label_to_related")):
                ontology_index = None
        except Exception:
            ontology_index = None

    scripts_dir = os.path.join(repo_root, "neurolab", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from query_to_map import get_map_with_ontology_on_low_similarity
    except ImportError:
        return None, False

    map_400, used_ontology = get_map_with_ontology_on_low_similarity(
        term,
        term_maps,
        term_vocab,
        ontology_index=ontology_index,
        similarity_threshold=similarity_threshold,
        similarity_threshold_ontology=similarity_threshold_ontology,
        encoder=None,
        cache_embeddings=None,
        use_tfidf_fallback=True,
    )
    return map_400, used_ontology


def main():
    parser = argparse.ArgumentParser(description="Query: term -> enrichment (local decoder cache)")
    parser.add_argument("term", help="Query term (e.g. attention, serotonin, memory)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cognitive decoder cache (default: unified_cache if present, else decoder_cache)",
    )
    parser.add_argument(
        "--neuromaps-cache-dir",
        default=None,
        help="Neuromaps cache for biological enrichment (default: neuromaps_cache if present)",
    )
    parser.add_argument(
        "--use-embedding-model",
        default=None,
        metavar="DIR",
        help="Use trained text->brain embedding (expandable term space). Skips NeuroQuery; local only.",
    )
    parser.add_argument(
        "--guardrail",
        choices=["on", "off", "warn"],
        default="on",
        help="Scope guardrail when using embedding model: on=block out-of-scope, warn=warn and continue, off=skip (default: on).",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Top N cognitive terms (default 15)")
    parser.add_argument(
        "--ontology-dir",
        default=None,
        help="Ontology dir for low-similarity fallback (default: neurolab/data/ontologies if present)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.15,
        help="Use ontology when max similarity to cache terms < this (default: 0.15)",
    )
    parser.add_argument(
        "--similarity-threshold-ontology",
        type=float,
        default=None,
        metavar="B",
        help="Use ontology weights only if similarity to ontology-derived terms >= B (default: same as --similarity-threshold)",
    )
    parser.add_argument(
        "--no-ontology",
        action="store_true",
        help="Disable ontology fallback for low similarity; use cache/NeuroQuery only",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        default=True,
        help="Use combined term→map: try neuromaps by name first, then cognitive (default: on).",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Disable combined term→map; use cognitive path only (no neuromaps-by-name).",
    )
    args = parser.parse_args()
    args.combined = args.combined and not args.no_combined

    cache_dir = args.cache_dir if args.cache_dir else _default_cognitive_cache()
    cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(repo_root, cache_dir)
    neuromaps_dir = args.neuromaps_cache_dir if args.neuromaps_cache_dir else _default_neuromaps_cache()
    if neuromaps_dir and not os.path.isabs(neuromaps_dir):
        neuromaps_dir = os.path.join(repo_root, neuromaps_dir)
    embedding_model_dir = None
    if args.use_embedding_model:
        embedding_model_dir = args.use_embedding_model if os.path.isabs(args.use_embedding_model) else os.path.join(repo_root, args.use_embedding_model)

    if not os.path.exists(os.path.join(cache_dir, "term_maps.npz")):
        print("Local decoder cache missing. Build the full system once:", file=sys.stderr)
        print("  python neurolab/scripts/build_all_maps.py --quick", file=sys.stderr)
        print("  (or build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 100 for NQ-only)", file=sys.stderr)
        sys.exit(1)

    from neurolab.enrichment.unified_enrichment import UnifiedEnrichment

    term = args.term.strip()
    print(f"Query: \"{term}\"")
    if embedding_model_dir:
        print("  Using text->brain embedding (expandable term space)...")
        if args.guardrail != "off":
            try:
                from neurolab.enrichment.scope_guard import ScopeGuard
                guard = ScopeGuard(embedding_model_dir, cache_dir=cache_dir)
                check = guard.check(term)
                if not check["in_scope"]:
                    print(f"  Guardrail: {check['message']}", file=sys.stderr)
                    if args.guardrail == "on":
                        print("  Skipping map prediction (use --guardrail=warn to run anyway, or --guardrail=off to disable).", file=sys.stderr)
                        sys.exit(2)
                    else:
                        print("  Proceeding anyway (--guardrail=warn).", file=sys.stderr)
                else:
                    print(f"  Guardrail: {check['message']}")
            except FileNotFoundError as e:
                if args.guardrail == "on":
                    print(f"  Guardrail unavailable: {e}", file=sys.stderr)
                    sys.exit(1)
                # warn/off: continue without guardrail
        try:
            parcellated, retrieval_result = get_parcellated_map_from_embedding(
                term, embedding_model_dir, cache_dir=cache_dir, with_retrieval=True
            )
            parcellated = retrieval_result["map"]
            retrieval = retrieval_result.get("retrieval", [])
            if retrieval:
                print(f"\n--- Top evidence (retrieval-first) ---")
                print(f"  Confidence: {retrieval_result.get('confidence', 0):.3f}")
                for i, r in enumerate(retrieval[:10], 1):
                    src = f" [{r['source']}]" if r.get("source") else ""
                    print(f"  {i}. {r['term'][:60]}{'...' if len(r['term'])>60 else ''}{src} sim={r['similarity']:.3f}")
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Combined term→map: neuromaps by name first, then cognitive (cache + ontology)
        ontology_dir = args.ontology_dir if args.ontology_dir else _default_ontology_dir()
        if args.combined and neuromaps_dir:
            print("  Fetching map (combined: neuromaps by name + cognitive)...")
            try:
                scripts_dir = os.path.join(repo_root, "neurolab", "scripts")
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from term_to_map import get_parcellated_map_combined
                n_parcels = _get_n_parcels()
                parcellated, map_source = get_parcellated_map_combined(
                    term,
                    cache_dir=cache_dir,
                    neuromaps_cache_dir=neuromaps_dir,
                    ontology_dir=ontology_dir,
                    similarity_threshold=args.similarity_threshold,
                    similarity_threshold_ontology=args.similarity_threshold_ontology,
                    use_ontology=not args.no_ontology,
                    prefer_neuromaps_if_matched=True,
                    n_parcels=n_parcels,
                    _get_cognitive=get_parcellated_map_from_cache_and_ontology,
                    _get_neuroquery=get_parcellated_map_for_term,
                )
                if map_source == "neuromaps":
                    print("  Map from neuromaps (biological term match).")
                elif map_source == "ontology":
                    print("  Used ontology fallback (low similarity to cache terms).")
            except Exception as e:
                print(f"  Combined path failed ({e}), falling back to cognitive-only.", file=sys.stderr)
                parcellated = None
                map_source = "none"
        else:
            map_source = None
            parcellated = None
            if not args.no_ontology and ontology_dir:
                print("  Fetching map (cache + ontology on low similarity)...")
            else:
                print("  Fetching map (cache weights over terms)...")
        if parcellated is None or map_source == "none":
            used_ontology = False
            try:
                parcellated, used_ontology = get_parcellated_map_from_cache_and_ontology(
                    term,
                    cache_dir,
                    ontology_dir,
                    args.similarity_threshold,
                    args.similarity_threshold_ontology,
                    use_ontology=not args.no_ontology,
                )
            except Exception as e:
                print(f"  Cache path failed ({e}), falling back to NeuroQuery.", file=sys.stderr)
                parcellated = None
            if parcellated is None:
                try:
                    parcellated = get_parcellated_map_for_term(term)
                except Exception as e:
                    print(f"  Error: {e}", file=sys.stderr)
                    sys.exit(1)
            elif used_ontology:
                print("  Used ontology fallback (low similarity to cache terms).")

    n_parcels = _get_n_parcels()
    unified = UnifiedEnrichment(
        cache_dir=cache_dir,
        receptor_path=None,
        neuromaps_cache_dir=neuromaps_dir,
        enable_cognitive=True,
        enable_biological=True,
        n_parcels=n_parcels,
    )
    result = unified.enrich(parcellated, cognitive_top_n=args.top_n)

    print("\n--- Summary ---")
    print(result["summary"])
    if "cognitive" in result:
        top = result["cognitive"].get("top_terms", [])[:10]
        print("\n--- Top cognitive terms ---")
        for t, r in top:
            print(f"  {t}: r={r:.3f}")
    if "biological" in result and result["biological"].get("top_hits"):
        print("\n--- Top biological hits ---")
        for r in result["biological"]["top_hits"][:10]:
            extra = f" ({r['system']})" if r.get("system") else ""
            print(f"  {r['name']}{extra}: r={r['r']:.3f}")


if __name__ == "__main__":
    main()
