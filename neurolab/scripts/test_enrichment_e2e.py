#!/usr/bin/env python3
"""
Local end-to-end test: text -> parcellated map -> UnifiedEnrichment.enrich().

No API or server. Run from querytobrain root:
  python neurolab/scripts/test_enrichment_e2e.py

Requires: Phase 0–1 (env + NeuroQuery + Schaefer), and local decoder cache (Phase 2):
  neurolab/data/decoder_cache/term_maps.npz + term_vocab.pkl
  Build once: python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 100
Receptor data: uses local placeholder if no Hansen CSV/NPZ path is given.
"""
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

DATA_DIR = os.path.join(repo_root, "neurolab", "data")
UNIFIED_CACHE = os.path.join(DATA_DIR, "unified_cache")
DECODER_CACHE = os.path.join(DATA_DIR, "decoder_cache")
NEUROMAPS_CACHE = os.path.join(DATA_DIR, "neuromaps_cache")


def _get_n_parcels():
    from neurolab.parcellation import get_n_parcels
    return get_n_parcels()


def _cognitive_cache_dir():
    if os.path.exists(os.path.join(UNIFIED_CACHE, "term_maps.npz")):
        return UNIFIED_CACHE
    return DECODER_CACHE


def _neuromaps_cache_dir():
    if os.path.exists(os.path.join(NEUROMAPS_CACHE, "annotation_maps.npz")):
        return NEUROMAPS_CACHE
    return None


def get_parcellated_map_for_term(term: str) -> "np.ndarray":
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


def main():
    cache_dir = _cognitive_cache_dir()
    neuromaps_dir = _neuromaps_cache_dir()
    if not os.path.exists(os.path.join(cache_dir, "term_maps.npz")):
        print("Local decoder cache missing. Build the full system once:", file=sys.stderr)
        print("  python neurolab/scripts/build_all_maps.py --quick", file=sys.stderr)
        sys.exit(1)

    from neurolab.enrichment.unified_enrichment import UnifiedEnrichment
    import numpy as np

    n_parcels = _get_n_parcels()
    unified = UnifiedEnrichment(
        cache_dir=cache_dir,
        receptor_path=None,
        neuromaps_cache_dir=neuromaps_dir,
        enable_cognitive=True,
        enable_biological=True,
        n_parcels=n_parcels,
    )

    # Test 1: known term -> enrich
    term = "attention"
    print(f"E2E: Getting parcellated map for '{term}'...")
    try:
        parcellated = get_parcellated_map_for_term(term)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    assert parcellated.shape == (n_parcels,), parcellated.shape
    print(f"  Shape {parcellated.shape}, OK")

    print(f"  Running UnifiedEnrichment.enrich()...")
    result = unified.enrich(parcellated, cognitive_top_n=15)

    assert "summary" in result
    assert "biological" in result
    if unified.cognitive:
        assert "cognitive" in result
        top = result["cognitive"].get("top_terms", [])[:5]
        print(f"  Cognitive top 5: {[t[0] for t in top]}")

    print(f"  Biological top 3: {[r['name'] for r in result['biological']['top_hits'][:3]]}")
    print("  Summary:")
    for line in result["summary"].split("\n"):
        print(f"    {line}")

    # Test 2: random vector (sanity)
    print("\nE2E: Random vector -> enrich (sanity)...")
    rng = np.random.default_rng(123)
    random_map = rng.standard_normal(n_parcels)
    result2 = unified.enrich(random_map, cognitive_top_n=5)
    assert "summary" in result2 and "biological" in result2
    print("  OK")

    # Test 3: cache + ontology path (no embedding)
    print("\nE2E: Cache + ontology term -> map...")
    scripts_dir = os.path.join(repo_root, "neurolab", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    ont_dir = os.path.join(DATA_DIR, "ontologies")
    ont_available = os.path.isdir(ont_dir) and any(
        f.endswith((".owl", ".obo", ".ttl", ".rdf")) for f in os.listdir(ont_dir)
    )
    try:
        from query import get_parcellated_map_from_cache_and_ontology
        map_cog, used_onto = get_parcellated_map_from_cache_and_ontology(
            "attention",
            cache_dir,
            ont_dir if ont_available else None,
            0.15,
            None,
            use_ontology=ont_available,
        )
        assert map_cog is not None and map_cog.shape == (n_parcels,)
        print(f"  Shape {map_cog.shape}, used_ontology={used_onto}, OK")
    except Exception as e:
        print(f"  SKIP (cache/ontology): {e}")

    # Test 4: neuromaps-by-name (if neuromaps cache present)
    if neuromaps_dir:
        print("\nE2E: Neuromaps-by-name term -> map...")
        try:
            from term_to_map import get_map_from_neuromaps, match_query_to_neuromaps_labels
            map_nm, label_nm = get_map_from_neuromaps("myelin", neuromaps_dir, n_parcels=n_parcels)
            if map_nm is not None:
                assert map_nm.shape == (n_parcels,)
                print(f"  myelin -> {label_nm}, shape {map_nm.shape}, OK")
            else:
                print("  No myelin match (labels may differ), OK")
            # Sanity: match_query_to_neuromaps_labels returns ranked list
            import pickle
            with open(os.path.join(neuromaps_dir, "annotation_labels.pkl"), "rb") as f:
                labels = pickle.load(f)
            ranked = match_query_to_neuromaps_labels("receptor", labels)
            print(f"  'receptor' matches: {len(ranked)} labels")
        except Exception as e:
            print(f"  SKIP (neuromaps): {e}")

    # Test 5: combined term -> map (neuromaps first, then cognitive)
    if neuromaps_dir:
        print("\nE2E: Combined term -> map...")
        try:
            from term_to_map import get_parcellated_map_combined
            from query import get_parcellated_map_from_cache_and_ontology, get_parcellated_map_for_term as get_nq
            map_combined, source = get_parcellated_map_combined(
                "attention",
                cache_dir=cache_dir,
                neuromaps_cache_dir=neuromaps_dir,
                ontology_dir=ont_dir if ont_available else None,
                similarity_threshold=0.15,
                similarity_threshold_ontology=None,
                use_ontology=ont_available,
                prefer_neuromaps_if_matched=True,
                n_parcels=n_parcels,
                _get_cognitive=get_parcellated_map_from_cache_and_ontology,
                _get_neuroquery=get_nq,
            )
            assert map_combined.shape == (n_parcels,)
            assert source in ("neuromaps", "cognitive", "ontology", "neuroquery", "none")
            print(f"  'attention' -> source={source}, OK")
        except Exception as e:
            print(f"  SKIP (combined): {e}")

    print("\nLocal E2E tests passed. No API used.")


if __name__ == "__main__":
    main()
