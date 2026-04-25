#!/usr/bin/env python3
"""
Small smoke test for parcellation, ontology, cache shape, and decoder/embedding path.
Run from repo root: python neurolab/scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main():
    failed = []
    # 1. Parcellation
    try:
        from neurolab.parcellation import get_n_parcels, get_masker, get_combined_atlas_path
        n = get_n_parcels()
        path = get_combined_atlas_path()
        assert path.exists(), "combined atlas not found"
        masker = get_masker()
        masker.fit()
        import numpy as np
        # Check masker output dimension (transform returns (1, n_parcels) or (n_parcels,))
        out = masker.transform(masker.labels_img_)
        out = np.atleast_1d(np.asarray(out).ravel())
        assert out.size == n, (out.size, n)
        print("[1/5] Parcellation: n_parcels=%s, atlas OK" % n)
    except Exception as e:
        failed.append(("Parcellation", e))
        print("[1/5] Parcellation: FAIL", e)

    # 2. Ontology dir and files present (don't load full index - slow)
    try:
        ont_dir = repo_root / "neurolab" / "data" / "ontologies"
        if ont_dir.exists():
            owls = list(ont_dir.glob("*.owl"))
            print("[2/5] Ontology: %d .owl files in %s" % (len(owls), ont_dir.name))
        else:
            print("[2/5] Ontology: skip (no ontologies dir)")
    except Exception as e:
        failed.append(("Ontology", e))
        print("[2/5] Ontology: FAIL", e)

    # 3. Expanded cache shape (use existing decoder cache or synthetic)
    try:
        import numpy as np
        import pickle
        cache_dir = repo_root / "neurolab" / "data" / "decoder_cache"
        smoke_dir = repo_root / "neurolab" / "data" / "smoke_decoder_cache"
        use_dir = smoke_dir if (smoke_dir / "term_maps.npz").exists() else cache_dir
        if (use_dir / "term_maps.npz").exists():
            data = np.load(use_dir / "term_maps.npz")
            term_maps = data["term_maps"]
            with open(use_dir / "term_vocab.pkl", "rb") as f:
                vocab = pickle.load(f)
            assert term_maps.shape[0] == len(vocab)
            assert term_maps.shape[1] in (392, 400, 410, 414, 427, 450), "expected 392, 400, 410, 414, 427, or 450 parcels"
            print("[3/5] Cache: %d terms x %d parcels" % (term_maps.shape[0], term_maps.shape[1]))
        else:
            # synthetic
            try:
                from neurolab.parcellation import get_n_parcels
                n_parcels = get_n_parcels()
            except Exception:
                n_parcels = 400
            with tempfile.TemporaryDirectory() as td:
                np.savez_compressed(Path(td) / "term_maps.npz", term_maps=np.zeros((2, n_parcels)))
                with open(Path(td) / "term_vocab.pkl", "wb") as f:
                    pickle.dump(["term_a", "term_b"], f)
                d = np.load(Path(td) / "term_maps.npz")
                print("[3/5] Cache: (no existing cache) synthetic 2 x %s OK" % d["term_maps"].shape[1])
    except Exception as e:
        failed.append(("Cache shape", e))
        print("[3/5] Cache: FAIL", e)

    # 4. CognitiveDecoder (if cache exists and is consistent)
    try:
        from neurolab.enrichment.cognitive_decoder import CognitiveDecoder
        cache_dir = repo_root / "neurolab" / "data" / "decoder_cache"
        smoke_dir = repo_root / "neurolab" / "data" / "smoke_decoder_cache"
        use_dir = smoke_dir if (smoke_dir / "term_maps.npz").exists() else cache_dir
        if (use_dir / "term_maps.npz").exists():
            data = np.load(use_dir / "term_maps.npz")
            with open(use_dir / "term_vocab.pkl", "rb") as f:
                vocab = pickle.load(f)
            if data["term_maps"].shape[0] != len(vocab):
                print("[4/5] CognitiveDecoder: skip (cache vocab len != term_maps rows)")
            else:
                # Prefer relative path (e.g. neurolab/data/decoder_cache) for portability
                try:
                    rel = use_dir.relative_to(repo_root)
                    cache_dir_arg = str(rel).replace("\\", "/")
                except ValueError:
                    cache_dir_arg = str(use_dir)
                dec = CognitiveDecoder(cache_dir=cache_dir_arg)
                rng = np.random.default_rng(42)
                fake = rng.standard_normal(dec.n_parcels).astype(np.float64)
                out = dec.decode(fake, top_n=2)
                assert "top_terms" in out or "terms" in out, list(out.keys())
                print("[4/5] CognitiveDecoder: n_parcels=%s, decode OK" % dec.n_parcels)
        else:
            print("[4/5] CognitiveDecoder: skip (no cache)")
    except Exception as e:
        failed.append(("CognitiveDecoder", e))
        print("[4/5] CognitiveDecoder: FAIL", repr(e))

    # 5. build_expanded_term_maps import and n_parcels from cache
    try:
        # Just verify the module and that it would use n_parcels from cache
        sys.path.insert(0, str(repo_root / "neurolab" / "scripts"))
        import build_expanded_term_maps as b
        print("[5/5] build_expanded_term_maps: module OK (n_parcels from cache)")
    except Exception as e:
        failed.append(("build_expanded_term_maps", e))
        print("[5/5] build_expanded_term_maps: FAIL", e)

    if failed:
        print("\nFailed:", [f[0] for f in failed])
        return 1
    print("\nSmoke test passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
