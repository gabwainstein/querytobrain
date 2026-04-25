#!/usr/bin/env python3
"""
Graph-distance seed analysis: how does brain-map correlation decay with
ontological distance?

**Distance marker:** We use *hierarchy distance* = number of parent_child (is_a)
edges along the shortest path. Synonym edges count as 0 (same concept). So:
  - d_hierarchy 0 = synonym only (own category)
  - d_hierarchy 1 = one child<->parent hop
  - d_hierarchy 2 = two hierarchy steps (e.g. child->parent->grandparent)

This separates "same concept, different label" (synonym) from "broader/narrower
in hierarchy" (parent_child). Raw hop count mixed both; at d=2 most hops were
parent_child, so reporting by hierarchy distance is clearer.

**Better marker / reasoning:** Hops are linked by *properties* (relation type).
Edges currently store relation='synonym'|'parent_child'. Ontologies (OBO/OWL) can
also define relationship: part_of, has_part, located_in, etc. Those properties
carry meaning we could measure:
  - Weighted path length: synonym=0, is_a=1, part_of=0.8, etc.
  - Information-content (IC) or Lin similarity from the graph (common ancestor).
  - Future: --edge-weights or IC-based semantic distance for better correlation.

Output:
  - Table: d_hierarchy (0=synonym, 1, 2, ...) | mean r | n | by path type
  - Optional plot: r vs hierarchy distance; percentiles; path-type stratification
  - Optional JSON: fitted gamma and per-distance stats

Usage (from repo root):
  python neurolab/scripts/graph_distance_correlation.py \
    --cache-dir neurolab/data/decoder_cache \
    --ontology-dir neurolab/data/ontologies

  # With plot and JSON export:
  python neurolab/scripts/graph_distance_correlation.py \
    --cache-dir neurolab/data/decoder_cache \
    --ontology-dir neurolab/data/ontologies \
    --plot neurolab/data/distance_decay.png \
    --output-json neurolab/data/graph_distance_stats.json \
    --max-distance 8

Requires: obonet, rdflib (for ontology loading), numpy, scipy
Optional: matplotlib (for --plot)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

N_PARCELS = 400


# ---------------------------------------------------------------------------
# Graph building: merge all OBO/OWL ontologies into one undirected NetworkX graph
# ---------------------------------------------------------------------------

def _normalize(t: str) -> str:
    return (t or "").strip().lower().replace("_", " ")


def build_unified_graph(ontology_dir: str) -> "nx.Graph":
    """
    Build a single undirected graph from all OBO/OWL files in ontology_dir.

    Nodes = normalized term labels (lowercase, stripped).
    Edges = ontological relationships (parent-child, synonym links).
    Edge weight = 1 for all (unweighted shortest path = hop count).

    We use an undirected graph because we want shortest path regardless of
    direction (parent→child or child→parent are both 1 hop).
    """
    import networkx as nx

    G = nx.Graph()
    ontology_dir = Path(ontology_dir)
    if not ontology_dir.exists():
        print(f"Ontology dir not found: {ontology_dir}", file=sys.stderr)
        return G

    # Try obonet for OBO files
    try:
        import obonet
        HAS_OBONET = True
    except ImportError:
        HAS_OBONET = False

    # Try rdflib for OWL/TTL files
    try:
        import rdflib
        HAS_RDFLIB = True
    except ImportError:
        HAS_RDFLIB = False

    for path in sorted(ontology_dir.glob("*")):
        suffix = path.suffix.lower()

        if suffix in (".obo", ".obo.gz", ".obo.xz") and HAS_OBONET:
            try:
                obo_graph = obonet.read_obo(str(path))
            except Exception as e:
                print(f"  Skip {path.name}: {e}")
                continue

            id_to_name = {}
            id_to_syns = {}

            for nid, data in obo_graph.nodes(data=True):
                raw = data.get("name")
                name = (raw[0] if isinstance(raw, list) and raw else raw) if raw else None
                if isinstance(name, str):
                    id_to_name[nid] = _normalize(name)

                syns = data.get("synonym") or []
                parsed_syns = []
                for s in syns:
                    if '"' in s:
                        parsed_syns.append(_normalize(s.split('"')[1]))
                    else:
                        parsed_syns.append(_normalize(s))
                if parsed_syns:
                    id_to_syns[nid] = parsed_syns

            # Add nodes
            for nid, name in id_to_name.items():
                if name:
                    G.add_node(name)

            # Parent-child edges (is_a in OBO). Edge property relation="parent_child".
            # Future: OBO also has "relationship: part_of", "has_part", etc.; load those
            # and set relation="part_of" etc. for weighted path (synonym=0, is_a=1, part_of=w).
            for nid in obo_graph.nodes():
                name = id_to_name.get(nid)
                if not name:
                    continue
                for parent_id in obo_graph.predecessors(nid):
                    parent_name = id_to_name.get(parent_id)
                    if parent_name and parent_name != name:
                        G.add_edge(name, parent_name, relation="parent_child")

            # Synonym edges: same concept, different label -> hierarchy distance 0
            for nid, syns in id_to_syns.items():
                name = id_to_name.get(nid)
                if not name:
                    continue
                for syn in syns:
                    if syn and syn != name:
                        G.add_edge(name, syn, relation="synonym")

            n_nodes = len([n for n in id_to_name.values() if n])
            print(f"  {path.name}: {n_nodes} concepts loaded into graph")

        elif suffix in (".owl", ".rdf", ".ttl") and HAS_RDFLIB:
            try:
                g = rdflib.Graph()
                if suffix in (".owl", ".rdf"):
                    g.parse(str(path), format="xml")
                elif suffix == ".ttl":
                    g.parse(str(path), format="turtle")
                else:
                    g.parse(str(path))
            except Exception as e:
                print(f"  Skip {path.name}: {e}")
                continue

            RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
            SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

            uri_to_label = {}
            uri_to_altlabels = defaultdict(list)

            for s, p, o in g:
                if p in (RDFS.label, SKOS.prefLabel) and isinstance(o, rdflib.Literal):
                    label = _normalize(str(o))
                    if label:
                        uri_to_label[str(s)] = label
                        G.add_node(label)
                if p == SKOS.altLabel and isinstance(o, rdflib.Literal):
                    alt = _normalize(str(o))
                    if alt:
                        uri_to_altlabels[str(s)].append(alt)

            # subClassOf edges
            for s, p, o in g:
                if p == RDFS.subClassOf and isinstance(o, rdflib.term.Node):
                    child_uri = str(s)
                    parent_uri = str(o)
                    child_label = uri_to_label.get(child_uri)
                    parent_label = uri_to_label.get(parent_uri)
                    if (child_label and parent_label
                            and child_label != parent_label
                            and "owl#" not in parent_uri
                            and not parent_uri.endswith("#Thing")):
                        G.add_edge(child_label, parent_label, relation="parent_child")

            # Synonym/altLabel edges
            for uri, alts in uri_to_altlabels.items():
                label = uri_to_label.get(uri)
                if not label:
                    continue
                for alt in alts:
                    if alt and alt != label:
                        G.add_node(alt)
                        G.add_edge(label, alt, relation="synonym")

            print(f"  {path.name}: {len(uri_to_label)} concepts loaded into graph")

    print(f"\nUnified graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_nxontology_for_ic(G: "nx.Graph") -> "NXOntology | None":
    """
    Build NXOntology from unified G for Lin/Resnik/Jiang-Conrath with intrinsic IC.
    Uses only parent_child edges; direction is superterm -> subterm (root -> leaf).
    Connects disconnected components via virtual root so IC is defined everywhere.
    Returns None if nxontology not installed.
    """
    try:
        from nxontology import NXOntology
        import networkx as nx
    except ImportError:
        return None
    # NXOntology expects DiGraph with edge superterm -> subterm (parent -> child)
    # Our G has (child, parent) for parent_child, so add (parent, child) to DAG
    dag = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data.get("relation") == "parent_child":
            dag.add_edge(v, u)  # v=parent, u=child -> parent -> child
    if dag.number_of_edges() == 0:
        return None
    # Virtual root for disconnected components
    roots = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    virtual_root = "__root__"
    dag.add_node(virtual_root)
    for r in roots:
        dag.add_edge(virtual_root, r)
    nxo = NXOntology(dag)
    nxo.freeze()
    return nxo


# ---------------------------------------------------------------------------
# Seed analysis: for each pair of cache terms in graph, compute (distance, r)
# ---------------------------------------------------------------------------

def compute_distance_correlations(
    G: "nx.Graph",
    term_maps: np.ndarray,
    term_vocab: list[str],
    max_distance: int = 8,
    max_seeds: int = 500,
    max_pairs_per_distance: int = 5000,
    seed: int = 42,
) -> dict:
    """
    For cache terms that appear in the graph, compute brain-map correlation
    at each graph distance.

    Strategy:
      1. Find all cache terms present in the graph
      2. For a sample of seed terms, BFS outward collecting terms at each distance
      3. For each (seed, neighbor_at_distance_d), compute Pearson r of their brain maps
      4. Bucket by d, report stats

    Returns dict with:
      per_distance: {d: {"mean_r": ..., "std_r": ..., "n": ..., "pairs": [...]}}
      random_baseline: {"mean_r": ..., "std_r": ..., "n": ...}
      fitted_gamma: float (best-fit γ for r(d) ≈ r(1) * γ^(d-1))
      cache_terms_in_graph: int
    """
    import networkx as nx

    # Map normalized cache terms to their index
    norm_to_idx = {}
    for i, t in enumerate(term_vocab):
        key = _normalize(t)
        if key:
            norm_to_idx[key] = i

    # Find cache terms that are also graph nodes
    graph_nodes = set(G.nodes())
    cache_in_graph = {k: idx for k, idx in norm_to_idx.items() if k in graph_nodes}
    print(f"Cache terms in graph: {len(cache_in_graph)} / {len(term_vocab)}")

    if len(cache_in_graph) < 20:
        print("Too few cache terms in graph for meaningful analysis.", file=sys.stderr)
        return {"per_distance": {}, "random_baseline": {}, "fitted_gamma": None,
                "cache_terms_in_graph": len(cache_in_graph)}

    # Sample seeds
    rng = np.random.default_rng(seed)
    all_cache_graph_terms = list(cache_in_graph.keys())
    n_seeds = min(max_seeds, len(all_cache_graph_terms))
    seed_terms = rng.choice(all_cache_graph_terms, n_seeds, replace=False)

    # BFS from each seed, collecting pairs at each distance.
    # Key subtlety: the same pair (A, B) can be found at different distances
    # from different seeds. We keep the MINIMUM distance (true shortest path
    # between A and B), not the distance from whatever seed found them.
    pair_to_min_distance: dict[tuple[int, int], int] = {}

    for i, seed_term in enumerate(seed_terms):
        if (i + 1) % 100 == 0:
            print(f"  Processing seed {i+1}/{n_seeds}...")

        seed_idx = cache_in_graph[seed_term]

        # BFS with distance tracking
        try:
            lengths = nx.single_source_shortest_path_length(G, seed_term, cutoff=max_distance)
        except nx.NetworkXError:
            continue

        for neighbor, d in lengths.items():
            if d == 0:
                continue  # skip self
            if d > max_distance:
                continue
            if neighbor in cache_in_graph:
                neighbor_idx = cache_in_graph[neighbor]
                if neighbor_idx != seed_idx:
                    pair = (min(seed_idx, neighbor_idx), max(seed_idx, neighbor_idx))
                    # Keep minimum distance found across all seeds
                    if pair not in pair_to_min_distance or d < pair_to_min_distance[pair]:
                        pair_to_min_distance[pair] = d

    # Bucket pairs by their true shortest distance
    pairs_by_distance: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair, d in pair_to_min_distance.items():
        pairs_by_distance[d].append(pair)

    # Subsample if needed
    for d in pairs_by_distance:
        if len(pairs_by_distance[d]) > max_pairs_per_distance:
            sampled = rng.choice(
                len(pairs_by_distance[d]), max_pairs_per_distance, replace=False
            )
            pairs_by_distance[d] = [pairs_by_distance[d][i] for i in sampled]

    total_pairs = sum(len(v) for v in pairs_by_distance.values())
    print(f"  Total unique pairs found: {total_pairs} across "
          f"raw hops {sorted(pairs_by_distance.keys())}")

    # Map index -> normalized term (for path lookup in G)
    idx_to_norm = {idx: norm for norm, idx in cache_in_graph.items()}
    idx_to_term = {idx: term_vocab[idx] for idx in set(cache_in_graph.values())}

    # Hierarchy distance = number of parent_child edges (synonym-only path -> 0).
    # So: d_hierarchy 0 = synonym (same concept); 1 = one child<->parent; 2 = two hierarchy steps; ...
    per_hierarchy: dict[int, list[tuple[float, str, str, str]]] = defaultdict(list)  # d_h -> [(r, path_type, ta, tb), ...]
    all_pairs_with_r: list[tuple[str, str, float, int]] = []  # (norm_a, norm_b, r, d_hierarchy) for IC and Mantel

    for d in sorted(pairs_by_distance.keys()):
        for idx_a, idx_b in pairs_by_distance[d]:
            r = np.corrcoef(term_maps[idx_a], term_maps[idx_b])[0, 1]
            if not np.isfinite(r):
                continue
            norm_a = idx_to_norm.get(idx_a)
            norm_b = idx_to_norm.get(idx_b)
            pt = "unknown"
            d_hierarchy = 0
            if norm_a and norm_b and norm_a != norm_b:
                try:
                    path_nodes = nx.shortest_path(G, norm_a, norm_b)
                    n_synonym = n_parent_child = 0
                    for i in range(len(path_nodes) - 1):
                        u, v = path_nodes[i], path_nodes[i + 1]
                        rel = G.edges.get((u, v), G.edges.get((v, u), {})).get("relation", "other")
                        if rel == "synonym":
                            n_synonym += 1
                        elif rel == "parent_child":
                            n_parent_child += 1
                    d_hierarchy = n_parent_child  # primary distance: hierarchy steps only
                    if n_parent_child == 0:
                        pt = "synonym"  # own category at d_hierarchy=0
                    elif n_synonym == 0:
                        pt = "all_parent_child"
                    else:
                        pt = "mixed"
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                    pass
            if norm_a and norm_b:
                all_pairs_with_r.append((norm_a, norm_b, float(r), d_hierarchy))
            ta = idx_to_term.get(idx_a, term_vocab[idx_a])
            tb = idx_to_term.get(idx_b, term_vocab[idx_b])
            per_hierarchy[d_hierarchy].append((float(r), pt, ta, tb))

    # Build per_distance keyed by d_hierarchy (0 = synonym, 1 = one hierarchy step, ...)
    per_distance = {}
    for d_h in sorted(per_hierarchy.keys()):
        items = per_hierarchy[d_h]
        corrs = [x[0] for x in items]
        path_types = [x[1] for x in items]
        if not corrs:
            continue
        corrs_arr = np.array(corrs)
        example_pairs = sorted(items, key=lambda x: -x[0])[:200]
        by_path_type = {}
        for pt in ("synonym", "all_parent_child", "mixed", "unknown"):
            mask = np.array([p == pt for p in path_types])
            if np.sum(mask) > 0:
                by_path_type[pt] = {
                    "mean_r": float(np.mean(corrs_arr[mask])),
                    "n": int(np.sum(mask)),
                }
        per_distance[d_h] = {
            "mean_r": float(np.mean(corrs)),
            "std_r": float(np.std(corrs)),
            "median_r": float(np.median(corrs)),
            "n": len(corrs),
            "q10": float(np.percentile(corrs, 10)),
            "q25": float(np.percentile(corrs, 25)),
            "q75": float(np.percentile(corrs, 75)),
            "q90": float(np.percentile(corrs, 90)),
            "r_values": [float(r) for r in corrs],
            "by_path_type": by_path_type,
            "examples_high": [(ta, tb, r) for r, _, ta, tb in example_pairs[:3]],
            "examples_low": [(ta, tb, r) for r, _, ta, tb in example_pairs[-3:]],
        }

    # Random baseline
    n_random = min(10000, len(term_vocab) * 5)
    random_corrs = []
    for _ in range(n_random):
        i, j = rng.integers(0, len(term_vocab), 2)
        if i == j:
            continue
        r = np.corrcoef(term_maps[i], term_maps[j])[0, 1]
        if np.isfinite(r):
            random_corrs.append(r)
    random_baseline = {
        "mean_r": float(np.mean(random_corrs)) if random_corrs else 0.0,
        "std_r": float(np.std(random_corrs)) if random_corrs else 0.0,
        "n": len(random_corrs),
    }

    # Fit γ: r(d) ≈ r(1) * γ^(d-1)
    # Taking log: log(r(d)) ≈ log(r(1)) + (d-1)*log(γ)
    # Linear regression of log(mean_r) on d gives slope = log(γ)
    fitted_gamma = None
    # Fit on hierarchy steps only (d_hier >= 1); synonym d_hier=0 is its own category
    distances_for_fit = sorted([d for d in per_distance if d >= 1 and per_distance[d]["mean_r"] > 0])
    if len(distances_for_fit) >= 2:
        ds = np.array(distances_for_fit, dtype=float)
        rs = np.array([per_distance[d]["mean_r"] for d in distances_for_fit])

        # Only fit on positive correlations (log requires > 0)
        # Subtract random baseline to get "signal above chance"
        baseline = random_baseline["mean_r"]
        rs_above = rs - baseline
        valid = rs_above > 0.01  # need meaningful signal

        if np.sum(valid) >= 2:
            ds_fit = ds[valid]
            rs_fit = rs_above[valid]
            log_rs = np.log(rs_fit)

            # Linear fit: log(r_above) = intercept + slope * d
            # γ = exp(slope)
            from scipy import stats as sp_stats
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(ds_fit, log_rs)
            fitted_gamma = float(np.exp(slope))

            # Clamp to reasonable range
            fitted_gamma = max(0.1, min(0.99, fitted_gamma))

    return {
        "per_distance": per_distance,
        "random_baseline": random_baseline,
        "fitted_gamma": fitted_gamma,
        "cache_terms_in_graph": len(cache_in_graph),
        "n_seeds": n_seeds,
        "all_pairs_with_r": all_pairs_with_r,
    }


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------

def print_report(results: dict):
    """Print a formatted table of distance vs correlation."""
    per_distance = results["per_distance"]
    random = results["random_baseline"]
    gamma = results["fitted_gamma"]

    print("\n" + "=" * 75)
    print("HIERARCHY DISTANCE vs BRAIN-MAP CORRELATION")
    print("  (distance = number of parent_child hops; 0 = synonym only, same concept)")
    print("=" * 75)
    print(f"Cache terms in graph: {results['cache_terms_in_graph']}")
    print(f"Seeds sampled: {results['n_seeds']}")
    print()

    print(f"{'d_hier':>8} {'Label':>12} {'Mean r':>10} {'Std r':>10} {'Median r':>10} "
          f"{'N pairs':>10} {'IQR':>15}")
    print("-" * 75)

    for d in sorted(per_distance.keys()):
        s = per_distance[d]
        iqr = f"[{s['q25']:.3f}, {s['q75']:.3f}]"
        label = "synonym" if d == 0 else f"{d} hop(s)"
        print(f"{d:>8d} {label:>12} {s['mean_r']:>10.4f} {s['std_r']:>10.4f} "
              f"{s['median_r']:>10.4f} {s['n']:>10d} {iqr:>15}")

    print("-" * 75)
    print(f"{'random':>10} {random['mean_r']:>10.4f} {random['std_r']:>10.4f} "
          f"{'':>10} {random['n']:>10d}")

    # Percentiles: show full distribution (why "some stay up, some go to 0")
    print("\n" + "-" * 75)
    print("PERCENTILES (full distribution at each hierarchy distance)")
    print("-" * 75)
    print(f"{'d_hier':>8} {'q10':>8} {'q25':>8} {'q50':>8} {'q75':>8} {'q90':>8}  (median = q50)")
    print("-" * 75)
    for d in sorted(per_distance.keys()):
        s = per_distance[d]
        q10 = s.get("q10", s["median_r"])
        q90 = s.get("q90", s["median_r"])
        print(f"{d:>8d} {q10:>8.3f} {s['q25']:>8.3f} {s['median_r']:>8.3f} {s['q75']:>8.3f} {q90:>8.3f}")
    print("-" * 75)
    print("  -> Spread (q90 - q10) shows heterogeneity: some pairs stay high, others drop to random.")

    # Path type: synonym vs parent_child vs mixed (explains why some stay high)
    print("\n" + "-" * 75)
    print("BY PATH TYPE (synonym vs parent_child along shortest path)")
    print("-" * 75)
    for d in sorted(per_distance.keys()):
        by_pt = per_distance[d].get("by_path_type", {})
        if not by_pt:
            continue
        label = "0 (synonym)" if d == 0 else f"d_hier={d}"
        print(f"\n  {label}:")
        for pt in ("synonym", "all_parent_child", "mixed", "unknown"):
            if pt not in by_pt:
                continue
            info = by_pt[pt]
            print(f"    {pt:18s}  mean r = {info['mean_r']:.4f}  n = {info['n']}")
    print("\n  -> Synonym (d_hier=0) = same concept; parent_child = hierarchy steps (child<->parent).")

    # Pattern summary
    print("\n" + "-" * 75)
    print("PATTERN SUMMARY")
    print("-" * 75)
    baseline = random["mean_r"]
    baseline_plus_std = baseline + random["std_r"]
    for d in sorted(per_distance.keys()):
        s = per_distance[d]
        rs = s.get("r_values", [])
        if not rs:
            continue
        pct_above_02 = 100 * sum(1 for r in rs if r > 0.2) / len(rs)
        pct_above_baseline = 100 * sum(1 for r in rs if r > baseline_plus_std) / len(rs)
        pct_below_0 = 100 * sum(1 for r in rs if r < 0) / len(rs)
        print(f"  d={d}:  {pct_above_02:.0f}% pairs r>0.2  |  {pct_above_baseline:.0f}% above random+std  |  {pct_below_0:.0f}% negative")
    print("  -> After d=1, most pairs fall to random; a minority stay high (often synonym paths).")

    # Show example pairs per distance for sanity checking
    print("\n" + "-" * 75)
    print("EXAMPLE PAIRS (sanity check)")
    print("-" * 75)
    for d in sorted(per_distance.keys()):
        s = per_distance[d]
        highs = s.get("examples_high", [])
        lows = s.get("examples_low", [])
        label = "0 (synonym)" if d == 0 else f"d_hier={d}"
        print(f"\n  {label} (mean r = {s['mean_r']:.3f}, n = {s['n']}):")
        if highs:
            print("    Highest-r pairs:")
            for ta, tb, r in highs[:3]:
                print(f"      r={r:+.3f}  '{ta}' vs '{tb}'")
        if lows:
            print("    Lowest-r pairs:")
            for ta, tb, r in lows[:3]:
                print(f"      r={r:+.3f}  '{ta}' vs '{tb}'")
    print()

    if gamma is not None:
        print(f"Fitted gamma (weight = gamma^d): {gamma:.4f}")
        print(f"  -> Use --graph-distance-gamma {gamma:.3f} in build_expanded_term_maps.py")
        print()
        print("Implied weights per hierarchy distance (gamma^d_hier):")
        for d in sorted(per_distance.keys()):
            w = gamma ** d
            observed = per_distance.get(d, {}).get("mean_r", None)
            obs_str = f"(observed: {observed:.4f})" if observed is not None else ""
            above_random = observed is not None and observed > random["mean_r"] + random["std_r"]
            flag = "  [ok] above random" if above_random else "  [x] at/below random" if observed is not None else ""
            lbl = "0 (synonym)" if d == 0 else f"d_hier={d}"
            print(f"    {lbl}: weight={w:.4f} {obs_str}{flag}")

        # Suggest max useful hierarchy distance
        for d in sorted(per_distance.keys()):
            if per_distance[d]["mean_r"] <= random["mean_r"] + random["std_r"]:
                if d == 0:
                    print("\n  [!] Synonym (d_hier=0) at/below random (unexpected).")
                else:
                    print(f"\n  [!] d_hier={d} hits random baseline -> expansion beyond {d-1} hierarchy hops adds noise.")
                    print(f"    Recommend: limit expansion to d_hierarchy <= {d-1}")
                break
    else:
        print("Could not fit gamma (too few positive-correlation distances).")
        print("This may mean ontology structure does not predict brain-map similarity.")


def _report_ic_similarity(G: "nx.Graph", results: dict) -> None:
    """
    Report Lin similarity (Sánchez intrinsic IC) vs brain-map r.
    Requires nxontology; uses same DAG as hierarchy (parent_child only).
    """
    nxo = build_nxontology_for_ic(G)
    if nxo is None:
        print("\n[--use-ic] nxontology not installed. pip install nxontology to enable Lin/Resnik/JC.")
        return
    pairs = results.get("all_pairs_with_r", [])
    if not pairs:
        return
    r_vals = []
    lin_vals = []
    for item in pairs:
        norm_a, norm_b, r = item[0], item[1], item[2]
        if norm_a == norm_b:
            continue
        if norm_a not in nxo.graph or norm_b not in nxo.graph:
            continue
        try:
            sim = nxo.similarity(norm_a, norm_b, ic_metric="intrinsic_ic_sanchez")
            lin_vals.append(sim.lin)
            r_vals.append(r)
        except (KeyError, ValueError, TypeError):
            continue
    if len(r_vals) < 10:
        print("\n[--use-ic] Too few pairs with valid Lin similarity (need shared hierarchy).")
        return
    r_vals = np.array(r_vals)
    lin_vals = np.array(lin_vals)
    pearson = np.corrcoef(r_vals, lin_vals)[0, 1] if np.isfinite(r_vals).all() else float("nan")
    from scipy import stats as sp_stats
    spearman_r, spearman_p = sp_stats.spearmanr(r_vals, lin_vals) if len(r_vals) > 2 else (float("nan"), float("nan"))
    print("\n" + "=" * 75)
    print("LIN SIMILARITY (Sanchez intrinsic IC) vs BRAIN-MAP r")
    print("  IC weights edges by information content; should extend predictive range vs hop count.")
    print("=" * 75)
    print(f"  Pairs with valid Lin (shared hierarchy): {len(r_vals)} / {len(pairs)}")
    print(f"  Pearson(r_brain, Lin_similarity):  {pearson:.4f}")
    print(f"  Spearman(r_brain, Lin_similarity): {spearman_r:.4f}  (p={spearman_p:.2e})" if np.isfinite(spearman_p) else "  Spearman: N/A")
    print("  -> Compare to hierarchy-distance table above; Lin often predicts r better at d_hier>=2.")
    print("  -> For spatial null (Mantel test), use neuromaps/BrainSMASH (see implementation note).")
    print()


def _run_mantel_test(results: dict, n_perm: int = 1000, seed: int = 42) -> None:
    """
    Mantel test: correlation between brain-map r and ontology similarity across pairs,
    with permutation null (shuffle ontology similarity across pairs).
    Reports Mantel r and p-value. Optional spatial null would use surrogate brain maps.
    """
    pairs = results.get("all_pairs_with_r", [])
    if len(pairs) < 20:
        print("\n[--mantel] Too few pairs for Mantel test.")
        return
    vec_r = np.array([p[2] for p in pairs])
    # Ontology similarity: 1/(1+d_hierarchy) so similar terms (low d) -> high value
    vec_onto = np.array([1.0 / (1.0 + max(0, p[3])) for p in pairs])
    mantel_obs = float(np.corrcoef(vec_r, vec_onto)[0, 1])
    if not np.isfinite(mantel_obs):
        print("\n[--mantel] Mantel statistic is NaN.")
        return
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        perm = rng.permutation(len(vec_onto))
        mantel_null = np.corrcoef(vec_r, vec_onto[perm])[0, 1]
        if np.isfinite(mantel_null):
            nulls.append(mantel_null)
    nulls = np.array(nulls)
    # One-tailed p: proportion of null >= observed (we expect positive association)
    p_val = float((1 + (nulls >= mantel_obs).sum()) / (1 + len(nulls)))
    r_sq = mantel_obs ** 2
    print("\n" + "=" * 75)
    print("MANTEL TEST (ontology similarity vs brain-map r)")
    print("  Permutation null: shuffle ontology similarity across pairs.")
    print("  (For spatial null: use neuromaps/BrainSMASH surrogates; see implementation note.)")
    print("=" * 75)
    print(f"  Pairs: {len(pairs)}")
    print(f"  Ontology similarity: 1/(1+d_hierarchy) per pair")
    print(f"  Mantel r (observed): {mantel_obs:.4f}")
    print(f"  r^2 (variance explained): {r_sq:.4f} ({100*r_sq:.2f}%)")
    print(f"  p-value (one-tailed, n_perm={n_perm}): {p_val:.4f}")
    if abs(mantel_obs) < 0.1:
        print("  -> Effect size is very small (|r| < 0.1): association is negligible in practice.")
    if p_val < 0.05:
        print("  -> p < 0.05: non-zero association, but small r means little predictive value.")
    else:
        print("  -> Non-significant at alpha=0.05 (permutation null).")
    print("  -> Try --use-ic (Lin similarity) for a stronger ontology measure; or accept weak link.")
    print()


def make_plot(results: dict, output_path: str, scatter_points: dict | None = None):
    """Plot correlation decay curve with random baseline.
    scatter_points: optional {d: list of r values} to show individual pairs in space.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return

    per_distance = results["per_distance"]
    random = results["random_baseline"]
    gamma = results["fitted_gamma"]

    if not per_distance:
        print("No distance data to plot.", file=sys.stderr)
        return

    distances = sorted(per_distance.keys())
    mean_rs = [per_distance[d]["mean_r"] for d in distances]
    std_rs = [per_distance[d]["std_r"] for d in distances]
    q25s = [per_distance[d]["q25"] for d in distances]
    q75s = [per_distance[d]["q75"] for d in distances]
    q10s = [per_distance[d].get("q10", per_distance[d]["median_r"]) for d in distances]
    q90s = [per_distance[d].get("q90", per_distance[d]["median_r"]) for d in distances]
    medians = [per_distance[d]["median_r"] for d in distances]
    ns = [per_distance[d]["n"] for d in distances]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: correlation vs distance (with optional scatter of individual pairs)
    if scatter_points:
        # Show how correlation distributes in space at each distance (jitter x slightly)
        rng = np.random.default_rng(42)
        for d in distances:
            rs = scatter_points.get(d, [])
            if not rs:
                continue
            rs = np.asarray(rs)
            # Subsample if huge for visibility
            if len(rs) > 800:
                rs = rng.choice(rs, 800, replace=False)
            jitter = rng.uniform(-0.15, 0.15, size=len(rs))
            x_coords = d + jitter
            ax1.scatter(x_coords, rs, alpha=0.25, s=8,
                        c="#1565C0", edgecolors="none")
        ax1.set_xlabel("Graph Distance (hops)", fontsize=12)
    ax1.errorbar(distances, mean_rs, yerr=std_rs, fmt="o-", color="#2196F3",
                 capsize=4, capthick=1.5, linewidth=2, markersize=8, label="Mean +/- std")
    ax1.fill_between(distances, q25s, q75s, alpha=0.15, color="#2196F3", label="IQR (q25-q75)")
    # Percentile curves: show full spread (some stay up, some go to 0)
    ax1.plot(distances, q10s, "s-", color="#0D47A1", linewidth=1, markersize=4, alpha=0.8, label="q10 (low tail)")
    ax1.plot(distances, medians, "^-", color="#1565C0", linewidth=1, markersize=4, alpha=0.8, label="Median")
    ax1.plot(distances, q90s, "v-", color="#42A5F5", linewidth=1, markersize=4, alpha=0.8, label="q90 (high tail)")

    # Random baseline
    ax1.axhline(random["mean_r"], color="#F44336", linestyle="--", linewidth=1.5,
                label=f"Random baseline ({random['mean_r']:.3f})")
    ax1.axhspan(random["mean_r"] - random["std_r"],
                random["mean_r"] + random["std_r"],
                alpha=0.1, color="#F44336")

    # Fitted decay curve (from d_hier=1 if 0 is synonym, else from min)
    if gamma is not None and len(distances) >= 2:
        d_min = min(distances)
        d_max = max(distances)
        d_fit = np.linspace(d_min, d_max, 100)
        baseline = random["mean_r"]
        r_ref = per_distance[d_min]["mean_r"] - baseline
        r_fit = baseline + r_ref * gamma ** (d_fit - d_min)
        ax1.plot(d_fit, r_fit, ":", color="#4CAF50", linewidth=2,
                 label=f"Fit: gamma = {gamma:.3f}")

    if not scatter_points:
        ax1.set_xlabel("Hierarchy distance (parent_child hops; 0 = synonym)", fontsize=12)
    ax1.set_ylabel("Brain-Map Pearson r", fontsize=12)
    ax1.set_title("Correlation vs hierarchy distance (synonym=0, then parent-child steps)", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.set_xticks(distances)
    ax1.grid(True, alpha=0.3)

    # Right panel: sample size per hierarchy distance (log scale)
    ax2.bar(distances, ns, color="#FF9800", alpha=0.7, edgecolor="#E65100")
    ax2.set_xlabel("Hierarchy distance (0 = synonym)", fontsize=12)
    ax2.set_ylabel("Number of Pairs", fontsize=12)
    ax2.set_title("Pair Count per Hierarchy Distance", fontsize=13)
    ax2.set_yscale("log")
    ax2.set_xticks(distances)
    ax2.grid(True, alpha=0.3, axis="y")

    for d, n in zip(distances, ns):
        ax2.text(d, n * 1.2, f"{n}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output_path}")
    plt.close()


def make_pearson_vs_distance_plot(
    results: dict,
    output_path: str,
    x_vals: np.ndarray | None = None,
    x_label: str | None = None,
    plot_title: str | None = None,
) -> None:
    """
    Scatter plot: x = ontology distance/similarity, y = brain-map Pearson r.
    If x_vals and x_label are provided, use them (e.g. Lin/Resnik/Jiang similarity).
    Otherwise x = hierarchy distance (d_hierarchy).
    plot_title: optional figure title (used when x_vals is provided).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping Pearson vs distance plot.", file=sys.stderr)
        return
    pairs = results.get("all_pairs_with_r", [])
    if not pairs:
        return
    r_vals = np.array([p[2] for p in pairs])
    random = results.get("random_baseline", {})
    random_mean = random.get("mean_r", 0.0)
    random_std = random.get("std_r", 0.0)
    per_distance = results.get("per_distance", {})

    if x_vals is not None and x_label is not None and len(x_vals) == len(r_vals):
        # IC-based (e.g. Lin similarity): use only pairs with finite x
        ok = np.isfinite(x_vals)
        if np.sum(ok) >= 20:
            x_scatter = np.asarray(x_vals[ok], dtype=float)
            r_vals = np.asarray(r_vals[ok], dtype=float)
            use_lin = True
        else:
            x_vals = None
            x_label = None
            use_lin = False
    else:
        use_lin = False
    if not use_lin:
        # Hierarchy distance
        d_vals = np.array([p[3] for p in pairs])
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.08, 0.08, size=len(d_vals))
        x_scatter = d_vals + jitter
        r_vals = np.asarray(r_vals, dtype=float)
        x_label = "Distance between 2 terms (# parent_child steps on shortest path; 0 = same concept)"
        use_lin = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_scatter, r_vals, alpha=0.35, s=12, c="#1565C0", edgecolors="none", label="Pairs")
    # Binned mean trend (same pairs): makes weak relationship visible
    if use_lin and len(x_scatter) >= 30:
        n_bins = min(10, max(3, len(x_scatter) // 30))
        try:
            bins = np.percentile(x_scatter, np.linspace(0, 100, n_bins + 1))
            bins[-1] += 1e-9
            bin_idx = np.searchsorted(bins[1:-1], x_scatter, side="right")
            bin_means_x, bin_means_r = [], []
            for b in range(n_bins):
                mask = bin_idx == b
                if np.sum(mask) >= 5:
                    bin_means_x.append(np.mean(x_scatter[mask]))
                    bin_means_r.append(np.mean(r_vals[mask]))
            if len(bin_means_x) >= 2:
                ax.plot(bin_means_x, bin_means_r, "o-", color="#0D47A1", linewidth=2, markersize=8, label="Mean r per similarity bin")
        except Exception:
            pass
    if not use_lin and per_distance:
        ds = sorted(per_distance.keys())
        mean_rs = [per_distance[d]["mean_r"] for d in ds]
        ax.plot(ds, mean_rs, "o-", color="#0D47A1", linewidth=2, markersize=10, label="Mean r per distance")
    ax.axhline(random_mean, color="#F44336", linestyle="--", linewidth=1.5, label=f"Random baseline ({random_mean:.3f})")
    ax.axhspan(random_mean - random_std, random_mean + random_std, alpha=0.15, color="#F44336")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Brain-map Pearson r", fontsize=12)
    if plot_title:
        title = plot_title
    else:
        title = "Pearson r vs Lin similarity (Sanchez IC)" if use_lin else "Pearson r vs distance between terms"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Graph-distance seed analysis: brain-map correlation vs ontological distance"
    )
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache",
                        help="Cache with term_maps.npz + term_vocab.pkl")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies",
                        help="Directory containing OBO/OWL ontology files")
    parser.add_argument("--max-distance", type=int, default=8,
                        help="Maximum graph distance to analyze (default 8)")
    parser.add_argument("--max-seeds", type=int, default=500,
                        help="Maximum seed terms for BFS (default 500)")
    parser.add_argument("--max-pairs-per-distance", type=int, default=5000,
                        help="Cap pairs per distance bucket (default 5000)")
    parser.add_argument("--plot", default=None,
                        help="Output path for decay plot (e.g. neurolab/data/distance_decay.png)")
    parser.add_argument("--output-json", default=None,
                        help="Output path for JSON stats (e.g. neurolab/data/graph_distance_stats.json)")
    parser.add_argument("--use-ic", action="store_true",
                        help="Compute Lin similarity (Sanchez intrinsic IC) via nxontology; report corr(r_brain, Lin)")
    parser.add_argument("--mantel", action="store_true",
                        help="Run Mantel test (ontology vs brain similarity) with permutation null; report r and p")
    parser.add_argument("--mantel-perms", type=int, default=1000,
                        help="Number of permutations for Mantel null (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve paths
    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(repo_root, args.cache_dir)
    ontology_dir = args.ontology_dir if os.path.isabs(args.ontology_dir) else os.path.join(repo_root, args.ontology_dir)

    # Load cache
    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        print("Cache not found. Build first.", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)
    term_maps = np.asarray(data["term_maps"])
    with open(pkl_path, "rb") as f:
        term_vocab = pickle.load(f)
    term_vocab = list(term_vocab)

    if term_maps.shape[0] != len(term_vocab) or term_maps.shape[1] != N_PARCELS:
        print("Cache shape mismatch.", file=sys.stderr)
        sys.exit(1)

    print(f"Cache: {len(term_vocab)} terms × {N_PARCELS} parcels")

    # Build unified graph
    print(f"\nBuilding unified ontology graph from {ontology_dir}...")
    G = build_unified_graph(ontology_dir)

    if G.number_of_nodes() < 10:
        print("Graph too small for analysis.", file=sys.stderr)
        sys.exit(1)

    # Run seed analysis
    print(f"\nRunning seed analysis (max_distance={args.max_distance}, "
          f"max_seeds={args.max_seeds})...")
    results = compute_distance_correlations(
        G, term_maps, term_vocab,
        max_distance=args.max_distance,
        max_seeds=args.max_seeds,
        max_pairs_per_distance=args.max_pairs_per_distance,
        seed=args.seed,
    )

    # Report
    print_report(results)

    # IC-based similarity (Lin + Sanchez intrinsic IC) when requested
    if getattr(args, "use_ic", False):
        _report_ic_similarity(G, results)

    # Mantel test (ontology vs brain similarity, permutation null)
    if getattr(args, "mantel", False):
        _run_mantel_test(results, n_perm=args.mantel_perms, seed=args.seed)

    # Plot
    if args.plot:
        plot_path = args.plot if os.path.isabs(args.plot) else os.path.join(repo_root, args.plot)
        scatter_points = {
            d: v.get("r_values", []) for d, v in results["per_distance"].items()
            if v.get("r_values")
        }
        make_plot(results, plot_path, scatter_points=scatter_points)
        # Pearson r vs ontological similarity: Lin, Resnik (scaled), Jiang-Conrath when nxontology available
        plot_dir = os.path.dirname(plot_path)
        plot_base = os.path.splitext(os.path.basename(plot_path))[0]
        nxo = build_nxontology_for_ic(G)
        if nxo is not None:
            lin_vals, resnik_vals, jiang_vals = [], [], []
            for item in results.get("all_pairs_with_r", []):
                norm_a, norm_b = item[0], item[1]
                if norm_a == norm_b or norm_a not in nxo.graph or norm_b not in nxo.graph:
                    lin_vals.append(np.nan)
                    resnik_vals.append(np.nan)
                    jiang_vals.append(np.nan)
                    continue
                try:
                    sim = nxo.similarity(norm_a, norm_b, ic_metric="intrinsic_ic_sanchez")
                    lin_vals.append(sim.lin)
                    resnik_vals.append(sim.resnik_scaled)
                    jiang_vals.append(sim.jiang)
                except (KeyError, ValueError, TypeError):
                    lin_vals.append(np.nan)
                    resnik_vals.append(np.nan)
                    jiang_vals.append(np.nan)
            lin_arr = np.array(lin_vals)
            resnik_arr = np.array(resnik_vals)
            jiang_arr = np.array(jiang_vals)
            n_valid = np.sum(np.isfinite(lin_arr))
            # Sanity check: same (norm_a, norm_b) -> r from term_maps, similarity from nxo
            r_all = np.array([p[2] for p in results.get("all_pairs_with_r", [])])
            d_hier_all = np.array([p[3] for p in results.get("all_pairs_with_r", [])])
            ok = np.isfinite(lin_arr)
            if n_valid >= 20:
                r_ok = r_all[ok]
                d_ok = d_hier_all[ok]
                corr_r_lin = np.corrcoef(r_ok, lin_arr[ok])[0, 1] if n_valid > 1 else float("nan")
                corr_r_resnik = np.corrcoef(r_ok, resnik_arr[ok])[0, 1] if n_valid > 1 else float("nan")
                corr_r_jiang = np.corrcoef(r_ok, jiang_arr[ok])[0, 1] if n_valid > 1 else float("nan")
                corr_d_lin = np.corrcoef(d_ok, lin_arr[ok])[0, 1] if n_valid > 1 else float("nan")
                print("\nOntology similarity vs brain r (valid pairs, same order as scatter):")
                print(f"  Pearson(r_brain, Lin):     {corr_r_lin:.4f}")
                print(f"  Pearson(r_brain, Resnik): {corr_r_resnik:.4f}")
                print(f"  Pearson(r_brain, Jiang):  {corr_r_jiang:.4f}")
                print(f"  Pearson(d_hierarchy, Lin): {corr_d_lin:.4f}  (expect <0: higher d -> lower Lin)")
                lin_ok = lin_arr[ok]
                print(f"  Lin range (valid pairs): min={lin_ok.min():.4f} mean={lin_ok.mean():.4f} max={lin_ok.max():.4f}")
                if not (np.isfinite(corr_d_lin) and corr_d_lin < -0.05):
                    print("  [!] d_hierarchy vs Lin not clearly negative -> check DAG/alignment.")
                else:
                    print("  -> Alignment OK: ontology similarity tracks hierarchy distance.")
                # Restriction of range: most pairs have low Lin (d_hier 2-8), diluting corr(r, Lin).
                # Report correlation for high-similarity pairs only (d_hierarchy <= 1).
                high_sim = (d_hier_all <= 1) & ok
                n_high = int(np.sum(high_sim))
                if n_high >= 15:
                    r_high = r_all[high_sim]
                    lin_high = lin_arr[high_sim]
                    corr_high = np.corrcoef(r_high, lin_high)[0, 1] if n_high > 1 else float("nan")
                    print(f"  Pearson(r, Lin) for d_hierarchy<=1 only (n={n_high}): {corr_high:.4f}")
                measures = [
                    (lin_arr, "Lin similarity (Sanchez intrinsic IC) [0,1]", "Pearson r vs Lin similarity (Sanchez IC)", "pearson_vs_lin"),
                    (resnik_arr, "Resnik similarity (scaled) [0,1]", "Pearson r vs Resnik (scaled)", "pearson_vs_resnik"),
                    (jiang_arr, "Jiang-Conrath similarity [0,1]", "Pearson r vs Jiang-Conrath similarity", "pearson_vs_jiang"),
                ]
                for x_arr, x_lbl, ptitle, suffix in measures:
                    out_path = os.path.join(plot_dir, plot_base + "_" + suffix + ".png")
                    make_pearson_vs_distance_plot(
                        results, out_path,
                        x_vals=x_arr, x_label=x_lbl, plot_title=ptitle,
                    )
                # Always also write: distance between 2 terms (steps to common ancestor) vs Pearson r
                dist_path = os.path.join(plot_dir, plot_base + "_pearson_vs_distance_between_terms.png")
                make_pearson_vs_distance_plot(
                    results, dist_path,
                    plot_title="Brain-map Pearson r vs distance between 2 terms (steps to common ancestor)",
                )
            else:
                scatter_path = os.path.join(plot_dir, plot_base + "_pearson_vs_distance_between_terms.png")
                make_pearson_vs_distance_plot(
                    results, scatter_path,
                    plot_title="Brain-map Pearson r vs distance between 2 terms (steps to common ancestor)",
                )
        else:
            scatter_path = os.path.join(plot_dir, plot_base + "_pearson_vs_distance_between_terms.png")
            make_pearson_vs_distance_plot(
                results, scatter_path,
                plot_title="Brain-map Pearson r vs distance between 2 terms (steps to common ancestor)",
            )

    # JSON export
    if args.output_json:
        json_path = args.output_json if os.path.isabs(args.output_json) else os.path.join(repo_root, args.output_json)
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        # Convert for JSON serialization (strip example tuples, convert numpy types)
        per_dist_clean = {}
        for d_str, v in {str(d): v for d, v in results["per_distance"].items()}.items():
            per_dist_clean[d_str] = {
                k: v2 for k, v2 in v.items()
                if k not in ("examples_high", "examples_low", "r_values")
            }
            # Add example pairs as serializable lists
            for key in ("examples_high", "examples_low"):
                if key in v:
                    per_dist_clean[d_str][key] = [
                        {"term_a": ta, "term_b": tb, "r": r}
                        for ta, tb, r in v[key]
                    ]

        export = {
            "per_distance": per_dist_clean,
            "random_baseline": results["random_baseline"],
            "fitted_gamma": results["fitted_gamma"],
            "cache_terms_in_graph": results["cache_terms_in_graph"],
            "n_seeds": results["n_seeds"],
            "max_distance_analyzed": args.max_distance,
            "recommended_max_distance": None,
        }

        # Determine recommended max distance (where correlation hits baseline)
        baseline = results["random_baseline"]["mean_r"]
        baseline_std = results["random_baseline"]["std_r"]
        for d in sorted(results["per_distance"].keys()):
            if results["per_distance"][d]["mean_r"] <= baseline + baseline_std:
                export["recommended_max_distance"] = d - 1
                break

        with open(json_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nJSON stats saved: {json_path}")


if __name__ == "__main__":
    main()
