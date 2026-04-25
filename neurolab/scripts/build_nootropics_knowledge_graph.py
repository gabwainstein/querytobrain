#!/usr/bin/env python3
"""
Verify nootropics_knowledge_base_expanded.csv and build knowledge graph / triples.
Outputs formats compatible with ontology embedding and the enrichment pipeline.

Usage:
  python neurolab/scripts/build_nootropics_knowledge_graph.py
  python neurolab/scripts/build_nootropics_knowledge_graph.py --csv path/to/nootropics.csv
  python neurolab/scripts/build_nootropics_knowledge_graph.py --output-dir neurolab/data/nootropics_kb
  python neurolab/scripts/build_nootropics_knowledge_graph.py --verify-only  # validation report only

Output:
  - nootropics_triples.json: (subject, predicate, object) for KG/RDF
  - nootropics_ontology_index.json: label_to_related format for ontology merge
  - nootropics_knowledge_graph.pkl: NetworkX graph (optional)
  - validation_report.json: accuracy/consistency checks
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    nx = None

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

REQUIRED_COLUMNS = ["compound", "category", "primary_targets", "mechanism_text", "brain_regions_expected"]
OPTIONAL_COLUMNS = ["aliases", "priority", "cognitive_domains", "pdsp_available", "chembl_id", "pubchem_cid", "examine_url"]

# Known mechanism terms for validation (partial list)
KNOWN_MECHANISMS = {
    "AMPA", "NMDA", "GABA", "acetylcholinesterase", "dopamine", "serotonin", "BDNF", "NGF",
    "DAT", "NET", "SERT", "5-HT", "nAChR", "nicotinic", "muscarinic", "opioid",
    "MAO", "PDE", "adenosine", "cortisol", "HPA", "glutamate", "acetylcholine",
}

# Spot-check: compound -> expected substring in mechanism or primary_targets (for accuracy verification)
ACCURACY_SPOT_CHECKS = {
    "Piracetam": "AMPA",
    "Modafinil": "DAT",
    "Huperzine A": "acetylcholinesterase",
    "Caffeine": "adenosine",
    "Nicotine": "nicotinic",
    "Methylphenidate": "dopamine",
    "L-Theanine": "GABA",
    "Creatine": "phosphocreatine",
    "Alpha-GPC": "acetylcholine",
    "Lion's Mane": "NGF",
}

# Known brain regions for validation
KNOWN_REGIONS = {
    "hippocampus", "prefrontal cortex", "amygdala", "striatum", "thalamus",
    "basal ganglia", "cortex", "cerebellum", "hypothalamus", "nucleus accumbens",
    "ventral tegmental area", "substantia nigra", "locus coeruleus", "raphe nuclei",
    "basal forebrain", "temporal cortex", "parietal cortex", "occipital cortex",
    "anterior cingulate", "entorhinal cortex", "pineal gland", "white matter",
}


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _split_list(s: str, sep: str = ",") -> list[str]:
    return [x.strip() for x in (s or "").split(sep) if x.strip()]


def validate_row(row: dict, idx: int) -> list[str]:
    """Return list of validation warnings/errors for a row."""
    issues = []
    compound = row.get("compound", "").strip()
    if not compound:
        issues.append("empty compound name")
    for col in REQUIRED_COLUMNS:
        if col not in row or not str(row.get(col, "")).strip():
            issues.append(f"missing or empty {col}")
    mechanism = (row.get("mechanism_text") or "").lower()
    if mechanism and len(mechanism) < 50:
        issues.append("mechanism_text unusually short")
    regions = _split_list(row.get("brain_regions_expected", ""))
    for r in regions:
        rn = _normalize(r)
        if rn and rn not in KNOWN_REGIONS and not any(kr in rn for kr in KNOWN_REGIONS):
            pass  # allow novel regions
    chembl = row.get("chembl_id", "")
    if chembl and not re.match(r"^CHEMBL\d+$", str(chembl).strip()):
        issues.append("chembl_id format unexpected (expected CHEMBL123)")
    pubchem = row.get("pubchem_cid", "")
    if pubchem and not str(pubchem).replace(".0", "").isdigit():
        issues.append("pubchem_cid should be numeric")
    pdsp = str(row.get("pdsp_available", "")).strip()
    if pdsp and pdsp not in ("Yes", "No", "Partial", "Low"):
        issues.append(f"pdsp_available unexpected value: {pdsp}")
    return issues


def row_to_triples(row: dict) -> list[dict]:
    """Convert a CSV row to (subject, predicate, object) triples."""
    compound = row.get("compound", "").strip()
    if not compound:
        return []
    triples = []
    for alias in _split_list(row.get("aliases", "")):
        if alias and alias != compound:
            triples.append({"subject": compound, "predicate": "has_alias", "object": alias})
    cat = row.get("category", "").strip()
    if cat:
        triples.append({"subject": compound, "predicate": "has_category", "object": cat})
    for target in _split_list(row.get("primary_targets", ""), ";"):
        if target:
            triples.append({"subject": compound, "predicate": "primary_target", "object": target.strip()})
    mech = (row.get("mechanism_text") or "").strip()[:500]
    if mech:
        triples.append({"subject": compound, "predicate": "mechanism", "object": mech})
    for domain in _split_list(row.get("cognitive_domains", "")):
        if domain:
            triples.append({"subject": compound, "predicate": "cognitive_domain", "object": domain})
    for region in _split_list(row.get("brain_regions_expected", "")):
        if region:
            triples.append({"subject": compound, "predicate": "affects_brain_region", "object": region})
    chembl = row.get("chembl_id", "")
    if chembl and str(chembl).strip():
        triples.append({"subject": compound, "predicate": "chembl_id", "object": str(chembl).strip()})
    pubchem = row.get("pubchem_cid", "")
    if pubchem and str(pubchem).replace(".0", "").isdigit():
        triples.append({"subject": compound, "predicate": "pubchem_cid", "object": str(int(float(pubchem)))})
    return triples


def row_to_ontology_related(row: dict) -> list[tuple[str, float, str]]:
    """Convert row to ontology label_to_related format: [(related_term, weight, relation_type)]"""
    compound = row.get("compound", "").strip()
    if not compound:
        return []
    related = [(compound, 1.0, "self")]
    for alias in _split_list(row.get("aliases", "")):
        if alias and alias != compound:
            related.append((alias, 0.95, "synonym"))
    cat = row.get("category", "").strip()
    if cat:
        related.append((cat, 0.8, "has_category"))
    for target in _split_list(row.get("primary_targets", ""), ";"):
        if target:
            related.append((target.strip(), 0.75, "primary_target"))
    mech = (row.get("mechanism_text") or "").strip()[:300]
    if mech:
        related.append((mech, 0.6, "mechanism"))
    for domain in _split_list(row.get("cognitive_domains", "")):
        if domain:
            related.append((domain, 0.7, "cognitive_domain"))
    for region in _split_list(row.get("brain_regions_expected", "")):
        if region:
            related.append((region, 0.7, "affects_region"))
    return related


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and build nootropics knowledge graph")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true", help="Only run validation, no output files")
    parser.add_argument("--no-graph", action="store_true", help="Skip NetworkX graph output")
    parser.add_argument("--copy-to-ontology-dir", type=Path, default=None,
                        help="Copy nootropics_ontology_index.json to ontology dir for embedding")
    args = parser.parse_args()

    root = _scripts.parent
    csv_path = Path(args.csv) if args.csv else root / "docs" / "implementation" / "nootropics_knowledge_base_expanded.csv"
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "nootropics_kb"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty.", file=sys.stderr)
        return 1

    # Validation
    validation = {"total": len(rows), "issues": [], "by_compound": {}}
    for i, row in enumerate(rows):
        issues = validate_row(row, i)
        if issues:
            validation["issues"].append({"row": i + 1, "compound": row.get("compound", ""), "issues": issues})
            validation["by_compound"][row.get("compound", "")] = issues

    # Accuracy spot-checks
    accuracy_checks = []
    row_by_compound = {r.get("compound", "").strip(): r for r in rows}
    for compound, expected in ACCURACY_SPOT_CHECKS.items():
        r = row_by_compound.get(compound)
        if not r:
            accuracy_checks.append({"compound": compound, "status": "not_found"})
            continue
        mech = (r.get("mechanism_text") or "").lower()
        targets = (r.get("primary_targets") or "").lower()
        if expected.lower() in mech or expected.lower() in targets:
            accuracy_checks.append({"compound": compound, "status": "ok", "expected": expected})
        else:
            accuracy_checks.append({"compound": compound, "status": "missing_expected", "expected": expected})
    validation["accuracy_spot_checks"] = accuracy_checks

    validation["summary"] = {
        "rows_with_issues": len(validation["issues"]),
        "rows_ok": len(rows) - len(validation["issues"]),
        "accuracy_checks_ok": sum(1 for c in accuracy_checks if c["status"] == "ok"),
        "accuracy_checks_total": len(accuracy_checks),
    }

    print(f"Loaded {len(rows)} compounds from {csv_path.name}")
    print(f"Validation: {validation['summary']['rows_ok']} OK, {validation['summary']['rows_with_issues']} with issues")
    acc = validation["summary"]
    print(f"Accuracy spot-checks: {acc['accuracy_checks_ok']}/{acc['accuracy_checks_total']} passed")
    for c in accuracy_checks:
        if c["status"] != "ok":
            print(f"  {c['compound']}: expected '{c['expected']}' in mechanism/targets")
    if validation["issues"]:
        for item in validation["issues"][:10]:
            print(f"  {item['compound']}: {item['issues']}")
        if len(validation["issues"]) > 10:
            print(f"  ... and {len(validation['issues']) - 10} more")

    if args.verify_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "validation_report.json", "w") as f:
            json.dump(validation, f, indent=2)
        return 0

    # Build triples
    all_triples = []
    for row in rows:
        all_triples.extend(row_to_triples(row))

    # Build ontology index format
    label_to_related = {}
    label_to_source = {}
    for row in rows:
        compound = row.get("compound", "").strip()
        if not compound:
            continue
        norm = _normalize(compound)
        related = row_to_ontology_related(row)
        if related:
            label_to_related[norm] = related
            label_to_source[norm] = "nootropics_kb"
        for alias in _split_list(row.get("aliases", "")):
            anorm = _normalize(alias)
            if anorm and anorm not in label_to_related:
                label_to_related[anorm] = [(compound, 0.95, "synonym")]
                label_to_source[anorm] = "nootropics_kb"

    ontology_index = {
        "label_to_related": {k: [[r[0], r[1], r[2]] for r in v] for k, v in label_to_related.items()},
        "label_to_source": label_to_source,
    }

    # Build NetworkX graph
    G = None
    if nx and not args.no_graph:
        G = nx.DiGraph()
        for t in all_triples:
            s, p, o = t["subject"], t["predicate"], t["object"]
            if s and o and len(str(o)) < 200:
                G.add_edge(s, o, predicate=p)

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "nootropics_triples.json", "w") as f:
        json.dump(all_triples, f, indent=2)

    with open(out_dir / "nootropics_ontology_index.json", "w") as f:
        json.dump(ontology_index, f, indent=2)

    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(validation, f, indent=2)

    if G is not None:
        import pickle
        with open(out_dir / "nootropics_knowledge_graph.pkl", "wb") as f:
            pickle.dump(G, f)

    if args.copy_to_ontology_dir:
        dest = Path(args.copy_to_ontology_dir)
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(out_dir / "nootropics_ontology_index.json", dest / "nootropics_ontology_index.json")
        print(f"  Copied to {dest}/nootropics_ontology_index.json (will be loaded with ontologies)")

    print(f"Output: {out_dir}")
    print(f"  - nootropics_triples.json: {len(all_triples)} triples")
    print(f"  - nootropics_ontology_index.json: {len(label_to_related)} labels (merge into ontology dir)")
    print(f"  - validation_report.json")
    if G is not None:
        print(f"  - nootropics_knowledge_graph.pkl: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return 0


if __name__ == "__main__":
    sys.exit(main())
