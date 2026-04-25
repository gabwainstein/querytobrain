#!/usr/bin/env python3
"""
Download canonical open-source ontologies for local ontology expansion.

Saves OWL/OBO/RDF/TTL files to data/ontologies/ (or --output-dir). Use with
ontology_expansion.py: load_ontology_index(output_dir) after running this once.

Default: MF + UBERON (unless --no-uberon).
Extra (--extra or individual flags): BFO, RO, Cognitive Atlas, CogPO, NBO, NIFSTD.
Clinical (--clinical): MONDO, HPO, ChEBI lite, MFOEM, PATO — disease, phenotype/symptom, drug classes.

Priority 1 — fills the clinical/pharmacological gap (ENIGMA disease maps, drug fingerprints):
  - MONDO (unified disease ontology, ~30K terms): mondo.owl — with --clinical
  - HPO (Human Phenotype Ontology, symptoms ~18K): hp.owl — with --clinical
  - ChEBI lite (drug roles/classes, manageable size): chebi_lite.obo — with --clinical (default)
  - DOID (Disease Ontology, alternative to MONDO, ~12K): doid.owl — with --doid (use DOID or MONDO, not both)

Priority 2 — enriches existing coverage:
  - MFOEM (emotion module): mfoem.owl — with --clinical or --mfoem
  - PATO (phenotypic qualities, e.g. decreased volume): pato.owl — with --clinical or --pato

Cognition/anatomy (--extra or individual):
  - Mental Functioning (MF): mf.owl
  - UBERON (anatomy): uberon.owl (optional, large ~50MB+; use --no-uberon to skip)
  - BFO, RO, Cognitive Atlas, CogPO, NBO — see --extra, --cognitive-atlas, etc.
  - GO (Gene Ontology): go-basic.owl — with --go
  - NIFSTD (neuroscience): scicrunch-registry.ttl — with --nifstd
  - NPT (NeuroPsychological Testing, tests→constructs): NPT.owl — with --npt

ChEBI: --clinical downloads ChEBI lite by default (drug class hierarchies). Use --chebi-full for full
ChEBI (~200K terms). Use --no-chebi to skip ChEBI entirely.

Usage:
  python download_ontologies.py [--output-dir path] [--no-uberon]
  python download_ontologies.py --extra
  python download_ontologies.py --clinical   # MONDO + HPO + ChEBI lite + MFOEM + PATO
  python download_ontologies.py --clinical --no-chebi   # disease + phenotype only
  python download_ontologies.py --clinical --chebi-full   # full ChEBI instead of lite
  python download_ontologies.py --doid   # DOID instead of/in addition to MONDO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request

# Ontology URLs
# OBO Foundry (purl.obolibrary.org)
MF_OWL = "http://purl.obolibrary.org/obo/mf.owl"
UBERON_OWL = "http://purl.obolibrary.org/obo/uberon.owl"
BFO_OWL = "http://purl.obolibrary.org/obo/bfo.owl"
RO_OWL = "http://purl.obolibrary.org/obo/ro.owl"
# Cognitive Atlas (GitHub raw)
COGAT_OWL = "https://raw.githubusercontent.com/CognitiveAtlas/cogat-ontology/master/ontology/cogat.v2.owl"
# CogPO (cognitive paradigms; task stimulus/instruction/response)
COGPO_OWL = "http://www.cogpo.org/ontologies/CogPOver1.owl"
# NBO (Neuro Behavior Ontology — OBO Foundry)
NBO_OWL = "http://purl.obolibrary.org/obo/nbo.owl"
# GO (Gene Ontology — go-basic, acyclic; BP/MF/CC, ~40MB+)
GO_OWL = "http://purl.obolibrary.org/obo/go/go-basic.owl"
# NIFSTD / SciCrunch registry (GitHub raw, Turtle — ontology_expansion loads .ttl; ~18MB)
NIFSTD_TTL = "https://raw.githubusercontent.com/SciCrunch/NIF-Ontology/master/scicrunch-registry.ttl"
# NPT (NeuroPsychological Testing — MMSE, MoCA, Trail-Making, ADAS, AVLT; maps tests to cognitive constructs)
NPT_OWL = "https://raw.githubusercontent.com/addiehl/neuropsychological-testing-ontology/master/ontology/NPT.owl"
# Clinical / disease / phenotype / drug (OBO Foundry)
MONDO_OWL = "http://purl.obolibrary.org/obo/mondo.owl"
DOID_OWL = "http://purl.obolibrary.org/obo/doid.owl"  # Disease Ontology (~12K); use DOID or MONDO
HP_OWL = "http://purl.obolibrary.org/obo/hp.owl"
CHEBI_OWL = "http://purl.obolibrary.org/obo/chebi.owl"  # full ~200K terms
# ChEBI lite: drug roles/classes, manageable for pharmacological fingerprint maps (EBI FTP)
CHEBI_LITE_OBO = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo"
MFOEM_OWL = "http://purl.obolibrary.org/obo/mfoem.owl"
PATO_OWL = "http://purl.obolibrary.org/obo/pato.owl"

CHUNK = 1024 * 256  # 256 KB
TIMEOUT = 300  # seconds


def download(url: str, dest: Path, timeout: int = TIMEOUT) -> bool:
    req = Request(url, headers={"User-Agent": "ontology-download/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Download failed {url}: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ontologies for local expansion.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save OWL/OBO/TTL files (default: neurolab/data/ontologies)",
    )
    parser.add_argument(
        "--no-uberon",
        action="store_true",
        help="Skip UBERON (large file, ~50MB+).",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also download BFO, RO, Cognitive Atlas, CogPO, NBO (and optionally NIFSTD with --nifstd).",
    )
    parser.add_argument("--bfo", action="store_true", help="Download BFO (Basic Formal Ontology).")
    parser.add_argument("--ro", action="store_true", help="Download RO (Relation Ontology).")
    parser.add_argument(
        "--cognitive-atlas",
        action="store_true",
        help="Download Cognitive Atlas (cognition/tasks, cogat.v2.owl).",
    )
    parser.add_argument(
        "--cogpo",
        action="store_true",
        help="Download CogPO (cognitive paradigms, task conditions).",
    )
    parser.add_argument(
        "--nbo",
        action="store_true",
        help="Download NBO (Neuro Behavior Ontology).",
    )
    parser.add_argument(
        "--go",
        action="store_true",
        help="Download Gene Ontology go-basic.owl (large, ~40MB+).",
    )
    parser.add_argument(
        "--nifstd",
        action="store_true",
        help="Download NIFSTD neuroscience ontology (large TTL).",
    )
    parser.add_argument(
        "--npt",
        action="store_true",
        help="Download NPT (NeuroPsychological Testing Ontology; tests → cognitive constructs).",
    )
    parser.add_argument(
        "--clinical",
        action="store_true",
        help="Download MONDO, HPO, ChEBI lite, MFOEM, PATO (disease, phenotype, drug classes).",
    )
    parser.add_argument(
        "--no-chebi",
        action="store_true",
        help="With --clinical: skip ChEBI entirely (disease + phenotype only).",
    )
    parser.add_argument(
        "--chebi-full",
        action="store_true",
        help="With --clinical: download full ChEBI (~200K terms) instead of ChEBI lite.",
    )
    parser.add_argument(
        "--mondo",
        action="store_true",
        help="Download MONDO (unified disease ontology, ~30K terms).",
    )
    parser.add_argument(
        "--doid",
        action="store_true",
        help="Download DOID (Disease Ontology, ~12K terms). Use DOID or MONDO, not both.",
    )
    parser.add_argument(
        "--hp",
        action="store_true",
        help="Download HPO (Human Phenotype Ontology, symptoms, ~18K terms).",
    )
    parser.add_argument(
        "--chebi",
        action="store_true",
        help="Download full ChEBI (standalone; with --clinical use --chebi-full).",
    )
    parser.add_argument(
        "--chebi-slim",
        action="store_true",
        help="Download ChEBI lite only (drug roles/classes, standalone).",
    )
    parser.add_argument(
        "--mfoem",
        action="store_true",
        help="Download MFOEM (Mental Functioning Ontology — Emotion Module).",
    )
    parser.add_argument(
        "--pato",
        action="store_true",
        help="Download PATO (Phenotype And Trait Ontology, qualities).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        root = script_dir.parent
        args.output_dir = root / "data" / "ontologies"

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")

    ok = True

    # MF (small)
    mf_dest = out / "mf.owl"
    print(f"Downloading MF -> {mf_dest} ...")
    if download(MF_OWL, mf_dest):
        print(f"  OK ({mf_dest.stat().st_size / 1024:.1f} KB)")
    else:
        ok = False

    # UBERON (optional, large)
    if not args.no_uberon:
        ub_dest = out / "uberon.owl"
        print(f"Downloading UBERON -> {ub_dest} ...")
        if download(UBERON_OWL, ub_dest):
            print(f"  OK ({ub_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False
    else:
        print("Skipping UBERON (--no-uberon).")

    # Extra ontologies
    if args.extra or args.bfo:
        bfo_dest = out / "bfo.owl"
        print(f"Downloading BFO -> {bfo_dest} ...")
        if download(BFO_OWL, bfo_dest):
            print(f"  OK ({bfo_dest.stat().st_size / 1024:.1f} KB)")
        else:
            ok = False

    if args.extra or args.ro:
        ro_dest = out / "ro.owl"
        print(f"Downloading RO -> {ro_dest} ...")
        if download(RO_OWL, ro_dest):
            print(f"  OK ({ro_dest.stat().st_size / 1024:.1f} KB)")
        else:
            ok = False

    if args.extra or args.cognitive_atlas:
        cogat_dest = out / "cogat.v2.owl"
        print(f"Downloading Cognitive Atlas -> {cogat_dest} ...")
        if download(COGAT_OWL, cogat_dest):
            print(f"  OK ({cogat_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    if args.extra or args.cogpo:
        cogpo_dest = out / "CogPOver1.owl"
        print(f"Downloading CogPO -> {cogpo_dest} ...")
        if download(COGPO_OWL, cogpo_dest):
            print(f"  OK ({cogpo_dest.stat().st_size / 1024:.1f} KB)")
        else:
            ok = False

    if args.extra or args.nbo:
        nbo_dest = out / "nbo.owl"
        print(f"Downloading NBO -> {nbo_dest} ...")
        if download(NBO_OWL, nbo_dest):
            print(f"  OK ({nbo_dest.stat().st_size / 1024:.1f} KB)")
        else:
            ok = False

    if args.go:
        go_dest = out / "go-basic.owl"
        print(f"Downloading GO (go-basic) -> {go_dest} ...")
        if download(GO_OWL, go_dest):
            print(f"  OK ({go_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    if args.nifstd:
        nif_dest = out / "scicrunch-registry.ttl"
        print(f"Downloading NIFSTD (SciCrunch registry) -> {nif_dest} ...")
        if download(NIFSTD_TTL, nif_dest):
            print(f"  OK ({nif_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    if args.npt:
        npt_dest = out / "NPT.owl"
        print(f"Downloading NPT (NeuroPsychological Testing) -> {npt_dest} ...")
        if download(NPT_OWL, npt_dest):
            print(f"  OK ({npt_dest.stat().st_size / 1024:.1f} KB)")
        else:
            ok = False

    # Clinical ontologies (disease, phenotype, drug)
    if args.clinical or args.mondo:
        mondo_dest = out / "mondo.owl"
        print(f"Downloading MONDO -> {mondo_dest} ...")
        if download(MONDO_OWL, mondo_dest):
            print(f"  OK ({mondo_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    if args.doid:
        doid_dest = out / "doid.owl"
        print(f"Downloading DOID (Disease Ontology) -> {doid_dest} ...")
        if download(DOID_OWL, doid_dest):
            print(f"  OK ({doid_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    if args.clinical or args.hp:
        hp_dest = out / "hp.owl"
        print(f"Downloading HPO -> {hp_dest} ...")
        if download(HP_OWL, hp_dest):
            print(f"  OK ({hp_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    # ChEBI: lite (default with --clinical) or full
    do_chebi_full = (args.clinical and not args.no_chebi and args.chebi_full) or args.chebi
    do_chebi_lite = (args.clinical and not args.no_chebi and not args.chebi_full) or args.chebi_slim
    if do_chebi_full:
        chebi_dest = out / "chebi.owl"
        print(f"Downloading ChEBI (full) -> {chebi_dest} ... (large)")
        if download(CHEBI_OWL, chebi_dest):
            print(f"  OK ({chebi_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False
    elif do_chebi_lite:
        chebi_lite_dest = out / "chebi_lite.obo"
        print(f"Downloading ChEBI lite (drug roles/classes) -> {chebi_lite_dest} ...")
        if download(CHEBI_LITE_OBO, chebi_lite_dest):
            print(f"  OK ({chebi_lite_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False
    elif args.clinical and args.no_chebi:
        print("Skipping ChEBI (--no-chebi).")

    if args.clinical or args.mfoem:
        mfoem_dest = out / "mfoem.owl"
        print(f"Downloading MFOEM -> {mfoem_dest} ...")
        if download(MFOEM_OWL, mfoem_dest):
            print(f"  OK ({mfoem_dest.stat().st_size / 1024:.1f} KB)")
        else:
            ok = False

    if args.clinical or args.pato:
        pato_dest = out / "pato.owl"
        print(f"Downloading PATO -> {pato_dest} ...")
        if download(PATO_OWL, pato_dest):
            print(f"  OK ({pato_dest.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            ok = False

    if ok:
        print("Done. Use: load_ontology_index({!r})".format(str(out)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
