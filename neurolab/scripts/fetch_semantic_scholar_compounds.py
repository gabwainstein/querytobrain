#!/usr/bin/env python3
"""
Fetch papers, citations, and facts for nootropic/compound vocabulary via Semantic Scholar API.
Used to expand triples for compounds not well-covered by NeuroSynth/NeuroQuery.
OpenScholar (AkariAsai/OpenScholar) uses this API; we use it directly for compound enrichment.

Output: compound_literature/ with papers, abstracts, citation counts per compound.
Optional: triples (compound, has_paper, paper_id), (compound, mechanism, abstract_snippet).

Usage:
  python neurolab/scripts/fetch_semantic_scholar_compounds.py --output-dir neurolab/data/compound_literature
  python neurolab/scripts/fetch_semantic_scholar_compounds.py --compounds alpha-GPC bacopa --max-papers 20
  python neurolab/scripts/fetch_semantic_scholar_compounds.py --from-pharma-terms  # use build_pharma_neurosynth PHARMA_TERMS

Requires: requests. Set S2_API_KEY for higher rate limits (100 req/5min without key; 429 without key is common).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Extended nootropic/compound vocabulary for literature expansion (beyond NeuroSynth coverage).
# Mechanism-of-action text can be added for training; Semantic Scholar fetches papers/facts.
NOOTROPIC_AND_COMPOUND_TERMS = [
    # Racetams
    "piracetam",
    "aniracetam",
    "oxiracetam",
    "pramiracetam",
    "phenylpiracetam",
    # Cholinergics
    "alpha-GPC",
    "citicoline",
    "CDP-choline",
    "acetyl-L-carnitine",
    "ALCAR",
    "huperzine A",
    "galantamine",
    "donepezil",
    "rivastigmine",
    "memantine",
    # Adaptogens / herbs
    "bacopa monnieri",
    "bacopa",
    "ashwagandha",
    "rhodiola rosea",
    "lion's mane",
    "Hericium erinaceus",
    "ginkgo biloba",
    "panax ginseng",
    "eleuthero",
    # Amino acids / precursors
    "L-theanine",
    "theanine",
    "N-acetyl cysteine",
    "NAC",
    "L-tyrosine",
    "phenylalanine",
    "5-HTP",
    "tryptophan",
    # Peptides (Khavinson, etc.)
    "cortexin",
    "cerebrolysin",
    "semax",
    "selank",
    "BPC-157",
    "epithalon",
    # Other nootropics
    "modafinil",
    "armodafinil",
    "methylphenidate",
    "caffeine",
    "nicotine",
    "noopept",
    "phenibut",
    "sulbutiamine",
    "vinpocetine",
    "idebenone",
    "coenzyme Q10",
    "ubiquinol",
    "creatine",
    "magnesium threonate",
    "omega-3",
    "DHA",
    "EPA",
    # Psychedelics / dissociatives (for triples expansion)
    "psilocybin",
    "LSD",
    "ketamine",
    "DMT",
    "ayahuasca",
    "MDMA",
]

S2_BASE = "https://api.semanticscholar.org/graph/v1"
# Without API key: 100 req/5min. Use 5+ sec delay to avoid 429.
DEFAULT_DELAY = 5.0
MAX_RETRIES = 5
RETRY_BACKOFF = 60  # seconds to wait on 429


def _search_papers(query: str, limit: int = 10, api_key: str | None = None) -> list[dict]:
    """Search Semantic Scholar for papers matching query. Retries on 429 with backoff."""
    if requests is None:
        raise ImportError("requests required: pip install requests")
    url = f"{S2_BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": "paperId,title,abstract,year,citationCount,url"}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(MAX_RETRIES):
        r = requests.get(url, params=params, headers=headers or None, timeout=30)
        if r.status_code == 429:
            wait = RETRY_BACKOFF * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                continue
        r.raise_for_status()
        data = r.json()
        return data.get("data", []) or []
    return []


def _get_paper_details(paper_id: str, api_key: str | None = None) -> dict | None:
    """Get full paper details including abstract."""
    if requests is None:
        return None
    url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": "paperId,title,abstract,year,citationCount,url,authors"}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    r = requests.get(url, params=params, headers=headers or None, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()


def fetch_for_compound(
    compound: str,
    max_papers: int = 15,
    api_key: str | None = None,
    delay: float = DEFAULT_DELAY,
) -> dict:
    """Fetch papers and facts for one compound."""
    results = {
        "compound": compound,
        "papers": [],
        "facts": [],
        "citation_count_total": 0,
    }
    try:
        papers = _search_papers(f"{compound} brain neuroimaging", limit=max_papers, api_key=api_key)
        if not papers:
            papers = _search_papers(compound, limit=max_papers, api_key=api_key)
    except Exception as e:
        results["error"] = str(e)
        if "429" in str(e):
            results["hint"] = "Rate limited. Set S2_API_KEY for higher limits, or increase --delay."
        return results

    for p in papers:
        pid = p.get("paperId")
        title = p.get("title", "")
        abstract = p.get("abstract") or ""
        year = p.get("year")
        cites = p.get("citationCount", 0) or 0
        url = p.get("url", "")
        results["papers"].append({
            "paperId": pid,
            "title": title,
            "abstract": abstract[:2000] if abstract else "",
            "year": year,
            "citationCount": cites,
            "url": url,
        })
        results["citation_count_total"] += cites
        if abstract:
            snippet = abstract[:500].replace("\n", " ")
            results["facts"].append({
                "type": "abstract_snippet",
                "paperId": pid,
                "text": snippet,
                "citationCount": cites,
            })
        time.sleep(delay)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Semantic Scholar papers for nootropic/compound vocabulary."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: neurolab/data/compound_literature)",
    )
    parser.add_argument(
        "--compounds",
        nargs="*",
        default=None,
        help="Compound names to fetch (default: built-in NOOTROPIC_AND_COMPOUND_TERMS)",
    )
    parser.add_argument(
        "--from-pharma-terms",
        action="store_true",
        help="Use PHARMA_TERMS from build_pharma_neurosynth_cache instead of NOOTROPIC_AND_COMPOUND_TERMS",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=15,
        help="Max papers per compound (default 15)",
    )
    parser.add_argument(
        "--max-compounds",
        type=int,
        default=0,
        help="Cap compounds (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between API requests (default {DEFAULT_DELAY}; use 5+ without API key)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip compounds already in compound_literature.json (incremental run)",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Detach and run in background (logs to compound_literature_fetch.log)",
    )
    args = parser.parse_args()

    if args.background:
        import subprocess
        root = _scripts.parent
        out = Path(args.output_dir) if args.output_dir else root / "data" / "compound_literature"
        log_path = out / "compound_literature_fetch.log"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(_scripts / "fetch_semantic_scholar_compounds.py")]
        if args.output_dir:
            cmd += ["--output-dir", str(args.output_dir)]
        if args.delay != DEFAULT_DELAY:
            cmd += ["--delay", str(args.delay)]
        if args.resume:
            cmd += ["--resume"]
        if args.max_papers:
            cmd += ["--max-papers", str(args.max_papers)]
        if args.from_pharma_terms:
            cmd += ["--from-pharma-terms"]
        print(f"Starting in background. Log: {log_path}")
        with open(log_path, "w") as log:
            popen_kw = {"stdout": log, "stderr": subprocess.STDOUT, "cwd": str(root.parent)}
            if sys.platform == "win32":
                popen_kw["creationflags"] = 0x00000008  # DETACHED_PROCESS
            else:
                popen_kw["start_new_session"] = True
            subprocess.Popen(cmd, **popen_kw)
        return 0

    if args.from_pharma_terms:
        try:
            from neurolab.scripts.build_pharma_neurosynth_cache import PHARMA_TERMS
            compounds = list(PHARMA_TERMS)
        except ImportError:
            compounds = NOOTROPIC_AND_COMPOUND_TERMS
    else:
        compounds = args.compounds or NOOTROPIC_AND_COMPOUND_TERMS

    compounds = [c.strip() for c in compounds if c.strip()]
    if args.max_compounds and len(compounds) > args.max_compounds:
        compounds = compounds[: args.max_compounds]

    api_key = __import__("os").environ.get("S2_API_KEY")

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "compound_literature"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip compounds already fetched
    done_compounds = set()
    if args.resume:
        prev_path = out_dir / "compound_literature.json"
        if prev_path.exists():
            try:
                prev = json.load(open(prev_path))
                done_compounds = {r.get("compound", "") for r in prev}
                print(f"Resume: {len(done_compounds)} compounds already fetched")
            except Exception:
                pass
    compounds = [c for c in compounds if c not in done_compounds]
    if not compounds:
        print("All compounds already fetched.")
        return 0

    all_results = []
    if args.resume and (out_dir / "compound_literature.json").exists():
        try:
            all_results = json.load(open(out_dir / "compound_literature.json"))
        except Exception:
            pass
    for i, compound in enumerate(compounds):
        print(f"[{i+1}/{len(compounds)}] {compound!r} ...")
        r = fetch_for_compound(compound, max_papers=args.max_papers, api_key=api_key, delay=args.delay)
        all_results.append(r)
        if "error" in r:
            print(f"  Error: {r['error']}")
        else:
            print(f"  {len(r['papers'])} papers, {r['citation_count_total']} total citations")

        # Incremental save after each compound (for resume + crash recovery)
        with open(out_dir / "compound_literature.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Triples for expansion: (compound, has_paper, paperId), (compound, mechanism, snippet)
    triples = []
    for r in all_results:
        c = r.get("compound", "")
        for p in r.get("papers", []):
            triples.append({"subject": c, "predicate": "has_paper", "object": p.get("paperId", "")})
        for fact in r.get("facts", []):
            triples.append({
                "subject": c,
                "predicate": "mechanism_or_abstract",
                "object": fact.get("text", "")[:300],
            })
    with open(out_dir / "compound_triples.json", "w") as f:
        json.dump(triples, f, indent=2)

    print(f"Wrote {out_dir}: {len(all_results)} compounds, {len(triples)} triples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
