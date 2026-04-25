#!/usr/bin/env python3
"""
Analyze all term labels in the training cache to find quality issues and improvement opportunities.

Identifies:
- Figure/table references (Figure 3, Fig. 4)
- Subject/study IDs (Subject0027, Study12Subject04)
- Opaque trm_ Cognitive Atlas IDs
- regparam / parametric contrast (pharma parametric regressors)
- Single-word or very short terms
- Junk placeholders (test, neurovault_image_N)
- ICA component refs (z-value voxel loadings IC14)
- Movie/experiment codes (0058_Movie_1)
- Island/study-specific jargon

Outputs: summary by category, examples, improvement suggestions, optional CSV.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Quality heuristics ---
def strip_prefix(label: str) -> str:
    """Remove modality prefix (fMRI:, Gene:, etc.) for analysis."""
    for p in ("fMRI:", "Pharmacological fMRI:", "PET:", "Structural:", "Gene:", "Cognitive:", "Perfusion:", "DTI:"):
        if label.startswith(p):
            return label[len(p):].strip()
    return label.strip()


def is_figure_reference(label: str) -> bool:
    s = strip_prefix(label)
    if re.match(r"^(Figure|Fig\.?)\s*\d+", s, re.I):
        return True
    if re.match(r"^SupFig\d+|^Fig\d+[A-Za-z]?", s, re.I):
        return True
    # e.g. "k=5_MAG-1" - component/figure refs, not "area 1" (Brodmann)
    if re.match(r"^[A-Za-z=]+\d+[A-Za-z_-]*\d*$", s) and len(s) < 25 and "area" not in s.lower():
        return True
    return False


def is_subject_id(label: str) -> bool:
    s = strip_prefix(label)
    if re.search(r"Subject\d{2,}", s, re.I):
        return True
    if re.search(r"Study\d+Subject\d+", s, re.I):
        return True
    if re.search(r"sub\d+[_ ](positive|negative|neutral)", s, re.I):
        return True
    return False


def has_trm_id(label: str) -> bool:
    return bool(re.search(r"trm_[a-f0-9]{12,}", label, re.I))


def has_regparam(label: str) -> bool:
    return "regparam" in label.lower() or "parametric contrast" in label.lower()


def is_single_word(label: str) -> bool:
    s = strip_prefix(label)
    # Exclude parenthetical (drug) for pharma
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s)
    return len(s.split()) <= 1 and len(s) > 0


def is_very_short(label: str) -> bool:
    s = strip_prefix(label)
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s)
    return len(s) < 5 and len(s) > 0


def is_junk_placeholder(label: str) -> bool:
    s = strip_prefix(label).lower()
    if re.match(r"^neurovault_image_\d+$", s):
        return True
    if re.match(r"^test\d*$", s):
        return True
    if re.match(r"^collection\s*\d+\s*image\s*\d+$", s):
        return True
    return False


def is_ica_component(label: str) -> bool:
    s = strip_prefix(label)
    return "z-value voxel loadings" in s or bool(re.search(r"IC\d+", s)) or "Experiment filtered IC" in s


def is_movie_code(label: str) -> bool:
    s = strip_prefix(label)
    return bool(re.match(r"^\d{4}_Movie_\d+", s)) or "Movie_" in s


def is_island_jargon(label: str) -> bool:
    s = strip_prefix(label).lower()
    return "island" in s and any(x in s for x in ["south", "north", "east", "west"])


def is_spm_technical(label: str) -> bool:
    s = strip_prefix(label)
    return bool(re.search(r"spmT\s*\d+|UT\d+mU|CT\d+mC", s, re.I))


def is_pe_contrast(label: str) -> bool:
    s = strip_prefix(label)
    return bool(re.match(r"^\d+:\s*PE_", s)) or "PE_plac" in s or "PE_" in s


def categorize(label: str) -> list[str]:
    """Return list of issue categories for this label."""
    cats = []
    if is_figure_reference(label):
        cats.append("figure_reference")
    if is_subject_id(label):
        cats.append("subject_id")
    if has_trm_id(label):
        cats.append("trm_opaque")
    if has_regparam(label):
        cats.append("regparam")
    if is_single_word(label):
        cats.append("single_word")
    if is_very_short(label):
        cats.append("very_short")
    if is_junk_placeholder(label):
        cats.append("junk_placeholder")
    if is_ica_component(label):
        cats.append("ica_component")
    if is_movie_code(label):
        cats.append("movie_code")
    if is_island_jargon(label):
        cats.append("island_jargon")
    if is_spm_technical(label):
        cats.append("spm_technical")
    if is_pe_contrast(label):
        cats.append("pe_contrast")
    if not cats:
        cats.append("ok")
    return cats


# Improvement suggestions per category
IMPROVEMENTS = {
    "figure_reference": "Replace with cognitive concept from paper; or prepend collection/study context",
    "subject_id": "Average by condition; use group-level label",
    "trm_opaque": "Fetch task name from Cognitive Atlas API (improve_neurovault_labels --resolve-trm)",
    "regparam": "Already improved to 'parametric contrast'; add drug from collection (--pharma-add-drug)",
    "single_word": "Consider expanding (e.g. acc to anterior cingulate cortex) or keep if domain-clear",
    "very_short": "Likely abbreviation; expand or verify meaning",
    "junk_placeholder": "DISCARD - no semantic content",
    "ica_component": "Relabel with cognitive interpretation (e.g. Hippocampus network)",
    "movie_code": "Relabel as HCP movie-watching tICA component N (naturalistic viewing)",
    "island_jargon": "Relabel with cognitive condition (IBC location task)",
    "spm_technical": "Replace with contrast description from design",
    "pe_contrast": "Add parametric regressor name/context",
    "ok": "-",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze term labels for quality issues and improvement opportunities")
    parser.add_argument("--cache-dir", default="neurolab/data/merged_sources", help="Cache with term_vocab.pkl, term_sources.pkl")
    parser.add_argument("--model-dir", default="neurolab/data/embedding_model", help="Optional: split_info.pkl for correlation analysis")
    parser.add_argument("--output-csv", default=None, help="Write full analysis to CSV")
    parser.add_argument("--min-correlation", type=float, default=None, help="If model-dir has correlations: only show terms with r below this")
    parser.add_argument("-n", "--examples", type=int, default=5, help="Examples per category (default 5)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    pkl = cache_dir / "term_vocab.pkl"
    if not pkl.exists():
        print(f"term_vocab.pkl not found at {cache_dir}", file=sys.stderr)
        return 1

    terms = pickle.load(open(pkl, "rb"))
    terms = list(terms)
    term_sources = None
    if (cache_dir / "term_sources.pkl").exists():
        term_sources = pickle.load(open(cache_dir / "term_sources.pkl", "rb"))

    # Correlation data (optional)
    corr_by_term = {}
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = REPO_ROOT / model_dir
        split_path = model_dir / "split_info.pkl"
        if split_path.exists():
            with open(split_path, "rb") as f:
                si = pickle.load(f)
            for lst, key in [(si.get("train_term_correlations"), "train"), (si.get("test_term_correlations"), "test")]:
                if lst:
                    for t, r in lst:
                        if r is not None:
                            corr_by_term[t] = (r, key)

    # Categorize all terms
    by_category = defaultdict(list)
    for t in terms:
        cats = categorize(t)
        for c in cats:
            by_category[c].append(t)

    # Report
    print("=" * 80)
    print("TERM LABEL ANALYSIS")
    print("=" * 80)
    print(f"\nCache: {cache_dir}")
    print(f"Total terms: {len(terms)}")
    if term_sources:
        src_counts = defaultdict(int)
        for s in term_sources:
            src_counts[s] += 1
        print("By source:", dict(sorted(src_counts.items(), key=lambda x: -x[1])))
    if corr_by_term:
        print(f"Correlations loaded: {len(corr_by_term)} terms (train+test)")
    print()

    # Categories (exclude "ok" for issue summary)
    issue_cats = [c for c in sorted(by_category.keys()) if c != "ok"]
    n_ok = len(by_category.get("ok", []))
    n_issues = len(terms) - n_ok

    print("SUMMARY BY ISSUE CATEGORY")
    print("-" * 80)
    print(f"{'Category':<25} {'Count':>8} {'%':>6}  Improvement")
    print("-" * 80)
    for cat in sorted(by_category.keys(), key=lambda c: (-len(by_category[c]), c)):
        n = len(by_category[cat])
        pct = 100 * n / len(terms) if terms else 0
        imp = IMPROVEMENTS.get(cat, "—")
        print(f"{cat:<25} {n:>8} {pct:>5.1f}%  {imp[:50]}")
    print("-" * 80)
    print(f"Terms with >=1 issue: {n_issues} ({100*n_issues/len(terms):.1f}%)" if terms else "")
    print()

    # Examples per category
    print("EXAMPLES PER CATEGORY")
    print("-" * 80)
    for cat in issue_cats:
        items = by_category[cat][: args.examples]
        print(f"\n{cat} ({len(by_category[cat])} total):")
        for t in items:
            corr_str = ""
            if t in corr_by_term:
                r, split = corr_by_term[t]
                corr_str = f"  [r={r:.3f} {split}]"
            print(f"  - {t[:75]}{'...' if len(t) > 75 else ''}{corr_str}")

    # Low-correlation terms (if requested)
    if args.min_correlation is not None and corr_by_term:
        print("\n" + "=" * 80)
        print(f"TERMS WITH CORRELATION < {args.min_correlation}")
        print("-" * 80)
        low = [(t, r, split) for t, (r, split) in corr_by_term.items() if r < args.min_correlation]
        low.sort(key=lambda x: x[1])
        for t, r, split in low[:100]:
            cats = categorize(t)
            cat_str = ",".join(cats) if cats else "ok"
            print(f"  r={r:7.4f} [{split}] {cat_str}")
            print(f"    {t[:70]}{'...' if len(t) > 70 else ''}")

    # CSV export
    if args.output_csv:
        out = Path(args.output_csv)
        if not out.is_absolute():
            out = REPO_ROOT / out
        with open(out, "w", encoding="utf-8") as f:
            f.write("term,source,categories,correlation,split\n")
            for i, t in enumerate(terms):
                src = term_sources[i] if term_sources and i < len(term_sources) else ""
                cats = ",".join(categorize(t))
                r, split = corr_by_term.get(t, (None, ""))
                r_str = f"{r:.4f}" if r is not None else ""
                f.write(f'"{t}","{src}","{cats}","{r_str}","{split}"\n')
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
