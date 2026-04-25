#!/usr/bin/env python3
"""
Batch LLM relabeling for NeuroVault cache terms. Fixes junk labels before generalizer training.

Pipeline:
  1. Load term_vocab, term_maps, term_collection_ids from cache
  2. Categorize: JUNK (discard), RELABEL (send to LLM), KEEP (as-is)
  3. For RELABEL: batch call OpenAI API to produce short cognitive labels
  4. Output: updated cache (filtered + relabeled) and relabel_diff.json for review

Usage:
  python neurolab/scripts/relabel_terms_with_llm.py --cache-dir neurolab/data/neurovault_cache
  python neurolab/scripts/relabel_terms_with_llm.py --cache-dir neurolab/data/neurovault_pharma_cache --dry-run
  python neurolab/scripts/relabel_terms_with_llm.py --cache-dir neurolab/data/neurovault_cache --output-dir neurolab/data/neurovault_cache_relabeled
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
API_BASE = "https://neurovault.org/api"
HEADERS = {"Accept": "application/json", "User-Agent": "NeuroLab/1.0"}

# Modality prefixes to preserve when relabeling
MODALITY_PREFIXES = ("fMRI:", "Pharmacological fMRI:", "PET:", "Structural:", "Gene:", "Cognitive:", "Perfusion:", "DTI:")

# Categories that cause DISCARD (no semantic content; exclude from output)
DISCARD_CATEGORIES = frozenset({
    "figure_reference", "subject_id", "junk_placeholder",
    "island_jargon", "spm_technical", "pe_contrast", "trm_opaque",
})

# Categories that trigger RELABEL (LLM fixes vague/technical labels)
# ica_component, movie_code: have extractable/interpretable content (e.g. IC14 Hippocampus -> Hippocampus network)
RELABEL_CATEGORIES = frozenset({
    "single_word", "very_short", "regparam", "ica_component", "movie_code",
})


def _get_prefix(label: str) -> tuple[str, str]:
    """Return (prefix, rest). Prefix is empty if none."""
    for p in MODALITY_PREFIXES:
        if label.startswith(p):
            return p, label[len(p):].strip()
    return "", label.strip()


def _has_contrast_syntax(label: str) -> bool:
    """True if label has contrast syntax (vs, minus, PE_) that encoders parse poorly."""
    s = label
    for p in MODALITY_PREFIXES:
        if label.startswith(p):
            s = label[len(p):]
            break
    s = s.strip().lower()
    if " vs " in s or " vs. " in s or " minus " in s:
        return True
    if "pe_plac" in s or re.search(r"\bpe_\w+", s):
        return True
    return False


def _categorize(label: str) -> list[str]:
    """Import and use categorize from analyze_term_labels."""
    _scripts = Path(__file__).resolve().parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    import analyze_term_labels as atl
    return atl.categorize(label)


def _decide_action(label: str) -> tuple[str, list[str]]:
    """Return (action, categories): 'discard' | 'relabel' | 'keep'."""
    cats = _categorize(label)
    for c in cats:
        if c in DISCARD_CATEGORIES:
            return "discard", cats
    for c in cats:
        if c in RELABEL_CATEGORIES:
            return "relabel", cats
    if cats == ["ok"] and _has_contrast_syntax(label):
        return "relabel", cats + ["contrast_syntax"]
    return "keep", cats


def _fetch_collection_metadata(cid: int, timeout: float = 15.0) -> dict:
    """Fetch collection name and description from NeuroVault API."""
    try:
        req = urllib.request.Request(f"{API_BASE}/collections/{cid}/", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def _relabel_batch_openai(
    items: list[tuple[str, str, str, str, bool]],
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
) -> list[str]:
    """Call OpenAI API to relabel terms. items = [(label, cname, cdesc, map_kind, dose_related), ...]. Returns list of new labels (same order)."""
    from openai import OpenAI
    client = OpenAI()
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        lines = []
        for j, (label, cname, cdesc, map_kind, dose_related) in enumerate(batch):
            ctx = f"Collection: {cname or 'N/A'}"
            if (cdesc or "").strip():
                ctx += f"\nDescription: {(cdesc or '')[:800]}"
            ctx += f"\nMap type: {map_kind or 'activation'}"
            if dose_related:
                ctx += " (dose/placebo study)"
            lines.append(f"[{j+1}]\n{ctx}\nLabel: {label}\n")
        # Map-kind-aware rules
        activation_rules = (
            "For ACTIVATION maps: PRESERVE direction when present (e.g. 'A > B', 'A minus B', 'reward > control'). "
            "Use '>' or 'minus' for signed contrasts; do not convert to 'vs'."
        )
        p_map_rules = (
            "For P-MAP (significance) maps: use UNSIGNED labels only. Convert 'A - B' or 'A > B' to 'A vs B (significance)'. "
            "No direction; these maps show where effects are significant, not which direction."
        )
        prompt = (
            "You are helping clean fMRI contrast labels for a brain activation prediction model. "
            "For each item below, use the collection name and description (when available) to produce a SHORT natural-language cognitive description (2-12 words). "
            "When the label is vague (e.g. regparam, main experiment, parametric contrast), use collection context to infer drug, dose, task, or design. "
            "For dose/placebo studies: include drug and dose when known (e.g. '200 mg ibuprofen emotion task', 'placebo-controlled reward'). "
            f"{activation_rules} "
            f"{p_map_rules} "
            "Rules: no figure/table refs; no subject IDs; no technical jargon (spmT, regparam, PE_). "
            "Output ONLY the new label, one per line, in the same order. No numbering or extra text.\n\n"
            + "\n".join(lines)
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Handle possible numbering (1. foo, 2. bar)
        cleaned = []
        for ln in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)$", ln)
            cleaned.append(m.group(1).strip() if m else ln)
        orig_labels = [item[0] for item in batch]
        if len(cleaned) < len(batch):
            # Pad with original if API returned fewer
            cleaned.extend(orig_labels[len(cleaned):])
        elif len(cleaned) > len(batch):
            cleaned = cleaned[: len(batch)]
        results.extend(cleaned)
        time.sleep(0.2)  # rate limit
    return results


def main() -> int:
    # Load .env so OPENAI_API_KEY is available when set there
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Batch LLM relabeling for NeuroVault cache terms")
    parser.add_argument("--cache-dir", default="neurolab/data/neurovault_cache", help="Cache with term_vocab.pkl, term_maps.npz")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: overwrite cache-dir)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without API calls or writes")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model (use gpt-4.1-mini for stronger biomedical summarization)")
    parser.add_argument("--batch-size", type=int, default=20, help="Terms per LLM request")
    parser.add_argument("--no-discard", action="store_true", help="Do not discard junk terms; only relabel")
    parser.add_argument("--relabel-all", action="store_true", help="Send ALL non-discarded terms to LLM for curation (default: only vague/bad labels)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    pkl_path = cache_dir / "term_vocab.pkl"
    npz_path = cache_dir / "term_maps.npz"
    if not pkl_path.exists() or not npz_path.exists():
        print(f"Cache not found: {pkl_path} or {npz_path}", file=sys.stderr)
        return 1

    with open(pkl_path, "rb") as f:
        terms = list(pickle.load(f))
    term_maps = np.load(npz_path)["term_maps"]
    cids = []
    if (cache_dir / "term_collection_ids.pkl").exists():
        cids = list(pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb")))
    else:
        cids = [0] * len(terms)
    weights = list(pickle.load(open(cache_dir / "term_sample_weights.pkl", "rb"))) if (cache_dir / "term_sample_weights.pkl").exists() else [1.0] * len(terms)

    assert len(terms) == term_maps.shape[0] == len(cids) == len(weights)

    # Categorize and decide actions
    actions = []
    categories_by_idx = []
    for lab in terms:
        act, cats = _decide_action(lab)
        if args.no_discard and act == "discard":
            act = "relabel"
        if args.relabel_all and act == "keep":
            act = "relabel"
        actions.append(act)
        categories_by_idx.append(cats)

    n_discard = sum(1 for a in actions if a == "discard")
    n_relabel = sum(1 for a in actions if a == "relabel")
    n_keep = sum(1 for a in actions if a == "keep")

    print(f"Loaded {len(terms)} terms from {cache_dir}")
    print(f"  discard: {n_discard}, relabel: {n_relabel}, keep: {n_keep}")

    if args.dry_run:
        print("\n[DRY RUN] Sample discard:")
        n = 0
        for lab, act in zip(terms, actions):
            if act == "discard":
                print(f"  {lab[:85]}{'...' if len(lab) > 85 else ''}")
                n += 1
                if n >= 10:
                    break
        print("\n[DRY RUN] Sample relabel:")
        n = 0
        for lab, act in zip(terms, actions):
            if act == "relabel":
                print(f"  {lab[:85]}{'...' if len(lab) > 85 else ''}")
                n += 1
                if n >= 10:
                    break
        return 0

    # Relabel via LLM (with collection context for dose/drug-aware labels)
    relabel_indices = [i for i, a in enumerate(actions) if a == "relabel"]
    new_labels_map = {}
    if relabel_indices:
        collection_metadata: dict[str, dict] = {}
        if (cache_dir / "collection_metadata.json").exists():
            collection_metadata = json.loads((cache_dir / "collection_metadata.json").read_text())
        to_send = []
        for i in relabel_indices:
            prefix, rest = _get_prefix(terms[i])
            label = rest if rest else terms[i]
            cid = cids[i] if i < len(cids) else 0
            meta = collection_metadata.get(str(cid))
            if not meta and cid:
                meta = _fetch_collection_metadata(cid)
                if meta:
                    collection_metadata[str(cid)] = {"name": meta.get("name"), "description": meta.get("description")}
                time.sleep(0.15)
            cname = (meta.get("name") or "").strip() if meta else ""
            cdesc = (meta.get("description") or "").strip() if meta else ""
            map_kind = (meta.get("map_kind") or "activation").strip() if meta else "activation"
            dose_related = bool(meta.get("dose_related")) if meta else False
            to_send.append((label, cname, cdesc, map_kind, dose_related))
        if os.environ.get("OPENAI_API_KEY"):
            print(f"Calling OpenAI for {len(to_send)} terms (with collection context)...")
            raw_new = _relabel_batch_openai(to_send, model=args.model, batch_size=args.batch_size)
            for idx, new in zip(relabel_indices, raw_new):
                prefix, _ = _get_prefix(terms[idx])
                new_labels_map[idx] = f"{prefix}{new}".strip() if prefix else new.strip()
        else:
            print("OPENAI_API_KEY not set; skipping LLM relabeling (relabel terms will keep original)", file=sys.stderr)
            for i in relabel_indices:
                new_labels_map[i] = terms[i]

    # Build output
    keep_mask = [a != "discard" for a in actions]
    new_terms = []
    new_maps = []
    new_cids = []
    new_weights = []
    diff_log = []

    for i in range(len(terms)):
        if not keep_mask[i]:
            diff_log.append({"action": "discard", "original": terms[i], "categories": categories_by_idx[i]})
            continue
        orig = terms[i]
        new = new_labels_map.get(i, orig)
        new_terms.append(new)
        new_maps.append(term_maps[i])
        new_cids.append(cids[i])
        new_weights.append(weights[i])
        if new != orig:
            diff_log.append({"action": "relabel", "original": orig, "new": new, "categories": categories_by_idx[i]})

    # Ensure uniqueness (duplicates get suffix)
    seen = {}
    for j in range(len(new_terms)):
        t = new_terms[j]
        if t in seen:
            k = 1
            while f"{t} ({k})" in seen:
                k += 1
            new_terms[j] = f"{t} ({k})"
        seen[new_terms[j]] = j

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "term_maps.npz", term_maps=np.array(new_maps))
    with open(output_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(new_terms, f)
    with open(output_dir / "term_collection_ids.pkl", "wb") as f:
        pickle.dump(new_cids, f)
    with open(output_dir / "term_sample_weights.pkl", "wb") as f:
        pickle.dump(new_weights, f)
    with open(output_dir / "relabel_diff.json", "w") as f:
        json.dump(diff_log, f, indent=2)

    print(f"Saved {len(new_terms)} terms to {output_dir}")
    print(f"  discarded {n_discard}, relabeled {len(new_labels_map)}, kept {n_keep - (n_relabel - len(new_labels_map))}")
    print(f"  diff log: {output_dir / 'relabel_diff.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
