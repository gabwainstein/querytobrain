#!/usr/bin/env python3
"""
Report NeuroVault collection status with stage-explicit definitions and reconciled accounting.

Definitions (all relative to current cache state):
  - terms_in_cache (N): Terms currently in term_vocab.pkl (after build, after improve_neurovault_labels).
  - in cache: Collections that contributed ≥1 term to term_vocab.pkl. Not "on disk" or "processed"—only those with output in the final cache.
  - discard/relabel/keep: What relabel_terms_with_llm would do to each term (pre-relabel pipeline).
  - eligible_for_averaging: cid in AVERAGE_FIRST (neurovault_ingestion).
  - actually_averaged: From collection_provenance.json (build-time) when present; else inferred from
    no subject_id pattern in terms. Prefer provenance over heuristic.

Discard subtypes (for prioritization):
  - discard_subject_id: Fixable via average_neurovault_cache or build --average-subject-level.
  - discard_figure_ref: Figure 1A, Fig3, etc. — mostly non-fixable.
  - discard_spmT: spmT 0001, CS2mCT2 — technical, non-fixable.
  - discard_placeholder: neurovault_image_N, test — discard.
  - discard_other: island_jargon, pe_contrast, trm_opaque.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "neurolab" / "scripts"))

import analyze_term_labels as atl
from neurolab.neurovault_ingestion import AVERAGE_FIRST

# Curated list: tiers 1-4 + WM atlas + pharma (matches download_neurovault_curated --all, no slugs)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dvc", REPO_ROOT / "neurolab" / "scripts" / "download_neurovault_curated.py"
)
_dvc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dvc)
TIER_1 = _dvc.TIER_1
TIER_2 = _dvc.TIER_2
TIER_3 = _dvc.TIER_3
TIER_4 = _dvc.TIER_4
WM_ATLAS = _dvc.WM_ATLAS
PHARMA = _dvc.PHARMA

ATLAS_COLLECTION_IDS = {262, 264, 1625, 6074, 9357}
CURATED_ALL = list(dict.fromkeys(TIER_1 + TIER_2 + TIER_3 + TIER_4 + WM_ATLAS + PHARMA))

# Short descriptions for collections in cache (from acquisition guide)
COLLECTION_NAMES = {
    457: "HCP group-level task maps",
    503: "PINES/IAPS emotion",
    504: "Pain NPS intensity",
    63: "Test-retest motor/language/attention",
    109: "Structure-function connectome",
    315: "Adaptive learning",
    426: "False belief ToM",
    445: "Why/How ToM",
    507: "Consensus decision-making",
    550: "7T subcortical variability",
    555: "Reward obesity/addiction meta",
    834: "Anterior midcingulate motor",
    839: "Vigilant attention meta",
    844: "Working memory meta",
    825: "DLPFC co-activation",
    833: "Motor learning meta",
    830: "Vestibular cortex meta",
    1057: "Yeo 7/17 networks",
    1206: "Serotonin PET normative",
    1425: "Pain NIDM-Results",
    1432: "Pain IBMA",
    1501: "Addiction reward SDM",
    1541: "Emotional contagion + sleep",
    1516: "Visual attention",
    1598: "Margulies gradient",
    1620: "Depression resting-state",
    1625: "Brainnetome atlas (excl)",
    16266: "Emotion regulation system ID",
    16284: "IAPS valence",
    17228: "NeuroT-Map supplementary",
    18197: "Peraza IBMA",
    19012: "Incentive-boosted inhibitory",
    20036: "Source vs item memory",
    20510: "Psychosis circuit",
    20820: "HCP-YA task-evoked network",
    2108: "Meaningful vs meaningless sentences",
    2138: "IBC 1st release",
    2462: "Social brain connectome",
    2485: "VTA/SN resting-state",
    2503: "Social Bayesian",
    2508: "Pharma fMRI",
    262: "Harvard-Oxford (excl)",
    2621: "WM face load",
    264: "JHU DTI (excl)",
    2884: "Anxious WM",
    2981: "Blood pressure VBM",
    3085: "2-back vs 0-back",
    3145: "Subcortical nuclei atlas",
    3158: "Meaningful inhibition Go/NoGo",
    3192: "Context-dependent WM",
    3245: "Extended amygdala connectivity",
    3264: "Pharmacological",
    3324: "Kragel pain/cog/emotion",
    3340: "Reward learning",
    3434: "UK Biobank DMN",
    3713: "Emotion OXT/PBO",
    3822: "BrainMap VBM",
    3884: "MDD reward meta",
    3887: "tMapNest",
    3902: "Haloperidol vs placebo",
    3960: "Striatal reward youth",
    4040: "Placebo vs dAMPH",
    4041: "Placebo vs dAMPH (alt)",
    4146: "Sleep restriction emotion reg",
    4343: "UCLA LA5C",
    4683: "Resting state",
    4804: "Fusiform-network",
    5070: "Reward clustering",
    5377: "Inhibition imitation meta",
    5488: "Schizophrenia oxytocin",
    5623: "Visual WM searchlight",
    5662: "FTD/AD multimodal",
    5673: "Memory integration",
    5943: "PTSD memory meta",
    6009: "Response inhibition",
    6047: "NARPS overlap",
    6051: "NARPS IBMA",
    6074: "Brainstem tract (excl)",
    6088: "Episodic memory replay",
    6126: "Hippocampal pain",
    6221: "Fear conditioning",
    6237: "Acute vs sustained fear",
    6262: "DMN parcellation",
    6618: "IBC 2nd release",
    6825: "Schizophrenia deformation",
    7114: "Child WM tract atlas",
    7793: "Social reward meta",
    8076: "Extended amygdala HCP",
    8306: "Proportion redo MPH/SUL",
    8448: "Executive function networks",
    8461: "Structural",
    8676: "Cued reward omission",
    9244: "Placebo analgesia",
    9246: "PPARgamma/ibuprofen",
    9357: "APOE structural (excl)",
    10410: "Pain value signature",
    1083: "LSD RSFC",
    11343: "GAD/FAD/MDD VBM",
    11584: "Finger tapping",
    11646: "ASD emotional egocentricity",
    1186: "Near-miss outcomes",
    12212: "Ketamine thalamic",
    12480: "DLPFC stimulation reward",
    12874: "Instructions vs experience pain",
    12992: "Nicotine abstinence",
    13042: "Language/WM epilepsy",
    13474: "Tensorial ICA MDD",
    13656: "Response time paradox",
    13665: "Placebo/other pharma",
    13705: "Tier3 language",
    13924: "Facial expression pain",
    15030: "Physical vs vicarious pain",
    15237: "NeuroT-Map neurotransmitter",
    15274: "Threat anticipation",
    15965: "Math anxiety ALE",
    1952: "BrainPedia",
    457: "HCP group-level",
    1274: "Cognitive Atlas NeuroSynth",
    7756: "WM function atlas (acoustic)",
    7757: "WM function atlas (decision)",
    7758: "WM function atlas (identification)",
    7759: "WM function atlas (object)",
    7760: "WM function atlas (saccade)",
    7761: "WM function atlas (valence)",
}


def _decide_action_and_subtype(label: str) -> tuple[str, str | None]:
    """Return (action, discard_subtype). discard_subtype only set when action=discard.
    Prefer subject_id over figure_reference (fixable over non-fixable) when both match."""
    cats = atl.categorize(label)
    discard_cats = [c for c in cats if c in {"figure_reference", "subject_id", "junk_placeholder", "island_jargon", "spm_technical", "pe_contrast", "trm_opaque"}]
    if discard_cats:
        # Prefer subject_id (fixable) over figure_reference when both match (e.g. Study01Subject01)
        priority = ["subject_id", "junk_placeholder", "spm_technical", "pe_contrast", "figure_reference", "island_jargon", "trm_opaque"]
        for p in priority:
            if p in discard_cats:
                subtype = {
                    "subject_id": "discard_subject_id",
                    "figure_reference": "discard_figure_ref",
                    "spm_technical": "discard_spmT",
                    "pe_contrast": "discard_pe_contrast",
                    "junk_placeholder": "discard_placeholder",
                    "island_jargon": "discard_island_jargon",
                    "trm_opaque": "discard_trm_opaque",
                }[p]
                return "discard", subtype
        return "discard", "discard_other"
    for c in cats:
        if c in {"single_word", "very_short", "regparam", "ica_component", "movie_code"}:
            return "relabel", None
    if cats == ["ok"] and (" vs " in label or " minus " in label.lower() or "pe_" in label.lower()):
        return "relabel", None
    return "keep", None


def _has_subject_id_pattern(terms: list[tuple[str, str]]) -> bool:
    """True if any term has subject_id category (Study01Subject01, PIP Subject0001, etc.)."""
    for t, act in terms:
        cats = atl.categorize(t)
        if "subject_id" in cats:
            return True
    return False


def main() -> None:
    cache_dir = REPO_ROOT / "neurolab" / "data" / "neurovault_cache"
    manifest_path = REPO_ROOT / "neurolab" / "data" / "neurovault_curated_data" / "manifest.json"

    if not (cache_dir / "term_vocab.pkl").exists():
        print("Cache not found:", cache_dir / "term_vocab.pkl", file=sys.stderr)
        sys.exit(1)

    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    cids = pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb"))

    # Raw image counts from manifest (if available)
    raw_by_col: dict[int, int] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for img in manifest.get("images", []):
            cid = img.get("collection_id")
            if cid:
                raw_by_col[cid] = raw_by_col.get(cid, 0) + 1

    by_col = defaultdict(lambda: {
        "terms": [], "discard": 0, "relabel": 0, "keep": 0,
        "discard_subject_id": 0, "discard_figure_ref": 0, "discard_spmT": 0,
        "discard_placeholder": 0, "discard_other": 0,
    })
    for t, cid in zip(terms, cids):
        act, subtype = _decide_action_and_subtype(t)
        by_col[cid]["terms"].append((t, act))
        if act == "discard":
            by_col[cid]["discard"] += 1
            if subtype:
                by_col[cid][subtype] = by_col[cid].get(subtype, 0) + 1
        elif act == "relabel":
            by_col[cid]["relabel"] += 1
        else:
            by_col[cid]["keep"] += 1

    in_cache = set(by_col.keys())
    curated_set = set(CURATED_ALL)
    atlas_set = ATLAS_COLLECTION_IDS
    missing = curated_set - atlas_set - in_cache
    excluded_atlas = curated_set & atlas_set

    # actually_averaged: from collection_provenance.json when available; else heuristic
    avg_eligible = AVERAGE_FIRST
    avg_in_cache = in_cache & avg_eligible
    provenance_path = cache_dir / "collection_provenance.json"
    if provenance_path.exists():
        prov = json.loads(provenance_path.read_text(encoding="utf-8"))
        avg_actually = {cid for cid in avg_in_cache if prov.get(str(cid), {}).get("was_averaged", False)}
    else:
        avg_actually = {cid for cid in avg_in_cache if not _has_subject_id_pattern(by_col[cid]["terms"])}

    # Sanity assertions
    placeholders_remaining = sum(1 for t in terms if atl.is_junk_placeholder(t))
    study_subject_in_avg = 0
    for cid in avg_actually:
        for t, _ in by_col[cid]["terms"]:
            if "Study" in t and "Subject" in t:
                study_subject_in_avg += 1
                break
    collections_zero_final = [c for c in in_cache if by_col[c]["discard"] == len(by_col[c]["terms"]) and len(by_col[c]["terms"]) > 0]

    # Output
    out_lines = [
        "# NeuroVault Collection Status Report",
        "",
        "**Generated by:** `python neurolab/scripts/report_collection_status.py`",
        "",
        "## Definitions",
        "",
        "| Term | Meaning |",
        "|------|---------|",
        "| `terms_in_cache` (N) | Terms in term_vocab.pkl (after build, after improve_neurovault_labels) |",
        "| `in cache` | Collections that contributed ≥1 term to term_vocab.pkl. Not \"on disk\" or \"processed\"—only those that have output in the final cache. |",
        "| `discard` / `relabel` / `keep` | What relabel_terms_with_llm would do to each term (pre-relabel) |",
        "| `eligible_for_averaging` | cid in AVERAGE_FIRST |",
        "| `actually_averaged` | From collection_provenance.json (build-time); else inferred from labels |",
        "",
        "## Accounting",
        "",
        "| Stage | Count |",
        "|-------|------:|",
        f"| Curated list (tiers 1–4 + WM + pharma) | {len(CURATED_ALL)} |",
        f"| Excluded at build (atlas) | {len(excluded_atlas)} |",
        f"| In cache | {len(in_cache)} |",
        f"| Missing (curated - atlas - in_cache) | {len(missing)} |",
        "",
        "**In cache** = collections that contributed ≥1 term to term_vocab.pkl (passed build + improve and have output). Not \"on disk\" or \"processed\"—only those with terms in the final cache.",
        "",
        "**Excluded at build (atlas IDs):** " + ", ".join(str(c) for c in sorted(excluded_atlas)),
        "",
        "## AVERAGE_FIRST (from neurovault_ingestion)",
        "",
        f"**Count:** {len(AVERAGE_FIRST)}",
        "",
        f"**IDs:** {sorted(AVERAGE_FIRST)}",
        "",
        "| In cache | Eligible | Actually averaged |",
        "|----------|----------|-------------------|",
        f"| {len(avg_in_cache)} | {len(avg_eligible)} | {len(avg_actually)} |",
        "",
        "**Actually averaged:** " + (", ".join(str(c) for c in sorted(avg_actually)) if avg_actually else "(none)"),
        "",
        "**Eligible but NOT averaged (subject_id pattern in terms):** " + (
            ", ".join(str(c) for c in sorted(avg_in_cache - avg_actually)) if (avg_in_cache - avg_actually) else "(none)"
        ),
        "",
        "## Missing / Not in cache",
        "",
        "| ID | In curated? | Status |",
        "|----|:-----------:|--------|",
    ]
    # Refined missing status (from NEUROVAULT_CURATED_CACHE_FORMATION diagnostics)
    AVG_DROPPED = {19012, 6825, 20510, 1620, 2503, 13474, 13705, 3887}
    QC_FAILED = {555, 2462}
    EMPTY_ON_NEUROVAULT = {109, 15965}  # number_of_images=0 on API; cannot download
    for cid in sorted(missing):
        raw = raw_by_col.get(cid, 0)
        if raw == 0:
            status = "not_downloaded" + (" (empty on NeuroVault)" if cid in EMPTY_ON_NEUROVAULT else "")
        elif cid in QC_FAILED:
            status = "qc_failed (std/zeros)"
        elif cid in AVG_DROPPED or (cid in AVERAGE_FIRST and raw > 0):
            status = "averaging_dropped (min_subjects)"
        else:
            status = "parcellation_failed_or_qc"
        out_lines.append(f"| {cid} | Yes | {status} |")
    if not missing:
        out_lines.append("| (none) | — | — |")

    out_lines.extend([
        "",
        "## High discard by subtype",
        "",
        "| Subtype | Fixable? | Collections (>=5) |",
        "|---------|----------|-------------------|",
        "| discard_subject_id | Yes (run average_neurovault_cache) | |",
        "| discard_figure_ref | No | |",
        "| discard_spmT | No | |",
        "| discard_placeholder | No | |",
    ])

    # Subtype breakdown
    high_subject_id = [(c, by_col[c]) for c in by_col if by_col[c]["discard_subject_id"] >= 5]
    high_figure_ref = [(c, by_col[c]) for c in by_col if by_col[c]["discard_figure_ref"] >= 5]
    high_spmT = [(c, by_col[c]) for c in by_col if by_col[c]["discard_spmT"] >= 5]
    high_placeholder = [(c, by_col[c]) for c in by_col if by_col[c]["discard_placeholder"] >= 5]

    out_lines.append("")
    if high_subject_id:
        ids = ", ".join(str(c) for c, _ in sorted(high_subject_id, key=lambda x: -x[1]["discard_subject_id"]))
        out_lines.append("**discard_subject_id (fixable):** " + ids)
        for cid, _ in sorted(high_subject_id, key=lambda x: -x[1]["discard_subject_id"]):
            fix = "run `average_neurovault_cache.py`" if cid in AVERAGE_FIRST else "add to AVERAGE_FIRST + get_cognitive_condition"
            out_lines.append(f"- {cid}: {fix}")
    else:
        out_lines.append("**discard_subject_id:** (none)")
    out_lines.append("")
    out_lines.append("**discard_figure_ref:** " + ", ".join(str(c) for c, _ in sorted(high_figure_ref, key=lambda x: -x[1]["discard_figure_ref"])) if high_figure_ref else "**discard_figure_ref:** (none)")
    out_lines.append("")
    out_lines.append("**discard_spmT:** " + ", ".join(str(c) for c, _ in sorted(high_spmT, key=lambda x: -x[1]["discard_spmT"])) if high_spmT else "**discard_spmT:** (none)")
    out_lines.append("")
    out_lines.append("**discard_placeholder:** " + ", ".join(str(c) for c, _ in sorted(high_placeholder, key=lambda x: -x[1]["discard_placeholder"])) if high_placeholder else "**discard_placeholder:** (none)")

    # High relabel
    high_relabel = [(c, by_col[c]) for c in by_col if by_col[c]["relabel"] >= 5 and by_col[c]["relabel"] > by_col[c]["keep"]]
    out_lines.extend([
        "",
        "## High relabel (>=5 terms, relabel > keep)",
        "",
    ])
    for cid, d in sorted(high_relabel, key=lambda x: -x[1]["relabel"])[:12]:
        ex = next((t for t, a in d["terms"] if a == "relabel"), "")
        out_lines.append(f"- **{cid}:** {d['relabel']} relabel — e.g. `{ex[:55]}...`" if len(ex) > 55 else f"- **{cid}:** {d['relabel']} relabel — e.g. `{ex}`")

    # Collections in cache: description + term example
    out_lines.extend([
        "",
        "## Collections in cache (description + term example)",
        "",
        "| ID | Description | Term example |",
        "|----|--------------|---------------|",
    ])
    for cid in sorted(by_col.keys(), key=lambda c: -len(by_col[c]["terms"])):
        desc = COLLECTION_NAMES.get(cid, f"col{cid}")
        ex = by_col[cid]["terms"][0][0] if by_col[cid]["terms"] else ""
        ex_short = (ex[:60] + "...") if len(ex) > 60 else ex
        out_lines.append(f"| {cid} | {desc} | `{ex_short}` |")

    # Full table
    out_lines.extend([
        "",
        "## Full collection list (in cache)",
        "",
        "| ID | N | discard | relabel | keep | d_subj | d_fig | d_spmT | d_ph | avg_ok? | avg_done? |",
        "|----|---|--------|---------|------|--------|-------|--------|------|---------|-----------|",
    ])
    for cid in sorted(by_col.keys(), key=lambda c: -len(by_col[c]["terms"])):
        d = by_col[cid]
        n = len(d["terms"])
        avg_ok = "Y" if cid in AVERAGE_FIRST else ""
        avg_done = "Y" if cid in avg_actually else ""
        out_lines.append(
            f"| {cid} | {n} | {d['discard']} | {d['relabel']} | {d['keep']} | "
            f"{d['discard_subject_id']} | {d['discard_figure_ref']} | {d['discard_spmT']} | {d['discard_placeholder']} | "
            f"{avg_ok} | {avg_done} |"
        )

    # Sanity assertions
    out_lines.extend([
        "",
        "## Sanity assertions",
        "",
        "| Assertion | Value |",
        "|-----------|------:|",
        f"| placeholders_remaining in final vocab | {placeholders_remaining} (expect 0) |",
        f"| Study??Subject?? in actually_averaged collections | {study_subject_in_avg} (expect 0) |",
        f"| Collections where relabeler would discard all current labels | {len(collections_zero_final)}: {collections_zero_final[:10]}{'...' if len(collections_zero_final) > 10 else ''} |",
        "",
    ])

    out_path = REPO_ROOT / "neurolab" / "docs" / "implementation" / "NEUROVAULT_COLLECTION_FULL_REPORT.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print("Wrote:", out_path)

    # Also print summary to stdout
    print("\n--- Summary ---")
    print(f"Curated: {len(CURATED_ALL)}, Atlas excluded: {len(excluded_atlas)}, In cache: {len(in_cache)}, Missing: {len(missing)}")
    print(f"AVERAGE_FIRST: {len(AVERAGE_FIRST)} eligible, {len(avg_actually)} actually averaged")
    print(f"Eligible but not averaged: {sorted(avg_in_cache - avg_actually)}")
    print(f"High discard_subject_id (fixable): {[c for c, _ in high_subject_id]}")
    print(f"placeholders_remaining: {placeholders_remaining}")


if __name__ == "__main__":
    main()
