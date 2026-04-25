# NeuroVault Pharma Cache: Audit and Curated Spec

**Date:** 2026-02-24  
**Question:** Is the neurovault_pharma cache polluted with non-pharma data, or is pharmacological information simply missing from labels?

---

## Verdict: **Pollution (legacy) — now fixed with curated spec**

The legacy `neurovault_pharma_data` (from `download_neurovault_pharma.py`) contained **non-pharmacological collections** (e.g. 1257 NeuroSynth Encoding). That path is **deprecated**.

**Current approach:** Pharma cache is built from **neurovault_curated_data** (curated collections from `download_neurovault_curated.py --include-pharma`), filtered to a curated pharma list, with schema-based relabeling. See `neurolab/data/neurovault_pharma_schema.json`.

---

## Curated Pharma Collection List

Source: `download_neurovault_curated.py` PHARMA + `neurovault_pharma_schema.json`.

### A) Drug challenge (drug vs placebo)

| Collection | Drug(s) | Paradigm |
|------------|---------|----------|
| 1083 | LSD vs placebo | RSFC seeds + CBF (rest) |
| 12212 | ketamine vs saline | resting-state thalamic dysconnectivity |
| 17403 | ketamine vs placebo | affective touch task |
| 8306 | methylphenidate / sulpiride / placebo | Ki→behavior regression |
| 3902 | haloperidol vs placebo | picture-processing task |
| 4040, 4041 | d-amphetamine vs placebo | PET [18F]fallypride BPND |
| 9246 | ibuprofen / PPARγ vs placebo | emotional faces |
| 5488, 3666, 3808, 13312 | oxytocin vs placebo | amygdala FC, faces, rest, speech |
| 13665 | L-DOPA / oxytocin / placebo | reinforcement learning RL |
| 4414 | alcohol vs placebo | emotional faces |

### B) Task-on-drug (null drug effect)

| Collection | Manipulation | Paradigm |
|------------|--------------|----------|
| 1186 | sulpiride (null; pooled) | slot-machine gambling |
| 12992 | nicotine abstinence vs SAU | threat anticipation |

### C) Placebo / expectancy

| Collection | Manipulation | Paradigm |
|------------|--------------|----------|
| 9206 | placebo analgesia (opioidergic) | empathy-for-pain |
| 9244 | placebo gel analgesia | empathy for pain |
| 20308 | cue-based vs placebo expectancy | pain modulation |

### D) Meta-analyses (keep separate)

| Collection | Type | Domain |
|------------|------|--------|
| 3713 | ALE meta | oxytocin vs placebo |
| 1501 | SDM meta | addiction reward |
| 19291 | MKDA meta | psychedelics+ketamine |

### E) Excluded from pharma

| Collection | Reason |
|------------|--------|
| **3264** | Control task battery (not drug manipulation); keep as `task_control_battery` |

---

## Label Schema (prevents task collapse)

Every pharma label must include at least: **drug + control/state + measure**.

Canonical format: `drug=<…> | control=<…> | measure=<…> | task=<…> | contrast=<…> | group=<…>`

**Do-not-pollute rules:**
- Separate sources: drug_challenge, task_on_drug_null, placebo_expectancy, meta_analysis, task_control_battery
- Hard-exclude 3264 from pharma
- For collections with metadata gaps (3713, 3666): drop/flag images missing mandatory metadata
- Require every pharma label to include at least: drug + control/state + measure

**Script:** `relabel_pharma_terms.py` applies schema prefixes from `label_prefix_by_collection` to the pharma cache.

---

## Pipeline

1. **Download:** `download_neurovault_curated.py --all` (includes curated pharma collections)
2. **Build pharma cache:** `build_neurovault_cache.py --data-dir neurovault_curated_data --collections <pharma_ids> --pharma-add-drug --cluster-by-description`
3. **Improve labels:** `improve_neurovault_labels.py` (strip prefix, etc.)
4. **Schema relabel:** `relabel_pharma_terms.py` (prepend drug/control/measure prefixes)
5. **Merge:** Uses `neurovault_pharma_cache_relabeled` in merged_sources

---

## Legacy (deprecated)

- **neurovault_pharma_data** from `download_neurovault_pharma.py` — keyword search returned polluted collections (e.g. 1257). Do not use.
- **neurovault_pharma_cache** built from neurovault_pharma_data — deprecated.

---

## Recommendations

1. **Use curated data:** Pharma comes from `neurovault_curated_data` with `--include-pharma` or `--all`.
2. **Run schema relabeling:** Always run `relabel_pharma_terms.py` after `improve_neurovault_labels.py` for pharma.
3. **Exclude 3264 from pharma:** It is a control task battery, not a drug manipulation.
4. **Add new collections:** Update `neurovault_pharma_schema.json` and `download_neurovault_curated.py` PHARMA list together.
