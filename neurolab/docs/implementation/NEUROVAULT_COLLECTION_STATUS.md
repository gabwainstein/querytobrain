# NeuroVault Collection Status

**Source:** [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md), [NEUROVAULT_TRAINING_SET_INGESTION_ALGORITHM.md](NEUROVAULT_TRAINING_SET_INGESTION_ALGORITHM.md).

---

## Pipeline (per docs)

1. **Download:** `download_neurovault_curated.py --all` → **126 collections**, ~29K images in `neurovault_curated_data/`
2. **Build:** `build_neurovault_cache.py ... --average-subject-level` → parcellates all, averages AVERAGE_FIRST by contrast
3. **Output:** ~2,700–5,000 neurovault maps → merged into `merged_sources` (~14–15K total with NQ, NS, neuromaps, enigma, abagen, etc.)

**Recommended command:**
```bash
python neurolab/scripts/build_neurovault_cache.py \
  --data-dir neurolab/data/neurovault_curated_data \
  --output-dir neurolab/data/neurovault_cache \
  --average-subject-level
```

Averaging is done **inside** the build via `neurovault_ingestion.ingest_collection()`.

---

## Collections by type

### AVERAGE_FIRST (subject-level) — ~25 collections

Need averaging by contrast during the build. The build with `--average-subject-level` does this.

| Priority | IDs |
|----------|-----|
| High | 1952 (BrainPedia), 6618 (IBC 2nd), 2138 (IBC 1st), 4343 (UCLA LA5C), 16284 (IAPS valence), 503 (PINES) |
| Medium | 426, 445, 507, 2503, 4804, 504, 13042, 13705, 2108, 4683, 3887, 1516, 13474, 20510, 11646, 437, 12992, 19012, 6825, 1620 |

### Use as-is (group-level) — ~95 collections

Already group-averaged or meta-analytic. No averaging needed.

Examples: 457, 1274, 3324, 20820, 18197, 844, 825, 1425, 7756–7761, 16266, …

---

## Alternative: Post-hoc averaging

If you have a cache built **without** `--average-subject-level` (subject-level maps, one per image), you can apply averaging without re-parcellating:

```bash
python neurolab/scripts/average_neurovault_cache.py --min-subjects 1
```

This uses existing parcellated maps. See script docstring for condition-based contrasts (503, 504, 16284).

---

## Regenerate collection list

```bash
python neurolab/scripts/list_neurovault_collections.py
```
