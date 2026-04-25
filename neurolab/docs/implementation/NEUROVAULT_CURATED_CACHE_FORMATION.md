# NeuroVault Curated Cache Formation

**Source:** `neurovault_curated_data` (manifest + downloads)
**Output:** `neurovault_cache` (term_maps.npz, term_vocab.pkl)

Pipeline: `build_neurovault_cache.py --average-subject-level` (parcellate → average AVERAGE_FIRST → QC → z-score)

## Collections in cache (by output terms, desc)

| ID | Name | Raw images | Output terms | Averaged? | Ratio |
|----|------|:----------:|:------------:|:---------:|-------|
| 16266 | Tier3 | 1074 | 1074 | No | 1074 → 1074 |
| 1274 | Cognitive Atlas | 399 | 399 | No | 399 → 399 |
| 3324 | Pain+Cog+Emotion | 270 | 263 | No | 270 → 263 |
| 15030 | Tier3 | 252 | 230 | No | 252 → 230 |
| 7760 | WM atlas | 228 | 228 | No | 228 → 228 |
| 7758 | WM atlas | 226 | 226 | No | 226 → 226 |
| 7759 | WM atlas | 204 | 204 | No | 204 → 204 |
| 6618 | IBC 2nd | 5043 | 196 | Yes | 5043 → 196 |
| 7756 | WM atlas | 192 | 192 | No | 192 → 192 |
| 1952 | BrainPedia | 6573 | 134 | Yes | 6573 → 134 |
| 2138 | IBC 1st | 1824 | 113 | Yes | 1824 → 113 |
| 7761 | WM atlas | 90 | 90 | No | 90 → 90 |
| 20820 | HCP-YA network | 77 | 77 | No | 77 → 77 |
| 15274 | Tier3 | 79 | 76 | No | 79 → 76 |
| 4146 | Tier3 | 76 | 57 | No | 76 → 57 |
| 457 | HCP | 48 | 48 | No | 48 → 48 |
| 3822 | Meta | 42 | 40 | No | 42 → 40 |
| 13665 | Pharma | 35 | 35 | No | 35 → 35 |
| 6237 | Tier3 | 40 | 33 | No | 40 → 33 |
| 2981 | col2981 | 29 | 29 | No | 29 → 29 |
| 4343 | UCLA LA5C | 5756 | 24 | Yes | 5756 → 24 |
| 3158 | col3158 | 36 | 21 | No | 36 → 21 |
| 5662 | col5662 | 75 | 21 | No | 75 → 21 |
| 8676 | col8676 | 21 | 21 | No | 21 → 21 |
| 1541 | col1541 | 24 | 19 | No | 24 → 19 |
| 2621 | col2621 | 25 | 17 | No | 25 → 17 |
| 3264 | col3264 | 16 | 16 | No | 16 → 16 |
| 6126 | col6126 | 20 | 16 | No | 20 → 16 |
| 3713 | col3713 | 60 | 15 | No | 60 → 15 |
| 6221 | col6221 | 15 | 15 | No | 15 → 15 |
| 7757 | col7757 | 15 | 15 | No | 15 → 15 |
| 3340 | col3340 | 14 | 14 | No | 14 → 14 |
| 5943 | col5943 | 18 | 14 | No | 18 → 14 |
| 12874 | col12874 | 18 | 14 | No | 18 → 14 |
| 15237 | col15237 | 13 | 13 | No | 13 → 13 |
| 550 | col550 | 52 | 12 | No | 52 → 12 |
| 1432 | col1432 | 11 | 11 | No | 11 → 11 |
| 7114 | col7114 | 23 | 11 | No | 23 → 11 |
| 63 | col63 | 11 | 10 | No | 11 → 10 |
| 1083 | col1083 | 10 | 10 | No | 10 → 10 |
| 1598 | col1598 | 10 | 10 | No | 10 → 10 |
| 6009 | col6009 | 11 | 10 | No | 11 → 10 |
| 6047 | col6047 | 9 | 9 | No | 9 → 9 |
| 6088 | col6088 | 9 | 9 | No | 9 → 9 |
| 8306 | col8306 | 9 | 9 | No | 9 → 9 |
| 9244 | col9244 | 9 | 9 | No | 9 → 9 |
| 3192 | col3192 | 10 | 8 | No | 10 → 8 |
| 8448 | col8448 | 8 | 8 | No | 8 → 8 |
| 4041 | col4041 | 7 | 7 | No | 7 → 7 |
| 6051 | col6051 | 7 | 7 | No | 7 → 7 |
| 830 | col830 | 9 | 6 | No | 9 → 6 |
| 834 | col834 | 6 | 6 | No | 6 → 6 |
| 1186 | col1186 | 6 | 6 | No | 6 → 6 |
| 1501 | col1501 | 6 | 6 | No | 6 → 6 |
| 3145 | col3145 | 20 | 6 | No | 20 → 6 |
| 3245 | col3245 | 12 | 6 | No | 12 → 6 |
| 10410 | col10410 | 12 | 6 | No | 12 → 6 |
| 11343 | col11343 | 7 | 6 | No | 7 → 6 |
| 4683 | Tier3 | 116 | 6 | Yes | 116 → 6 |
| 844 | col844 | 5 | 5 | No | 5 → 5 |
| 5488 | col5488 | 5 | 5 | No | 5 → 5 |
| 6262 | col6262 | 9 | 5 | No | 9 → 5 |
| 8076 | col8076 | 5 | 5 | No | 5 → 5 |
| 17228 | col17228 | 5 | 5 | No | 5 → 5 |
| 315 | col315 | 4 | 4 | No | 4 → 4 |
| 1057 | col1057 | 5 | 4 | No | 5 → 4 |
| 1206 | col1206 | 4 | 4 | No | 4 → 4 |
| 1425 | col1425 | 54 | 4 | No | 54 → 4 |
| 4040 | col4040 | 4 | 4 | No | 4 → 4 |
| 5673 | col5673 | 4 | 4 | No | 4 → 4 |
| 7793 | col7793 | 4 | 4 | No | 4 → 4 |
| 13656 | col13656 | 8 | 4 | No | 8 → 4 |
| 833 | col833 | 4 | 3 | No | 4 → 3 |
| 2884 | col2884 | 18 | 3 | No | 18 → 3 |
| 3884 | col3884 | 9 | 3 | No | 9 → 3 |
| 3960 | col3960 | 6 | 3 | No | 6 → 3 |
| 5070 | col5070 | 16 | 3 | No | 16 → 3 |
| 5377 | col5377 | 21 | 3 | No | 21 → 3 |
| 5623 | col5623 | 3 | 3 | No | 3 → 3 |
| 8461 | col8461 | 3 | 3 | No | 3 → 3 |
| 9246 | col9246 | 3 | 3 | No | 3 → 3 |
| 12480 | col12480 | 3 | 3 | No | 3 → 3 |
| 13042 | Language/WM epilepsy | 20 | 3 | Yes | 20 → 3 |
| 504 | Pain NPS | 84 | 3 | Yes | 84 → 3 |
| 825 | col825 | 4 | 2 | No | 4 → 2 |
| 3085 | col3085 | 2 | 2 | No | 2 → 2 |
| 13924 | col13924 | 4 | 2 | No | 4 → 2 |
| 18197 | col18197 | 10 | 2 | No | 10 → 2 |
| 507 | Consensus decision | 0 | 2 | Yes | → 2 |
| 16284 | IAPS valence | 168 | 2 | Yes | 168 → 2 |
| 839 | col839 | 1 | 1 | No | 1 → 1 |
| 2814 | col2814 | 12 | 1 | No | 12 → 1 |
| 3902 | col3902 | 1 | 1 | No | 1 → 1 |
| 11584 | col11584 | 2 | 1 | No | 2 → 1 |
| 12212 | col12212 | 1 | 1 | No | 1 → 1 |
| 20036 | col20036 | 4 | 1 | No | 4 → 1 |
| 445 | Why/How ToM | 5 | 1 | Yes | 5 → 1 |
| 503 | PINES/IAPS | 5067 | 1 | Yes | 5067 → 1 |
| 1516 | Tier3 | 46 | 1 | Yes | 46 → 1 |
| 2108 | Tier3 | 3 | 1 | Yes | 3 → 1 |

## Totals

| Metric | Value |
|--------|------:|
| Total raw images (manifest) | 29,271 |
| Raw images from collections in cache | 28,953 |
| Total output terms in cache | 4,308 |
| Collections with output | 100 |

## Excluded from cache

- **Atlas/structural (262, 264, 1625, 6074, 9357):** Harvard-Oxford, JHU DTI, Brainnetome, brainstem tract atlas, APOE structural masks — excluded by `--exclude-atlas-collections`
- **Collections with 0 output:** Failed QC, parcellation, or dropped by averaging (min_subjects)

## Zero-output diagnostics

**AVERAGE_FIRST moved to use-as-is (≤7 images):** 19012, 6825, 20510, 1620, 2503, 13474, 13705, 3887 — each contrast had n=1, min_subjects=3 dropped them. Removing from AVERAGE_FIRST recovers ~50 maps.

**Use-as-is QC failures (diagnostic via `diagnose_parcellation_failure.py`):**
- **555 (20 images):** All pass load/resample/parcellate; fail QC — **std &lt; 0.01** (maps are near-constant after parcellation).
- **2462 (36 images):** Same pipeline; fail QC — **zeros &gt; 0.95** (sparse/thresholded meta-analysis maps), std &lt; 0.01.

Recovery options: relax `qc_filter` (lower std threshold, allow sparse) for meta-analytic collections, or `--no-qc` for specific builds.

**Use-as-is collections worth recovering (555, 2485, 2508):** Reward/addiction meta-analysis (20), VTA/SN connectivity (7), pharma fMRI (3) — ~30 maps. Likely QC (std/zeros) or parcellation. Diagnose with `diagnose_parcellation_failure.py`.

**Borderline (2462, 3434):** Seed-region definitions, not activation contrasts; may parcellate to mostly-zero. Probably not useful for training.
