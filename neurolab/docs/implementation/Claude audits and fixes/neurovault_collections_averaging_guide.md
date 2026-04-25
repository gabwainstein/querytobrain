# NeuroVault Curated Collections: Full List and Averaging Priority

**126 collections** from `download_neurovault_curated.py --all`. Use this to decide which need subject-level averaging before training.

---

## Curation and preprocessing pipeline (build_neurovault_cache.py)

All NeuroVault maps are brought into a single, consistent space before training:

| Step | What | Implementation |
|------|------|----------------|
| **Atlas** | Glasser 360 (cortex) + Tian S2 32 (subcortex) = **392 parcels** | `neurolab.parcellation`: `get_combined_atlas_path()`, `get_n_parcels()` |
| **Resampling** | Every image is resampled to the pipeline atlas before parcellation | `resample_to_atlas(img)` in `parcellation.py` — handles different MNI/template/origin so alignment is correct regardless of source |
| **Group averages** | Subject-level collections (1952, 6618, 2138, 4343, 16284, etc.) are averaged by contrast within collection | `--average-subject-level`; `neurovault_ingestion.ingest_collection()` with `AVERAGE_FIRST`; use-as-is for group/meta-analytic collections |
| **QC** | Reject all-zero, high-NaN, extreme, or constant maps | `neurovault_ingestion.qc_filter()` (per collection when averaging, global when use-as-is); `--no-qc` to skip |
| **Z-score** | **Global** across all 392 parcels (fMRI: cross-compartment pattern is informative) | `neurovault_ingestion.zscore_maps(term_maps, axis=1)`; `--no-zscore` to skip |
| **Sample weights** | Meta-analytic 2×, group 1×, subject-averaged 0.8× | `term_sample_weights.pkl` from `get_sample_weight()`; used by training |

**Recommended command (curated data):**
```bash
python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level
```

---

## Quick reference: Average first (IDs only)

**High priority (subject-level, many maps):** 1952, 6618, 2138, 4343, 16284

**Medium priority:** 426, 445, 507, 2503, 4804, 504, 13042, 13705, 2108, 4683, 3887, 1516, 13474, 20510, 11646, 437, 12992, 19012, 6825, 1620

**Check (may already be group):** 503

---

## Use as-is (already group-level or meta-analytic)

These collections contain group averages, meta-analytic consensus maps, or atlases. **No averaging needed.**

| ID | Tier | Name / Notes |
|----|------|--------------|
| **457** | 1 | HCP group-level (~1,200 subjects) |
| **1274** | 1 | Cognitive Atlas × NeuroSynth decoding (399 concepts) |
| **3324** | 1 | Kragel 2018 pain/cog/emotion (15×18 studies) |
| **20820** | 1 | HCP-YA task-evoked network atlas |
| **18197** | 2 | Peraza IBMA (WM, motor, emotion) |
| **844** | 2 | Working memory meta-analysis |
| **833** | 2 | Motor learning meta-analysis |
| **830** | 2 | Vestibular cortex meta-analysis |
| **825** | 2 | DLPFC co-activation parcellation |
| **839** | 2 | Vigilant attention meta-analysis |
| **1425** | 2 | Pain NIDM-Results (21 studies) |
| **1432** | 2 | Pain IBMA results |
| **1501** | 2 | Addiction reward SDM (JAMA Psych) |
| **2462** | 2 | Social brain connectome meta-analysis |
| **3884** | 2 | MDD reward meta-analysis |
| **5070** | 2 | Reward processing clustering |
| **5377** | 2 | Inhibition of automatic imitation meta-analysis |
| **5943** | 2 | PTSD autobiographical memory meta-analysis |
| **6262** | 2 | DMN parcellation meta-analysis |
| **7793** | 2 | Social reward meta-analysis |
| **8448** | 2 | Executive function networks (166 SPMs) |
| **11343** | 2 | GAD/FAD/MDD VBM meta-analysis |
| **20036** | 2 | Source vs item memory meta-analysis (66 studies) |
| **555** | 2 | Reward in obesity/addiction meta-analysis |
| **3822** | 2 | BrainMap VBM meta-analysis |
| **15965** | 2 | Math anxiety ALE |
| **2884** | 3 | Anxious WM (group T-maps) |
| **2621** | 3 | WM face load (group T-maps) |
| **3085** | 3 | 2-back vs 0-back (group T-maps) |
| **5623** | 3 | Visual WM searchlight (group maps) |
| **3192** | 3 | Context-dependent WM (group maps) |
| **3158** | 3 | Meaningful inhibition Go/No-Go |
| **6009** | 3 | Non-selective response inhibition |
| **13656** | 3 | Response time paradox (ANT N=91, Stroop N=94) |
| **6221** | 3 | Proximal threat fear conditioning |
| **6237** | 3 | Acute vs sustained fear |
| **15274** | 3 | Threat anticipation |
| **1541** | 3 | Emotional contagion + sleep deprivation |
| **4146** | 3 | Sleep restriction emotional regulation |
| **16266** | 3 | Emotion regulation system ID |
| **6126** | 3 | Hippocampal pain reactivation |
| **10410** | 3 | Pain value signature (PVP) |
| **12874** | 3 | Instructions vs experience in pain |
| **13924** | 3 | Facial expression pain signature |
| **15030** | 3 | Physical vs vicarious pain |
| **9244** | 3 | Placebo analgesia and empathy |
| **6088** | 3 | Episodic memory replay |
| **5673** | 3 | Memory integration/reactivation |
| **2814** | 3 | Memory reinstatement |
| **3340** | 3 | Reward learning (weeks vs minutes) |
| **8676** | 3 | Cued reward omission |
| **3960** | 3 | Striatal reward in youth depression |
| **12480** | 3 | DLPFC stimulation and reward |
| **63** | 3 | Test-retest motor/language/attention |
| **834** | 3 | Anterior midcingulate motor control |
| **11584** | 3 | Finger tapping PET/fMRI |
| **315** | 3 | Adaptive learning |
| **6047** | 4 | NARPS overlap maps |
| **6051** | 4 | NARPS IBMA results |
| **262** | 4 | Harvard-Oxford atlas |
| **264** | 4 | JHU DTI white matter atlas |
| **550** | 4 | 7T subcortical variability |
| **3145** | 4 | Subcortical brain nuclei atlas |
| **1625** | 4 | Human Brainnetome Atlas |
| **2981** | 4 | Blood pressure and gray matter (VBM) |
| **6074** | 4 | Brainstem pathway atlas |
| **7114** | 4 | Child WM tract atlas |
| **5662** | 4 | Multimodal MRI FTD/AD |
| **9357** | 4 | UK Biobank APOE (~28K MRI) |
| **8461** | 4 | (Tier 4 structural) |
| **1057** | 4 | Yeo 7/17 networks (1,000 subjects) |
| **1598** | 4 | Margulies principal gradient |
| **3434** | 4 | UK Biobank DMN (10K subjects) |
| **3245** | 4 | Extended amygdala connectivity |
| **8076** | 4 | Extended amygdala (HCP N=1,073) |
| **2485** | 4 | VTA/SN resting-state networks |
| **109** | 4 | Structure-function connectome |
| **1206** | 4 | Serotonin PET normative (~210 subjects) |
| **15237** | 4 | NeuroT-Map neurotransmitter projections |
| **17228** | 4 | Supplementary NeuroT-Map |
| **7756–7761** | 4 | WM function atlas (6 parts) |
| **1083** | P | (Pharmacological) |
| **12212** | P | (Pharmacological) |
| **4040** | P | (Pharmacological) |
| **4041** | P | (Pharmacological) |
| **9246** | P | (Pharmacological) |
| **8306** | P | (Pharmacological) |
| **13665** | P | (Pharmacological) |
| **1186** | P | (Pharmacological) |
| **3902** | P | (Pharmacological) |
| **5488** | P | (Pharmacological) |
| **3713** | P | (Pharmacological) |
| **2508** | P | (Pharmacological) |
| **3264** | P | (Pharmacological) |
| **EBAYVDBZ** | S | GC-LDA 200-topic atlas |
| **OCAMCQFK** | S | Improved DMN model |
| **UOWUSAMV** | S | PTSD meta-analysis |
| **ZSVLTNSF** | S | Depression neurofeedback |

---

## Average first (subject-level maps)

These collections have **subject-level** or **individual** maps. Average by contrast/condition before adding to the cache for best training quality.

| ID | Tier | Name / Notes | Priority |
|----|------|--------------|----------|
| **1952** | 1 | BrainPedia — subject-level SPMs; 196 conditions | **High** |
| **6618** | 1 | IBC 2nd release — 13 subjects × 205 contrasts; individual SPMs | **High** |
| **2138** | 1 | IBC 1st release — 12 subjects × 59 contrasts; unsmoothed SPMs | **High** |
| **4343** | 1 | UCLA LA5C — 130 healthy + 142 clinical; subject-level | **High** |
| **426** | 3 | False belief vs physical reasoning — OpenfMRI ds000109 | Medium |
| **445** | 3 | Why/How ToM validation — 3 fMRI studies | Medium |
| **507** | 3 | Consensus decision-making — Neuron | Medium |
| **2503** | 3 | Social Bayesian inference | Medium |
| **4804** | 3 | Fusiform-network coupling | Medium |
| **503** | 3 | PINES emotion — N=182; multivariate pattern (may be group) | Check |
| **16284** | 3 | IAPS emotional valence — **Individual betas; N=56** | **High** |
| **504** | 3 | Pain comparison — NPS; multiple intensity levels | Medium |
| **13042** | 3 | Language and WM in epilepsy | Medium |
| **13705** | 3 | (Tier 3 language) | Medium |
| **2108** | 3 | (Tier 3) | Medium |
| **4683** | 3 | (Tier 3) | Medium |
| **3887** | 3 | (Tier 3) | Medium |
| **1516** | 3 | (Tier 3) | Medium |
| **13474** | 4 | Tensorial ICA for MDD | Medium |
| **20510** | 4 | Lesions causing psychosis | Medium |
| **11646** | 4 | ASD emotional egocentricity (21 ASD vs 21 controls) | Medium |
| **437** | 4 | Autism functional subnetworks | Medium |
| **12992** | 4 | Nicotine abstinence threat | Medium |
| **19012** | 4 | Incentive-boosted inhibitory control (N=76 adolescents) | Medium |
| **6825** | 4 | Schizophrenia deformation | Medium |
| **1620** | 4 | Depression resting-state and task | Medium |

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Use as-is | ~95 | No averaging |
| Average first | ~25 | Average by contrast/condition before cache |
| Slug-based | 4 | EBAYVDBZ, OCAMCQFK, UOWUSAMV, ZSVLTNSF (resolve to numeric ID) |

**Recommended workflow:** Build cache with use-as-is collections first. For average-first collections, add a preprocessing step that groups images by `collection_id` + `contrast_definition` (or similar metadata), computes mean map per group, then adds those to the cache.
