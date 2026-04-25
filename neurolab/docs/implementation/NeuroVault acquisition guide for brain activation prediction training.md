# NeuroVault acquisition guide for brain activation prediction training

**Abbreviations:** ASD = autism spectrum disorder; CBMA = coordinate-based meta-analysis; DLPFC = dorsolateral prefrontal cortex; DMN = default mode network; FAD = fear-related anxiety disorders; GAD = generalized anxiety disorder; IAPS = International Affective Picture System; IBMA = image-based meta-analysis; IBC = Individual Brain Charting; MDD = major depressive disorder; NPS = Neurologic Pain Signature; ToM = theory of mind; VBM = voxel-based morphometry; WM = working memory (cognitive) or white matter (structural).

**Implementation:** The curated collection list below is implemented in `neurolab/scripts/download_neurovault_curated.py`. Run `python neurolab/scripts/download_neurovault_curated.py --all` to fetch all tiers (1–4, **126 collections**). The **download** stores all **raw images** in those collections. After **`build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level`** you get **~2.7–5K maps** (the curated training set). Output dir: `neurolab/data/neurovault_curated_data/`. The pipeline **prefers curated over bulk** (`neurovault_data`): `run_full_cache_build.py` and `rebuild_all_caches.py` use curated when present. Use `--download-neurovault-curated` to auto-download if missing.

**Resume:** If the download is interrupted (restart, network failure), run the same command again. The script uses `mode=download_new` by default, so it skips already-downloaded images and continues from where it left off.

**Recover manifest:** If `manifest.json` is missing (interrupted before completion), run `python neurolab/scripts/download_neurovault_curated.py --recover-manifest` to build the manifest from existing `downloads/neurovault/` files. Then run `--all` again to continue downloading.

---

**The most impactful additions to your training data will come from ~90 NeuroVault collections spanning meta-analyses, multi-domain compilations, and domain-specific group maps — collectively adding an estimated 2,000–4,000 high-quality group-level statistical maps to complement your existing NeuroSynth/NeuroQuery base.** These collections fill critical gaps in clinical, pharmacological, social-cognitive, and pain domains that are underrepresented in coordinate-based databases. The Peraza et al. 2025 analysis confirms that only **2.7% of NeuroVault's ~238K images** (roughly 6,400 maps) meet strict IBMA inclusion criteria (fMRI-BOLD, group-level, N>10, T/Z stats, unthresholded, MNI space, >40% brain coverage), so aggressive curation is essential.

---

## Tier 1: Multi-domain compilations and mega-collections

These collections provide the highest value-per-download because each one covers multiple cognitive domains with consistent preprocessing and well-documented contrasts. They should be your first additions.

| ID | Name | Maps (est.) | Domains | DOI | Modality | Notes |
|---|---|---|---|---|---|---|
| **457** | HCP group-level task maps (WU-Minn overview) | ~23 | WM, emotion, motor, language, social, gambling, relational | 10.1016/j.neuroimage.2013.05.041 | fMRI | Group Z-stats from ~1,200 subjects; **benchmark reference** used in Peraza 2025 and NiCLIP |
| **1952** | BrainPedia / Atlases of cognition | ~hundreds | ~30 protocols (OpenfMRI + HCP + Neurospin) | 10.1371/journal.pcbi.1006565 | fMRI | Subject-level SPMs; 196 conditions across diverse cognitive functions |
| **6618** | Individual Brain Charting, 2nd release | ~thousands | 25 tasks, 205 contrasts (HCP + ARCHI + language + ToM + pain + reward) | 10.1038/s41597-020-00670-4 | fMRI | 13 subjects at 1.5mm; smoothed individual SPMs; most comprehensive multi-task collection |
| **2138** | Individual Brain Charting, 1st release | ~hundreds | 12 tasks, 59 contrasts | 10.1038/sdata.2018.105 | fMRI | 12 subjects; unsmoothed SPMs |
| **4343** | UCLA Consortium for Neuropsychiatric Phenomics (LA5C) | ~dozens | Memory, cognitive control, risk-taking | 10.1038/sdata.2016.110 | fMRI | 130 healthy + 142 clinical (schizophrenia, bipolar, ADHD); rare clinical + healthy in same collection |
| **3324** | Generalizable representations of pain, cognitive control, and negative emotion | **270** | Pain, cognitive control, negative emotion | 10.1038/s41593-017-0051-7 | fMRI | Kragel et al. 2018 Nature Neuroscience; 15 subjects × 18 studies |
| **EBAYVDBZ** | GC-LDA brain decoding atlas | **200** | 200 topics from 11K+ studies | Rubin et al. 2017 PLoS Comp Bio | fMRI | Topic-specific brain maps covering full cognitive ontology |
| **1274** | Cognitive Atlas terms decoded via NeuroSynth | ~399 | 399 Cognitive Atlas concepts | — | Meta-analytic | Logistic regression parameter maps for each concept; excellent for text-to-brain grounding |
| **20820** | Task-Evoked Network Atlas: HCP-YA | ~dozens | Emotion, gambling, WM, social, movie | — | fMRI | Tensor ICA network maps from HCP |

**Estimated Tier 1 total: ~1,500–3,000 maps** (most are subject-level requiring aggregation, except 457 and 1274 which are group-level).

---

## Tier 2: Meta-analyses deposited on NeuroVault

Meta-analytic result maps are exceptionally valuable for your model because each one represents the consensus activation pattern for a specific cognitive process, derived from dozens of studies. These are effectively "distilled" brain maps with high signal-to-noise.

| ID | Name | Maps (est.) | Domain | DOI | Type |
|---|---|---|---|---|---|
| **18197** | Peraza et al. IBMA results | ~6–12 | Working memory, motor, emotion | 10.1038/s41598-025-20328-8 | IBMA |
| **844** | Working memory coordinate-based meta-analysis | ~5–15 | Working memory | ANIMA import | ALE |
| **833** | Motor learning meta-analysis (70 experiments) | ~5–10 | Motor learning | 10.1016/j.neuroimage.2012.11.020 | ALE |
| **830** | Vestibular cortex meta-analysis (28 experiments) | ~5–10 | Vestibular/multisensory | ANIMA import | ALE |
| **825** | DLPFC co-activation parcellation | ~5–10 | Executive function | ANIMA import | Meta-analytic |
| **839** | Vigilant attention meta-analysis | ~5–15 | Sustained attention | — | ALE |
| **1425** | Pain NIDM-Results packs (21 studies) | ~21+ | Pain | Maumet et al. 2016 Sci Data | IBMA data |
| **1432** | Pain IBMA results | ~5–10 | Pain | Maumet et al. 2016 | IBMA results |
| **1501** | Addiction reward processing (25 studies, JAMA Psych) | ~5–15 | Addiction/reward | Published JAMA Psychiatry | SDM |
| **2462** | Social brain connectome meta-analysis | ~10–20 | Social cognition | 10.1093/cercor/bhx121 | Meta-analytic |
| **3884** | MDD reward processing meta-analysis (41 studies) | ~5–15 | Depression/reward | — | CBMA |
| **5070** | Reward processing clustering (749 experiments) | ~10–20 | Reward processing subtypes | — | BrainMap clustering |
| **5377** | Inhibition of automatic imitation meta-analysis | ~5–10 | Inhibition/imitation | — | ALE |
| **5943** | PTSD autobiographical memory meta-analysis (28 studies) | ~10–20 | PTSD | 10.1002/da.22977 | SDM |
| **6262** | DMN functional parcellation meta-analysis | ~10–20 | Default mode network | — | Neurosynth-based |
| **7793** | Social reward/punishment meta-analysis (SID task) | ~5–10 | Social reward | — | Voxel-based MA |
| **8448** | Executive function network labels meta-analysis (166 SPMs) | ~166 | Executive function networks | 10.1007/s10548-021-00847-z | Network label MA |
| **11343** | GAD, FAD, and MDD VBM meta-analysis | ~10–20 | Anxiety/depression (structural) | — | SDM-PSI |
| **15965** | Math anxiety ALE meta-analysis | ~5–10 | Math anxiety | — | ALE |
| **20036** | Source vs item memory retrieval meta-analysis (66 studies) | ~10–20 | Episodic memory | 10.1162/imag.a.124 | Network MA |
| **555** | Reward in obesity, substance, and behavioral addictions | ~5–15 | Reward/addiction | — | Meta-analysis |
| **3822** | BrainMap VBM structural meta-analysis | ~10–20 | Structural (VBM) | 10.1002/hbm.24078 | VBM meta-analysis |

**Estimated Tier 2 total: ~300–500 meta-analytic result maps.** These are the single most information-dense maps you can add.

---

## Tier 3: Domain-specific published collections

Organized by cognitive domain, these fill specific gaps. Collections marked with ★ are particularly high-value for training because they cover domains poorly represented in NeuroSynth/NeuroQuery.

### Working memory and executive function

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **2884** | Anxious WM mis-allocation | ~5 | WM/anxiety | 10.1038/s41598-017-08443-7 | Face WM, group T-maps; used in NiMARE |
| **2621** | WM face load | ~5 | Working memory | — | Group T-maps; NiMARE-referenced |
| **3085** | 2-back vs 0-back | ~5 | Working memory | — | Group T-maps; NiMARE-referenced |
| **5623** | Visual WM searchlight decoding | ~5 | Working memory | — | Group maps; NiMARE-referenced |
| **3192** | Context-dependent WM | ~5 | Working memory | — | Group maps; NiMARE-referenced |
| **13042** | Language and WM in epilepsy | ~20+ | WM/language | 10.1093/brain/awac150 | Multiple WM and language contrasts |
| **3158** | Meaningful inhibition (Go/No-Go) | ~10 | Response inhibition | 10.1016/j.neuroimage.2018.06.074 | Modality effects on inhibition |
| **6009** | Non-selective response inhibition | ~10 | Response inhibition | 10.1038/s41598-022-14221-x | Bayesian fMRI + Go/NoGo meta-analysis |
| **13656** | Response time paradox across 7 tasks | ~20+ | Executive function | — | ANT N=91, Stroop N=94, stop-signal N=91 — large samples |

### Social cognition and theory of mind ★

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **426** | False belief vs physical reasoning | ~10 | Theory of mind | 10.1523/JNEUROSCI.5511-11.2012 | Classic ToM paradigm; OpenfMRI ds000109 |
| **445** | Why/How ToM validation | ~10 | Theory of mind | — | 3 fMRI studies validating ToM contrast |
| **507** | Consensus decision-making | ~10 | Social decision | 10.1016/j.neuron.2015.03.019 | Published in Neuron |
| **2503** | Social Bayesian inference in groups | ~10 | Social decision | 10.1371/journal.pbio.2001958 | Social information integration |
| **4804** | Fusiform-network coupling for social motivation | ~10 | Social motivation | — | DMN/ECN coupling with FFA |

### Emotion processing and fear ★

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **503** | PINES emotion signature | ~120+ | Negative emotion | 10.1371/journal.pbio.1002180 | N=182 total; multivariate pattern signature |
| **6221** | Proximal threat fear conditioning | ~10 | Fear | 10.1073/pnas.2004258117 | PNAS 2020; fear acquisition/persistence |
| **6237** | Acute vs sustained fear | ~10 | Fear | — | Unthresholded + thresholded maps |
| **15274** | Threat anticipation (temporal uncertainty) | ~10 | Threat/anxiety | — | Maryland Threat Countdown paradigm |
| **16284** | IAPS emotional valence (3 conditions) | ~56+ | Emotion | 10.1101/2023.07.29.551128 | Individual betas; N=56; positive/negative/neutral |
| **1541** | Emotional contagion + sleep deprivation | ~10 | Emotion/faces | 10.1038/s41598-020-74489-9 | FACES task; Stockholm Sleepy Brain |
| **4146** | Sleep restriction emotional regulation | ~10 | Emotion regulation | 10.1098/rsos.181704 | Cognitive reappraisal paradigm |
| **16266** | Emotion regulation system identification | ~10+ | Emotion regulation | — | Bayes factor and system ID results |

### Pain processing ★

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **504** | Pain comparison data (Chang/Kragel) | ~30+ | Pain | PLoS Biology 2015 | Pain intensity levels; NPS comparison |
| **6126** | Hippocampal pain reactivation | ~10 | Pain/memory | 10.1523/JNEUROSCI.1350-20.2021 | Heat pain with decision-making |
| **10410** | Pain value signature (PVP) | ~5 | Pain valuation | PNAS 2022 | Unthresholded multivariate pattern |
| **12874** | Instructions vs experience in pain | ~10 | Pain learning | — | Thermal pain reversal learning |
| **13924** | Facial expression pain signature (FEPS) | ~5 | Pain expression | eLife | Unthresholded pattern maps |
| **15030** | Physical vs vicarious pain | ~10 | Pain/empathy | — | NPS/VPS across 3 datasets |
| **9244** | Placebo analgesia and empathy | ~10 | Pain/placebo | 10.1093/texcom/tgab039 | N=45 |

### Episodic memory

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **6088** | Episodic memory replay | ~10 | Memory replay | 10.1038/s41593-020-0649-z | Nature Neuroscience; MEG source maps |
| **5673** | Memory integration/reactivation | ~10 | Memory | 10.1038/s41598-020-61737-1 | AB-AC inference; MTL/mPFC |
| **2814** | Memory reinstatement | ~10 | Memory | — | T-images for reinstatement |

### Decision making and reward

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **3340** | Reward learning (weeks vs minutes) | ~10 | Reward | J Neurosci 2018 | Spaced vs massed learning |
| **8676** | Cued reward omission | ~10 | Reward | 10.3389/fnhum.2021.615313 | Conditioned inhibition |
| **3960** | Striatal reward in youth depression | ~10 | Reward/depression | 10.1037/abn0000389 | Clinical + reward |
| **12480** | DLPFC stimulation and reward cue reactivity | ~10 | Reward/TMS | — | TMS modulation |

### Motor and learning

| ID | Name | Maps | Domain | DOI | Notes |
|---|---|---|---|---|---|
| **63** | Test-retest motor/language/attention | ~20 | Motor/language/attention | 10.1186/2047-217X-2-6 | OpenfMRI ds000114; reliability data |
| **834** | Anterior midcingulate motor control | ~10 | Motor control | — | aMCC connectivity |
| **11584** | Finger tapping PET/fMRI | ~10 | Motor/dopamine | — | Simultaneous PET/fMRI |
| **315** | Adaptive learning (3 factors) | ~10 | Learning | — | Surprise, uncertainty, reward-driven |

**Estimated Tier 3 total: ~500–800 maps across all cognitive domains.**

---

## Tier 4: Clinical, structural, connectivity, and pharmacological

### Clinical disorder collections

NeuroVault's clinical coverage is notably sparse — confirmed by the official 2024 analysis stating clinical topics are "significantly under-represented." These are the best available:

| ID | Name | Maps | Condition | DOI | Notes |
|---|---|---|---|---|---|
| **13474** | Tensorial ICA for MDD (HCP data) | ~10 | Depression | — | Social/reward networks in MDD |
| **1620** | Depression resting-state and task | ~10 | Depression | — | Rest + task comparison |
| **ZSVLTNSF** | Depression neurofeedback | ~5 | MDD treatment | — | Uncorrected maps |
| **6825** | Schizophrenia-related deformation | ~5 | Schizophrenia | — | Structural deformation maps |
| **20510** | Lesions causing psychosis | ~10 | Psychosis | JAMA Psychiatry 2025 | Lesion network mapping |
| **11646** | ASD emotional egocentricity bias | ~10 | Autism | — | Cyberball task; 21 ASD vs 21 controls |
| **437** | Autism functional subnetworks | ~10 | Autism | — | Graph analysis; consensus clustering |
| **12992** | Nicotine abstinence threat fMRI | ~10 | Nicotine/smoking | — | Threat paradigm in smokers |
| **19012** | Incentive-boosted inhibitory control (adolescents) | ~10 | ADHD-relevant | — | Go/No-Go; N=76 adolescents |

### Structural and atlas collections

| ID | Name | Maps | Type | DOI | Notes |
|---|---|---|---|---|---|
| **262** | Harvard-Oxford atlas | ~10 | Probabilistic atlas | — | Most re-used collection (8 citations) |
| **264** | JHU DTI white matter atlas | ~50 | WM tract atlas | Mori 2005 | Standard DTI reference |
| **550** | 7T subcortical variability | ~10 | Subcortical atlas | 10.1016/j.neuroimage.2014.03.032 | High-resolution 7T |
| **3145** | Subcortical brain nuclei atlas | ~30 | Probabilistic atlas | 10.1038/sdata.2018.63 | High-resolution MNI space |
| **1625** | Human Brainnetome Atlas | ~246+ | Connectivity-based parcellation | Published | 210 cortical + 36 subcortical |
| **2981** | Blood pressure and gray matter (N=423) | ~10 | VBM | 10.1212/WNL.0000000000006947 | GMV maps from AES-SDM |
| **6074** | Brainstem pathway atlas | ~20 | DWI/tractography | 10.1016/j.neuroimage.2017.12.042 | Probabilistic brainstem WM |
| **7114** | Child WM tract atlas (6–8y) | ~20 | DTI atlas | 10.1089/brain.2021.0058 | Age-specific pediatric |
| **7756–7761** | Atlas of WM function (stroke, 6 parts) | ~100+ | Disconnection atlas | — | 1,333 stroke patients; A-Z terms |
| **5662** | Multimodal MRI in FTD/AD carriers | ~30 | VBM+TBSS+rs-fMRI | 10.1186/s12883-019-1567-0 | Rare multimodal clinical |
| **9357** | UK Biobank APOE analysis (~28K MRI) | ~20 | WM tracts | 10.1038/s41398-024-02848-5 | Massive sample |

### Resting-state and connectivity

| ID | Name | Maps | Type | DOI | Notes |
|---|---|---|---|---|---|
| **1057** | Yeo 7/17 network parcellation | ~20 | RS-fMRI networks | 10.1152/jn.00338.2011 | **Landmark**; 1,000 subjects |
| **1598** | Margulies principal gradient | ~5 | Functional gradient | 10.1073/pnas.1608282113 | Sensorimotor-to-transmodal axis |
| **3434** | UK Biobank DMN subspecialization (10K subjects) | ~10 | DMN atlas | 10.1073/pnas.1804876115 | Massive sample |
| **3245** | Extended amygdala connectivity | ~10 | Seed-based FC | — | BST and CeA connectivity |
| **8076** | Extended amygdala connectivity (HCP) | ~10 | Seed-based FC | — | N=1,073 HCP subjects |
| **2485** | VTA/SN resting-state networks | ~20 | RS-fMRI + atlas | 10.1016/j.neuroimage.2014.06.047 | Dopaminergic midbrain |
| **109** | Structure-function connectome | ~10 | DTI + RS-fMRI | 10.1016/j.neuroimage.2013.09.069 | Structure-function overlap |
| **OCAMCQFK** | Improved DMN neuroanatomical model | ~10 | DMN maps | Alves et al. Commun Biol | Thalamus/basal forebrain in DMN |

### Additional pharmacological (beyond user's existing list)

| ID | Name | Maps | System | DOI | Notes |
|---|---|---|---|---|---|
| **1206** | Serotonin PET normative database | ~10+ | 5-HT1A/1B/2A/4/6 + SERT | 10.1016/j.neuroimage.2012.07.001 | **Landmark** multi-tracer atlas; ~210 subjects |
| **15237** | NeuroT-Map neurotransmitter projections | ~15+ | Multi-receptor WM projections | 10.1038/s41467-025-57680-2 | D1/D2/DAT/5HT/NAT/M1/VAChT |
| **17228** | Supplementary NeuroT-Map | ~10+ | GABA-A/mGluR5/MOR/H3/CB1 | 10.1038/s41467-025-57680-2 | Complements 15237 |
| **2485** | VTA/SN dopaminergic atlas | ~20 | Dopamine | 10.1016/j.neuroimage.2014.06.047 | Also listed under connectivity |
| **11584** | Finger tapping dopamine PET/fMRI | ~10 | Dopamine displacement | — | Simultaneous PET/fMRI |

**Estimated Tier 4 total: ~400–700 maps.**

---

## NARPS: A unique multi-analytic resource

The Neuroimaging Analysis Replication and Prediction Study (Botvinik-Nezer et al. 2020, Nature) deposited results from **70 independent analysis teams** analyzing the same dataset. Key collections:

| ID | Name | Value |
|---|---|---|
| **6047** | NARPS overlap maps | Proportion of teams with significant voxels; ground truth for analytic variability |
| **6051** | NARPS IBMA results | Image-based meta-analysis across all 70 teams; FDR-corrected |

These are uniquely informative for understanding how analysis choices affect brain maps — potentially useful for training data augmentation strategies.

---

## What NeuroVault does not cover well

The Peraza et al. 2025 analysis and the official NeuroVault "Decade" report (March 2024) confirm several systematic gaps that no amount of searching will fill. **No substantial collections were found** on NeuroVault for these domains:

- **Bipolar disorder** — ENIGMA has working groups but results not deposited on NeuroVault
- **OCD** — extremely sparse
- **Alzheimer's disease / MCI** — ADNI data hosted elsewhere
- **Parkinson's disease** — no discoverable collections
- **Epilepsy** — despite being a major neuroimaging research area
- **Eating disorders** — absent
- **Sleep disorders / insomnia** — only tangentially covered through sleep deprivation studies
- **Spatial navigation** — nearly absent as a primary domain
- **Creativity / divergent thinking** — only one small collection (10350)
- **Psychedelic-specific collections** beyond LSD (Collection 1083) — psilocybin and MDMA data typically deposited on OpenNeuro or OSF instead
- **Auditory processing** as a primary domain — only incidentally covered

For these gaps, consider supplementing from **OpenNeuro** (raw data you'd need to process), **ENIGMA consortium** results (request access), the **Hansen et al. 2022 receptor atlas** (via GitHub neuromaps), or the **BrainMap** database.

---

## Priority acquisition list with collection IDs

Copy these directly into your download script. Collections are ordered by priority within each tier.

```python
# TIER 1: Multi-domain compilations (highest ROI per collection)
TIER_1_COLLECTIONS = [
    457,    # HCP group-level (7 domains, ~1200 subjects)
    1274,   # Cognitive Atlas × NeuroSynth decoding (399 concepts)
    1952,   # BrainPedia (30 protocols)
    3324,   # Pain + CogControl + Emotion (270 maps)
    6618,   # IBC 2nd release (25 tasks, 205 contrasts)
    2138,   # IBC 1st release (12 tasks, 59 contrasts)
    4343,   # UCLA LA5C (healthy + clinical)
    20820,  # HCP-YA task-evoked network atlas
]

# TIER 2: Meta-analyses (consensus maps, highest information density)
TIER_2_META_ANALYSES = [
    18197,  # Peraza IBMA (WM, motor, emotion)
    844,    # Working memory meta-analysis (ANIMA)
    833,    # Motor learning meta-analysis (ANIMA)
    830,    # Vestibular cortex meta-analysis (ANIMA)
    825,    # DLPFC co-activation (ANIMA)
    839,    # Vigilant attention meta-analysis
    1425,   # Pain NIDM-Results (21 studies)
    1432,   # Pain IBMA results
    1501,   # Addiction reward SDM (JAMA Psych)
    2462,   # Social brain connectome meta-analysis
    3884,   # MDD reward meta-analysis
    5070,   # Reward processing clustering (BrainMap)
    5377,   # Imitation inhibition meta-analysis
    5943,   # PTSD memory meta-analysis
    6262,   # DMN parcellation meta-analysis
    7793,   # Social reward meta-analysis
    8448,   # Executive function networks (166 SPMs)
    11343,  # Anxiety/depression VBM meta-analysis
    20036,  # Memory retrieval meta-analysis (66 studies)
    555,    # Reward in obesity/addiction meta-analysis
    3822,   # BrainMap VBM meta-analysis
    15965,  # Math anxiety ALE
]

# TIER 3: Cognitive domain group maps (fill specific gaps)
TIER_3_COGNITIVE = [
    # Working memory / executive function
    2884, 2621, 3085, 5623, 3192,  # NiMARE WM collections
    13042, 3158, 6009, 13656,
    # Social cognition / ToM
    426, 445, 507, 2503, 4804,
    # Emotion / fear
    503, 6221, 6237, 15274, 16284, 1541, 4146, 16266,
    # Pain
    504, 6126, 10410, 12874, 13924, 15030, 9244,
    # Episodic memory
    6088, 5673, 2814,
    # Reward / decision making
    3340, 8676, 3960, 12480,
    # Motor / learning / attention
    63, 834, 11584, 315,
    # Language / semantics
    13705, 2108, 4683, 3887, 1516,
]

# TIER 4: Clinical, structural, connectivity, pharma
TIER_4_SUPPLEMENTARY = [
    # Clinical
    13474, 20510, 11646, 437, 12992, 19012, 6825,
    # Structural / atlases
    3145, 2981, 6074, 7114, 5662, 9357, 8461,
    # Connectivity
    1057, 1598, 3434, 3245, 8076, 2485, 109,
    # Pharmacological (new)
    1206, 15237, 17228,
    # NARPS
    6047, 6051,
    # Brain decoding
    # EBAYVDBZ,  # Use slug-based ID
]

# Slug-based IDs (need special handling)
SLUG_COLLECTIONS = [
    "EBAYVDBZ",   # GC-LDA 200-topic atlas
    "OCAMCQFK",   # Improved DMN model
    "UOWUSAMV",   # PTSD meta-analysis maps
]

# White matter function atlas (6 parts)
WM_ATLAS_COLLECTIONS = list(range(7756, 7762))  # 7756-7761

# Already identified pharmacological (for reference, do not re-download)
EXISTING_PHARMA = [
    1083, 12212, 4040, 4041, 9246, 8306,
    13665, 1186, 3902, 5488, 3713, 1501, 2508, 3264,
]
```

---

## Estimated total yield by tier

| Tier | Collections | Estimated maps | Primary value |
|---|---|---|---|
| Tier 1 | 8 | 1,500–3,000 | Multi-domain coverage; benchmark quality |
| Tier 2 | 21 | 300–500 | Consensus activation patterns; highest information density |
| Tier 3 | ~45 | 500–800 | Domain-specific gap-filling |
| Tier 4 | ~30 | 400–700 | Clinical, structural, connectivity, pharma |
| **Total** | **~104** | **~2,700–5,000** | — |

After filtering for group-level, unthresholded, MNI-space T/Z maps, expect roughly **2,000–3,500 usable maps** — a meaningful **15–25% expansion** of your current NeuroSynth + NeuroQuery training set, with disproportionately more coverage in social cognition, pain, emotion, reward, and pharmacological domains where coordinate-based databases are weakest.

---

## Key practical recommendations

The Peraza et al. 2025 framework provides a three-stage selection process that you should apply post-download: metadata filtering (modality, map type, analysis level), automated outlier detection via PCA, and manual contrast verification against linked publications. Their analysis found that **working memory IBMA from curated NeuroVault maps correlated r=0.74 with HCP reference**, validating the approach. Motor maps achieved r=0.52 and emotion r=0.31, suggesting that domain difficulty varies considerably.

For your Glasser 360 + Tian S2 parcellation pipeline, prioritize downloading **unthresholded volumetric maps in MNI space** — NeuroVault reports that ~75% of its images are unthresholded and ~75% are T or Z statistics, which is encouraging. However, verify MNI registration quality after download, as the Peraza analysis flagged significant metadata inaccuracy (25% of images lack modality specification, many have incorrect annotations).

Consider also accessing the **Hansen et al. 2022 neurotransmitter receptor atlas** directly from GitHub (netneurolab/hansen_receptors) rather than NeuroVault — it provides **19 PET-derived receptor/transporter maps** across 9 neurotransmitter systems from >1,200 subjects, and is distributed via the neuromaps toolbox in a more standardized format than individual NeuroVault collections. This single resource would dramatically strengthen your model's ability to predict pharmacologically-relevant brain patterns from text queries.