# Basal Forebrain & Hypothalamus Atlas Integration Plan

Extension of the pipeline atlas to cover **cholinergic basal forebrain** (Zaborszky) and **hypothalamic nuclei** (Neudorfer). These fill gaps left by Glasser+Tian+Brainstem Navigator for neuromodulation-focused analyses.

**Project focus:** Neuromodulatory systems, cognitive flexibility, pharmacology/nootropics, LC-NE system. Parcel selection prioritizes structures with strong signal in training data and relevance to cholinergic drugs, stimulants, and stress-cognition paradigms.

---

## 1. Basal Forebrain: Zaborszky et al. (2008)

**Citation:** Zaborszky L, Hoemke L, Mohlberg H, Schleicher A, Amunts K, Zilles K (2008). Stereotaxic probabilistic maps of the magnocellular cell groups in human basal forebrain. *NeuroImage*, 42(3), 1127–1141.

**Source:** Cytoarchitectonic probabilistic maps from 10 postmortem brains, warped to MNI space.

### Parcels

| Label | Structure | Projection |
|-------|-----------|------------|
| Ch1-2 | Medial septum + vertical limb of diagonal band | Cholinergic → hippocampus |
| Ch3 | Horizontal limb of diagonal band | Cholinergic → olfactory bulb |
| Ch4 | Nucleus basalis of Meynert (NBM) | Cholinergic → all neocortex |
| Ch4p | Posterior NBM | Cholinergic → temporal cortex, amygdala |

### Acquisition Options

| Option | Source | Notes |
|--------|--------|-------|
| **JuBrain Anatomy Toolbox** | [GitHub](https://github.com/inm7/jubrain-anatomy-toolbox) / [FZ Jülich](https://www.fz-juelich.de/en/inm/inm-7/resources/tools/jubrain-anatomy-toolbox) | Probability maps as NIfTI in MNI; threshold 40–50% for binary masks |
| **EBRAINS / Julich-Brain** | [julich-brain-atlas.de](https://julich-brain-atlas.de/atlas) | Programmatic access via `siibra` Python package |

### Parcel Recommendation (Neuromodulation-Focused)

| Parcel | Why |
|--------|-----|
| **Ch4 (NBM)** | Non-negotiable. Main cholinergic projection to all neocortex. Central to attention, cognitive flexibility, cholinergic drug targets (donepezil, galantamine, nicotine). Dozens of nootropics terms will map here. |
| **Ch1-2** (medial septum/diagonal band) | Cholinergic + GABAergic projection to hippocampus. Memory consolidation, theta rhythm. Relevant for racetams, memory-enhancing compounds. |

**Skip Ch3** — primarily olfactory; unlikely to capture meaningful signal in training data.

**L/R merging:** Average both sides together into single parcels (Ch1-2_l + Ch1-2_r → Ch1-2), same convention as Brainstem Navigator.

**→ 4 new parcels** (Ch1-2_L, Ch1-2_R, Ch4_L, Ch4_R; or 2 merged: Ch1-2, Ch4).

---

## 2. Hypothalamus: Neudorfer et al. (2020)

**Citation:** Neudorfer C, Germann J, Elias GJB, Gramer R, Boutet A, Lozano AM (2020). A high-resolution in vivo magnetic resonance imaging atlas of the human hypothalamic region. *Scientific Data*, 7, 305. doi: 10.1038/s41597-020-00644-6.

**Source:** 13 hypothalamic nuclei manually delineated from high-resolution MRI, MNI152 NLIN 2009b.

### Parcels (13 hypothalamic)

| Label | Nucleus | Role |
|-------|---------|------|
| LH | Lateral hypothalamus | Orexin/hypocretin — wake/arousal |
| TM | Tuberomammillary nucleus | Histamine — arousal/wake |
| PA | Paraventricular nucleus | CRH/oxytocin/vasopressin — stress/HPA |
| VM | Ventromedial nucleus | Feeding/satiety |
| DM | Dorsomedial nucleus | Circadian/feeding |
| AN | Arcuate nucleus | Feeding, GnRH |
| SCh | Suprachiasmatic nucleus | Circadian clock |
| SO | Supraoptic nucleus | Vasopressin/oxytocin |
| MPO | Medial preoptic nucleus | Thermoregulation, parenting |
| PH | Posterior hypothalamus | Arousal, DBS target |
| PE | Periventricular nucleus | — |
| DP | Dorsal periventricular nucleus | — |
| AH | Anterior hypothalamic area | — |

**Extrahypothalamic (12):** Mammillary bodies, STN, zona incerta, NBM, diagonal band, BNST, RN, SN — **overlap with Bianciardi and Tian.**

### Acquisition Options

| Option | Source | Notes |
|--------|--------|------|
| **Lead-DBS** | [lead-dbs.org](https://www.lead-dbs.org/) | Atlas preinstalled; NIfTIs in atlases directory, MNI152 2009b |
| **Scientific Data supplementary** | doi: 10.1038/s41597-020-00644-6 | T1/T2 templates + segmentation NIfTIs |

### Parcel Recommendation (Neuromodulation-Focused)

| Parcel | Why |
|--------|-----|
| **LH** (lateral hypothalamus) | Orexin/hypocretin neurons. Wake-promoting, arousal, motivation, reward. Modafinil's primary mechanism. Critical for stimulant/wakefulness pharmacology. |
| **TM** (tuberomammillary) | Sole source of brain histamine. Wake/arousal. H3 antagonists are active nootropic targets (pitolisant, ciproxifan). Directly relevant to DeSci/nootropics. |
| **PA** (paraventricular) | CRH, oxytocin, vasopressin. HPA stress axis hub. Relevant for anxiolytics, stress-cognition, cortisol/stress paradigms. |

**Skip:** SCh (circadian — rarely activated in task fMRI), SO (vasopressin — endocrine, not cognitive), VM/DM/AN (feeding/metabolic — off-topic), MPO (thermoregulation), AH/PE/DP (too small, too general).

**L/R merging:** Average both sides together into single parcels (LH_l + LH_r → LH), same convention as Brainstem Navigator.

**→ 6 new parcels** (LH_L, LH_R, TM_L, TM_R, PA_L, PA_R; or 3 merged: LH, TM, PA).

---

## 3. Overlap Handling

Neudorfer includes **NBM, SN, RN, STN** — already in Bianciardi or Tian.

| Strategy | Implementation |
|----------|----------------|
| **Preferred** | Use Neudorfer **only for hypothalamic nuclei**; use Zaborszky **only for basal forebrain**. Exclude Neudorfer NBM, SN, RN, STN from our atlas. |
| **Alternative** | Add Neudorfer hypothalamic nuclei only (LH, TM, PA, SCh, PH, VM); all basal forebrain from Zaborszky. |

---

## 4. Parcel Count Impact

| Component | Parcels |
|-----------|---------|
| Glasser | 360 |
| Tian S2 | 32 |
| Brainstem Navigator | ~64 |
| Zaborszky (Ch1-2 + Ch4) | 4 |
| Neudorfer (LH + TM + PA) | 6 |
| **Total** | **~427 parcels** (actual: 360+32+32+3 from Glasser+Tian+Brainstem+BFB/Hyp) |

(L/R merge reduces to 5 parcels → 461 total.)

This covers the four missing neuromodulatory sources: cortical acetylcholine (Ch4), hippocampal acetylcholine (Ch1-2), orexin (LH), histamine (TM), plus the stress axis hub (PA). Serotonin, dopamine, norepinephrine, and pontine acetylcholine are already in Bianciardi brainstem parcels.

---

## 5. Practical Considerations

| Issue | Notes |
|-------|-------|
| **Resolution** | At 2–3 mm fMRI, expect ~5–20 voxels per nucleus. Noisy but usable; better for PET and pharmacological fMRI. |
| **Template alignment** | Both atlases are MNI; Neudorfer uses MNI152 2009b (matches fMRIPrep). May need resampling to pipeline 2 mm reference. |
| **Cache rebuild** | Full cache rebuild required when parcel count changes. |
| **Priority order** | Glasser > Tian > Brainstem > Basal forebrain > Hypothalamus (later layers fill only where prior layers are 0). |

---

## 6. Implementation Checklist

- [x] Obtain Zaborszky NIfTIs (JuBrain or siibra); threshold to binary masks
- [x] Obtain Neudorfer hypothalamic NIfTIs (Lead-DBS or Scientific Data supplementary)
- [x] Add `_fetch_zaborszky_basal_forebrain()` and `_fetch_neudorfer_hypothalamus()` to `build_combined_atlas.py`
- [x] Merge L/R into single parcels for both atlases (same logic as `_merge_brainstem_nuclei`)
- [x] Implement overlap exclusion: use Neudorfer only for hypothalamic nuclei (LH, TM, PA); Zaborszky for basal forebrain
- [x] Add new atlas option: `glasser+tian+brainstem+bfb+hyp`
- [x] Update `parcellation.py` and downstream configs for new parcel count
- [ ] Rebuild all map caches (after obtaining atlas files)
