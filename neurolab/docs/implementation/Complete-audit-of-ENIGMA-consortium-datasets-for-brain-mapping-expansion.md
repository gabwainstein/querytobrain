# Complete audit of ENIGMA consortium datasets for brain mapping expansion

**Your current 12-map pipeline can expand to approximately 100+ disorder×measure maps.** Beyond cortical thickness and subcortical volume for your 13 disorders, ENIGMA has published cortical surface area effect sizes for most of those same disorders, DTI white matter tract effect sizes for 7+ disorders, laterality maps for 4 conditions, gyrification data for PTSD, and entirely new disorder maps for PTSD, substance use disorders, four anxiety subtypes, obesity, suicidality, and multiple copy number variants. Nearly all published ENIGMA data uses Cohen's d with standard errors in Desikan-Killiany parcellation, making integration straightforward.

---

## What the ENIGMA Toolbox provides programmatically

The ENIGMA Toolbox (v2.0.3, February 2024) offers the most efficient entry point, covering **8 disorders** via `load_summary_stats()` in Python/MATLAB. Every dataset includes **Cohen's d** (`d_icv`), **standard errors** (`se_icv`), confidence intervals, sample sizes, and FDR-corrected p-values. The toolbox does **not** include DTI, gyrification, or laterality data.

| Disorder | Toolbox argument | CT (DK-68) | SA (DK-68) | SubVol (16) | Age splits | N cases / N controls |
|---|---|---|---|---|---|---|
| 22q11.2 deletion | `'22q'` | ✅ | ✅ | ✅ | — | 474 / 470 |
| ADHD | `'adhd'` | ✅ | ✅ | ✅ | Adult, Child | 2,246 / 1,934 |
| ASD | `'asd'` | ✅ | ❓ | ✅ | — | 1,571 / 1,651 |
| Bipolar disorder | `'bipolar'` | ✅ | ✅ | ✅ | — | 2,447 / 4,056 |
| Epilepsy | `'epilepsy'` | ✅ | ❌ | ✅ | LTLE, RTLE, GGE, All | 2,149 / 1,727 |
| Major depression | `'depression'` | ✅ | ✅ | ✅ | Adult, Adolescent | 2,148 / 7,957 |
| OCD | `'ocd'` | ✅ | ✅ | ✅ | Adult, Child | 1,905 / 1,760 |
| Schizophrenia | `'schizophrenia'` | ✅ | ✅ | ✅ | — | 4,474 / 5,098 |

This yields roughly **37–44 distinct statistical tables** (counting age and subtype splits). Usage is straightforward:

```python
from enigmatoolbox.datasets import load_summary_stats
stats = load_summary_stats('depression')
ct_adult = stats['CortThick_case_vs_controls_adult']
d_values = ct_adult['d_icv']    # Cohen's d
se_values = ct_adult['se_icv']  # Standard errors
```

**Critical gap in your current pipeline**: The toolbox already includes **cortical surface area** maps for 22q, ADHD, bipolar, MDD, OCD, and schizophrenia — these represent ~6 additional maps you may not yet be using if you only extracted CT and SubVol.

---

## New disorders with published effect sizes not in your pipeline

These ENIGMA papers report Cohen's d effect sizes in Desikan-Killiany parcellation and are extractable from supplementary tables. Each represents a potential new addition to your platform.

### PTSD — three complementary papers

ENIGMA-PGC PTSD is one of the most data-rich working groups outside the toolbox. **Logue et al. (2018)** in *Biological Psychiatry* (DOI: 10.1016/j.biopsych.2017.09.006) reported subcortical volumes across **794 PTSD cases and 1,074 trauma-exposed controls** from 16 cohorts, finding hippocampal volume reduction of **d = −0.17**. **Wang et al. (2021)** in *Molecular Psychiatry* (DOI: 10.1038/s41380-020-00967-1) covered cortical volumes for **1,379 cases and 2,192 controls** across 32 sites in DK parcellation. The DTI paper by **Dennis et al. (2021)** in *Molecular Psychiatry* (DOI: 10.1038/s41380-019-0631-x) reported FA and RD effect sizes across JHU white matter tracts in **~3,047 adults**. Cohen's d with standard errors available in all three papers' supplementary materials.

### Substance use disorders — ENIGMA-Addiction

**Mackey et al. (2019)** in *American Journal of Psychiatry* (DOI: 10.1176/appi.ajp.2018.17040415) performed a mega-analysis of **~2,277 substance-dependent individuals and ~963 controls** with substance-specific breakdowns for alcohol, nicotine, cocaine, methamphetamine, and cannabis. Measures include cortical thickness and subcortical volumes in DK parcellation. **Alcohol dependence showed the largest effects.** Cohen's d values are reported for both substance-general and substance-specific contrasts.

### Anxiety disorders — four separate conditions

The ENIGMA-Anxiety Working Group has been remarkably productive, publishing separate papers for each anxiety subtype:

- **GAD**: Harrewijn et al. (2021) in *Translational Psychiatry* (DOI: 10.1038/s41398-021-01622-1) — CT, SA, SubVol from 28 sites. Main finding: no significant main effects; diagnosis-by-sex interactions only.
- **Social anxiety disorder**: Groenewold et al. (2023) in *Molecular Psychiatry* (DOI: 10.1038/s41380-022-01933-9) — subcortical volumes for **1,115 SAD cases and 2,775 controls** from 37 samples. Putamen d = −0.10, pallidum d = +0.13 in adults.
- **Panic disorder**: ENIGMA-Anxiety (2025) in *Molecular Psychiatry* (DOI: 10.1038/s41380-025-03376-4) — CT, SA, SubVol for **1,146 cases and 3,778 controls** from 28 sites. Effect sizes d = −0.08 to −0.13.
- **Specific phobia**: Hilbert et al. (2024) in *American Journal of Psychiatry* (DOI: 10.1176/appi.ajp.20230032) — CT, SA, SubVol for **1,452 cases and 2,991 controls** from 31 studies, with animal and blood-injection-injury subtype breakdowns.

### Suicidality — transdiagnostic dimension

ENIGMA-STB has published at least three papers. **Rentería et al. (2017)** in *Translational Psychiatry* (DOI: 10.1038/tp.2017.84) covered subcortical volumes in MDD with suicidal symptoms (**N = 451 suicidal, 650 non-suicidal MDD, 1,996 controls**). A cortical paper (~2022) in *Biological Psychiatry* examined suicide attempt in **18,925 participants** across 18 cohorts. A youth-focused paper (2022) in *Molecular Psychiatry* (DOI: 10.1038/s41380-022-01734-0) analyzed 21 international studies. All report Cohen's d in DK parcellation.

### Obesity — studied within ENIGMA-MDD

**Opel et al. (2021)** in *Molecular Psychiatry* (DOI: 10.1038/s41380-020-0774-9) reported CT, SA, and SubVol for obese vs. normal-weight individuals in **6,420 participants** from 28 ENIGMA-MDD sites. Maximum effect was **d = −0.33** for left fusiform thickness. Effect sizes available in supplementary tables for all DK regions.

---

## DTI white matter tract data — a major untapped modality

ENIGMA-DTI papers use the **JHU white matter atlas with 25 ROIs** (24 regional tracts + whole-skeleton average) and report Cohen's d for FA, MD, RD, and AD. This modality alone could add **~28 new maps** (7 disorders × 4 DTI metrics) to your pipeline.

| Disorder | Paper | Year | Journal | N cases / N controls | FA | MD | RD | AD |
|---|---|---|---|---|---|---|---|---|
| Schizophrenia | Kelly et al. | 2018 | Mol Psychiatry | 1,963 / 2,359 | ✅ d=−0.42 avg | ✅ | ✅ | ✅ |
| Bipolar disorder | Favre et al. | 2019 | Neuropsychopharmacology | 1,482 / 1,551 | ✅ d=−0.46 CC | ✅ | ✅ | ✅ |
| MDD | van Velzen et al. | 2020 | Psychol Med | ~1,300 / ~1,600 | ✅ | ✅ | ✅ | ✅ |
| OCD | Piras et al. | 2021 | Transl Psychiatry | Consortium scale | ✅ | ✅ | ✅ | ✅ |
| PTSD | Dennis et al. | 2021 | Mol Psychiatry | ~1,300 / ~1,700 | ✅ | — | ✅ | — |
| TBI | Dennis et al. | 2018 | Hum Brain Mapp | Preliminary | ✅ | ✅ | ✅ | ✅ |
| 22q11.2 deletion | Villalón-Reina et al. | 2019 | — | Consortium scale | ✅ | ✅ | ✅ | ✅ |

The **cross-disorder DTI comparison** by **Kochunov et al. (2022)** in *Human Brain Mapping* (DOI: 10.1002/hbm.24998) is especially valuable — Table 2 compiles Cohen's d values for all 24 JHU tract ROIs across schizophrenia, bipolar, MDD, OCD, PTSD, TBI, and 22q, providing a single source for 7 disorders. **This paper alone could populate your entire DTI pipeline.** Standard errors and 95% CIs are included.

---

## Laterality, gyrification, and specialty modalities

**ENIGMA-Laterality** has published hemisphere asymmetry maps that could add a unique dimension to your platform. **Kong et al. (2018)** in *PNAS* (DOI: 10.1073/pnas.1718418115) mapped cortical thickness and surface area asymmetry in **17,141 healthy individuals** across 99 datasets using DK parcellation, with scripts on GitHub (github.com/Conxz/neurohemi). Disorder-specific laterality papers exist for **MDD** (de Kovel et al., 2019, *Am J Psychiatry*), **ASD** (Postema et al., 2019, *Nature Communications*), and **OCD** (Kong et al., 2020, *Biological Psychiatry*).

For **cortical gyrification**, the first large-scale ENIGMA paper is **Hussain et al. (2026)** on PTSD in *Biological Psychiatry: Global Open Science*, covering 789 PTSD cases and 1,087 controls across 24 sites using FreeSurfer local gyrification index in DK parcellation. This is currently the only ENIGMA gyrification dataset available.

---

## Copy number variants beyond 22q11.2

The ENIGMA-CNV working group has published structural effect sizes for three additional genomic deletions/duplications, though these use regression β coefficients rather than Cohen's d due to dose-response modeling:

- **15q11.2 BP1-BP2**: van der Meer et al. (2020), *JAMA Psychiatry* — 203 deletion carriers, 306 duplication carriers, 45,247 noncarriers. CT, SA, SubVol in DK parcellation.
- **16p11.2 distal**: Sønderby et al. (2020), *Molecular Psychiatry* — 12 deletions, 12 duplications, 6,882 noncarriers. SubVol only.
- **1q21.1 distal**: Sønderby et al. (2021), *Translational Psychiatry* — SubVol and cortical measures.
- **Multi-CNV analysis**: Kumar et al. (2023), *American Journal of Psychiatry* — 675 carriers across 7 CNV loci vs. 782 controls. Effect sizes **2–6× larger** than idiopathic psychiatric disorders.

---

## Cross-disorder resources that compile multiple ENIGMA datasets

Several papers serve as convenient single-source compilations. **Cheon et al. (2022)** in *Psychiatry and Clinical Neurosciences* (DOI: 10.1111/pcn.13323) compiled ranked Cohen's d effect sizes across **four modalities** (SubVol, CT, SA, DTI FA) for schizophrenia, bipolar, MDD, and 22q — a critical reference for cross-disorder comparison. **Patel et al. (2021)** in *JAMA Psychiatry* (DOI: 10.1001/jamapsychiatry.2020.2694) aggregated cortical thickness profiles for 6 disorders (ADHD, ASD, BD, MDD, OCD, SCZ) from 145 cohorts with virtual histology cell-type mapping. **Opel et al. (2020)** in *Biological Psychiatry* (DOI: 10.1016/j.biopsych.2020.04.027) found cross-disorder correlations of **r = 0.44–0.78** among MDD, BD, SCZ, and OCD structural profiles. **Hettwer et al. (2022)** in *Nature Communications* identified transdiagnostic co-alteration networks across 12,024 patients and 18,969 controls.

---

## Complete working group status inventory

Of ENIGMA's **60+ working groups**, the following status applies to those relevant for your platform:

| Status | Working groups |
|---|---|
| **In ENIGMA Toolbox** (8) | 22q, ADHD, ASD, Bipolar, Epilepsy, MDD, OCD, Schizophrenia |
| **Published d-maps, extractable** (~10) | Parkinson's, Anorexia, Antisocial/CD, PTSD, Addiction/SUD, GAD, Social Anxiety, Panic Disorder, Specific Phobia, Suicidality, Obesity |
| **Published DTI d-maps** (7) | Schizophrenia, Bipolar, MDD, OCD, PTSD, TBI, 22q |
| **Published laterality maps** (4) | Healthy population, MDD, ASD, OCD |
| **Active WG, no d-maps yet** | Tourette, Sleep/Insomnia, Irritability, Chronic Pain, Dissociation, Stroke Recovery, Ataxia, FTD, Diabetes, Cancer/Chemo |
| **No WG identified** | Borderline PD, Multiple Sclerosis, Huntington's, Preterm birth |

The **ENIGMA-Tourette** working group has a rationale paper (Paschou et al., 2022) but no structural brain paper yet. **ENIGMA-HIV** (Nir et al., 2021, *JAMA Network Open*) analyzed subcortical volumes in 1,295 HIV+ adults but without an HIV-negative control group, limiting its use for case-control effect size maps. **ENIGMA-Stroke Recovery** focuses on recovery biomarkers rather than case-control structural differences.

---

## Data access methods at a glance

| Source | URL | What's there | Format |
|---|---|---|---|
| ENIGMA Toolbox | github.com/MICA-MNI/ENIGMA | 8 disorders, CT/SA/SubVol | Python/MATLAB API, CSV |
| ENIGMA Viewer | enigma-viewer.org | Interactive effect size visualization | Web portal |
| ENIGMA GWAS downloads | enigma.ini.usc.edu/research/download-enigma-gwas-results/ | GWAS summary stats for brain volumes | Text/CSV (application required) |
| ENIGMA-git GitHub | github.com/ENIGMA-git | DTI protocols, analysis scripts, templates | R scripts, NIfTI |
| Supplementary tables | Journal websites | All published Cohen's d + SE per region | PDF, Excel, CSV |
| Kochunov 2022 Table 2 | Paper supplement | Cross-disorder DTI FA for 7 disorders × 25 tracts | Tabular |
| Cheon 2022 | Paper supplement | Cross-disorder CT/SA/SubVol/DTI compiled | Tabular |

---

## Recommended expansion roadmap

Your most efficient path from 12 maps to 100+ maps follows three tiers of effort.

**Tier 1 — Immediate, programmatic (add ~15 maps in hours):** Load surface area effect sizes from the ENIGMA Toolbox for 22q, ADHD, bipolar, MDD, OCD, and schizophrenia. These are already in the toolbox but you may only be extracting CT and SubVol. Also extract epilepsy subtype splits (LTLE, RTLE, GGE) and age-group splits for ADHD, MDD, and OCD.

**Tier 2 — Supplementary table extraction (add ~25–30 maps in days):** Extract Cohen's d tables from published ENIGMA papers for PTSD (3 papers), addiction/SUD (substance-specific breakdowns), Parkinson's, panic disorder, specific phobia, social anxiety, suicidality, and obesity. All use DK parcellation with standard errors. Prioritize PTSD and the anxiety subtypes as the largest untapped datasets.

**Tier 3 — DTI pipeline (add ~28+ maps in a week):** Build a DTI tract-level module using Kochunov et al. (2022) as the primary source for 7 disorders × 4 DTI metrics (FA, MD, RD, AD) across 25 JHU ROIs. This requires adding a new parcellation scheme (JHU white matter atlas) to your platform but represents the single largest expansion opportunity.

The total realistic expansion reaches **~80–100 disorder×measure×parcellation maps** with published Cohen's d and standard errors, covering structural MRI (CT, SA, SubVol), diffusion MRI (FA, MD, RD, AD), laterality, and gyrification across 20+ distinct conditions. The limiting factor is not data availability but rather the engineering effort to extract supplementary tables and integrate the JHU white matter parcellation alongside your existing DK cortical atlas.

## Conclusion

ENIGMA's data footprint is substantially larger than what the current 12-map pipeline captures. The most impactful additions are **surface area maps** already sitting in the toolbox (zero extraction effort), the **PTSD and anxiety disorder families** (4+ new conditions with large samples), and the **DTI white matter modality** (7 disorders with tract-level effect sizes from a single cross-disorder paper). The ENIGMA-Anxiety Working Group's recent output (2021–2025) is particularly notable — they have produced condition-specific maps for GAD, social anxiety, panic, and specific phobia that did not exist when most ENIGMA platform tools were built. Together with the CNV datasets and cross-disorder compilations, these resources position a comprehensive neuroscience prediction platform to cover the majority of psychiatric and neurological conditions that have been studied at consortium scale.