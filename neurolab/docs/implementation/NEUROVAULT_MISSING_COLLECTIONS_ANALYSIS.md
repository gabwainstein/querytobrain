# NeuroVault: 15 Collections with 0 Output (Averaging Verification)

These AVERAGE_FIRST collections produced no terms in the current cache. Root cause: **images likely not downloaded** in the curated data directory — a data availability issue, not a code bug.

| ID | Name | What it is | Expected contrasts | Worth recovering? |
|----|------|------------|--------------------|-------------------|
| **426** | False belief vs physical reasoning | Theory-of-mind (ToM) task (OpenfMRI ds000109), 48 subjects | ~2–3 contrasts | ✓ Yes — classic ToM paradigm |
| **437** | Autism functional subnetworks | Consensus clustering (network consistency, not activation) | ~10 subnetwork maps | ✓ Yes — relabel as connectivity/network |
| **504** | Single-subject thermal pain (NPS) | Individual pain response maps, multiple intensity levels | ~4–6 intensity contrasts | ✓ Yes — pain processing, ties to Neurologic Pain Signature (NPS) |
| **1620** | Depression resting-state and task | Clinical depression functional connectivity (FC)/task maps | ~5–10 contrasts | ✓ Yes — clinical relevance |
| **2503** | Social Bayesian inference | Social learning task fMRI | ~3–5 contrasts | ✓ Yes — social cognition |
| **3887** | (Tier 3 — unspecified) | Unknown | Unknown | Check NeuroVault |
| **4804** | Fusiform-network coupling | Face processing / fusiform connectivity | ~3–5 contrasts | ✓ Yes — face perception |
| **6825** | Schizophrenia deformation | Structural deformation maps in schizophrenia | ~2–5 maps | ✓ Yes — relabel as structural/schizophrenia |
| **11646** | ASD emotional egocentricity | 21 autism spectrum disorder (ASD) vs 21 controls, emotional egocentricity bias | ~3–5 contrasts | ✓ Yes — clinical + emotion |
| **12992** | Nicotine abstinence threat | Smoking/nicotine + threat processing | ~3–5 contrasts | ✓ Yes — pharmacological relevance |
| **13474** | Tensorial ICA for MDD (major depressive disorder) | ICA decomposition maps for depression | ~10–20 components | ✓ Yes — relabel as ICA/depression components |
| **13705** | (Tier 3 language) | Language task fMRI | ~3–5 contrasts | ✓ Yes — language processing |
| **16284** | IAPS emotional valence | 56 subjects, positive/negative/neutral International Affective Picture System (IAPS) blocks | ~3 contrasts (pos/neg/neu) | ✓ Yes — emotion, well-controlled |
| **19012** | Incentive-boosted inhibitory control | 76 adolescents, reward × inhibition | ~4–6 contrasts | ✓ Yes — reward + cognitive control |
| **20510** | Lesions causing psychosis | Lesion maps associated with psychosis | ~5–10 maps | ✓ Yes — relabel as lesion/psychosis |

## Summary

- **All 15 worth recovering** — include structural, ICA, connectivity, and lesion maps with appropriate labels
- **Task fMRI** (426, 504, 1620, 2503, 4804, 11646, 12992, 13705, 16284, 19012): use contrast-derived labels
- **Structural/ICA/connectivity/lesion** (437, 6825, 13474, 20510): use domain-specific labels (e.g. "autism functional network", "schizophrenia structural deformation", "major depressive disorder ICA component", "psychosis lesion map")
- **Total impact**: ~40–70 averaged maps → **0.3–0.5% of training set** (~14,815 terms)

## Priority for future data refresh

1. **16284** — IAPS emotional valence (clean emotion paradigm)
2. **504** — pain / Neurologic Pain Signature (NPS)
3. **12992** — pharmacological relevance

Not worth blocking training for; add during next curated download refresh.
