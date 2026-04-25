# NeuroLab Problem Statement

## The problem

- **Existing tools** (NeuroSynth, NeuroQuery, Nilearn, JuSpace, DANDI, etc.) each do one thing well but require expertise, coding, or stay within a single database.  
- **Non-experts** (e.g. students, independent researchers, and technically curious users) cannot easily ask: â€œWhich brain regions does modafinil affect compared to caffeine?â€ or â€œWhere are 5-HT2A receptors densest, overlaid with working-memory activation?â€ and get a single, evidence-aware answer.  
- **Map provenance and evidence strength** are often opaque: users may treat predictive or coordinate-derived maps as if they were voxelwise effect-size maps, leading to overinterpretation (â€œblobologyâ€).

## What we solve

- **Single natural-language interface** over multiple open neuroscience data sources (Neurosynth, NeuroVault, NeuroQuery, receptor atlases, etc.).  
- **Evidence-tiered outputs** so users always see whether a map is direct image evidence (Tier A), coordinate-based (Tier B), or predictive/hypothesis (Tier C), plus â€œHow to validateâ€ guidance.  
- **3D brain explorer** and map comparison with spatially-constrained nulls (e.g. spin tests), so users can explore and compare without writing code.  
- **Protocol generation** (hypothesis â†’ experiment â†’ analysis plan) via a reproducible analysis layer, with guardrails and interpretation checklists.

## What we do not solve (MVP)

- Full raw-data preprocessing (fMRIPrep, etc.) in â€œchat time.â€  
- Clinical decision support, diagnosis, or treatment recommendation.  
- Causal claims from spatial overlap alone.  
- Providing or recommending controlled substances.

---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3 Â§1â€“2](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md).


