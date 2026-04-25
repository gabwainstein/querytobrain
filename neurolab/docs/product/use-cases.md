# NeuroLab Use Cases

## UC-1: Term or drug â†’ brain map(s)

- **Actor:** Student or non-expert researcher.  
- **Goal:** Get brain map(s) for a term (e.g. â€œworking memoryâ€) or drug (e.g. â€œmodafinilâ€) with Evidence Tier and short explanation.  
- **Flow:** User asks in natural language â†’ Orchestrator routes to NeuroLab tools â†’ map registry / NeuroSynth/NeuroQuery/NeuroVault â†’ map(s) + manifest + â€œHow to validateâ€ (if B/C) â†’ 3D viewer or inline preview.  
- **Success:** Map(s) returned in &lt;5â€“15 s; 100% have Evidence Tier and provenance.

## UC-2: Drug vs drug (or condition vs condition) comparison

- **Actor:** Non-expert or scientist.  
- **Goal:** Compare brain maps for two drugs/conditions (e.g. modafinil vs caffeine).  
- **Flow:** User asks â†’ Data Analyst runs map retrieval + comparison (e.g. spatial correlation, difference map, receptor correlation) â†’ figures + stats + manifest â†’ report with interpretation checklist.  
- **Success:** Difference map and/or correlation with receptor maps; clear Tier labels and limitations.

## UC-3: Receptor density lookup and overlay

- **Actor:** Non-expert or scientist.  
- **Goal:** â€œWhere are 5-HT2A receptors densest?â€ or â€œOverlay receptor map with working-memory activation.â€  
- **Flow:** User asks â†’ NeuroLab plugin serves receptor atlas (e.g. Hansen) + optional overlay with meta-analytic map â†’ viewer manifest.  
- **Success:** Correct overlay in standard space; caveats shown (receptor map â‰  activation).

## UC-4: Upload + contextualize (BYOL)

- **Actor:** Scientist.  
- **Goal:** Upload own map(s) and compare to public meta-analytic or receptor maps.  
- **Flow:** User provides map (or link) â†’ license-aware path (BYOL) â†’ comparison tools + null model â†’ report + manifest.  
- **Success:** User map never mixed with ODbL/CC0 in a way that violates license; attribution clear.

## UC-5: Hypothesis â†’ experiment + analysis plan

- **Actor:** Student or scientist.  
- **Goal:** Get a suggested experiment and analysis plan to test a hypothesis (e.g. from a Tier C map).  
- **Flow:** User asks â†’ Data Analyst generates protocol: design, confounds, model, prereg snippet, sample-size heuristics (educational).  
- **Success:** Structured protocol; no medical/clinical advice; â€œhow to validateâ€ aligned with Evidence Tier.

## UC-6: 3D explorer (inspect map + provenance)

- **Actor:** Any.  
- **Goal:** Explore brain map in 3D; see Evidence Tier and â€œWhat data generated this?â€  
- **Flow:** Viewer loads from manifest (layers, transforms, provenance) â†’ surface/volume exploration â†’ Tier badge + provenance panel.  
- **Success:** Correct rendering; Tier and provenance visible.

## UC-7: Stack guidance for community experiments

- **Actor:** Community member planning a multi-factor intervention or comparison.  
- **Goal:** Describe a combination of compounds (e.g. "modafinil, l-theanine, choline") and receive evidence-tiered cognitive + receptor maps plus "how to validate" suggestions.  
- **Flow:** User submits plain-language stack ??' NeuroLab resolves each ingredient to cached maps or predictive embeddings ??' merges into a report with Evidence Tier labels, receptor context, and suggested validation steps ??' manifest for Data Analyst follow-up.  
- **Success:** Report delivered in <15 s with Tier badges and receptor overlays; includes at least one validation checklist item for Tier B/C sections.

## UC-8: Compound comparison for DAO treasury decisions

- **Actor:** DAO research lead comparing candidate compounds or clinical collaborations.  
- **Goal:** Contrast two compounds/conditions (e.g., "ketamine vs psilocybin for cognitive flexibility") with quantitative map differences and receptor alignment to prioritize investments.  
- **Flow:** User specifies compounds ??' system retrieves existing maps and/or runs predictive decoder ??' computes spatial correlation/difference, receptor overlay, null-model stats ??' produces figures + manifest + interpretation checklist.  
- **Success:** Output includes numeric comparison (correlation/difference), receptor correlation summary, and evidence-tiered caveats; ready to share in governance discussions.

## UC-9: Community BYOL share-and-compare

- **Actor:** User uploading personal fMRI/PET-derived maps to contextualize results.  
- **Goal:** Upload privately-held map(s), compare against public NeuroLab datasets, and obtain attribution-safe manifests for community sharing.  
- **Flow:** User uploads map ??' BYOL pipeline stores in separate bucket with license note ??' comparison service finds nearest public maps/receptors and runs null models ??' manifest + report returned; user can optionally publish summary to DAO forums.  
- **Success:** User map never co-mingles licensing; manifest lists both private and public sources with Evidence Tiers; comparison metrics highlight overlap and limitations.

## UC-10: Pharma semantics exploration dashboard

- **Actor:** Community curator exploring pharmacological literature (e.g., dosing studies, challenge paradigms).  
- **Goal:** Filter curated neuropharma collections by semantic labels (dose, challenge, chronic, receptor focus) to quickly find relevant activation patterns.  
- **Flow:** User queries "dopamine challenge studies" ??' NeuroLab consults the pharma schema (`neurovault_pharma_schema.json`) and semantic embeddings ??' returns category-specific map sets with Tier badges and provenance, plus shortcuts to run enrichment or comparisons.  
- **Success:** Dashboard responds in <10 s with clustered collections, each linked to manifests; schema labels align with curated definitions and direct the user to the right maps.


---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3 Â§1.1](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md) and [neurolab_implementation_guide Â§6](../../../docs/external-specs/neurolab_implementation_guide.md).

