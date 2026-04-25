# NeuroLab Personas

## Primary personas (MVP)

### Student / learner

- Asks: “What brain regions are implicated in spatial attention while taking methylphenidate (Ritalin)?”
- Needs: brain map(s) with uncertainty + provenance, short explanation (meta-analysis vs prediction), 3D view, validation pathway (what study would confirm/refute).
- Does **not** have: deep neuroimaging or stats background; expects plain-language answers.

### Non-expert researcher (e.g. nootropics community)

- Asks: “Which brain areas might be impacted by SSRI use?” or “Where are D2 receptors densest?”
- Needs: cautious synthesis with limitations, receptor overlays, drug comparisons.
- Has: conceptual understanding of neurotransmitters/receptors; may not know coordinate systems or preprocessing.

### Scientist (expert)

- Uploads own results (thresholded clusters or unthresholded maps); wants to contextualize against public meta-analytic resources.
- Needs: map comparison, null models (spin tests), protocol generation (experiment + analysis plan).
- Has: familiarity with fMRI/neuroimaging; may use NeuroLab to avoid re-running pipelines for standard comparisons.

## Secondary / out-of-scope

- **Clinician** — we do not support clinical decision support or treatment recommendation.
- **Raw-data analyst** — full BIDS→fMRIPrep workflows are Phase 2, not MVP.

---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3 §1.1](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md).
