# NeuroLab Constraints

## Technical constraints

- **Integration-optional:** NeuroLab must support standalone use and should integrate cleanly with external orchestrators, analysis sandboxes, and viewer-manifest workflows.
- **Evidence Tier mandatory:** Every map exposed to users must have an Evidence Tier (A/B/C) and, for B/C, â€œHow to validateâ€ guidance. No unlabeled maps.
- **Provenance mandatory:** Export-eligible artifacts include dataset + version, algorithm + parameters, coordinate space, and content-addressing (hash) where applicable.
- **License separation:** Data services must separate ODbL, CC0, and BYOL (user uploads) so license obligations can be enforced and attributed correctly.
- **Standard spaces (MVP):** Volumetric MNI152 (e.g. 2 mm); surfaces fsaverage/Conte69. NIfTI (volumes), GIfTI (surfaces), glTF/OBJ for viewer meshes.

## Legal / compliance constraints

- **ODbL (Neurosynth):** Produced Work vs Derivative Database distinction; attribution and share-alike where required. Engineering must support blocking or attributing correctly.
- **AGPL:** If any server component is AGPL and modified, network use may trigger source-offer obligations. Keep clear boundaries between AGPL and non-AGPL services.
- **No medical endorsement:** Product and token positioning must not imply medical or clinical endorsement.
- **Attribution:** Every export must support required attribution text for upstream datasets and algorithms.

## Operational constraints

- **Analysis runs where the user has rights:** Preferred model is bring-your-own-execution or a reference container (local/cluster). Hosted UI can remain a thin client that displays produced works and manifests.
- **Cost/secondary use:** Capability first; cost-at-scale and release policy decided separately. Test in small environments before broad release.

---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3 Â§4, 6.3, 8](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md).

