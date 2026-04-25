# NeuroLab Interfaces (Artifact Contracts)

This document defines the **canonical artifact contracts** for NeuroLab. Implementation must conform; changes here require review and updated tests.

## 1. Viewer manifest (output of `neuro.viewer.manifest` / `neuro.export_viewer_bundle`)

- **layers:** List of volume/surface/ROI layers; each has:
  - `type` (volume | surface | roi)
  - `url` or `content_hash` (content-addressed)
  - `space` (e.g. MNI152, fsaverage)
  - `colormap`, `threshold`, `opacity` (UI hints)
- **transforms:** Affine and/or surface space mapping as needed for alignment.
- **provenance:** For each layer or composite:
  - `dataset_id`, `version`
  - `algorithm`, `parameters`
  - `evidence_tier` (A | B | C)
  - `attribution_text` (required for ODbL/CC-BY etc.)
- **metadata:** `software_versions`, `hash_of_inputs` (optional but recommended).

## 2. Map registry entry (dataset registry)

- **id:** Unique dataset/source id.
- **modality:** e.g. CBMA, IBMA, receptor_atlas, predictive.
- **cognitive_terms:** Optional mapping (term → map id or query params).
- **license:** ODbL | CC0 | CC-BY | BYOL | non_commercial | DUA.
- **attribution_required:** Boolean; if true, export must include attribution text.
- **transform_space:** MNI152 | fsaverage | Conte69 | other.
- **access_tier:** 1 (ready-to-go) | 2 (guided/BYOD) for data access only (do not conflate with Evidence Tier A/B/C).
- **provenance_hash:** Content-addressed hash for cache/dedup.

(Full schema may extend; see [neurolab_implementation_guide §5](../../../docs/external-specs/neurolab_implementation_guide.md).)

## 3. Tool response (plugin → orchestrator / Data Analyst)

Every tool that returns maps or figures must include:

- **artifacts:** Figures (PNG/SVG), tables (CSV), map refs (URL or hash).
- **manifest:** Same structure as viewer manifest where applicable (layers, provenance, evidence_tier, attribution).
- **reason_codes:** Optional; if a step was skipped or failed, stable reason code (e.g. `license_blocked`, `missing_dataset`).

## 4. Evidence Tier (required on every user-facing map)

- **A:** Direct image evidence (IBMA / shared statistical maps).  
- **B:** Coordinate-based evidence (CBMA).  
- **C:** Predictive / model-based hypothesis.  

Tier B/C must have “How to validate” (min experiment, analysis plan, alternatives, links). Stored in manifest or report.

## 5. Run audit / reconstruction

For traceability (SR-10):

- **run_config:** Snapshot of config (dataset versions, tool versions, parameters).
- **artifact_lineage:** Which inputs produced which outputs; hashes where applicable.
- **audit_events:** Optional list of decisions (e.g. license check passed, tier assigned).

Implementation details (e.g. exact JSON schema) can live in [implementation](../implementation/) specs; see [Schema Index](../implementation/SCHEMA_INDEX.md) for canonical references. This document defines the **contract boundaries** that must not be broken.

---

**See also:** [architecture.md](architecture.md), [components.md](components.md).  
**Canonical reference:** [NeuroLab_Plugin_Spec_v0.3 §6.4, 7.3](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md).
