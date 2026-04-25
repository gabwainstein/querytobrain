# Supabase Deployment Plan

This plan describes how to host NeuroLab’s metadata, maps, and models using Supabase (Postgres + storage) plus lightweight services for inference.

## 1. Goals

1. Provide a multi-tenant API for researchers and biohackers to query NeuroLab without rebuilding caches locally.
2. Persist canonical metadata (datasets, term descriptions, provenance) in relational tables.
3. Store large artifacts (maps, embeddings, trained weights) in Supabase Storage (or external object storage) with signed URLs.
4. Keep inference code (decoder, embedding model) in a stateless service that reads from Supabase and local cache volumes.

## 2. Core artifacts to host

| Artifact | Notes | Storage target |
|----------|-------|----------------|
| Dataset registry (id, modality, license, evidence tier, version, provenance hash) | Matches schema in `SCHEMA_INDEX.md` (dataset registry entry). | Postgres table `datasets`. |
| Map metadata (term id, label, evidence tier, source, map type, term_sources weights) | Derived from merged cache (`term_vocab.pkl`, `term_sources.pkl`, `term_map_types.pkl`). | `term_metadata` table. |
| Map files (parcellated vectors, compressed `.npz`) | Up to ~12k maps; each vector ~427 floats. | Supabase Storage bucket `maps/` (content-addressed). |
| Viewer manifests & provenance logs | JSON manifest per query run. | Table `manifests` + Storage (optional) for full JSON. |
| Ontologies & expansion graph | `neurolab/data/ontologies/*.owl|obo|ttl` and derived meta-graph (see `ONTOLOGY_ENRICHMENT_IMPLEMENTATION_CHECKLIST.md`). | Storage bucket `ontologies/` + table `ontology_nodes`. |
| Pharma schema | `neurolab/data/neurovault_pharma_schema.json` (versioned). | Table `pharma_collections` (fields from JSON) plus Storage for raw JSON. |
| Trained embedding model | `neurolab/data/embedding_model` (config, weights, PCA, training history). | Storage bucket `models/embedding/<version>/`. |
| Encoder embeddings / guardrail vectors | `training_embeddings.npy`, `training_terms.pkl`. | Storage `models/embedding/<version>/embeddings/`. |
| Receptor knowledge base | `receptor_knowledge_base.json`, `receptor_gene_names_v2.json`. | Table `receptor_genes` + Storage for JSON. |
| Audit snapshots | Files under `neurolab/docs/implementation/*AUDIT*.md`. | GitHub only (no Supabase storage required) but link from dashboard for context. |

## 3. Database schema (Supabase Postgres)

### 3.1 `datasets`
- `id` (text, PK)
- `name`
- `modality` (enum: ibma, cbma, receptor, structural, predictive, etc.)
- `evidence_tier` (enum A/B/C)
- `license`
- `access_tier`
- `provenance_hash`
- `version`
- `last_built_at`

### 3.2 `term_metadata`
- `term_id` (uuid or hash, PK)
- `label`
- `source_dataset_id` (FK -> `datasets`)
- `map_type` (fmri_activation, structural, pet_receptor, gene_expression, etc.)
- `evidence_tier`
- `term_source_weight` (jsonb: {direct:1.0,...})
- `map_storage_path` (Storage key or external URL)
- `sha256`
- `n_parcels`
- `created_at`

### 3.3 `manifests`
- `id`
- `query_text`
- `evidence_tier_summary`
- `manifest_json` (jsonb)
- `storage_path` (optional)
- `created_by`
- `created_at`

### 3.4 `pharma_collections`
- `collection_id`
- `name`
- `semantic_label`
- `description`
- `label_prefix`
- `source_bucket`
- `exclude` (bool)
- `metadata_json`

### 3.5 `ontology_nodes` / `ontology_edges`
- `ontology_nodes`: `id`, `ontology`, `label`, `definition`, `parent_ids` (array), `ontology_version`.
- `ontology_edges`: `src`, `dst`, `relation`, `weight`.

### 3.6 `receptor_genes`
- `gene_symbol`
- `gene_name`
- `system`
- `category`
- `notes`
- `description`

## 4. Storage layout (Supabase Storage or S3-compatible)

```
maps/
  merged_sources/<sha256>.npz
  neuromaps/<...>.npz
models/
  embedding/<version>/
    config.pkl
    head_weights.pt
    training_embeddings.npy
    training_history.pkl
    split_info.pkl
ontologies/
  <ontology>/<version>/<file.owl>
schemas/
  neurovault_pharma_schema.json
  receptor_knowledge_base.json
```

Each stored object should include metadata headers (`dataset_id`, `evidence_tier`, `n_parcels`, etc.) for quick filtering.

## 5. Services / deployment components

1. **Metadata API (Supabase PostgREST/GraphQL):** Expose `datasets`, `term_metadata`, `pharma_collections`, `ontologies`.
2. **Inference service (FastAPI/Cloud Run):**
   - Mounts supabase storage (via signed URLs) or keeps hot caches on disk.
   - Loads `neurolab/data/embedding_model` weights from storage on startup.
   - Implements `query`, `predict_map`, `compare_maps` endpoints.
3. **Job runner / pipeline orchestrator:**
   - Rebuild caches via `run_full_cache_build.py` in a container.
   - Writes new map files + metadata to storage/db.
   - Records build logs in `build_history` table (optional).
4. **Monitoring & provenance:**
   - Each build writes a manifest to `manifests` table and uploads technology stack versions.

## 6. Steps to implement

1. **Schema migration:** Use Supabase SQL migrations to create tables above. Seed from existing JSON/PKL files.
2. **Storage sync:** Write scripts to upload `merged_sources` vectors, embedding models, ontologies.
3. **Metadata import:** Convert `term_vocab.pkl`, `term_sources.pkl`, `term_map_types.pkl` into CSV/SQL inserts for `term_metadata`.
4. **Service layer:** Containerize inference (`scripts/query.py` logic) with environment variables pointing to Supabase (for metadata) and storage (for maps/models).
5. **Access control:** Configure Supabase policies so public clients can read tier-1 data while sensitive caches (BYOL) remain private.
6. **CI/CD:** Add workflows that rebuild caches, sync to Supabase, and update the `datasets` + `term_metadata` tables whenever new data is ingested.

With this plan, ResearchAgent/NeuroLab can run as a hosted service where Supabase keeps the authoritative metadata/configs and storage holds large binaries, while the inference layer loads trained networks/embeddings as needed.***
