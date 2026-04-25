# querytobrain

Personal neuroscience agent and brain-mapping research platform: turn natural-language queries into brain maps with provenance and enrichment.

## Contents

| Path | Purpose |
|------|--------|
| **neurolab/** | Primary subproject — neuroscience map retrieval, enrichment, and text-to-brain training. Start at [neurolab/README.md](neurolab/README.md) and [neurolab/docs/README.md](neurolab/docs/README.md). |
| **twitter-agent-kit/** | Reference social / research agent tooling (ElizaOS-based). |
| **docs/** | Reference material retained for implementation context. |

## What this repo is

A reusable neuroscience research codebase:

- Brain-map retrieval, enrichment, and text-to-brain modeling (`neurolab/`)
- Agent tooling for research-facing assistants and social integrations (`twitter-agent-kit/`)
- Reference docs retained where they carry useful implementation detail

## NeuroLab — quick links

- **Docs spine:** [neurolab/docs/README.md](neurolab/docs/README.md) — product, architecture, ADRs, pipeline slices, process
- **Training pipelines:** `python neurolab/scripts/train_neurolab_model.py --pipeline <fmri_cbma|gene_receptor|multimodal>`
- **Full pipeline build:** `python neurolab/scripts/run_full_cache_build.py`
- **Query (after build):** `python neurolab/scripts/query.py "working memory" --use-embedding-model neurolab/data/embedding_model --cache-dir neurolab/data/merged_sources`

See [HANDOVER.md](HANDOVER.md) for a full audit-and-run guide.

## Contributing

Follow the process in [neurolab/docs/process/](neurolab/docs/process/) (definition of ready/done, documentation rules, issue types). Work is documentation-first: product and architecture docs constrain implementation.
