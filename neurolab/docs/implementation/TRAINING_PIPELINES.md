# Training Pipelines

This repo now exposes a neutral training entrypoint:

```bash
python neurolab/scripts/train_neurolab_model.py --pipeline <fmri_cbma|gene_receptor|multimodal>
```

The goal is to separate common MLP training runs by **data regime** without
breaking the existing trainer implementation in
`neurolab/scripts/train_text_to_brain_embedding.py`.

## Pipelines

### `fmri_cbma`

Use when the main supervision comes from:

- direct task / cognitive maps
- NeuroVault
- CBMA / neurocontext-like sources

Behavior:

- uses `merged_sources_cbma`
- enables the CBMA auxiliary head
- keeps settings focused on cognitive/fMRI map learning

### `gene_receptor`

Use when the main supervision comes from:

- curated abagen gene maps
- receptor / neuromaps maps
- residual biology-aware targets

Behavior:

- points at the curated tiered gene/receptor cache
- keeps genes on the **main head**
- enables receptor ranking consistency loss

### `multimodal`

Use when training should combine:

- cognitive / CBMA maps
- gene and receptor maps
- ontology / KG context
- multimodal regularization

Behavior:

- starts from the curated gene/receptor cache
- enables semantic KG context
- enables ontology alignment and KG teacher distillation
- keeps CBMA auxiliary supervision enabled

## Why this wrapper exists

`train_text_to_brain_embedding.py` contains a large amount of logic for many
sources and objectives. This wrapper is the first step toward separating MLP
training by data family while preserving backward compatibility.

## Relationship to the legacy trainer

- `train_text_to_brain_embedding.py` remains the source of truth for actual execution.
- `train_neurolab_model.py` selects a preset and forwards arguments.
- `neurolab/training/pipeline_presets.py` centralizes the default flag bundles.

## Override behavior

You can append additional flags after `--`:

```bash
python neurolab/scripts/train_neurolab_model.py --pipeline multimodal -- --epochs 10 --output-dir neurolab/data/tmp_run
```

That preserves the preset while allowing one-off changes.
