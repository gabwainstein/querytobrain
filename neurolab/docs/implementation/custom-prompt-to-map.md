# Custom prompt → brain map (steps)

**Goal:** You give a custom prompt (any text); the system returns a brain map. Local only, no API.

---

## Steps

### One-time setup (2 steps)

| Step | What | Command |
|------|------|--------|
| **1** | Build decoder cache (term → map pairs for training) | `python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0` (or `--max-terms 5000` for faster build) |
| **2** | Train text→brain embedding (so any text can be mapped to 400-D) | `python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir neurolab/data/embedding_model --max-terms 0` (optional: add `--encoder sentence-transformers --encoder-model all-mpnet-base-v2 --epochs 30`) |

After this, the “custom prompt → map” path is ready.

---

### Per custom prompt (1 step)

| Step | What | Command |
|------|------|--------|
| **3** | Give your prompt; get the predicted map | `python neurolab/scripts/predict_map.py "YOUR CUSTOM PROMPT HERE"` |

- Output: parcellated map (400-D) printed to console (min/max/mean); optional `--save-nifti out.nii.gz` to save a 3D NIfTI for a viewer.

So: **2 steps one-time**, then **1 step per prompt**.

---

## In code (for a custom agent or UI)

```python
from neurolab.enrichment import TextToBrainEmbedding

emb = TextToBrainEmbedding("neurolab/data/embedding_model")
# Predicted parcellated map (400,) for any prompt
map_400 = emb.predict_map("Your custom prompt here")

# Optional: save as 3D NIfTI
emb.predict_map_to_nifti("Your custom prompt here", "output.nii.gz")
```

---

## Summary

| Phase | Steps | Total so far |
|-------|--------|---------------|
| One-time setup | 1) Build decoder cache<br>2) Train embedding model | 2 |
| Each prompt | 3) Run `predict_map.py "prompt"` (or call `predict_map()` in code) | 1 per prompt |

**No steps left** for the minimal “custom prompt → map” flow: do the one-time setup once, then one command or one function call per prompt.
