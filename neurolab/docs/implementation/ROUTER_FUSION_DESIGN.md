# Router + Fusion Module: Multi-Head Integration Design

**Goal:** Multiple heads (task-fMRI, structural, biology, genes) produce separate maps; a **router + fusion** layer combines them into one user-facing "integrated brain answer" with explainable breakdown.

---

## 1. Head Definitions (Exact)

| Head ID | Map type | Sources | Output | Reliability prior (from memorizer) |
|---------|----------|---------|--------|-----------------------------------|
| `task_fmri` | fmri_activation | direct, neurovault, ontology, pharma_neurosynth | (427,) parcel map | 0.90 (direct), 0.42 (neurovault) |
| `structural` | structural | enigma | (427,) parcel map | −0.05 |
| `pet_receptor` | pet_receptor | neuromaps, receptor, neuromaps_residual, receptor_residual | (427,) parcel map | 0.89 (neuromaps) |
| `gene_expression` | gene_expression | abagen | (427,) via gene PCA inverse | 0.21 |

**Future (not yet implemented):** `rsfc`, `dti` (connectivity heads).

---

## 2. Routing Policy (Which Head Dominates Which Query Types)

| Query pattern | Primary head | Secondary | Rationale |
|---------------|--------------|-----------|------------|
| Task/cognitive ("working memory", "n-back", "attention", "reward") | task_fmri | — | Activation contrasts |
| Drug/compound + task ("ibuprofen emotion", "caffeine attention") | task_fmri | pet_receptor | Pharma fMRI + receptor distribution |
| Drug/compound only ("dopamine", "serotonin receptor") | pet_receptor | gene_expression | Receptor/gene maps |
| Gene/symbol ("HTR2A", "DRD2 gene expression") | gene_expression | pet_receptor | Abagen retrieval or gene head |
| Structural ("cortical thickness", "schizophrenia", "atrophy", "ENIGMA") | structural | — | CT/SA/SubVol |
| Anatomy + function ("amygdala fear", "DLPFC working memory") | task_fmri | pet_receptor | Task dominates; receptor as context |
| Mechanism ("noradrenergic tone", "dopaminergic modulation") | pet_receptor | gene_expression | Receptor/gene biology |

**Keyword rules for router (MVP):**
- `structural`, `thickness`, `volume`, `atrophy`, `enigma`, `cortical`, `subcortical` → boost structural
- `receptor`, `binding`, `pet`, `transmitter`, `dopamine receptor`, `gene expression`, `gene` → boost pet_receptor, gene_expression
- `task`, `activation`, `fmri`, `contrast`, `working memory`, `n-back`, `attention`, `reward` → boost task_fmri
- Drug names (from PDSP/ChEBI list) + no task → boost pet_receptor
- Drug names + task → boost task_fmri

---

## 3. Router Architecture

**Input:** Query embedding `q` (encoder output, dim 1536 for text-embedding-3-small).

**Output:** Head weights `w ∈ R^4` (one per head), normalized.

```
w_raw = router_mlp(q)           # e.g. Linear(1536, 64) → ReLU → Linear(64, 4)
w = softmax(w_raw)
w_tilde_h = w_h * ReliabilityPrior_h   # element-wise
w_tilde = w_tilde / sum(w_tilde)        # renormalize
```

**Reliability priors (from memorizer diagnostics):**
```python
RELIABILITY_PRIOR = {
    "task_fmri": 0.85,      # direct 0.90, neurovault 0.42 → blend
    "structural": 0.10,     # enigma −0.05 → low trust
    "pet_receptor": 0.90,   # neuromaps 0.89
    "gene_expression": 0.30,  # abagen 0.21
}
```

---

## 4. Fusion: Uncertainty-Weighted Mixture (MVP)

Each head returns:
- `map_h`: (427,) parcel vector
- `uncertainty_h`: scalar (0 = confident, 1 = uncertain). MVP: use fixed per-head or 1 − reliability.

**Fused map:**
```
μ_fused = Σ_h w_tilde_h * map_h
σ²_fused ≈ Σ_h w_tilde_h² * σ²_h   (if per-head uncertainty available)
```

**Later:** Product-of-Experts when heads agree → sharpen; when they disagree → increase uncertainty.

---

## 5. Return Schema (JSON + NIfTI)

### 5.1 JSON Output

```json
{
  "query": "ibuprofen 600 mg emotion task",
  "consensus_map": [0.12, -0.34, ...],
  "n_parcels": 427,
  "head_weights": {
    "task_fmri": 0.52,
    "structural": 0.05,
    "pet_receptor": 0.28,
    "gene_expression": 0.15
  },
  "head_confidence": {
    "task_fmri": "high",
    "structural": "low",
    "pet_receptor": "medium",
    "gene_expression": "medium"
  },
  "per_head_maps": {
    "task_fmri": [0.1, -0.2, ...],
    "pet_receptor": [0.15, -0.4, ...]
  },
  "cross_head_agreement": {
    "task_fmri_vs_pet_receptor": 0.67,
    "task_fmri_vs_gene_expression": 0.23
  },
  "evidence": {
    "task_fmri": [
      {"term": "fMRI: ibuprofen 600 mg, emotion task, placebo-controlled", "similarity": 0.92, "source": "neurovault"},
      {"term": "fMRI: pain modulation emotion", "similarity": 0.85, "source": "direct"}
    ],
    "pet_receptor": [
      {"term": "PET: COX-2 receptor", "similarity": 0.78, "source": "neuromaps"}
    ]
  },
  "ontology_expansion_used": ["n-back", "working memory", "executive control"]
}
```

### 5.2 NIfTI Output

- `consensus_map.nii.gz`: Parcellated map back-projected to volumetric NIfTI (using atlas).
- Optional: `per_head_task_fmri.nii.gz`, etc., for inspection.

### 5.3 Fields

| Field | Type | Description |
|-------|------|-------------|
| `consensus_map` | list[float] | Fused (427,) parcel vector |
| `head_weights` | dict | Router output (which heads contributed) |
| `head_confidence` | dict | high/medium/low per head |
| `per_head_maps` | dict | Top 2–3 heads only (to limit payload) |
| `cross_head_agreement` | dict | corr(head_i, head_j) for top heads |
| `evidence` | dict | Top-k nearest training terms per head |
| `ontology_expansion_used` | list[str] | Concepts used (e.g. n-back → working memory) |

---

## 6. Training: Multi-Head + Router

1. **Train heads separately** (or jointly with per-head loss weighting).
2. **Train router** on query embedding → head weights. Supervision: for each training term, the "correct" head is the one that produced that term (from term_sources / map_type). Loss: cross-entropy between router output and true head index.
3. **Optional:** Fine-tune router with reward (e.g. correlation of fused map with held-out retrieval map when available).

---

## 7. Implementation Order

1. Implement separate heads (structural, biology) in trainer.
2. Add router MLP (query_embedding → 4 logits).
3. Add fusion: `consensus = Σ w_h * map_h`.
4. Add evidence retrieval (top-k neighbors per head).
5. Add return schema (JSON).
6. Add NIfTI export for consensus map.
