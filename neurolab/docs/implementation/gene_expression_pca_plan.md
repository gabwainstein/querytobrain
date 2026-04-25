# Gene Expression PCA Pipeline: Complete Implementation Plan

## Overview

Decompose Allen Human Brain Atlas gene expression data (via abagen) into principal components that serve as:
1. **Biologically-labeled axes** of cortical organization
2. **Pharmacological fingerprinting space** for any drug via receptor gene co-expression
3. **Dimensionality-reduced output basis** for the text-to-brain MLP
4. **Structural context layer** for enrichment reports

The pipeline transforms ~15,000 genes × 400 parcels into ~15 interpretable biological gradients that bridge pharmacology, transcriptomics, and neuroimaging.

---

## Phase 1: Data Acquisition & Preprocessing

### 1.1 Fetch Gene Expression via abagen

```python
import abagen
import numpy as np
import pandas as pd
from neuromaps import datasets as nmdata

# Use Schaefer 400 parcellation (7 networks) for consistency with rest of pipeline
# abagen handles donor-to-atlas registration, probe selection, normalization
atlas = abagen.fetch_desikan_killiany()  # or load Schaefer 400 from nilearn

# For Schaefer 400:
from nilearn import datasets as nidatasets
schaefer = nidatasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=1)

expression = abagen.get_expression_data(
    schaefer['maps'],
    atlas_info=schaefer.get('labels'),  # or build label info manually
    
    # Critical parameters:
    ibf_threshold=0.5,        # intensity-based filtering threshold
    probe_selection='diff_stability',  # most robust probe selection
    donor_probes='aggregate',  # aggregate across donors first, then select
    lr_mirror='bidirectional', # mirror left/right to fill both hemispheres
    missing='interpolate',     # interpolate missing regions (Allen only has 6 donors)
    tolerance=2,               # mm tolerance for sample-to-parcel assignment
    sample_norm='srs',         # scaled robust sigmoid normalization
    gene_norm='srs',           # same for genes
    norm_matched=True,         # normalize matched samples only
    return_donors=False,       # return single aggregated expression matrix
)

# Result: DataFrame (400 parcels × ~15,000 genes)
print(f"Expression matrix: {expression.shape}")
print(f"Parcels: {expression.shape[0]}, Genes: {expression.shape[1]}")
```

### 1.2 Gene Filtering

Not all 15,000 genes are informative. Filter before PCA:

```python
# Strategy 1: Differential stability (DS) filtering
# DS = average correlation of each gene's spatial pattern across donor pairs
# Keep genes with DS > threshold (typically 0.1-0.2)
# abagen already does this internally if probe_selection='diff_stability'

# Strategy 2: Variance filtering — remove near-constant genes
gene_var = expression.var(axis=0)
high_var_mask = gene_var > np.percentile(gene_var, 10)  # drop bottom 10%
expression_filtered = expression.loc[:, high_var_mask]
print(f"After variance filter: {expression_filtered.shape[1]} genes")

# Strategy 3: Brain-relevant gene set filtering (optional, more aggressive)
# Use a curated set of brain-expressed genes from, e.g., the HPA Brain Atlas
# This reduces noise from genes not meaningfully expressed in brain
# ~8,000-10,000 genes typically survive this filter

# Strategy 4: Receptor/transporter/channel gene subset (for pharmacology focus)
# If you want PCs specifically relevant to drug targets:
RECEPTOR_GENE_FAMILIES = [
    'HTR',   # serotonin receptors
    'DRD',   # dopamine receptors
    'GABA',  # GABA receptors
    'GRI',   # glutamate receptors (GRIA, GRIK, GRIN, GRID, GRM)
    'GRM',
    'GRIA',
    'GRIK',
    'GRIN',
    'CHRN',  # nicotinic cholinergic
    'CHRM',  # muscarinic cholinergic
    'OPRM',  # opioid receptors
    'OPRD',
    'OPRK',
    'ADRA',  # adrenergic
    'ADRB',
    'HRH',   # histamine
    'SLC6A',  # monoamine transporters (DAT, SERT, NET)
    'CNR',   # cannabinoid receptors
    'CACNA', # calcium channels
    'SCN',   # sodium channels
    'KCNA',  # potassium channels
    'KCNJ',
]
# This gives ~200-500 genes. PCA on this subset = purely pharmacological gradients.
# Decision: run TWO PCAs — one on all genes, one on receptor subset only.
```

### 1.3 Standardization

```python
from sklearn.preprocessing import StandardScaler

# Center and scale: each gene has mean=0, std=1 across parcels
# This ensures PCA captures spatial co-variation patterns, not magnitude differences
scaler = StandardScaler()
expression_scaled = scaler.fit_transform(expression_filtered.values)

# Save for later reconstruction
gene_names = expression_filtered.columns.tolist()
parcel_labels = expression_filtered.index.tolist()

print(f"Scaled matrix: {expression_scaled.shape}")
# Expected: (400, ~12000-15000)
```

---

## Phase 2: PCA Decomposition

### 2.1 Full-genome PCA

```python
from sklearn.decomposition import PCA

# Fit PCA — keep enough components to explain meaningful variance
n_components = 50  # start generous, trim later based on variance explained
pca_full = PCA(n_components=n_components, random_state=42)
pc_scores_full = pca_full.fit_transform(expression_scaled)

# pc_scores_full: (400 parcels, 50 components)
# pca_full.components_: (50, n_genes) — gene loadings
# pca_full.explained_variance_ratio_: fraction of variance per PC

print("Variance explained per PC:")
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
for i in range(20):
    print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f} "
          f"(cumulative: {cumvar[i]:.4f})")

# Typical result:
# PC1: ~0.25 (cumulative: 0.25)  — sensorimotor-transmodal gradient
# PC2: ~0.08 (cumulative: 0.33)  — anterior-posterior
# PC3: ~0.05 (cumulative: 0.38)  
# ...
# PC10: ~0.02 (cumulative: ~0.55)
# PC15: ~0.01 (cumulative: ~0.62)

# Decision: keep components until cumulative variance > 0.60 or until
# the scree plot elbows. Typically 10-15 components.
n_keep = 15
pc_scores = pc_scores_full[:, :n_keep]  # (400, 15)
gene_loadings = pca_full.components_[:n_keep, :]  # (15, n_genes)
```

### 2.2 Receptor-specific PCA

```python
# Subset expression matrix to receptor/transporter/channel genes only
receptor_mask = np.array([
    any(gene.startswith(prefix) for prefix in RECEPTOR_GENE_FAMILIES)
    for gene in gene_names
])
expression_receptors = expression_scaled[:, receptor_mask]
receptor_gene_names = [g for g, m in zip(gene_names, receptor_mask) if m]

print(f"Receptor genes: {expression_receptors.shape[1]}")

# PCA on receptor subset
n_receptor_pcs = 10
pca_receptor = PCA(n_components=n_receptor_pcs, random_state=42)
receptor_pc_scores = pca_receptor.fit_transform(expression_receptors)
receptor_gene_loadings = pca_receptor.components_

# These PCs capture PURELY neurotransmitter-related spatial gradients
# PC1_receptor might be "dopaminergic vs serotonergic cortex"
# PC2_receptor might be "excitatory vs inhibitory receptor density"
```

### 2.3 Validation: Compare PC1 to known gradients

```python
from neuromaps.datasets import fetch_annotation
from neuromaps.stats import compare_images
from neuromaps.parcellate import Parcellater
import nibabel as nib

# Fetch known neuromaps annotations
myelin = fetch_annotation(source='hcps1200', desc='myelinmap', space='fsLR', den='32k')
thickness = fetch_annotation(source='hcps1200', desc='thickness', space='fsLR', den='32k')

# Parcellate to Schaefer 400
parcellater = Parcellater(schaefer['maps'], 'mni152')  # adjust space as needed

# PC1 should correlate strongly with:
# - T1w/T2w myelin (r ~ 0.5-0.7): both capture sensorimotor-transmodal axis
# - Cortical thickness (r ~ 0.3-0.5): thinner cortex in transmodal regions
# - Evolutionary expansion (r ~ -0.5): transmodal = more expanded

# Also compare to existing neuromaps genePC1
genepc1 = fetch_annotation(source='abagen', desc='genepc1', space='fsaverage', den='10k')
# Our PC1 should correlate r > 0.95 with this (same data, same method)
```

### 2.4 Save Everything

```python
import json

output_dir = 'data/gene_pca/'
os.makedirs(output_dir, exist_ok=True)

# Core PCA outputs
np.save(f'{output_dir}/pc_scores_full.npy', pc_scores)           # (400, 15)
np.save(f'{output_dir}/gene_loadings_full.npy', gene_loadings)    # (15, n_genes)
np.save(f'{output_dir}/explained_variance.npy', 
        pca_full.explained_variance_ratio_[:n_keep])

np.save(f'{output_dir}/receptor_pc_scores.npy', receptor_pc_scores)  # (400, 10)
np.save(f'{output_dir}/receptor_gene_loadings.npy', receptor_gene_loadings)

# Metadata
with open(f'{output_dir}/gene_names.json', 'w') as f:
    json.dump(gene_names, f)
with open(f'{output_dir}/receptor_gene_names.json', 'w') as f:
    json.dump(receptor_gene_names, f)
with open(f'{output_dir}/parcel_labels.json', 'w') as f:
    json.dump(parcel_labels, f)

# Scaler for potential inverse transforms
import joblib
joblib.dump(scaler, f'{output_dir}/expression_scaler.pkl')
joblib.dump(pca_full, f'{output_dir}/pca_full_model.pkl')
joblib.dump(pca_receptor, f'{output_dir}/pca_receptor_model.pkl')
```

---

## Phase 3: Biological Labeling of PCs

### 3.1 Gene Ontology Enrichment per PC

```python
import gseapy as gp

def get_pc_enrichment(gene_loadings, gene_names, pc_idx, n_top=300):
    """
    For a given PC, get GO enrichment for top positive and negative loading genes.
    Returns biological labels for each pole of the PC.
    """
    loadings = gene_loadings[pc_idx, :]
    
    # Top positive-loading genes (high values of this PC)
    pos_idx = np.argsort(loadings)[-n_top:]
    pos_genes = [gene_names[i] for i in pos_idx]
    
    # Top negative-loading genes (low values of this PC)
    neg_idx = np.argsort(loadings)[:n_top]
    neg_genes = [gene_names[i] for i in neg_idx]
    
    # Run enrichment against multiple gene set libraries
    libraries = [
        'GO_Biological_Process_2023',
        'GO_Molecular_Function_2023',
        'GO_Cellular_Component_2023',
        'KEGG_2021_Human',
        'Allen_Brain_Atlas_up',      # brain-specific
        'Allen_Brain_Atlas_down',
        'Human_Gene_Atlas',
        'Descartes_Cell_Types_and_Tissue_2021',  # cell type enrichment
    ]
    
    results = {}
    for pole, genes in [('positive', pos_genes), ('negative', neg_genes)]:
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=libraries,
                organism='human',
                outdir=None,  # don't save files
                no_plot=True,
            )
            # Filter significant results
            sig = enr.results[enr.results['Adjusted P-value'] < 0.05]
            sig = sig.sort_values('Adjusted P-value')
            results[pole] = sig[['Gene_set', 'Term', 'Adjusted P-value', 
                                  'Overlap', 'Combined Score']].head(20)
        except Exception as e:
            print(f"  Enrichment failed for PC{pc_idx+1} {pole}: {e}")
            results[pole] = pd.DataFrame()
    
    return results

# Run for each PC
pc_labels = {}
for pc_idx in range(n_keep):
    print(f"\n=== PC{pc_idx+1} ({pca_full.explained_variance_ratio_[pc_idx]:.1%} variance) ===")
    results = get_pc_enrichment(gene_loadings, gene_names, pc_idx)
    
    print(f"\n  Positive pole (top 5 terms):")
    if len(results['positive']) > 0:
        for _, row in results['positive'].head(5).iterrows():
            print(f"    {row['Term']} (p={row['Adjusted P-value']:.2e})")
    
    print(f"\n  Negative pole (top 5 terms):")
    if len(results['negative']) > 0:
        for _, row in results['negative'].head(5).iterrows():
            print(f"    {row['Term']} (p={row['Adjusted P-value']:.2e})")
    
    pc_labels[pc_idx] = results
```

### 3.2 Cell-Type Deconvolution per PC

```python
# Known cell-type marker genes from literature
# These are well-established markers for major brain cell classes
CELL_TYPE_MARKERS = {
    'excitatory_neurons': ['SLC17A7', 'CAMK2A', 'NRGN', 'SATB2', 'TBR1', 'SLC17A6'],
    'inhibitory_neurons': ['GAD1', 'GAD2', 'SLC32A1', 'PVALB', 'SST', 'VIP', 'RELN'],
    'astrocytes': ['GFAP', 'AQP4', 'ALDH1L1', 'GJA1', 'SLC1A2', 'SLC1A3'],
    'oligodendrocytes': ['MBP', 'MOG', 'PLP1', 'OLIG1', 'OLIG2', 'CNP'],
    'microglia': ['CX3CR1', 'P2RY12', 'TMEM119', 'AIF1', 'CSF1R', 'CD68'],
    'endothelial': ['CLDN5', 'FLT1', 'VWF', 'PECAM1', 'CDH5'],
    'OPC': ['PDGFRA', 'CSPG4', 'GPR17', 'OLIG2'],
}

def compute_celltype_loadings(gene_loadings, gene_names, cell_markers):
    """
    For each PC, compute average loading of each cell type's marker genes.
    Tells us which cell types drive each PC.
    """
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    results = {}
    for pc_idx in range(gene_loadings.shape[0]):
        loadings = gene_loadings[pc_idx, :]
        ct_scores = {}
        for cell_type, markers in cell_markers.items():
            # Find markers present in our gene list
            present = [gene_name_to_idx[m] for m in markers if m in gene_name_to_idx]
            if len(present) > 0:
                ct_scores[cell_type] = np.mean(loadings[present])
            else:
                ct_scores[cell_type] = 0.0
        results[f'PC{pc_idx+1}'] = ct_scores
    
    return pd.DataFrame(results).T

celltype_df = compute_celltype_loadings(gene_loadings, gene_names, CELL_TYPE_MARKERS)
print("\nCell-type loadings per PC:")
print(celltype_df.round(3))

# Typical pattern:
# PC1: high oligodendrocyte (positive pole = myelinated sensorimotor)
#       high excitatory neuron (negative pole = transmodal cortex)
# PC2: might separate astrocyte-rich vs neuron-rich regions
```

### 3.3 Neurotransmitter System Enrichment per PC

```python
# Specific receptor genes corresponding to Hansen's receptor atlas
NEUROTRANSMITTER_GENES = {
    # Serotonin system
    '5HT1a': 'HTR1A', '5HT1b': 'HTR1B', '5HT2a': 'HTR2A',
    '5HT4': 'HTR4', '5HT6': 'HTR6',
    # Dopamine system
    'D1': 'DRD1', 'D2': 'DRD2',
    # GABA system
    'GABAa': 'GABRA1',  # representative subunit
    # Glutamate system
    'NMDA': 'GRIN1', 'mGluR5': 'GRM5',
    # Acetylcholine system
    'VAChT': 'SLC18A3', 'M1': 'CHRM1', 'a4b2': 'CHRNA4',
    # Norepinephrine
    'NET': 'SLC6A2',
    # Opioid
    'MU': 'OPRM1',
    # Cannabinoid
    'CB1': 'CNR1',
    # Histamine
    'H3': 'HRH3',
    # Transporters
    'DAT': 'SLC6A3', 'SERT': 'SLC6A4',
}

def compute_receptor_loadings(gene_loadings, gene_names, receptor_genes):
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    results = {}
    for pc_idx in range(gene_loadings.shape[0]):
        loadings = gene_loadings[pc_idx, :]
        receptor_scores = {}
        for receptor_name, gene in receptor_genes.items():
            if gene in gene_name_to_idx:
                receptor_scores[receptor_name] = loadings[gene_name_to_idx[gene]]
        results[f'PC{pc_idx+1}'] = receptor_scores
    return pd.DataFrame(results).T

receptor_loading_df = compute_receptor_loadings(gene_loadings, gene_names, NEUROTRANSMITTER_GENES)
print("\nNeurotransmitter receptor loadings per PC:")
print(receptor_loading_df.round(3))
```

### 3.4 Build Human-Readable PC Label Registry

```python
# After running enrichment + cell-type + receptor analyses,
# compile a structured label for each PC

# This is manually curated after reviewing automated results
# but the structure is programmatic

PC_REGISTRY = {
    'PC1': {
        'variance_explained': 0.25,
        'short_label': 'Sensorimotor–Transmodal Hierarchy',
        'positive_pole': {
            'description': 'Sensorimotor cortex (heavily myelinated)',
            'enriched_go': ['myelination', 'axon ensheathment', 'oligodendrocyte differentiation'],
            'cell_types': ['oligodendrocytes'],
            'receptors_high': ['5HT1b', 'NMDA'],
            'brain_regions': ['primary motor', 'primary somatosensory', 'primary visual'],
        },
        'negative_pole': {
            'description': 'Transmodal association cortex',
            'enriched_go': ['synaptic signaling', 'glutamate receptor signaling', 'cognition'],
            'cell_types': ['excitatory_neurons'],
            'receptors_high': ['5HT2a', 'D1', 'CB1'],
            'brain_regions': ['prefrontal', 'temporal pole', 'angular gyrus'],
        },
        'known_correlates': {
            'myelin_t1t2': 0.65,
            'evolutionary_expansion': -0.55,
            'cortical_thickness': -0.40,
        }
    },
    'PC2': {
        'variance_explained': 0.08,
        'short_label': 'Anterior–Posterior Axis',
        # ... fill in after running enrichment
    },
    # ... etc for PC3-PC15
}

# Save registry
with open(f'{output_dir}/pc_registry.json', 'w') as f:
    json.dump(PC_REGISTRY, f, indent=2)
```

---

## Phase 4: Drug-to-PC-Space Projection (Pharmacological Fingerprinting)

### 4.1 Load PDSP Ki Data

```python
# Download PDSP Ki database as CSV from pdspdb.unc.edu
# Columns: Drug Name, Gene Target, Ki (nM), Species, etc.

pdsp = pd.read_csv('data/pdsp_ki_database.csv')

# Filter to human targets only
pdsp_human = pdsp[pdsp['species'] == 'human'].copy()

# For each drug, build a binding profile: {gene_name: affinity_weight}
def build_drug_profile(drug_name, pdsp_df, gene_names_available):
    """
    Build a drug's receptor binding profile as a weight vector over genes.
    
    Weight = 1/Ki (higher affinity = higher weight), normalized.
    Only includes genes present in our expression matrix.
    """
    drug_data = pdsp_df[pdsp_df['drug_name'].str.lower() == drug_name.lower()]
    
    if len(drug_data) == 0:
        return None
    
    profile = {}
    for _, row in drug_data.iterrows():
        gene = row.get('gene_symbol', row.get('target_gene', ''))
        ki = row.get('ki_nM', row.get('Ki (nM)', np.nan))
        
        if pd.isna(ki) or ki <= 0 or gene not in gene_names_available:
            continue
        
        # If multiple Ki values for same target, take geometric mean
        if gene in profile:
            profile[gene].append(ki)
        else:
            profile[gene] = [ki]
    
    # Geometric mean of Ki values, then convert to affinity weight
    weights = {}
    for gene, ki_values in profile.items():
        mean_ki = np.exp(np.mean(np.log(ki_values)))  # geometric mean
        weights[gene] = 1.0 / mean_ki  # higher affinity = higher weight
    
    # Normalize weights to sum to 1
    if len(weights) > 0:
        total = sum(weights.values())
        weights = {g: w/total for g, w in weights.items()}
    
    return weights

# Also supplement with ChEMBL data for drugs missing from PDSP
# ChEMBL API: https://www.ebi.ac.uk/chembl/api/data/activity?...
```

### 4.2 Project Drug into Gene Expression Space

```python
def drug_to_spatial_profile(drug_weights, expression_scaled, gene_names):
    """
    Convert a drug's receptor binding weights into a spatial brain profile
    by computing the weighted average expression of its target genes.
    
    Returns: (n_parcels,) vector — the drug's "transcriptomic brain fingerprint"
    """
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    spatial_profile = np.zeros(expression_scaled.shape[0])  # (400,)
    total_weight = 0
    
    for gene, weight in drug_weights.items():
        if gene in gene_name_to_idx:
            idx = gene_name_to_idx[gene]
            spatial_profile += weight * expression_scaled[:, idx]
            total_weight += weight
    
    if total_weight > 0:
        spatial_profile /= total_weight
    
    return spatial_profile


def drug_to_pc_coordinates(drug_weights, expression_scaled, gene_names, 
                            pc_scores, gene_loadings):
    """
    Project a drug's binding profile into PC space.
    
    Two approaches:
    
    Approach A: Project spatial profile onto PC score vectors
      1. Build spatial profile (400,)
      2. Project onto PC scores: coords = pc_scores.T @ spatial_profile
    
    Approach B: Project directly through gene loadings (more principled)
      1. Build gene weight vector (n_genes,) with drug affinities
      2. Dot with gene loadings: coords = gene_loadings @ gene_weight_vector
    
    Approach B is better because it doesn't require going through
    the spatial domain — it operates directly in gene space.
    """
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Approach B: Direct gene-space projection
    gene_weight_vector = np.zeros(len(gene_names))
    for gene, weight in drug_weights.items():
        if gene in gene_name_to_idx:
            gene_weight_vector[gene_name_to_idx[gene]] = weight
    
    # Project through gene loadings
    # gene_loadings: (n_pcs, n_genes)
    # gene_weight_vector: (n_genes,)
    pc_coords = gene_loadings @ gene_weight_vector  # (n_pcs,)
    
    # Also compute spatial profile for visualization
    spatial_profile = drug_to_spatial_profile(drug_weights, expression_scaled, gene_names)
    
    # And the PC-reconstructed version (smooth, denoised)
    spatial_from_pcs = pc_scores @ pc_coords  # (400,)
    
    return {
        'pc_coordinates': pc_coords,           # (15,) — position in PC space
        'spatial_profile_raw': spatial_profile, # (400,) — direct weighted expression
        'spatial_profile_pc': spatial_from_pcs, # (400,) — PC-reconstructed (smoother)
    }
```

### 4.3 Batch Process All Drugs

```python
# Get unique drug list from PDSP
all_drugs = pdsp_human['drug_name'].unique()
gene_names_set = set(gene_names)

drug_profiles = {}
drug_pc_coords = {}
drug_spatial_maps = {}

for drug in all_drugs:
    weights = build_drug_profile(drug, pdsp_human, gene_names_set)
    if weights is None or len(weights) < 2:  # need at least 2 targets
        continue
    
    result = drug_to_pc_coordinates(
        weights, expression_scaled, gene_names, pc_scores, gene_loadings
    )
    
    drug_profiles[drug] = weights
    drug_pc_coords[drug] = result['pc_coordinates']
    drug_spatial_maps[drug] = result['spatial_profile_pc']

print(f"Processed {len(drug_pc_coords)} drugs with valid binding profiles")

# Save
drug_pc_matrix = np.array([drug_pc_coords[d] for d in sorted(drug_pc_coords.keys())])
np.save(f'{output_dir}/drug_pc_coordinates.npy', drug_pc_matrix)

drug_names_sorted = sorted(drug_pc_coords.keys())
with open(f'{output_dir}/drug_names.json', 'w') as f:
    json.dump(drug_names_sorted, f)

drug_spatial_matrix = np.array([drug_spatial_maps[d] for d in drug_names_sorted])
np.save(f'{output_dir}/drug_spatial_maps.npy', drug_spatial_matrix)  # (n_drugs, 400)
```

### 4.4 Drug Similarity in PC Space

```python
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise drug similarity in PC space
drug_pc_matrix_normed = drug_pc_matrix / np.linalg.norm(drug_pc_matrix, axis=1, keepdims=True)
drug_similarity = cosine_similarity(drug_pc_matrix_normed)

# Find most similar drugs to a query
def find_similar_drugs(query_drug, drug_pc_coords, drug_names, top_k=10):
    if query_drug not in drug_pc_coords:
        return []
    
    query_vec = drug_pc_coords[query_drug]
    similarities = {}
    for drug, coords in drug_pc_coords.items():
        if drug != query_drug:
            sim = 1 - cosine(query_vec, coords)
            similarities[drug] = sim
    
    sorted_drugs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_drugs[:top_k]

# Example: "What drugs have similar brain engagement profiles to psilocybin?"
similar = find_similar_drugs('Psilocin', drug_pc_coords, drug_names_sorted)
# Expected: LSD, DMT, mescaline (all 5-HT2A agonists with similar spatial profiles)
```

---

## Phase 5: MLP Integration — PC-Based Output Architecture

### 5.1 Architecture: Dual-Head with PC Prior

```python
import torch
import torch.nn as nn

class GenePCBrainModel(nn.Module):
    """
    Text-to-brain prediction using gene expression PCs as output basis.
    
    Architecture:
        text_embedding (1536) → shared trunk → two heads:
        
        Head 1 (PC-constrained): predicts 15 PC coefficients, 
            reconstructs brain map via PC scores
        Head 2 (residual): predicts 400-dim residual correction
        
        Final = gated combination of PC-reconstructed map + residual
    
    The PC head acts as a strong biological prior (smooth, interpretable),
    the residual head captures patterns not explained by gene expression.
    """
    
    def __init__(self, 
                 embed_dim=1536,     # text embedding dimension
                 n_parcels=400,      # Schaefer 400
                 n_pcs=15,           # gene expression PCs to predict
                 pc_scores=None,     # (400, 15) numpy array — fixed basis
                 hidden_dim=512,
                 dropout=0.2):
        super().__init__()
        
        # Register PC scores as a non-trainable buffer
        # These are the spatial patterns from gene expression PCA
        self.register_buffer('pc_basis', 
                             torch.FloatTensor(pc_scores))  # (400, 15)
        
        self.n_pcs = n_pcs
        self.n_parcels = n_parcels
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Head 1: PC coefficient prediction (biologically constrained)
        self.pc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_pcs),
        )
        
        # Head 2: Direct residual prediction (unconstrained)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_parcels),
        )
        
        # Learned gate: how much to weight PC vs residual
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, text_embedding):
        """
        Args:
            text_embedding: (batch, 1536) from OpenAI text-embedding-3-large
        
        Returns:
            predicted_map: (batch, 400) brain activation prediction
            pc_coefficients: (batch, 15) for interpretability
            gate_value: (batch, 1) PC vs residual weighting
        """
        # Shared features
        h = self.trunk(text_embedding)  # (batch, hidden_dim)
        
        # PC-constrained prediction
        pc_coefs = self.pc_head(h)  # (batch, 15)
        map_from_pcs = pc_coefs @ self.pc_basis.T  # (batch, 400)
        
        # Residual prediction
        residual = self.residual_head(h)  # (batch, 400)
        
        # Gated combination
        alpha = self.gate(h)  # (batch, 1) — how much PC contributes
        predicted_map = alpha * map_from_pcs + (1 - alpha) * residual
        
        return predicted_map, pc_coefs, alpha
    
    def predict_from_pcs_only(self, text_embedding):
        """Pure PC-based prediction (fully interpretable, no residual)."""
        h = self.trunk(text_embedding)
        pc_coefs = self.pc_head(h)
        map_from_pcs = pc_coefs @ self.pc_basis.T
        return map_from_pcs, pc_coefs
```

### 5.2 Training with PC Regularization

```python
class PCRegularizedLoss(nn.Module):
    """
    Loss function that encourages PC-based predictions to explain most variance,
    with the residual head only activating for patterns that truly need it.
    """
    
    def __init__(self, lambda_residual=0.1, lambda_pc_smooth=0.01):
        super().__init__()
        self.lambda_residual = lambda_residual
        self.lambda_pc_smooth = lambda_pc_smooth
    
    def forward(self, predicted_map, target_map, pc_coefs, gate_value):
        # Primary loss: MSE on predicted vs actual brain map
        mse_loss = nn.functional.mse_loss(predicted_map, target_map)
        
        # Regularization 1: Penalize residual head usage
        # Encourage gate to stay close to 1 (favor PC pathway)
        # This biases toward biologically interpretable predictions
        residual_penalty = self.lambda_residual * (1 - gate_value).mean()
        
        # Regularization 2: Smooth PC coefficients
        # Discourage extreme coefficient values (keeps predictions moderate)
        pc_smooth = self.lambda_pc_smooth * (pc_coefs ** 2).mean()
        
        total = mse_loss + residual_penalty + pc_smooth
        
        return total, {
            'mse': mse_loss.item(),
            'residual_penalty': residual_penalty.item(),
            'pc_smooth': pc_smooth.item(),
            'gate_mean': gate_value.mean().item(),
        }
```

### 5.3 Training Loop Sketch

```python
# Training data: (text_embedding, brain_map) pairs
# from NeuroQuery, Neurosynth, NeuroVault, ENIGMA, OpenNeuro pharma datasets

# Also include drug spatial maps from Phase 4 as training data:
# ("haloperidol receptor binding", drug_spatial_maps['Haloperidol'])
# This teaches the model to use the PC pathway for pharmacological queries

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    metrics_acc = defaultdict(float)
    
    for batch in dataloader:
        text_emb = batch['embedding'].to(device)
        target_map = batch['brain_map'].to(device)
        
        predicted_map, pc_coefs, gate = model(text_emb)
        loss, metrics = criterion(predicted_map, target_map, pc_coefs, gate)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_acc[k] += v
    
    n = len(dataloader)
    return total_loss / n, {k: v/n for k, v in metrics_acc.items()}

# Training schedule:
# Phase A (epochs 1-50): lambda_residual=1.0 (force PC pathway only)
# Phase B (epochs 50-100): lambda_residual=0.1 (gradually allow residual)
# Phase C (epochs 100+): lambda_residual=0.01 (fine-tune with minimal constraint)
```

---

## Phase 6: Drug Enrichment Reports (Platform Output)

### 6.1 Query Processing Pipeline

```python
def process_drug_query(query_text, model, pc_registry, drug_pc_coords,
                       receptor_loading_df, embedder, device):
    """
    Full pipeline for a user query about a drug.
    
    Example query: "What brain regions are affected by sertraline?"
    
    Returns a structured report with:
    1. Predicted brain map
    2. PC decomposition of the prediction
    3. Biological interpretation per significant PC
    4. Similar drugs by PC-space proximity
    5. Receptor system involvement
    """
    
    # Step 1: Embed the query
    query_embedding = embedder.encode(query_text)  # (1536,)
    query_tensor = torch.FloatTensor(query_embedding).unsqueeze(0).to(device)
    
    # Step 2: Predict brain map + PC coefficients
    model.eval()
    with torch.no_grad():
        predicted_map, pc_coefs, gate = model(query_tensor)
    
    predicted_map = predicted_map.cpu().numpy().squeeze()  # (400,)
    pc_coefs = pc_coefs.cpu().numpy().squeeze()            # (15,)
    gate_value = gate.cpu().item()
    
    # Step 3: Identify dominant PCs
    # Which PCs contribute most to this prediction?
    pc_contributions = np.abs(pc_coefs) * np.sqrt(
        np.array([pc_registry[f'PC{i+1}']['variance_explained'] 
                  for i in range(len(pc_coefs))])
    )
    dominant_pcs = np.argsort(pc_contributions)[::-1]  # sorted by importance
    
    # Step 4: Build interpretation for top PCs
    interpretations = []
    for pc_idx in dominant_pcs[:5]:  # top 5 contributing PCs
        pc_key = f'PC{pc_idx+1}'
        pc_info = pc_registry[pc_key]
        coef = pc_coefs[pc_idx]
        
        # Which pole of this PC does the prediction favor?
        if coef > 0:
            pole = pc_info['positive_pole']
            pole_name = 'positive'
        else:
            pole = pc_info['negative_pole']
            pole_name = 'negative'
        
        interpretations.append({
            'pc': pc_key,
            'coefficient': float(coef),
            'contribution': float(pc_contributions[pc_idx]),
            'label': pc_info['short_label'],
            'pole': pole_name,
            'description': pole['description'],
            'enriched_processes': pole['enriched_go'][:3],
            'cell_types': pole['cell_types'],
            'associated_receptors': pole.get('receptors_high', []),
        })
    
    # Step 5: Find pharmacologically similar drugs
    # Extract drug name from query (simplified — use NER in production)
    drug_name = extract_drug_name(query_text)
    similar_drugs = []
    if drug_name and drug_name in drug_pc_coords:
        similar_drugs = find_similar_drugs(drug_name, drug_pc_coords, 
                                           list(drug_pc_coords.keys()), top_k=5)
    
    # Step 6: Receptor system summary
    receptor_involvement = {}
    for receptor, gene in NEUROTRANSMITTER_GENES.items():
        # How much does this receptor's gene load on the dominant PCs?
        involvement = sum(
            pc_coefs[pc_idx] * receptor_loading_df.iloc[pc_idx].get(receptor, 0)
            for pc_idx in dominant_pcs[:5]
        )
        receptor_involvement[receptor] = float(involvement)
    
    # Sort by absolute involvement
    receptor_involvement = dict(
        sorted(receptor_involvement.items(), 
               key=lambda x: abs(x[1]), reverse=True)
    )
    
    return {
        'predicted_map': predicted_map,
        'pc_coefficients': pc_coefs.tolist(),
        'gate_value': gate_value,
        'dominant_pcs': interpretations,
        'similar_drugs': similar_drugs,
        'receptor_involvement': receptor_involvement,
        'query': query_text,
    }
```

### 6.2 Report Generation (Natural Language)

```python
def generate_enrichment_text(result, pc_registry):
    """
    Generate human-readable enrichment report from PC analysis.
    
    This text describes and interprets findings using facts derived from 
    the analysis, not reproducing raw data values. This keeps the output
    in the domain of factual interpretation (not copyrightable expression).
    """
    
    lines = []
    lines.append(f"## Transcriptomic Context: {result['query']}\n")
    
    # Overall summary
    gate = result['gate_value']
    if gate > 0.7:
        lines.append("This pattern is well-explained by known transcriptomic "
                     "gradients of the cortex, suggesting strong biological grounding.\n")
    else:
        lines.append("This pattern includes features beyond canonical transcriptomic "
                     "gradients, suggesting task-specific or network-specific effects.\n")
    
    # PC-by-PC interpretation
    lines.append("### Key Biological Axes\n")
    for interp in result['dominant_pcs'][:3]:
        direction = "aligns with" if interp['coefficient'] > 0 else "opposes"
        lines.append(
            f"**{interp['label']}** (PC{interp['pc'][-1]}): "
            f"This pattern {direction} the {interp['description'].lower()} pole. "
            f"Regions involved are enriched for {', '.join(interp['enriched_processes'][:2])} "
            f"and associated with {', '.join(interp['cell_types'])} cell populations."
        )
        if interp['associated_receptors']:
            lines.append(
                f"Relevant receptor systems: {', '.join(interp['associated_receptors'][:3])}."
            )
        lines.append("")
    
    # Receptor summary
    top_receptors = list(result['receptor_involvement'].items())[:5]
    if top_receptors:
        lines.append("### Neurotransmitter System Involvement\n")
        for receptor, score in top_receptors:
            if abs(score) > 0.01:
                direction = "positively" if score > 0 else "inversely"
                lines.append(f"- **{receptor}**: {direction} implicated")
    
    # Similar drugs
    if result['similar_drugs']:
        lines.append("\n### Pharmacologically Similar Compounds\n")
        lines.append("Based on transcriptomic brain engagement profiles:")
        for drug, sim in result['similar_drugs'][:5]:
            lines.append(f"- {drug} (similarity: {sim:.2f})")
    
    return '\n'.join(lines)
```

---

## Phase 7: Visualization Components

### 7.1 PC Radar Chart

```python
# For each drug/query, show a radar chart of PC loadings
# with biological labels on each axis

def make_pc_radar_data(pc_coefs, pc_registry, n_pcs=8):
    """
    Prepare data for a radar chart showing the query's position
    in transcriptomic gradient space.
    
    Returns dict suitable for recharts or plotly rendering in React frontend.
    """
    categories = []
    values = []
    
    for i in range(n_pcs):
        pc_key = f'PC{i+1}'
        label = pc_registry[pc_key]['short_label']
        # Truncate label for radar chart readability
        short = label[:25] + '...' if len(label) > 25 else label
        categories.append(short)
        values.append(float(pc_coefs[i]))
    
    return {
        'categories': categories,
        'values': values,
        'description': 'Position of query in transcriptomic gradient space. '
                      'Each axis represents a principal component of gene expression '
                      'across the cortex, labeled by its dominant biological process.'
    }
```

### 7.2 Drug Similarity UMAP

```python
from umap import UMAP

def compute_drug_umap(drug_pc_matrix, drug_names, n_neighbors=15, min_dist=0.1):
    """
    2D UMAP embedding of all drugs in PC space.
    Shows clusters of pharmacologically similar compounds.
    """
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, 
                   min_dist=min_dist, random_state=42)
    embedding_2d = reducer.fit_transform(drug_pc_matrix)
    
    # Classify drugs by primary mechanism for coloring
    # (this would use PDSP data to determine dominant receptor)
    
    return pd.DataFrame({
        'drug': drug_names,
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
    })

# The UMAP shows natural clusters:
# - Typical antipsychotics (D2 antagonists) cluster together
# - SSRIs cluster together
# - Psychedelics (5-HT2A agonists) cluster together
# - Benzodiazepines (GABAa modulators) form their own cluster
# User's query drug is highlighted on the map
```

### 7.3 PC Brain Maps for 3D Viewer

```python
import nibabel as nib

def save_pc_maps_as_nifti(pc_scores, schaefer_atlas_path, output_dir):
    """
    Save each PC's spatial pattern as a NIfTI volume for 3D rendering.
    These become the "gene expression gradient" overlays in the viewer.
    """
    atlas_img = nib.load(schaefer_atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    for pc_idx in range(pc_scores.shape[1]):
        # Create volume where each parcel has the PC score value
        pc_vol = np.zeros_like(atlas_data, dtype=np.float32)
        for parcel_idx in range(pc_scores.shape[0]):
            parcel_label = parcel_idx + 1  # labels are 1-indexed
            pc_vol[atlas_data == parcel_label] = pc_scores[parcel_idx, pc_idx]
        
        pc_img = nib.Nifti1Image(pc_vol, atlas_img.affine, atlas_img.header)
        nib.save(pc_img, f'{output_dir}/genePC{pc_idx+1}_schaefer400.nii.gz')
    
    print(f"Saved {pc_scores.shape[1]} PC maps as NIfTI volumes")
```

---

## Phase 8: File Manifest & Dependencies

### 8.1 Output Files

```
data/gene_pca/
├── pc_scores_full.npy              # (400, 15) — spatial patterns
├── gene_loadings_full.npy          # (15, ~15000) — gene weights per PC
├── explained_variance.npy          # (15,) — variance explained
├── receptor_pc_scores.npy          # (400, 10) — receptor-only PCA
├── receptor_gene_loadings.npy      # (10, ~300) — receptor gene weights
├── gene_names.json                 # list of all gene names (ordered)
├── receptor_gene_names.json        # subset of receptor genes
├── parcel_labels.json              # Schaefer 400 parcel labels
├── expression_scaler.pkl           # sklearn StandardScaler
├── pca_full_model.pkl              # sklearn PCA model
├── pca_receptor_model.pkl          # sklearn PCA model
├── pc_registry.json                # human-readable PC labels + enrichment
├── drug_pc_coordinates.npy         # (n_drugs, 15) — all drugs in PC space
├── drug_spatial_maps.npy           # (n_drugs, 400) — PC-reconstructed maps
├── drug_names.json                 # drug names (ordered)
├── celltype_loadings.csv           # cell-type loadings per PC
├── receptor_loadings.csv           # neurotransmitter receptor loadings per PC
├── enrichment/
│   ├── PC1_positive_GO.csv         # GO enrichment results per PC pole
│   ├── PC1_negative_GO.csv
│   ├── PC2_positive_GO.csv
│   └── ...
└── nifti/
    ├── genePC1_schaefer400.nii.gz  # volumetric PC maps for 3D rendering
    ├── genePC2_schaefer400.nii.gz
    └── ...
```

### 8.2 Python Dependencies

```
# Core
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
joblib>=1.3

# Neuroimaging
abagen>=0.1.3
neuromaps>=0.0.5
nilearn>=0.10
nibabel>=5.0

# Gene enrichment
gseapy>=1.0      # Enrichr API wrapper

# Visualization
matplotlib>=3.7
seaborn>=0.12
umap-learn>=0.5

# Deep learning
torch>=2.0

# Pharmacology data
requests          # for ChEMBL API
chembl_webresource_client  # official ChEMBL Python client
```

---

## Execution Order

| Step | Phase | Time Est. | Depends On |
|------|-------|-----------|------------|
| 1 | 1.1 Fetch expression | ~20 min | abagen installed |
| 2 | 1.2-1.3 Filter & scale | ~5 min | Step 1 |
| 3 | 2.1-2.2 Run PCA | ~2 min | Step 2 |
| 4 | 2.3 Validation vs neuromaps | ~10 min | Step 3 |
| 5 | 3.1 GO enrichment | ~30 min | Step 3 (Enrichr API calls) |
| 6 | 3.2-3.3 Cell-type & receptor loadings | ~5 min | Step 3 |
| 7 | 3.4 Curate PC registry | ~2 hrs | Steps 5-6 (manual review) |
| 8 | 4.1 Load PDSP data | ~10 min | PDSP CSV download |
| 9 | 4.2-4.3 Drug projections | ~15 min | Steps 3, 8 |
| 10 | 4.4 Drug similarity | ~5 min | Step 9 |
| 11 | 5.1-5.3 MLP training | ~2-4 hrs | Steps 3, 9, training data |
| 12 | 6.1-6.2 Report generation | ~1 hr | Steps 7, 9, 11 |
| 13 | 7.1-7.3 Visualization | ~2 hrs | Steps 3, 9 |

**Total estimated time: ~1-2 days of focused work** (excluding MLP training data assembly which is a separate pipeline).

---

## Critical Design Decisions

### How many PCs to keep?

**Recommendation: 15 for full-genome, 10 for receptor-only.**

Rationale: With 6 Allen donors and ~400 parcels, the effective degrees of freedom are limited. Beyond PC15, individual PCs explain <1% variance and become unreliable. The first 15 PCs explain ~60-65% of total variance, which is the stable, reproducible fraction. The remaining ~35% is donor-specific noise and very local expression patterns.

For the receptor-only PCA, 10 components from ~300 genes is appropriate (rule of thumb: n_components < sqrt(n_features)).

### Full-genome PCA vs. receptor-only PCA?

**Use both. They answer different questions.**

Full-genome PCA captures the complete landscape of cortical organization — cell types, developmental gradients, metabolic patterns. This is your "biological context" layer.

Receptor-only PCA captures specifically neurotransmitter-related gradients. This is your "pharmacological specificity" layer. When a user queries a drug, the receptor-only PCs tell you about receptor-level engagement, while the full-genome PCs tell you about the broader biological context of the affected regions.

In the MLP, use the full-genome PCs as the output basis (they capture more variance and are more general). Use the receptor-only PCs as additional input features or as a separate enrichment analysis.

### PC-constrained vs. unconstrained MLP?

**Start PC-constrained, gradually relax.**

The gated architecture lets you control this. During early training, heavily penalize the residual head (lambda_residual=1.0), forcing the model to work through PCs. This acts as strong regularization and prevents overfitting on small training sets. As training progresses and if the model plateau, gradually lower lambda_residual to let the residual head capture task-specific patterns the PCs miss.

Monitor the gate value during training. If it stays near 1.0 even without penalty, your PC basis is sufficient for most predictions. If it drops to 0.3-0.5, the data contains important patterns orthogonal to gene expression gradients (e.g., task-specific networks that don't follow transcriptomic axes).
