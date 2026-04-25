"""
NeuroVault → NeuroLab Training Set Ingestion Algorithm
=======================================================

Full pipeline: raw NeuroVault collections → consistent 392-D training vectors
with proper averaging, QC, deduplication, labeling, and balanced sampling.

This is the SPECIFICATION. Implement each stage as a module.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0: COLLECTION METADATA & CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class AggLevel(Enum):
    """How the maps in a collection were generated."""
    META_ANALYTIC = "meta"       # ALE/MKDA/IBMA across studies (highest SNR)
    GROUP_AVERAGE = "group"      # Group-level GLM from one study (medium SNR)  
    SUBJECT_LEVEL = "subject"    # Individual subject SPMs/betas (lowest SNR, needs averaging)
    ATLAS = "atlas"              # Parcellation/atlas maps (not activation data)
    MULTIVARIATE = "multivariate"  # Decoder weights, signatures (e.g. NPS, PINES)

class DomainTag(Enum):
    """Coarse domain for stratified sampling."""
    COGNITIVE = "cognitive"         # WM, attention, executive function, language, learning
    EMOTION = "emotion"             # Fear, reward, emotion regulation
    SENSORIMOTOR = "sensorimotor"   # Motor, pain, sensory
    SOCIAL = "social"               # ToM, social reward, empathy
    CLINICAL = "clinical"           # MDD, PTSD, SZ, ASD, addiction, anxiety
    PHARMACOLOGICAL = "pharma"      # Drug effects
    STRUCTURAL = "structural"       # VBM, CT, DTI-derived
    CONNECTIVITY = "connectivity"   # FC-derived, network maps, gradients
    REFERENCE = "reference"         # Atlases, normative maps, neurotransmitter maps
    META_MIXED = "meta_mixed"       # Multi-domain meta-analytic collections

@dataclass
class CollectionSpec:
    collection_id: int              # NeuroVault collection ID
    name: str
    agg_level: AggLevel
    domain: DomainTag
    tier: int                       # 1-4, P(harma), S(pecial)
    n_subjects: Optional[int]       # If known
    needs_averaging: bool           # True → must average before training
    contrast_field: str             # Metadata field to group by for averaging
    notes: str = ""

# -----------------------------------------------------------------------
# Registry of all 126 collections with their classification
# -----------------------------------------------------------------------

# This would be the full registry. Abbreviated example:
COLLECTION_REGISTRY = {
    # === USE AS-IS (group-level / meta-analytic) ===
    457:   CollectionSpec(457,  "HCP group-level", AggLevel.GROUP_AVERAGE, DomainTag.COGNITIVE, 1, 1200, False, ""),
    1274:  CollectionSpec(1274, "CogAtlas × NeuroSynth decoding", AggLevel.META_ANALYTIC, DomainTag.META_MIXED, 1, None, False, ""),
    3324:  CollectionSpec(3324, "Kragel pain/cog/emotion", AggLevel.META_ANALYTIC, DomainTag.META_MIXED, 1, None, False, ""),
    # ... (all ~95 use-as-is collections)
    
    # === AVERAGE FIRST (subject-level) ===
    1952:  CollectionSpec(1952, "BrainPedia", AggLevel.SUBJECT_LEVEL, DomainTag.COGNITIVE, 1, None, True, "contrast_definition", "196 conditions; HIGH priority"),
    6618:  CollectionSpec(6618, "IBC 2nd release", AggLevel.SUBJECT_LEVEL, DomainTag.COGNITIVE, 1, 13, True, "contrast_definition", "13 subj × 205 contrasts"),
    2138:  CollectionSpec(2138, "IBC 1st release", AggLevel.SUBJECT_LEVEL, DomainTag.COGNITIVE, 1, 12, True, "contrast_definition", "12 subj × 59 contrasts"),
    4343:  CollectionSpec(4343, "UCLA LA5C", AggLevel.SUBJECT_LEVEL, DomainTag.CLINICAL, 1, 272, True, "contrast_definition", "130 healthy + 142 clinical"),
    16284: CollectionSpec(16284, "IAPS emotional valence", AggLevel.SUBJECT_LEVEL, DomainTag.EMOTION, 3, 56, True, "contrast_definition", "Individual betas"),
    # ... (all ~25 average-first collections)
    
    # === PHARMACOLOGICAL ===
    1083:  CollectionSpec(1083, "Pharma collection", AggLevel.GROUP_AVERAGE, DomainTag.PHARMACOLOGICAL, 0, None, False, ""),
    # ... etc
}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DOWNLOAD & PARCELLATE
# ═══════════════════════════════════════════════════════════════════════════════
"""
INPUT:  Raw NIfTI stat maps from NeuroVault API
OUTPUT: 392-D vectors (one per image) in a staging dict

For each collection:
  1. Download all images via NeuroVault API (or from local cache)
  2. For each NIfTI image:
     a. Check it's in MNI space (reject if not)
     b. Resample to your atlas space if needed (2mm MNI152)
     c. Parcellate: for each of 392 parcels, compute mean value within parcel mask
     d. Result: 392-D vector
  3. Store: {image_id: {"vector": np.array(392), "metadata": {...}}}
"""

@dataclass
class RawImage:
    image_id: int
    collection_id: int
    vector: np.ndarray            # shape (392,)
    name: str                     # NeuroVault image name
    contrast_definition: str      # e.g. "2-back > 0-back"
    cognitive_paradigm: str       # CogAtlas paradigm if available
    cognitive_contrast: str       # CogAtlas contrast if available
    map_type: str                 # "T", "Z", "beta", "other"
    analysis_level: str           # "group", "individual" from NV metadata
    subject_id: Optional[str]     # If extractable
    modality: str                 # "fMRI-BOLD", "PET", "VBM", etc.


def parcellate_nifti(nifti_path: str, atlas_path: str) -> np.ndarray:
    """
    Standard parcellation: NIfTI → 392-D vector.
    
    atlas_path = "combined_atlas_392.nii.gz" (Glasser 360 + Tian S2)
    
    Algorithm:
      1. Load atlas and image, check they're in same space
      2. If different resolution, resample image to atlas space
         (nilearn.image.resample_to_img with interpolation='continuous')
      3. For each parcel label 1..392:
         mask = atlas_data == label
         value = np.nanmean(image_data[mask])
      4. Return 392-D vector
      
    Handle edge cases:
      - NaN parcels (no voxels after masking): set to 0.0
      - All-zero images: flag for QC rejection
      - Negative-only images (e.g. deactivation maps): keep as-is
    """
    pass  # implement with nilearn


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: QUALITY CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
"""
Reject bad maps BEFORE averaging or training.

QC criteria (applied to each 392-D vector):
"""

@dataclass
class QCResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)

def qc_check(vec: np.ndarray, metadata: dict) -> QCResult:
    """
    QC pipeline for a single 392-D vector.
    
    REJECT if ANY of:
    
    1. ALL-ZERO CHECK
       - More than 95% of parcels are exactly 0.0
       - Indicates failed registration or empty mask
       
    2. NaN CHECK
       - More than 10% of parcels are NaN
       - Indicates partial brain coverage or registration failure
       - Note: a few NaN parcels (e.g. ventral cerebellum in some FOVs)
         is acceptable; replace with 0.0 after passing QC
    
    3. EXTREME VALUE CHECK
       - Any parcel value > 50 or < -50 (for T/Z maps)
       - For beta maps: any value > 1000 or < -1000
       - Indicates unmasked non-brain voxels contaminating parcels
       
    4. CONSTANT CHECK
       - Standard deviation across parcels < 0.01
       - Indicates a flat map (intercept-only or failed contrast)
    
    5. SPATIAL IMPLAUSIBILITY CHECK
       - Correlation with the mean map of same collection < -0.3
       - Indicates a map that's flipped, inverted, or from wrong subject
       - Only apply within collections with >10 maps
    
    6. MODALITY CHECK
       - Reject if map_type is "parcellation", "ROI", "atlas"
       - These are not activation/effect maps
       
    7. MAP TYPE NORMALIZATION
       - Accept: T, Z, beta, cope, contrast
       - Flag but accept: F-maps (convert to Z), chi-square
       - Reject: p-value maps, label maps, probability maps
    """
    reasons = []
    
    # 1. All-zero
    zero_frac = np.sum(vec == 0.0) / len(vec)
    if zero_frac > 0.95:
        reasons.append(f"all_zero: {zero_frac:.1%} zeros")
    
    # 2. NaN
    nan_frac = np.sum(np.isnan(vec)) / len(vec)
    if nan_frac > 0.10:
        reasons.append(f"too_many_nan: {nan_frac:.1%} NaN")
    
    # 3. Extreme values
    vec_clean = vec[~np.isnan(vec)]
    if len(vec_clean) > 0:
        map_type = metadata.get("map_type", "T")
        threshold = 1000 if map_type == "beta" else 50
        if np.max(np.abs(vec_clean)) > threshold:
            reasons.append(f"extreme_values: max={np.max(np.abs(vec_clean)):.1f}")
    
    # 4. Constant
    if len(vec_clean) > 0 and np.std(vec_clean) < 0.01:
        reasons.append(f"constant_map: std={np.std(vec_clean):.4f}")
    
    # 5. Spatial implausibility — requires collection context, done in batch
    # 6. Modality — checked at download time
    # 7. Map type — checked at download time
    
    return QCResult(passed=len(reasons) == 0, reasons=reasons)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: SUBJECT-LEVEL AVERAGING
# ═══════════════════════════════════════════════════════════════════════════════
"""
For collections flagged needs_averaging=True:
  Group images by contrast, average within group.

This is the CRITICAL stage for data consistency.
"""

def determine_grouping_key(image_metadata: dict, collection_spec: CollectionSpec) -> str:
    """
    Determine the grouping key for averaging.
    
    Priority order for finding the "contrast identity":
    
    1. cognitive_contrast_cogatlas_id  (most reliable — standardized CogAtlas URI)
       e.g. "trm_4f244b6f13850" → "2-back vs 0-back"
       
    2. contrast_definition  (NeuroVault field; usually populated for well-curated collections)
       e.g. "2-back > 0-back", "faces > shapes"
       
    3. cognitive_paradigm_cogatlas_id + name parsing
       e.g. paradigm="N-back" + name contains "2back_gt_0back"
       
    4. image name parsing (last resort)
       Strip subject IDs, run numbers, file extensions
       e.g. "sub-01_task-nback_contrast-2bk_gt_0bk_tmap.nii.gz" 
            → "task-nback_contrast-2bk_gt_0bk"
       
    For UCLA LA5C (4343) specifically:
       - Parse BIDS-like filenames
       - Group by task + contrast, SEPARATELY for healthy vs clinical
       - This gives you both "healthy group average" and "clinical group average"
       
    For IBC (6618, 2138):
       - contrast_definition is well-populated
       - Direct grouping works
       
    For BrainPedia (1952):
       - Heterogeneous sources
       - May need manual curation of grouping keys
       - Fall back to name-based clustering
    """
    # Try in priority order
    if image_metadata.get("cognitive_contrast_cogatlas_id"):
        return image_metadata["cognitive_contrast_cogatlas_id"]
    
    if image_metadata.get("contrast_definition"):
        cd = image_metadata["contrast_definition"].strip().lower()
        if cd and cd != "unknown" and cd != "nan":
            return cd
    
    # Parse image name
    name = image_metadata.get("name", "")
    return _parse_contrast_from_name(name, collection_spec)


def _parse_contrast_from_name(name: str, spec: CollectionSpec) -> str:
    """
    Extract contrast identity from image filename.
    
    Strategy:
      1. Remove known subject-ID patterns: sub-XX, subj_XX, s01, etc.
      2. Remove known run patterns: run-XX, run_XX, ses-XX
      3. Remove file extensions: .nii.gz, .nii, .img
      4. Remove collection-specific noise (configurable per collection)
      5. What remains is the contrast key
      
    Example:
      "sub-01_task-emotionalregulation_contrast-regulate_gt_look_neg_zstat.nii.gz"
      → "task-emotionalregulation_contrast-regulate_gt_look_neg"
    """
    import re
    
    # Strip extensions
    name = re.sub(r'\.(nii\.gz|nii|img|hdr)$', '', name)
    
    # Strip subject identifiers
    name = re.sub(r'sub[_-]?\d+', '', name)
    name = re.sub(r'subj[_-]?\d+', '', name)
    name = re.sub(r's\d{2,3}(?=[_-])', '', name)
    
    # Strip run/session
    name = re.sub(r'run[_-]?\d+', '', name)
    name = re.sub(r'ses[_-]?\w+', '', name)
    
    # Strip stat type suffixes (keep the contrast, drop the stat label)
    name = re.sub(r'[_-]?(zstat|tstat|tmap|zmap|cope|varcope)\d*', '', name)
    
    # Clean up
    name = re.sub(r'[_-]+', '_', name).strip('_')
    
    return name if name else "unknown"


def average_collection(
    images: List[RawImage], 
    spec: CollectionSpec,
    min_subjects: int = 3
) -> Dict[str, np.ndarray]:
    """
    Average subject-level maps within a collection.
    
    INPUT:  List of RawImage objects (all from same collection, all passed QC)
    OUTPUT: Dict mapping contrast_key → averaged 392-D vector
    
    Algorithm:
    
    1. GROUP images by contrast key
       groups = defaultdict(list)
       for img in images:
           key = determine_grouping_key(img.metadata, spec)
           groups[key].append(img.vector)
    
    2. For each group:
       a. If n_images < min_subjects (default 3): SKIP this contrast
          - Too few subjects for a reliable group average
          - Exception: if collection has ≤3 subjects total (e.g. dense sampling),
            use min_subjects=1
            
       b. OUTLIER REMOVAL within group:
          - Compute pairwise correlations between all maps in the group
          - If any map has mean correlation with others < 0.2: exclude it
          - This catches misclassified contrasts or failed subjects
          
       c. AVERAGE:
          - Simple arithmetic mean across remaining maps
          - avg = np.nanmean(np.stack(vectors), axis=0)
          
       d. VERIFY:
          - Run QC on the averaged map (should pass easily)
          - If it fails: flag collection for manual review
          
    3. Return: {contrast_key: avg_vector, ...}
    
    SPECIAL CASES:
    
    UCLA LA5C (4343):
      - Has healthy controls AND clinical groups (SZ, ADHD, bipolar)
      - Average separately: "faces_gt_shapes__healthy" vs "faces_gt_shapes__schizophrenia"
      - Both go into training with different labels
      - The clinical averages go into DomainTag.CLINICAL
    
    IBC (6618):
      - 13 subjects × 205 contrasts = very clean
      - Some contrasts may have <13 subjects (missed sessions)
      - min_subjects=5 is safe here
      
    BrainPedia (1952):
      - Heterogeneous; some "contrasts" may have only 1-2 maps
      - Be permissive: min_subjects=2
      - Accept that some will be noisy; the training set is large enough
    """
    groups = defaultdict(list)
    
    for img in images:
        key = determine_grouping_key(
            {"contrast_definition": img.contrast_definition,
             "cognitive_contrast_cogatlas_id": img.cognitive_contrast,
             "cognitive_paradigm_cogatlas_id": img.cognitive_paradigm,
             "name": img.name},
            spec
        )
        groups[key].append(img.vector)
    
    results = {}
    stats = {"n_contrasts": 0, "n_skipped_too_few": 0, "n_outliers_removed": 0}
    
    for key, vectors in groups.items():
        # Skip if too few subjects
        if len(vectors) < min_subjects:
            stats["n_skipped_too_few"] += 1
            continue
        
        # Stack for batch operations
        mat = np.stack(vectors)  # shape (n_subjects, 392)
        
        # Outlier removal via pairwise correlation
        if len(vectors) >= 5:  # only if enough to meaningfully compute
            corr_mat = np.corrcoef(mat)
            mean_corr = np.nanmean(corr_mat, axis=1) - 1/(len(vectors))  # subtract self-corr
            keep = mean_corr > 0.2
            if np.sum(keep) >= min_subjects:
                n_removed = np.sum(~keep)
                if n_removed > 0:
                    stats["n_outliers_removed"] += n_removed
                    mat = mat[keep]
        
        # Average
        avg = np.nanmean(mat, axis=0)
        
        # Replace any remaining NaN with 0
        avg = np.nan_to_num(avg, nan=0.0)
        
        results[key] = avg
        stats["n_contrasts"] += 1
    
    return results, stats


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: MAP TYPE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
Different collections use different stat types (T, Z, beta, F).
We need to normalize so they're comparable in training.

The problem: T-values and Z-values scale with sample size (N=20 study gives
smaller T-values than N=200 study for same effect). Betas are in arbitrary
units that depend on the design matrix scaling.

Options:
  A) Z-score each map to zero mean, unit variance across parcels
     → Preserves spatial pattern, removes scale
     → RECOMMENDED for training; your model learns PATTERNS not magnitudes
     
  B) Convert everything to Cohen's d (requires knowing N)
     → More principled but N often unknown for NeuroVault maps
     
  C) Leave as-is
     → Only if all maps are same stat type (they're not)
"""

def normalize_map(vec: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a 392-D activation vector.
    
    method="zscore":  (x - mean) / std  across 392 parcels
       - Each map becomes a spatial pattern with mean=0, std=1
       - The model learns relative activation patterns, not absolute magnitudes
       - This is what you want: "this compound activates temporal > frontal"
         not "this compound produces T=4.2 in temporal cortex"
    
    method="rank":  rank-based normalization (Spearman-like)
       - Convert to ranks, then to Gaussian quantiles
       - More robust to extreme values
       - Loses fine-grained amplitude information
       
    method="minmax":  scale to [0, 1]
       - Preserves relative scaling within map
       - But makes all maps "equal loudness" which removes information
       
    RECOMMENDED: "zscore" for most cases.
    Exception: If you later want to compare MAGNITUDES across compounds
    (e.g. "ketamine has stronger effects than modafinil"), you'd need
    to preserve scale somehow. But for training the embedding model,
    zscore is correct.
    """
    if method == "zscore":
        clean = vec[~np.isnan(vec)]
        if len(clean) == 0 or np.std(clean) < 1e-10:
            return np.zeros_like(vec)
        return (vec - np.nanmean(vec)) / np.nanstd(vec)
    
    elif method == "rank":
        from scipy.stats import rankdata, norm
        ranks = rankdata(np.nan_to_num(vec, nan=0.0))
        # Convert ranks to Gaussian quantiles
        quantiles = norm.ppf(ranks / (len(ranks) + 1))
        return quantiles
    
    elif method == "minmax":
        vmin, vmax = np.nanmin(vec), np.nanmax(vec)
        if vmax - vmin < 1e-10:
            return np.zeros_like(vec)
        return (vec - vmin) / (vmax - vmin)
    
    else:
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: SEMANTIC LABELING
# ═══════════════════════════════════════════════════════════════════════════════
"""
Each training sample needs a TEXT LABEL for the Generalizer (text → brain).
The quality of this label determines the quality of your semantic embeddings.

The label is what gets embedded by text-embedding-3-large and paired with
the 392-D brain vector during training.
"""

@dataclass  
class TrainingSample:
    vector: np.ndarray              # 392-D normalized brain map
    label: str                      # Semantic text label for embedding
    source: str                     # Source identifier for stratified sampling
    domain: DomainTag               # For stratified sampling
    agg_level: AggLevel             # For potential weighting
    collection_id: int
    contrast_key: str               # Original contrast key
    
    # Optional enrichment
    cognitive_atlas_id: Optional[str] = None
    cognitive_atlas_concept: Optional[str] = None
    drug_name: Optional[str] = None
    disorder_name: Optional[str] = None


def generate_label(contrast_key: str, collection_spec: CollectionSpec, 
                   metadata: dict) -> str:
    """
    Generate the text label for a training sample.
    
    This is CRITICAL — the label determines what the model learns.
    
    HIERARCHY of label sources (best → worst):
    
    1. COGNITIVE ATLAS CONCEPT (best)
       If cognitive_contrast_cogatlas_id maps to a CogAtlas concept,
       use the concept's full name.
       e.g. "working memory maintenance during n-back task"
       → Clean, standardized, semantically rich
       
    2. CURATED CONTRAST DEFINITION (good)
       If contrast_definition is well-formed:
       e.g. "2-back > 0-back" → "working memory load: 2-back greater than 0-back"
       Expand common abbreviations:
         "WM" → "working memory"
         "EF" → "executive function"  
         "SST" → "stop signal task"
         "MID" → "monetary incentive delay"
         "GT" / ">" → "greater than"
         "LT" / "<" → "less than"
         "VS" / "vs" → "versus"
       
    3. COLLECTION NAME + CONTRAST (acceptable)
       Combine collection theme with contrast key:
       e.g. collection="Fear conditioning" + contrast="CS+ > CS-"
       → "fear conditioning conditioned stimulus greater than unconditioned"
       
    4. RAW FILENAME (worst, but better than nothing)
       Parse what you can from the filename.
       e.g. "task-nback_contrast-2bk_gt_0bk"
       → "n-back task 2-back greater than 0-back"
       
    ENRICHMENT for specific domains:
    
    Pharmacological:
      Label = drug name + " acute effect on brain activation"
      e.g. "psilocybin acute effect on brain activation versus placebo"
      
    Clinical/Disorder:
      Label = disorder + contrast
      e.g. "schizophrenia patients versus controls during working memory"
      
    Pain:
      Label = pain type + intensity  
      e.g. "thermal pain high intensity versus warm control"
    
    The goal: when text-embedding-3-large embeds this label, it should
    land near other semantically similar concepts in embedding space.
    "2-back > 0-back" should be near "working memory load" and
    "prefrontal executive function" in embedding space.
    """
    # Try CogAtlas first
    cogatlas_id = metadata.get("cognitive_contrast_cogatlas_id")
    if cogatlas_id:
        concept_name = COGATLAS_LOOKUP.get(cogatlas_id)
        if concept_name:
            return concept_name
    
    # Try contrast_definition
    cd = metadata.get("contrast_definition", "")
    if cd and cd.lower() not in ("", "nan", "unknown", "none"):
        return _expand_contrast_label(cd, collection_spec)
    
    # Fall back to parsed key
    return _expand_contrast_label(contrast_key, collection_spec)


# Common abbreviation expansions
ABBREVIATIONS = {
    "wm": "working memory",
    "ef": "executive function",
    "sst": "stop signal task",
    "mid": "monetary incentive delay",
    "ant": "attention network task",
    "tom": "theory of mind",
    "dmn": "default mode network",
    "dlpfc": "dorsolateral prefrontal cortex",
    "acc": "anterior cingulate cortex",
    "nback": "n-back",
    "2bk": "2-back",
    "0bk": "0-back",
    "gt": "greater than",
    "lt": "less than",
    "vs": "versus",
    "ctrl": "control",
    "neg": "negative",
    "pos": "positive",
    "neut": "neutral",
    "rew": "reward",
    "pun": "punishment",
    "cs+": "conditioned stimulus positive",
    "cs-": "conditioned stimulus negative",
}

def _expand_contrast_label(raw: str, spec: CollectionSpec) -> str:
    """Expand abbreviations and clean up a contrast label."""
    label = raw.lower().replace("_", " ").replace("-", " ")
    for abbr, full in ABBREVIATIONS.items():
        label = label.replace(abbr, full)
    # Add collection context if label is too short/vague
    if len(label.split()) < 3:
        label = f"{spec.name}: {label}"
    return label.strip()


# Placeholder for CogAtlas ID → name lookup
COGATLAS_LOOKUP = {}  # Load from CogAtlas API or local cache


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
Multiple collections may contain overlapping maps:
  - Same study uploaded to multiple collections
  - Re-analyses of the same dataset  
  - NeuroSynth/NeuroQuery terms that partially overlap with NeuroVault maps

Dedup strategy:
"""

def deduplicate(samples: List[TrainingSample], 
                corr_threshold: float = 0.95) -> List[TrainingSample]:
    """
    Remove near-duplicate training samples.
    
    Algorithm:
    
    1. EXACT DUPLICATE CHECK
       - Hash each 392-D vector (round to 3 decimal places, then hash)
       - Remove exact duplicates (keep first encountered)
       
    2. NEAR-DUPLICATE CHECK (within same domain)
       - For each pair of samples in the same DomainTag:
         compute Pearson correlation between vectors
       - If corr > 0.95: these are effectively the same map
         Keep the one with:
           a. Better label quality (longer, more specific)
           b. Higher aggregation level (meta > group > subject-average)
           c. If tied: keep the one from higher-tier collection
           
    3. CROSS-SOURCE DUPLICATE CHECK
       - Compare NeuroVault maps against NeuroSynth/NeuroQuery maps
       - If a NeuroVault map correlates >0.95 with a NeuroSynth term:
         keep BOTH (they reinforce the same mapping with different labels)
         BUT don't count both in the training set size for sampling weights
    
    Computational note:
       - With 20K+ maps, pairwise correlation is 20K×20K = 400M pairs
       - Solution: block by domain, then use random projections for 
         approximate nearest-neighbor search
       - Or: just do within-collection dedup (fast) and accept some
         cross-collection duplicates (they'll have different labels anyway)
    
    PRACTICAL RECOMMENDATION:
       - Do exact dedup (Stage 6.1) always
       - Do within-collection near-dedup (Stage 6.2) always
       - Skip cross-collection dedup for v1 — the label differences
         actually help training by giving the model multiple text→brain pairings
    """
    # Stage 6.1: Exact dedup
    seen_hashes = set()
    unique = []
    
    for s in samples:
        h = hash(tuple(np.round(s.vector, 3)))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(s)
    
    n_exact_dupes = len(samples) - len(unique)
    
    # Stage 6.2: Within-collection near-dedup
    by_collection = defaultdict(list)
    for s in unique:
        by_collection[s.collection_id].append(s)
    
    final = []
    n_near_dupes = 0
    
    for coll_id, coll_samples in by_collection.items():
        if len(coll_samples) <= 1:
            final.extend(coll_samples)
            continue
        
        # Compute pairwise correlations within collection
        vecs = np.stack([s.vector for s in coll_samples])
        # For large collections, subsample
        if len(coll_samples) > 500:
            # Skip near-dedup for very large collections
            final.extend(coll_samples)
            continue
            
        corr_matrix = np.corrcoef(vecs)
        
        # Greedy removal: mark duplicates
        removed = set()
        for i in range(len(coll_samples)):
            if i in removed:
                continue
            for j in range(i+1, len(coll_samples)):
                if j in removed:
                    continue
                if corr_matrix[i, j] > corr_threshold:
                    # Keep i (encountered first), remove j
                    removed.add(j)
                    n_near_dupes += 1
        
        for i, s in enumerate(coll_samples):
            if i not in removed:
                final.append(s)
    
    print(f"Dedup: {len(samples)} → {len(final)} "
          f"(removed {n_exact_dupes} exact, {n_near_dupes} near-duplicates)")
    
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7: SOURCE-STRATIFIED SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════
"""
The core problem: your training set is dominated by NeuroVault task maps
(~20K) with small numbers of pharma (632), disorder (49), receptor (40).

If you train with uniform sampling, the model learns task activation patterns
and barely sees the rare categories that matter most for nootropics predictions.

Solution: stratified sampling that balances across sources and domains.
"""

@dataclass
class SamplingConfig:
    """Configuration for stratified batch sampling."""
    
    # Target proportions per domain in each training batch
    # These should sum to 1.0
    domain_weights: Dict[DomainTag, float] = field(default_factory=lambda: {
        DomainTag.COGNITIVE:       0.20,   # Largest source, downsample
        DomainTag.EMOTION:         0.10,
        DomainTag.SENSORIMOTOR:    0.08,
        DomainTag.SOCIAL:          0.07,
        DomainTag.CLINICAL:        0.15,   # Important for disorder predictions
        DomainTag.PHARMACOLOGICAL: 0.15,   # Critical for drug predictions — upsample
        DomainTag.STRUCTURAL:      0.05,
        DomainTag.CONNECTIVITY:    0.05,
        DomainTag.REFERENCE:       0.05,   # Receptor maps, neurotransmitter maps
        DomainTag.META_MIXED:      0.10,
    })
    
    # Weighting by aggregation level
    # Higher-quality maps (meta-analytic) get higher weight in loss
    agg_weights: Dict[AggLevel, float] = field(default_factory=lambda: {
        AggLevel.META_ANALYTIC: 2.0,    # Highest quality — weight up
        AggLevel.GROUP_AVERAGE: 1.0,    # Standard
        AggLevel.SUBJECT_LEVEL: 0.8,    # After averaging, still noisier
        AggLevel.ATLAS: 1.0,
        AggLevel.MULTIVARIATE: 1.5,     # Decoder weights are high-quality
    })
    
    batch_size: int = 256
    oversample_rare: bool = True        # If a domain has fewer maps than
                                        # its target proportion, allow repeats


class StratifiedSampler:
    """
    Samples training batches with domain-balanced proportions.
    
    Each batch of size B contains approximately:
      B * domain_weights[domain] samples from each domain.
      
    Within each domain, samples are drawn uniformly.
    Rare domains are oversampled (with replacement) to meet targets.
    Abundant domains are undersampled (without replacement per epoch).
    
    This ensures the model sees:
      - Pharmacological maps in ~15% of every batch (vs ~3% with uniform)
      - Clinical maps in ~15% of every batch (vs ~0.2% with uniform)
      - Cognitive maps in ~20% of every batch (vs ~88% with uniform)
    """
    
    def __init__(self, samples: List[TrainingSample], config: SamplingConfig):
        self.config = config
        
        # Index samples by domain
        self.domain_indices: Dict[DomainTag, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            self.domain_indices[s.domain].append(i)
        
        self.samples = samples
        self.n_total = len(samples)
        
        # Pre-compute per-sample loss weights based on aggregation level
        self.loss_weights = np.array([
            config.agg_weights.get(s.agg_level, 1.0) for s in samples
        ])
        
        # Compute actual proportions vs target
        print("\nSampling distribution:")
        print(f"{'Domain':<20} {'Actual':>8} {'Target':>8} {'Over/Under':>12}")
        for domain, target_w in sorted(config.domain_weights.items(), 
                                        key=lambda x: -x[1]):
            actual_n = len(self.domain_indices[domain])
            actual_frac = actual_n / self.n_total if self.n_total > 0 else 0
            ratio = actual_frac / target_w if target_w > 0 else float('inf')
            status = "oversample" if ratio < 0.5 else "undersample" if ratio > 2 else "~balanced"
            print(f"  {domain.value:<18} {actual_frac:>7.1%} {target_w:>7.1%} {status:>12}")
    
    def sample_batch(self, rng: np.random.Generator) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Sample one training batch.
        
        Returns:
          vectors: (batch_size, 392) array of brain maps
          labels:  list of text labels (for embedding)
          weights: (batch_size,) array of loss weights
        """
        B = self.config.batch_size
        batch_indices = []
        
        for domain, target_w in self.config.domain_weights.items():
            n_from_domain = max(1, int(B * target_w))
            domain_pool = self.domain_indices.get(domain, [])
            
            if len(domain_pool) == 0:
                continue
            
            if len(domain_pool) >= n_from_domain:
                # Undersample: draw without replacement
                chosen = rng.choice(domain_pool, size=n_from_domain, replace=False)
            else:
                # Oversample: draw with replacement
                chosen = rng.choice(domain_pool, size=n_from_domain, replace=True)
            
            batch_indices.extend(chosen)
        
        # Shuffle the batch
        rng.shuffle(batch_indices)
        
        # Trim to exact batch size
        batch_indices = batch_indices[:B]
        
        vectors = np.stack([self.samples[i].vector for i in batch_indices])
        labels = [self.samples[i].label for i in batch_indices]
        weights = np.array([self.loss_weights[i] for i in batch_indices])
        
        return vectors, labels, weights


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8: EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
Generate text embeddings for all training labels using text-embedding-3-large.

Do this ONCE and cache. Don't re-embed during training.
"""

def generate_embeddings(samples: List[TrainingSample], 
                        model: str = "text-embedding-3-large",
                        batch_size: int = 100) -> np.ndarray:
    """
    Embed all training labels.
    
    Algorithm:
    
    1. Collect unique labels (many samples may share labels)
       unique_labels = list(set(s.label for s in samples))
       
    2. Batch-embed via OpenAI API
       - text-embedding-3-large produces 3072-D vectors
       - Process in batches of 100 to avoid rate limits
       - Cache results to disk: {label_hash: embedding}
       
    3. Map back to samples
       embeddings[i] = embedding_cache[samples[i].label]
       
    4. Save as numpy array: shape (n_samples, 3072)
    
    COST ESTIMATE:
      - 20K unique labels × ~20 tokens each = ~400K tokens
      - text-embedding-3-large: $0.00013 per 1K tokens
      - Total: ~$0.05 (trivial)
      
    CACHING:
      - Save embedding cache as {label: embedding} pickle/npz
      - On subsequent runs, only embed NEW labels
      - This matters when adding new collections incrementally
    """
    import hashlib, pickle
    from pathlib import Path
    
    cache_path = Path("embeddings_cache.pkl")
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    
    # Find labels not yet cached
    unique_labels = list(set(s.label for s in samples))
    to_embed = [l for l in unique_labels if l not in cache]
    
    print(f"Labels: {len(unique_labels)} unique, {len(to_embed)} need embedding")
    
    if to_embed:
        # Call OpenAI API in batches
        # from openai import OpenAI
        # client = OpenAI()
        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i:i+batch_size]
            # response = client.embeddings.create(model=model, input=batch)
            # for j, emb in enumerate(response.data):
            #     cache[batch[j]] = np.array(emb.embedding)
            pass
        
        # Save updated cache
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    
    # Map to samples
    embeddings = np.stack([cache[s.label] for s in samples])
    return embeddings  # shape (n_samples, 3072)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9: FINAL TRAINING SET ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_set(
    collections_dir: str,
    atlas_path: str = "combined_atlas_392.nii.gz",
    output_path: str = "training_set_v2.npz"
):
    """
    MASTER PIPELINE: collections → training set.
    
    Full flow:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ For each collection in COLLECTION_REGISTRY:                    │
    │                                                                │
    │   STAGE 1: Download & Parcellate                               │
    │     raw NIfTI → 392-D vectors                                  │
    │                                                                │
    │   STAGE 2: QC                                                  │
    │     reject bad maps (all-zero, NaN, extreme, constant)         │
    │                                                                │
    │   STAGE 3: Average (if needs_averaging=True)                   │
    │     subject-level maps → group averages per contrast           │
    │                                                                │
    │   STAGE 4: Normalize                                           │
    │     z-score each map across parcels                            │
    │                                                                │
    │   STAGE 5: Label                                               │
    │     generate semantic text label for each map                  │
    │                                                                │
    │   → Append to master list of TrainingSamples                   │
    └─────────────────────────────────────────────────────────────────┘
    
    Then globally:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ STAGE 6: Deduplicate                                           │
    │   remove exact and near-duplicate maps                         │
    │                                                                │
    │ STAGE 7: Configure stratified sampler                          │
    │   set domain weights for balanced training                     │
    │                                                                │
    │ STAGE 8: Generate text embeddings                              │
    │   embed all labels with text-embedding-3-large                 │
    │                                                                │
    │ STAGE 9: Save                                                  │
    │   vectors (N, 392) + embeddings (N, 3072) + metadata           │
    └─────────────────────────────────────────────────────────────────┘
    
    Output file format:
      training_set_v2.npz containing:
        - "vectors":      (N, 392) float32  — brain maps
        - "embeddings":   (N, 3072) float32 — text embeddings  
        - "labels":       (N,) string array  — text labels
        - "sources":      (N,) string array  — source identifiers
        - "domains":      (N,) string array  — domain tags
        - "agg_levels":   (N,) string array  — aggregation levels
        - "collection_ids": (N,) int32       — NeuroVault collection IDs
        - "loss_weights":   (N,) float32     — per-sample loss weights
        
    Expected sizes after full build:
      - NeuroVault task (after averaging): ~2,000-3,000 maps
      - NeuroQuery decoder: ~5,000-7,000 maps  
      - NeuroSynth: ~100-3,400 maps
      - NeuroVault pharma: ~300-600 maps (after dedup)
      - ENIGMA disorder: ~49-100 maps
      - neuromaps receptor: ~40 maps
      - abagen gene expression: NOT in text→brain training 
        (separate pathway via gene PCA)
      
      TOTAL: ~8,000-14,000 training samples for text→brain Generalizer
      
    Note: abagen (15K gene maps) and PDSP projections go through
    the PHARMACOLOGICAL pathway (receptor→gene→brain), not through
    the semantic embedding pathway. They're not in this training set.
    """
    all_samples = []
    
    for coll_id, spec in COLLECTION_REGISTRY.items():
        print(f"\nProcessing collection {coll_id}: {spec.name}")
        
        # Stage 1: Load parcellated vectors
        images = load_collection_vectors(collections_dir, coll_id, atlas_path)
        print(f"  Loaded {len(images)} images")
        
        # Stage 2: QC
        passed = []
        for img in images:
            qc = qc_check(img.vector, {"map_type": img.map_type})
            if qc.passed:
                passed.append(img)
        print(f"  QC: {len(passed)}/{len(images)} passed")
        
        # Stage 3: Average if needed
        if spec.needs_averaging:
            min_subj = 2 if coll_id == 1952 else 3  # BrainPedia is heterogeneous
            averaged, stats = average_collection(passed, spec, min_subjects=min_subj)
            print(f"  Averaged: {stats['n_contrasts']} contrasts "
                  f"(skipped {stats['n_skipped_too_few']} with too few subjects, "
                  f"removed {stats['n_outliers_removed']} outlier maps)")
            
            # Convert averaged maps to TrainingSamples
            for contrast_key, vec in averaged.items():
                vec_norm = normalize_map(vec, method="zscore")
                label = generate_label(contrast_key, spec, {})
                sample = TrainingSample(
                    vector=vec_norm,
                    label=label,
                    source=f"neurovault_{coll_id}",
                    domain=spec.domain,
                    agg_level=AggLevel.GROUP_AVERAGE,  # now group-level after averaging
                    collection_id=coll_id,
                    contrast_key=contrast_key,
                )
                all_samples.append(sample)
        else:
            # Use as-is (already group-level)
            for img in passed:
                vec_norm = normalize_map(img.vector, method="zscore")
                label = generate_label(img.name, spec, {
                    "contrast_definition": img.contrast_definition,
                    "cognitive_contrast_cogatlas_id": img.cognitive_contrast,
                })
                sample = TrainingSample(
                    vector=vec_norm,
                    label=label,
                    source=f"neurovault_{coll_id}",
                    domain=spec.domain,
                    agg_level=spec.agg_level,
                    collection_id=coll_id,
                    contrast_key=img.contrast_definition or img.name,
                )
                all_samples.append(sample)
    
    print(f"\n{'='*60}")
    print(f"Total samples before dedup: {len(all_samples)}")
    
    # Stage 6: Dedup
    all_samples = deduplicate(all_samples)
    
    # Stage 8: Embeddings
    embeddings = generate_embeddings(all_samples)
    
    # Stage 9: Save
    np.savez_compressed(
        output_path,
        vectors=np.stack([s.vector for s in all_samples]).astype(np.float32),
        embeddings=embeddings.astype(np.float32),
        labels=np.array([s.label for s in all_samples]),
        sources=np.array([s.source for s in all_samples]),
        domains=np.array([s.domain.value for s in all_samples]),
        agg_levels=np.array([s.agg_level.value for s in all_samples]),
        collection_ids=np.array([s.collection_id for s in all_samples], dtype=np.int32),
        loss_weights=np.array([
            SamplingConfig().agg_weights.get(s.agg_level, 1.0) 
            for s in all_samples
        ], dtype=np.float32),
    )
    
    print(f"\nSaved {len(all_samples)} samples to {output_path}")
    print(f"  Vectors: ({len(all_samples)}, 392)")
    print(f"  Embeddings: ({len(all_samples)}, 3072)")
    
    return all_samples


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 10: TRAINING LOOP INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
How the training set plugs into your V2 Generalizer training:

    training_set = np.load("training_set_v2.npz")
    vectors = training_set["vectors"]       # (N, 392)
    embeddings = training_set["embeddings"] # (N, 3072)
    weights = training_set["loss_weights"]  # (N,)
    domains = training_set["domains"]       # (N,) string
    
    sampler = StratifiedSampler(samples, SamplingConfig())
    
    for epoch in range(n_epochs):
        for batch_idx in range(n_batches_per_epoch):
            
            # Get balanced batch
            batch_vecs, batch_labels, batch_weights = sampler.sample_batch(rng)
            
            # Look up pre-computed embeddings for these labels
            batch_embs = embeddings_lookup[batch_labels]  # (B, 3072)
            
            # Forward pass: embedding → predicted brain map
            predicted = generalizer(batch_embs)  # (B, 392)
            
            # Loss: weighted MSE
            # Each sample's loss is scaled by its aggregation-level weight
            mse = ((predicted - batch_vecs) ** 2).mean(dim=1)  # (B,)
            loss = (mse * batch_weights).mean()
            
            loss.backward()
            optimizer.step()
"""


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX: MERGING WITH NON-NEUROVAULT SOURCES
# ═══════════════════════════════════════════════════════════════════════════════
"""
Your training set includes sources beyond NeuroVault:

SOURCE                  MAPS     DOMAIN            HOW IT ENTERS
─────────────────────────────────────────────────────────────────
NeuroQuery decoder      5-7K     meta_mixed        Already 392-D; label = NQ term
NeuroSynth              100-3.4K meta_mixed        Already 392-D; label = NS term
NeuroVault (this pipe)  2-3K     all domains       This pipeline
NeuroVault pharma       300-600  pharmacological   This pipeline, tagged domain=pharma
ENIGMA                  49-100   clinical          Already 392-D; label = disorder name
neuromaps               40       reference         Already 392-D; label = receptor/annotation name

These all get concatenated AFTER their respective pipelines produce
TrainingSamples, then go through Stages 6-9 together.

NOT in text→brain training:
  - abagen (15K gene expression) → gene PCA pathway
  - PDSP Ki binding → pharmacological projection pathway
  - FC matrices → separate FC cache
  
These feed into the Memorizer and pharmacological pathway, not the Generalizer.


MERGING PROCEDURE:

1. Process each source through its own pipeline to get TrainingSamples
2. Tag each with its source string:
     - "neuroquery" 
     - "neurosynth"
     - "neurovault_{collection_id}"
     - "enigma"
     - "neuromaps"
3. Concatenate all into one master list
4. Run Stages 6-9 on the combined list
5. The stratified sampler handles the domain balancing across all sources

EXPECTED FINAL TRAINING SET:
  ~8,000-14,000 samples with text→brain pairings
  Each sample: (label, 3072-D embedding, 392-D brain vector, loss weight, domain)
"""

if __name__ == "__main__":
    print("This is a specification module. Import and call build_training_set().")
    print("\nPipeline stages:")
    stages = [
        "Stage 0: Collection classification & metadata",
        "Stage 1: Download & parcellate (NIfTI → 392-D)",
        "Stage 2: Quality control (reject bad maps)",
        "Stage 3: Subject-level averaging (group by contrast)",
        "Stage 4: Map type normalization (z-score)",
        "Stage 5: Semantic labeling (text for embeddings)", 
        "Stage 6: Deduplication (exact + near-duplicate removal)",
        "Stage 7: Stratified sampling config (domain-balanced batches)",
        "Stage 8: Text embedding generation (text-embedding-3-large)",
        "Stage 9: Save training set (.npz)",
        "Stage 10: Training loop integration (weighted loss)",
    ]
    for s in stages:
        print(f"  {s}")
