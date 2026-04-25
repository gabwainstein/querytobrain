"""
Expand common neuro/fMRI acronyms and jargon in contrast descriptions before encoding.

Use so the embedding model sees full phrases (e.g. "BART (Balloon Analog Risk Task)")
instead of bare acronyms, improving semantic representation. Applied at encode time
only; stored terms stay verbatim.
"""
import re
from typing import Union

# Acronym or jargon -> expansion (inserted as "ACRONYM (expansion)" on first occurrence)
NEURO_GLOSSARY = {
    "BART": "Balloon Analog Risk Task",
    "RSVP": "rapid serial visual presentation",
    "N-back": "n-back working memory",
    "n-back": "n-back working memory",
    "0-back": "0-back",
    "2-back": "2-back",
    "fMRI": "functional MRI",
    "MRI": "magnetic resonance imaging",
    "SPM": "statistical parametric map",
    "IBC": "Individual Brain Charting",
    "HCP": "Human Connectome Project",
    "DMN": "default mode network",
    "PFC": "prefrontal cortex",
    "ACC": "anterior cingulate cortex",
    "ROI": "region of interest",
    "GLM": "general linear model",
    "FWE": "family-wise error",
    "FDR": "false discovery rate",
}


def expand_abbreviations(text: Union[str, list]) -> Union[str, list]:
    """
    Replace known acronyms/jargon with "ACRONYM (expansion)" using word boundaries.
    If text is a list, expand each element.
    """
    if isinstance(text, list):
        return [expand_abbreviations(t) for t in text]
    if not text or not isinstance(text, str):
        return text
    out = text
    for acronym, expansion in NEURO_GLOSSARY.items():
        # Word-boundary replacement: avoid replacing inside words
        pattern = r"\b" + re.escape(acronym) + r"\b"
        # Only add expansion if not already present (e.g. "BART (Balloon...)" already there)
        out = re.sub(pattern, f"{acronym} ({expansion})", out)
    return out
