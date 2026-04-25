"""
Receptor gene knowledge base: canonical gene list and metadata for receptor atlas and training.

**Canonical source:** `neurolab/docs/implementation/receptor_gene_list_v2.csv` (curated list with
gene_symbol, gene_name, system, category, notes). Rows with category=EXCLUDE are omitted.

Example:
  from neurolab.receptor_kb import load_receptor_genes, load_receptor_kb

  genes = load_receptor_genes()  # list of 250 curated genes
  kb = load_receptor_kb()        # full KB with metadata from CSV
  desc = kb["metadata"]["HTR2A"]["description"]  # "5-hydroxytryptamine receptor 2A gene expression"
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _REPO_ROOT / "neurolab" / "docs" / "implementation" / "receptor_gene_list_v2.csv"
_DEFAULT_JSON = _REPO_ROOT / "neurolab" / "data" / "receptor_gene_names_v2.json"


def _load_csv(path: Path) -> list[dict]:
    """Parse CSV, exclude EXCLUDE rows."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("gene_symbol") or str(row.get("gene_symbol", "")).strip().startswith("=="):
                continue
            if str(row.get("category", "")).strip().upper() == "EXCLUDE":
                continue
            rows.append(row)
    return rows


def load_receptor_genes(path: Path | str | None = None) -> list[str]:
    """Load canonical receptor gene list. Prefers CSV; falls back to JSON."""
    p = Path(path) if path else _DEFAULT_CSV
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if p.suffix.lower() == ".csv" and p.exists():
        rows = _load_csv(p)
        return [r["gene_symbol"] for r in rows]
    # Fallback: JSON (e.g. neurolab/data/receptor_gene_names_v2.json)
    json_path = p if p.suffix.lower() == ".json" else _DEFAULT_JSON
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    if _DEFAULT_CSV.exists():
        rows = _load_csv(_DEFAULT_CSV)
        return [r["gene_symbol"] for r in rows]
    raise FileNotFoundError(f"Receptor list not found: {p} or {_DEFAULT_JSON}")


def load_receptor_kb(path: Path | str | None = None) -> dict:
    """Load full receptor knowledge base (genes, metadata). From CSV when available."""
    p = Path(path) if path else _DEFAULT_CSV
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if p.suffix.lower() == ".csv" and p.exists():
        rows = _load_csv(p)
        genes = [r["gene_symbol"] for r in rows]
        metadata = {}
        for r in rows:
            g = r["gene_symbol"]
            desc = r.get("gene_name", g) + " gene expression"
            metadata[g] = {
                "gene_name": r.get("gene_name", ""),
                "system": r.get("system", ""),
                "category": r.get("category", ""),
                "notes": r.get("notes", ""),
                "description": desc,
            }
        return {
            "_source": str(p),
            "_description": "Curated receptor/transporter/channel gene list from CSV",
            "genes": genes,
            "metadata": metadata,
        }
    # Fallback: JSON
    json_path = _REPO_ROOT / "neurolab" / "data" / "receptor_knowledge_base.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    if _DEFAULT_CSV.exists():
        return load_receptor_kb(_DEFAULT_CSV)
    raise FileNotFoundError(f"Receptor KB not found: {p} or {json_path}")


def _format_enriched_label(gene: str, meta: dict[str, str]) -> str:
    """Build a descriptive label for receptor genes."""
    name = (meta.get('gene_name') or gene).strip() or gene
    system = (meta.get('system') or '').strip()
    category = (meta.get('category') or '').strip()
    notes = (meta.get('notes') or '').strip()

    parts = [f"{gene} ({name}) gene expression"]
    context = []
    if system:
        context.append(f"{system} system")
    if category:
        context.append(category)
    if context:
        parts.append(' - ' + ', '.join(context))
    label = ''.join(parts)
    if notes:
        label = f"{label}; {notes}"
    return label



def get_gene_descriptions(path: Path | str | None = None) -> dict[str, str]:
    """Return {gene: description} for training (e.g. abagen term labels)."""
    kb = load_receptor_kb(path)
    meta = kb.get("metadata") or {}
    return {g: m.get("description", m.get("gene_name", g) + " gene expression") for g, m in meta.items()}


def get_rich_gene_descriptions(path: Path | str | None = None) -> dict[str, str]:
    """Return {gene: rich_label} for embedding quality. Combines gene_name, system, category, notes."""
    kb = load_receptor_kb(path)
    meta = kb.get("metadata") or {}
    out = {}
    for g, m in meta.items():
        label = _format_enriched_label(g, m)
        out[g] = label.replace("gene expression", "gene expression across cortex", 1)
    return out


def get_enriched_gene_labels(path: Path | str | None = None) -> dict[str, str]:
    """
    Return {gene: enriched_label} for text-to-brain training with detailed semantics.
    Format: "SYMBOL (full name) gene expression - System, Category; notes".
    """
    kb = load_receptor_kb(path)
    meta = kb.get("metadata") or {}
    return {g: _format_enriched_label(g, m) for g, m in meta.items()}

