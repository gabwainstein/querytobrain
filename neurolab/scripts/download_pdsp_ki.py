#!/usr/bin/env python3
"""
Download the full PDSP Ki Database (~98,685 Ki values).

The PDSP Ki DB (https://pdspdb.unc.edu/databases/kiDownload/) is an Angular app
backed by a PHP API. This script attempts multiple download strategies:

1. Direct download.php endpoint (whole database CSV)
2. Dev server direct CSV link
3. Paginated JSON API fallback (fetches all records in chunks)

Output: KiDatabase.csv in the specified output directory (default: neurolab/data/pdsp_ki/).
build_pdsp_cache.py expects this filename.

Usage:
  python neurolab/scripts/download_pdsp_ki.py
  python neurolab/scripts/download_pdsp_ki.py --output-dir /path/to/pdsp_ki
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Default output: neurolab/data/pdsp_ki/KiDatabase.csv
_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
_DEFAULT_OUT = _repo_root / "neurolab" / "data" / "pdsp_ki"

# ── Known endpoints ──────────────────────────────────────────────────────────

# Primary: the Angular app's download button calls this
DOWNLOAD_URLS = [
    # Main production server - whole database download
    "https://pdspdb.unc.edu/databases/kiDownload/download.php",
    # Dev server - whole database download
    "https://kidbdev.med.unc.edu/databases/kiDownload/download.php",
    # Dev server - incremental (fromDate=0 means all)
    "https://kidbdev.med.unc.edu/databases/kiDownload/download.php?fromDate=0",
    # Older static file endpoint (may be stale)
    "https://kidbdev.med.unc.edu/databases/downloadKi.html",
]

# JSON API endpoints (Angular app uses these to load paginated data)
JSON_API_URLS = [
    "https://pdspdb.unc.edu/databases/kiDownload/getKiData.php",
    "https://kidbdev.med.unc.edu/databases/kiDownload/getKiData.php",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept": "text/csv, text/plain, application/json, */*",
    "Referer": "https://pdspdb.unc.edu/databases/kiDownload/",
}

CSV_COLUMNS = [
    "ki_id", "receptor", "unigene_code", "ligand_id", "ligand_name",
    "smiles", "cas", "nsc", "hot_ligand", "species", "source",
    "ki_note", "ki_value", "reference", "pubmed_link"
]


def try_direct_download(output_path: Path) -> bool:
    """Try downloading the whole database as a CSV from known endpoints."""
    for url in DOWNLOAD_URLS:
        print(f"  Trying: {url}")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()

                # Check if we got something useful
                if len(data) < 1000:
                    print(f"    Response too small ({len(data)} bytes), skipping")
                    continue

                text = data.decode("utf-8", errors="replace")

                # Check if it's CSV-like (has commas and multiple lines)
                lines = text.strip().split("\n")
                if len(lines) > 100 and ("," in lines[0] or "\t" in lines[0]):
                    print(f"    [OK] Got CSV data: {len(lines)} lines, {len(data)} bytes")
                    output_path.write_text(text, encoding="utf-8")
                    return True

                # Check if it's JSON
                if text.strip().startswith("[") or text.strip().startswith("{"):
                    try:
                        records = json.loads(text)
                        if isinstance(records, list) and len(records) > 100:
                            print(f"    [OK] Got JSON data: {len(records)} records")
                            json_to_csv(records, output_path)
                            return True
                    except json.JSONDecodeError:
                        pass

                print(f"    Response doesn't look like data ({len(lines)} lines, content-type: {content_type})")

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"    [X] Error: {e}")
        except Exception as e:
            print(f"    [X] Unexpected error: {e}")

    return False


def try_paginated_json(output_path: Path, page_size: int = 5000) -> bool:
    """
    Try fetching all records via paginated JSON API.
    The Angular app loads data in pages; we replicate that.
    """
    for base_url in JSON_API_URLS:
        print(f"  Trying paginated API: {base_url}")
        all_records = []
        page = 0
        consecutive_errors = 0

        while consecutive_errors < 3:
            # Try various pagination parameter styles
            urls_to_try = [
                f"{base_url}?offset={page * page_size}&limit={page_size}",
                f"{base_url}?page={page}&pageSize={page_size}",
                f"{base_url}?start={page * page_size}&count={page_size}",
            ]

            got_data = False
            for url in urls_to_try:
                try:
                    req = urllib.request.Request(url, headers={
                        **HEADERS,
                        "Accept": "application/json",
                        "X-Requested-With": "XMLHttpRequest",
                    })
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = resp.read().decode("utf-8", errors="replace")

                        if not data.strip():
                            continue

                        records = json.loads(data)
                        if isinstance(records, dict) and "data" in records:
                            records = records["data"]

                        if isinstance(records, list):
                            if len(records) == 0:
                                # Reached the end
                                if all_records:
                                    print(f"    End of data at page {page}")
                                    json_to_csv(all_records, output_path)
                                    return True
                                continue

                            all_records.extend(records)
                            got_data = True
                            print(f"    Page {page}: got {len(records)} records (total: {len(all_records)})")

                            if len(records) < page_size:
                                # Last page
                                print(f"    [OK] Complete: {len(all_records)} total records")
                                json_to_csv(all_records, output_path)
                                return True
                            break

                except (urllib.error.URLError, urllib.error.HTTPError) as e:
                    continue
                except json.JSONDecodeError:
                    continue

            if got_data:
                page += 1
                consecutive_errors = 0
                time.sleep(0.5)  # Be polite
            else:
                consecutive_errors += 1
                if page == 0:
                    break  # This API style doesn't work, try next base URL

        if all_records:
            print(f"    [OK] Got {len(all_records)} records (may be incomplete)")
            json_to_csv(all_records, output_path)
            return True

    return False


def try_post_download(output_path: Path) -> bool:
    """
    Try POST request to download endpoint (some Angular apps use POST for downloads).
    """
    for base in ["https://pdspdb.unc.edu", "https://kidbdev.med.unc.edu"]:
        endpoints = [
            f"{base}/databases/kiDownload/download.php",
            f"{base}/databases/kiDownload/getKiData.php",
        ]
        for url in endpoints:
            print(f"  Trying POST: {url}")
            try:
                # POST with empty body or with download flag
                for post_data in [b"", b"download=true", b"action=downloadAll"]:
                    req = urllib.request.Request(
                        url,
                        data=post_data if post_data else None,
                        headers={
                            **HEADERS,
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                        method="POST" if post_data else "GET",
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = resp.read()
                        if len(data) > 10000:
                            text = data.decode("utf-8", errors="replace")
                            lines = text.strip().split("\n")
                            if len(lines) > 100:
                                print(f"    [OK] Got data: {len(lines)} lines")
                                # Try to detect if CSV or JSON
                                if text.strip().startswith("["):
                                    records = json.loads(text)
                                    json_to_csv(records, output_path)
                                else:
                                    output_path.write_text(text, encoding="utf-8")
                                return True
            except Exception as e:
                continue
    return False


def json_to_csv(records: list, output_path: Path) -> None:
    """Convert JSON records to CSV. Uses build_pdsp_cache-compatible column names."""
    if not records:
        return

    # Map Angular/API field names -> build_pdsp_cache expected columns
    raw_to_pdsp = {
        "number": "Ki ID", "ki_id": "Ki ID",
        "name": "Receptor", "receptor": "Receptor",
        "unigene": "UniGene Code", "unigene_code": "UniGene Code",
        "ligandid": "Ligand ID", "ligand_id": "Ligand ID",
        "ligandname": "Ligand Name", "ligand_name": "Ligand Name",
        "smiles": "SMILES", "cas": "CAS", "nsc": "NSC",
        "hotligand": "Hot Ligand", "hot_ligand": "Hot Ligand",
        "species": "Species", "source": "Source",
        "kinote": "Ki note", "ki_note": "Ki note",
        "kival": "Ki (nM)", "ki_value": "Ki (nM)",
        "reference": "Reference", "link": "PubMed link", "pubmed_link": "PubMed link",
    }

    sample = records[0]
    if isinstance(sample, dict):
        raw_fields = list(sample.keys())
        columns = [raw_to_pdsp.get(k, k) for k in raw_fields]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for rec in records:
                row = {raw_to_pdsp.get(k, k): v for k, v in rec.items()}
                writer.writerow(row)
    else:
        # List of lists
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if len(CSV_COLUMNS) == len(records[0]):
                writer.writerow(CSV_COLUMNS)
            for rec in records:
                writer.writerow(rec)

    print(f"    Wrote {len(records)} records to {output_path}")


def validate_csv(path: Path) -> dict:
    """Quick validation and summary of downloaded CSV."""
    stats = {"rows": 0, "receptors": set(), "ligands": set(), "ki_values": 0}

    try:
        with open(path, "r", errors="replace") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

            # Find the right column names (case-insensitive)
            def find_col(candidates):
                for c in candidates:
                    for f_name in fields:
                        if f_name.lower().strip() == c.lower():
                            return f_name
                return None

            receptor_col = find_col(["receptor", "name", "receptor_name", "Receptor"])
            ligand_col = find_col(["ligand_name", "ligandname", "Ligand name", "ligand"])
            ki_col = find_col(["ki_value", "kival", "Ki value", "ki", "Ki (nM)"])

            for row in reader:
                stats["rows"] += 1
                if receptor_col and row.get(receptor_col):
                    stats["receptors"].add(str(row[receptor_col]).strip())
                if ligand_col and row.get(ligand_col):
                    stats["ligands"].add(str(row[ligand_col]).strip())
                if ki_col and row.get(ki_col):
                    val = str(row[ki_col]).strip()
                    if val and val not in ("", "NA", "N/A", "-"):
                        stats["ki_values"] += 1

        stats["n_receptors"] = len(stats["receptors"])
        stats["n_ligands"] = len(stats["ligands"])
        del stats["receptors"], stats["ligands"]

    except Exception as e:
        print(f"  Validation error: {e}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PDSP Ki database (CSV).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save KiDatabase.csv (default: neurolab/data/pdsp_ki)",
    )
    parser.add_argument(
        "--output-name",
        default="KiDatabase.csv",
        help="Output filename (default: KiDatabase.csv for build_pdsp_cache compatibility)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output_name

    print("=" * 60)
    print("PDSP Ki Database Downloader")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Strategy 1: Direct CSV download
    print("\n[1/3] Trying direct CSV download...")
    if try_direct_download(output_path):
        stats = validate_csv(output_path)
        if stats.get("rows", 0) > 1000:
            print(f"\n[OK] SUCCESS: {stats}")
            return 0
        print("  Downloaded file seems too small, trying other methods...")

    # Strategy 2: POST-based download
    print("\n[2/3] Trying POST-based download...")
    if try_post_download(output_path):
        stats = validate_csv(output_path)
        if stats.get("rows", 0) > 1000:
            print(f"\n[OK] SUCCESS: {stats}")
            return 0

    # Strategy 3: Paginated JSON API
    print("\n[3/3] Trying paginated JSON API...")
    if try_paginated_json(output_path):
        stats = validate_csv(output_path)
        if stats.get("rows", 0) > 1000:
            print(f"\n[OK] SUCCESS: {stats}")
            return 0

    # If all automated methods fail, print manual instructions
    print("\n" + "=" * 60)
    print("AUTOMATED DOWNLOAD FAILED")
    print("=" * 60)
    print(f"""
The PDSP Ki Database uses an Angular frontend that may require
a browser session to trigger the download. Manual options:

1. BROWSER DOWNLOAD (recommended):
   - Go to: https://pdspdb.unc.edu/databases/kiDownload/
   - Click "Download Whole Database" button
   - Save the CSV file as KiDatabase.csv

2. ALTERNATIVE (dev server):
   - Go to: https://kidbdev.med.unc.edu/databases/kiDownload/
   - Click "Download Whole Database"

3. After manual download, place the CSV at:
   {output_path}
""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
