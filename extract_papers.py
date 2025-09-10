#!/usr/bin/env python3
"""
PDF → CSV/Excel extractor for AlloyMind papers (writes EXACT clean schema: 66 columns).

Run with defaults:
  python extract_papers.py
(Reads PDFs from ./AlloyMind and writes alloymind_dataset1.csv)

Override:
  python extract_papers.py --pdf_dir AlloyMind/papers --out_csv alloymind_dataset_clean.csv
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
from pdfminer.high_level import extract_text

# Quiet pdfminer noise
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# === The target schema (same order as your clean dataset) ===
TARGET_COLUMNS = [
    # Metadata
    "id","title","authors","year","venue","doi","url","keywords","domain_tag",
    # Unstructured text
    "abstract","introduction","methods","results_text","discussion","conclusion","fig_tbl_captions",
    # Structured (primary)
    "Alloy ID","Alloy","Temperature [C]","Yield Stress [MPa]","Ultimate Tensile Stress [MPa]",
    "Tensile elongation [%]","Reduction area [%]",
    # Composition (wt%)
    "Fe","C","Cr","Mn","Si","Ni","Co","Mo","W","Nb","Al","P","Cu","Ti","Ta","Hf","Re","V","B","N","O","S","Zr","Y",
    # Process / test metadata
    "process","heat_treatment","test_environment","load_type","standard_used",
    # Property key/val (generic)
    "property_name","property_value","units","notes",
    # Practical tags
    "application_tags","performance_category","cost_tag","availability_tag",
    # Synonyms / alternate columns used in some sources
    "alloy","composition_wt%","temperature_C","Temperature, [C]","Yield Stress, [MPa]","Ultimate Tensile Stress, [MPa]",
]

# Section headers to search for
SECTION_HEADERS = [
    "abstract","introduction","methods","materials and methods","experimental",
    "results","results and discussion","discussion","conclusion","conclusions",
]

# ---------- Heuristics ----------
def find_doi(text: str) -> Optional[str]:
    m = re.search(r'\b10\.\d{4,9}/[^\s"<>]+', text, flags=re.IGNORECASE)
    if m:
        doi = m.group(0).strip().rstrip(".;,)")
        doi = re.sub(r'[\]\)}>]+$', "", doi)
        return doi
    return None

def find_year(text: str) -> Optional[int]:
    snippet = text[:3000]
    hits = re.findall(r"\b(19[8-9]\d|20[0-2]\d)\b", snippet) or re.findall(r"\b(19[8-9]\d|20[0-2]\d)\b", text)
    if hits:
        try:
            return int(hits[0])
        except Exception:
            return None
    return None

def guess_title_and_authors(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    title, authors = None, None
    for i, line in enumerate(lines[:80]):
        l = line.strip()
        if not l:
            continue
        if title is None and len(l.split()) >= 4 and len(l) <= 200:
            title = l
            for j in range(i+1, min(i+10, len(lines))):
                la = lines[j].strip()
                if 0 < len(la) <= 200 and ("," in la or ";" in la or re.search(r"\b[A-Z]\.\s*[A-Z]\.", la)):
                    authors = la
                    break
            break
    return title, authors

def _header_regex(header: str) -> str:
    return rf'^\s*(?:\d+(?:\.\d+)*|\b[IVXLC]+\b)?\s*{re.escape(header)}\s*:?\s*$'

def split_sections(text: str) -> Dict[str, str]:
    norm = text.replace("\r", "\n")
    positions: List[Tuple[int, str]] = []
    for h in SECTION_HEADERS:
        for m in re.finditer(_header_regex(h), norm, flags=re.IGNORECASE | re.MULTILINE):
            positions.append((m.start(), h.lower()))
    positions.sort(key=lambda t: t[0])
    out: Dict[str, str] = {}
    for idx, (pos, h) in enumerate(positions):
        start = pos
        end = positions[idx+1][0] if idx+1 < len(positions) else len(norm)
        out[h] = norm[start:end].strip()
    return out

# ---------- Writing helpers ----------
def safe_to_csv(df: pd.DataFrame, out_path: Path, append: bool = False) -> None:
    """Write CSV robustly across pandas versions; always quote to avoid escape errors."""
    if append and out_path.exists():
        try:
            old = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        except Exception:
            old = pd.read_csv(out_path)
        # Align union of columns in correct order
        all_cols = list(dict.fromkeys(TARGET_COLUMNS + [c for c in old.columns if c not in TARGET_COLUMNS]))
        old = old.reindex(columns=all_cols)
        df = df.reindex(columns=all_cols)
        df = pd.concat([old, df], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(
            out_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            doublequote=True,
            escapechar='\\',
            lineterminator="\n",
        )
    except TypeError:
        # Some very old pandas: retry without lineterminator
        df.to_csv(
            out_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            doublequote=True,
            escapechar='\\',
        )

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Extract paper metadata/sections from PDFs into the exact AlloyMind clean schema.")
    ap.add_argument("--pdf_dir", default="AlloyMind", help="Folder containing PDFs (default: AlloyMind)")
    ap.add_argument("--out_excel", default=None, help="Excel output file (optional)")
    ap.add_argument("--out_csv", default="alloymind_dataset2.csv", help="CSV output (default: alloymind_dataset1.csv)")
    ap.add_argument("--default_alloy", default="", help="Default alloy name")
    ap.add_argument("--default_env", default="", help="Default environment (e.g., '3.5% NaCl')")
    ap.add_argument("--default_temp", type=float, default=None, help="Default temperature °C")
    ap.add_argument("--default_load", default="", help="Default load type (tension/fatigue/...)")
    ap.add_argument("--append", action="store_true", help="Append to existing output")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Not a directory: {pdf_dir}", file=sys.stderr)
        sys.exit(2)

    files = sorted(pdf_dir.glob("*.pdf"))
    if not files:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in files:
        try:
            text = extract_text(str(p)) or ""
        except Exception as e:
            print(f"[WARN] Could not parse {p.name}: {e}", file=sys.stderr)
            continue

        lines = [l for l in text.splitlines() if l.strip()]
        doi = find_doi(text) or ""
        year = find_year(text)
        title, authors = guess_title_and_authors(lines)
        sections = split_sections(text)

        abstract = sections.get("abstract","")
        introduction = sections.get("introduction","")
        methods = sections.get("methods","") or sections.get("materials and methods","") or sections.get("experimental","")
        results_text = sections.get("results","") or sections.get("results and discussion","")
        discussion = sections.get("discussion","")
        conclusion = sections.get("conclusion","") or sections.get("conclusions","")

        # Start with all fields blank so we always emit the full schema
        row = {col: "" for col in TARGET_COLUMNS}

        # Map the fields we know
        row.update({
            "id": doi or p.stem,
            "title": title or p.stem,
            "authors": authors or "",
            "year": str(year or ""),
            "venue": "",
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "keywords": "",
            "domain_tag": "materials; alloys",

            "abstract": abstract,
            "introduction": introduction,
            "methods": methods,
            "results_text": results_text,
            "discussion": discussion,
            "conclusion": conclusion,
            "fig_tbl_captions": "",

            # Set both the canonical and synonym alloy/temperature columns
            "Alloy": args.default_alloy,
            "alloy": args.default_alloy,
            "test_environment": args.default_env,
            "temperature_C": str(args.default_temp) if args.default_temp is not None else "",
            "Temperature, [C]": str(args.default_temp) if args.default_temp is not None else "",
            "Temperature [C]": str(args.default_temp) if args.default_temp is not None else "",
            "load_type": args.default_load,

            "property_name": "",
            "property_value": "",
            "units": "",
            "notes": f"Auto-extracted from {p.name}",

            "application_tags": "",
            "performance_category": "",
            "cost_tag": "",
            "availability_tag": "",
        })

        rows.append(row)

    # Build DataFrame in the exact column order
    df = pd.DataFrame(rows, columns=TARGET_COLUMNS)

    # Optional Excel
    if args.out_excel:
        out_x = Path(args.out_excel)
        out_x.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_x, index=False)
        print("Wrote", out_x)

    # CSV (robust)
    out_c = Path(args.out_csv)
    safe_to_csv(df, out_c, append=args.append)
    print("Wrote", out_c)

    # Preview if no outputs (not typical with defaults)
    if not args.out_csv and not args.out_excel:
        with pd.option_context("display.max_colwidth", 80, "display.width", 160):
            print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
