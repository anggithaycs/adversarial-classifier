"""
parse_termbase.py – Extract (Swedish → English) term pairs from Catena CSV termbases.

Usage:
    python scripts/parse_termbase.py

Output:
    data/catena_baseline/processed/termbase.jsonl
    data/catena_baseline/processed/termbase_summary.txt
"""

import csv
import json
import logging
from pathlib import Path

RAW_DIR = Path("data/catena_baseline/raw/Catena_package")
OUT_DIR  = Path("data/catena_baseline/processed")
OUT_FILE = OUT_DIR / "termbase.jsonl"

# (filename, en_col_indices, sv_col_indices)  — 0-based
CSV_CONFIGS = [
    ("Catena termbase.csv",        "catena",        [11,14,17,20], [24,27,30,33]),
    ("Fluid Finance termbase.csv", "fluid_finance",  [15,18,21],    [42,45,48]),
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SKIP_VALS = {"CasePermissive", "CaseInsense", "HalfPrefix"}


def get_terms(row: list[str], indices: list[int]) -> list[str]:
    terms = []
    for i in indices:
        if i >= len(row):
            continue
        val = row[i].strip()
        if val and val not in SKIP_VALS and not any(s in val for s in SKIP_VALS):
            terms.append(val)
    return terms


def parse_csv(filename: str, source: str, en_cols: list, sv_cols: list) -> list[dict]:
    path = RAW_DIR / filename
    if not path.exists():
        log.warning(f"Not found, skipping: {path}")
        return []

    records = []
    seen = set()

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if not row:
                continue

            en_terms = get_terms(row, en_cols)
            sv_terms = get_terms(row, sv_cols)

            if not en_terms or not sv_terms:
                continue

            sv = sv_terms[0]
            en = en_terms[0]
            key = (sv.lower(), en.lower())
            if key in seen:
                continue
            seen.add(key)

            records.append({
                "sv":          sv,
                "en":          en,
                "sv_variants": sv_terms[1:],
                "en_variants": en_terms[1:],
                "source":      source,
            })

    log.info(f"  → {len(records)} term pairs from {filename}")
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    for filename, source, en_cols, sv_cols in CSV_CONFIGS:
        log.info(f"Parsing {filename} ...")
        all_records.extend(parse_csv(filename, source, en_cols, sv_cols))

    # Deduplicate across files
    seen = set()
    deduped = []
    for r in all_records:
        key = (r["sv"].lower(), r["en"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = OUT_DIR / "termbase_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total term pairs: {len(deduped)}\n\n")
        f.write(f"{'Swedish':<45} {'English':<45} {'Source'}\n")
        f.write("-" * 100 + "\n")
        for r in deduped[:80]:
            f.write(f"{r['sv']:<45} {r['en']:<45} {r['source']}\n")

    log.info(f"Total term pairs : {len(deduped)}")
    log.info(f"Output → {OUT_FILE}")
    log.info(f"Preview → {summary_path}")


if __name__ == "__main__":
    main()