"""
parse_tmx.py  –  Extract (sv, en-gb) sentence pairs from Catena TMX files.

Usage:
    python scripts/parse_tmx.py

Output:
    data/catena_baseline/processed/sentence_pairs.jsonl
"""

import xml.etree.ElementTree as ET
import json
import re
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/catena_baseline/raw/Catena_package")
OUT_DIR = Path("data/catena_baseline/processed")
OUT_FILE = OUT_DIR / "sentence_pairs.jsonl"

TMX_FILES = [
    "Catena-master-sv-en-gb.tmx",
    "Catena-2281 Q1 2026-sv-en-gb-sv-en-gb.tmx",
]

MIN_CHAR_LEN = 30
MAX_CHAR_LEN = 1000
DEDUP = True

RE_NUMBERS  = re.compile(r'\b\d[\d\s,\.]*\b')
RE_CURRENCY = re.compile(r'(SEK|MSEK|KSEK|EUR|USD|%|Mkr|Mnkr)', re.IGNORECASE)
RE_DATE     = re.compile(r'\b(20\d{2}|Q[1-4]|jan|feb|mar|apr|maj|jun|'
                          r'jul|aug|sep|okt|nov|dec)\b', re.IGNORECASE)
RE_JUNK_TAG = re.compile(r'<[^>]{0,50}>')

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_text(tu_element, lang: str) -> str | None:
    for tuv in tu_element.findall("tuv"):
        tuv_lang = (
            tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
            or tuv.get("lang")
            or ""
        ).lower()
        if tuv_lang.startswith(lang.lower()):
            seg = tuv.find("seg")
            if seg is not None:
                raw = "".join(seg.itertext())
                return clean_text(raw)
    return None


def clean_text(text: str) -> str:
    text = RE_JUNK_TAG.sub("", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def flag_content(text: str) -> dict:
    return {
        "has_numbers":  bool(RE_NUMBERS.search(text)),
        "has_currency": bool(RE_CURRENCY.search(text)),
        "has_date":     bool(RE_DATE.search(text)),
    }


def is_valid(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    if len(src) < MIN_CHAR_LEN or len(tgt) < MIN_CHAR_LEN:
        return False
    if len(src) > MAX_CHAR_LEN or len(tgt) > MAX_CHAR_LEN:
        return False
    alpha_ratio = sum(c.isalpha() for c in src) / len(src)
    if alpha_ratio < 0.4:
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_tmx(tmx_path: Path, prefix: str, seen_sources: set) -> list[dict]:
    log.info(f"Parsing {tmx_path.name} ...")
    records = []

    tree = ET.parse(tmx_path)
    root = tree.getroot()
    body = root.find("body") or root

    for i, tu in enumerate(body.findall("tu")):
        src = get_text(tu, "sv")
        tgt = get_text(tu, "en")

        if not is_valid(src, tgt):
            continue
        if DEDUP and src in seen_sources:
            continue
        seen_sources.add(src)

        flags = flag_content(src)
        record = {
            "id": f"{prefix}_{i:05d}",
            "source_lang": "sv",
            "target_lang": "en-GB",
            "source": src,
            "translation": tgt,
            "source_file": tmx_path.name,
            "char_len_src": len(src),
            "char_len_tgt": len(tgt),
            **flags,
        }
        records.append(record)

    log.info(f"  → {len(records)} valid pairs extracted from {tmx_path.name}")
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_sources: set[str] = set()
    all_records: list[dict] = []

    for tmx_name in TMX_FILES:
        tmx_path = RAW_DIR / tmx_name
        if not tmx_path.exists():
            log.warning(f"File not found, skipping: {tmx_path}")
            continue
        prefix = "master" if "master" in tmx_name.lower() else "q1_2026"
        records = parse_tmx(tmx_path, prefix, seen_sources)
        all_records.extend(records)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(all_records)
    with_numbers  = sum(1 for r in all_records if r["has_numbers"])
    with_currency = sum(1 for r in all_records if r["has_currency"])
    with_date     = sum(1 for r in all_records if r["has_date"])

    log.info("─" * 50)
    log.info(f"Total pairs written : {total}")
    log.info(f"  Contains numbers  : {with_numbers} ({with_numbers/total:.1%})")
    log.info(f"  Contains currency : {with_currency} ({with_currency/total:.1%})")
    log.info(f"  Contains dates    : {with_date} ({with_date/total:.1%})")
    log.info(f"Output → {OUT_FILE}")


if __name__ == "__main__":
    main()
