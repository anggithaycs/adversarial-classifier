"""
parse_report_rtf.py – Extract bilingual Swedish-English report segments from Catena RTF exports.

Usage:
    python scripts/parse_report_rtf.py

Output:
    data/catena_baseline/processed/q3_2025_segments.jsonl
    data/catena_baseline/processed/q1_2026_segments.jsonl
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/catena_baseline/raw/Catena_package")
OUT_DIR = Path("data/catena_baseline/processed")

REPORTS = [
    {
        "doc_id": "catena_q3_2025",
        "input_file": "SWE-ENG_Catena Q3 2025.rtf",
        "output_file": "q3_2025_segments.jsonl",
    },
    {
        "doc_id": "catena_q1_2026",
        "input_file": "SWE-ENG_Catena Q1 2026.rtf",
        "output_file": "q1_2026_segments.jsonl",
    },
]

CPG_RE = re.compile(r"\\ansicpg(\d+)")
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


def detect_encoding(raw: str) -> str:
    """
    Detect RTF ANSI codepage.
    Q1 uses cp1252; Q3 may use cp1250.
    """
    m = CPG_RE.search(raw[:3000])
    if not m:
        return "cp1252"

    enc = f"cp{m.group(1)}"
    try:
        b"test".decode(enc)
        return enc
    except LookupError:
        return "cp1252"


def decode_signed_unicode(n: int) -> str:
    """
    RTF Unicode values may be signed 16-bit integers.
    """
    if n < 0:
        n += 65536
    try:
        return chr(n)
    except ValueError:
        return ""


def rtf_fragment_to_text(s: str, encoding: str) -> str:
    """
    Minimal RTF-to-text converter for these translation-tool exports.
    Handles:
    - hex escapes: \\'e4
    - unicode escapes: \\u228
    - escaped braces: \\{ and \\}
    - line breaks
    - common dash controls
    """
    out = []
    i = 0
    n = len(s)
    uc_skip = 1

    while i < n:
        ch = s[i]

        if ch == "\\":
            i += 1
            if i >= n:
                break

            nxt = s[i]

            # Escaped literal characters: \\ \\{ \\}
            if nxt in "\\{}":
                out.append(nxt)
                i += 1
                continue

            # Hex-encoded byte, e.g. \\'e4
            if nxt == "'" and i + 2 < n:
                hx = s[i + 1 : i + 3]
                try:
                    out.append(bytes.fromhex(hx).decode(encoding, errors="replace"))
                except Exception:
                    pass
                i += 3
                continue

            # Control word
            if nxt.isalpha():
                start = i
                while i < n and s[i].isalpha():
                    i += 1
                word = s[start:i]

                sign = 1
                if i < n and s[i] in "+-":
                    if s[i] == "-":
                        sign = -1
                    i += 1

                num_start = i
                while i < n and s[i].isdigit():
                    i += 1

                num = None
                if i > num_start:
                    num = sign * int(s[num_start:i])

                # A space after a control word is a delimiter.
                if i < n and s[i] == " ":
                    i += 1

                if word == "uc" and num is not None:
                    uc_skip = max(num, 0)

                elif word == "u" and num is not None:
                    out.append(decode_signed_unicode(num))
                    for _ in range(uc_skip):
                        if i < n:
                            i += 1

                elif word in {"par", "line"}:
                    out.append("\n")

                elif word == "tab":
                    out.append("\t")

                elif word == "endash":
                    out.append("–")

                elif word == "emdash":
                    out.append("—")

                # Ignore formatting/control words.
                continue

            # Control symbols
            if nxt == "~":
                out.append(" ")
            elif nxt in {"-", "_"}:
                out.append("-")

            i += 1
            continue

        if ch in "{}":
            i += 1
            continue

        out.append(ch)
        i += 1

    text = "".join(out)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_segments(raw: str, doc_id: str) -> list[dict]:
    encoding = detect_encoding(raw)
    records = []

    for row in raw.split(r"\row"):
        # Important: split only on real \cell, not on \cellx table-width controls.
        parts = re.split(r"\\cell(?![a-zA-Z])", row)

        if len(parts) < 4:
            continue

        cells = [
            clean_cell(rtf_fragment_to_text(part, encoding))
            for part in parts[:-1]
        ]
        cells = [c for c in cells if c]

        if len(cells) < 3:
            continue

        first_cell = cells[0]
        uuid_match = UUID_RE.search(first_cell)

        if not uuid_match:
            continue

        uuid = uuid_match.group(0)

        before_uuid = first_cell[: uuid_match.start()]
        id_match = re.search(r"\b(\d{1,6})\b", before_uuid)

        if not id_match:
            continue

        segment_id = id_match.group(1)

        source_sv = cells[1]
        reference_en = cells[2]

        # Skip obvious placeholder rows.
        if source_sv in {"", "[1]"} and reference_en in {"", "[1]"}:
            continue

        record = {
            "doc_id": doc_id,
            "segment_id": segment_id,
            "uuid": uuid,
            "source_sv": source_sv,
            "reference_en": reference_en,
            "source_len": len(source_sv),
            "reference_len": len(reference_en),
            "has_tags": bool(re.search(r"[\[\{]\d+[\]\}]", source_sv + reference_en)),
            "has_numbers": bool(re.search(r"\d", source_sv + reference_en)),
        }

        records.append(record)

    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for report in REPORTS:
        input_path = RAW_DIR / report["input_file"]
        output_path = OUT_DIR / report["output_file"]

        if not input_path.exists():
            print(f"WARNING: missing file: {input_path}")
            continue

        raw = input_path.read_bytes().decode("latin-1", errors="ignore")
        records = extract_segments(raw, report["doc_id"])
        write_jsonl(records, output_path)

        print(f"{report['doc_id']}: {len(records)} segments → {output_path}")

        if records:
            print("Preview:")
            print(json.dumps(records[0], ensure_ascii=False, indent=2))
            print()


if __name__ == "__main__":
    main()