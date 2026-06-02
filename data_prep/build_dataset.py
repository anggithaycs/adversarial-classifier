"""
build_dataset.py 
=> To merge positive and negative examples into a labeled dataset
and split into train / val / test sets.
 
Usage:
    python build_dataset.py
 
Input:
    processed/sentence_pairs.jsonl   (label = 1, compliant)
    processed/violations.jsonl        (label = 0, violation)
 
Output:
    splits/train.jsonl
    splits/val.jsonl
    splits/test.jsonl
    splits/dataset_stats.txt
 
Design decisions (justified by Zhang et al., 2024):
    - No undersampling: keeps all 17k positive pairs; undersampling hurts
      performance on binary datasets (Zhang et al., Table 2).
    - Focal loss handled at training time (gamma=0.2 recommended as starting
      point per Zhang et al., Figure 2).
    - Class weights computed here and saved to stats for use in training.
    - Split: 80 / 10 / 10 (train / val / test), stratified by label so that
      the imbalance ratio is preserved across all three splits.
    - Inline translation-tool tags (e.g. [1], {2}) stripped before saving.
    - Each record gets a clean unified schema regardless of source file.
 
Label convention (detection framing):
    label = 1  →  violation (non-compliant)   ← minority / positive class
    label = 0  →  compliant                   ← majority / negative class
    
FN (missed violation) costs more than FP (false alarm) in production,
so we optimise recall on label=1 (violation) via focal loss + class weights at train time.

"""
 
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
 
# ── Config ────────────────────────────────────────────────────────────────────
 
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
TEST_RATIO    = 0.10
 
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
 
IN_POSITIVES  = Path("processed/sentence_pairs.jsonl")
IN_NEGATIVES  = Path("processed/violations.jsonl")
OUT_DIR       = Path("splits")
 
# Label convention
LABEL_COMPLIANT  = 0   # majority class
LABEL_VIOLATION  = 1   # minority class  ← what we want to detect
 
# ── Tag stripper ──────────────────────────────────────────────────────────────
 
TAG_RE = re.compile(r"[\[\{]\d+[\]\}]")
 
def strip_tags(text: str) -> str:
    """Remove translation-tool inline tags like [1], {2], {3}."""
    cleaned = TAG_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
 
 
# ── Schema normaliser ─────────────────────────────────────────────────────────
 
def normalise(record: dict, label: int) -> dict:
    """
    Return a clean, unified record regardless of which input file it came from.
    Fields kept:
        id, label, source_sv, translation, source_file,
        violation_id (None for positives), violation_desc (None for positives)
    """
    return {
        "id":             record.get("id", record.get("segment_id", "")),
        "label":          label,
        "source_sv":      strip_tags(record.get("source", record.get("source_sv", ""))),
        "translation":    strip_tags(record.get("translation", record.get("reference_en", ""))),
        "source_file":    record.get("source_file", record.get("doc_id", "")),
        "violation_id":   record.get("violation_id", None),
        "violation_desc": record.get("violation_desc", None),
    }
 
 
# ── Stratified split ──────────────────────────────────────────────────────────
 
def stratified_split(
    records: list[dict],
    train_r: float,
    val_r: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records into train/val/test preserving the label distribution.
    Works by splitting each class separately then combining.
    """
    rng = random.Random(seed)
 
    by_label: dict[int, list[dict]] = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec)
 
    train, val, test = [], [], []
 
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = math.floor(n * train_r)
        n_val   = math.floor(n * val_r)
 
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
 
    # Shuffle each split so labels are interleaved
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
 
    return train, val, test
 
 
# ── Stats ─────────────────────────────────────────────────────────────────────
 
def compute_class_weights(label_counts: Counter) -> dict[int, float]:
    """
    Inverse-frequency class weights, normalised so they sum to num_classes.
    Formula: w_c = N / (num_classes * N_c)
    This is the standard sklearn-style 'balanced' weighting.
    """
    total = sum(label_counts.values())
    n_classes = len(label_counts)
    return {
        label: total / (n_classes * count)
        for label, count in label_counts.items()
    }
 
 
def write_stats(
    all_records: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
    out_path: Path,
) -> None:
    all_counts   = Counter(r["label"] for r in all_records)
    train_counts = Counter(r["label"] for r in train)
    val_counts   = Counter(r["label"] for r in val)
    test_counts  = Counter(r["label"] for r in test)
 
    weights = compute_class_weights(all_counts)
    imbalance_ratio = all_counts[LABEL_COMPLIANT] / all_counts[LABEL_VIOLATION]
 
    # Violation breakdown
    violation_counts = Counter(
        r["violation_id"] for r in all_records if r["label"] == LABEL_VIOLATION
    )
 
    lines = [
        "=" * 60,
        "DATASET STATISTICS",
        "=" * 60,
        "",
        f"Total records       : {len(all_records)}",
        f"  Compliant  (0)    : {all_counts[0]}",
        f"  Violation  (1)    : {all_counts[1]}",
        f"  Imbalance ratio   : {imbalance_ratio:.2f}:1  (majority:minority)",
        "",
        "CLASS WEIGHTS  (use in focal loss / weighted CE)",
        f"  w[0] compliant    : {weights[0]:.4f}",
        f"  w[1] violation    : {weights[1]:.4f}",
        "",
        "FOCAL LOSS",
        "  Recommended starting gamma : 0.2",
        "  (Zhang et al. 2024, Figure 2 — best on binary datasets)",
        "",
        "SPLITS",
        f"  Train : {len(train):>6}  "
        f"(compliant={train_counts[0]}, violation={train_counts[1]})",
        f"  Val   : {len(val):>6}  "
        f"(compliant={val_counts[0]}, violation={val_counts[1]})",
        f"  Test  : {len(test):>6}  "
        f"(compliant={test_counts[0]}, violation={test_counts[1]})",
        "",
        "VIOLATION BREAKDOWN (across full dataset)",
    ]
 
    for vid, count in sorted(violation_counts.items()):
        lines.append(f"  {vid:<6} : {count}")
 
    lines += [
        "",
        "LABEL CONVENTION",
        "  label=1  violation (non-compliant)  ← minority / detection target",
        "  label=0  compliant                  ← majority",
        "",
        "NOTE: FN (missed violation) costs more than FP in production.",
        "Optimise recall on label=1 via focal loss + class weights.",
        "=" * 60,
    ]
 
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
 
    # Also print to console
    print("\n".join(lines))
 
 
# ── Inpuy output helpers ───────────────────────────────────────────────────────────────
 
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
 
 
def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading positives from {IN_POSITIVES} ...")
    raw_pos = load_jsonl(IN_POSITIVES)
    print(f"  → {len(raw_pos)} records")
 
    print(f"Loading negatives from {IN_NEGATIVES} ...")
    raw_neg = load_jsonl(IN_NEGATIVES)
    print(f"  → {len(raw_neg)} records")
 
    # ── Normalise ─────────────────────────────────────────────────────────────
    positives = [normalise(r, LABEL_COMPLIANT) for r in raw_pos]
    negatives = [normalise(r, LABEL_VIOLATION) for r in raw_neg]
 
    # Drop records where source or translation is empty after tag stripping
    def is_valid(rec: dict) -> bool:
        return bool(rec["source_sv"]) and bool(rec["translation"])
 
    positives = [r for r in positives if is_valid(r)]
    negatives = [r for r in negatives if is_valid(r)]
 
    print(f"\nAfter tag-stripping and validity filter:")
    print(f"  Compliant  : {len(positives)}")
    print(f"  Violation  : {len(negatives)}")
 
    all_records = positives + negatives
 
    # ── Stratified split ──────────────────────────────────────────────────────
    train, val, test = stratified_split(
        all_records,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=RANDOM_SEED,
    )
 
    # ── Write splits ──────────────────────────────────────────────────────────
    write_jsonl(train, OUT_DIR / "train.jsonl")
    write_jsonl(val,   OUT_DIR / "val.jsonl")
    write_jsonl(test,  OUT_DIR / "test.jsonl")
 
    print(f"\nWrote splits to {OUT_DIR}/")
    print(f"  train.jsonl : {len(train)}")
    print(f"  val.jsonl   : {len(val)}")
    print(f"  test.jsonl  : {len(test)}")
 
    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_path = OUT_DIR / "dataset_stats.txt"
    write_stats(all_records, train, val, test, stats_path)
    print(f"\nStats written to {stats_path}")
 
 
if __name__ == "__main__":
    main()