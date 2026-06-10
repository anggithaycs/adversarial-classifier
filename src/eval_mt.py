"""
eval_mt.py — run the trained classifier over the machine-translation (MT)
segments and export the ones it flags as style-guide violations.

INFERENCE ONLY: no labels, no F1. Answers Fluid's question "does the model also
catch issues in MT output?" by listing the MT segments predicted as violations.

Reuses the project's own model/tokeniser/dataset (same as confusion_matrix.py).
Run from the repo ROOT, after parsing the MT RTFs into *_mt_segments.jsonl:

    python src/eval_mt.py
"""
import csv
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import StyleGuideClassifier
from train import SimpleTokeniser, ComplianceDataset, collate_fn, PAD_IDX

MODEL_DIR = Path("models")
MT_FILES = [
    ("Q1_2026", Path("data/mt_eval/q1_mt_segments.jsonl")),
    ("Q3_2025", Path("data/mt_eval/q3_mt_segments.jsonl")),
]
OUT_CSV   = Path("data/mt_eval/mt_predictions.csv")
MIN_LEN   = 30      # skip page numbers / fragments (lower this to include short rows)
THRESHOLD = 0.5

TAG_RE = re.compile(r"[\[\{]\d+[\]\}]")
def strip_tags(t):
    return re.sub(r"\s+", " ", TAG_RE.sub("", t or "")).strip()

def load_mt(path):
    """Read MT segments; map reference_en -> translation; strip tags; drop junk."""
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        src = strip_tags(r.get("source_sv", ""))
        tgt = strip_tags(r.get("reference_en", ""))   # column 3 = the MT translation
        if len(src) < MIN_LEN or len(tgt) < MIN_LEN:
            continue
        out.append({
            "segment_id": r.get("segment_id", ""),
            "source_sv": src,
            "translation": tgt,        # the field name ComplianceDataset expects
        })
    return out

@torch.no_grad()
def predict(model, records, tokeniser, max_len, device):
    ds = ComplianceDataset(records, tokeniser, max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    probs = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device))
        probs.extend(torch.sigmoid(logits).cpu().tolist())
    return probs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokeniser = SimpleTokeniser.load(MODEL_DIR / "tokeniser.json")
    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = StyleGuideClassifier(
        vocab_size=len(tokeniser), d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"], max_len=cfg["max_len"],
        dropout=0.0, pad_idx=PAD_IDX,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded best_model.pt (val macro F1 = {ckpt.get('val_macro_f1', float('nan')):.4f})\n")

    rows = []
    for doc, path in MT_FILES:
        if not path.exists():
            print(f"WARNING: missing {path} — run the parse step first.")
            continue
        recs = load_mt(path)
        probs = predict(model, recs, tokeniser, cfg["max_len"], device)
        n_flag = 0
        for r, p in zip(recs, probs):
            flag = int(p >= THRESHOLD)
            n_flag += flag
            rows.append({
                "doc": doc,
                "segment_id": r["segment_id"],
                "source_sv": r["source_sv"],
                "mt_translation": r["translation"],
                "prediction": "VIOLATION" if flag else "compliant",
                "violation_prob": round(p, 4),
            })
        rate = n_flag / max(len(recs), 1)
        print(f"{doc}: {len(recs)} segments scored — {n_flag} flagged as violation ({rate:.1%})")

    rows.sort(key=lambda x: (x["prediction"] != "VIOLATION", -x["violation_prob"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "segment_id", "source_sv",
                                          "mt_translation", "prediction", "violation_prob"])
        w.writeheader()
        w.writerows(rows)
    n_viol = sum(r["prediction"] == "VIOLATION" for r in rows)
    print(f"\nWrote {len(rows)} rows ({n_viol} flagged as violation) -> {OUT_CSV}")
    print("Open in Excel; flagged violations are at the top, sorted by confidence.")

if __name__ == "__main__":
    main()