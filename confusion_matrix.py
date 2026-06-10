"""
confusion_matrix.py — evaluate the trained classifier and print a confusion
matrix + per-class metrics on the held-out test set (and optionally validation).

Reuses the project's own modules so the encoding / threshold match training.
Run from the repo root (where model.py, train.py, models/, splits/ live):

    python confusion_matrix.py              # test set (default)
    python confusion_matrix.py --split val
    python confusion_matrix.py --split both
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from model import StyleGuideClassifier
from train import SimpleTokeniser, ComplianceDataset, collate_fn, PAD_IDX

MODEL_DIR = Path("models")
SPLITS = {"test": Path("splits/test.jsonl"), "val": Path("splits/val.jsonl")}


def load_records(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def predict(model, records, tokeniser, max_len, device, threshold=0.5):
    ds = ComplianceDataset(records, tokeniser, max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    preds, labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device))
        preds.extend((torch.sigmoid(logits) >= threshold).long().cpu().tolist())
        labels.extend(batch["label"].long().cpu().tolist())
    return labels, preds


def report_split(name, labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(labels, preds)
    print("\n" + "=" * 58)
    print(f"{name.upper()} SET  —  {len(labels)} examples   (0 = compliant, 1 = violation)")
    print("=" * 58)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("                    pred compliant    pred violation")
    print(f"  actual compliant      {tn:>7}           {fp:>7}")
    print(f"  actual violation      {fn:>7}           {tp:>7}")
    print(f"\n  TN={tn}   FP={fp}   FN={fn}   TP={tp}")
    print(f"  accuracy = {acc:.4f}\n")
    print(classification_report(labels, preds,
          target_names=["compliant", "violation"], digits=4, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "both"], default="test")
    args = ap.parse_args()

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
    print(f"Loaded best_model.pt  (epoch {ckpt.get('epoch', '?')}, "
          f"checkpoint val macro F1 = {ckpt.get('val_macro_f1', float('nan')):.4f})")

    splits = ["test", "val"] if args.split == "both" else [args.split]
    for s in splits:
        labels, preds = predict(model, load_records(SPLITS[s]), tokeniser, cfg["max_len"], device)
        report_split(s, labels, preds)

        if s == "test":
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(4.2, 3.6))
                ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["compliant", "violation"])
                ax.set_yticklabels(["compliant", "violation"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                ax.set_title("Confusion matrix — test set")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
                fig.tight_layout(); fig.savefig("confusion_matrix_test.png", dpi=150)
                print("Saved figure -> confusion_matrix_test.png")
            except Exception as e:
                print(f"(figure skipped — matplotlib not available: {e})")


if __name__ == "__main__":
    main()