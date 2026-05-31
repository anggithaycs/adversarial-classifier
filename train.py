
"""
train.py 
=> Train the StyleGuideClassifier on the Catena compliance dataset.
 
Usage:
    python train.py
 
    # Override defaults:
    python train.py --epochs 20 --lr 3e-4 --gamma 0.5
 
Input:
    splits/train.jsonl
    splits/val.jsonl
 
Output:
    models/best_model.pt          ← best checkpoint (by val F1)
    models/tokeniser.json         ← vocabulary
    models/training_log.jsonl     ← per-epoch metrics
 
Design:
    - Simple whitespace + subword-free tokeniser built from training data.
      Good enough for a from-scratch model; keeps the pipeline self-contained.
    - Focal loss (Lin et al. 2020) with gamma=0.2 (Zhang et al. 2024 default).
    - Class weights from dataset_stats: w[0]=0.5911, w[1]=3.2453.
    - AdamW optimiser with linear warmup + cosine decay.
    - Early stopping on validation macro F1 (patience=5).
    - Macro F1 as primary metric (Zhang et al. 2024; standard for imbalanced).
"""
 
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
 
from model import StyleGuideClassifier
 
# ── Paths ─────────────────────────────────────────────────────────────────────
 
TRAIN_FILE  = Path("splits/train.jsonl")
VAL_FILE    = Path("splits/val.jsonl")
MODEL_DIR   = Path("models")
 
# ── Defaults ──────────────────────────────────────────────────────────────────
 
DEFAULTS = dict(
    seed        = 42,
    max_len     = 256,      # tokens per example (source + sep + translation)
    vocab_size  = 8000,     # built from training data
    d_model     = 256,
    n_heads     = 4,
    n_layers    = 4,
    d_ff        = 1024,
    dropout     = 0.1,
    batch_size  = 32,
    epochs      = 30,
    lr          = 3e-4,
    warmup_steps= 200,
    gamma       = 0.2,      # focal loss; Zhang et al. 2024 best default
    # class weights from dataset_stats.txt
    w_compliant = 0.5911,
    w_violation = 3.2453,
    patience    = 5,        # early stopping on val macro F1
)
 
# ── Special tokens ────────────────────────────────────────────────────────────
 
PAD   = "<PAD>"
UNK   = "<UNK>"
CLS   = "<CLS>"
SEP   = "<SEP>"
SPECIAL_TOKENS = [PAD, UNK, CLS, SEP]
PAD_IDX = 0
 
 
# ── Tokeniser ─────────────────────────────────────────────────────────────────
 
class SimpleTokeniser:
    """
    Whitespace tokeniser with a fixed vocabulary built from training data.
    Lowercases everything; OOV tokens map to <UNK>.
    """
 
    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
 
    def build(self, texts: list[str], vocab_size: int) -> None:
        counts: Counter = Counter()
        for text in texts:
            for tok in self._tokenize(text):
                counts[tok] += 1
 
        vocab = SPECIAL_TOKENS + [
            tok for tok, _ in counts.most_common(vocab_size - len(SPECIAL_TOKENS))
        ]
        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
 
    def _tokenize(self, text: str) -> list[str]:
        # Lowercase + split on whitespace and punctuation boundaries
        text  = text.lower()
        tokens = re.findall(r"[a-zA-ZåäöÅÄÖ]+|[0-9]+|[^\w\s]", text)
        return tokens
 
    def encode(self, text: str) -> list[int]:
        return [
            self.token2id.get(tok, self.token2id[UNK])
            for tok in self._tokenize(text)
        ]
 
    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
 
    @classmethod
    def load(cls, path: Path) -> "SimpleTokeniser":
        tok = cls()
        with path.open(encoding="utf-8") as f:
            tok.token2id = json.load(f)
        tok.id2token = {i: t for t, i in tok.token2id.items()}
        return tok
 
    def __len__(self) -> int:
        return len(self.token2id)
 
 
# ── Dataset ───────────────────────────────────────────────────────────────────
 
class ComplianceDataset(Dataset):
    """
    Returns tokenised tensors in the format:
        [CLS] source_sv_tokens [SEP] translation_tokens [SEP] [PAD...]
    """
 
    def __init__(
        self,
        records   : list[dict],
        tokeniser : SimpleTokeniser,
        max_len   : int,
    ):
        self.records   = records
        self.tokeniser = tokeniser
        self.max_len   = max_len
 
        cls_id = tokeniser.token2id[CLS]
        sep_id = tokeniser.token2id[SEP]
 
        self.cls_id = cls_id
        self.sep_id = sep_id
 
    def __len__(self) -> int:
        return len(self.records)
 
    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
 
        src_ids  = self.tokeniser.encode(rec["source_sv"])
        tgt_ids  = self.tokeniser.encode(rec["translation"])
 
        # [CLS] src [SEP] tgt [SEP]
        # Truncate src and tgt proportionally if needed
        budget = self.max_len - 3   # 3 special tokens
        src_budget = budget // 2
        tgt_budget = budget - src_budget
 
        src_ids = src_ids[:src_budget]
        tgt_ids = tgt_ids[:tgt_budget]
 
        ids = [self.cls_id] + src_ids + [self.sep_id] + tgt_ids + [self.sep_id]
 
        # Pad to max_len
        pad_len = self.max_len - len(ids)
        ids = ids + [PAD_IDX] * pad_len
 
        return {
            "input_ids" : torch.tensor(ids, dtype=torch.long),
            "label"     : torch.tensor(rec["label"], dtype=torch.float),
        }
 
 
def collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids" : torch.stack([b["input_ids"] for b in batch]),
        "label"     : torch.stack([b["label"]     for b in batch]),
    }
 
 
# ── Focal Loss ────────────────────────────────────────────────────────────────
 
class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al. 2020).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 
    alpha: per-class weight tensor [w_compliant, w_violation]
    gamma: focusing parameter (0 = standard BCE; 0.2 recommended)
    """
 
    def __init__(self, alpha: torch.Tensor, gamma: float = 0.2):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,)  targets: (B,) in {0, 1}
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t is the probability of the true class
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
 
        # Per-sample alpha weight
        alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
 
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()
 
 
# ── LR Schedule ───────────────────────────────────────────────────────────────
 
def get_scheduler(optimiser, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
 
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
 
 
# ── Evaluation ────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    all_preds, all_labels = [], []
 
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(ids)
        preds  = (torch.sigmoid(logits) >= threshold).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.long().cpu().tolist())
 
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=["compliant", "violation"],
        zero_division=0,
    )
    return {"macro_f1": macro_f1, "report": report}
 
 
# ── Training loop ─────────────────────────────────────────────────────────────
 
def train(cfg: dict) -> None:
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
    # ── Load data ─────────────────────────────────────────────────────────────
    def load_jsonl(path):
        with path.open(encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
 
    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")
 
    # ── Build tokeniser ───────────────────────────────────────────────────────
    all_texts = [
        r["source_sv"] + " " + r["translation"]
        for r in train_records
    ]
    tokeniser = SimpleTokeniser()
    tokeniser.build(all_texts, vocab_size=cfg["vocab_size"])
    tokeniser.save(MODEL_DIR / "tokeniser.json")
    actual_vocab = len(tokeniser)
    print(f"Vocabulary size: {actual_vocab}")
 
    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = ComplianceDataset(train_records, tokeniser, cfg["max_len"])
    val_ds   = ComplianceDataset(val_records,   tokeniser, cfg["max_len"])
 
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2,
        shuffle=False, collate_fn=collate_fn, num_workers=2,
    )
 
    # ── Model ─────────────────────────────────────────────────────────────────
    model = StyleGuideClassifier(
        vocab_size = actual_vocab,
        d_model    = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        d_ff       = cfg["d_ff"],
        max_len    = cfg["max_len"],
        dropout    = cfg["dropout"],
        pad_idx    = PAD_IDX,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
 
    # ── Loss ──────────────────────────────────────────────────────────────────
    alpha = torch.tensor(
        [cfg["w_compliant"], cfg["w_violation"]], dtype=torch.float, device=device
    )
    criterion = FocalLoss(alpha=alpha, gamma=cfg["gamma"])
 
    # ── Optimiser + schedule ──────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=0.01
    )
    total_steps   = len(train_loader) * cfg["epochs"]
    scheduler     = get_scheduler(optimiser, cfg["warmup_steps"], total_steps)
 
    # ── Training ──────────────────────────────────────────────────────────────
    best_f1      = 0.0
    patience_cnt = 0
    log_path     = MODEL_DIR / "training_log.jsonl"
 
    print("\nStarting training...")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val F1':>8}  {'LR':>10}")
    print("-" * 45)
 
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
 
        for batch in train_loader:
            ids    = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
 
            optimiser.zero_grad()
            logits = model(ids)
            loss   = criterion(logits, labels)
            loss.backward()
 
            # Gradient clipping — important for small Transformers
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            optimiser.step()
            scheduler.step()
            total_loss += loss.item()
 
        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        val_f1      = val_metrics["macro_f1"]
        current_lr  = scheduler.get_last_lr()[0]
 
        print(f"{epoch:>5}  {avg_loss:>10.4f}  {val_f1:>8.4f}  {current_lr:>10.2e}")
 
        # Log
        log_entry = {
            "epoch"    : epoch,
            "train_loss": round(avg_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr"       : round(current_lr, 8),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
 
        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": val_f1,
                "cfg"        : cfg,
            }, MODEL_DIR / "best_model.pt")
            patience_cnt = 0
            print(f"         ↑ New best F1={best_f1:.4f} — checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                break
 
    # ── Final evaluation on val ───────────────────────────────────────────────
    print(f"\nBest val macro F1: {best_f1:.4f}")
    print("\nLoading best checkpoint for final val report...")
 
    #ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=device)
    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    final = evaluate(model, val_loader, device)
    print(final["report"])
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    for key, val in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args = vars(parser.parse_args())
    return args
 
 
if __name__ == "__main__":
    cfg = parse_args()
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k:<15} = {v}")
    print()
    train(cfg)