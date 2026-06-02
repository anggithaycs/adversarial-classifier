"""
analyze.py – Gradient-based saliency analysis for the StyleGuideClassifier.
 
What this does
--------------
1. Loads the best model checkpoint + tokeniser.
2. For each example, computes the gradient of the predicted violation logit
   with respect to the input token embeddings:
       saliency[t] = || d(logit) / d(embedding[t]) ||_2
3. Saves per-example saliency scores to:
       results/saliency/saliency_scores.jsonl
4. Prints a colour-coded terminal heatmap for quick inspection.
5. Saves an HTML heatmap file for each analysed example to:
       results/saliency/heatmaps/
 
Usage
-----
    # Analyse a random sample of 50 examples from the test set
    python analyze.py
 
    # Analyse more examples and filter by violation type
    python analyze.py --n_samples 100 --violation_id V01
 
    # Analyse both compliant and violation examples
    python analyze.py --labels both
 
Why gradient saliency?
----------------------
For a token t with embedding e_t, the saliency score is:
    s_t = || d(logit) / d(e_t) ||_2
 
High s_t means the model's prediction is sensitive to that token.
We expect violation tokens (e.g. "per cent", "Mkr", "direct yield")
to have high saliency on non-compliant examples.
Tokens with high saliency that are NOT violation tokens are the
"vulnerable" tokens targeted by the adversarial attack (attack.py).
"""
 
import argparse
import json
import math
import random
from pathlib import Path
 
import torch
import torch.nn as nn
 
from model import StyleGuideClassifier
from train import SimpleTokeniser, ComplianceDataset, PAD_IDX
 
# ── Paths ─────────────────────────────────────────────────────────────────────
 
TEST_FILE    = Path("splits/test.jsonl")
MODEL_DIR    = Path("models")
RESULTS_DIR  = Path("results/saliency")
HEATMAP_DIR  = RESULTS_DIR / "heatmaps"
 
# ── Colour palette for terminal output ───────────────────────────────────────
# ANSI background colours mapped to saliency intensity buckets
 
ANSI_RESET  = "\033[0m"
ANSI_COLORS = [
    "\033[48;5;255m\033[38;5;0m",   # white bg   — very low
    "\033[48;5;226m\033[38;5;0m",   # yellow     — low
    "\033[48;5;214m\033[38;5;0m",   # orange     — medium
    "\033[48;5;196m\033[38;5;255m", # red        — high
    "\033[48;5;88m\033[38;5;255m",  # dark red   — very high
]
 
 
# ── Saliency computation ──────────────────────────────────────────────────────
 
def compute_saliency(
    model     : StyleGuideClassifier,
    input_ids : torch.Tensor,   # (1, T)
    device    : torch.device,
) -> tuple[float, list[float]]:
    """
    Returns:
        logit     : raw model logit (float)
        saliency  : list of per-token L2 gradient norms, length T
    """
    model.eval()
    input_ids = input_ids.to(device)
 
    # Get embeddings with gradient tracking
    B, T = input_ids.shape
    positions = torch.arange(T, device=device).unsqueeze(0)
 
    token_emb = model.token_emb(input_ids)          # (1, T, d_model)
    pos_emb   = model.pos_emb(positions)             # (1, T, d_model)
 
    # Combine and mark as requiring grad
    embeddings = (token_emb + pos_emb).detach().requires_grad_(True)
    x = model.emb_norm(embeddings)
    x = model.emb_drop(x)
 
    # Padding mask
    key_padding_mask = (input_ids == PAD_IDX)
 
    # Forward through encoder layers
    for layer in model.layers:
        x = layer(x, key_padding_mask)
    x = model.final_norm(x)
 
    # CLS pooling + classification head
    cls_hidden = x[:, 0, :]
    logit = model.classifier(cls_hidden).squeeze(-1)  # scalar
 
    # Backward pass to get d(logit)/d(embedding)
    logit.backward()
 
    # Gradient: (1, T, d_model) → L2 norm per token → (T,)
    grad = embeddings.grad                            # (1, T, d_model)
    saliency = grad[0].norm(dim=-1).tolist()          # (T,)
 
    return logit.item(), saliency
 
 
# ── Token colour ──────────────────────────────────────────────────────────────
 
def saliency_to_colour(score: float, max_score: float) -> str:
    if max_score == 0:
        return ANSI_COLORS[0]
    ratio = score / max_score
    idx = min(int(ratio * len(ANSI_COLORS)), len(ANSI_COLORS) - 1)
    return ANSI_COLORS[idx]
 
 
# ── HTML heatmap ──────────────────────────────────────────────────────────────
 
def make_html_heatmap(
    tokens   : list[str],
    saliency : list[float],
    logit    : float,
    label    : int,
    example_id: str,
    violation_id: str | None,
) -> str:
    max_s  = max(saliency) if saliency else 1.0
    pred   = "violation" if torch.sigmoid(torch.tensor(logit)).item() >= 0.5 else "compliant"
    true_l = "violation" if label == 1 else "compliant"
 
    spans = []
    for tok, s in zip(tokens, saliency):
        ratio    = s / max_s if max_s > 0 else 0
        # Red channel scales with saliency; background goes white→red
        r = int(255)
        g = int(255 * (1 - ratio))
        b = int(255 * (1 - ratio))
        bg = f"rgb({r},{g},{b})"
        opacity = 0.2 + 0.8 * ratio
        style = (
            f"background-color:{bg};"
            f"opacity:{opacity:.2f};"
            f"padding:2px 4px;"
            f"margin:1px;"
            f"border-radius:3px;"
            f"display:inline-block;"
            f"font-family:monospace;"
            f"font-size:14px;"
        )
        score_tip = f"saliency={s:.4f}"
        spans.append(f'<span style="{style}" title="{score_tip}">{tok}</span>')
 
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Saliency: {example_id}</title></head>
<body style="font-family:sans-serif;padding:20px;max-width:900px;">
  <h2>Gradient Saliency Heatmap</h2>
  <table style="border-collapse:collapse;margin-bottom:20px;">
    <tr><td><b>ID</b></td><td>{example_id}</td></tr>
    <tr><td><b>True label</b></td><td>{true_l}</td></tr>
    <tr><td><b>Predicted</b></td><td>{pred}</td></tr>
    <tr><td><b>Logit</b></td><td>{logit:.4f}</td></tr>
    <tr><td><b>Violation rule</b></td><td>{violation_id or 'N/A (compliant)'}</td></tr>
  </table>
  <div style="line-height:2.2;word-wrap:break-word;">
    {"".join(spans)}
  </div>
  <hr>
  <p style="color:#888;font-size:12px;">
    Colour intensity = || &part;logit / &part;embedding ||&#8322; (L2 gradient norm).
    Darker red = higher saliency = more influential token.
  </p>
</body>
</html>"""
    return html
 
 
# ── Terminal print ────────────────────────────────────────────────────────────
 
def print_terminal_heatmap(
    tokens   : list[str],
    saliency : list[float],
    logit    : float,
    label    : int,
    violation_id: str | None,
    top_k    : int = 10,
) -> None:
    prob  = torch.sigmoid(torch.tensor(logit)).item()
    pred  = "VIOLATION" if prob >= 0.5 else "compliant"
    true_l = "VIOLATION" if label == 1 else "compliant"
    max_s  = max(saliency) if saliency else 1.0
 
    print(f"\n  True: {true_l}  |  Pred: {pred}  |  P(violation)={prob:.3f}"
          f"  |  Rule: {violation_id or 'N/A'}")
    print("  " + "─" * 70)
 
    # Print coloured tokens
    line = "  "
    for tok, s in zip(tokens, saliency):
        colour = saliency_to_colour(s, max_s)
        line  += f"{colour} {tok} {ANSI_RESET}"
        if len(line) > 120:
            print(line)
            line = "  "
    if line.strip():
        print(line)
 
    # Top-k most salient tokens
    print(f"\n  Top-{top_k} salient tokens:")
    ranked = sorted(zip(tokens, saliency), key=lambda x: x[1], reverse=True)
    for tok, s in ranked[:top_k]:
        bar = "█" * int(s / max_s * 20)
        print(f"    {tok:<20} {s:.4f}  {bar}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main(cfg: dict) -> None:
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
 
    # ── Load tokeniser + model ────────────────────────────────────────────────
    tokeniser = SimpleTokeniser.load(MODEL_DIR / "tokeniser.json")
    print(f"Vocabulary size: {len(tokeniser)}")
 
    ckpt = torch.load(
        MODEL_DIR / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    saved_cfg = ckpt["cfg"]
 
    model = StyleGuideClassifier(
        vocab_size = len(tokeniser),
        d_model    = saved_cfg["d_model"],
        n_heads    = saved_cfg["n_heads"],
        n_layers   = saved_cfg["n_layers"],
        d_ff       = saved_cfg["d_ff"],
        max_len    = saved_cfg["max_len"],
        dropout    = 0.0,        # disable dropout at inference
        pad_idx    = PAD_IDX,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val F1={ckpt['val_macro_f1']:.4f})")
 
    # ── Load test records ─────────────────────────────────────────────────────
    with TEST_FILE.open(encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]
 
    # Filter by label and violation_id if requested
    if cfg["labels"] == "violation":
        records = [r for r in all_records if r["label"] == 1]
    elif cfg["labels"] == "compliant":
        records = [r for r in all_records if r["label"] == 0]
    else:
        records = all_records
 
    if cfg["violation_id"]:
        records = [r for r in records if r.get("violation_id") == cfg["violation_id"]]
 
    if not records:
        print("No records match the filter. Try --labels both or remove --violation_id.")
        return
 
    # Sample
    n = min(cfg["n_samples"], len(records))
    sample = random.sample(records, n)
    print(f"\nAnalysing {n} examples (filtered from {len(all_records)} test records)\n")
 
    # ── Special token ids ─────────────────────────────────────────────────────
    cls_id = tokeniser.token2id["<CLS>"]
    sep_id = tokeniser.token2id["<SEP>"]
    max_len = saved_cfg["max_len"]
 
    def encode_example(rec: dict) -> tuple[list[int], list[str]]:
        src_ids  = tokeniser.encode(rec["source_sv"])
        tgt_ids  = tokeniser.encode(rec["translation"])
        budget   = max_len - 3
        src_ids  = src_ids[:budget // 2]
        tgt_ids  = tgt_ids[:budget - budget // 2]
        ids      = [cls_id] + src_ids + [sep_id] + tgt_ids + [sep_id]
        pad_len  = max_len - len(ids)
        ids      = ids + [PAD_IDX] * pad_len
 
        # Reconstruct token strings for display
        id2tok = tokeniser.id2token
        tokens = [id2tok.get(i, "<UNK>") for i in ids]
        return ids, tokens
 
    # ── Analyse ───────────────────────────────────────────────────────────────
    all_results = []
 
    for i, rec in enumerate(sample):
        ids, tokens = encode_example(rec)
        input_ids = torch.tensor([ids], dtype=torch.long)
 
        logit, saliency = compute_saliency(model, input_ids, device)
 
        # Strip PAD tokens from display
        real_len = sum(1 for t in tokens if t != "<PAD>")
        disp_tokens   = tokens[:real_len]
        disp_saliency = saliency[:real_len]
 
        print(f"\n[{i+1}/{n}] id={rec.get('id', '?')}")
        print_terminal_heatmap(
            disp_tokens, disp_saliency, logit,
            rec["label"], rec.get("violation_id"),
            top_k=cfg["top_k"],
        )
 
        # Save HTML heatmap
        html = make_html_heatmap(
            disp_tokens, disp_saliency, logit,
            rec["label"],
            example_id=rec.get("id", str(i)),
            violation_id=rec.get("violation_id"),
        )
        html_path = HEATMAP_DIR / f"{rec.get('id', i)}.html"
        html_path.write_text(html, encoding="utf-8")
 
        # Save saliency record
        result = {
            "id"           : rec.get("id", str(i)),
            "label"        : rec["label"],
            "violation_id" : rec.get("violation_id"),
            "logit"        : round(logit, 4),
            "p_violation"  : round(torch.sigmoid(torch.tensor(logit)).item(), 4),
            "predicted"    : int(torch.sigmoid(torch.tensor(logit)).item() >= 0.5),
            "tokens"       : disp_tokens,
            "saliency"     : [round(s, 4) for s in disp_saliency],
            "top_tokens"   : sorted(
                zip(disp_tokens, [round(s,4) for s in disp_saliency]),
                key=lambda x: x[1], reverse=True
            )[:cfg["top_k"]],
        }
        all_results.append(result)
 
    # ── Save all saliency scores ──────────────────────────────────────────────
    out_path = RESULTS_DIR / "saliency_scores.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
 
    print(f"\n\nSaliency scores saved → {out_path}")
    print(f"HTML heatmaps saved  → {HEATMAP_DIR}/")
 
    # ── Summary: which tokens appear most in top-k across examples ────────────
    print("\n" + "=" * 60)
    print("AGGREGATE: most frequently high-saliency tokens")
    print("(appears in top-k for the most examples)")
    print("=" * 60)
 
    from collections import Counter
    token_freq: Counter = Counter()
    for r in all_results:
        for tok, _ in r["top_tokens"]:
            if tok not in {"<CLS>", "<SEP>", "<PAD>", "<UNK>"}:
                token_freq[tok] += 1
 
    for tok, cnt in token_freq.most_common(20):
        bar = "█" * cnt
        print(f"  {tok:<25} {cnt:>4}  {bar}")
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",    type=int, default=50,
                        help="Number of test examples to analyse")
    parser.add_argument("--top_k",        type=int, default=10,
                        help="Top-k salient tokens to show per example")
    parser.add_argument("--labels",       type=str, default="violation",
                        choices=["violation", "compliant", "both"],
                        help="Which examples to analyse")
    parser.add_argument("--violation_id", type=str, default=None,
                        help="Filter to a specific violation rule e.g. V01")
    return vars(parser.parse_args())
 
 
if __name__ == "__main__":
    cfg = parse_args()
    print("Config:", cfg)
    main(cfg)