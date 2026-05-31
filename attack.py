"""
attack.py
=> Adversarial attack on the StyleGuideClassifier.

Strategy
--------
For each violation example the model correctly predicts:

1. Compute gradient saliency for every token.
2. Identify "vulnerable" tokens: high saliency BUT not the actual
   violation token (i.e. not in the known bad-form vocabulary).
3. Replace vulnerable tokens one at a time with synonyms from a
   hand-crafted financial/neutral synonym dictionary.
4. Re-run inference. If the prediction flips (violation → compliant)
   the attack succeeded — without fixing the actual violation.

This directly operationalises the assignment brief:
   "find vulnerable words, i.e. words with a large gradient but which
    intuitively should not be important for the prediction."

Output
------
    results/attack/attack_results.jsonl   ← per-example outcomes
    results/attack/attack_summary.txt     ← aggregate success rate

Usage
-----
    python attack.py
    python attack.py --n_samples 100 --top_vulnerable 5
    python attack.py --violation_id V01
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from model import StyleGuideClassifier
from train import SimpleTokeniser, PAD_IDX
from analyze import compute_saliency

# ── Paths ─────────────────────────────────────────────────────────────────────

TEST_FILE   = Path("splits/test.jsonl")
MODEL_DIR   = Path("models")
RESULTS_DIR = Path("results/attack")

# ── Violation token vocabulary ────────────────────────────────────────────────
# These are the actual BAD-form tokens we injected.
# Tokens in this set are NOT considered vulnerable even if saliency is high —
# they are the legitimate signal the model should use.

VIOLATION_TOKENS = {
    # V01
    "per", "cent",
    # V03
    "mkr", "mnkr",
    # V04
    "square", "meters", "sq",
    # V05
    "kpi",
    # V06
    "direct", "yield",
    # V07 / V07b
    "occupancy", "ratio", "letting",
    # V08
    "disposal", "disposals",
    # V09
    "significant", "policies",
    # V10
    "parent", "company",
    # V14
    "average", "contract", "period",
    # V15
    "land", "holdings",
    # V16
    "carcass", "carcasses",
    # V17
    "rights", "issue",
    # V18
    "taxable", "values",
    # V19
    "participations",
    # V20
    "assets",
    # V21
    "figures",
    # V22
    "profit",
    # V23
    "balance", "sheet", "items",
    # V24
    "directors",
    # V25
    "measurement", "hierarchy",
    # V26
    "rate",
    # V27
    "refrigeration", "freezing", "facility",
    # V28
    "presented", "online",
    # V29
    "disposals",
    # V30
    "financial",
}

# ── Synonym dictionary ────────────────────────────────────────────────────────
# Financially neutral synonyms for common words that might appear as
# high-saliency vulnerable tokens.
# Key = token as it appears after tokenisation (lowercase).
# Value = list of replacement options (also lowercase).

SYNONYMS: dict[str, list[str]] = {
    # Common verbs
    "signed"      : ["executed", "entered"],
    "acquire"     : ["purchase", "obtain"],
    "tecknade"    : ["ingick", "genomförde"],       # Swedish
    "fastigheter" : ["tillgångar", "objekt"],       # Swedish
    "signed"      : ["concluded", "finalised"],
    "increase"    : ["rise", "grow"],
    "increased"   : ["rose", "grew"],
    "decreased"   : ["fell", "declined"],
    "decrease"    : ["fall", "decline"],
    "shows"       : ["indicates", "demonstrates"],
    "showed"      : ["indicated", "demonstrated"],
    "reported"    : ["stated", "disclosed"],
    "report"      : ["statement", "disclosure"],
    "noted"       : ["stated", "recorded"],
    "based"       : ["founded", "grounded"],
    "compared"    : ["relative", "measured"],
    "related"     : ["associated", "linked"],
    "following"   : ["below", "subsequent"],
    "used"        : ["applied", "utilised"],
    "using"       : ["applying", "utilising"],
    "resulting"   : ["arising", "stemming"],
    "current"     : ["present", "existing"],
    "total"       : ["aggregate", "combined"],
    "property"    : ["asset", "holding"],
    "properties"  : ["assets", "holdings"],
    "value"       : ["amount", "figure"],
    "values"      : ["amounts", "figures"],
    "period"      : ["interval", "duration"],
    "end"         : ["close", "conclusion"],
    "level"       : ["degree", "extent"],
    "impact"      : ["effect", "influence"],
    "change"      : ["shift", "movement"],
    "changes"     : ["shifts", "movements"],
    "estimated"   : ["assessed", "calculated"],
    "fair"        : ["assessed", "market"],
    "underlying"  : ["fundamental", "core"],
    "loan"        : ["debt", "borrowing"],
    "agreement"   : ["contract", "arrangement"],
    "company"     : ["group", "entity"],
    "illustrate"  : ["demonstrate", "show"],
    "sensitivity" : ["impact", "effect"],
    "analysis"    : ["assessment", "review"],
    "parameters"  : ["variables", "factors"],
    "impact"      : ["effect", "consequence"],
    "at"          : ["of", "with"],
    "from"        : ["sourced", "originating"],
    "two"         : ["2", "a pair of"],
    "500"         : ["five hundred"],
    "the"         : ["this", "a"],
    "an"          : ["a", "one"],
    "can"         : ["may", "could"],
    "be"          : ["become", "remain"],
    "on"          : ["upon", "regarding"],
    "in"          : ["within", "across"],
    "of"          : ["for", "regarding"],
    "and"         : ["as well as", "along with"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_record(
    rec       : dict,
    tokeniser : SimpleTokeniser,
    max_len   : int,
    cls_id    : int,
    sep_id    : int,
) -> tuple[list[int], list[str]]:
    src_ids = tokeniser.encode(rec["source_sv"])
    tgt_ids = tokeniser.encode(rec["translation"])
    budget  = max_len - 3
    src_ids = src_ids[:budget // 2]
    tgt_ids = tgt_ids[:budget - budget // 2]

    ids = [cls_id] + src_ids + [sep_id] + tgt_ids + [sep_id]
    pad_len = max_len - len(ids)
    ids = ids + [PAD_IDX] * pad_len

    tokens = [tokeniser.id2token.get(i, "<UNK>") for i in ids]
    return ids, tokens


def predict(
    model     : StyleGuideClassifier,
    ids       : list[int],
    device    : torch.device,
    threshold : float = 0.5,
) -> tuple[int, float]:
    """Returns (predicted_label, p_violation)."""
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logit = model(input_ids)
    prob = torch.sigmoid(logit).item()
    return int(prob >= threshold), prob


def is_vulnerable(token: str, saliency: float, saliency_threshold: float) -> bool:
    """
    A token is vulnerable if:
    - It has saliency above the threshold (model is sensitive to it)
    - It is NOT a known violation token
    - It is NOT a special token
    - It has a synonym available
    """
    if token in {"<CLS>", "<SEP>", "<PAD>", "<UNK>"}:
        return False
    if token.lower() in VIOLATION_TOKENS:
        return False
    if saliency < saliency_threshold:
        return False
    if token.lower() not in SYNONYMS:
        return False
    return True


# ── Attack ────────────────────────────────────────────────────────────────────

def attack_example(
    rec              : dict,
    model            : StyleGuideClassifier,
    tokeniser        : SimpleTokeniser,
    device           : torch.device,
    max_len          : int,
    cls_id           : int,
    sep_id           : int,
    top_vulnerable   : int,
    saliency_pct     : float,   # top X% saliency tokens are "high"
) -> dict:
    """
    Attempts to flip the prediction of a single violation example.
    Returns a result dict describing what happened.
    """
    ids, tokens = encode_record(rec, tokeniser, max_len, cls_id, sep_id)

    # Original prediction
    orig_pred, orig_prob = predict(model, ids, device)

    # Only attack examples the model gets right
    if orig_pred != 1:
        return {
            "id"           : rec.get("id"),
            "violation_id" : rec.get("violation_id"),
            "skipped"      : True,
            "reason"       : "model already wrong on original",
            "attack_success": False,
        }

    # Compute saliency
    input_tensor = torch.tensor([ids], dtype=torch.long)
    logit, saliency = compute_saliency(model, input_tensor, device)

    # Determine saliency threshold (top X% of non-pad tokens)
    real_saliency = [
        s for t, s in zip(tokens, saliency)
        if t != "<PAD>"
    ]
    if not real_saliency:
        return {"id": rec.get("id"), "skipped": True,
                "reason": "empty", "attack_success": False}

    real_saliency_sorted = sorted(real_saliency, reverse=True)
    threshold_idx = max(0, int(len(real_saliency_sorted) * saliency_pct) - 1)
    sal_threshold = real_saliency_sorted[threshold_idx]

    # Rank vulnerable tokens by saliency
    vulnerable = []
    for pos, (tok, sal) in enumerate(zip(tokens, saliency)):
        if is_vulnerable(tok, sal, sal_threshold):
            vulnerable.append((pos, tok, sal))

    vulnerable = sorted(vulnerable, key=lambda x: x[2], reverse=True)
    vulnerable = vulnerable[:top_vulnerable]

    if not vulnerable:
        return {
            "id"            : rec.get("id"),
            "violation_id"  : rec.get("violation_id"),
            "skipped"       : False,
            "reason"        : "no vulnerable tokens with synonyms found",
            "attack_success": False,
            "vulnerable_tokens": [],
        }

    # Try replacing each vulnerable token with each synonym
    attack_log = []
    flipped    = False
    flip_detail = None

    for pos, tok, sal in vulnerable:
        synonyms = SYNONYMS.get(tok.lower(), [])
        for syn in synonyms:
            # Encode synonym (take first token if multi-token)
            syn_ids = tokeniser.encode(syn)
            if not syn_ids:
                continue
            syn_id = syn_ids[0]

            # Build modified sequence
            modified_ids = ids.copy()
            modified_ids[pos] = syn_id

            new_pred, new_prob = predict(model, modified_ids, device)

            entry = {
                "position"   : pos,
                "original"   : tok,
                "synonym"    : syn,
                "orig_prob"  : round(orig_prob, 4),
                "new_prob"   : round(new_prob, 4),
                "saliency"   : round(sal, 4),
                "flipped"    : new_pred == 0,
            }
            attack_log.append(entry)

            if new_pred == 0 and not flipped:
                flipped     = True
                flip_detail = entry

    return {
        "id"               : rec.get("id"),
        "violation_id"     : rec.get("violation_id"),
        "source_sv"        : rec.get("source_sv", "")[:120],
        "translation"      : rec.get("translation", "")[:120],
        "skipped"          : False,
        "orig_pred"        : orig_pred,
        "orig_prob"        : round(orig_prob, 4),
        "attack_success"   : flipped,
        "flip_detail"      : flip_detail,
        "vulnerable_tokens": [(t, round(s, 4)) for _, t, s in vulnerable],
        "attack_log"       : attack_log,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    tokeniser = SimpleTokeniser.load(MODEL_DIR / "tokeniser.json")
    ckpt      = torch.load(
        MODEL_DIR / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    saved_cfg = ckpt["cfg"]
    max_len   = saved_cfg["max_len"]

    model = StyleGuideClassifier(
        vocab_size = len(tokeniser),
        d_model    = saved_cfg["d_model"],
        n_heads    = saved_cfg["n_heads"],
        n_layers   = saved_cfg["n_layers"],
        d_ff       = saved_cfg["d_ff"],
        max_len    = max_len,
        dropout    = 0.0,
        pad_idx    = PAD_IDX,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt['epoch']}  val_F1={ckpt['val_macro_f1']:.4f}")

    cls_id = tokeniser.token2id["<CLS>"]
    sep_id = tokeniser.token2id["<SEP>"]

    # ── Load test records ─────────────────────────────────────────────────────
    with TEST_FILE.open(encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]

    # Only violation examples
    violations = [r for r in all_records if r["label"] == 1]
    if cfg["violation_id"]:
        violations = [r for r in violations if r.get("violation_id") == cfg["violation_id"]]

    n = min(cfg["n_samples"], len(violations))
    sample = random.sample(violations, n)
    print(f"Attacking {n} violation examples...\n")

    # ── Run attacks ───────────────────────────────────────────────────────────
    results   = []
    n_skipped = 0
    n_success = 0
    n_tried   = 0

    by_violation: dict[str, dict] = defaultdict(lambda: {"tried": 0, "success": 0})

    for i, rec in enumerate(sample):
        result = attack_example(
            rec, model, tokeniser, device,
            max_len, cls_id, sep_id,
            top_vulnerable = cfg["top_vulnerable"],
            saliency_pct   = cfg["saliency_pct"],
        )
        results.append(result)

        vid = result.get("violation_id", "?")

        if result["skipped"]:
            n_skipped += 1
            print(f"[{i+1:>3}/{n}] SKIP  {rec.get('id')}  — {result['reason']}")
            continue

        n_tried += 1
        by_violation[vid]["tried"] += 1

        if result["attack_success"]:
            n_success += 1
            by_violation[vid]["success"] += 1
            fd = result["flip_detail"]
            print(
                f"[{i+1:>3}/{n}] ✓ FLIP  {rec.get('id'):<20} "
                f"rule={vid}  "
                f"'{fd['original']}' → '{fd['synonym']}'  "
                f"P: {fd['orig_prob']:.3f} → {fd['new_prob']:.3f}"
            )
        else:
            vt = [t for t, _ in result["vulnerable_tokens"]]
            print(
                f"[{i+1:>3}/{n}] ✗ HOLD  {rec.get('id'):<20} "
                f"rule={vid}  "
                f"vulnerable={vt[:3]}"
            )

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "attack_results.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    success_rate = n_success / n_tried if n_tried > 0 else 0.0

    summary_lines = [
        "=" * 60,
        "ADVERSARIAL ATTACK SUMMARY",
        "=" * 60,
        f"Examples sampled   : {n}",
        f"Skipped (wrong orig): {n_skipped}",
        f"Attacked           : {n_tried}",
        f"Successful flips   : {n_success}",
        f"Attack success rate: {success_rate:.1%}",
        "",
        "BY VIOLATION RULE:",
    ]
    for vid, counts in sorted(by_violation.items()):
        t, s = counts["tried"], counts["success"]
        r = s / t if t > 0 else 0
        bar = "█" * s + "░" * (t - s)
        summary_lines.append(f"  {vid:<6} {s}/{t} ({r:.0%})  {bar}")

    summary_lines += [
        "",
        "INTERPRETATION:",
        f"  {success_rate:.1%} of correctly-predicted violations were",
        "  flipped to 'compliant' by replacing a semantically neutral",
        "  high-saliency token with a synonym — without fixing the",
        "  actual style-guide violation.",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = RESULTS_DIR / "attack_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nResults → {out_path}")
    print(f"Summary → {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",     type=int,   default=100,
                        help="Number of violation examples to attack")
    parser.add_argument("--top_vulnerable", type=int,  default=5,
                        help="Max vulnerable tokens to try per example")
    parser.add_argument("--saliency_pct",  type=float, default=0.2,
                        help="Top X fraction of tokens considered high-saliency")
    parser.add_argument("--violation_id",  type=str,   default=None,
                        help="Focus on a specific rule e.g. V01")
    return vars(parser.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    print("Config:", cfg)
    main(cfg)