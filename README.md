# Adversarial Style-Guide Compliance Classifier

*Language Engineering Mini Project*

A binary Transformer-based text classifier trained to detect style-guide
violations in Swedish→English financial translations (Client X, translated
by Fluid Translation). The project covers all four assignment steps:
model training, gradient saliency analysis, adversarial attack, and defense.

> **Note on data:** the client translation assets (TMX, term bases, RTF reports,
> style guide) and all derived files (`data/`, `models/`, `results/`, `splits/`)
> are **not tracked in this repository** — they are confidential and are
> git-ignored. To reproduce, place the raw client files under
> `data/raw/ClientX_package/` and run the pipeline below.

## Repository structure

```
.
├── src/                      # core model + training + analysis
│   ├── model.py              #   from-scratch bidirectional Transformer
│   ├── train.py              #   training loop, tokeniser, dataset, focal loss
│   ├── analyze.py            #   gradient saliency
│   ├── attack.py             #   synonym-swap adversarial attack
│   └── defense.py            #   adversarial-training defense
├── data_prep/                # raw exports → clean JSONL
│   ├── parse_tmx.py          #   translation memories  → sentence pairs
│   ├── parse_termbase.py     #   term-base CSVs         → glossary
│   ├── parse_report_rtf.py   #   bilingual RTF reports  → segments
│   ├── generate_violations.py#   inject style-guide violations (negatives)
│   └── build_dataset.py      #   merge + stratified 80/10/10 split
├── notebooks/                # exploratory / visualization notebooks
├── docs/                     # workflow diagram
├── requirements.txt
└── README.md
```

> **How to run:** execute every script **from the repository root** (e.g.
> `python src/train.py`), not from inside `src/`. Scripts read and write
> relative paths such as `models/` and `splits/`, which resolve against the
> repo root.

## Installation

```bash
pip install torch scikit-learn
```

> Tested on Python 3.11, PyTorch 2.6, CUDA 12. Runs on KTH GPU nodes.

## Replication — run in order

### Step 0 · Create output directories

```bash
mkdir -p data/raw/ClientX_package
mkdir -p data/processed
mkdir -p splits
mkdir -p models results/saliency/heatmaps results/attack results/defense
```

### Step 1 · Parse raw data (Phase 1)

Extract positive training pairs from the translation memory:

```bash
python data_prep/parse_tmx.py
```
Output: `data/processed/sentence_pairs.jsonl` (17,102 pairs)

Extract evaluation segments from the bilingual RTF reports:

```bash
python data_prep/parse_report_rtf.py
```
Output: `q1_2026_segments.jsonl` (2,581 rows), `q3_2025_segments.jsonl` (2,713 rows)

Extract terminology pairs from the CSV term bases:

```bash
python data_prep/parse_termbase.py
```
Output: `termbase.jsonl` (1,937 term pairs), `termbase_summary.txt`

### Step 2 · Generate negative examples (Phase 2)

Inject style-guide violations into positive pairs to create the negative class:

```bash
python data_prep/generate_violations.py
```
Output: `data/processed/violations.jsonl` (3,115 violations)

31 violation rules derived directly from the Client X style guide:

- **Tier 1 (26 rules)** — unconditional string substitutions
  e.g. `percent → per cent`, `SEK X million → X Mkr`, `m² → square meters`
- **Tier 2 (5 rules)** — context-guarded substitutions
  e.g. `material accounting policy information → significant accounting policies`

### Step 3 · Build labeled dataset (Phase 3)

Merge positives + negatives, strip inline tags, stratified 80/10/10 split:

```bash
python data_prep/build_dataset.py
```
Output: `train.jsonl` (16,174), `val.jsonl` (2,021), `test.jsonl` (2,023)

Key design decisions:

- No undersampling — keeps all 17k positives (Zhang et al. 2024)
- Focal loss with γ=0.2, class weights [0.59, 3.25] (inverse frequency)
- Label convention: `0 = compliant`, `1 = violation`

### Step 4 · Train classifier (Phase 4 · Week 1)

Train the bidirectional Transformer from scratch:

```bash
python src/train.py
```
Output: `models/best_model.pt`, `models/tokeniser.json`, `models/training_log.jsonl`

Optional — tune hyperparameters:

```bash
python src/train.py --epochs 20 --gamma 0.5 --batch_size 64
```

Expected results:

```
Best val macro F1 : 0.9618
Violation precision: 0.94  recall: 0.93  F1: 0.94
Compliant precision: 0.99  recall: 0.99  F1: 0.99
```

Model architecture:

- 4-layer bidirectional Transformer encoder (no causal mask)
- d_model=256, 4 heads, d_ff=1024, max_len=256
- Input: `[CLS] source_sv [SEP] translation_en [SEP]`
- Binary classification head → single logit

### Step 5a · Gradient saliency analysis (Phase 5a · Week 2)

Compute ∂logit/∂embedding per token and visualize as heatmaps:

```bash
# Default: 50 random violation examples
python src/analyze.py

# Focus on V01 (percent → per cent)
python src/analyze.py --violation_id V01 --n_samples 30

# Include compliant examples
python src/analyze.py --labels both --n_samples 50
```
Output: `results/saliency/saliency_scores.jsonl`, `results/saliency/heatmaps/*.html`

Expected finding: violation tokens (`cent`, `mkr`) rank #1 in saliency,
confirming the model learned the correct surface signal.

### Step 5b · Adversarial attack (Phase 5b · Week 2)

Find high-saliency neutral tokens and replace with synonyms to flip predictions:

```bash
python src/attack.py

# Focus on specific rule
python src/attack.py --violation_id V01 --n_samples 80

# Try more tokens per example
python src/attack.py --top_vulnerable 8 --saliency_pct 0.3
```
Output: `results/attack/attack_results.jsonl`, `results/attack/attack_summary.txt`

Expected results:

```
Attack success rate: 1.0%  (1/96 examples flipped)
Successful flip: V08 · 'change' → 'movement'  P: 0.993 → 0.350
```

Note: low success rate is a finding, not a failure — the model's decision is
driven by the exact violation token, making neutral token substitution ineffective.

### Step 6 · Defense (Phase 6 · Week 3)

Augment training set with adversarial examples and retrain:

```bash
python src/defense.py

# Generate more augments
python src/defense.py --augment_multiplier 3
```
Output: `models/defense_model.pt`, `results/defense/defense_comparison.txt`

Expected results:

```
Original model attack success : 1.0%
Defense model attack success  : 0.0%
Defense model val F1          : 0.9563  (vs 0.9618 original)
```

Answer to Q4: Yes, it is harder to break — attack success dropped from 1% to 0%.
