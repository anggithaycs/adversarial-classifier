"""
generate_negatives.py 
=> To enerate non-compliant (negative) training synthetic examples
by injecting known Catena style-guide violations into positive pairs.
 
!! Remember !!
Usage:
    python generate_violations.py
 
Input:
    processed/sentence_pairs.jsonl   (positive pairs)
 
Output:
    processed/violations.jsonl        (label = 0)
 
Violation tiers
---------------
Tier 1 – Unconditional string substitutions (safe to automate):
    V01  percent → per cent
    V02  website → web site
    V03  SEK X million → X Mkr
    V04  m2 / m² → square meters
    V05  CPI → KPI
    V06  yield → direct yield  (property; excludes dividend yield)
    V07  occupancy rate → occupancy ratio
    V07b occupancy rate → letting ratio
    V08  divestment → disposal
    V14  weighted average lease expiry → average contract period
    V15  land properties → land holdings
    V16  building frames → carcasses
    V17  new share issue → rights issue
    V18  property tax value → taxable values
    V19  Profit from investments in → Profit from participations in
    V20  Equity ratio → Equity/assets ratio
    V21  EPRA performance measures → EPRA key figures
    V22  Net profit for the year → Profit for the year
    V23  Foreign currency transactions and balances →
         Foreign currency transactions and balance sheet items
    V24  Board fees → Directors' fees
    V25  valuation hierarchy → measurement hierarchy
    V26  capitalised interest → capitalised interest rate
    V27  cold storage → refrigeration and freezing facility
    V28  live-streamed → presented online
    V29  Acquisitions, investments and divestments →
         Acquisitions, investments and disposals
    V30  key performance indicators → key financial figures
 
Tier 2 – Context-guarded (only inject when a guard phrase is present):
    V09  material accounting policy information → significant accounting policies
    V10  Parent Company financial statements → Parent Company's financial statements
    V11  Nomination Committee → nomination committee  (capitalisation)
    V12  Remuneration Committee → remuneration committee
    V13  Audit Committee → audit committee
"""
 
import json
import re
import random
from pathlib import Path
from dataclasses import dataclass
 
# ── Config ────────────────────────────────────────────────────────────────────
 
RANDOM_SEED = 42
IN_FILE  = Path("processed/sentence_pairs.jsonl")
OUT_FILE = Path("processed/violations.jsonl")
 
# Max negatives per rule — keeps dataset balanced
MAX_PER_RULE = 2000
 
 
# ────────────────────  Violation definitions ────────────────────────
 
@dataclass
class Violation:
    vid: str
    desc: str
    tier: int
    guard: str | None
    pattern: re.Pattern
    replacement: str
 
 
def build_violations() -> list[Violation]:
    return [
 
        # ══════════════════════════════════════════════════════════════════
        # TIER 1 — unconditional
        # ══════════════════════════════════════════════════════════════════
 
        Violation(
            vid="V01",
            desc="'percent' → forbidden 'per cent'",
            tier=1, guard=None,
            pattern=re.compile(r"\bpercent\b", re.IGNORECASE),
            replacement="per cent",
        ),
        Violation(
            vid="V02",
            desc="'website' → forbidden 'web site'",
            tier=1, guard=None,
            pattern=re.compile(r"\bwebsite\b", re.IGNORECASE),
            replacement="web site",
        ),
        Violation(
            vid="V03",
            desc="Currency: 'SEK X million' → forbidden 'X Mkr'",
            tier=1, guard=None,
            pattern=re.compile(r"SEK\s+(-?\d[\d,\.]*)\s+million", re.IGNORECASE),
            replacement=r"\1 Mkr",
        ),
        Violation(
            vid="V04",
            desc="Area unit: 'm²/m2' → forbidden 'square meters'",
            tier=1, guard=None,
            pattern=re.compile(r"\bm²\b|\bm2\b"),
            replacement="square meters",
        ),
        Violation(
            vid="V05",
            desc="'CPI' (consumer price index) → forbidden 'KPI'",
            tier=1, guard=None,
            pattern=re.compile(r"\bCPI\b"),
            replacement="KPI",
        ),
        Violation(
            vid="V06",
            desc="'yield' → forbidden 'direct yield' (excludes dividend yield)",
            tier=1, guard=None,
            # Negative lookbehind: don't match if preceded by 'direct' or 'dividend'
            pattern=re.compile(r"(?<!direct\s)(?<!dividend\s)\byield\b", re.IGNORECASE),
            replacement="direct yield",
        ),
        Violation(
            vid="V07",
            desc="'occupancy rate' → forbidden 'occupancy ratio'",
            tier=1, guard=None,
            pattern=re.compile(r"\boccupancy rate\b", re.IGNORECASE),
            replacement="occupancy ratio",
        ),
        Violation(
            vid="V07b",
            desc="'occupancy rate' → forbidden 'letting ratio'",
            tier=1, guard=None,
            pattern=re.compile(r"\boccupancy rate\b", re.IGNORECASE),
            replacement="letting ratio",
        ),
        Violation(
            vid="V08",
            desc="'divestment' → forbidden 'disposal'",
            tier=1, guard=None,
            pattern=re.compile(r"\bdivestment\b", re.IGNORECASE),
            replacement="disposal",
        ),
        Violation(
            vid="V14",
            desc="'weighted average lease expiry' → forbidden 'average contract period'",
            tier=1, guard=None,
            pattern=re.compile(r"\bweighted average lease expiry\b", re.IGNORECASE),
            replacement="average contract period",
        ),
        Violation(
            vid="V15",
            desc="'land properties' → forbidden 'land holdings'",
            tier=1, guard=None,
            pattern=re.compile(r"\bland properties\b", re.IGNORECASE),
            replacement="land holdings",
        ),
        Violation(
            vid="V16",
            desc="'building frame(s)' → forbidden 'carcass(es)'",
            tier=1, guard=None,
            pattern=re.compile(r"\bbuilding frames?\b", re.IGNORECASE),
            replacement="carcasses",
        ),
        Violation(
            vid="V17",
            desc="'new share issue' → forbidden 'rights issue'",
            tier=1, guard=None,
            pattern=re.compile(r"\bnew share issue\b", re.IGNORECASE),
            replacement="rights issue",
        ),
        Violation(
            vid="V18",
            desc="'property tax value' → forbidden 'taxable values'",
            tier=1, guard=None,
            pattern=re.compile(r"\bproperty tax value\b", re.IGNORECASE),
            replacement="taxable values",
        ),
        Violation(
            vid="V19",
            desc="'Profit from investments in' → forbidden 'Profit from participations in'",
            tier=1, guard=None,
            pattern=re.compile(r"\bProfit from investments in\b", re.IGNORECASE),
            replacement="Profit from participations in",
        ),
        Violation(
            vid="V20",
            desc="'Equity ratio' → forbidden 'Equity/assets ratio'",
            tier=1, guard=None,
            pattern=re.compile(r"\bEquity ratio\b", re.IGNORECASE),
            replacement="Equity/assets ratio",
        ),
        Violation(
            vid="V21",
            desc="'EPRA performance measures' → forbidden 'EPRA key figures'",
            tier=1, guard=None,
            pattern=re.compile(r"\bEPRA performance measures\b", re.IGNORECASE),
            replacement="EPRA key figures",
        ),
        Violation(
            vid="V22",
            desc="'Net profit for the year' → forbidden 'Profit for the year'",
            tier=1, guard=None,
            pattern=re.compile(r"\bNet profit for the year\b", re.IGNORECASE),
            replacement="Profit for the year",
        ),
        Violation(
            vid="V23",
            desc="'Foreign currency transactions and balances' → forbidden '... balance sheet items'",
            tier=1, guard=None,
            pattern=re.compile(
                r"\bForeign currency transactions and balances\b", re.IGNORECASE
            ),
            replacement="Foreign currency transactions and balance sheet items",
        ),
        Violation(
            vid="V24",
            desc="'Board fees' → forbidden 'Directors' fees'",
            tier=1, guard=None,
            pattern=re.compile(r"\bBoard fees\b", re.IGNORECASE),
            replacement="Directors' fees",
        ),
        Violation(
            vid="V25",
            desc="'valuation hierarchy' → forbidden 'measurement hierarchy'",
            tier=1, guard=None,
            pattern=re.compile(r"\bvaluation hierarchy\b", re.IGNORECASE),
            replacement="measurement hierarchy",
        ),
        Violation(
            vid="V26",
            desc="'capitalised interest' → forbidden 'capitalised interest rate'",
            tier=1, guard=None,
            # Negative lookahead: don't match if 'rate' already follows
            pattern=re.compile(r"\bcapitalised interest\b(?!\s+rate)", re.IGNORECASE),
            replacement="capitalised interest rate",
        ),
        Violation(
            vid="V27",
            desc="'cold storage' → forbidden 'refrigeration and freezing facility'",
            tier=1, guard=None,
            pattern=re.compile(r"\bcold storage\b", re.IGNORECASE),
            replacement="refrigeration and freezing facility",
        ),
        Violation(
            vid="V28",
            desc="'live-streamed' → forbidden 'presented online'",
            tier=1, guard=None,
            pattern=re.compile(r"\blive-streamed\b", re.IGNORECASE),
            replacement="presented online",
        ),
        Violation(
            vid="V29",
            desc="'Acquisitions, investments and divestments' → forbidden '... disposals'",
            tier=1, guard=None,
            pattern=re.compile(
                r"\bAcquisitions, investments and divestments\b", re.IGNORECASE
            ),
            replacement="Acquisitions, investments and disposals",
        ),
        Violation(
            vid="V30",
            desc="'key performance indicators' → forbidden 'key financial figures'",
            tier=1, guard=None,
            pattern=re.compile(r"\bkey performance indicators\b", re.IGNORECASE),
            replacement="key financial figures",
        ),
 
        # ══════════════════════════════════════════════════════════════════
        # TIER 2 — context-guarded
        # Has GUARD AND PATTERN!!!!
        # ══════════════════════════════════════════════════════════════════
 
        Violation(
            vid="V09",
            desc="IAS 1: 'material accounting policy information' → 'significant accounting policies'",
            tier=2,
            guard=r"accounting polic",
            pattern=re.compile(
                r"material\s+accounting\s+policy\s+information", re.IGNORECASE
            ),
            replacement="significant accounting policies",
        ),
        Violation(
            vid="V10",
            desc="'Parent Company financial statements' → forbidden possessive form",
            tier=2,
            guard=r"Parent Company",
            pattern=re.compile(r"Parent Company financial statements", re.IGNORECASE),
            replacement="Parent Company's financial statements",
        ),
        Violation(
            vid="V11",
            desc="'Nomination Committee' → wrong lowercase",
            tier=2,
            guard=r"Nomination Committee",
            pattern=re.compile(r"\bNomination Committee\b"),
            replacement="nomination committee",
        ),
        Violation(
            vid="V12",
            desc="'Remuneration Committee' → wrong lowercase",
            tier=2,
            guard=r"Remuneration Committee",
            pattern=re.compile(r"\bRemuneration Committee\b"),
            replacement="remuneration committee",
        ),
        Violation(
            vid="V13",
            desc="'Audit Committee' → wrong lowercase",
            tier=2,
            guard=r"Audit Committee",
            pattern=re.compile(r"\bAudit Committee\b"),
            replacement="audit committee",
        ),
    ]
 
 
# ── Inline tag stripper ───────────────────────────────────────────────────────
 
TAG_RE = re.compile(r"[\[\{]\d+[\]\}]") #regex to match any digit inside [] {} but not ()
 
def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text).strip()
 
 
# ── Core injection logic ──────────────────────────────────────────────────────
"""
Our guide:
If this is a tier-2 violation AND a guard is defined, the guard pattern must be found somewhere in the translation. 
If it's not there, don't even attempt the injection.

Since this works as a bouncer with two checks:
Guard check: Is the right context present? (only for tier 2, conditionally)
Pattern check:  Is the thing I want to replace actually there?

In example: v.pattern is something generic that could match in many places, but the translation can only be injected if its also talking about a specific topic. The guard lets us say: "only do this replacement if the text also contains X."
Furthermore:
guard: r"price" → only proceed if the word "price" is somewhere in the text
pattern: replaces some number → now you know you're replacing a price-related number, not a random one

So guard is like a safety condition to avoid false positive injections.

"""

def try_inject(translation: str, v: Violation) -> str | None:
    if v.tier == 2 and v.guard:
        if not re.search(v.guard, translation):
            return None
    if not v.pattern.search(translation):
        return None
    mutated = v.pattern.sub(v.replacement, translation, count=1)
    if mutated == translation:
        return None
    return mutated
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    random.seed(RANDOM_SEED)
    violations = build_violations()
 
    positives = []
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                positives.append(json.loads(line))
 
    print(f"Loaded {len(positives)} positive pairs from {IN_FILE}")
    random.shuffle(positives)
 
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {v.vid: 0 for v in violations}
    written = 0
 
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for v in violations:
            count = 0
            for rec in positives:
                if count >= MAX_PER_RULE:
                    break
                translation_raw = rec.get("translation", "")
                translation = strip_tags(translation_raw)
                if not translation:
                    continue
                mutated = try_inject(translation, v)
                if mutated is None:
                    continue
 
                negative = {
                    **rec,
                    "label": 0,
                    "violation_id": v.vid,
                    "violation_tier": v.tier,
                    "violation_desc": v.desc,
                    "original_en": translation,
                    "translation": mutated,
                }
                out.write(json.dumps(negative, ensure_ascii=False) + "\n")
                count += 1
                written += 1
 
            totals[v.vid] = count
            tier_label = f"T{v.tier}"
            print(f"  {v.vid:<5} ({tier_label}) – {count:>5} negatives  |  {v.desc}")
 
    print(f"\nTotal negatives written: {written} → {OUT_FILE}")
    print("\nViolation breakdown:")
    for vid, n in totals.items():
        bar = "█" * (n // 50)
        print(f"  {vid}: {n:>5}  {bar}")
 
 
if __name__ == "__main__":
    main()