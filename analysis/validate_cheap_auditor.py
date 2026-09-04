"""Can a cheap non-reasoning model replace the reasoning auditor?

    python analysis/validate_cheap_auditor.py --n 120

## Motivation

The current auditor `deepseek-v4-pro` is a **reasoning model**: a batch of 20
candidates burns tens of thousands of tokens of thinking, ~200s per call. 19
sites take 100+ minutes; scaling to a hundred-plus sites is not feasible.

The task itself is simple: given a place name plus a distance, decide whether
it works as a reference point, rate its referenceability, and give canonical
Chinese/English names. **It does not need a reasoning model.**

## Existing indirect evidence

notes/07 records auditor agreement of **kappa = 0.919 across models**
(deepseek-v4-pro vs claude-opus-5). In other words, this task is **already
insensitive to which model does it** -- which is even less reason to use the
slowest one.

## Method: re-audit already-decided candidates rather than run an A/B

There are already 1431 `deepseek-v4-pro@en` decisions. Have a cheap model
audit the **same** candidates and measure agreement with the existing
decisions. This costs half of rerunning both arms as an A/B, and compares
against labels that are actually in use.

Criteria (matching notes/07):

    ok            binary -> Cohen's kappa >= 0.8
    familiarity   tier agreement (high 4-5 / mid 2-3 / low 0-1) >= 90%
                  **tier only**, because the ladder only ever uses the tier,
                  never the raw score

Timing is reported alongside -- that is the actual motivation for switching models.
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import audit, config, providers  # noqa: E402

TIERS = {"high": (4, 5), "mid": (2, 3), "low": (0, 1)}
REF_TAG = "deepseek-v4-pro@en"


def tier_of(f):
    if not isinstance(f, (int, float)) or f != f:
        return None
    for name, (lo, hi) in TIERS.items():
        if lo <= f <= hi:
            return name
    return None


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    p1, q1 = sum(a) / n, sum(b) / n
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120, help="how many already-audited candidates to re-audit")
    ap.add_argument("--models", nargs="+",
                    default=["deepseek-vision", "kimi-vision"],
                    help="candidate cheap auditors (must not be in EVAL_MODELS)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # reference labels: the existing deepseek-v4-pro@en decisions
    ref = {}
    for line in audit.DECISIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("model") == REF_TAG:
            ref[r["qid"]] = r
    print(f"reference labels ({REF_TAG}): {len(ref)}")

    # pull the full row (needs label / dist_km / en_title) for these qids from
    # any candidate table
    frames = []
    for f in sorted((config.LADDERS).glob("*_candidates_*.pkl")):
        frames.append(pd.read_pickle(f))
    if not frames:
        print("no candidate pkl found, run build_ladder.py first")
        return
    cand = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset="qid", keep="first"))
    cand = cand[cand.qid.isin(ref)].reset_index(drop=True)
    samp = cand.sample(min(args.n, len(cand)), random_state=args.seed)
    n = len(samp)
    print(f"sampled {n} for re-audit\n")

    for name in args.models:
        if name in providers.EVAL_MODELS:
            print(f"skipping {name}: it is in EVAL_MODELS and cannot be an auditor")
            continue
        try:
            client = providers.REGISTRY[name]()
        except Exception as e:
            print(f"{name}: could not create a client -- {e}")
            continue

        got, t0, calls = {}, time.time(), 0
        for start in range(0, n, args.batch):
            chunk = samp.iloc[start:start + args.batch]
            try:
                arr = audit._ask(client, audit._fmt(chunk), True, "en")
                calls += 1
            except Exception as e:
                print(f"  batch failed {type(e).__name__}: {str(e)[:60]}")
                continue
            if not arr:
                continue
            for d in arr:
                i = d.get("idx")
                if isinstance(i, int) and 0 <= i < len(chunk):
                    got[chunk.iloc[i].qid] = d
        dt = time.time() - t0

        common = [q for q in samp.qid if q in got]
        if not common:
            print(f"{name}: parsed nothing at all, skipping\n")
            continue
        ok_a = [bool(ref[q].get("ok")) for q in common]
        ok_b = [bool(got[q].get("ok")) for q in common]
        fa = [ref[q].get("familiarity") for q in common]
        fb = [got[q].get("familiarity") for q in common]
        pairs = [(x, y) for x, y in zip(fa, fb)
                 if isinstance(x, (int, float)) and isinstance(y, (int, float))]
        tier = (sum(tier_of(x) == tier_of(y) for x, y in pairs) / len(pairs)
                if pairs else float("nan"))
        exact = (sum(x == y for x, y in pairs) / len(pairs)
                 if pairs else float("nan"))

        print(f"=== {name} ===")
        print(f"  time {dt:.1f}s / {calls} calls = {dt/max(calls,1):.1f}s per batch"
              f" (reference model is ~200s per batch)")
        print(f"  parsed {len(common)}/{n}")
        print(f"  ok  kappa                {kappa(ok_a, ok_b):.3f}   (criterion >= 0.80)")
        print(f"  familiarity-tier agreement {tier:.1%}   (criterion >= 90%)")
        print(f"  familiarity exact match    {exact:.1%}   (reference only, the ladder never uses the raw score)")
        verdict = ("PASS: can replace" if kappa(ok_a, ok_b) >= 0.8 and tier >= 0.90
                   else "FAIL: below criteria")
        print(f"  -> {verdict}\n")

    print("NOTE: if replaced, the cache key must change from deepseek-v4-pro@en")
    print("      to the new model's tag, and **must not be mixed with old decisions** --")
    print("      either re-audit everything, or keep the old decisions for old sites only.")


if __name__ == "__main__":
    main()
