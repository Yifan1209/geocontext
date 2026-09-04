"""Did the models actually fill in the forced_chain evidence checklist?

    python analysis/chain_compliance.py

## Why this exists

`chain_results.py` reports that `forced_chain` is null on every pre-registered
criterion. That leaves one trivial explanation open: perhaps the models simply
ignored the checklist. This script closes it. They did not -- an `evidence`
object is present in at least 99.7% of responses from every model -- so the null
result is a null result about the intervention, not about compliance.

## What it reads

`data/forced_chain_raw.jsonl`, the raw response text for the `forced_chain`
schema. `answers_judged.csv` cannot answer this question: it keeps the parsed
`city` / `area` / `place` fields and drops `evidence` entirely, so the only
output the checklist produces is absent from the judged table.

## Parsing caveat, learned the hard way

The first parser stripped a leading ```` ```json ```` fence and nothing else,
and reported 566 of 2799 responses as unparseable -- a 20% failure rate that
would have been a finding if it were true. It was not: taking the substring from
the first `{` to the last `}` recovers 564 of them. Only 2 are genuinely
unparseable. Any "failure rate" produced by a parser must be re-derived with a
second parser before it is reported.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geocontext.config import DATA  # noqa: E402

RAW = DATA / "forced_chain_raw.jsonl"

#: The five checklist categories, matched against whatever key the model chose
#: (models merge the fifth into names like `terrain_climate_vehicles`).
CATS = ["infrastructure", "vegetation", "architecture", "language", "terrain"]

FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.S)


def parse(raw: str):
    """Recover the JSON object from a response, or None.

    Two independent steps, because either alone leaves valid responses behind:
    strip a markdown fence, then take the outermost brace-delimited span.
    """
    t = (raw or "").strip()
    m = FENCE.match(t)
    if m:
        t = m.group(1)
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        t = t[i:j + 1]
    try:
        return json.loads(t)
    except Exception:
        return None


def main():
    rows = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    models = sorted({r["model"] for r in rows})

    n = Counter()
    parsed = Counter()
    with_ev = Counter()
    filled = defaultdict(Counter)
    notvis = defaultdict(Counter)
    unparseable = []

    for r in rows:
        m = r["model"]
        n[m] += 1
        obj = parse(r.get("raw"))
        if obj is None:
            unparseable.append(r)
            continue
        parsed[m] += 1
        ev = obj.get("evidence")
        if not isinstance(ev, dict):
            continue
        with_ev[m] += 1
        for k, v in ev.items():
            cat = next((c for c in CATS if c in k.lower()), None)
            if cat is None:
                continue
            s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if s.strip().lower().startswith("not visible"):
                notvis[m][cat] += 1
            else:
                filled[m][cat] += 1

    sites = {r["path"].replace("\\", "/").split("/")[1] for r in rows}
    print(f"{len(rows)} responses, {len(sites)} sites, {len(models)} models")
    print(f"unparseable: {len(unparseable)}")
    for r in unparseable:
        print(f"    {r['model']}  {r['path']}  context={r.get('context')}")

    print()
    print("evidence object present:")
    for m in models:
        print(f"  {m:<18} {with_ev[m]:>4}/{n[m]:<4} = {with_ev[m] / n[m]:6.1%}")

    print()
    print('share of slots answered "not visible":')
    print("  " + f"{'model':<18}" + "".join(f"{c[:7]:>9}" for c in CATS)
          + f"{'slots':>8}")
    for m in models:
        line = f"  {m:<18}"
        tot = 0
        for c in CATS:
            f_, v_ = filled[m][c], notvis[m][c]
            tot += f_ + v_
            line += f"{(v_ / (f_ + v_) if f_ + v_ else 0):>8.1%} "
        print(line + f"{tot:>7}")

    tf = sum(filled[m][c] for m in models for c in CATS)
    tv = sum(notvis[m][c] for m in models for c in CATS)
    print(f"\n  pooled: {tv}/{tf + tv} = {tv / (tf + tv):.1%}")
    print("\nNote: architecture is never answered \"not visible\" by any model,")
    print("while vegetation is 21-30%. The exit is available and used; the")
    print("checklist simply creates no pressure to decline on architecture.")


if __name__ == "__main__":
    main()
