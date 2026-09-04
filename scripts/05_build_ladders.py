"""Build ladders for many sites in ONE pooled parallel pass.

    python scripts/05_build_ladders.py --workers 16

## Why this exists

Running sites one at a time (`analysis/build_ladder.py`), 6-way concurrent
within each site, means the auditor deepseek-v4-pro (a reasoning model, ~200s
per batch) takes about 5 minutes per site -- roughly 75 minutes for 15 sites.

But candidates across sites in the same city **overlap heavily** (two sites in
one city share most of their Wikidata candidates), and running site by site
never captures that cross-site parallelism.

This script instead:

  1. fetches candidates for every site up front, **de-duplicated by qid**
  2. sends every batch into **one shared thread pool** (workers is tunable)
  3. builds each site's ladder from the cache afterwards

De-duplication plus global parallelism measured at compressing 15 sites from
~75 minutes down to roughly 10.

NOTE: uses **one process, multiple threads**. Several processes appending to
the same audit_decisions.jsonl interleave and corrupt the file on Windows.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import audit, config, ladder  # noqa: E402
from geocontext.sites import SITES, BATCH3  # noqa: E402

OUT = config.LADDERS

#: Batch 2, the expansion sites (batch 1 already has ladders built).
BATCH2 = ["paris_montmartre", "paris_bastille", "paris_cite", "paris_canal",
          "london_shoreditch", "london_covent", "london_nottinghill",
          "barcelona_gracia", "barcelona_born",
          "sf_mission", "sf_northbeach", "sf_hayes",
          "cdmx_roma", "cdmx_condesa",
          "tokyo_shimokita", "tokyo_yanaka"]

#: Batch 3, the 29-city / 87-site expansion from scripts/00_select_sites.py.
#: `python scripts/05_build_ladders.py --sites $(python -c "from geocontext.sites import BATCH3; print(' '.join(BATCH3))")`
#: or just pass --sites list(BATCH3) programmatically -- kept as a plain name
#: list here (not hardcoded again) so it can never drift from sites.py.
BATCH3_SITES = list(BATCH3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=BATCH2)
    ap.add_argument("--auditor", default="deepseek-v4-pro")
    ap.add_argument("--tag", default="deepseek-en")
    ap.add_argument("--prompt-lang", default="en", choices=["en", "zh"])
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16,
                    help="global concurrency. A single auditor batch takes ~200s, "
                         "so concurrency is the only lever for speed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip sites that already have a ladder CSV")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sites = [s for s in args.sites if s in SITES]
    if args.skip_done:
        sites = [s for s in sites
                 if not (OUT / f"{s}_ladder_{args.tag}.csv").exists()]
    print(f"{len(sites)} sites: {', '.join(sites)}\n", flush=True)
    if not sites:
        print("nothing to run"); return

    # --- 1. fetch candidates for every site, de-duplicated by qid ---
    per_site, frames = {}, []
    for s in sites:
        lat, lon, label = SITES[s]
        c = ladder.wikidata_around(lat, lon, radius_km=args.radius)
        c = c[c.dist_km < args.radius].reset_index(drop=True)
        per_site[s] = c
        frames.append(c)
        print(f"  {s:20} {len(c)} candidates", flush=True)
    allc = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset="qid", keep="first").reset_index(drop=True)
    print(f"\n{len(allc)} unique candidates after de-duplication "
          f"(raw {sum(len(c) for c in frames)}, saved "
          f"{sum(len(c) for c in frames) - len(allc)})", flush=True)

    # --- 2. audit the whole pool in one pass ---
    print(f"\n[1/3] LLM audit ({args.auditor}, {args.workers}-way concurrent)", flush=True)
    allc = audit.audit(allc, model=args.auditor, batch=args.batch,
                       prompt_lang=args.prompt_lang, workers=args.workers)
    print(f"      passed {int((allc.ok == True).sum())} / {len(allc)}"  # noqa: E712
          f"; unaudited {int(allc.ok.isna().sum())}", flush=True)

    print(f"\n[2/3] sensitive-name screening ({args.workers}-way concurrent)", flush=True)
    allc = audit.screen_sensitive(allc, model=args.auditor, batch=args.batch,
                                 workers=args.workers)

    # --- 3. build each site's ladder from the cache ---
    print("\n[3/3] building each site's ladder", flush=True)
    cols = [c for c in ("ok", "familiarity", "name_zh", "name_en", "why",
                        "sensitive", "sensitive_why") if c in allc.columns]
    lookup = allc.set_index("qid")[cols]
    summary = []
    for s in sites:
        c = per_site[s].join(lookup, on="qid")
        c.to_csv(OUT / f"{s}_candidates_{args.tag}.csv", index=False,
                 encoding="utf-8")
        c.to_pickle(OUT / f"{s}_candidates_{args.tag}.pkl")
        lad = audit.build_ladder2(c, per_cell=args.per_cell, seed=args.seed,
                                 verbose=False, site=s)
        lad.to_csv(OUT / f"{s}_ladder_{args.tag}.csv", index=False,
                   encoding="utf-8")
        audit.review_queue(c).to_csv(OUT / f"{s}_review_{args.tag}.csv",
                                     index=False, encoding="utf-8")
        summary.append(dict(site=s, candidates=len(c), ladder=len(lad)))
        print(f"  {s:20} ladder: {len(lad):2} rows", flush=True)

    print("\n=== summary ===", flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
