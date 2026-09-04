"""Pipeline 2, steps 1-3: fetch candidates -> LLM audit -> build the context ladder.

    # see how much this will cost without actually calling anything
    python analysis/build_ladder.py --site paris_marais --dry-run

    # run it for real (already-audited candidates hit the cache, no repeat cost)
    python analysis/build_ladder.py --site paris_marais

    # switch auditor, for a consistency comparison
    python analysis/build_ladder.py --site paris_marais --auditor claude-opus-5

    # an arbitrary coordinate
    python analysis/build_ladder.py --lat 40.7233 --lon -74.0030 --name nyc_soho

Outputs (all under results/ladders/):
    <site>_candidates_<auditor>.csv   every candidate + its audit rating (including rejected ones)
    <site>_ladder_<auditor>.csv       the final ladder (9 cells x per_cell)
    <site>_review_<auditor>.csv       the human review queue, priority-ordered

Audit decisions are also appended row by row to data/audit_decisions.jsonl,
which ships with the dataset.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import ladder, audit, providers, config  # noqa: E402
from geocontext.sites import SITES  # noqa: E402

OUT = config.LADDERS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=list(SITES), help=f"preset site: {list(SITES)}")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name", help="output filename prefix when using a custom coordinate")
    ap.add_argument("--auditor", default="deepseek-v4-pro",
                    help="auditor model. **Must not be an evaluated model** "
                         "(see providers.EVAL_MODELS)")
    ap.add_argument("--prompt-lang", choices=["en", "zh"], default="en",
                    help="language of the audit prompt. English by default -- "
                         "the prompt IS the method, a reviewer needs to be able "
                         "to read it; the Chinese variant's negative examples "
                         "are China-specific and would skew other countries")
    ap.add_argument("--tag",
                    help="auditor suffix used in output filenames, defaults to "
                         "--auditor. The first batch of English-prompt outputs "
                         "used 'deepseek-en'; new sites keep that tag for consistency")
    ap.add_argument("--radius", type=float, default=6.0, help="candidate search radius, km")
    ap.add_argument("--per-cell", type=int, default=4, help="reference points drawn per cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=15, help="candidates sent per audit call")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.site:
        lat, lon, label = SITES[args.site]
        name = args.site
    elif args.lat is not None and args.lon is not None:
        lat, lon, label = args.lat, args.lon, args.name or "custom"
        name = args.name or "custom"
    else:
        ap.error("need --site, or --lat/--lon")

    if args.auditor in providers.EVAL_MODELS:
        ap.error(f"auditor model {args.auditor} is in the evaluated-model list -- "
                 f"that would let the model set its own exam. Pick one outside "
                 f"EVAL_MODELS (a text-only model such as deepseek-v4-pro is recommended)")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"site: {name} ({label})  coordinates {lat}, {lon}", flush=True)
    tag = args.tag or args.auditor
    print(f"auditor: {args.auditor} / prompt={args.prompt_lang}"
          f"  output suffix: {tag}\n")

    # --- step 1: fetch candidates (Wikidata, free, disk-cached) ---
    cand = ladder.wikidata_around(lat, lon, radius_km=args.radius)
    cand = cand[cand.dist_km < args.radius].reset_index(drop=True)
    print(f"[1/3] {len(cand)} Wikidata candidates within {args.radius} km", flush=True)
    if cand.empty:
        print("      no candidates, try a different coordinate or a larger radius"); return

    if args.dry_run:
        done = audit._load()
        key = f"{args.auditor}@{args.prompt_lang}"
        todo = sum(1 for q in cand.qid if (q, key) not in done)
        print(f"[2/3] {todo} to audit ({len(cand)-todo} already cached)", flush=True)
        print(f"      ~{-(-todo//args.batch)} batches; deepseek ~$0.05, opus-5 ~$1.5", flush=True)
        print("[3/3] skipped (dry-run)", flush=True)
        return

    # --- step 2: LLM audit ---
    print(f"[2/3] LLM audit ({args.auditor})", flush=True)
    a = audit.audit(cand, model=args.auditor, batch=args.batch,
                    prompt_lang=args.prompt_lang)
    n_ok = int((a.ok == True).sum())          # noqa: E712
    print(f"      passed {n_ok} / {len(a)}; unaudited {int(a.ok.isna().sum())}", flush=True)

    # --- step 2.5: sensitive-name screening (separate pass, see
    # audit.PROMPT_SENSITIVE_EN) ---
    # Kept separate rather than folded into the `ok` criterion: referenceability
    # is a **measurement**, an exclusion criterion is **dataset policy**; mixed
    # into one field a reader cannot tell "nobody navigates by this" from
    # "unsuitable to include".
    print("[2.5] sensitive-name screening", flush=True)
    a = audit.screen_sensitive(a, model=args.auditor, batch=args.batch)

    f_cand = OUT / f"{name}_candidates_{tag}.csv"
    a.drop(columns=["classes"], errors="ignore").to_csv(f_cand, index=False, encoding="utf-8")
    a.to_pickle(OUT / f"{name}_candidates_{tag}.pkl")   # keep every column, for rebuilding the ladder later

    # --- step 3: build the ladder ---
    print("[3/3] building the ladder", flush=True)
    lad = audit.build_ladder2(a, per_cell=args.per_cell, seed=args.seed, site=name)
    f_lad = OUT / f"{name}_ladder_{tag}.csv"
    lad.to_csv(f_lad, index=False, encoding="utf-8")

    q = audit.review_queue(a)
    f_rev = OUT / f"{name}_review_{tag}.csv"
    q.to_csv(f_rev, index=False, encoding="utf-8")

    print(f"\n{len(lad)} ladder rows:", flush=True)
    print(lad.groupby(["band", "tier"]).size().rename("n").reset_index().to_string(index=False), flush=True)
    print(f"\noutputs:\n  candidates+ratings  {f_cand}\n  ladder              {f_lad}\n  review queue        {f_rev}", flush=True)
    print(f"\nnext: review {f_rev.name} by hand, then run scripts/06_run_eval.py", flush=True)


if __name__ == "__main__":
    main()
