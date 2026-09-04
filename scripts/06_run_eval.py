"""GeoHint evaluation runner: context-anchoring experiment.

    # hand-picked baseline (4 hand-chosen reference points)
    python scripts/06_run_eval.py --contexts taikooli tianfu wuhou shieryuan

    # Pipeline 2 automatic ladder (the main experiment)
    python scripts/06_run_eval.py --ladder london_covent --limit-images 20

Design: the image is held fixed and only the context varies. Every context is
**true**, only decreasing in precision. The question is whether the model
switches to naming the most iconic building near the reference point, or keeps
reading the image.

context=none is the baseline and is byte-identical to the prompt used in the
existing 800-plus baseline calls, so it is reused rather than rerun.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, prompts, providers, runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["gemini-flash", "claude-haiku-4-5", "claude-opus-5",
                             "qwen3-vl-235b", "qwen3-vl-8b"])
    ap.add_argument("--langs", nargs="+", default=["en"],
                    help="only 'en' ships in this release; see prompts.py")
    ap.add_argument("--ladder", metavar="SITE",
                    help="use the Pipeline-2 auto-generated ladder for this site, "
                         "e.g. --ladder london_covent. If omitted, uses the 4 "
                         "hand-picked baseline reference points")
    ap.add_argument("--auditor", default="deepseek-en",
                    help="which auditor model generated the ladder (locates the CSV file)")
    ap.add_argument("--contexts", nargs="+", default=None,
                    help="run only these context keys; default is all of them (baseline excluded)")
    ap.add_argument("--source", choices=["mmsvpr", "streetview"], default="streetview",
                    help="image source. streetview = fetched from Mapillary "
                         "(what the released benchmark uses)")
    ap.add_argument("--site", help="use only one site's images (with --source streetview)")
    ap.add_argument("--meta", default="streetview_meta_selected.csv",
                    help="street-view sample table (under data/). The released "
                         "experiment uses streetview_meta_selected.csv -- the 48 "
                         "images left after manual picking, auditing all 159 "
                         "candidates, and the explicit exclusion list")
    ap.add_argument("--schema", default="forced",
                    choices=["v1", "forced", "forced_hedge", "forced_warn", "forced_chain"],
                    help="output format / mitigation level. v1 permits 'unknown' "
                         "(legacy data); forced requires a committed answer; "
                         "forced_hedge adds 'the location may be approximate'; "
                         "forced_warn adds 'the location may be inaccurate, trust "
                         "the image'; forced_chain adds a GeoGuessr-style "
                         "structured evidence checklist")
    ap.add_argument("--serial", action="store_true",
                    help="run serially. Parallel is the default -- calls are "
                         "almost entirely network wait, so serial is 5-10x slower")
    ap.add_argument("--condition", choices=["day", "night", "both"], default="both",
                    help="restrict to daytime/night images. Street-view sources "
                         "(Mapillary etc.) are usually daytime-only; fix this to "
                         "'day' for cross-city comparability")
    ap.add_argument("--limit-images", type=int, default=None, metavar="N",
                    help="use only N images per run (stratified random sample "
                         "by site, at most 1 per site). If omitted, uses every "
                         "image in the sample table")
    ap.add_argument("--image-seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== key check ===", flush=True)
    config.check_keys()

    if args.ladder:
        prompts.use_ladder(args.ladder, args.auditor)
        print(f"loaded automatic ladder: {args.ladder} / {args.auditor}", flush=True)
    contexts = args.contexts or [k for k in prompts.CONTEXTS if k != "none"]
    missing = [c for c in contexts if c not in prompts.CONTEXTS]
    if missing:
        ap.error(f"unknown context key(s): {missing}\navailable: {list(prompts.CONTEXTS)}")

    print(f"\n=== {len(contexts)} context conditions ===", flush=True)
    for c in contexts:
        m = prompts.CONTEXTS[c]
        d = f"{m['dist_km']:.2f} km" if m.get("dist_km") is not None else "--"
        print(f"  {c:32} {d:>9}  {m.get('note', '')}", flush=True)
        print(f"      en: {m['en']}", flush=True)

    if args.source == "streetview":
        m = pd.read_csv(config.DATA / args.meta)
        if args.site:
            m = m[m.site == args.site]
        # Images with leaked answers (an overlay printing a place name or GPS
        # coordinate) must be excluded -- the model would be reading text, not
        # the street scene. The image itself is kept on disk, just not entered
        # into evaluation.
        if "answer_overlay" in m:
            n0 = len(m)
            m = m[m.answer_overlay != True]      # noqa: E712
            if n0 != len(m):
                print(f"excluded {n0-len(m)} answer-leaking images", flush=True)
        # Normalise to the column names the runner expects. Mapillary has no
        # site index, so `site` doubles as `location`; there is no day/night
        # label either (nearly everything is daytime), so condition is fixed
        # to "day".
        sample = pd.DataFrame({"location": m.site, "condition": "day",
                               "hour": 12, "path": m.path})
        print(f"street-view source: {len(sample)} images, sites "
              f"{sorted(sample.location.unique())}", flush=True)
    else:
        sample = pd.read_csv(config.DATA / "e1_sample.csv")
    if args.condition != "both":
        sample = sample[sample.condition == args.condition].reset_index(drop=True)
        print(f"restricted to {args.condition}: {len(sample)} images", flush=True)
    if args.limit_images:
        n_loc = sample.location.nunique()
        if n_loc > 1:
            # MMS-VPR: 208 sites, several images each. Stratify by site, at
            # most 1 per site -- taking the first N rows would only cover a
            # handful of sites (the sample table is sorted by site), which
            # confounds the site effect with the context effect.
            pool = (sample.groupby("location", group_keys=False)
                          .sample(1, random_state=args.image_seed))
        else:
            # Street view: several images under one site, which should just be
            # sampled directly. An earlier version stratified by site
            # unconditionally, which collapsed this case to a single image.
            pool = sample
        n = min(args.limit_images, len(pool))
        sample = pool.sample(n, random_state=args.image_seed).reset_index(drop=True)
        print(f"sampled {len(sample)} images, covering {sample.location.nunique()} sites",
              flush=True)
    todo, done = runner.plan(sample, args.models, args.langs,
                             args.repeats, contexts=contexts, schema=args.schema)
    total = len(sample) * len(args.models) * len(args.langs) * args.repeats * len(contexts)
    print(f"\n{len(sample)} images x {len(args.models)} models x {len(args.langs)} "
          f"languages x {len(contexts)} contexts = {total}", flush=True)
    print(f"done {len(done)}, to run {len(todo)}\n", flush=True)
    if args.dry_run:
        return

    fn = runner.run if args.serial else runner.run_parallel
    kw = dict(repeats=args.repeats, contexts=contexts)
    if not args.serial:
        kw["schema"] = args.schema
    if args.serial:
        kw["sleep"] = args.sleep
    stats = fn(sample, args.models, args.langs, **kw)
    print(f"\n=== done ===\n{stats}", flush=True)


if __name__ == "__main__":
    main()
