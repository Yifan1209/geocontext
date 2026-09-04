"""Turn the picker's JSON export into the final sample table.

    python scripts/04_select_images.py ~/Downloads/image_selection.json --per-site 3

After a human has checked boxes in results/picker.html, this step:
  1. reads the checked selection
  2. randomly samples `--per-site` images per site (fixed seed, reproducible)
  3. writes data/streetview_meta_selected.csv for scripts/06_run_eval.py

NOTE: no original image is touched -- images that were not selected stay on
disk, they simply do not enter the sample table.
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config  # noqa: E402

OUT = config.DATA / "streetview_meta_selected.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="image_selection.json exported by the picker")
    ap.add_argument("--per-site", type=int, default=3, help="images to take per site")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclusions", default="image_audit.csv",
                    help="unified audit table (under data/), one row per image. "
                         "Only rows with human_verdict=exclude actually drop an "
                         "image -- the ai_* columns are advisory, measured "
                         "precision is only ~50%%")
    ap.add_argument("--exclude-audit", action="append", default=[],
                    metavar="CSV",
                    help="an image-audit results CSV; drops images with "
                         "answer_overlay=True or usable=False. Can be given "
                         "more than once.\n"
                         "Sample -> audit -> resample is iterative: newly "
                         "swapped-in images need auditing too")
    args = ap.parse_args()

    keep = json.loads(Path(args.json).read_text(encoding="utf-8"))["keep"]
    df = pd.DataFrame({"path": keep})
    df["site"] = df.path.str.split("/").str[1]
    print(f"{len(df)} images kept, covering {df.site.nunique()} sites")

    # Human verdicts: **the only source that actually drops an image**.
    # They live in the unified audit table alongside the model's verdicts, one
    # row per image; only `human_verdict == "exclude"` removes anything.
    ex = config.DATA / args.exclusions
    if ex.exists():
        e = pd.read_csv(ex)
        drop = e[e.human_verdict == "exclude"]
        n0 = len(df)
        df = df[~df.path.isin(set(drop.image.astype(str)))]
        print(f"  dropped {n0 - len(df)} via {ex.name} (human_verdict=exclude):")
        for r in drop.itertuples():
            print(f"    [{r.human_reason}] {str(r.image).split('/')[-1]}")

    # NOTE: dropping images automatically based on AI-audit results is **not
    # recommended**, see the module docstring. Measured precision is only ~50%.
    for f in args.exclude_audit:
        a = pd.read_csv(f)
        bad = set(a.loc[(a.answer_overlay == True) | (a.usable != True),   # noqa: E712
                        "key"].astype(str))
        if not bad:
            continue
        n0 = len(df)
        df = df[~df.path.isin(bad)]
        print(f"  dropped {n0-len(df)} via {Path(f).name}:")
        for r in a[a.key.isin(bad)].itertuples():
            flag = "answer leak" if r.answer_overlay == True else "quality/scene"   # noqa: E712
            print(f"    [{flag}] {r.key}  -- {r.why}")
    print()

    counts = df.groupby("site").size().rename("checked").to_frame()
    counts["taken"] = counts.checked.clip(upper=args.per_site)
    thin = counts[counts.checked < args.per_site]
    print(counts.to_string())
    if len(thin):
        print(f"\nWARNING: the following sites have fewer than {args.per_site} "
              f"checked images and all of them will be used:")
        print("   " + ", ".join(f"{i}({r.checked})" for i, r in thin.iterrows()))

    # Not using groupby.apply -- pandas 2.2's include_groups=False drops the
    # site column. Sampling per group and concatenating is more explicit.
    samp = pd.concat([g.sample(min(args.per_site, len(g)), random_state=args.seed)
                      for _, g in df.groupby("site")], ignore_index=True)
    samp["path"] = samp.path.str.replace("/", "\\", regex=False)   # the runner expects Windows separators
    samp["location"] = samp.site
    samp["condition"] = "day"
    samp["hour"] = 12
    samp[["site", "location", "condition", "hour", "path"]].to_csv(
        OUT, index=False, encoding="utf-8-sig")
    print(f"\nfinal sample: {len(samp)} images -> {OUT}")


if __name__ == "__main__":
    main()
