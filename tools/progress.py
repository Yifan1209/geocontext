"""Check evaluation progress. Safe to run any time, does not disturb a running job.

    python tools/progress.py                      # current progress
    python tools/progress.py --watch              # refresh every 20 seconds

Progress is counted directly from results/e1_raw.jsonl -- the runner flushes
one line per call, so this number is always accurate, more reliable than
watching stdout (which buffers when redirected to a file).
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, runner  # noqa: E402


def snapshot(models, langs, repeats):
    raw = runner.load_raw()
    sample = pd.read_csv(config.DATA / "streetview_meta_selected.csv")
    total = len(sample) * len(models) * len(langs) * repeats

    ok = raw[raw["error"].isna()]
    print(f"{len(raw)} records ({len(ok)} succeeded, {len(raw) - len(ok)} failed)")
    print(f"target combinations: {total}\n")

    # Only count repeats within the current target range. Historical runs that
    # went through more rounds (e.g. an earlier gemini/zh pass with
    # repeats=3) must not all be counted, or completion would read over 100%.
    ok = ok[ok["repeat"] < repeats].drop_duplicates(subset=["path", "model", "lang", "repeat"])

    want = len(sample) * repeats
    rows = []
    for m in models:
        for lg in langs:
            sub = ok[(ok.model == m) & (ok.lang == lg)]
            err = raw[(raw.model == m) & (raw.lang == lg) & raw.error.notna()]
            rows.append(dict(model=m, lang=lg, done=len(sub), target=want,
                             pct=f"{100*len(sub)/want:.0f}%", errors=len(err)))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    done = int(t.done.sum())
    print(f"\noverall progress {done}/{total} = {100*done/total:.1f}%")
    if len(raw) and "ts" in raw:
        idle = time.time() - raw["ts"].max()
        print(f"last write was {idle:.0f}s ago "
              f"{'(looks stopped or finished)' if idle > 120 else '(still running)'}")
    return done, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemini-flash", "claude-haiku-4-5", "claude-opus-5",
                             "qwen3-vl-235b", "qwen3-vl-8b"])
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--watch", action="store_true", help="refresh every 20 seconds")
    args = ap.parse_args()

    while True:
        print("=" * 60)
        done, total = snapshot(args.models, args.langs, args.repeats)
        if not args.watch or done >= total:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
