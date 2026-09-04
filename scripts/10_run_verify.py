"""Run the GeoVerify location-claim verification pilot.

    python scripts/10_run_verify.py --models gemini-flash claude-haiku-4-5 > results/_verify.log 2>&1

Reads `data/verify_trials.csv` (see `09_build_verify_trials.py`), asks each
model one yes/no question per trial, appends one JSON line per call to
`results/verify_raw.jsonl`. Re-running skips whatever is already there.

## Two deliberate prompt design choices

1. **Neutral wording, not pushed toward either answer.** Writing "answer no if
   unsure" would depress both the false-alarm rate and the hit rate at once,
   baking response bias into the data. This forces a binary choice instead and
   lets uncertainty be carried by `confidence` -- criterion c is a quantity to
   **measure**, not to pre-determine.

2. **The candidate is given by name only, no coordinates, city, or
   description.** Giving them away would hand over half the answer, and the
   deployed scenario (an app showing "your destination: X") only has a name too.

## Why not the existing runner.run_parallel

That runner's plan/cache key is (path, model, lang, context, schema), designed
for the open-ended localisation task. Here the key is (trial_id, model), a
different shape; forcing it through would treat "same image, different
candidate" as the same call.
"""
import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, providers  # noqa: E402

RAW = config.DATA / "verify_raw.jsonl"

#: Uniform grey field used for the no-image control. A grey image rather than
#: no image at all, so that **the request shape stays constant** -- only the
#: visual evidence is removed.
BLANK = config.DATA / "blank_gray.jpg"

PROMPT = """You are checking whether a photo was taken at a claimed location.

CLAIM: This photo was taken within 150 metres of "{cand}".

Look at the image and decide whether the claim is true.

Output only JSON, no other text:
{{
  "verdict": "yes" or "no",
  "confidence": a number between 0.0 and 1.0,
  "evidence": ["specific things you actually see in the image that led to your verdict"]
}}

You must choose "yes" or "no". Put your uncertainty in `confidence`, not in the
verdict. In `evidence`, list only what you can actually see in this image."""

#: Concurrency, using the same caps as scripts/06_run_eval.py so no single
#: vendor's API gets hammered.
LIMITS = {"claude-opus-5": 4, "claude-haiku-4-5": 8, "gemini-flash": 8,
          "qwen3-vl-235b": 6, "qwen3-vl-8b": 6}


def ensure_blank():
    if BLANK.exists():
        return
    from PIL import Image
    Image.new("RGB", (768, 512), (128, 128, 128)).save(BLANK, quality=90)
    print(f"generated the no-image control's grey field at {BLANK}")


def done_keys() -> set:
    if not RAW.exists():
        return set()
    keys = set()
    for line in RAW.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("error") is None:
            keys.add((d["trial_id"], d["model"]))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default="data/verify_trials.csv")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, help="run only the first N trials (for a smoke test)")
    args = ap.parse_args()

    ensure_blank()
    t = pd.read_csv(args.trials)
    t["trial_id"] = [f"{r.image_id}|{r.arm}|{r.cand}" for r in t.itertuples()]
    dup = t.trial_id.duplicated().sum()
    if dup:
        # The same image + same candidate should appear at most once in the
        # main arm; a duplicate means decoy sampling collided.
        t = t.drop_duplicates(subset="trial_id").reset_index(drop=True)
        print(f"dropped {dup} duplicate trials")
    if args.limit:
        t = t.head(args.limit)

    have = done_keys()
    todo = [(r, m) for r in t.itertuples() for m in args.models
            if (r.trial_id, m) not in have]
    print(f"{len(t)} trials x {len(args.models)} models = {len(t)*len(args.models)}; "
          f"{len(have)} done; {len(todo)} to run", flush=True)
    if not todo:
        return

    clients, clock, wlock = {}, threading.Lock(), threading.Lock()
    stats = {"n": 0, "err": 0, "tin": 0, "tout": 0}
    t0 = time.time()
    fout = RAW.open("a", encoding="utf-8")

    def client(name):
        with clock:
            if name not in clients:
                clients[name] = providers.REGISTRY[name]()
            return clients[name]

    def work(item):
        r, mname = item
        rec = dict(trial_id=r.trial_id, image_id=r.image_id, site=r.site,
                   arm=r.arm, band=r.band, cand=r.cand, truth=r.truth,
                   d_km=float(r.d_km), fam=float(r.fam), model=mname,
                   ts=time.time())
        img = BLANK if r.arm == "ctrl_noimage" else config.DATA / r.path
        delay = 2.0
        for attempt in range(4):
            try:
                out = client(mname).ask(img, PROMPT.format(cand=r.cand))
                rec.update(raw=out["raw"], usage=out["usage"], error=None)
                break
            except Exception as e:
                transient = any(c in str(e) for c in
                                ("503", "429", "529", "overloaded",
                                 "high demand", "Timeout"))
                if transient and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                rec.update(raw=None, usage=None,
                           error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc()[-600:])
                break
        with wlock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            stats["n"] += 1
            if rec["error"]:
                stats["err"] += 1
            else:
                stats["tin"] += (rec["usage"] or {}).get("input_tokens") or 0
                stats["tout"] += (rec["usage"] or {}).get("output_tokens") or 0
            if stats["n"] % 50 == 0:
                el = time.time() - t0
                rate = stats["n"] / el
                left = (len(todo) - stats["n"]) / rate / 60 if rate else 0
                print(f"  {stats['n']}/{len(todo)}  errors {stats['err']}  "
                        f"{rate:.1f}/s  ~{left:.0f}min left", flush=True)

    # One pool per model: vendors rate-limit differently, and pooling them
    # together lets the slowest vendor stall everyone else.
    with ThreadPoolExecutor(max_workers=sum(LIMITS.get(m, 4) for m in args.models)) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for f in as_completed(futs):
            f.result()
    fout.close()

    el = time.time() - t0
    print(f"\ndone: {stats['n']} calls, {stats['err']} errors, {el/60:.1f} min")
    print(f"tokens in {stats['tin']:,} / out {stats['tout']:,}")
    print(f"written to {RAW}")


if __name__ == "__main__":
    main()
