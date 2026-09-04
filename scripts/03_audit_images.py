"""Image quality audit: answer leakage, scene type, viewpoint, usability.

    python scripts/03_audit_images.py --source streetview          # Mapillary images
    python scripts/03_audit_images.py --source mmsvpr              # MMS-VPR sample
    python scripts/03_audit_images.py --source streetview --model kimi-vision

Two levels of judgement:

- **answer_overlay is a hard exclusion.** Text burned into the image (an app
  screenshot, a news caption, a watermark) that names the place makes the task
  fail outright -- the model would be reading text, not the street scene.
  Found in MMS-VPR: 4 of 80 images (5%), and 66% carried some kind of watermark.

- **Every other dimension is kept but flagged.** Window reflections, blur, poor
  framing are noise sources; excluding them just shrinks the sample. Keeping
  and flagging them instead lets "does image quality affect the anchoring
  effect" become its own robustness check.

The line that must stay clear: **text present in the scene is not leakage.**
A shop sign or a street sign is a legitimate geographic cue -- that is what
geolocation runs on. Text in an app's UI bar naming the exact location is
metadata leakage instead.

The auditor model must not be one of the evaluated models
(`providers.EVAL_MODELS` blocks this). Before putting a new auditor model into
service, validate it on an image with a known answer -- measured:
`mistral-small-3.2-24b` hallucinated a Taipei address on a photo taken
somewhere else entirely.

## 2026-09-02 addition: `tourist_infrastructure`

The third batch of sites uses `select_sites.py`, picking sites via a map API,
and the site itself can be a popular landmark (Anne Frank House, Brandenburg
Gate and similar, with sitelinks in the tens). Even when the photo is a nearby
street scene that does not show the landmark itself, imagery right next to a
top attraction can still carry **visual tells of tourist density**: souvenir
shop signage, queues, tour guides holding little flags, dense multilingual
directional signage. That is enough for a model to guess "there is a major
attraction nearby" without ever seeing the landmark.

Unlike `answer_overlay`, this is not a task failure where the model reads text
instead of the street scene -- the site is just **inherently** easier to
recognise, and how easy a site is to recognise is exactly what this benchmark
measures (that is what the no-context baseline is for). So this field is
**flagged only, never a hard exclusion**, the same treatment as
`through_glass` / `quality`: kept for robustness checks, and used to monitor
whether the newer sites are systematically easier than the 19 hand-picked
ones, which would push too many sites into the "legible" regime and starve the
"illegible" regime of samples.
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, providers  # noqa: E402

PROMPT = """Audit this street-level photo for a geolocation benchmark.

Report on the IMAGE ITSELF, not on where you think it was taken.

1. `answer_overlay`: is there text OVERLAID on the photo (app UI, news subtitle,
   watermark, caption bar) that names a place, street, district, city or landmark?
   Text physically present in the scene — shop signs, street signs, banners,
   billboards — does NOT count. Those are legitimate visual cues.
2. `overlay_text`: any overlaid text you can read (empty string if none).
3. `scene_type`: one of "street", "indoor", "vehicle_interior", "closeup", "other".
   "street" = an outdoor street-level view a pedestrian or car would see.
   "vehicle_interior" = shot from inside a vehicle with dashboard/pillars visible.
   "closeup" = a wall, sign or object filling the frame with no street context.
4. `through_glass`: shot through a windscreen or window (reflections, glare, wipers)?
5. `tourist_infrastructure`: does the scene show visual signs of being right next
   to a major tourist attraction, EVEN IF the attraction itself is not in frame?
   Look for: souvenir/gift shops, ticket booths or entrance queues, tour groups
   or guides with flags/umbrellas, dense multilingual directional signage aimed
   at visitors. Ordinary shops, ordinary street signs, and ordinary pedestrians
   do NOT count.
6. `quality`: 0-5 overall usability for geolocation.
   5 = sharp, well framed, plenty of context; 3 = usable; 0 = unusable.
7. `usable`: false if answer_overlay is true, or scene_type is not "street",
   or quality <= 1. (`tourist_infrastructure` does NOT affect `usable` --- it is
   recorded for robustness analysis, not filtered out.)

Output ONLY JSON:
{"answer_overlay": true/false, "overlay_text": "", "scene_type": "street",
 "through_glass": false, "tourist_infrastructure": false, "quality": 4,
 "usable": true, "why": "under 15 words"}"""

FIELDS = ["answer_overlay", "overlay_text", "scene_type",
          "through_glass", "tourist_infrastructure", "quality", "usable", "why"]


def load_images(source: str, meta_file: str = "streetview_meta.csv") -> pd.DataFrame:
    """Returns a DataFrame with at least `key` (a unique identifier) and `abspath`.

    `meta_file` selects which sample table to audit -- **this is how audit order
    is controlled**, there is no separate queue system. Images actually used in
    the released experiment (streetview_meta_selected.csv) are audited first;
    the rest follow later. `key` is a relative path rather than image_id: not
    every sample table has an image_id column, but the path is present and
    unique in all of them.
    """
    if source == "streetview":
        meta = pd.read_csv(config.DATA / meta_file)
        rel = meta.path.astype(str).str.replace("\\", "/", regex=False)
        return pd.DataFrame({"key": rel, "site": meta.site,
                             "abspath": [str(config.DATA / p) for p in meta.path]})
    if source == "mmsvpr":
        s = pd.read_csv(config.DATA / "e1_sample.csv")
        return pd.DataFrame({"key": s.path, "site": s.location,
                             "abspath": [str(config.MMSVPR / p) for p in s.path]})
    raise ValueError(f"unknown source: {source}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["streetview", "mmsvpr"], default="streetview")
    ap.add_argument("--model", default="deepseek-vision",
                    help="vision model to audit with. **Must not be an evaluated model**")
    ap.add_argument("--meta", default="streetview_meta.csv",
                    help="which sample table to use. Use "
                         "streetview_meta_selected.csv for the 48 images actually "
                         "used in the released experiment; "
                         "streetview_meta_candidates.csv for all 159")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    if args.model in providers.EVAL_MODELS:
        ap.error(f"{args.model} is in the evaluated-model list and cannot be used as an auditor")

    stem = args.meta.replace("streetview_meta", "sv").replace(".csv", "")
    out = config.RESULTS / f"image_audit_{stem}_{args.model}.csv"
    # The **single source of truth** for decisions is this append-only jsonl,
    # shared across sample tables. An earlier version used the CSV as both
    # cache and output, so switching --meta overwrote and lost the previous
    # batch of records wholesale (four excluded images' audit results were
    # lost this way, surviving only in a log).
    log = config.DATA / f"image_audit_{args.model}.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r

    imgs = load_images(args.source, args.meta)
    todo = [r for r in imgs.itertuples() if str(r.key) not in done]
    print(f"{len(imgs)} images, {len(imgs)-len(todo)} already audited, "
          f"{len(todo)} to go", flush=True)

    client = providers.REGISTRY[args.model]() if todo else None
    rows = []
    for i, r in enumerate(imgs.itertuples(), 1):
        if str(r.key) in done:
            rows.append(done[str(r.key)])
            continue
        rec = {"key": str(r.key), "site": r.site, "abspath": r.abspath}
        try:
            raw = client.ask(r.abspath, PROMPT)["raw"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            d = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            missing = [f for f in ("answer_overlay", "scene_type", "usable") if f not in d]
            if missing:
                raise ValueError(f"response missing fields {missing}")
            for f in FIELDS:
                rec[f] = d.get(f)
        except Exception as e:
            for f in FIELDS:
                rec[f] = None
            rec["why"] = f"ERR {type(e).__name__}: {str(e)[:50]}"
        rows.append(rec)
        with open(log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if i % 10 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8")

    n = len(df)
    print(f"\n{n} images total")
    print(f"  **answer leakage (hard exclusion)** {int(df.answer_overlay.sum() or 0)}  "
          f"({(df.answer_overlay.mean() or 0):.1%})")
    print(f"  usable              {int(df.usable.sum() or 0)}  ({(df.usable.mean() or 0):.1%})")
    print(f"  shot through glass  {int(df.through_glass.sum() or 0)}  "
          f"({(df.through_glass.mean() or 0):.1%})")
    print(f"  mean quality score  {pd.to_numeric(df.quality, errors='coerce').mean():.2f}")
    print("\nscene types:", dict(df.scene_type.value_counts()))
    if "site" in df:
        print("\nusable count per site:")
        print(df.groupby("site").agg(n=("usable", "size"),
                                     usable=("usable", "sum"),
                                     leaked=("answer_overlay", "sum")).to_string())
    leak = df[df.answer_overlay == True]        # noqa: E712
    if len(leak):
        print("\nleak examples:")
        for _, r in leak.head(6).iterrows():
            print(f"  {str(r.site):10} {str(r.overlay_text)[:56]}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
