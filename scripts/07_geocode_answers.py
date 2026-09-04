"""Geocode model answers into continuous error_km, for all sites and schemas.

    python scripts/07_geocode_answers.py                      # everything
    python scripts/07_geocode_answers.py --site nyc_soho      # one site only

Why a continuous metric: binary scoring (does the answer contain "Shibuya"?)
is acutely sensitive to whether a neighbourhood has a nameable identity, which
makes it incomparable across sites. New York's model answers tend toward the
coarser "Manhattan", and binary scoring counts "Lower Manhattan" (off by ~1 km)
and "Chicago" (off by 1200 km) as the same kind of wrong.

Switching to continuous distance moved New York's distance effect, on the same
set of images, from p=0.61 (undetectable) to p=1.4e-11 (under the forced schema).

WARNING: **the median is not usable**. Many answers resolve only to city level,
and a city QID's coordinate is fixed, so the distance to the site is pinned at
a constant (1.20 km for New York) -- the median gets nailed to that constant.
Use a regression coefficient or the `<1km` hit rate instead; both are sensitive
to the shape of the distribution.

Produces results/answers_geocoded.pkl, with error_km / resolved_level / resolved_label.
"""
import sys
import json
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, geocode, runner  # noqa: E402
from geocontext.sites import SITES  # noqa: E402

#: Ground-truth coordinates. Derived from geocontext.sites.SITES (the same
#: centre points used to build each ladder and to search Mapillary) rather
#: than duplicated here, so this can never silently drift out of sync with
#: them. `sanjose_rose` is excluded: it is a validation control, not an
#: experimental site with any collected responses.
GT = {k: (lat, lon) for k, (lat, lon, _) in SITES.items() if k != "sanjose_rose"}

OUT = config.DATA / "answers_judged.csv"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)
_JSON = re.compile(r"\{.*\}", re.S)


def parse(raw):
    if not raw:
        return {}
    s = _FENCE.sub("", str(raw).strip())
    m = _JSON.search(s)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", help="process only this site")
    ap.add_argument("--schemas", nargs="+", default=None, help="process only these schemas")
    args = ap.parse_args()

    rows = []
    for line in runner.RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        site = geocode.site_of(r.get("path", ""))
        if site not in GT:
            continue
        if args.site and site != args.site:
            continue
        sch = r.get("schema", "v1")
        if args.schemas and sch not in args.schemas:
            continue
        d = parse(r["raw"])
        # the forced schemas call the finest level `building`; v1 calls it `place`
        fine = d.get("building") if sch.startswith("forced") else d.get("place")
        rows.append(dict(
            site=site, schema=sch, model=r["model"], lang=r["lang"],
            context=r.get("context"), path=r["path"],
            city=d.get("city"), area=d.get("area"), place=fine,
            conf_area=d.get("confidence_area", d.get("confidence")),
            conf_fine=d.get("confidence_building", d.get("confidence")),
            clues=d.get("clues")))
    df = pd.DataFrame(rows)
    print(f"{len(df)} rows to process:")
    print(df.groupby(["site", "schema"]).size().rename("n").reset_index().to_string(index=False))

    # Reuse anything already computed (geocoding itself is disk-cached too, but
    # skipping the loop entirely is faster still).
    if OUT.exists():
        prev = pd.read_pickle(OUT)
        key = ["site", "schema", "model", "lang", "context", "path"]
        done = set(map(tuple, prev[key].values.tolist()))
        mask = ~df.set_index(key).index.isin(done)
        todo = df[mask.values] if hasattr(mask, "values") else df[mask]
        print(f"\n{len(prev)} already done, {len(todo)} new this run")
    else:
        prev, todo = None, df

    if len(todo):
        # Checkpoint every 500 rows. This step runs thousands of rows x network
        # lookups, and a background task can be killed along with its parent
        # CLI process (one run was lost at 1400/4518 when the terminal
        # disconnected). Wikidata results are disk-cached so a rerun is not
        # slow, but there is no reason to redo the work.
        todo = geocode.add_error_km(todo, GT,
                                    checkpoint=OUT.with_suffix(".partial.pkl"))
        df = pd.concat([prev, todo], ignore_index=True) if prev is not None else todo
    else:
        df = prev

    df.to_pickle(OUT)
    print(f"\nresolved {df.error_km.notna().sum()}/{len(df)} = {df.error_km.notna().mean():.1%}")
    print("levels:", dict(df.resolved_level.value_counts(dropna=False)))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
