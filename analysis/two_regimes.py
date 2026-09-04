"""Test the two-regime reading of the data.

    python analysis/two_regimes.py

The reading being tested:

  Regime 1  the image carries little distinguishable information -> accuracy
            rises when a reference point is supplied, and falls as it gets
            farther away. **Plausibly the model is not reading the image at
            all, just guessing near the reference point.**
  Regime 2  the image is informative and the baseline is already accurate ->
            supplying a reference point is interference instead.

Regime 2 already has direct evidence (the legibility-tier x distance-band
table). **Regime 1's "not reading the image, just guessing" is a stronger
claim and needs its own check.**

## Criterion: error versus the reference-point distance itself

If the model is purely reciting the reference point, its error should equal
the distance from the reference point to the true location, i.e.
`error_km / ref_km ~ 1`. If it is using image evidence, the ratio should be
significantly below 1.

WARNING: this ratio was used once before and **was invalid then** -- error_km
was pinned to a constant by geocoding resolution (a city-level fallback), so
the ratio was `constant / ref_km`, mechanically decreasing with ref_km. The
city-level fallback rate has since dropped from 51% to 9%, and the
computation below **only uses non-city-level resolutions**, avoiding that trap.

## Three complementary criteria

1. The distribution of `error_km / ref_km`: ~1 means the model recites the
   reference point
2. **Echo rate**: the fraction of responses that name the reference point directly
3. **A direct control on image legibility**: the same images' behaviour under baseline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts  # noqa: E402

pd.set_option("display.width", 240)
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]
#: Every shipped ladder carries the same auditor tag, so the per-site override
#: table that used to live here is gone; `load_ladder`'s default is enough.
AUDITOR_TAG = "deepseek-en"


def gm(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(10 ** np.log10(x).mean()) if len(x) else np.nan


def echoed(area, place, zh, en):
    t = f"{area} {place}".lower()
    for n in (zh, en):
        s = str(n).strip().lower()
        if not s or s == "nan":
            continue
        key = s[:4] if any("一" <= c <= "鿿" for c in s) else s.split()[0]
        if len(key) >= 3 and key in t:
            return True
    return False


def load():
    df = pd.read_csv(config.DATA / "answers_judged.csv")
    meta = []
    for site in sorted(df.site.dropna().unique()):
        try:
            lad = prompts.load_ladder(site, AUDITOR_TAG,
                                      include_baseline=False)
        except FileNotFoundError:
            continue
        for k, v in lad.items():
            meta.append(dict(site=site, context=k, ref_km=v["dist_km"],
                             band=v["band"], tier=v["tier"],
                             ref_zh=v["name_zh"], ref_en=v["name_en"]))
    df = df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")
    df["cond"] = np.where(df.context == "none", "baseline", df.band)
    df["hit"] = geocode.hit(df)
    # A record has context but its qid can no longer be found in the ladder --
    # usually because it was later removed from the ladder (for instance the 5
    # "self-leaking" reference points removed on 2026-09-01: candidates whose
    # name was the site's own place name, e.g. london_nottinghill had a
    # candidate literally called "Notting Hill"). `cond` is NaN here, but
    # `cond != "baseline"` evaluates to True for NaN, so without dropping these
    # explicitly they quietly leak into "overall/pooled" aggregations that do
    # not group by cond (e.g. chain_results.py's strict pairing test), while
    # never showing up in any table grouped by band -- the kind of error that
    # only surfaces when a total does not add up.
    df = df[df.context.eq("none") | df.cond.notna()]
    return df


def main():
    d = load()
    f = d[d.schema == "forced"].copy()
    base = f[f.cond == "baseline"].groupby("site").hit.mean() * 100
    f["baseline_hit"] = f.site.map(base)
    f["regime"] = np.where(f.baseline_hit < 50, "regime1_illegible", "regime2_legible")

    print("Split criterion: the site's hit rate at context=none; <50% -> regime 1")
    print(f"Regime 1 sites: {sorted(base[base < 50].index)}")
    print(f"Regime 2 sites: {sorted(base[base >= 50].index)}\n")

    g = f[(f.cond != "baseline") & f.ref_km.notna()].copy()
    # Ratio computed only on non-city-level resolutions -- city-level error is
    # a constant, which would mechanically shrink the ratio as ref_km grows.
    g = g[(g.resolved_level != "city") & g.error_km.notna() & (g.error_km > 0)]
    g["ratio"] = g.error_km / g.ref_km
    g["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                 zip(g.area.astype(str), g.place.astype(str), g.ref_zh, g.ref_en)]

    print("=" * 104)
    print("Criterion 1: error_km / ref_km -- ~1 means the answer is basically the reference point")
    print("             (non-city-level resolutions only, avoiding the constant trap)")
    print("=" * 104)
    rows = []
    for (rg, band), gg in g.groupby(["regime", "cond"]):
        rows.append(dict(
            regime=rg, band=band, n=len(gg),
            ref_dist_gm=round(gm(gg.ref_km), 2),
            error_gm=round(gm(gg.error_km), 2),
            ratio_median=round(gg.ratio.median(), 2),
            ratio_gm=round(gm(gg.ratio), 2),
            pct_error_below_dist=f"{100*(gg.ratio < 1).mean():.1f}%"))
    t = pd.DataFrame(rows)
    t["band"] = pd.Categorical(t.band, BANDS, ordered=True)
    print(t.sort_values(["regime", "band"]).to_string(index=False))

    print("\n" + "=" * 104)
    print("Criterion 2: echo rate -- fraction of responses naming the reference point (echoed/total)")
    print("=" * 104)
    rows = []
    for (rg, band), gg in g.groupby(["regime", "cond"]):
        rows.append(dict(regime=rg, band=band,
                         echoed=f"{int(gg.echo.sum())}/{len(gg)}",
                         echo_rate=f"{100*gg.echo.mean():.1f}%"))
    t = pd.DataFrame(rows)
    t["band"] = pd.Categorical(t.band, BANDS, ordered=True)
    print(t.sort_values(["regime", "band"]).to_string(index=False))

    print("\n" + "=" * 104)
    print("Criterion 3: error when echoing vs not echoing (within the same set of records)")
    print("             if echoing really is 'reciting the reference point', its error should be close to ref_km")
    print("=" * 104)
    rows = []
    for (rg, e_), gg in g.groupby(["regime", "echo"]):
        rows.append(dict(regime=rg, echoed="echo" if e_ else "no_echo", n=len(gg),
                         error_gm=round(gm(gg.error_km), 2),
                         ref_dist_gm=round(gm(gg.ref_km), 2),
                         ratio_gm=round(gm(gg.ratio), 2),
                         hit=f"{int(gg.hit.sum())}/{len(gg)}"
                              f"={100*gg.hit.mean():.1f}%"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 104)
    print("Criterion 4: each regime's own baseline -- is regime 1 really 'no information in the image'?")
    print("=" * 104)
    b = f[f.cond == "baseline"]
    rows = []
    for rg, gg in b.groupby("regime"):
        rows.append(dict(regime=rg, n_sites=gg.site.nunique(), n=len(gg),
                         hit=f"{int(gg.hit.sum())}/{len(gg)}"
                              f"={100*gg.hit.mean():.1f}%",
                         error_gm=round(gm(gg.error_km), 2),
                         error_median=round(gg.error_km.median(), 2),
                         pct_building_level=f"{100*(gg.resolved_level=='place').mean():.1f}%"))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
