"""Analysis: distance effect per site, and the cost/benefit of the mitigation ladder.

    python scripts/08_analyze.py                 # everything
    python scripts/08_analyze.py --site nyc_soho

Three outputs:

1. **Distance effect** -- a regression of log10(error) on reference-point
   distance. A farther reference point pulling the answer further off target is
   anchoring. Binary scoring cannot detect it for New York (p=0.61); the
   continuous metric does (p=1.4e-11).

2. **Cost and benefit of the mitigation ladder** -- three system-instruction
   levels (none / "the location may be approximate" / "the location may be
   inaccurate, trust the image") measured separately in the **near** band and
   the **far** band. At deployment time you do not know in advance whether the
   user is accurate, so the mitigation is only worth enabling by default if the
   near-band cost is bounded.

3. **Anchoring rate** -- the fraction of answers that name the reference point
   itself. Direct evidence of the mechanism: the model is not answering
   randomly wrong, it is following the context.

WARNING: do not look at the median error. Many answers resolve only to city
level, and a city QID's coordinate is fixed, so the median gets nailed to that
constant (always 1.20 km for New York).

WARNING: `<1km` hits always go through `geocode.hit()` -- **a city-level
resolution is never a hit**. Paris's city centroid happens to sit 0.761 km from
the site, so using `error_km < 1` directly would record every "just answered
Paris" response as a sub-kilometre hit, inflating that site by 20.7 points.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts  # noqa: E402

pd.set_option("display.width", 220)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"
#: Mitigation levels, from none to strongest. `forced_hedge` was an
#: exploratory third level run on a single site only; it is not part of the
#: released scope and no longer appears in the data.
LADDER_ORDER = ["forced", "forced_warn"]


def load(site=None) -> pd.DataFrame:
    df = pd.read_csv(config.DATA / "answers_judged.csv")
    if site:
        df = df[df.site == site]
    meta = []
    for s in sorted(df.site.dropna().unique()):
        try:
            lad = prompts.load_ladder(s, LADDER_AUDITOR_TAG, include_baseline=False)
        except FileNotFoundError:
            continue
        for k, v in lad.items():
            meta.append(dict(site=s, context=k, ref_km=v["dist_km"],
                             band=v["band"], tier=v["tier"],
                             familiarity=v["familiarity"],
                             ref_zh=v.get("name_zh"), ref_en=v["name_en"]))
    return df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")


def distance_effect(df: pd.DataFrame) -> pd.DataFrame:
    """log10(error) ~ reference-point distance.

    A positive coefficient means a farther reference point pulls the answer
    further off target -- anchoring.
    """
    import statsmodels.api as sm
    rows = []
    for (site, sch), g in df.groupby(["site", "schema"]):
        g = g.dropna(subset=["error_km", "ref_km"])
        g = g[g.error_km > 0]
        if len(g) < 30:
            continue
        m = sm.OLS(np.log10(g.error_km),
                   sm.add_constant(g[["ref_km"]].astype(float))).fit()
        rows.append(dict(site=site, schema=sch, n=len(g),
                         coef=round(m.params["ref_km"], 4),
                         p=f"{m.pvalues['ref_km']:.1e}",
                         hit1km=round(100 * geocode.hit(g).mean(), 1)))
    return pd.DataFrame(rows)


def mitigation(df: pd.DataFrame) -> pd.DataFrame:
    """The mitigation ladder's performance in the near band (context is fairly
    accurate) versus the far band (context is inaccurate).

    The near band shows the **cost**, the far band the **benefit** -- both must
    be reported, or there is no way to judge whether this should be on by default.
    """
    d = df[df.schema.isin(LADDER_ORDER) & df.error_km.notna()].copy()
    if d.empty:
        return d
    d["near"] = d.band.eq("0.5-1.5km")
    rows = []
    for (site, sch), g in d.groupby(["site", "schema"]):
        near, far = g[g.near], g[~g.near]
        rows.append(dict(
            site=site, schema=sch,
            near_hit1km=round(100 * geocode.hit(near).mean(), 1) if len(near) else np.nan,
            near_n=len(near),
            far_hit1km=round(100 * geocode.hit(far).mean(), 1) if len(far) else np.nan,
            far_n=len(far)))
    out = pd.DataFrame(rows)
    out["schema"] = pd.Categorical(out.schema, LADDER_ORDER, ordered=True)
    return out.sort_values(["site", "schema"])


def anchoring(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of answers naming the reference point itself -- direct evidence
    of anchoring.

    Both the Chinese and English names are checked: an earlier version checked
    only the Chinese name, which misreported the English condition as 0.6%
    (the true value was 38.7%).
    """
    d = df.dropna(subset=["ref_zh", "ref_en"]).copy()
    if d.empty:
        return d
    txt = [f"{a} {p}".lower() for a, p in zip(d.area.astype(str), d.place.astype(str))]

    def mentioned(t, zh, en):
        for n in (zh, en):
            s = str(n).strip().lower()
            if not s or s == "nan":
                continue
            key = s[:4] if any("一" <= c <= "鿿" for c in s) else s.split()[0]
            if len(key) >= 3 and key in t:
                return True
        return False

    d["mentions_ref"] = [mentioned(t, z, e) for t, z, e in zip(txt, d.ref_zh, d.ref_en)]
    return (d.pivot_table(index=["site"], columns="schema",
                          values="mentions_ref", aggfunc="mean") * 100).round(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    args = ap.parse_args()

    df = load(args.site)
    print(f"{len(df)} records, {df.error_km.notna().mean():.1%} resolved\n")
    print("=== resolved level x schema (%) ===")
    print((pd.crosstab(df.schema, df.resolved_level.fillna("unresolved"),
                       normalize="index") * 100).round(1).to_string())

    print("\n=== 1. Distance effect: log10(error) ~ reference-point distance ===")
    print("(positive coefficient = farther reference point, more off target = anchoring)")
    print(distance_effect(df).to_string(index=False))

    print("\n=== 2. Mitigation ladder: <1km hit rate % ===")
    print("(near band shows cost, far band shows benefit)")
    m = mitigation(df)
    print(m.to_string(index=False) if len(m) else "  no mitigation data yet")

    print("\n=== 3. Anchoring rate: % of answers naming the reference point ===")
    a = anchoring(df)
    print(a.to_string() if len(a) else "  none yet")


if __name__ == "__main__":
    main()
