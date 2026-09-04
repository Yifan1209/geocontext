"""Is error really monotone in reference-point distance? Test it properly.

    python analysis/distance_monotonicity.py

## Why this needed its own check

A reviewer pointed out that the near/mid/far bands do not look monotone, and
they were right. Two different things had been conflated up to that point:

  A. **Legibility -> harm**  across sites, the more accurate the baseline, the
     more harmful context is -- this holds
  B. **Distance -> error**   within a site, a farther reference point pulls the
     answer further off -- **this is much weaker**

And "compare the geometric mean of three bands" is not the right way to test
monotonicity at all: it is only three points, ignores within-band variance, and
throws away `ref_km`'s continuous information.

## The right way to do it

1. **Regress** log10(error) on ref_km, using continuous distance rather than banding
2. **Spearman rank correlation**, which only needs monotonicity, not linearity
3. Do both **split by legibility**, since A may moderate B

WARNING: run per site, each site's baseline is only 9 rows and underpowered;
pooling is needed for power. Both are reported below.
"""
import sys
import glob
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from geocontext import config, geocode  # noqa: E402

pd.set_option("display.width", 240)
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]


def load():
    df = pd.read_csv(config.DATA / "answers_judged.csv")
    meta = []
    for f in glob.glob(str(config.LADDERS / "*_ladder_*.csv")):
        site = os.path.basename(f).split("_ladder_")[0]
        for r in pd.read_csv(f).itertuples():
            key = (f"{r.tier}_{str(r.band).replace('.', '').replace('-', '_')}"
                   f"_{r.qid}")
            meta.append(dict(site=site, context=key, ref_km=float(r.dist_km),
                             band=str(r.band), tier=str(r.tier)))
    m = pd.DataFrame(meta).drop_duplicates(subset=["site", "context"])
    df = df.merge(m, on=["site", "context"], how="left")
    df["hit"] = geocode.hit(df)
    return df


def main():
    df = load()
    f = df[(df.schema == "forced") & df.error_km.notna() & (df.error_km > 0)].copy()

    base = (f[f.context == "none"].groupby("site").hit.mean() * 100)
    f["baseline_legibility"] = f.site.map(base)
    ctx = f.dropna(subset=["ref_km"]).copy()          # context conditions only
    ctx["y"] = np.log10(ctx.error_km)
    print(f"{len(ctx)} resolved records with context, covering {ctx.site.nunique()} sites\n")

    print("=" * 96)
    print("Table A  all sites pooled: log10(error) ~ reference-point distance")
    print("=" * 96)
    m = sm.OLS(ctx.y, sm.add_constant(ctx[["ref_km"]])).fit()
    rho, prho = stats.spearmanr(ctx.ref_km, ctx.y)
    print(f"  OLS coefficient {m.params['ref_km']:+.4f}  p={m.pvalues['ref_km']:.2e}"
          f"  n={len(ctx)}")
    print(f"  Spearman rho {rho:+.4f}  p={prho:.2e}")
    print(f"  -> each extra 1 km of reference-point distance multiplies error by {10**m.params['ref_km']:.3f}")

    print("\n" + "=" * 96)
    print("Table B  split by baseline legibility (tests whether A moderates B)")
    print("=" * 96)
    ctx["grp"] = pd.cut(ctx.baseline_legibility, [-1, 25, 50, 75, 101],
                       labels=["low <25%", "mid 25-50%", "high 50-75%", "very_high >75%"])
    rows = []
    for g, d in ctx.groupby("grp", observed=True):
        mm = sm.OLS(d.y, sm.add_constant(d[["ref_km"]])).fit()
        r, p = stats.spearmanr(d.ref_km, d.y)
        rows.append(dict(group=g, n_sites=d.site.nunique(), n=len(d),
                         ols_coef=round(mm.params["ref_km"], 4),
                         p_ols=f"{mm.pvalues['ref_km']:.1e}",
                         ratio_per_km=round(10 ** mm.params["ref_km"], 3),
                         spearman=round(r, 3), p_rho=f"{p:.1e}"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 96)
    print("Table C  per site (separate regression each) -- is the direction consistent?")
    print("=" * 96)
    rows = []
    for site, d in ctx.groupby("site"):
        if len(d) < 40:
            continue
        mm = sm.OLS(d.y, sm.add_constant(d[["ref_km"]])).fit()
        rows.append(dict(site=site, n=len(d),
                         baseline_hit=round(base.get(site, np.nan), 1),
                         coef=round(mm.params["ref_km"], 4),
                         p=round(mm.pvalues["ref_km"], 4),
                         direction="positive" if mm.params["ref_km"] > 0 else "negative"))
    t = pd.DataFrame(rows).sort_values("baseline_hit")
    print(t.to_string(index=False))
    pos = int((t.coef > 0).sum())
    sig = int(((t.p < .05) & (t.coef > 0)).sum())
    print(f"\n  of {len(t)} sites: {pos} have a positive coefficient, {sig} of those significant")
    print(f"  sign test (positive vs negative) p = "
          f"{stats.binomtest(pos, len(t), 0.5).pvalue:.4f}")

    print("\n" + "=" * 96)
    print("Table D  geometric mean of the three bands (the one originally shown, "
          "with within-band spread to explain why it alone is not enough)")
    print("=" * 96)
    rows = []
    for g, d in ctx.groupby("grp", observed=True):
        r = {"group": g}
        for b in BANDS:
            v = d.loc[d.band == b, "error_km"]
            r[b] = (f"{10**np.log10(v).mean():.2f} (n={len(v)}, "
                    f"sd={np.log10(v).std():.2f})") if len(v) else "--"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  sd is the standard deviation of log10 error. sd~=1 means within-band")
    print("  error spans an order of magnitude, so when the three bands' means differ")
    print("  by only 0.2-0.3 orders of magnitude, ranking them by mean alone is noise.")


if __name__ == "__main__":
    main()
