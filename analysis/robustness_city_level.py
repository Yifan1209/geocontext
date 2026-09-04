"""Robustness: does the conclusion survive dropping city-level resolutions?

    python analysis/robustness_city_level.py

Background: 17.6% of records resolve only to **city** level, and their
error_km is a constant (Paris 0.761 / New York 1.199 / Tokyo
3.424 km). Scoring already excludes these from "hits" via geocode.hit(), but
**continuous metrics (regressions, geometric means) still carry that constant
along with them**.

That constant carries no information about model performance -- it only
reflects "the model could not resolve anything finer". So this checks: does
the conclusion still hold once these are dropped?

Both readings are shown side by side, both reporting the raw numbers (km,
counts), never a ratio alone:
  full           -- includes city-level records
  fine-grained   -- keeps only records resolved to district/shop level

WARNING: the exclusion itself carries selection bias -- the model only answers
finely when confident, and confident answers tend to be the easy samples. So
**both readings must be shown**, never just the one that is picked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from geocontext import config, prompts  # noqa: E402

pd.set_option("display.width", 240)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]
RNG = np.random.default_rng(42)
N_PERM = 20000


def gm(x):
    """Geometric mean. Error spans 4 orders of magnitude, and an ordinary mean
    would be dominated by extreme values."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return float(10 ** np.log10(x).mean()) if len(x) else np.nan


def load():
    df = pd.read_csv(config.DATA / "answers_judged.csv")
    df = df[df.error_km.notna()].copy()
    meta = []
    for s in sorted(df.site.dropna().unique()):
        try:
            lad = prompts.load_ladder(s, LADDER_AUDITOR_TAG,
                                      include_baseline=False)
        except FileNotFoundError:
            continue
        for k, v in lad.items():
            meta.append(dict(site=s, context=k, ref_km=v["dist_km"], band=v["band"]))
    df = df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")
    df["cond"] = np.where(df.context == "none", "baseline", df.band)
    return df


def perm_p(a, b):
    """Two-sided permutation test on the difference in log10 means."""
    a, b = np.log10(a[a > 0]), np.log10(b[b > 0])
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    n, cnt = len(a), 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)


def table(d, label):
    print(f"\n--- {label} ---")
    print("geometric-mean error (km) and counts, per site x condition")
    rows = []
    for site, g in d.groupby("site"):
        r = {"site": site}
        for c in ["baseline"] + BANDS:
            v = g.loc[g.cond == c, "error_km"]
            r[c] = f"{gm(v):.2f} (n={len(v)})" if len(v) else "--"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    df = load()
    fine = df[df.resolved_level != "city"]
    print(f"{len(df)} records total; {len(fine)} after dropping city-level "
          f"({len(df)-len(fine)} dropped = {1-len(fine)/len(df):.1%})")

    for d, lab in [(df, "Reading A: all records"), (fine, "Reading B: fine-grained only (city-level dropped)")]:
        table(d[d.schema == "forced"], lab + ", schema=forced")

    print("\n\n=== Paris: is context harmful? (raw geometric mean + test) ===")
    print("this is the single most important cell in the paper, verified under both readings")
    for d, lab in [(df, "Reading A, all"), (fine, "Reading B, fine-grained only")]:
        g = d[(d.site == "paris_marais") & (d.schema == "forced")]
        b = g.loc[g.cond == "baseline", "error_km"].values
        print(f"\n  {lab}: no context {gm(b):.3f} km (n={len(b)})")
        for band in BANDS:
            a = g.loc[g.cond == band, "error_km"].values
            if len(a) < 10 or len(b) < 10:
                print(f"    {band:10} n too small, skipped")
                continue
            obs, p = perm_p(a, b)
            verdict = ("harmful" if obs > 0 and p < .05
                       else "helpful" if obs < 0 and p < .05 else "n.s.")
            print(f"    {band:10} {gm(a):6.3f} km (n={len(a)})  "
                  f"p={p:.4f}  {verdict}")

    print("\n\n=== Distance-effect regression: log10(error) ~ reference-point distance ===")
    print("(positive coefficient = farther reference point, more off target)")
    rows = []
    for (site, sch), g0 in df[df.schema == "forced"].groupby(["site", "schema"]):
        for d, lab in [(g0, "A_all"),
                       (g0[g0.resolved_level != "city"], "B_fine_only")]:
            d = d.dropna(subset=["error_km", "ref_km"])
            d = d[d.error_km > 0]
            if len(d) < 30:
                continue
            m = sm.OLS(np.log10(d.error_km),
                       sm.add_constant(d[["ref_km"]].astype(float))).fit()
            rows.append(dict(site=site, reading=lab, n=len(d),
                             coef=round(m.params["ref_km"], 4),
                             p=f"{m.pvalues['ref_km']:.1e}"))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
