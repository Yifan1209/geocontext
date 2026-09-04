"""Every p-value in the paper, recomputed with the site as the unit of clustering.

    python analysis/cluster_inference.py

## Why

The paper's replication unit is the site: multiple images at one site show the
same buildings and the same street, and the reference points offered at a site
are drawn from one pool, so responses within a site are correlated. The
regressions and permutation tests as first written treat each response as an
independent draw. With a median of about 76 responses per site across 109
sites, that overstates the effective sample size and produces p-values far
smaller than the design supports.

Point estimates do not change here. Clustering changes only the standard errors
and the p-values, so every percentage, slope and ratio in the paper stands as
printed; what this script replaces is the claim about how surely they differ
from zero.

## How

Regressions: ordinary least squares, then cluster-robust (CR1 sandwich)
standard errors grouped by site.

Paired tests: the sign-flip permutation is moved from the item to the site.
Every paired difference belonging to one site receives the same flip, so the
resampling respects the correlation the clustering is meant to model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from two_regimes import load, BANDS  # noqa: E402

N_PERM = 20000
RNG = np.random.default_rng(42)
MODELS = ["claude-haiku-4-5", "qwen3-vl-235b", "qwen3-vl-8b",
          "gemini-flash", "claude-opus-5"]


def ols(y, x, groups):
    """Slope with naive and site-clustered p-values."""
    X = sm.add_constant(np.asarray(x, float))
    m = sm.OLS(np.asarray(y, float), X)
    naive = m.fit()
    clus = m.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    return (naive.params[1], naive.pvalues[1], clus.pvalues[1],
            clus.bse[1] / naive.bse[1])


def perm_site(diff, sites, n=N_PERM):
    """Sign-flip permutation at the site level.

    Flipping each paired difference on its own assumes the differences are
    independent. Flipping whole sites together keeps whatever correlation
    exists inside a site intact, which is the point of clustering.
    """
    diff = np.asarray(diff, float)
    codes, uniq = pd.factorize(pd.Series(sites))
    obs = diff.mean()
    hits = 0
    for _ in range(n):
        flip = RNG.choice([-1.0, 1.0], size=len(uniq))[codes]
        if abs((diff * flip).mean()) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def perm_item(diff, n=N_PERM):
    diff = np.asarray(diff, float)
    obs = diff.mean()
    hits = sum(abs((diff * RNG.choice([-1.0, 1.0], size=len(diff))).mean())
               >= abs(obs) - 1e-12 for _ in range(n))
    return obs, (hits + 1) / (n + 1)


def fmt(p):
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def main():
    d = load()
    f = d[d.schema == "forced"].copy()
    base = f[f.cond == "baseline"].groupby("site").hit.mean() * 100
    f["regime"] = np.where(f.site.map(base) < 50, "illegible", "legible")
    d["regime"] = np.where(d.site.map(base) < 50, "illegible", "legible")

    g = f[(f.cond != "baseline") & f.ref_km.notna()
          & f.error_km.notna() & (f.error_km > 0)]
    ly = np.log10(g.error_km)

    print("=" * 84)
    print("DISTANCE SLOPE  log10(err_km) ~ d_ref")
    print("=" * 84)
    print(f"  {'subset':<26}{'beta':>9}{'p naive':>12}{'p clustered':>14}"
          f"{'SE ratio':>10}")
    rows = [("all 109 sites", g), ("fine-grained only",
                                   g[g.resolved_level != "city"])]
    for m in MODELS:
        rows.append((m, g[g.model == m]))
    for rg in ["illegible", "legible"]:
        rows.append((f"all models, {rg}", g[g.regime == rg]))
    for name, sub in rows:
        b, pn, pc, r = ols(np.log10(sub.error_km), sub.ref_km, sub.site)
        print(f"  {name:<26}{b:>+9.4f}{fmt(pn):>12}{fmt(pc):>14}{r:>10.1f}")

    print()
    print("=" * 84)
    print("PAIRED INTERVENTIONS  sign-flip permutation, item vs site level")
    print("=" * 84)
    key = ["site", "path", "model", "lang", "context"]

    def paired(schema_b, label):
        w = d[d.schema.isin(["forced", schema_b]) & (d.cond != "baseline")].copy()
        cnt = w.groupby(key).schema.nunique()
        w["k"] = list(zip(*[w[c] for c in key]))
        w = w[w.k.isin(set(cnt[cnt == 2].index))]
        print(f"\n  {label}")
        print(f"    {'group':<12}{'n pairs':>9}{'delta':>9}{'p item':>11}"
              f"{'p site':>11}")
        for gname, sel in [("illegible", w[w.regime == "illegible"]),
                           ("legible", w[w.regime == "legible"]),
                           ("pooled", w)]:
            piv = sel.pivot_table(index="k", columns="schema",
                                  values="hit").dropna()
            if piv.empty:
                continue
            diff = (piv[schema_b] - piv["forced"]).values * 100
            sites = [k[0] for k in piv.index]
            _, pi = perm_item(diff)
            obs, ps = perm_site(diff, sites)
            print(f"    {gname:<12}{len(piv):>9}{obs:>+9.2f}{fmt(pi):>11}"
                  f"{fmt(ps):>11}")

    paired("forced_warn", "forced_warn vs forced (warning)")
    paired("forced_chain", "forced_chain vs forced (evidence checklist)")

    print()
    print("=" * 84)
    print("PRECISION COST  share resolving at building level, warn vs forced")
    print("=" * 84)
    d2 = d.copy()
    d2["place_level"] = d2.resolved_level.eq("place")
    w = d2[d2.schema.isin(["forced", "forced_warn"])
           & (d2.cond != "baseline")].copy()
    cnt = w.groupby(key).schema.nunique()
    w["k"] = list(zip(*[w[c] for c in key]))
    w = w[w.k.isin(set(cnt[cnt == 2].index))]
    piv = w.pivot_table(index="k", columns="schema",
                        values="place_level").dropna()
    diff = (piv["forced_warn"] - piv["forced"]).values * 100
    sites = [k[0] for k in piv.index]
    _, pi = perm_item(diff)
    obs, ps = perm_site(diff, sites)
    print(f"  n={len(piv)}  delta={obs:+.2f}  p item={fmt(pi)}  "
          f"p site={fmt(ps)}")

    print()
    print("=" * 84)
    print("CONTEXT VS BASELINE  building-level share, per band")
    print("=" * 84)
    b = d2[(d2.schema == "forced") & (d2.cond == "baseline")]
    for band in BANDS:
        c = d2[(d2.schema == "forced") & (d2.cond == band)]
        m = (b.groupby("path").place_level.mean().to_frame("base")
             .join(c.groupby("path").place_level.mean().to_frame("ctx"),
                   how="inner").dropna())
        sites = [p.replace("\\", "/").split("/")[1] for p in m.index]
        diff = (m.ctx - m.base).values * 100
        _, pi = perm_item(diff)
        obs, ps = perm_site(diff, sites)
        print(f"  {band:<12} n_img={len(m):<5} delta={obs:+.2f}  "
              f"p item={fmt(pi)}  p site={fmt(ps)}")


if __name__ == "__main__":
    main()
