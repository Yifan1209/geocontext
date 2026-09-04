"""The decisive test: is the distance effect anchoring, or just guessing geometry?

    python analysis/baseline_test.py

An alternative explanation raised during design discussion: the model has no
idea where the image is and just guesses somewhere near the given reference
point; the chance of guessing right naturally falls as the reference point gets
farther away. If so, the "distance effect" would be pure geometry, not anchoring.

Telling the two apart needs the **context=none baseline** (same image, same
model, same schema, just no reference point supplied):

    far-band error > baseline    context is a **net harm** -> anchoring holds
                                  (pure guessing cannot explain getting worse)
    far-band error < baseline    context is still a net gain -> the geometry
                                  explanation holds
    far-band ~ baseline          context has no net effect -> the distance
                                  effect is pure geometry

The key is the direction "getting worse": under pure guessing, context can only
ever add information, so the worst case is no change -- **it cannot fall below
a baseline with no information at all**. Dropping below baseline requires some
other mechanism.

WARNING: schema=forced only, because the baseline was only run under forced.
Comparing across schemas would mix in the mitigation effects.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts  # noqa: E402

pd.set_option("display.width", 240)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"
N_PERM = 20000
#: Band labels read from the ladder CSVs, not hard-coded -- the real values are
#: 1.5-3.0km / 3.0-6.0km.
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]
RNG = np.random.default_rng(42)


def load():
    df = pd.read_csv(config.DATA / "answers_judged.csv")
    df = df[(df.schema == "forced") & df.error_km.notna()].copy()
    meta = []
    for s in sorted(df.site.dropna().unique()):
        try:
            lad = prompts.load_ladder(s, LADDER_AUDITOR_TAG,
                                      include_baseline=False)
        except FileNotFoundError:
            continue
        for k, v in lad.items():
            meta.append(dict(site=s, context=k, ref_km=v["dist_km"],
                             band=v["band"], tier=v["tier"]))
    df = df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")
    df["cond"] = np.where(df.context == "none", "baseline", df.band)
    return df


def perm_p(a, b, stat=np.median):
    """Two-sided permutation test on the difference of `stat`.

    Samples are small and heavy-tailed, so a t-test is not appropriate.
    """
    obs = stat(a) - stat(b)
    pool = np.concatenate([a, b])
    n = len(a)
    cnt = 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        if abs(stat(pool[:n]) - stat(pool[n:])) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)


def main():
    df = load()
    base_n = (df.cond == "baseline").sum()
    print(f"forced schema, {len(df)} resolved records, of which {base_n} are baseline (no context)\n")

    # The baseline only ran 3 images x 5 models x 2 languages. For
    # comparability, every other condition is also **restricted to that same
    # set of images**.
    imgs = set(df.loc[df.cond == "baseline", "path"])
    d = df[df.path.isin(imgs)].copy()
    print(f"restricted to the {len(imgs)} images the baseline used: {len(d)} records\n")

    print("=== 1. Error distribution (km) and <1km hit rate, by condition ===")
    d["hit"] = geocode.hit(d)      # city-level never counts as a hit, see geocode.hit
    tab = (d.groupby(["site", "cond"])
             .agg(n=("error_km", "size"),
                  median=("error_km", "median"),
                  geomean=("error_km", lambda x: 10 ** np.log10(x[x > 0]).mean()),
                  hit1km=("hit", lambda x: 100 * x.mean()))
             .round(2))
    order = ["baseline"] + BANDS
    tab = tab.reindex(pd.MultiIndex.from_product(
        [sorted(d.site.unique()), order], names=["site", "cond"])).dropna(how="all")
    print(tab.to_string())

    print("\n=== 2. The decisive comparison: far band vs baseline ===")
    print("(geomean difference >0 = context makes answers more off-target = pure guessing cannot explain it)")
    rows = []
    for site, g in d.groupby("site"):
        b = g.loc[g.cond == "baseline", "error_km"].values
        b = np.log10(b[b > 0])
        for band in BANDS:
            a = g.loc[g.cond == band, "error_km"].values
            a = np.log10(a[a > 0])
            if len(a) < 10 or len(b) < 10:
                continue
            obs, p = perm_p(a, b, np.mean)
            rows.append(dict(site=site, band=band, n_band=len(a), n_base=len(b),
                             ratio=round(10 ** obs, 2), p=round(p, 4),
                             verdict="context harmful" if obs > 0 and p < .05 else
                                  "context helpful" if obs < 0 and p < .05 else "no significant difference"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 3. Pooled across the four sites (centred within each site first, to remove site-difficulty differences) ===")
    d["l"] = np.log10(d.error_km.clip(lower=1e-3))
    d["lc"] = d.l - d.groupby("site").l.transform("mean")
    b = d.loc[d.cond == "baseline", "lc"].values
    for band in BANDS:
        a = d.loc[d.cond == band, "lc"].values
        obs, p = perm_p(a, b, np.mean)
        print(f"  {band:10} vs baseline: error x{10**obs:.2f}  p={p:.4f}  "
              f"(n={len(a)} vs {len(b)})")

    print("\n=== 4. Per model: far band vs baseline (sites pooled) ===")
    rows = []
    for mdl, g in d.groupby("model"):
        bb = g.loc[g.cond == "baseline", "lc"].values
        ff = g.loc[g.cond == BANDS[-1], "lc"].values
        if len(bb) < 5 or len(ff) < 5:
            continue
        obs, p = perm_p(ff, bb, np.mean)
        rows.append(dict(model=mdl, n_far=len(ff), n_base=len(bb),
                         ratio=round(10 ** obs, 2), p=round(p, 4)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
