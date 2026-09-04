"""Does the location context make the model give *vaguer* answers?

    python analysis/answer_specificity.py

## The problem

The `forced` schema does not allow the model to say "unknown", so it invents
**unlocatable phrases** to comply instead -- "Friday's or a nearby commercial
building", "Outfront Media billboard wall". 12% of records currently fail to
parse, and 57% of those fall into this category.

This is not a measurement problem, it is a **behaviour worth reporting**. And
it raises a new question: **does supplying location context make the model
more likely to give this kind of evasive answer?**

## Method: use resolved_level, not a text classifier

A regex classifier was tried (`classify_geocode_failures.py`) but it **is known
to misclassify** (it judged "Rockefeller Foundation Building" as a generic
category, and "Industrial and Commercial Bank of China" as a real place name).
It is not reliable as a primary metric.

Using the **already-computed objective quantity** `resolved_level` instead:

    place  -> resolved to a specific building/shop = specific and real
    area   -> resolved only to a neighbourhood
    city   -> resolved only to a city
    NaN    -> could not be resolved at all         = probably not a real place name

The geocoder is identical across every condition, so **comparing across
conditions** automatically cancels out the confound of "the geocoder's own
coverage gaps". Absolute values are not interpretable; **differences between
conditions are**.

## Three comparisons

1. Baseline (no context) vs each distance band -- does context make answers vaguer?
2. `forced` vs `forced_warn` -- does the warning make the model more conservative/vague?
3. Per model -- is this widespread, or one model's quirk?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, prompts  # noqa: E402

pd.set_option("display.width", 220)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]
RNG = np.random.default_rng(42)
N_PERM = 20000


def load():
    df = pd.read_csv(config.DATA / "answers_judged.csv")
    meta = []
    for s in sorted(df.site.dropna().unique()):
        try:
            lad = prompts.load_ladder(s, LADDER_AUDITOR_TAG,
                                      include_baseline=False)
        except FileNotFoundError:
            continue
        for k, v in lad.items():
            meta.append(dict(site=s, context=k, band=v["band"]))
    df = df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")
    df["cond"] = np.where(df.context == "none", "baseline", df.band)
    # resolved to building/shop level = the answer is both specific and real
    df["specific"] = df.resolved_level.eq("place")
    df["unresolvable"] = df.resolved_level.isna()
    return df


def perm_p(a, b):
    """Two-sided permutation test on the difference in two proportions."""
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    n, cnt = len(a), 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)


def main():
    df = load()
    f = df[df.schema == "forced"]
    print(f"{len(df)} records total; {len(f)} with schema=forced\n")

    print("=== 0. Baseline fact: share of each resolved level (%) ===")
    print("(place = specific and real; NaN = probably not a real place name)")
    t = (f.resolved_level.fillna("unresolved").value_counts(normalize=True) * 100).round(1)
    print(t.to_string())

    print("\n\n=== 1. Does context make answers vaguer? (restricted to images that also have a baseline call) ===")
    imgs = set(f.loc[f.cond == "baseline", "path"])
    d = f[f.path.isin(imgs)]
    rows = []
    for site, g in d.groupby("site"):
        r = {"site": site}
        for c in ["baseline"] + BANDS:
            s = g.loc[g.cond == c, "specific"]
            r[c] = f"{100*s.mean():.1f}% (n={len(s)})" if len(s) else "--"
        rows.append(r)
    print("share resolving to place level (a specific building/shop):")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\npooled across all sites, each band vs baseline:")
    b = d.loc[d.cond == "baseline", "specific"].values.astype(float)
    print(f"  baseline        {100*b.mean():5.1f}%  (n={len(b)})")
    for band in BANDS:
        a = d.loc[d.cond == band, "specific"].values.astype(float)
        obs, p = perm_p(a, b)
        print(f"  {band:14} {100*a.mean():5.1f}%  (n={len(a)})  "
              f"diff {100*obs:+5.1f}pp  p={p:.4f}")

    print("\n\n=== 2. Share unresolvable (a more direct evasiveness measure) ===")
    b = d.loc[d.cond == "baseline", "unresolvable"].values.astype(float)
    print(f"  baseline        {100*b.mean():5.1f}%  (n={len(b)})")
    for band in BANDS:
        a = d.loc[d.cond == band, "unresolvable"].values.astype(float)
        obs, p = perm_p(a, b)
        print(f"  {band:14} {100*a.mean():5.1f}%  (n={len(a)})  "
              f"diff {100*obs:+5.1f}pp  p={p:.4f}")

    print("\n\n=== 3. Does the warning make the model vaguer? (New York, has forced_warn data) ===")
    n = df[(df.site == "nyc_soho") & df.schema.isin(
        ["forced", "forced_hedge", "forced_warn"])]
    t = (n.groupby("schema")
          .agg(n=("specific", "size"),
               pct_place=("specific", lambda x: round(100 * x.mean(), 1)),
               pct_unresolvable=("unresolvable", lambda x: round(100 * x.mean(), 1))))
    print(t.to_string())
    a = n.loc[n.schema == "forced_warn", "specific"].values.astype(float)
    b2 = n.loc[n.schema == "forced", "specific"].values.astype(float)
    if len(a) and len(b2):
        obs, p = perm_p(a, b2)
        print(f"\n  forced_warn vs forced: place-level share diff {100*obs:+.1f}pp  p={p:.4f}")

    print("\n\n=== 4. Per model: share resolving to place level (%) ===")
    print("(is this widespread, or one model's quirk?)")
    t = (f.pivot_table(index="model", columns="site", values="specific",
                       aggfunc="mean") * 100).round(1)
    t["overall"] = (f.groupby("model").specific.mean() * 100).round(1)
    print(t.to_string())

    print("\n\n=== 5. 12 sampled 'unresolvable' answers, for inspection ===")
    pool = f[f.unresolvable & f.place.notna()]
    u = pool.sample(min(12, len(pool)), random_state=42)   # not f.unresolvable.sum():
    #                                    that is the pre-filter total and can
    #                                    exceed pool's length, which errors out
    for r in u.itertuples():
        print(f"  [{str(r.site)[:16]:16}] {str(r.place)[:72]}")


if __name__ == "__main__":
    main()
