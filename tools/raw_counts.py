"""Raw counts, not derived statistics: right/wrong counts by model x site x condition.

    python tools/raw_counts.py

What is wanted is a table the reader can verify themselves -- which model, on
which site, how many right, how many wrong -- not an already-computed ratio.
Every table below gives **numerator and denominator**.

Definition of "right" (geocode.hit): `error_km < 1` **and** the resolved level
is not city. City-level resolutions have a constant error (Paris's 0.761 km
happens to be <1km), and must be excluded, or "could not answer with any
detail" gets recorded as "answered correctly".

Also breaks down "wrong, but how wrong" into bins, because "off by 1.2 km" and
"off by 1200 km" are both just "wrong" under binary scoring, throwing away all
the information.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts  # noqa: E402

pd.set_option("display.width", 250)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]


def load():
    df = pd.read_pickle(config.RESULTS / "answers_geocoded.pkl")
    meta = []
    for s, aud in LADDER_AUDITOR.items():
        for k, v in prompts.load_ladder(s, aud, include_baseline=False).items():
            meta.append(dict(site=s, context=k, band=v["band"]))
    df = df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")
    df["cond"] = np.where(df.context == "none", "baseline", df.band)
    df["correct"] = geocode.hit(df)
    return df


def counts(g):
    """One group of records -> counts of right/wrong/unresolved."""
    n = len(g)
    ok = int(g["correct"].sum())
    na = int(g.error_km.isna().sum())
    return pd.Series({"total": n, "right": ok, "wrong": n - ok - na, "unresolved": na,
                      "pct_correct": round(100 * ok / n, 1) if n else np.nan})


def main():
    df = load()
    f = df[df.schema == "forced"]
    print(f"{len(df)} records total. Unless noted, everything below uses "
          f"schema=forced ({len(f)} records)")
    print("correct = error_km < 1km and the resolved level is not city\n")

    print("=" * 92)
    print("Table 1  model x site (every condition pooled, baseline + three distance bands)")
    print("=" * 92)
    t = f.groupby(["model", "site"]).apply(counts, include_groups=False)
    print(t.to_string())

    print("\n" + "=" * 92)
    print("Table 2  model totals (pooled across sites) -- answers 'which model is most accurate'")
    print("=" * 92)
    print(f.groupby("model").apply(counts, include_groups=False)
          .sort_values("pct_correct", ascending=False).to_string())

    print("\n" + "=" * 92)
    print("Table 3  model x condition -- answers 'which model is most affected by context'")
    print("(note: this answers a different question from Table 2 -- higher accuracy != less affected)")
    print("=" * 92)
    t3 = f.groupby(["model", "cond"]).apply(counts, include_groups=False)
    print(t3.to_string())

    print("\n  per model: baseline accuracy -> far-band accuracy")
    for m, g in f.groupby("model"):
        b = g[g.cond == "baseline"]["correct"]
        far = g[g.cond == BANDS[-1]]["correct"]
        if len(b) and len(far):
            print(f"    {m:18} baseline {int(b.sum()):3}/{len(b):3} = {100*b.mean():5.1f}%"
                  f"   ->   far {int(far.sum()):4}/{len(far):4} = {100*far.mean():5.1f}%"
                  f"   ({100*(far.mean()-b.mean()):+5.1f}pp)")

    print("\n" + "=" * 92)
    print("Table 4  site x condition (every model pooled)")
    print("=" * 92)
    t4 = f.groupby(["site", "cond"]).apply(counts, include_groups=False)
    print(t4.to_string())

    print("\n" + "=" * 92)
    print("Table 5  how wrong is 'wrong' (error bucketed, counts)")
    print("=" * 92)
    e = f.dropna(subset=["error_km"]).copy()
    bins = [0, 1, 5, 25, 100, 1000, 1e9]
    lab = ["<1km", "1-5km", "5-25km", "25-100km", "100-1000km", ">1000km"]
    e["bucket"] = pd.cut(e.error_km, bins=bins, labels=lab, right=False)
    print(pd.crosstab(e.model, e.bucket).to_string())
    print("\nby site:")
    print(pd.crosstab(e.site, e.bucket).to_string())

    print("\n" + "=" * 92)
    print("Table 6  resolved-level distribution (counts) -- scoring reliability")
    print("=" * 92)
    print(pd.crosstab(f.model, f.resolved_level.fillna("unresolved")).to_string())

    print("\n" + "=" * 92)
    print("Table 7  record counts across all schemas (which cells have data)")
    print("=" * 92)
    print(pd.crosstab(df.site, df.schema).to_string())

    out = config.RESULTS / "raw_counts.csv"
    t.reset_index().to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nTable 1 saved to {out}")


if __name__ == "__main__":
    main()
