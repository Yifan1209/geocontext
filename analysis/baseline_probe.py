"""Follow-ups after the baseline test: what does the model do with no context,
and is its answer distinguishable from "just name the reference point"?

    python analysis/baseline_probe.py

Three questions:

1. **What does the model answer with no context?** Where the baseline
   geometric-mean error runs to hundreds of kilometres, it needs checking:
   is that "wrong city" or "wrong country"? If the image itself is
   completely indistinguishable, the claim "context stops the model reading
   the image" is vacuous -- there was nothing much to read in the first place.

2. **error_km / ref_km ratio** -- pure guessing (just repeating the reference
   point) gives a ratio of ~1. A ratio significantly below 1 means the model
   is really using image evidence; ~1 means it is essentially reciting the
   reference point. This is the cleanest test between "rational use of a weak
   hint" and "anchoring".

3. **Legibility stratification** -- some sites' baselines sit above 80%, where
   the imagery is legible; others sit in the low single digits. If "legible image ->
   context is useless or harmful; illegible image -> context is the only
   information source" turns out to hold, the paper's claim should be
   **conditional**, not unconditional anchoring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"
BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]


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
            meta.append(dict(site=s, context=k, ref_km=v["dist_km"], band=v["band"],
                             tier=v["tier"], ref_zh=v["name_zh"], ref_en=v["name_en"]))
    df = df.merge(pd.DataFrame(meta), on=["site", "context"], how="left")
    df["cond"] = np.where(df.context == "none", "baseline", df.band)
    imgs = set(df.loc[df.cond == "baseline", "path"])
    return df[df.path.isin(imgs)].copy()


def main():
    d = load()

    print("=== 1. What the model answers with no context (per site, by frequency) ===")
    b = d[d.cond == "baseline"]
    for site, g in b.groupby("site"):
        print(f"\n-- {site}  (geometric-mean error against ground truth "
              f"{10 ** np.log10(g.error_km.clip(lower=1e-3)).mean():.1f} km)")
        t = (g.groupby(["city", "area"]).agg(n=("error_km", "size"),
                                             error_km=("error_km", "median"))
             .sort_values("n", ascending=False).head(6))
        print(t.to_string())

    print("\n\n=== 2. error_km / ref_km ratio ===")
    print("(~1 = essentially reciting the reference point; <1 = real image "
          "evidence was used; >1 = worse than just reciting the reference point)")
    g = d[d.cond != "baseline"].copy()
    g["ratio"] = g.error_km / g.ref_km
    t = (g.groupby(["site", "cond"])
          .agg(n=("ratio", "size"), ratio_median=("ratio", "median"),
               ratio_gm=("ratio", lambda x: 10 ** np.log10(x.clip(lower=1e-3)).mean()),
               pct_better_than_reciting=("ratio", lambda x: 100 * (x < 1).mean()))
          .round(2))
    print(t.to_string())
    print("\npooled across the four sites:")
    print(g.groupby("cond").ratio.describe(percentiles=[.25, .5, .75]).round(2).to_string())

    print("\n\n=== 3. Image legibility x benefit of context ===")
    print("(baseline hit rate = how legible the image is; benefit = near-band "
          "geomean / baseline geomean, <1 is an improvement)")
    d["hit"] = geocode.hit(d)      # city-level never counts as a hit, see geocode.hit
    rows = []
    for site, g2 in d.groupby("site"):
        bb = g2.loc[g2.cond == "baseline", "error_km"]
        gm = lambda x: 10 ** np.log10(x.clip(lower=1e-3)).mean()  # noqa: E731
        row = dict(site=site,
                   baseline_hit1km=round(100 * g2.loc[g2.cond == "baseline", "hit"].mean(), 1),
                   baseline_gm=round(gm(bb), 1))
        for band in BANDS:
            aa = g2.loc[g2.cond == band, "error_km"]
            row[band] = round(gm(aa) / gm(bb), 2) if len(aa) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("baseline_hit1km")
    print(out.to_string(index=False))

    print("\n\n=== 4. Echo rate: the response directly names the reference point ===")
    print("(split by band: if the far band's echo rate does not drop, the model "
          "recites the reference point regardless of how far away it is)")

    def mentioned(area, place, zh, en):
        t = f"{area} {place}".lower()
        for n in (zh, en):
            s = str(n).strip().lower()
            if not s or s == "nan":
                continue
            key = s[:4] if any("一" <= c <= "鿿" for c in s) else s.split()[0]
            if len(key) >= 3 and key in t:
                return True
        return False

    g = d[d.cond != "baseline"].dropna(subset=["ref_zh", "ref_en"]).copy()
    g["hit_ref"] = [mentioned(a, p, z, e) for a, p, z, e in
                    zip(g.area.astype(str), g.place.astype(str), g.ref_zh, g.ref_en)]
    print((g.pivot_table(index="site", columns="cond", values="hit_ref",
                         aggfunc="mean") * 100).round(1).to_string())
    print("\nby referenceability tier:")
    print((g.pivot_table(index="tier", columns="cond", values="hit_ref",
                         aggfunc="mean") * 100).round(1).to_string())


if __name__ == "__main__":
    main()
