"""Raw counts and raw statistics, for the reader to judge from -- not conclusions.

    python analysis/raw_tables.py > results/raw_tables.txt

Report the raw statistic, not just a ratio. "x1.50" alone cannot distinguish
0.3->0.45 from 30->45, and the two carry very different credibility.

So every table below gives: numerator, denominator, n, and the raw km values.
A ratio appears only as an extra column, always alongside the counts it came from.

## Terminology (fixed)

    echo rate           the fraction of responses whose `area` or `building`
                         field directly names the reference point given in the
                         prompt. Measures how much the model "recites" the context.
    referenceability     0-5, how often a local/visitor actually uses a place to
                         say where they are. **Not fame.** (The code field is
                         still called `familiarity`; renaming it would
                         invalidate 4000+ cached audit decisions, so it stays.)
    baseline legibility  the model's own <1km hit rate at context=none.
                         **A behavioural measure of the model, not a quality
                         score from an image auditor.**
    hit                  error_km < 1km AND the resolved level is not city
                         (see geocode.hit)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 300)

BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]
#: Every shipped ladder carries the same auditor tag, so the per-site override
#: table that used to live here is gone; `load_ladder`'s default is enough.
AUDITOR_TAG = "deepseek-en"


def gm(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(10 ** np.log10(x).mean()) if len(x) else np.nan


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
    return df


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


def hit_cell(g):
    """One cell -> 'hit/total = xx.x%' plus the geometric-mean error."""
    n, k = len(g), int(g.hit.sum())
    if n == 0:
        return "--"
    return f"{k}/{n}={100*k/n:.1f}%  gm={gm(g.error_km):.2f}km"


def main():
    d = load()
    f = d[d.schema == "forced"]
    print(f"data: {len(d)} records, {d.site.nunique()} sites, {d.model.nunique()} models")
    print(f"schema breakdown: {dict(d.schema.value_counts())}\n")

    print("=" * 118)
    print("Table 1  per site x per condition: hits/total, hit rate, geometric-mean error")
    print("         (hit = error_km<1km and resolved level is not city)")
    print("=" * 118)
    rows = []
    for site, g in f.groupby("site"):
        r = {"site": site}
        for c, lab in [("baseline", "no_context")] + list(zip(BANDS, ["near", "mid", "far"])):
            r[lab] = hit_cell(g[g.cond == c])
        rows.append(r)
    t = pd.DataFrame(rows)
    order = f[f.cond == "baseline"].groupby("site").hit.mean().sort_values().index
    t = t.set_index("site").reindex(order).reset_index()
    print(t.to_string(index=False))

    print("\n" + "=" * 118)
    print("Table 2  echo rate, raw counts: echoed/total")
    print("         echo = the response's area or building directly names the prompt's reference point")
    print("=" * 118)
    e = f[f.cond != "baseline"].dropna(subset=["ref_zh", "ref_en"]).copy()
    e["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                 zip(e.area.astype(str), e.place.astype(str), e.ref_zh, e.ref_en)]
    print("\n-- by referenceability tier x distance band --")
    rows = []
    for tier in ["high", "mid", "low"]:
        r = {"tier": tier}
        for b, lab in zip(BANDS, ["near", "mid", "far"]):
            gg = e[(e.tier == tier) & (e.cond == b)]
            r[lab] = (f"{int(gg.echo.sum())}/{len(gg)}={100*gg.echo.mean():.1f}%"
                      if len(gg) else "--")
        gg = e[e.tier == tier]
        r["total"] = f"{int(gg.echo.sum())}/{len(gg)}={100*gg.echo.mean():.1f}%"
        rows.append(r)
    tot = {"tier": "total"}
    for b, lab in zip(BANDS, ["near", "mid", "far"]):
        gg = e[e.cond == b]
        tot[lab] = f"{int(gg.echo.sum())}/{len(gg)}={100*gg.echo.mean():.1f}%"
    tot["total"] = f"{int(e.echo.sum())}/{len(e)}={100*e.echo.mean():.1f}%"
    rows.append(tot)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 118)
    print("Table 3  per model x condition: hits/total")
    print("=" * 118)
    rows = []
    for m, g in f.groupby("model"):
        r = {"model": m}
        for c, lab in [("baseline", "no_context")] + list(zip(BANDS, ["near", "mid", "far"])):
            r[lab] = hit_cell(g[g.cond == c])
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 118)
    print("Table 4  error bucketed, raw counts")
    print("=" * 118)
    x = f.dropna(subset=["error_km"]).copy()
    bins = [0, .5, 1, 2, 5, 25, 100, 1000, 1e9]
    lab = ["<0.5km", "0.5-1", "1-2", "2-5", "5-25", "25-100", "100-1000", ">1000km"]
    x["bucket"] = pd.cut(x.error_km, bins=bins, labels=lab, right=False)
    print("\n-- by condition --")
    print(pd.crosstab(x.cond, x.bucket).reindex(["baseline"] + BANDS).to_string())
    print("\n-- by model --")
    print(pd.crosstab(x.model, x.bucket).to_string())

    print("\n" + "=" * 118)
    print("Table 5  resolved-level raw counts (scoring reliability)")
    print("=" * 118)
    print(pd.crosstab(f.cond, f.resolved_level.fillna("unresolved"))
          .reindex(["baseline"] + BANDS).to_string())

    print("\n" + "=" * 118)
    print("Table 6  mitigation, forced vs forced_warn: strictly paired "
          "(same site/image/model/language/reference point)")
    print("=" * 118)
    w = d[d.schema.isin(["forced", "forced_warn"]) & (d.cond != "baseline")].copy()
    key = ["site", "path", "model", "lang", "context"]
    cnt = w.groupby(key).schema.nunique()
    w["k"] = list(zip(*[w[c] for c in key]))
    w = w[w.k.isin(set(cnt[cnt == 2].index))]
    print(f"{len(w)} rows after pairing, covering {w.site.nunique()} sites\n")
    rows = []
    for sch, g in w.groupby("schema"):
        r = {"schema": sch}
        for b, lab in zip(BANDS, ["near", "mid", "far"]):
            r[lab] = hit_cell(g[g.cond == b])
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    ee = w.dropna(subset=["ref_zh", "ref_en"]).copy()
    ee["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                  zip(ee.area.astype(str), ee.place.astype(str),
                      ee.ref_zh, ee.ref_en)]
    print("\necho rate:")
    for sch, g in ee.groupby("schema"):
        print(f"  {sch:14} {int(g.echo.sum())}/{len(g)} = {100*g.echo.mean():.1f}%")

    print("\n" + "=" * 118)
    print("Table 7  per reference point, raw performance (top 25 by echo rate)")
    print("         for manual review: are the high-echo places really 'good for navigating by'?")
    print("=" * 118)
    per = (e.groupby(["site", "ref_en", "tier", "ref_km"])
             .agg(n=("echo", "size"), echoed=("echo", "sum"),
                  hit=("hit", "sum"), error_gm=("error_km", gm))
             .reset_index())
    per["echo_pct"] = (100 * per.echoed / per.n).round(1)
    per["hit_pct"] = (100 * per.hit / per.n).round(1)
    per["error_gm"] = per.error_gm.round(2)
    print(per.sort_values("echo_pct", ascending=False).head(25).to_string(index=False))
    out = config.RESULTS / "per_reference_point.csv"
    per.sort_values(["site", "ref_km"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nfull detail for all {len(per)} reference points -> {out}")


if __name__ == "__main__":
    main()
