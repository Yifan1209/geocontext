"""Did the structured-evidence checklist (forced_chain) work?

    python analysis/chain_results.py

## Read-out criteria (fixed before the run on 2026-09-01, not changed after)

| Observation | Conclusion |
|---|---|
| Regime 1's `error/ref-distance` median drops significantly below 1.00 | the reasoning chain made the model genuinely start reading the image -- the strongest evidence |
| Regime 1 hit-rate gain > forced_warn's +1.24pp | there is a contribution beyond just "trust the context less" |
| Regime 2 hit rate does not drop | does no harm, safe to recommend on by default |
| Echo rate drops but the ratio does not move | just a change in wording, **counts as failure** |

## Unfavourable prior

EarthWhere / WhereBench (arXiv 2510.10880) found across 17 models that "deeper
reasoning and web search do not reliably help when visual cues are limited"
(Gemini 3.1 Pro with search 62.56% vs without 62.22%). This is exactly the
situation we are trying to fix, so **measuring no effect is itself a result**
-- an independent confirmation of that finding, and we can supply the
mechanism it lacks (a ratio of 1.00 shows the deficit is not insufficient
reasoning, it is not reading the image at all).

WARNING: the reasoning chain only ran on 3 models (gemini-flash /
claude-haiku-4-5 / qwen3-vl-235b), so the comparison against `forced` **must be
restricted to those 3 models**, otherwise a different model mix would
manufacture a spurious effect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_regimes import load, gm, echoed, BANDS  # noqa: E402

pd.set_option("display.width", 230)
RNG = np.random.default_rng(42)
N_PERM = 20000
CHAIN_MODELS = ["gemini-flash", "claude-haiku-4-5", "qwen3-vl-235b"]


def paired_perm(diff):
    """Sign-flip paired permutation test."""
    obs = diff.mean()
    cnt = sum(abs((diff * RNG.choice([-1, 1], size=len(diff))).mean())
              >= abs(obs) - 1e-12 for _ in range(N_PERM))
    return obs, (cnt + 1) / (N_PERM + 1)


def main():
    d = load()
    # The regime split is a property of the site, so it is estimated from every
    # model's baseline -- the same assignment the rest of the paper uses. An
    # earlier version computed it after restricting to CHAIN_MODELS, which put
    # 2 of these 18 sites on the other side of the threshold from where every
    # other table has them. The *comparison* still runs on CHAIN_MODELS only;
    # it is the grouping that must not vary between tables.
    base = (d[(d.schema == "forced") & (d.cond == "baseline")]
            .groupby("site").hit.mean() * 100)
    d = d[d.model.isin(CHAIN_MODELS)].copy()
    d["regime"] = np.where(d.site.map(base) < 50, "regime1_illegible", "regime2_legible")

    print(f"restricted to the 3 models the reasoning chain ran on: {CHAIN_MODELS}")
    print(f"{len(d)} records, {d.site.nunique()} sites\n")

    # ---------- strict pairing ----------
    w = d[d.schema.isin(["forced", "forced_chain"]) & (d.cond != "baseline")].copy()
    key = ["site", "path", "model", "lang", "context"]
    cnt = w.groupby(key).schema.nunique()
    w["k"] = list(zip(*[w[c] for c in key]))
    w = w[w.k.isin(set(cnt[cnt == 2].index))]
    print(f"strictly paired (same site/image/model/language/reference point): {len(w)} rows\n")

    print("=" * 100)
    print("Criterion 2/3: hit rate, raw counts hit/total")
    print("=" * 100)
    rows = []
    for (rg, sch), g in w.groupby(["regime", "schema"]):
        r = {"regime": rg, "schema": sch}
        for b, lab in zip(BANDS, ["near", "mid", "far"]):
            gg = g[g.cond == b]
            r[lab] = (f"{int(gg.hit.sum())}/{len(gg)}={100*gg.hit.mean():.1f}%"
                      if len(gg) else "--")
        r["overall"] = f"{int(g.hit.sum())}/{len(g)}={100*g.hit.mean():.1f}%"
        r["error_gm"] = round(gm(g.error_km), 2)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\npaired test (chain - forced):")
    for rg, g in w.groupby("regime"):
        piv = g.pivot_table(index="k", columns="schema", values="hit").dropna()
        obs, p = paired_perm((piv["forced_chain"] - piv["forced"]).values)
        flag = "significant" if p < .05 else "n.s."
        print(f"  {rg:20} n={len(piv):5}  {100*obs:+.2f}pp  p={p:.4f}  {flag}")
    print("  for comparison: forced_warn was +0.46pp in regime 1 / +3.94pp in regime 2")

    # ---------- criterion 1: ratio ----------
    print("\n" + "=" * 100)
    print("Criterion 1 (strongest evidence): error / reference-point distance. 1.00 = the answer IS the reference point")
    print("non-city-level resolutions only, avoiding the city-level constant trap")
    print("=" * 100)
    r = w[(w.resolved_level != "city") & w.error_km.notna() & (w.error_km > 0)].copy()
    r["ratio"] = r.error_km / r.ref_km
    rows = []
    for (rg, sch, b), g in r.groupby(["regime", "schema", "cond"]):
        rows.append(dict(regime=rg, schema=sch, band=b, n=len(g),
                         ratio_median=round(g.ratio.median(), 2),
                         ratio_gm=round(gm(g.ratio), 2),
                         error_gm=round(gm(g.error_km), 2)))
    t = pd.DataFrame(rows)
    t["band"] = pd.Categorical(t.band, BANDS, ordered=True)
    print(t.sort_values(["regime", "band", "schema"]).to_string(index=False))

    # ---------- criterion 4: echo rate ----------
    print("\n" + "=" * 100)
    print("Criterion 4: echo rate. If only the echo rate drops while the ratio does not move, that just renames the failure")
    print("=" * 100)
    e = w.dropna(subset=["ref_zh", "ref_en"]).copy()
    e["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                 zip(e.area.astype(str), e.place.astype(str), e.ref_zh, e.ref_en)]
    for (rg, sch), g in e.groupby(["regime", "schema"]):
        print(f"  {rg:20} {sch:14} {int(g.echo.sum()):5}/{len(g):5} "
              f"= {100*g.echo.mean():.1f}%")

    # ---------- per model ----------
    print("\n" + "=" * 100)
    print("Per model (paired, percentage points chain - forced)")
    print("=" * 100)
    for (rg, m), g in w.groupby(["regime", "model"]):
        piv = g.pivot_table(index="k", columns="schema", values="hit").dropna()
        if len(piv) < 30:
            continue
        obs, p = paired_perm((piv["forced_chain"] - piv["forced"]).values)
        print(f"  {rg:20} {m:18} n={len(piv):5} {100*obs:+6.2f}pp  p={p:.4f}")


if __name__ == "__main__":
    main()
