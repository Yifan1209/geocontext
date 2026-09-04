"""The forced vs forced_warn strictly-paired mitigation table (paper Table 3).

    python bin/mitigation_results.py

This didn't have a standalone script before —— the numbers currently in
`paper/main.tex` (\\label{tab:mitigation}) were produced by an ad-hoc scratchpad
computation earlier in the project. Written here as a permanent script,
mirroring `chain_results.py`'s pairing/permutation machinery exactly, so the
table is reproducible and can be rerun whenever the underlying ladders change
(as they did on 2026-09-01: 5 self-name-leaking reference points removed,
see `geolab/self_names.py`).

Pairing key is (site, path, model, lang, context): same image, same model,
same reference point, forced vs forced_warn only.

## Why the pairing is strict, and what happens without it

Do not compare the two arms as independent group means. This effect was
reported wrongly three separate times before the pairing was enforced, each
time for a different reason:

1. Read off a single site, the effect came out NEGATIVE ("obeyed but useless,
   costing 8.3 points").
2. One site had `forced` coverage but no `forced_warn` coverage. Averaging the
   arms separately let that site drag down the `forced` arm only, manufacturing
   a uniform +4.9 point gain that did not exist.
3. At 19 sites the illegible-regime difference was +1.26 points (p=0.046);
   extending to 109 sites it fell to +0.46 (p=0.33). An effect that only just
   clears p=0.05 should not be treated as established until the sample grows.

Any item present in one arm and missing from the other must be dropped, which
is what the inner join below does. The same applies to any future cross-schema
comparison built on this data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_regimes import load, echoed, BANDS  # noqa: E402

pd.set_option("display.width", 230)
RNG = np.random.default_rng(42)
N_PERM = 20000


def paired_perm(diff):
    obs = diff.mean()
    cnt = sum(abs((diff * RNG.choice([-1, 1], size=len(diff))).mean())
              >= abs(obs) - 1e-12 for _ in range(N_PERM))
    return obs, (cnt + 1) / (N_PERM + 1)


def main():
    d = load()
    base = (d[(d.schema == "forced") & (d.cond == "baseline")]
            .groupby("site").hit.mean() * 100)
    d = d.copy()
    d["regime"] = np.where(d.site.map(base) < 50, "Illegible", "Legible")

    print(f"{len(d)} records, {d.site.nunique()} sites\n")

    w = d[d.schema.isin(["forced", "forced_warn"]) & (d.cond != "baseline")].copy()
    key = ["site", "path", "model", "lang", "context"]
    cnt = w.groupby(key).schema.nunique()
    w["k"] = list(zip(*[w[c] for c in key]))
    w = w[w.k.isin(set(cnt[cnt == 2].index))]
    print(f"strictly paired (same site/image/model/language/reference point): {len(w)} rows\n")

    print("=" * 100)
    print("hit rate, raw counts hit/total (this is Table 3's data source)")
    print("=" * 100)
    rows = []
    for (rg, sch), g in w.groupby(["regime", "schema"]):
        r = {"regime": rg, "schema": sch}
        for b, lab in zip(BANDS, ["Near", "Mid", "Far"]):
            gg = g[g.cond == b]
            r[lab] = (f"{int(gg.hit.sum())}/{len(gg)}={100*gg.hit.mean():.1f}%"
                      if len(gg) else "—")
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\npaired test (forced_warn - forced), overall (three bands pooled):")
    for rg, g in w.groupby("regime"):
        piv = g.pivot_table(index="k", columns="schema", values="hit").dropna()
        obs, p = paired_perm((piv["forced_warn"] - piv["forced"]).values)
        flag = "significant" if p < .05 else "n.s."
        print(f"  {rg:10} n={len(piv):5}  {100*obs:+.2f}pp  p={p:.4f}  {flag}")

    print("\n" + "=" * 100)
    print("echo rate: was the warning actually heeded?")
    print("=" * 100)
    e = w.dropna(subset=["ref_zh", "ref_en"]).copy()
    e["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                 zip(e.area.astype(str), e.place.astype(str), e.ref_zh, e.ref_en)]
    for (rg, sch), g in e.groupby(["regime", "schema"]):
        print(f"  {rg:10} {sch:14} {int(g.echo.sum()):5}/{len(g):5} "
              f"= {100*g.echo.mean():.1f}%")


if __name__ == "__main__":
    main()
