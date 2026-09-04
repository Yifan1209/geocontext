"""Per-model raw hit counts and context-sensitivity table (paper Table 1 & 2).

    python bin/model_tables.py

Table 1 (`tab:models`): raw hit counts by model, forced schema, all conditions
pooled (baseline + context).

Table 2 (`tab:ctxsens`): per-model echo rate and the near-vs-far accuracy drop,
context conditions only (baseline excluded). Split out because the pooled
Table 1 conflates "how often a model repeats the reference point" with "how
much accuracy it loses as the reference point moves away."

Both cover every shipped site.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from two_regimes import load, echoed  # noqa: E402

pd.set_option("display.width", 200)


def main():
    df = load()
    f = df[df.schema == "forced"].copy()

    print("=" * 90)
    print("Table 1 (tab:models): raw hit counts, forced schema, all conditions pooled")
    print("=" * 90)
    t1 = f.groupby("model").hit.agg(hit="sum", n="size")
    t1["miss"] = t1.n - t1.hit
    t1["pct"] = (100 * t1.hit / t1.n).round(1)
    print(t1.sort_values("pct", ascending=False)[["hit", "miss", "pct"]].to_string())

    print("\n" + "=" * 90)
    print("Table 2 (tab:ctxsens): per-model echo rate + near-band-to-far-band hit-rate drop, context conditions only")
    print("=" * 90)
    ctx = f[f.cond != "baseline"].copy()
    ctx["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                  zip(ctx.area, ctx.place, ctx.ref_zh, ctx.ref_en)]
    rows = []
    for m, g in ctx.groupby("model"):
        near = g[g.cond == "0.5-1.5km"]
        far = g[g.cond == "3.0-6.0km"]
        rows.append(dict(model=m, n=len(g), echo_n=int(g.echo.sum()),
                         echo_pct=round(100 * g.echo.mean(), 1),
                         near_pct=round(100 * near.hit.mean(), 1), n_near=len(near),
                         far_pct=round(100 * far.hit.mean(), 1), n_far=len(far)))
    t2 = pd.DataFrame(rows)
    t2["drop"] = (t2.near_pct - t2.far_pct).round(1)
    print(t2.sort_values("echo_pct", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
