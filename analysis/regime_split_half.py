"""Is the two-regime baseline comparison inflated by regression to the mean?

    python analysis/regime_split_half.py

## The problem

`two_regimes.py` assigns each site to a regime from its `context=none` hit
rate, and the regime tables then report that same no-context rate as a column
to compare the context conditions against. Selection and measurement share the
responses, so a site that lands in the legible group partly because its
baseline draw came out high carries that upward noise into the column it is
compared against. The legible group's baseline is biased up, the illegible
group's down, and the context effect is pulled negative at legible sites and
positive at illegible ones -- with no true effect required.

The dilution is real but partial: each site has a median of 5 baseline
responses across 5 models, so any one model contributes about 20% of the
variable that assigns the regime.

## The correction

Split each site's baseline responses at random into halves A and B. Assign the
regime from A. Report the no-context column from B. B never touches the
assignment, so it is an unbiased estimate of the group's true baseline. Repeat
and average, because with a median of 5 responses per site each half is small
and a single split is noisy.

## What the read-out means

Split-half removes the bias and adds variance; the added variance pushes group
contrasts toward zero. So a corrected effect that stays large is trustworthy,
while a corrected effect that vanishes is ambiguous between "the effect was an
artefact" and "the split-half assignment is too noisy to see it". The
regime-stability count printed below separates those: if few sites change side,
the assignment is stable and a vanished effect really was an artefact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from two_regimes import load, BANDS  # noqa: E402

R = 200
SEED = 42
THRESHOLD = 50.0
MODELS = ["claude-opus-5", "gemini-flash", "qwen3-vl-235b",
          "qwen3-vl-8b", "claude-haiku-4-5"]


def naive(f):
    """The grouping the paper currently uses: assign and measure on the same
    responses."""
    base = f[f.cond == "baseline"].groupby("site").hit.mean() * 100
    regime = np.where(f.site.map(base) < THRESHOLD, "illegible", "legible")
    return base, pd.Series(regime, index=f.index)


def report(f, regime, base_rows, label):
    """Per-model no-context vs near-band within each regime."""
    out = {}
    for rg in ["illegible", "legible"]:
        for m in MODELS:
            sel = (regime == rg) & (f.model == m)
            b = f[sel & base_rows]
            n = f[sel & (f.cond == BANDS[0])]
            if len(b) == 0 or len(n) == 0:
                continue
            out[(rg, m)] = (100 * b.hit.mean(), 100 * n.hit.mean(), len(b))
    return out


def main():
    rng = np.random.default_rng(SEED)
    d = load()
    f = d[d.schema == "forced"].copy()

    # ---------------------------------------------------------- naive
    base, regime_naive = naive(f)
    is_base = f.cond == "baseline"
    nv = report(f, regime_naive, is_base, "naive")

    n_illeg = int((base < THRESHOLD).sum())
    print(f"{len(f)} forced records, {f.site.nunique()} sites")
    print(f"naive split at {THRESHOLD:.0f}%: {n_illeg} illegible / "
          f"{len(base) - n_illeg} legible\n")

    print("=" * 78)
    print("NAIVE (what the paper reports): no-context vs near band, per model")
    print("=" * 78)
    print(f"  {'regime':<11}{'model':<18}{'no ctx':>9}{'near':>9}{'diff':>9}"
          f"{'n_base':>8}")
    for (rg, m), (b, n, nb) in nv.items():
        print(f"  {rg:<11}{m:<18}{b:>8.1f}%{n:>8.1f}%{n - b:>+9.1f}{nb:>8}")

    # ---------------------------------------------------------- split-half
    bidx = f.index[is_base].to_numpy()
    bsite = f.loc[bidx, "site"].to_numpy()
    order = np.argsort(bsite, kind="stable")
    bidx, bsite = bidx[order], bsite[order]
    bounds = np.searchsorted(bsite, np.unique(bsite), side="left").tolist()
    bounds.append(len(bsite))
    groups = [bidx[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

    acc = {k: [] for k in nv}
    flips = []
    for _ in range(R):
        a_parts, b_parts = [], []
        for g in groups:
            perm = rng.permutation(g)
            half = len(perm) // 2
            a_parts.append(perm[:half])
            b_parts.append(perm[half:])
        a_idx = np.concatenate(a_parts)
        b_idx = np.concatenate(b_parts)

        base_a = f.loc[a_idx].groupby("site").hit.mean() * 100
        reg = pd.Series(
            np.where(f.site.map(base_a) < THRESHOLD, "illegible", "legible"),
            index=f.index)
        flips.append(int((reg[is_base].groupby(f.site[is_base]).first()
                          != regime_naive[is_base].groupby(
                              f.site[is_base]).first()).sum()))

        in_b = pd.Series(False, index=f.index)
        in_b.loc[b_idx] = True
        r = report(f, reg, in_b, "split")
        for k in acc:
            if k in r:
                acc[k].append(r[k][2] and (r[k][1] - r[k][0]))

    print()
    print("=" * 78)
    print(f"SPLIT-HALF CORRECTED ({R} splits, mean +- SD of the difference)")
    print("=" * 78)
    print(f"  {'regime':<11}{'model':<18}{'naive':>9}{'corrected':>12}"
          f"{'shift':>9}")
    for (rg, m) in nv:
        b, n, _ = nv[(rg, m)]
        v = np.array(acc[(rg, m)], float)
        v = v[np.isfinite(v)]
        print(f"  {rg:<11}{m:<18}{n - b:>+8.1f}"
              f"{v.mean():>+9.1f}+-{v.std():>3.1f}{v.mean() - (n - b):>+9.1f}")

    print()
    print(f"regime stability: {np.mean(flips):.1f} of {len(base)} sites change "
          f"side per split (SD {np.std(flips):.1f})")


if __name__ == "__main__":
    main()
