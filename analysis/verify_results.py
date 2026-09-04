"""Score GeoVerify with signal detection theory.

    python bin/verify_results.py

## Read-out criteria, fixed before the run and not changed afterwards

| Observation | Conclusion |
|---|---|
| `ctrl_mismatch` false alarms > 20% | **Task invalid** -- the image was swapped for another continent and the model still says yes, so it is reading the name, not the photo |
| `ctrl_noimage` false alarms within 5 pts of the main arm | **Task invalid** -- the image contributes no information |
| Nearest band (0.15-0.3 km) d' > 2 | **Kill it** -- models already solve this, there is no benchmark here |
| Nearest band d' < 0.5 **and** farthest band (3-6 km) d' > 1.5 | **Informative** -- discrimination rises with distance, the curve carries signal |

## Why d' rather than accuracy

A model that answers "no" to everything scores perfectly on every noise trial
while having no spatial discrimination whatsoever, and the signal-to-noise
ratio of the trial set is a property of our sampling rather than of the model.
Signal detection theory separates the two: d' = z(H) - z(F) measures
**discriminability**, and criterion c = -(z(H)+z(F))/2 measures **response
bias** -- how willing the model is to say yes at all. H is the hit rate on
signal trials (one value per model); F(d) is the false-alarm rate per distance
band.

d' = 0 means the model is exactly as likely to confirm a candidate 200 m away
as the true one, whatever its raw accuracy happens to be.

WARNING: the pilot has only 33 signal trials per model (limited by how many
audited, English-named candidates fall within 150 m of an image), so z(H)
carries a large standard error that propagates equally into every d'. That is
enough for a go/no-go decision and **not** enough for publication -- the full
run needs more near-field POIs mined around each image.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from geocontext import config  # noqa: E402

RAW = config.DATA / "verify_raw.jsonl"
BAND_ORDER = ["0.15-0.3km", "0.3-0.7km", "0.7-1.5km", "1.5-3.0km", "3.0-6.0km"]

pd.set_option("display.width", 200)


def parse(raw: str) -> tuple[str | None, float | None]:
    """Extract verdict and confidence from a model reply.

    Models like to wrap JSON in ```json fences and occasionally add prose
    around it. Strip fences, find the first balanced brace block, and fall back
    to a regex on "verdict" if that fails.
    """
    if not raw:
        return None, None
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i = s.find("{")
    if i >= 0:
        depth = 0
        for j, ch in enumerate(s[i:], i):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                try:
                    d = json.loads(s[i:j + 1])
                    v = str(d.get("verdict", "")).strip().lower()
                    c = d.get("confidence")
                    return (v if v in ("yes", "no") else None,
                            float(c) if isinstance(c, (int, float)) else None)
                except Exception:
                    break
    m = re.search(r'"verdict"\s*:\s*"(yes|no)"', s, re.I)
    return (m.group(1).lower() if m else None), None


def load() -> pd.DataFrame:
    rows = []
    for line in RAW.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("error"):
            continue
        v, c = parse(d.get("raw"))
        rows.append({**{k: d[k] for k in ("trial_id", "image_id", "site", "arm",
                                          "band", "cand", "truth", "d_km",
                                          "fam", "model")},
                     "verdict": v, "conf": c})
    df = pd.DataFrame(rows).drop_duplicates(subset=["trial_id", "model"])
    df["yes"] = df.verdict == "yes"
    return df


def dprime(h, f, nh, nf):
    """Sensitivity and criterion, with the log-linear correction.

    z(.) diverges at rates of 0 or 1, so the standard fix (Hautus 1995) adds
    0.5 to each cell count and 1 to each total before taking the rate.
    """
    h = (h * nh + 0.5) / (nh + 1)
    f = (f * nf + 0.5) / (nf + 1)
    return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2


def main():
    df = load()
    bad = df.verdict.isna().sum()
    print(f"{len(df)} records, {bad} unparseable ({100*bad/len(df):.1f}%)")
    df = df[df.verdict.notna()]
    print(f"models: {sorted(df.model.unique())}")
    print(f"{df.site.nunique()} sites, {df.image_id.nunique()} images\n")

    main_arm = df[df.arm == "main"]

    print("=" * 92)
    print("CONTROLS -- if these fail, nothing below is worth reading")
    print("=" * 92)
    for arm, note in [("ctrl_mismatch", "image swapped to another continent"),
                      ("ctrl_noimage", "uniform grey field, no visual evidence")]:
        g = df[df.arm == arm]
        if g.empty:
            continue
        n = g[g.truth == "no"]
        print(f"{arm:16} false alarms {int(n.yes.sum()):4}/{len(n):4} = "
              f"{100*n.yes.mean():5.1f}%   ({note})")
    mn = main_arm[main_arm.truth == "no"]
    print(f"{'main':16} false alarms {int(mn.yes.sum()):4}/{len(mn):4} = "
          f"{100*mn.yes.mean():5.1f}%   (reference)")

    print("\n" + "=" * 92)
    print("Hit rate H  (signal trials: candidate really is within 150 m)")
    print("=" * 92)
    H = {}
    for m, g in main_arm[main_arm.truth == "yes"].groupby("model"):
        H[m] = (g.yes.mean(), len(g))
        print(f"  {m:18} {int(g.yes.sum()):3}/{len(g):3} = {100*g.yes.mean():5.1f}%")

    print("\n" + "=" * 92)
    print("False-alarm rate F(d) and sensitivity d'  -- raw counts / d' per cell")
    print("=" * 92)
    rows = []
    for m, g in main_arm[main_arm.truth == "no"].groupby("model"):
        h, nh = H.get(m, (np.nan, 0))
        rec = {"model": m}
        for b in BAND_ORDER:
            gb = g[g.band == b]
            if gb.empty:
                continue
            f, nf = gb.yes.mean(), len(gb)
            d, c = dprime(h, f, nh, nf)
            rec[b] = f"{int(gb.yes.sum())}/{nf}={100*f:.0f}%  d'={d:.2f}"
        rows.append(rec)
    print(pd.DataFrame(rows).set_index("model").to_string())

    print("\n" + "=" * 92)
    print("Criterion c per model -- response bias, negative = biased toward "
          "accepting")
    print("=" * 92)
    rows = []
    for m, g in main_arm[main_arm.truth == "no"].groupby("model"):
        h, nh = H.get(m, (np.nan, 0))
        rec = {"model": m}
        for b in BAND_ORDER:
            gb = g[g.band == b]
            if gb.empty:
                continue
            rec[b] = round(dprime(h, gb.yes.mean(), nh, len(gb))[1], 2)
        rows.append(rec)
    print(pd.DataFrame(rows).set_index("model").to_string())

    print("\n" + "=" * 92)
    print("Controls per model -- the pooled figure hides the spread")
    print("=" * 92)
    for arm, note in [("ctrl_mismatch", "image swapped to another continent"),
                      ("ctrl_noimage", "uniform grey field")]:
        print(f"  {note}:")
        for m, g in df[(df.arm == arm) & (df.truth == "no")].groupby("model"):
            print(f"    {m:18} {int(g.yes.sum()):3}/{len(g):3} = "
                  f"{100 * g.yes.mean():5.1f}%")

    print("\n" + "=" * 92)
    print("Pooled d' curve and criterion c (all models combined)")
    print("=" * 92)
    hh = main_arm[(main_arm.truth == "yes")]
    h, nh = hh.yes.mean(), len(hh)
    print(f"  pooled hit rate H = {int(hh.yes.sum())}/{nh} = {100*h:.1f}%")
    for b in BAND_ORDER:
        gb = main_arm[(main_arm.truth == "no") & (main_arm.band == b)]
        if gb.empty:
            continue
        f, nf = gb.yes.mean(), len(gb)
        d, c = dprime(h, f, nh, nf)
        print(f"  {b:12} F = {int(gb.yes.sum()):3}/{nf:3} = {100*f:5.1f}%   "
              f"d' = {d:5.2f}   c = {c:5.2f}")

    print("\n" + "=" * 92)
    print("Self-reported confidence: how sure is the model when it false-alarms?")
    print("=" * 92)
    fa = main_arm[(main_arm.truth == "no") & main_arm.yes & main_arm.conf.notna()]
    hit = main_arm[(main_arm.truth == "yes") & main_arm.yes & main_arm.conf.notna()]
    if len(fa):
        print(f"  median confidence on false alarms {fa.conf.median():.2f} (n={len(fa)})")
        print(f"  median confidence on hits         {hit.conf.median():.2f} (n={len(hit)})")
        print(f"  false alarms with confidence >= 0.8: {100*(fa.conf>=.8).mean():.1f}%")

    print("\n" + "=" * 92)
    print("READ-OUT")
    print("=" * 92)
    mm = df[(df.arm == "ctrl_mismatch") & (df.truth == "no")]
    near = main_arm[(main_arm.truth == "no") & (main_arm.band == BAND_ORDER[0])]
    far = main_arm[(main_arm.truth == "no") & (main_arm.band == BAND_ORDER[-1])]
    d_near = dprime(h, near.yes.mean(), nh, len(near))[0] if len(near) else np.nan
    d_far = dprime(h, far.yes.mean(), nh, len(far))[0] if len(far) else np.nan
    if len(mm) and mm.yes.mean() > .20:
        print(f"  FAIL: mismatch-control false alarms {100*mm.yes.mean():.1f}% > 20% "
              "-- task invalid")
    elif d_near > 2:
        print(f"  FAIL: nearest band d'={d_near:.2f} > 2 -- models already solve this")
    elif d_near < 0.5 and d_far > 1.5:
        print(f"  PASS: nearest band d'={d_near:.2f} < 0.5, "
              f"farthest band d'={d_far:.2f} > 1.5")
    else:
        print(f"  INCONCLUSIVE: nearest d'={d_near:.2f}, farthest d'={d_far:.2f} "
              "-- outside the pre-registered envelope, inspect per-model tables")


if __name__ == "__main__":
    main()
