"""Build the trial table for GeoVerify, the location-claim verification task.

    python bin/build_verify_trials.py --out data/verify_trials.csv

## What this task measures

GeoHint, the open-ended task, asks the model to name a place. Scoring it means
geocoding free text, hoping the geocode resolves at a usable granularity, and
picking a distance threshold. All four of the scoring faults documented in the
paper live in that chain -- including a city-centroid fallback that happened to
land 0.761 km from one site, under the 1 km hit threshold, which inflated that
site's accuracy by 20.7 points and reversed the headline result.

GeoVerify asks a binary question instead:

    "Was this photo taken within 150 metres of {X}?"  -> yes / no + confidence

Ground truth is mechanical: `haversine(image_gps, candidate_gps) < 0.15`.
No geocoder, no judge model, no tunable threshold.

The 150 m tolerance is not arbitrary. It is the order of magnitude at which
ride-hailing and delivery platforms treat an arrival as complete -- i.e. the
scenario where a driver drops a passenger at the wrong place, spoofs GPS, and
the platform asks for a photo to confirm the drop-off.

## Trial structure

| Type     | Candidate                                          | Correct answer |
|----------|----------------------------------------------------|----------------|
| `signal` | a real landmark within 150 m of the image          | yes            |
| `noise`  | a decoy at 0.15-0.3 / 0.3-0.7 / 0.7-1.5 / 1.5-3 / 3-6 km | no       |

Hit rate H = P(yes | signal) and false-alarm rate F(d) = P(yes | decoy at d)
are tracked separately, then combined into sensitivity d' = z(H) - z(F(d)).
Raw accuracy would be gamed by response bias: a model that always answers "no"
scores perfectly on every noise trial while having no spatial discrimination
at all. d' separates ability from bias.

## Two controls, without which the results cannot be interpreted

- `mismatch`: same candidate, but the image is swapped for street-level imagery
  from a different continent. F should be near zero here. If it is not, the
  model is reasoning from the candidate's name rather than from the photograph,
  and the task does not measure what it claims to.
- `noimage`: a uniform grey field in place of the photo. Bounds what can be
  achieved from the name alone, with the request shape held constant.

## Decoy selection

Only audited candidates (`ok=True`) are used, and decoys are matched to the
signal candidate on referenceability (within one tier) wherever the cell allows
it. Without that matching a model could reject a decoy on type alone --
"this is a famous museum and I see no museum" is a category judgement, not a
spatial one.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext import config, streetview  # noqa: E402
from geocontext.ladder import haversine  # noqa: E402

#: Arrival tolerance, matching what delivery / ride-hailing platforms use.
TOL_KM = 0.15

#: Decoy distance bands (km). The nearest band starts right above the
#: tolerance on purpose: "dropped off one street away" is both the hardest
#: discrimination and the one a real deployment actually has to make.
BANDS = [(0.15, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 6.0)]

N_PER_BAND = 3          # decoys drawn per image per band
SEED = 42


def image_table() -> pd.DataFrame:
    """Selected evaluation images, with GPS backfilled from Mapillary."""
    m = pd.read_csv(config.DATA / "streetview_meta_selected.csv")
    m["image_id"] = m.path.str.replace("\\", "/", regex=False).str.split("/").str[-1]
    m["image_id"] = m.image_id.str.replace(".jpg", "", regex=False)
    lat, lon = [], []
    for i, r in enumerate(m.itertuples(), 1):
        g = streetview.geometry(r.image_id)
        lat.append(g[0] if g else np.nan)
        lon.append(g[1] if g else np.nan)
        if i % 10 == 0:
            print(f"  backfilling coordinates {i}/{len(m)}", flush=True)
    m["lat"], m["lon"] = lat, lon
    bad = m.lat.isna().sum()
    if bad:
        print(f"  WARNING: {bad} images have no retrievable coordinate, dropped")
    return m.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def candidates(site: str) -> pd.DataFrame:
    """Audited candidate POIs for one site."""
    hits = sorted((config.LADDERS).glob(f"{site}_candidates_*.csv"))
    if not hits:
        return pd.DataFrame()
    d = pd.read_csv(hits[0])
    d = d[d.ok.fillna(False).astype(bool)]
    # English names only at this stage -- v1 runs no Chinese-language condition.
    d["name"] = d.name_en.fillna("").astype(str).str.strip()
    d = d[d.name.str.len() > 0]
    return d.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/verify_trials.csv")
    ap.add_argument("--n-per-band", type=int, default=N_PER_BAND)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    imgs = image_table()
    print(f"{len(imgs)} images across {imgs.site.nunique()} sites\n")

    rows = []
    for r in imgs.itertuples():
        cand = candidates(r.site)
        if cand.empty:
            print(f"  {r.site}: no candidates, skipped")
            continue
        # Distances are recomputed against THIS image, not against the site
        # centre that the candidate table's own dist_km refers to.
        cand = cand.assign(d_img=[haversine((r.lat, r.lon), (c.lat, c.lon))
                                  for c in cand.itertuples()])

        sig = cand[cand.d_img < TOL_KM]
        # Referenceability anchor for decoy matching. When an image has no
        # signal candidate nearby, fall back to the pool median -- do NOT skip
        # the image, since noise trials do not require a signal trial to exist.
        fam0 = (float(sig.sort_values("d_img").iloc[0].familiarity)
                if len(sig) else float(cand.familiarity.median()))

        for s in sig.itertuples():
            rows.append(dict(image_id=r.image_id, site=r.site, path=r.path,
                             img_lat=r.lat, img_lon=r.lon,
                             cand=s.name, cand_lat=s.lat, cand_lon=s.lon,
                             fam=s.familiarity, d_km=round(s.d_img, 4),
                             band="signal", truth="yes", arm="main"))

        for lo, hi in BANDS:
            pool = cand[(cand.d_img >= lo) & (cand.d_img < hi)]
            near = pool[(pool.familiarity - fam0).abs() <= 1]
            pool = near if len(near) >= args.n_per_band else pool
            if pool.empty:
                continue
            take = pool.sample(min(args.n_per_band, len(pool)),
                               random_state=int(rng.integers(1e6)))
            for s in take.itertuples():
                rows.append(dict(image_id=r.image_id, site=r.site, path=r.path,
                                 img_lat=r.lat, img_lon=r.lon,
                                 cand=s.name, cand_lat=s.lat, cand_lon=s.lon,
                                 fam=s.familiarity, d_km=round(s.d_img, 4),
                                 band=f"{lo}-{hi}km", truth="no", arm="main"))

    t = pd.DataFrame(rows)
    if t.empty:
        print("no trials generated")
        return

    # --- Control 1: mismatched image, from a different continent ---
    # continent code "AN", not "NA": pandas.read_csv treats the literal
    # string "NA" as a missing value by default, and this column is written
    # straight to verify_trials.csv -- a plain read of that file would
    # silently blank it for these rows.
    CONT = {"paris": "EU", "london": "EU", "barcelona": "EU",
            "tokyo": "AS", "sf": "AN", "cdmx": "AN", "nyc": "AN"}
    t["cont"] = t.site.str.split("_").str[0].map(CONT)
    mism = []
    for r in t[t.arm == "main"].sample(min(200, len(t)), random_state=SEED).itertuples():
        other = imgs[imgs.site.str.split("_").str[0].map(CONT) != r.cont]
        if other.empty:
            continue
        o = other.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        mism.append(dict(image_id=o.image_id, site=o.site, path=o.path,
                         img_lat=o.lat, img_lon=o.lon,
                         cand=r.cand, cand_lat=r.cand_lat, cand_lon=r.cand_lon,
                         fam=r.fam,
                         d_km=round(haversine((o.lat, o.lon),
                                              (r.cand_lat, r.cand_lon)), 3),
                         band="mismatch", truth="no", arm="ctrl_mismatch",
                         cont=None))
    # --- Control 2: no image, candidate name only ---
    noimg = []
    for r in t[t.arm == "main"].sample(min(200, len(t)),
                                       random_state=SEED + 1).itertuples():
        noimg.append(dict(image_id=r.image_id, site=r.site, path="",
                          img_lat=r.img_lat, img_lon=r.img_lon,
                          cand=r.cand, cand_lat=r.cand_lat, cand_lon=r.cand_lon,
                          fam=r.fam, d_km=r.d_km, band=r.band, truth=r.truth,
                          arm="ctrl_noimage", cont=None))

    t = pd.concat([t, pd.DataFrame(mism), pd.DataFrame(noimg)], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n=== {len(t)} trials written to {out} ===")
    print(t.groupby(["arm", "band"]).size().to_string())
    print(f"\nmain arm: {(t[(t.arm=='main')&(t.truth=='yes')]).shape[0]} signal, "
          f"{(t[(t.arm=='main')&(t.truth=='no')]).shape[0]} noise")
    print(f"covering {t.site.nunique()} sites and {t.image_id.nunique()} images")
    print(f"median signal distance {t[t.band=='signal'].d_km.median():.3f} km")


if __name__ == "__main__":
    main()
