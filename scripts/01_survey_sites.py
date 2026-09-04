"""Survey Mapillary coverage for candidate expansion sites — before committing to them.

    python scripts/01_survey_sites.py

Why this step exists first, learned the hard way on the pilot site: its
Mapillary coverage is
**100% spherical panoramas** (141/141), which had to be reprojected into
perspective crops before it was usable at all, and the reprojected images were
poor enough quality that the baseline error was 372 km. Before committing to a
new city, three things need checking, or the plan falls apart later:

1. **Is there coverage at all?**   zero coverage rules the site out
2. **Is it perspective or spherical?**  spherical needs reprojection and is
   lower quality -- the perspective share needs to be high enough
3. **How recent is it?**   street view older than ~5 years may show buildings
   that no longer exist

Site-selection principle (set by the user): **avoid each city's single most
iconic landmark** (the Eiffel Tower, the Arc de Triomphe, La Defense, Sagrada
Familia, the Zocalo, and similar). Two reasons:
  - too iconic pushes baseline accuracy to the ceiling, leaving no room to
    discriminate between models
  - the failure mode under study is precisely "the model substitutes the most
    iconic nearby landmark" -- if the target itself already is that landmark,
    the failure has nowhere to show up

NOTE: this script **only measures coverage, it does not download images**.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import streetview  # noqa: E402

pd.set_option("display.width", 200)

from geocontext.sites import CANDIDATES  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, default=300, help="search radius, metres")
    ap.add_argument("--only", nargs="+", help="survey only these sites")
    args = ap.parse_args()

    items = {k: v for k, v in CANDIDATES.items()
             if not args.only or k in args.only}
    rows = []
    for name, (lat, lon, note) in items.items():
        try:
            df = streetview.search(lat, lon, radius_m=args.radius, limit=200)
        except Exception as e:
            print(f"  {name:22} search failed {type(e).__name__}: {e}", flush=True)
            rows.append(dict(site=name, total=0, perspective=0, spherical=0,
                             pct_perspective=0.0, year_max=None, year_median=None,
                             note=note))
            continue
        if df is None or len(df) == 0:
            print(f"  {name:22} no coverage", flush=True)
            rows.append(dict(site=name, total=0, perspective=0, spherical=0,
                             pct_perspective=0.0, year_max=None, year_median=None,
                             note=note))
            continue

        pano = (df.is_pano.fillna(False).astype(bool) if "is_pano" in df
                else pd.Series(False, index=df.index))
        yr = None
        if "captured_at" in df:
            # captured_at is a millisecond timestamp
            t = pd.to_datetime(pd.to_numeric(df.captured_at, errors="coerce"),
                               unit="ms", errors="coerce")
            yr = t.dt.year
        rows.append(dict(
            site=name, total=len(df),
            perspective=int((~pano).sum()), spherical=int(pano.sum()),
            pct_perspective=round(100 * (~pano).mean(), 1),
            year_max=int(yr.max()) if yr is not None and yr.notna().any() else None,
            year_median=int(yr.median()) if yr is not None and yr.notna().any() else None,
            note=note))
        print(f"  {name:22} {len(df):3} total, {int((~pano).sum()):3} perspective "
              f"({100*(~pano).mean():.0f}%)", flush=True)

    out = pd.DataFrame(rows)
    print("\n=== coverage survey (radius %d m, counting only, nothing downloaded) ===" % args.radius)
    print(out.to_string(index=False))

    print("\n=== verdict ===")
    # Judge by the **absolute count** of perspective images, not the share.
    # sf_northbeach is the counter-example: only 21% of 2074 images are
    # perspective, but that is still 441 images, and each site only needs 3-5.
    # A low share just means that area has many panorama-only cameras and says
    # nothing about usability.
    print("usable if perspective images >= 30 (each site only needs 3-5, share does not matter)")
    ok = out[out.perspective >= 30]
    bad = out[~out.site.isin(ok.site)]
    print(f"\n  usable ({len(ok)}): {list(ok.site)}")
    print(f"  not usable ({len(bad)}): {list(bad.site)}")


if __name__ == "__main__":
    main()
