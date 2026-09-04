"""Measure the characteristic size of a named neighbourhood, per city (OSM Overpass).

    python analysis/district_scale.py

Why measure this: a fixed-kilometre distance band is not comparable across
cities -- Toranomon is 4 km from Shibuya and in Tokyo that is "a completely
different place", while the same 4 km in San Jose might not even leave one
neighbourhood. The scale that matters is not POI density (measured nearly
identical across four sites, 1.15-1.67 km) but **how big the neighbourhood
itself is**.

Method: take every OSM `place=neighbourhood|suburb|quarter` node within radius
R of the centre point, and compute the median nearest-neighbour distance among
them = the characteristic neighbourhood spacing L_district.

San Jose is the control (the counter-example raised during design discussion).

WARNING: **addressing units must be excluded first** (Japanese "X-chome",
Mexican Unidad Habitacional). Without that filter the numbers falsely suggest
"Tokyo's neighbourhood grain is twice as fine as Paris's" -- that is purely a
difference in OSM tagging convention. See the comment at `_ADDRESS_UNIT`.
"""
import re
import sys
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geocontext.config import DATA  # noqa: E402
from geocontext.ladder import haversine  # noqa: E402

#: Public Overpass instances 429/504 often. Rotate mirrors + back off, or a
#: single run failing partway loses even the cities that already succeeded.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    # overpass.osm.jp removed: SSL hostname mismatch, fails every time after a
    # wasted 30s wait.
    "https://overpass.private.coffee/api/interpreter",
]
CACHE = DATA / "cache"
UA = {"User-Agent": ("geo-context-research/0.1 "
                     "(https://github.com/; powerfan233@gmail.com) python-urllib")}

#: The four original experimental sites plus San Jose (the counter-example
#: raised during design discussion, used only to validate the scale definition).
CENTERS = {
    # batch 1 experimental sites
    "nyc_soho":         (40.7233, -74.0030),
    "paris_marais":     (48.8590, 2.3620),
    "tokyo_shibuya":    (35.6595, 139.7005),
    "san_jose_ref":     (37.3350, -121.8900),   # control, not an experimental site
    # batch 2 expansion sites (coordinates match geocontext.sites.SITES)
    "paris_montmartre":   (48.8840, 2.3380),
    "paris_bastille":     (48.8530, 2.3690),
    "paris_cite":         (48.8560, 2.3410),
    "paris_canal":        (48.8710, 2.3660),
    "london_shoreditch":  (51.5250, -0.0780),
    "london_covent":      (51.5120, -0.1230),
    "london_nottinghill": (51.5150, -0.2050),
    "barcelona_gracia":   (41.4030, 2.1560),
    "barcelona_born":     (41.3850, 2.1810),
    "sf_mission":         (37.7600, -122.4190),
    "sf_northbeach":      (37.8000, -122.4090),
    "sf_hayes":           (37.7760, -122.4240),
    "cdmx_roma":          (19.4190, -99.1620),
    "cdmx_condesa":       (19.4110, -99.1710),
    "tokyo_shimokita":    (35.6610, 139.6680),
    "tokyo_yanaka":       (35.7270, 139.7660),
}
RADIUS_M = 6000
PLACE_KINDS = "neighbourhood|suburb|quarter"


# --------------------------------------------------------------- addressing-unit filter
#
# OSM's place=neighbourhood tagging convention varies hugely by country, and
# counting it naively produces a false conclusion. Measured: Tokyo Shibuya has
# 800 "neighbourhoods" within 6 km, and **573 of them (72%) are "X-chome"** --
# address-numbering blocks, not neighbourhoods in the perceptual sense. Without
# filtering, Tokyo's L=0.24 km looks twice as fine as Paris's (0.51), producing
# the **false conclusion** "Tokyo's neighbourhood grain is finer".
#
# After filtering, Tokyo's three sites become 0.34 / 0.44 / 0.58, the same
# order of magnitude as every other city; the spread across all 19
# experimental sites drops from a 3.12x range to 1.98x.
#
# Mexico's Unidad Habitacional / Fraccionamiento (individual housing
# developments) are excluded for the same reason. This list is **certainly
# incomplete** -- when adding a new country, re-check a sample of `name` values
# by eye.
_ADDRESS_UNIT = re.compile(
    r"([一二三四五六七八九十\d]+丁目$"          # Japanese address block ("-chome")
    r"|^Unidad Habitacional"                   # Mexican housing development
    r"|^Fraccionamiento)")                     # Mexican housing subdivision


def is_address_unit(name: str) -> bool:
    return bool(_ADDRESS_UNIT.search(str(name or "")))


def fetch(lat, lon, radius_m=RADIUS_M):
    key = CACHE / f"overpass_place_{lat:.4f}_{lon:.4f}_{radius_m}.json"
    if key.exists():
        return json.loads(key.read_text(encoding="utf-8"))
    q = (f'[out:json][timeout:90];'
         f'node["place"~"^({PLACE_KINDS})$"]'
         f'(around:{radius_m},{lat},{lon});out body;')
    last = None
    for attempt in range(6):
        url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        req = urllib.request.Request(
            url, data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
        try:
            # 2026-09-02: timeout dropped from 180s to 40s -- a request that is
            # genuinely stuck is rare, almost every failure returns 429/504
            # within seconds, so 180s was mostly wasted keeping a truly
            # unresponsive request alive (a city hitting this a few times in a
            # row could burn 10+ minutes). 40s leaves enough margin for a
            # normal response while still cycling to the next mirror faster
            # when something really is stuck.
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read().decode())
            CACHE.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            time.sleep(5)      # a public instance, don't hammer it
            return d
        except Exception as e:
            last = e
            wait = 5 * (attempt + 1)
            print(f"      {url.split('/')[2]} failed {type(e).__name__}, "
                  f"retrying with the next mirror in {wait}s", flush=True)
            time.sleep(wait)
    raise last


def nn_stats(pts):
    if len(pts) < 3:
        return {}
    d = [min(haversine(a, b) for j, b in enumerate(pts) if j != i)
         for i, a in enumerate(pts)]
    return dict(nn_median=float(np.median(d)), nn_mean=float(np.mean(d)))


def main():
    rows = []
    for name, (lat, lon) in CENTERS.items():
        try:
            d = fetch(lat, lon)
        except Exception as e:
            print(f"  {name}: Overpass failed {type(e).__name__}: {e}", flush=True)
            continue
        els = [e for e in d.get("elements", []) if "lat" in e and "lon" in e]
        n_raw = len(els)
        # Keep only perceptual neighbourhoods, excluding address/housing units
        # -- see the _ADDRESS_UNIT comment.
        els = [e for e in els if not is_address_unit(e.get("tags", {}).get("name"))]
        pts = [(e["lat"], e["lon"]) for e in els]
        s = nn_stats(pts)
        # neighbourhood count density: count within radius / area
        area = np.pi * (RADIUS_M / 1000) ** 2
        rows.append(dict(city=name, n_neighbourhoods=len(pts), n_raw=n_raw,
                         per_sqkm=round(len(pts) / area, 3),
                         L_district=round(s.get("nn_median", np.nan), 2),
                         nn_mean=round(s.get("nn_mean", np.nan), 2)))
        print(f"  {name}: {len(pts)} neighbourhoods"
              f" (raw {n_raw}, {n_raw-len(pts)} addressing units excluded)", flush=True)

    df = pd.DataFrame(rows).sort_values("L_district")
    out = DATA / "district_scale.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"written to {out}")
    print("\n=== characteristic neighbourhood spacing L_district (km, within 6 km radius) ===")
    print(df.to_string(index=False))

    ref = df[df.city == "san_jose_ref"]
    if not ref.empty and np.isfinite(ref.L_district.iloc[0]):
        Lsj = ref.L_district.iloc[0]
        print(f"\nSan Jose L_district = {Lsj:.2f} km, other sites relative to it:")
        for r in df[df.city != "san_jose_ref"].itertuples():
            if np.isfinite(r.L_district):
                print(f"  {r.city:18} {r.L_district:5.2f} km  = {r.L_district/Lsj:.2f}x San Jose")
        print(f"\nnormalised by L_district, the current 3-6 km far band corresponds to, per city:")
        for r in df.itertuples():
            if np.isfinite(r.L_district):
                print(f"  {r.city:18} {3/r.L_district:.1f}-{6/r.L_district:.1f} neighbourhood spacings")


if __name__ == "__main__":
    main()
