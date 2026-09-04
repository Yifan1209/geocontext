"""Fetch street-level imagery from Mapillary for arbitrary cities.

The original MMS-VPR data covers a single pilot site. Extending the
experiment to New York / Paris / Tokyo needs a source that can fetch imagery at
any coordinate.

## Why Mapillary rather than Google Street View

| | Mapillary | Google Street View |
|---|---|---|
| License | **CC-BY-SA, images can be repackaged and published** | redistribution forbidden, only coordinates can be shared |
| Reproducibility | images ship with the dataset, permanently reproducible | official docs say pano IDs change and should not be persisted |
| Barrier to entry | free token | requires a card on file |

A benchmark's value depends on whether someone else can run it in one line, so
licensing is the decisive factor.

## Known biases (must be stated in the paper)

1. **Almost daytime-only** -- crowdsourced dashcam data, very little at night.
   Cross-city experiments should fix a daytime condition.
2. **Vehicle-biased viewpoint** -- mostly a driving perspective, unlike
   MMS-VPR's pedestrian viewpoint. This is a confound in any cross-source
   comparison; absolute accuracy across two different sites cannot be compared
   directly.
3. **Uneven coverage** -- dense in developed cities, sparse elsewhere.

Needs MAPILLARY_TOKEN (free at https://www.mapillary.com/dashboard/developer).
"""
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import config
from .config import DATA, load_env
from .ladder import _cached, _get_json, haversine

IMAGES = DATA / "streetview"
META = DATA / "streetview_meta.csv"
API = "https://graph.mapillary.com/images"

#: Fields for the search step. Do **not** request thumb_2048_url here -- that
#: field needs a freshly signed URL per image, which makes dense cities like
#: Tokyo return a flat 500 "Please reduce the amount of data you're asking
#: for", and drops New York's result count from 42 to 3. Image URLs are
#: fetched one at a time after selection instead (see `thumb_url`).
FIELDS = "id,computed_geometry,captured_at,is_pano,compass_angle,camera_type"

#: Used when fetching a single image's URL.
FIELD_THUMB = "thumb_2048_url"


def _token() -> str:
    load_env()
    import os
    t = os.environ.get("MAPILLARY_TOKEN")
    if not t:
        raise RuntimeError(
            "MAPILLARY_TOKEN is missing. Get a free one at "
            "https://www.mapillary.com/dashboard/developer and add it to .env")
    return t


def bbox_around(lat: float, lon: float, radius_m: float = 300) -> str:
    """Bounding box centred on a point. Mapillary accepts only a bbox, not a
    circular radius."""
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * max(0.1, abs(math.cos(math.radians(lat)))))
    return f"{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}"


def _query_bbox(bbox: str, limit: int) -> list:
    url = API + "?" + urllib.parse.urlencode({
        "access_token": _token(), "fields": FIELDS, "bbox": bbox, "limit": limit})
    return _get_json(url, timeout=60).get("data", [])


def search(lat, lon, radius_m=300, limit=200, max_depth=3,
           verbose=False) -> pd.DataFrame:
    """Search street-view image metadata near a coordinate (does not download
    the images themselves).

    Mapillary accepts only a **bbox**, not a circular radius, and returns 500
    "Please reduce the amount of data you're asking for" when a bbox is too
    dense -- independent of `limit`, the server is objecting to how dense the
    area itself is (measured in Shibuya, Tokyo).

    An earlier version handled this by shrinking the radius and retrying
    globally, which **sacrifices coverage**: at 150m there are too few
    candidates left to satisfy the 40m minimum separation. This version instead
    does **quadtree recursion**: split an unresponsive area into four quadrants
    and query each separately, then merge. Coverage is unchanged; only the
    request count grows.
    """
    def fetch(clat, clon, r, depth):
        try:
            return _query_bbox(bbox_around(clat, clon, r), limit)
        except Exception as e:
            if "500" not in str(e) or depth >= max_depth:
                raise
            if verbose and depth == 0:
                print("      area too dense, recursing into quadrants")
            half = r / 2
            dlat = half / 111_320
            dlon = half / (111_320 * max(0.1, abs(math.cos(math.radians(clat)))))
            out = []
            for dy in (-1, 1):
                for dx in (-1, 1):
                    try:
                        out += fetch(clat + dy * dlat, clon + dx * dlon, half, depth + 1)
                    except Exception:
                        pass          # one quadrant failing does not affect the others
            return out

    data = fetch(lat, lon, radius_m, 0)
    seen, uniq = set(), []
    for d in data:                    # adjacent quadrants overlap; de-dup by id
        if d.get("id") not in seen:
            seen.add(d.get("id"))
            uniq.append(d)
    data = uniq
    if verbose:
        print(f"      found {len(data)} (radius {radius_m:.0f}m)")
    rows = []
    for d in data:
        g = (d.get("computed_geometry") or {}).get("coordinates")
        if not g:
            continue
        plon, plat = g[0], g[1]
        rows.append(dict(
            image_id=str(d["id"]), lat=plat, lon=plon,
            dist_m=round(haversine((lat, lon), (plat, plon)) * 1000, 1),
            captured_at=d.get("captured_at"),
            is_pano=bool(d.get("is_pano")),
            compass_angle=d.get("compass_angle"),
            camera_type=d.get("camera_type")))
    df = pd.DataFrame(rows)
    return df.sort_values("dist_m").reset_index(drop=True) if len(df) else df


def thumb_url(image_id: str) -> str | None:
    """Fetch one image's URL. Requesting this field in bulk during search
    overwhelms the server (see the FIELDS comment)."""
    url = (f"https://graph.mapillary.com/{image_id}?"
           + urllib.parse.urlencode({"access_token": _token(), "fields": FIELD_THUMB}))
    try:
        return _get_json(url, timeout=30).get(FIELD_THUMB)
    except Exception:
        return None


def geometry(image_id: str) -> tuple[float, float] | None:
    """Fetch one image's (lat, lon).

    The search DataFrame already carries coordinates, but early batches only
    persisted site/path and lost them -- a real problem for a geolocation
    benchmark, since no downstream "how far is this image from point X"
    computation can run without them. This function backfills it.

    Goes through `_cached`, so repeated calls do not re-hit the API.
    """
    def fetch():
        url = (f"https://graph.mapillary.com/{image_id}?"
               + urllib.parse.urlencode({"access_token": _token(),
                                         "fields": "computed_geometry,geometry"}))
        return _get_json(url, timeout=30)

    try:
        d = _cached(f"imggeom|{image_id}", fetch)
    except Exception:
        return None
    g = d.get("computed_geometry") or d.get("geometry") or {}
    c = g.get("coordinates")
    return (float(c[1]), float(c[0])) if c else None


def _hour_solar(captured_at_ms, lon) -> float | None:
    """Estimate **solar time** from a UTC timestamp and longitude. Reference
    only -- do not use this to classify day/night.

    Legal timezone is not solar time. France sits at longitude 2.36 but uses
    UTC+1/+2, and a country spanning five geographic time zones may still run
    on one legal offset, putting local solar time over an hour out. Using this
    to filter for daytime reduced
    185 Paris images to 1 (a UTC 06:57 photo was locally 08:57, but got
    classified as 7am). Use `brightness()` for day/night instead -- it looks
    at the image directly and makes no timezone assumption.
    """
    if not captured_at_ms:
        return None
    utc_h = (int(captured_at_ms) / 1000 / 3600) % 24
    return (utc_h + lon / 15.0) % 24


def brightness(path) -> float:
    """Mean image brightness, 0-255. Measures day/night directly, sidestepping
    timezone issues.

    Empirical thresholds: daytime street scenes are usually > 90, night scenes
    usually < 60.
    """
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L").resize((64, 64))
        px = list(im.getdata())
    return sum(px) / len(px)


def pick(df: pd.DataFrame, n=3, min_sep_m=40, camera="perspective",
         oversample=4, seed=42, verbose=True) -> pd.DataFrame:
    """Select candidates: restrict camera type and enforce separation between
    them. **Day/night filtering does not happen at this step.**

    `min_sep_m` is deliberate: Mapillary is a sequence of dashcam frames, and
    adjacent frames are nearly the same image (70% of MMS-VPR turned out to be
    video frame extraction -- a lesson learned the hard way). Skipping the
    separation is equivalent to sampling the same shot repeatedly.

    `camera="perspective"` excludes fisheye (heavy distortion) and
    equirectangular (full panorama). Measured: 71 of 200 Paris candidates were
    fisheye, 15 were panoramas.

    Oversamples by `oversample`x, leaving room for the brightness filter applied
    after download (see `collect`).
    """
    if df.empty:
        return df
    d = df.copy()
    n0 = len(d)
    if camera:
        d = d[d.camera_type == camera]
    if verbose:
        print(f"      camera-type filter {n0} -> {len(d)}", end="")
    if d.empty:
        return d

    chosen = []
    for _, r in d.sample(frac=1.0, random_state=seed).iterrows():
        if all(haversine((r.lat, r.lon), (c.lat, c.lon)) * 1000 >= min_sep_m
               for c in chosen):
            chosen.append(r)
        if len(chosen) >= n * oversample:
            break
    out = pd.DataFrame(chosen).reset_index(drop=True)
    if verbose:
        print(f", {len(out)} after >= {min_sep_m}m separation")
    out["hour_solar"] = [_hour_solar(t, lo) for t, lo in zip(out.captured_at, out.lon)]
    return out


def download(df: pd.DataFrame, site: str, dest: Path = None,
             verbose=False) -> pd.DataFrame:
    """Download the selected images, returning a table with local paths.

    Existing files are skipped; a failed download is dropped rather than
    aborting the whole run.
    """
    dest = (dest or IMAGES) / site
    dest.mkdir(parents=True, exist_ok=True)
    rows, paths = [], []
    for _, r in df.iterrows():
        p = dest / f"{r.image_id}.jpg"
        if not p.exists():
            u = getattr(r, "url", None) or thumb_url(r.image_id)
            if not u:
                if verbose:
                    print(f"      could not get a URL for {r.image_id}")
                continue
            try:
                with urllib.request.urlopen(u, timeout=60) as resp:
                    p.write_bytes(resp.read())
            except Exception as e:
                if verbose:
                    print(f"      download failed {r.image_id}: {type(e).__name__}")
                continue
            time.sleep(0.2)
        rows.append(r)
        paths.append(str(p.relative_to(DATA)))
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["site"] = site
    out["path"] = paths
    return out


def collect_panoramic(lat, lon, site, n=3, radius_m=300, min_sep_m=40,
                      crops_per_pano=2, min_brightness=90, seed=42,
                      verbose=True) -> pd.DataFrame:
    """Fallback path for a site with only 360-degree panoramas: download the
    panorama, then crop it into perspective views.

    One pilot site's Mapillary coverage turned out to be **100% spherical**
    (141 candidates, not one perspective shot), so without this conversion a
    site like that cannot be included at all.

    Cropped images are not equivalent to native perspective shots (field of
    view, distortion and the imaging pipeline all differ), so results carry
    `from_panorama=True` and must be treated as a covariate in analysis.
    """
    from . import panorama as pano

    found = search(lat, lon, radius_m, verbose=verbose)
    if found.empty:
        return pd.DataFrame()
    sph = found[found.camera_type.isin(["spherical", "equirectangular"])]
    if sph.empty:
        return pd.DataFrame()

    # Panoramas need separation too, otherwise adjacent frames crop to nearly
    # the same view.
    chosen = []
    for _, r in sph.sample(frac=1.0, random_state=seed).iterrows():
        if all(haversine((r.lat, r.lon), (c.lat, c.lon)) * 1000 >= min_sep_m
               for c in chosen):
            chosen.append(r)
        if len(chosen) >= -(-n // crops_per_pano) + 2:      # a couple extra as spares
            break
    if verbose:
        print(f"      {len(sph)} panoramas -> {len(chosen)} after "
              f">= {min_sep_m}m separation", flush=True)

    got = download(pd.DataFrame(chosen).reset_index(drop=True),
                   f"{site}_pano", verbose=verbose)
    dest = IMAGES / site
    dest.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in got.iterrows():
        src = DATA / r.path
        for yaw, im in pano.best_crops(src, n=crops_per_pano):
            out = dest / f"{r.image_id}_yaw{int(yaw):03d}.jpg"
            im.save(out, quality=92)
            b = brightness(out)
            if b < min_brightness:
                config.trash(out, f"panorama crop too dark: {b:.0f} < {min_brightness}")
                continue
            rows.append(dict(image_id=f"{r.image_id}_y{int(yaw):03d}",
                             lat=r.lat, lon=r.lon, dist_m=r.dist_m,
                             captured_at=r.captured_at, is_pano=False,
                             compass_angle=(r.compass_angle or 0) + yaw,
                             camera_type="spherical_crop",
                             site=site, path=str(out.relative_to(DATA)),
                             brightness=round(b, 1), from_panorama=True,
                             pano_id=r.image_id, crop_yaw=round(yaw, 1)))
    df = pd.DataFrame(rows)
    if verbose:
        print(f"      cropped {len(df)} perspective views "
              f"({crops_per_pano} directions per panorama)", flush=True)
    return df.head(n)


def collect(sites: dict, n=3, radius_m=300, min_brightness=90,
            verbose=True) -> pd.DataFrame:
    """One-stop pipeline: search -> select candidates -> download -> filter for
    daytime by brightness -> combine.

    sites: {"nyc_soho": (40.7233, -74.0030), ...}

    Day/night is decided **after** download, by image brightness rather than
    timestamp -- legal timezone and solar time can differ by two hours, and an
    earlier longitude-based estimate reduced 185 Paris images to 1. Downloading
    a few extra and filtering by brightness costs only a few hundred KB of
    bandwidth.
    """
    frames = []
    for site, (lat, lon) in sites.items():
        if verbose:
            print(f"  {site}")
        try:
            found = search(lat, lon, radius_m, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"      search failed: {type(e).__name__}: {str(e)[:90]}")
            continue
        if found.empty:
            if verbose:
                print("      no results")
            continue
        sel = pick(found, n=n, verbose=verbose)
        if sel.empty:
            # No perspective shots at this site (it happens: some sites are all
            # 141 candidates spherical) -- fall back to panorama cropping.
            if verbose:
                print("      no perspective shots, falling back to panorama crops",
                      flush=True)
            pano_df = collect_panoramic(lat, lon, site, n=n, radius_m=radius_m,
                                        min_brightness=min_brightness, verbose=verbose)
            if not pano_df.empty:
                frames.append(pano_df)
            continue

        got = download(sel, site, verbose=verbose)
        got["brightness"] = [round(brightness(DATA / p), 1) for p in got.path]
        got["from_panorama"] = False
        day = got[got.brightness >= min_brightness].head(n)
        if verbose:
            print(f"      downloaded {len(got)}, "
                  f"{len(got[got.brightness>=min_brightness])} >= brightness "
                  f"{min_brightness}, keeping {len(day)}; brightness range "
                  f"{got.brightness.min():.0f}-{got.brightness.max():.0f}")
        if day.empty:
            if verbose:
                print("      all too dark, skipped (this site may only have night data)")
            continue
        frames.append(day)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META, index=False, encoding="utf-8")
    return df
