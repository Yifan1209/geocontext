"""Pick photo sites: Wikidata landmark discovery + street-view coverage check.

    python scripts/00_select_sites.py --per-city 3 --out data/site_pool.csv
    python scripts/00_select_sites.py --cities London Paris --per-city 3

## What this produces, and what it does not

This picks *candidate* sites for a city already confirmed dense/walkable
enough to be in scope (see `L_district` in the paper and the README's Scope
section) -- it is not a city screener. `CITIES` below is exactly the shortlist
that passed that screening (median nearest-neighbour spacing among OSM
`place=neighbourhood|suburb|quarter` nodes, 0.32-0.65 km within 6 km); the
screening script itself is a research-process tool (Overpass-heavy, hours to
run against public mirrors) kept in the private project rather than shipped
here, since reproducing this exact 29-city table does not require rerunning
it.

## Selection criteria

**Type** (hard constraint, for a fine-grained landmark rather than a whole
district): museum / park / church / theatre / market / bridge / square /
library / castle and similar entity types with a physical footprint.

**Fame ceiling** `sitelinks <= 60`: cut only the extreme top. An earlier
attempt at a *middle* sitelinks band (5-40, to select "known but not
iconic") failed to backtest against 1507 already-audited candidates: that
band covers only 67.5% of high-referenceability candidates, and every
referenceability level appears throughout it. Sitelinks correlates only
weakly with referenceability -- useful for excluding extremes, not for
picking a middle. True legibility stratification is measured after the
fact from baseline hit rate, not engineered in at selection time.

**Street-view coverage**: >= 8 perspective images within 150 m of the point.
The most expensive step (one Mapillary query per candidate), so it runs
last, over a **shuffled** candidate order.

## Why shuffled, not sorted by fame

A first pass ordered candidates `DESC(sitelinks)` and took the first N with
coverage -- which is "the most famous eligible landmark in this city",
mechanically, regardless of the coverage check. That produced a batch of
almost entirely postcard landmarks (Anne Frank House, Brandenburg Gate, the
Van Gogh Museum), all comfortably under any reasonable sitelinks cap, so a
tighter cap alone would not have fixed it -- the bias was in the traversal
order, not the threshold. Shuffling before the coverage check removes the
systematic "always the single most iconic instance" bias; it does not
guarantee no famous site is ever picked (a shuffle can land on one), and it
does not replace the post-hoc legibility measurement above.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import streetview  # noqa: E402
from geocontext.ladder import _get_json, haversine, _cached  # noqa: E402,F401

SPARQL = "https://query.wikidata.org/sparql"

#: (city, country, continent, centre lat, centre lon). Exactly the cities
#: that passed the L_district screen described in the paper (0.32-0.65 km).
CITIES = [
    ("London", "UK", "EU", 51.5100, -0.1300),
    ("Paris", "FR", "EU", 48.8600, 2.3400),
    ("Barcelona", "ES", "EU", 41.3870, 2.1700),
    ("Amsterdam", "NL", "EU", 52.3700, 4.8950),
    ("Vienna", "AT", "EU", 48.2080, 16.3730),
    ("Lisbon", "PT", "EU", 38.7100, -9.1400),
    ("Berlin", "DE", "EU", 52.5170, 13.3900),
    ("Madrid", "ES", "EU", 40.4160, -3.7040),
    ("Milan", "IT", "EU", 45.4650, 9.1900),
    ("Budapest", "HU", "EU", 47.4980, 19.0400),
    ("Copenhagen", "DK", "EU", 55.6800, 12.5800),
    ("Athens", "GR", "EU", 37.9760, 23.7350),
    ("Stockholm", "SE", "EU", 59.3300, 18.0700),
    ("Edinburgh", "UK", "EU", 55.9500, -3.1900),
    ("Singapore", "SG", "AS", 1.3000, 103.8500),
    ("Hong Kong", "HK", "AS", 22.2800, 114.1700),
    ("Tel Aviv", "IL", "AS", 32.0700, 34.7750),
    ("Hanoi", "VN", "AS", 21.0300, 105.8500),
    # continent code "AN", not "NA": pandas.read_csv treats the literal
    # string "NA" as a missing value by default, which silently blanked
    # this column for every North American row the first time this table
    # round-tripped through pandas. AN/EU/AS/SA/AF are not in pandas'
    # default na_values list, so this sidesteps the whole bug class instead
    # of relying on every future read remembering keep_default_na=False.
    ("New York", "US", "AN", 40.7400, -73.9900),
    ("San Francisco", "US", "AN", 37.7800, -122.4150),
    ("Boston", "US", "AN", 42.3570, -71.0600),
    ("Toronto", "CA", "AN", 43.6500, -79.3850),
    ("Mexico City", "MX", "AN", 19.4300, -99.1500),
    ("Santiago", "CL", "SA", -33.4400, -70.6500),
    ("Lima", "PE", "SA", -12.0500, -77.0400),
    ("Bogota", "CO", "SA", 4.6000, -74.0800),
    ("Marrakech", "MA", "AF", 31.6300, -8.0000),
    ("Mumbai", "IN", "AS", 18.9350, 72.8350),
    ("Delhi", "IN", "AS", 28.6330, 77.2190),
]

#: Photographable entity classes. Deliberately excludes administrative
#: divisions (those are district names, too coarse-grained).
#: Q ids: museum/park/church/theatre/market/bridge/square/library/castle/
#:        botanical garden/monument/railway station/university/stadium/
#:        fountain/clock tower
TYPES = ["Q33506", "Q22698", "Q16970", "Q24354", "Q330284", "Q12280",
         "Q174782", "Q7075", "Q23413", "Q167346", "Q4989906",
         "Q55488", "Q3918", "Q483110", "Q483453", "Q200334"]

MAX_SITELINKS = 60
MIN_SV = 8
SV_RADIUS = 120


#: Leading articles: on their own they say nothing about the landmark, so
#: they are filtered out before slug generation. Covers the languages in
#: wikidata_landmarks' label fallback chain (en/mul/fr/es/de).
_STOPWORDS = {
    "the", "a", "an",                                        # English
    "le", "la", "les", "l", "du", "de", "des", "un", "une",   # French
    "el", "los", "las", "del",                                # Spanish
    "der", "die", "das", "den", "dem", "ein", "eine",         # German
}


def _slug(landmark_name: str, city_name: str, taken: set) -> str:
    """A short slug from the landmark name: filter out words that duplicate
    the city name or are a bare leading article ("The Morgan Library" ->
    "the" reads as nothing; "La Moneda" -> "la" likewise), then take the
    first surviving word not already used for this city. If filtering
    leaves nothing, or every surviving word collides, fall back to the
    unfiltered first word and a numeric suffix -- this is the table behind
    the released benchmark, so the fallback is there specifically to
    guarantee no collision ever slips through, even in that edge case.

    An earlier version only excluded city-name words, since a municipal
    landmark's English name commonly starts with its own city ("Hanoi
    Botanic Garden", "Hanoi College of Fine Arts") and naively taking the
    first word collapsed both into the same id, silently dropping the
    second from downstream de-duplication. Bare articles are the same
    failure mode by a different name, so both are filtered the same way
    rather than only ever handling the one pattern noticed first.
    """
    city_words = set(city_name.lower().split())
    raw, filtered = [], []
    for w in landmark_name.lower().split():
        # ASCII alnum only -- accents are dropped, not transliterated
        # ("gärdet" -> "grdet"), matching how the very first version of
        # this table stripped non-ASCII characters.
        w = "".join(ch for ch in w if ch.isalnum() and ch.isascii())[:12]
        if not w:
            continue
        raw.append(w)
        if w not in city_words and w not in _STOPWORDS:
            filtered.append(w)
    for w in filtered or raw or ["site"]:
        if w not in taken:
            return w
    base = (filtered or raw or ["site"])[0]
    slug, n = base, 2
    while slug in taken:       # every candidate word collided
        slug = f"{base}{n}"
        n += 1
    return slug


def wikidata_landmarks(lat, lon, radius_km=3.0, limit=120):
    """Entities near the city centre, of one of TYPES, with coordinates."""
    import urllib.parse
    vals = " ".join(f"wd:{t}" for t in TYPES)
    q = f"""SELECT ?item ?itemLabel ?coord ?sitelinks ?typeLabel WHERE {{
      SERVICE wikibase:around {{
        ?item wdt:P625 ?coord .
        bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
        bd:serviceParam wikibase:radius "{radius_km}" .
      }}
      VALUES ?type {{ {vals} }}
      ?item wdt:P31/wdt:P279* ?type .
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks <= {MAX_SITELINKS})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul,fr,es,de". }}
    }} ORDER BY DESC(?sitelinks) LIMIT {limit}"""

    def fetch():
        u = SPARQL + "?" + urllib.parse.urlencode({"query": q, "format": "json"})
        return _get_json(u, timeout=120)["results"]["bindings"]

    rows = _cached(f"landmarks|{lat:.4f}|{lon:.4f}|{radius_km}|{limit}", fetch)
    out = []
    for r in rows:
        c = r["coord"]["value"].replace("Point(", "").replace(")", "").split()
        out.append(dict(qid=r["item"]["value"].rsplit("/", 1)[1],
                        name=r["itemLabel"]["value"],
                        lat=float(c[1]), lon=float(c[0]),
                        sitelinks=int(r["sitelinks"]["value"]),
                        wd_type=r.get("typeLabel", {}).get("value", "")))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # An entity belonging to several TYPES at once (e.g. the Rijksmuseum is
    # both a museum and a building) gets one row per type from
    # wdt:P31/wdt:P279* -- without de-duplication the same landmark can be
    # picked more than once (Amsterdam's first pass picked the Rijksmuseum
    # for all three of its slots).
    df = df.drop_duplicates(subset="qid").reset_index(drop=True)
    # If none of the requested languages has a label, wikibase:label falls
    # back to the entity's own QID as ?itemLabel -- silently, not as an
    # error. Seen once: Marrakech's Q3496072 (a stadium with only
    # French/Arabic names) was written into a ladder as its own QID. The
    # fallback language chain above should avoid it going forward; this is
    # a second line of defence.
    return df[~df.name.eq(df.qid)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-city", type=int, default=3)
    ap.add_argument("--cities", nargs="+")
    ap.add_argument("--city-radius", type=int, default=300,
                    help="city-centre coverage-check radius, metres. Kept small: "
                         "a large radius triggers Mapillary's dense-area 500 and a "
                         "slow quadrant-recursion fallback")
    ap.add_argument("--check-top", type=int, default=25,
                    help="check street-view coverage for at most the first N "
                         "shuffled candidates per city (the expensive step)")
    ap.add_argument("--out", default="data/site_pool.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cities = [c for c in CITIES if not args.cities or c[0] in args.cities]
    print(f"{len(cities)} cities\n", flush=True)

    # Append per city as it finishes rather than only at the very end: this
    # script routinely runs tens of minutes against public APIs, and a
    # mid-run interruption should not throw away already-completed (and
    # already street-view-checked) cities.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False

    def append_picked(picked_rows):
        nonlocal header_written
        if not picked_rows:
            return
        d = pd.DataFrame(picked_rows)
        d.to_csv(out_path, index=False, encoding="utf-8-sig",
                 mode="a" if header_written else "w", header=not header_written)
        header_written = True

    rows, skipped = [], []
    for name, cc, cont, lat, lon in cities:
        # --- 1. any street-view coverage at all? (cheap) ---
        try:
            df = streetview.search(lat, lon, radius_m=args.city_radius, limit=100)
            pano = (df.is_pano.fillna(False).astype(bool)
                    if len(df) and "is_pano" in df else pd.Series(dtype=bool))
            n_persp = int((~pano).sum()) if len(df) else 0
        except Exception as e:
            print(f"{name:16} street-view search failed {type(e).__name__}", flush=True)
            skipped.append((name, "search failed"))
            continue
        if n_persp < 15:
            print(f"{name:16} insufficient coverage ({n_persp} perspective)", flush=True)
            skipped.append((name, f"insufficient coverage {n_persp}"))
            continue

        # --- 2. Wikidata landmark candidates ---
        try:
            cand = wikidata_landmarks(lat, lon)
        except Exception as e:
            print(f"{name:16} Wikidata query failed {type(e).__name__}", flush=True)
            skipped.append((name, "Wikidata failed"))
            continue
        if cand.empty:
            print(f"{name:16} no candidates of the required types", flush=True)
            skipped.append((name, "no candidates"))
            continue

        # --- 3. shuffle, then check street-view coverage until per-city quota is met ---
        cand = cand.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        picked = []
        for r in cand.head(args.check_top).itertuples():
            try:
                d = streetview.search(r.lat, r.lon, radius_m=SV_RADIUS, limit=60)
                pn = (d.is_pano.fillna(False).astype(bool)
                      if len(d) and "is_pano" in d else pd.Series(dtype=bool))
                n = int((~pn).sum()) if len(d) else 0
            except Exception:
                n = 0
            # Drop candidates within 250 m of an already-picked one -- two
            # landmarks on the same block would otherwise count as two
            # independent sites despite near-identical street-view coverage.
            if any(haversine((r.lat, r.lon), (q["lat"], q["lon"])) < 0.25
                   for q in picked):
                continue
            if n >= MIN_SV:
                taken = {p["site"].rsplit("_", 1)[1] for p in picked}
                slug = _slug(r.name, name, taken)
                picked.append(dict(
                    site=f"{name.lower().replace(' ', '')}_{slug}",
                    city=name, country=cc, continent=cont,
                    landmark=r.name, qid=r.qid, wd_type=r.wd_type,
                    lat=r.lat, lon=r.lon, sitelinks=r.sitelinks, sv_perspective=n,
                    selection_method="algorithm"))
            if len(picked) >= args.per_city:
                break
        if not picked:
            skipped.append((name, "no landmark had street-view coverage"))
        rows += picked
        append_picked(picked)
        print(f"{name:16} candidates {len(cand):3} -> picked {len(picked)}: "
              f"{', '.join(p['landmark'][:22] for p in picked)}", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        print("\nno sites picked")
        return
    out = out.drop_duplicates(subset="site")
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\n=== picked {len(out)} sites across {out.city.nunique()} cities ===")
    print(out.groupby("continent").agg(cities=("city", "nunique"),
                                       sites=("site", "size")).to_string())
    print(f"\nsitelinks: median {out.sitelinks.median():.0f}, "
          f"range {out.sitelinks.min()}-{out.sitelinks.max()}")
    print(f"street-view perspective images: median {out.sv_perspective.median():.0f}, "
          f"min {out.sv_perspective.min()}")
    if skipped:
        print(f"\nskipped {len(skipped)} cities:")
        for c, why in skipped:
            print(f"  {c:16} {why}")
    print(f"\nwritten to {args.out}")
    print("next: scripts/02_fetch_images.py")


if __name__ == "__main__":
    main()
