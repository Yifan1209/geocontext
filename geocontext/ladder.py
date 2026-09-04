"""Pipeline 1: automatically build a context ladder for a given photo coordinate,
generating a set of reference points that are **true but under-specified**.

The ladder is two-dimensional: **distance x prominence**. The two axes must be
orthogonal, because the core hypothesis is "the model anchors on the most
famous reference point" -- only the distance axis can separate "using
geographic knowledge" from "using fame".

Data sources and why they were chosen (measured, not guessed):

- **Wikidata Query Service** (coordinates + entity classes) is cleaner than
  OSM: `tourism=hotel` alone accounted for two-thirds of OSM candidates within
  1.5 km, pure noise.

- **Wikipedia monthly pageviews** (prominence). Wikidata's sitelink count is
  **not usable**: measured, one square had 4 sitelinks and a metro station had
  5, an inverted ordering relative to actual fame. Pageviews gave 4,149 for the
  square versus 328 for the station, a 13x gap that matches intuition.

Validated on the first site worked on: this recovers 5 of 6 hand-picked reference
points automatically, with distance error < 0.2 km (Taikoo Li itself has no
Wikidata entry -- a known gap, since commercial complexes are poorly covered).

NOTE: this is the earlier, pageview-based pipeline, superseded by the
LLM-audited referenceability ladder in `audit.py`. Kept for reference and for
the cross-validation described above.
"""
import json
import math
import time
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA

CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Wikimedia requires a UA with contact information. A generic UA gets 429'd
# immediately -- that, not request frequency, was the cause of an earlier
# string of rate-limit failures.
UA = {"User-Agent": ("geo-context-research/0.1 "
                     "(https://github.com/; powerfan233@gmail.com) python-urllib"),
      "Accept": "application/sparql-results+json"}

# Excluded Wikidata classes: administrative divisions, events, etc. -- none of
# these work as "I took this near X".
EXCLUDE_CLASSES = {
    "Q515",        # city
    "Q1065118",    # municipal district
    "Q174844",     # megacity
    "Q250811",     # sub-provincial city
    "Q27020041",   # season (sports)
    "Q56061",      # administrative territorial entity
    "Q1637706",    # million-population city
    "Q13220204",   # administrative district
    "Q15284",      # municipality
    "Q3957",       # town
    "Q532",        # village
    "Q6256",       # country
    "Q1970725",    # national-park-like class
    # Events: the coordinate is merely "where it happened", not a place you
    # can stand next to.
    "Q12890393",   # accident / incident (measured to leak in from real data)
    "Q1190554",    # event
    "Q1656682",    # event (generic)
    "Q13418847",   # historical event
    "Q3839081",    # disaster
    "Q198",        # war
    # The classes below were found one at a time while running on real
    # data, not anticipated in advance.
    "Q986065",     # street (a township-level administrative division in China)
                   # -- an administrative division, not a nameable spot
    "Q5503",       # metro (a transit system, not a place: "I took this near
                   # a metro line" does not parse)
    "Q5398426",    # TV series
    "Q11424",      # film
    "Q4167410",    # disambiguation page
}

# Only these "place-like" classes may serve as a reference point. An empty set
# degrades to pure blacklist mode. A whitelist is used because a blacklist can
# never be complete -- every new city surfaces some new odd class.
INCLUDE_CLASSES = {
    "Q41176",      # building
    "Q811979",     # architectural structure
    "Q33506",      # museum
    "Q44539",      # temple
    "Q32815",      # mosque
    "Q16970",      # church building
    "Q22698",      # park
    "Q10300916",   # heritage site
    "Q928830",     # metro station
    "Q55488",      # railway station
    "Q22808403",   # underground station
    "Q3918",       # university
    "Q9826",       # senior high school
    "Q34442",      # road
    "Q174782",     # square
    "Q11315",      # shopping mall
    "Q173387",     # grave
    "Q249027",     # ancestral shrine
    "Q839954",     # archaeological site
    "Q570116",     # tourist attraction
    "Q12280",      # bridge
    "Q1497364",    # library building
    "Q41253",      # cinema
    "Q483110",     # stadium
}


class Transient(Exception):
    """A retryable failure (rate limit, 5xx, network jitter). **Never cache this**."""


def _cached(key: str, fn, ttl_days: int = 30):
    """Disk cache. SPARQL and pageview lookups are both slow and rate-limited,
    so repeated calls are pointless.

    Only deterministic results are cached. A transient failure (429/5xx) must
    raise Transient and propagate -- an earlier version cached the None
    returned by a rate limit, permanently and silently poisoning the entry
    after a single throttled request.
    """
    p = CACHE / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl_days * 86400:
        return json.loads(p.read_text(encoding="utf-8"))
    val = fn()                      # not written to disk if this raises Transient
    p.write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
    return val


def _get_json(url, timeout=60, retries=5):
    """GET with exponential backoff. 404 raises FileNotFoundError (deterministic);
    everything else raises Transient after retries are exhausted."""
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"],
                                                       "Accept": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay); delay *= 2
                continue
            raise Transient(f"HTTP {e.code}") from e
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2
                continue
            raise Transient(str(e)) from e
    raise Transient("retries exhausted")


def haversine(a, b) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- Wikidata

SPARQL = """SELECT ?item ?itemLabel ?coord ?sitelinks ?cls ?zhTitle ?enTitle WHERE {{
  SERVICE wikibase:around {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius}" . }}
  ?item wikibase:sitelinks ?sitelinks ; wdt:P31 ?cls .
  OPTIONAL {{ ?a schema:about ?item ; schema:isPartOf <https://zh.wikipedia.org/> ;
              schema:name ?zhTitle . }}
  OPTIONAL {{ ?b schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ;
              schema:name ?enTitle . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}" . }}
}} ORDER BY DESC(?sitelinks) LIMIT {limit}"""


def wikidata_around(lat, lon, radius_km=6.0, lang="zh,en", limit=400) -> pd.DataFrame:
    """Query Wikidata entities near a coordinate.

    Returns qid/label/lat/lon/dist_km/sitelinks/classes/titles.
    """
    q = SPARQL.format(lat=lat, lon=lon, radius=radius_km, lang=lang, limit=limit)

    def fetch():
        u = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
            {"query": q, "format": "json"})
        return _get_json(u, timeout=120)["results"]["bindings"]

    rows = _cached(f"wd|{lat:.4f}|{lon:.4f}|{radius_km}|{limit}|{lang}", fetch)

    # An entity can carry several P31 values; aggregate them into a set.
    agg = {}
    for r in rows:
        qid = r["item"]["value"].rsplit("/", 1)[1]
        if qid not in agg:
            c = r["coord"]["value"].replace("Point(", "").replace(")", "").split()
            plon, plat = float(c[0]), float(c[1])
            agg[qid] = dict(qid=qid, label=r["itemLabel"]["value"],
                            lat=plat, lon=plon,
                            dist_km=round(haversine((lat, lon), (plat, plon)), 3),
                            sitelinks=int(r["sitelinks"]["value"]),
                            zh_title=r.get("zhTitle", {}).get("value"),
                            en_title=r.get("enTitle", {}).get("value"),
                            classes=set())
        agg[qid]["classes"].add(r["cls"]["value"].rsplit("/", 1)[1])

    df = pd.DataFrame(agg.values())
    if df.empty:
        return df
    # Blacklist first, then whitelist. An empty whitelist means blacklist-only.
    df["excluded"] = df["classes"].map(
        lambda cs: bool(cs & EXCLUDE_CLASSES)
        or (bool(INCLUDE_CLASSES) and not (cs & INCLUDE_CLASSES)))
    return df.sort_values("dist_km").reset_index(drop=True)


# ---------------------------------------------------------------- prominence

def _missing(v) -> bool:
    """pandas represents missing as float('nan'), and `if nan` is truthy, so
    the check has to be explicit."""
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == ""


def pageviews(title, wiki="zh.wikipedia", months=12) -> int | None:
    """Cumulative Wikipedia pageviews over the last N months.

    Returns None (distinct from 0) when the title cannot be found.
    """
    if _missing(title):
        return None
    title = str(title)
    # Align to calendar-month boundaries, so the API does not undercount a
    # month at a non-aligned boundary.
    end = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    start = end - pd.DateOffset(months=months)

    def fetch():
        t = urllib.parse.quote(title.replace(" ", "_"), safe="")
        u = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{wiki}"
             f"/all-access/user/{t}/monthly/{start:%Y%m%d}00/{end:%Y%m%d}00")
        try:
            return sum(i["views"] for i in _get_json(u, timeout=30)["items"])
        except FileNotFoundError:
            return None          # the article does not exist -- a deterministic
                                 # result, safe to cache

    return _cached(f"pv|{wiki}|{title}|{start:%Y%m}|{months}", fetch)


def add_prominence(df: pd.DataFrame, sleep=1.0) -> pd.DataFrame:
    """Add a pageviews column to a candidate table. Prefers zh, falls back to en."""
    df = df.copy()
    vals = []
    for _, r in df.iterrows():
        v = pageviews(r["zh_title"], "zh.wikipedia")
        if v is None:
            v = pageviews(r["en_title"], "en.wikipedia")
        vals.append(v)
        if v is not None:          # no need to rate-limit on a cache hit
            time.sleep(sleep)
    df["pageviews"] = vals
    return df


# ---------------------------------------------------------------- ladder building

# Distance bands (km). Within 0.5 km is essentially "this is the place", not
# under-specified, so the bands start at 0.5.
DEFAULT_BANDS = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0)]


def build_ladder(lat, lon, bands=None, radius_km=6.0, per_cell=1,
                 min_pageviews=50, lang="zh,en", verbose=True) -> pd.DataFrame:
    """Build the distance x prominence context ladder.

    Each distance band contributes one **most famous** and one **least
    famous** reference point, so distance and prominence stay orthogonal and
    can be attributed separately.
    """
    bands = bands or DEFAULT_BANDS
    cand = wikidata_around(lat, lon, radius_km, lang=lang)
    if cand.empty:
        return cand
    n0 = len(cand)
    cand = cand[~cand["excluded"]]
    # Trim to the distance range before fetching pageviews: pageviews are one
    # HTTP request each, and candidates outside every band are of no use.
    lo_min, hi_max = min(b[0] for b in bands), max(b[1] for b in bands)
    cand = cand[(cand.dist_km >= lo_min) & (cand.dist_km < hi_max)]
    n1 = len(cand)
    # An entity with no Wikipedia article of either language cannot have
    # pageviews, so skip the request.
    cand = cand[cand.zh_title.notna() | cand.en_title.notna()]
    n2 = len(cand)
    cand = add_prominence(cand)
    cand = cand[cand["pageviews"].notna() & (cand["pageviews"] >= min_pageviews)]
    if verbose:
        print(f"candidates {n0} -> excluding admin/event classes -> "
              f"{n1} within {lo_min}-{hi_max}km -> {n2} with a Wikipedia article "
              f"-> {len(cand)} with pageviews >= {min_pageviews}")

    out = []
    for lo, hi in bands:
        band = cand[(cand.dist_km >= lo) & (cand.dist_km < hi)]
        if band.empty:
            if verbose:
                print(f"  [{lo}-{hi}km] no candidates")
            continue
        band = band.sort_values("pageviews", ascending=False)
        for tier, sel in (("famous", band.head(per_cell)),
                          ("obscure", band.tail(per_cell))):
            for _, r in sel.iterrows():
                out.append(dict(band=f"{lo}-{hi}km", tier=tier, label=r.label,
                                qid=r.qid, dist_km=r.dist_km,
                                pageviews=int(r.pageviews), sitelinks=r.sitelinks,
                                n_in_band=len(band)))
    res = pd.DataFrame(out)
    # The same entity can be both the most and the least famous in a band that
    # has only one candidate; de-duplicate.
    return res.drop_duplicates(subset=["qid", "band"]).reset_index(drop=True)


def to_context(label: str, lang="en") -> str:
    """Turn a reference point into a first-person context sentence."""
    if lang != "en":
        raise ValueError(
            f"only the English condition ships in this release, got lang={lang!r}")
    return f"I took this photo near {label}. "
