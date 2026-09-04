"""Geocode model answers, so accuracy becomes a continuous distance instead of a binary hit.

## Why a continuous metric is necessary

Binary scoring ("does the answer contain *Shibuya*?") is acutely sensitive to
whether a neighbourhood has a usable name, which makes it incomparable across
sites. Measured:

| Site           | Area-level accuracy | Frequent model answer |
|----------------|--------------------|------------------------|
| Tokyo Shibuya  | 25%                | Shibuya (the model volunteers the word) |
| Paris Marais   | 31%                | Le Marais / **Latin Quarter** (a different district across the river) |
| New York SoHo  | **15%**            | **Midtown / Lower Manhattan** (one level coarser) |

The 15% for New York is not a scoring artefact -- adding aliases did not change
it, models really do prefer the coarser "Manhattan". But binary scoring counts
"Lower Manhattan" (off by ~1 km) and "Chicago" (off by 1200 km) as equally
wrong, discarding all of the information.

Scoring by the distance from the geocoded answer to the true coordinate removes
the dependence on what a neighbourhood happens to be called, and makes sites
comparable. This is also the Near-Miss measure the project set out to build.

## Handling ambiguous place names

One name can map to several entities ("Marais" is a district of Paris and also
a place elsewhere). We must **not** pick the candidate nearest the ground truth
-- that peeks at the answer and launders a wrong answer into a right one.
Instead we pick the one with the most sitelinks (the mainstream reading), i.e.
what someone who does not know the answer would think of on hearing the name.

## Four scoring faults, and why each fix is what it is

Every one of these was invisible in aggregate statistics and was found only by
reading individual records. Together they had reversed the sign of the headline
result before they were found. None of them raises an exception -- each produces
a plausible-looking number, which is what makes them dangerous.

### 1. Parenthetical aliases silently degrade to city level

`wbsearchentities` is exact-label search, so `Shibuya` resolves while
`Shibuya (Miyamasuzaka)` does not, falling back to city level -- not an error
message but a plausible-looking 3.42 km.

Fix: generate cleaned label variants in a fixed **specific-first** order and take
the first that resolves. The order matters: preferring the broader form would
systematically pull answers toward the city centre and depress error.

### 2. Linear features cannot be scored as points

Meiji-dori is a long street with a single Wikidata coordinate. Human judgement on
two responses with *identical* `place` fields is opposite:
`place=Meiji-dori, area=Shibuya` is correct (that stretch is in Shibuya) while
`place=Meiji-dori, area=Harajuku` is wrong. The information lives in `area`.

Fix: skip linear features at the `place` level and fall through to `area`.
Do **not** compute distance to the street geometry -- Meiji-dori passes through
both Harajuku and Shibuya, so that rule scores both responses correct, which
contradicts human judgement. Street-type words are detected per language
convention: suffixal in English and Japanese, prefixal in French, Italian and
German, with `-dori` / `-zaka` bound morphemes carrying no word boundary.

### 3. A global fallback fabricates enormous errors

When no candidate lay within the anchor radius, an earlier version fell back to
the globally most-linked entity, so `city=New York` with `place=Ralphs`
(a California grocery chain) scored 16609 km. The model never claimed the photo
was in California; it named a New York store we cannot geocode. Recording
"unverifiable" as "wrong by sixteen thousand kilometres" invents error.

Fix: treat out-of-anchor as a level failure. This cut >1000 km errors from 976 to
459 and the maximum from 16609 km to 3938 km.

### 4. The city-level fallback distance is a constant

When an answer resolves no finer than the city, error_km is the fixed distance
from the city centroid to the site:

| Site          | city-level error_km |
|---------------|---------------------|
| Tokyo Shibuya | 3.424 km            |
| New York SoHo | 1.199 km            |
| Paris Marais  | **0.761 km**        |

Which side of the 1 km hit threshold that constant lands on is an accident of
where the centroid happens to sit. Paris fell below it, so every answer resolving
only to "Paris" counted as a sub-kilometre hit, inflating that site by 20.7
points and concealing the effect reported in the paper.

Fix: `hit()` treats city-level resolution as **censored** -- it states only that
the answer is no finer than the city, never that it is within 1 km. Across the
full release these four corrections cut city-level resolution from 51.0% to 9.0%
of responses.

General rule this leaves behind: any error value that recurs as a constant is a
signal of a resolution failure, not a measurement.
"""
import json
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd

from .config import DATA
from .ladder import _get_json, haversine, _cached, UA  # noqa: F401

CACHE = DATA / "cache"
SEARCH = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"


def _search_entities(label: str, lang="en", limit=8) -> list[str]:
    """Search Wikidata by label, returning QIDs."""
    def fetch():
        u = SEARCH + "?" + urllib.parse.urlencode({
            "action": "wbsearchentities", "search": label, "language": lang,
            "uselang": lang, "limit": limit, "format": "json", "formatversion": "2"})
        d = _get_json(u, timeout=30)
        return [x["id"] for x in d.get("search", [])]
    return _cached(f"wbsearch|{lang}|{label}|{limit}", fetch)


def _coords_for(qids: list[str]) -> list[dict]:
    """Fetch coordinates and sitelink counts for a batch of QIDs.

    Entities without coordinates are dropped automatically.
    """
    if not qids:
        return []
    values = " ".join(f"wd:{q}" for q in qids)
    q = f"""SELECT ?item ?coord ?sitelinks WHERE {{
      VALUES ?item {{ {values} }}
      ?item wdt:P625 ?coord ; wikibase:sitelinks ?sitelinks .
    }}"""

    def fetch():
        u = SPARQL + "?" + urllib.parse.urlencode({"query": q, "format": "json"})
        return _get_json(u, timeout=90)["results"]["bindings"]

    rows = _cached(f"qcoord|{'|'.join(sorted(qids))}", fetch)
    out = []
    for r in rows:
        c = r["coord"]["value"].replace("Point(", "").replace(")", "").split()
        out.append(dict(qid=r["item"]["value"].rsplit("/", 1)[1],
                        lat=float(c[1]), lon=float(c[0]),
                        sitelinks=int(r["sitelinks"]["value"])))
    return out



# --------------------------------------------------------------- label cleaning
#
# Models like to append an alias or a clarification after a place name, and
# `wbsearchentities` is an exact-label search: anything carrying parentheses
# fails to resolve and **silently falls back to city level**. City-level error
# is a constant (Tokyo 3.42 km, New York 1.20 km), so it looks like "the model
# was 3 km off" when in fact parsing failed.
#
#   Shibuya (in Japanese script)   -> no match -> city level 3.42
#                                     (the real area-level answer was 0.53)
#   Shibuya (Miyamasuzaka)         -> same
#   Miyamasuzaka, Shibuya          -> same
#
# Variants are tried in a fixed **specific-first** order and the first one that
# resolves wins. The order matters: preferring the broader form would
# systematically pull answers towards the city centre and understate the error.

_PAREN = re.compile(r"[（(【\[][^）)】\]]*[）)】\]]")
_INNER = re.compile(r"[（(【\[]([^）)】\]]+)[）)】\]]")
_SEP = re.compile(r"\s*[,，、/｜|·・;；—–~〜]+\s*|\s+-\s+")


def variants(label: str) -> list[str]:
    """Ordered cleaned forms of a raw model-produced place label, specific first."""
    label = str(label).strip()
    bare = _PAREN.sub("", label).strip(" 　-–—,，、/")
    segs = [x.strip() for x in _SEP.split(bare) if x.strip()]
    inner = _INNER.search(label)
    inner_segs = ([x.strip() for x in _SEP.split(inner.group(1)) if x.strip()]
                  if inner else [])
    out = [label, bare]
    if segs:
        out.append(segs[0])
    if inner_segs:
        out.append(inner_segs[0])
    if len(segs) > 1:
        out.append(segs[-1])          # broadest form last, as a fallback
    seen, res = set(), []
    for v in out:
        if len(v) >= 2 and v not in seen:
            seen.add(v)
            res.append(v)
    return res


# --------------------------------------------------------------- linear features
#
# Roads, slopes and similar **linear features** cannot be scored as points.
# Meiji-dori is a long street and Wikidata gives it one representative point
# (3 km from Shibuya), so:
#
#   place=Meiji-dori + area=Shibuya   human verdict: fully correct
#                                     -- that stretch of road IS in Shibuya
#   place=Meiji-dori + area=Harajuku  human verdict: wrong -- wrong district
#
# The `place` field is identical in both, yet the verdicts are opposite, which
# says **the information is in `area`, not in `place`**. Hence the rule: a
# linear feature does not participate in scoring; fall through to the next
# bounded level.
#
# Why not measure distance to the road geometry instead: Meiji-dori runs
# through both Harajuku and Shibuya, so a geometric distance would score both
# examples as 0 km and correct -- the opposite of the human verdict.
#
# Safety: this rule fires only at `place` level, so the worst case is falling
# back to the coarser `area` level. It can only make scores more conservative
# and can never manufacture accuracy.

# Street-type words sit predictably: English and Japanese put them at the
# **end** (Canal Street, Meiji-dori), French / Italian / German at the
# **start** (rue de Rivoli, via del Corso). Kept as two patterns rather than
# one big \b(...)\b, otherwise "Ave Maria" matches `ave` and is misread as an
# avenue.
_LINEAR_SUFFIX = re.compile(
    r"(通り|通|銀座通|坂|坂上|坂下|坂道|街道|大街|大道|大路|路|街|"
    r"\b(?:street|st|road|rd|avenue|ave|boulevard|blvd|drive|dr|lane|ln|alley|"
    r"broadway|highway|parkway|expressway|freeway|crossing|intersection)|"
    # Romanised -dori / -zaka / -bashi are bound morphemes with no preceding
    # word boundary (Dogenzaka).
    r"d[oō]ri|zaka|bashi)\s*$", re.I)
_LINEAR_PREFIX = re.compile(
    r"^\s*(rue|avenue|boulevard|via|corso|viale|calle|carrer|"
    r"stra(?:ss|ß)e|allee)\b", re.I)


def is_linear(label: str) -> bool:
    """A road, slope, crossing or similar linear feature that cannot locate a photo."""
    t = str(label).strip()
    return bool(_LINEAR_SUFFIX.search(t) or _LINEAR_PREFIX.search(t))


def geocode(label: str, langs=("en", "zh"), sleep=0.1,
            near: tuple | None = None, near_km: float = 60.0) -> dict | None:
    """Resolve a place name to a coordinate.

    Returns {qid, lat, lon, sitelinks, label}, or None if it cannot be resolved.

    `near` is a **disambiguation anchor** for cross-country name collisions.
    Without it the results are wildly wrong in practice:

    - `SoHo` -> Soho in London (sitelinks 50), not the New York one
    - `Le Marais` -> a small town in Normandy (48.88, **-0.02**), not the Paris
      district (48.86, **2.36**)

    The anchor MUST come from the city the model itself answered, never from the
    ground truth. Disambiguating with the truth peeks at the answer and launders
    "answered London Soho" into "correctly answered New York SoHo". Candidates
    within `near_km` win; if none qualify, see the comment below.
    """
    if not label or str(label).strip().lower() in ("nan", "none", "unknown", "未知", ""):
        return None
    for label in variants(label):
        qids = []
        for lg in langs:
            try:
                qids += _search_entities(label, lg)
            except Exception:
                pass
            if qids:
                break
        try:
            cands = _coords_for(list(dict.fromkeys(qids))[:8])
        except Exception:
            cands = []
        if cands:
            break
    else:
        return None
    if not cands:
        return None
    if not sleep:
        pass
    else:
        time.sleep(sleep)
    if not cands:
        return None
    if near:
        close = [c for c in cands
                 if haversine(near, (c["lat"], c["lon"])) <= near_km]
        if close:
            return {**max(close, key=lambda c: c["sitelinks"]),
                    "label": label, "disambiguated": True}
        # No candidate inside the anchor radius -> **this level fails to
        # resolve**, and the caller falls through to a coarser level.
        #
        # We must NOT fall back to "the globally most-linked entity". When the
        # model answers city=New York, place=`Ralphs` (a California grocery
        # chain), that fallback yields a 16,609 km error -- but the model did
        # not claim the photo was in California, it named a New York store we
        # simply cannot geocode. Recording "unverifiable" as "wrong by sixteen
        # thousand kilometres" manufactures error out of nothing.
        #
        # Measured: this fallback started firing constantly once variants() was
        # introduced (MUJI headquarters 16,343 km, WIFC 11,497 km), because
        # cleaned fragment labels match same-named entities abroad more easily.
        return None
    return {**max(cands, key=lambda c: c["sitelinks"]), "label": label,
            "disambiguated": False}


#: Resolution order: try the most specific level first, degrade step by step.
#: The level used is recorded (`resolved_level`), because an error that only
#: resolved to a city means something different from one that resolved to a shop.
LEVELS = ("place", "area", "city")


def error_km(row, gt_lat, gt_lon, cache: dict | None = None) -> tuple:
    """Returns (error km, resolved level, resolved label), or (nan, None, None).

    The model's answered **city** is resolved first and used as the
    disambiguation anchor for the finer levels -- if the model says "SoHo in New
    York", it should not be scored against Soho in London. The anchor comes from
    the model's own answer, never from the ground truth.
    """
    cache = cache if cache is not None else {}
    anchor = None
    ckey = str(row.get("city") or "").strip().lower()
    if ckey and ckey not in ("nan", "none", "unknown", "未知"):
        if ("city", ckey) not in cache:
            cache[("city", ckey)] = geocode(row.get("city"))
        cg = cache[("city", ckey)]
        if cg:
            anchor = (cg["lat"], cg["lon"])

    for lvl in LEVELS:
        lab = row.get(lvl)
        if lab is None:
            continue
        key = str(lab).strip().lower()
        if key in ("", "nan", "none", "unknown", "未知"):
            continue
        if lvl == "place" and is_linear(lab):
            continue          # linear features cannot locate a photo; fall to area
        ck = (lvl, key, anchor)
        if ck not in cache:
            cache[ck] = geocode(lab, near=anchor)
        g = cache[ck]
        if g:
            return (round(haversine((gt_lat, gt_lon), (g["lat"], g["lon"])), 3),
                    lvl, g["label"])
    return (float("nan"), None, None)


def hit(df, km: float = 1.0):
    """`<km` hit test. A city-level resolution is **never** a hit, whatever the number.

    Why this is mandatory: the error of a city-level fallback is a constant --
    the distance from the city centroid to the site -- and which side of the
    threshold that constant lands on is pure accident:

        Paris Marais      0.761 km   <- **below the 1 km threshold**
        New York SoHo     1.199 km
        Tokyo Shibuya     3.424 km

    So every Paris response that only got as far as "Paris" was recorded as a
    sub-kilometre hit, inflating that site's accuracy by 20.7 points
    (54.2% -> 33.5%). An earlier draft's claim that "Paris is the one site with
    legible imagery (baseline 82.8%)" was entirely this artefact.

    The correct treatment is to regard a city-level resolution as a **censored
    observation**: it says the answer was coarser than a city, and does not
    constitute a sub-kilometre localisation, regardless of where the centroid
    happens to fall.
    """
    return (df.error_km < km) & (df.resolved_level != "city")


def site_of(path: str) -> str | None:
    """Extract the site name from an image path.

    The separator is `\\` on Windows and `/` on Linux, and it gets escaped again
    once written into jsonl -- handling that with a regex is very easy to get
    wrong through a second round of shell escaping. Splitting on both separators
    is simpler and safer.
    """
    parts = str(path).replace("\\", "/").split("/")
    if "streetview" in parts:
        i = parts.index("streetview")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def add_error_km(df: pd.DataFrame, gt: dict, verbose=True,
                 checkpoint: "Path | None" = None, every: int = 500) -> pd.DataFrame:
    """Add error_km / resolved_level / resolved_label columns to a results table.

    gt: {site: (lat, lon)}, the ground-truth coordinates.

    `checkpoint`: write to disk every `every` rows. This step is slow (thousands
    of rows, each with network lookups) and **can be killed by something
    external** -- a background task once died with its parent CLI process and a
    loop that had reached 1400/4518 was lost entirely. Wikidata results are
    cached on disk so a rerun is not slow, but there is no reason to rerun.
    """
    df = df.copy()
    if "site" not in df.columns:
        df["site"] = df["path"].map(site_of)
    cache, out = {}, []
    cols = ["error_km", "resolved_level", "resolved_label"]
    for i, r in enumerate(df.itertuples(), 1):
        site = getattr(r, "site", None)
        if site not in gt:
            out.append((float("nan"), None, None))
            continue
        lat, lon = gt[site]
        out.append(error_km({"place": getattr(r, "place", None),
                             "area": getattr(r, "area", None),
                             "city": getattr(r, "city", None)}, lat, lon, cache))
        if verbose and i % 200 == 0:
            print(f"  geocode {i}/{len(df)} ({len(cache)} names cached)", flush=True)
        if checkpoint is not None and i % every == 0:
            part = df.iloc[:len(out)].copy()
            part[cols] = pd.DataFrame(out, index=part.index, columns=cols)
            part.to_pickle(checkpoint)
            if verbose:
                print(f"    checkpointed {len(out)} rows -> {checkpoint.name}", flush=True)
    if not out:      # empty input: create the columns explicitly, so an empty
                     # DataFrame does not end up with zero columns
        for c in cols:
            df[c] = pd.NA
        return df
    df[cols] = pd.DataFrame(out, index=df.index, columns=cols)
    return df
