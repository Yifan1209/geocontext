"""Pipeline 2: LLM-audited reference-point selection.

Replaces the hard-coded category rules of `ladder.INCLUDE_CLASSES`. It does
three things in one pass: decide whether a candidate is usable, rate its
"referenceability", and produce canonical Simplified-Chinese and English names.

## Why neither hand-written rules nor pageviews are enough (both measured)

**Rules do not transfer.** `INCLUDE_CLASSES` was tuned on the first site
worked on. Applying it to New York cut candidates from 199 to 18. On the
original site it rejected the
Tianfu Panda Tower (a genuine landmark) while keeping a gravestone that nobody
would ever use to say where they are.

**Pageviews measure the wrong thing.** In San Jose, Rosicrucian Park has 5,846
views against the Municipal Rose Garden's 2,294, so pageviews call the former
better known. But the former is a curiosity (an Egyptian museum); locals
navigate by the Rose Garden. **Pageviews measure how many people look a place
up, not how many people use it to describe where they are.**

**Simplified/Traditional is not a mechanical conversion.** Wikidata's Chinese
labels are largely Traditional, and mechanically simplifying the Traditional
form of "San Jose" yields a string that is not the standard mainland rendering.
Normalisation has to happen at the semantic level.

## Design constraints

1. **The auditor model must be disjoint from the evaluated models** -- otherwise
   the models are setting their own exam.
2. **Decisions are persisted** (`data/audit_decisions.jsonl`) and shipped with
   the dataset; without them the benchmark cannot be reproduced.
3. **Disagreement between the LLM and pageviews becomes a human review queue**,
   rather than one of them being adopted automatically.
"""
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import providers
from .config import DATA

DECISIONS = DATA / "audit_decisions.jsonl"
#: Sensitive-name screening decisions are persisted separately, not mixed
#: into the referenceability ratings file.
SENSITIVE = DATA / "audit_sensitive.jsonl"

#: Output cap for the auditor LLM. deepseek-v4-pro is a reasoning model: a
#: batch of 15 candidates measurably burns ~10,000 tokens of reasoning before
#: it emits any answer. Set this too low and you get finish_reason=length with
#: empty content -- see the note in _text_only.
MAX_TOKENS = 20000

PROMPT_EN = """You are selecting LOCATION REFERENCE POINTS for a visual geolocation experiment.

Scenario: a tourist photographs something, asks an AI "where is this?", and casually
adds their rough whereabouts — "I took this near <somewhere>". Judge whether each
candidate works as that <somewhere>.

For each candidate give four fields:

1. `ok`: usable or not. Typical rejects:
   - Administrative divisions (districts, subdistricts, wards, boroughs) — nobody says
     "I'm near the 3rd Municipal Subdistrict"
   - Abstract systems (a metro network, a railway line) — not a spot you can stand next to
   - Events and incidents — the coordinate is merely where it happened
   - Objects too small to navigate by (gravestones, plaques, individual rooms)

2. `familiarity`: **referenceability**, 0-5. This is NOT "how famous is it".
   It is "how often would locals or tourists use it to describe where they are".
   A curiosity attraction may get many encyclopedia readers yet nobody navigates by it
   — score those low.
   5 = everyone uses it to locate themselves; 3 = locals use it; 1 = almost nobody says it.

3. `zh`: canonical **Simplified Chinese** name. Use the rendering standard in mainland
   China (e.g. San Jose is 圣何塞, not 圣荷西). Use null if there is no common Chinese name.

4. `en`: canonical English name. Use null if there is none.

Candidates (with distance from the reference point):

{items}

Output ONLY a JSON array, elements like:
{{"idx":0,"ok":true,"familiarity":4,"zh":"圣何塞玫瑰园","en":"San Jose Municipal Rose Garden","why":"under 12 words"}}
No other text."""


#: Language of the audit prompt. Only English ships in this release: the prompt
#: IS the method, so a reviewer has to be able to read it. A Chinese variant
#: exists internally for v2 but its negative examples are China-specific and
#: skew judgements on non-Chinese candidates; agreement between the two was
#: measured before the Chinese one was dropped from the release.
PROMPTS_AUDIT = {"en": PROMPT_EN,
                 "sensitive": None}   # see PROMPT_SENSITIVE_EN below


def _load() -> dict:
    if not DECISIONS.exists():
        return {}
    out = {}
    for line in DECISIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["qid"], r["model"])] = r
    return out


def _text_only(client, prompt: str) -> str:
    """Text-only call. The Provider interface is built for images; this bypasses that."""
    if isinstance(client, providers.AnthropicProvider):
        r = client.client.messages.create(
            model=client.model, max_tokens=MAX_TOKENS, timeout=600,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")
    if isinstance(client, providers.GeminiProvider):
        return client.client.models.generate_content(
            model=client.model, contents=[prompt]).text or ""
    # NOTE deepseek-v4-pro is a **reasoning model**: it spends thousands of
    # tokens thinking before emitting any answer.
    #
    # Measured (2026-09-01, batches of 15 candidates):
    #   max_tokens=4000  -> finish_reason=length, 4000 tok out, **0 visible characters**
    #   max_tokens=12000 -> finish_reason=stop, 11541 tok out, only 1671 visible
    #
    # So the whole 4000-token budget goes into reasoning, not one character of
    # answer comes out -> JSON parsing fails -> _ask_recursive splits the batch
    # in half and retries -> each half fails the same way. Shrinking the batch
    # does not help at all (batch=8 also hits finish=length). This once pushed
    # the estimated ladder-build time to 42 hours.
    #
    # max_tokens must therefore leave room for the reasoning, and the timeout
    # has to grow with it: a 12000-token call measures at 200s.
    r = client.client.chat.completions.create(
        model=client.model, max_tokens=MAX_TOKENS, timeout=600,
        messages=[{"role": "user", "content": prompt}])
    txt = r.choices[0].message.content or ""
    if not txt and r.choices[0].finish_reason == "length":
        # Surface this explicitly rather than letting it masquerade as a
        # plain parse failure.
        raise RuntimeError(
            f"reasoning exhausted max_tokens={MAX_TOKENS} without producing content; raise MAX_TOKENS")
    return txt


def _fmt(chunk) -> str:
    return "\n".join(
        f"{i}. {r.label}"
        + (f" (English name: {r.en_title})" if isinstance(r.en_title, str) else "")
        + f"  {r.dist_km:.1f} km from the site"
        for i, r in enumerate(chunk.itertuples()))


def _ask_recursive(client, chunk, verbose, depth=0, prompt_lang="en"):
    """Send one batch; on failure split in half recursively down to single items.

    Returns results with idx aligned to the chunk.
    """
    got = _ask(client, _fmt(chunk), verbose and depth == 0, prompt_lang)
    if got is not None:
        # Seen once: the JSON itself parsed, but the array had a non-dict
        # element mixed in (a model formatting slip), and d.get("idx") threw
        # AttributeError and took the whole pipeline down with it. Same
        # tolerance already used elsewhere: treat a malformed element as "not
        # returned" rather than crashing; the next run's cache-miss filter
        # picks it back up.
        return [d for d in got
                if isinstance(d, dict)
                and isinstance(d.get("idx"), int) and 0 <= d["idx"] < len(chunk)]
    if len(chunk) == 1:
        return []                      # a single item still failed; give up on it
    half = len(chunk) // 2
    out = _ask_recursive(client, chunk.iloc[:half], verbose, depth + 1, prompt_lang)
    right = _ask_recursive(client, chunk.iloc[half:], verbose, depth + 1, prompt_lang)
    return out + [{**d, "idx": d["idx"] + half} for d in right]


def _ask(client, items: str, verbose: bool, prompt_lang="en"):
    """Send one batch and return the parsed array, or None on failure
    (the caller decides whether to split and retry)."""
    try:
        return json.loads(_strip_fence(_text_only(
            client, PROMPTS_AUDIT[prompt_lang].format(items=items))))
    except Exception as e:
        if verbose:
            print(f"    parse failed: {type(e).__name__}", flush=True)
        return None


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    i, j = s.find("["), s.rfind("]")
    return s[i:j + 1] if i >= 0 and j > i else s


def audit(cand: pd.DataFrame, model="deepseek-v4-pro", batch=20,
          prompt_lang="en", verbose=True, workers=6) -> pd.DataFrame:
    """Add ok / familiarity / name_zh / name_en / why columns to a candidate table.

    Cached by qid, not by label -- labels change with the language setting.
    """
    cand = cand.copy()
    cache = _load()
    tag = f"{model}@{prompt_lang}"      # language is part of the cache key: switching
                                   # language requires a re-audit, not a reuse
    todo = cand[~cand.qid.map(lambda q: (q, tag) in cache)]
    if verbose:
        print(f"{len(cand)} candidates, {len(cand)-len(todo)} already audited, "
              f"{len(todo)} to go", flush=True)

    if len(todo):
        client = providers.REGISTRY[model]()
        todo = todo.reset_index(drop=True)
        n_batch = -(-len(todo) // batch)
        # Batch-level parallelism. A single call measures ~200s (reasoning
        # model), so 2300 candidates serially would take 8+ hours. Threads in
        # one process rather than multiple processes: several processes
        # appending to the same jsonl interleave and corrupt it on Windows.
        # One writer, protected by a lock.
        lock = threading.Lock()
        done_n = [0]

        def one(start):
            chunk = todo.iloc[start:start + batch]
            # A whole-batch failure usually means output was truncated by
            # max_tokens. Split recursively down to single items rather than
            # dropping the batch: dropping leaves silent holes in the pool.
            arr = _ask_recursive(client, chunk, verbose, prompt_lang=prompt_lang)
            with lock:
                done_n[0] += 1
                if verbose:
                    tail = "" if len(arr) == len(chunk) else f" ({len(arr)}/{len(chunk)} succeeded)"
                    print(f"  batch {done_n[0]}/{n_batch} done{tail}", flush=True)
            return chunk, arr

        with open(DECISIONS, "a", encoding="utf-8") as f, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk, arr in pool.map(one, range(0, len(todo), batch)):
                for d in arr:
                    i = d.get("idx")
                    if not isinstance(i, int) or i >= len(chunk):
                        continue
                    row = chunk.iloc[i]
                    rec = dict(qid=row.qid, label=row.label, model=tag,
                               auditor=model, prompt_lang=prompt_lang,
                               ok=bool(d.get("ok")),
                               familiarity=d.get("familiarity"),
                               name_zh=d.get("zh"), name_en=d.get("en"),
                               why=str(d.get("why", ""))[:40], ts=time.time())
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache[(row.qid, tag)] = rec

    get = lambda q, k: cache.get((q, tag), {}).get(k)
    for col, key in (("ok", "ok"), ("familiarity", "familiarity"),
                     ("name_zh", "name_zh"), ("name_en", "name_en"), ("why", "why")):
        cand[col] = [get(q, key) for q in cand.qid]
    return cand


# ------------------------------------------------- sensitive place-name screening
#
# Why this is a separate pass rather than another `ok` criterion in PROMPT_EN:
#
# 1. Editing the main prompt would invalidate 4000+ cached referenceability
#    ratings, and the auditor-agreement statistics computed from them
#    (kappa 0.919 across models / 0.906 across languages) would need rerunning.
# 2. In the paper these are two different things: **referenceability** is a
#    measurement, an **exclusion criterion** is dataset policy. Folded into one
#    `ok` flag, a reader cannot tell whether a candidate was dropped because
#    nobody navigates by it or because it is politically contentious.
#
# Some of these are genuinely places tourists ask about, but including them in
# a public benchmark would import unrelated controversy into the dataset, so
# they are excluded.

PROMPT_SENSITIVE_EN = """You are screening place names for a PUBLIC research benchmark.

The bar is DELIBERATELY HIGH: flag only names whose inclusion in an academic dataset
would itself be objectionable. Ordinary political, historical or commercial
controversy is NOT a reason to flag. When in doubt, do NOT flag.

Flag `sensitive: true` ONLY for these four categories:

1. **Japanese militarism** — shrines, monuments or institutions that honour
   Imperial Japanese aggression or convicted Class-A war criminals
   (e.g. Yasukuni Shrine); the Empire of Japan itself as an entity.
2. **Nazism / fascism** — sites that memorialise or glorify the Third Reich,
   the Holocaust perpetrators, or comparable fascist regimes.
   (Memorials **to victims** are fine and must NOT be flagged.)
3. **Severe racism** — monuments erected to celebrate slavery, segregation,
   apartheid, or ethnic cleansing (e.g. Confederate "Lost Cause" monuments).
4. **Israeli-Palestinian conflict** — sites whose naming or sovereignty is
   directly contested within that conflict.

Do NOT flag anything else. In particular, these are all FINE:
- Explorers and colonial-era figures as ordinary landmarks
  — **Columbus Circle is just a traffic circle; do not flag it**
- Political parties, party offices, government buildings, embassies, consulates
- Protest movements and their sites (e.g. Occupy Wall Street)
- Statues of political leaders of any country, including communist leaders
- Crime scenes, accident sites, assassination sites, terrorist-attack sites
  (these fail the separate "is it a usable reference point" test, not this one)
- Religious sites of any faith
- War memorials that mourn victims
- Military museums, decommissioned fortifications, heritage sites
- Anything merely old, colonial-era, or historically contested long ago

Candidates:

{items}

Output ONLY a JSON array, elements like:
{{"idx":0,"sensitive":false,"why":"ordinary landmark"}}
Keep `why` under 15 words. No other text."""


PROMPTS_AUDIT["sensitive"] = PROMPT_SENSITIVE_EN


def screen_sensitive(cand: pd.DataFrame, model="deepseek-v4-pro", batch=20,
                     verbose=True, workers=6) -> pd.DataFrame:
    """Add `sensitive` / `sensitive_why` columns to a candidate table.

    Separate cache file and separate cache key; `audit_decisions.jsonl` is not
    touched, so existing referenceability ratings and the kappa validation are
    entirely unaffected.
    """
    cand = cand.copy()
    cache = {}
    if SENSITIVE.exists():
        for line in SENSITIVE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[(r["qid"], r["model"])] = r

    # The criteria were narrowed, so the prompt changed and the cache key must
    # change with it; otherwise the old, broader verdicts get reused. The old
    # records stay in the jsonl rather than being deleted.
    tag = f"sens2|{model}"
    todo = cand[~cand.qid.map(lambda q: (q, tag) in cache)]
    if verbose:
        print(f"{len(cand)} candidates, {len(cand)-len(todo)} already screened, "
              f"{len(todo)} to go", flush=True)

    if len(todo):
        client = providers.REGISTRY[model]()
        todo = todo.reset_index(drop=True)
        with open(SENSITIVE, "a", encoding="utf-8") as f:
            for start in range(0, len(todo), batch):
                chunk = todo.iloc[start:start + batch]
                arr = _ask_recursive(client, chunk, verbose,
                                     prompt_lang="sensitive")
                if verbose:
                    tail = "" if len(arr) == len(chunk) else f"（{len(arr)}/{len(chunk)}）"
                    print(f"  batch {start//batch+1}/{-(-len(todo)//batch)} done{tail}",
                          flush=True)
                for d in arr:
                    i = d.get("idx")
                    if not isinstance(i, int) or i >= len(chunk):
                        continue
                    row = chunk.iloc[i]
                    rec = dict(qid=row.qid, label=row.label, model=tag,
                               auditor=model, sensitive=bool(d.get("sensitive")),
                               why=str(d.get("why", ""))[:60], ts=time.time())
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache[(row.qid, tag)] = rec

    cand["sensitive"] = [bool(cache.get((q, tag), {}).get("sensitive", False))
                         for q in cand.qid]
    cand["sensitive_why"] = [cache.get((q, tag), {}).get("why") for q in cand.qid]
    if verbose:
        n = int(cand.sensitive.sum())
        print(f"{n} flagged as sensitive", flush=True)
        for r in cand[cand.sensitive].itertuples():
            print(f"    ✗ {r.label}  —— {r.sensitive_why}", flush=True)
    return cand


# ------------------------------------------------------------- ladder building v2

#: Referenceability tiers. The middle tier matters as much as the extremes:
#: "locals use it, visitors do not know it" is the most interesting case, and
#: in practice it produced the condition that separated models most sharply.
FAMILIARITY_TIERS = {"high": (4, 5), "mid": (2, 3), "low": (0, 1)}


def build_ladder2(cand: pd.DataFrame, bands=None, per_cell=4,
                  tiers=None, seed=42, verbose=True,
                  site: str | None = None, self_qid: str | None = None) -> pd.DataFrame:
    """Build the ladder over distance x referenceability.

    Referenceability comes from the LLM rather than from pageviews: pageviews
    measure how many people look a place up, not how many people use it to say
    where they are (see the San Jose example at the top of this module).
    Pageviews are kept as an objective reference column but take no part in
    selection.

    `cand` must already have been through audit(); if ladder.add_prominence()
    was also run it will carry a pageviews column.

    `site` / `self_qid` exclude candidates that **leak the answer** -- ones whose
    name is the site's own place name (e.g. a candidate literally called
    "Notting Hill" near `london_nottinghill`). The prompt tells the model "I
    took this near X"; if X is the correct answer, that reference point is not
    testing reliance on an external hint of varying distance and
    referenceability, it is reading the answer aloud. See `self_names.py` for
    the full account. The hand-picked batch of 19 sites passes `site` (looked up
    in `SELF_NAMES`); batches whose sites are single Wikidata entities (chosen by
    `select_sites.py`) pass `self_qid` to exclude by qid directly. Both may be
    given together.
    """
    from .ladder import DEFAULT_BANDS
    from .self_names import is_self_name
    bands = bands or DEFAULT_BANDS
    tiers = tiers or FAMILIARITY_TIERS
    ok = cand[(cand.ok == True) & cand.familiarity.notna()].copy()  # noqa: E712
    ok["familiarity"] = pd.to_numeric(ok.familiarity, errors="coerce")
    ok = ok[ok.familiarity.notna()]
    # Politically contentious names, excluded by the separate screen_sensitive
    # pass. Some really are places tourists ask about, but a public benchmark
    # including them would import unrelated controversy.
    if "sensitive" in ok.columns:
        n0 = len(ok)
        ok = ok[ok.sensitive != True]                                # noqa: E712
        if verbose and n0 != len(ok):
            print(f"excluded {n0 - len(ok)} politically contentious names", flush=True)
    if site or self_qid:
        n0 = len(ok)
        if self_qid:
            ok = ok[ok.qid != self_qid]
        if site:
            leak = ok.name_en.fillna("").apply(lambda s: is_self_name(s, site))
            ok = ok[~leak]
        if verbose and n0 != len(ok):
            print(f"excluded {n0 - len(ok)} answer-leaking names "
                  "(candidate carries the site's own place name)", flush=True)
    if verbose:
        print(f"{len(ok)} candidates passed the LLM audit; referenceability "
              f"distribution {dict(ok.familiarity.astype(int).value_counts().sort_index())}")

    out = []
    for lo, hi in bands:
        band = ok[(ok.dist_km >= lo) & (ok.dist_km < hi)]
        if band.empty:
            if verbose:
                print(f"  [{lo}-{hi}km] no candidates", flush=True)
            continue
        for tname, (flo, fhi) in tiers.items():
            cell = band[(band.familiarity >= flo) & (band.familiarity <= fhi)]
            if cell.empty:
                continue
            # Sample **at random** within the cell; do not sort by distance.
            #
            # An earlier version ranked by proximity to the band centre, which
            # was wrong: the 1.5-3km band had 30 candidates at fam=5, and one
            # museum won purely because it happened to sit at 2.25km, the band
            # centre, beating a square at 2.02km. That preference has no
            # theoretical basis and only introduces uncontrolled selection bias.
            # Random sampling treats the cell as exchangeable and leaves
            # distance as a continuous covariate for the analysis stage.
            # One place can have several Wikidata entries (a station appeared
            # twice in practice), so de-duplicate by name before sampling,
            # otherwise one reference point occupies two ladder slots.
            cell = cell.drop_duplicates(subset=["name_zh"]).drop_duplicates(subset=["label"])
            take = min(per_cell, len(cell))
            for _, r in cell.sample(take, random_state=seed).iterrows():
                out.append(dict(band=f"{lo}-{hi}km", tier=tname,
                                name_zh=r.name_zh, name_en=r.name_en,
                                label_raw=r.label, qid=r.qid, dist_km=r.dist_km,
                                familiarity=int(r.familiarity),
                                pageviews=r.get("pageviews"),
                                n_in_cell=len(cell), n_taken=take))
    res = pd.DataFrame(out)
    return res.drop_duplicates(subset=["qid", "band"]).reset_index(drop=True)


NAME_PROMPT = """The places below are missing a {lang_name} name. Supply one for each.

Rules:
- If an **established** {lang_name} name exists, use it and set `established: true`
- Otherwise {fallback}, and set `established: false`
- Never leave it empty, never output null

{items}

Output only a JSON array, with elements shaped like:
{{"idx":0,"name":"Yin Changheng's Residence","established":false}}
Output nothing else."""

FALLBACK = {"en": "transliterate (romanise proper nouns; generic nouns may be "
                  "translated, e.g. a heritage site becomes 'X Former Site')"}
LANG_NAME = {"en": "English"}


def fill_missing_names(cand: pd.DataFrame, model="deepseek-v4-pro",
                       lang="en", batch=20, verbose=True) -> pd.DataFrame:
    """Fill in missing names, flagging whether each is an established name.

    **Why this is necessary**: the missing values are not random. Of 182 audited
    candidates at one site, 12.6% lacked an English name, and 100% of those were
    obscure heritage buildings with no English Wikidata entry -- all of them in
    the low referenceability tier. Without filling them, the low tier collapses
    under the English condition and the two language conditions stop being
    comparable, and the low tier is precisely the control.

    Transliterated entries are flagged `<lang>_established=False`, so the
    analysis can check robustness: does the conclusion survive dropping them?
    """
    col = f"name_{lang}"
    cand = cand.copy()
    if f"{lang}_established" not in cand:
        cand[f"{lang}_established"] = cand[col].notna()

    todo = cand[cand.ok.astype("boolean").fillna(False) & cand[col].isna()]
    if verbose:
        print(f"usable candidates missing a {lang} name: {len(todo)}", flush=True)
    if todo.empty:
        return cand

    client = providers.REGISTRY[model]()
    todo = todo.reset_index()
    filled = {}
    for start in range(0, len(todo), batch):
        chunk = todo.iloc[start:start + batch]
        items = "\n".join(f"{i}. {r.name_zh or r.label}" for i, r in enumerate(chunk.itertuples()))
        try:
            arr = json.loads(_strip_fence(_text_only(client, NAME_PROMPT.format(
                items=items, lang_name=LANG_NAME[lang], fallback=FALLBACK[lang]))))
        except Exception as e:
            if verbose:
                print(f"  batch {start//batch+1} failed: {type(e).__name__}", flush=True)
            continue
        for d in arr:
            i = d.get("idx")
            if isinstance(i, int) and i < len(chunk) and d.get("name"):
                filled[chunk.iloc[i]["index"]] = (str(d["name"]), bool(d.get("established")))
        if verbose:
            print(f"  batch {start//batch+1}/{-(-len(todo)//batch)} filled {len(filled)}", flush=True)

    for idx, (name, est) in filled.items():
        cand.loc[idx, col] = name
        cand.loc[idx, f"{lang}_established"] = est
    if verbose:
        print(f"still missing after filling: {int(cand[cand.ok.astype('boolean').fillna(False)][col].isna().sum())}", flush=True)
    return cand


def review_queue(cand: pd.DataFrame) -> pd.DataFrame:
    """Human review queue.

    Prioritises three kinds of row: rejected by the LLM, missing a canonical
    name, and sitting on a tier boundary (referenceability 1 or 2). Boundary
    ratings are the easiest to get wrong and they decide which tier a candidate
    lands in.
    """
    d = cand[cand.ok.notna()].copy()
    if d.empty:
        return d
    fam = pd.to_numeric(d.familiarity, errors="coerce")
    d["priority"] = (
        (~d.ok.astype(bool)).astype(int) * 2                  # rejected
        + (d.name_zh.isna() | d.name_en.isna()).astype(int)   # missing a name
        + fam.isin([1, 2]).astype(int)                        # on a tier boundary
    )
    cols = [c for c in ["label", "name_zh", "name_en", "dist_km", "ok",
                        "familiarity", "pageviews", "why", "priority"] if c in d]
    return d.sort_values(["priority", "dist_km"], ascending=[False, True])[cols]


def _first(*vals):
    """First non-null value. pandas represents missing as float(nan), and
    `nan or x` returns nan, so the check has to be explicit."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return None


def _first(*vals):
    """First non-null value. pandas represents missing as float(nan), and
    `nan or x` returns nan, so the check has to be explicit."""
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        t = str(v).strip()
        if t and t.lower() != "nan":
            return t
    return None


def to_context(row, lang="en") -> str:
    """Build the context sentence, falling back between languages when one name
    is missing."""
    if lang != "en":
        raise ValueError(
            f"only the English condition ships in this release, got lang={lang!r}")
    zh, en, raw = row.get("name_zh"), row.get("name_en"), row.get("label_raw")
    name = _first(en, zh, raw)     # name_zh survives in the CSVs as a fallback
    return f"I took this photo near {name}. "
