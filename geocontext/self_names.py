"""Per-site self-name tokens, for excluding reference points that leak the answer.

## What this fixes

Some candidates in the ladder are named after **the site itself** -- for example
a candidate literally called "Notting Hill" sitting 600 m from the
`london_nottinghill` site. The prompt shown to the model is "I took this photo
near X". If X is itself the correct answer's name, that reference point is not
testing whether the model over-relies on an external hint of a given distance
and referenceability. It is reading the answer aloud.

Found when a reader opened the `london_nottinghill` ladder CSV and saw two rows
named "Notting Hill" and "Notting Hill Gate tube station". A sweep across all 19
sites then found **5 of 387 (1.3%)** such rows in the ladders actually used for
evaluation, spread over 3 sites. Checking hit rates row by row showed the impact
is uneven: most matched their cell-mates, but `sf_mission`'s "Mission Dolores
Park" (high referenceability, near band) scored 93.3% against 13.3% for the
other reference point in the same cell. That one was genuinely inflating results.

The two existing audit passes do not catch this. Content auditing (`ok`) and
sensitive-name screening both pass these candidates, because there is nothing
wrong with the name itself -- "Notting Hill" is a perfectly ordinary place. It
needs its own check.

## Why this table is maintained by hand

The 19 hand-picked sites are **neighbourhoods**, not single Wikidata entities,
so there is no clean "official name" field to compare against; the mapping has
to be written down.

Sites chosen by `select_sites.py` do not need this table: each is a single
Wikidata entity, so `build_ladder2(self_qid=...)` can exclude by QID directly.

## A trap worth recording

The first attempted fix removed the leaking rows from the candidate pool and
re-ran per-cell sampling. That was wrong: `pandas.sample(random_state=seed)` is
sensitive to the *size* of the frame, so dropping rows re-rolled the sampling in
**cells that had nothing to do with the leak**. One site swapped a reference
point in an entirely unrelated band, which would have invalidated already
collected model responses for that cell. The applied fix deletes exactly the
offending rows and leaves every other row untouched, letting the affected cells
shrink rather than resampling around them.
"""

#: site_id -> name fragments that would leak the answer (lower-case, substring match)
SELF_NAMES: dict[str, list[str]] = {
    "london_nottinghill": ["notting hill"],
    "london_shoreditch":  ["shoreditch"],
    "london_covent":      ["covent garden"],
    "paris_marais":       ["marais"],
    "paris_montmartre":   ["montmartre"],
    "paris_bastille":     ["bastille"],
    "paris_cite":         ["île de la cité", "ile de la cite", "cité"],
    "paris_canal":        ["canal saint-martin", "canal st-martin", "canal st martin"],
    "barcelona_gracia":   ["gràcia", "gracia"],
    "barcelona_born":     ["el born", "born"],
    "sf_mission":         ["mission"],
    "sf_northbeach":      ["north beach"],
    "sf_hayes":           ["hayes valley", "hayes"],
    "cdmx_roma":          ["roma norte", "roma"],
    "cdmx_condesa":       ["condesa"],
    "tokyo_shimokita":    ["shimokitazawa", "shimokita"],
    "tokyo_yanaka":       ["yanaka"],
    "tokyo_shibuya":      ["shibuya"],
    "nyc_soho":           ["soho"],
}


def is_self_name(name: str, site: str) -> bool:
    """Whether a candidate's name contains the site's own place name.

    Case-insensitive substring match.
    """
    toks = SELF_NAMES.get(site)
    if not toks or not isinstance(name, str):
        return False
    s = name.lower()
    return any(t in s for t in toks)
