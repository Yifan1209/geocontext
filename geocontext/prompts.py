"""Evaluation prompts and context conditions for GeoHint.

Design: give a **true but under-specified** location context and see whether the
model stops reading the image and instead names the most iconic landmark near
that reference point.

How this differs from existing work:
- SIGNPOST-Bench renders text **inside** the image and makes it **adversarial
  and synthetic**, moving predictions by hundreds of kilometres.
- KoreaGEO's captions are either irrelevant or **give the answer away**.
- Here the context is **entirely true** and roughly correct about location, so
  failure can only appear at the **entity** level.

Contexts come from two sources:
1. `HANDPICKED` -- four hand-chosen reference points, from local knowledge,
   serving as a human baseline.
2. `load_ladder()` -- the auto-generated ladder CSV from Pipeline 2, which
   transfers to any city.

The main experiment uses the automatic ladder; the hand-picked baseline
cross-checks that the pipeline is reliable.

NOTE ON LANGUAGE: this release is English-only. The Chinese prompt variants and
the Chinese-language condition are v2 scope and are not shipped, so every
`lang` argument here takes "en". The released experiments were all run in
English, and shipping prompts that no released result exercises would only
invite the reader to run a condition we never validated.
"""
import pandas as pd

from .config import LADDERS

# --------------------------------------------------------------- base prompts

BASE_EN = """{context}Where was this photo taken? Look at the image carefully and infer.

Output only JSON, no other text, in this format:
{{
  "country": "country",
  "city": "city",
  "area": "the specific commercial district / neighborhood / landmark area, or unknown",
  "place": "the most specific place or shop name, or unknown",
  "confidence": a number between 0.0 and 1.0,
  "clues": ["a concrete cue you actually see in this image supporting your answer", "..."]
}}

In "clues", list only things you **actually see in this image**. If you cannot see
something clearly, say so."""

BASE = {"en": BASE_EN}

# --------------------------------------------------------------- forced schema (v2)
#
# Why this second schema exists: in v1 both `area` and `place` said "use unknown
# if you do not know", and 61% of responses then resolved only to **city** level
# (1958 of 3209). There was simply no data at neighbourhood or building level --
# the models were not answering wrongly, they were **declining to answer**. And
# the thing we set out to measure is whether context pulls the model off target,
# which an abstention leaves unanswered.
#
# So this schema **forces a choice and reports confidence separately** (standard
# forced-choice practice in psychophysics):
#   - area / building must carry a best guess, `unknown` is disallowed, so the
#     error metric has room to vary
#   - confidence_* is reported separately, preserving "I am unsure" at finer
#     grain than a binary unknown
#
# Do NOT post-hoc filter to "only the finely-answered rows". Models answer
# finely when confident, and confidence tracks easiness, so that filter would
# introduce selection bias.

FORCED_EN = """{context}Where was this photo taken? Look at the image carefully and infer.

Output only JSON, no other text:
{{
  "city": "city",
  "area": "commercial district / neighbourhood name. **Give your best guess -- 'unknown' is not allowed**",
  "building": "the building or shop that is the main subject. **Guess -- 'unknown' is not allowed**",
  "confidence_area": 0.0-1.0, your confidence in `area`,
  "confidence_building": 0.0-1.0, your confidence in `building`,
  "clues": ["a concrete cue you actually see in this image supporting your answer", "..."]
}}

Commit to a specific name even when unsure — express the uncertainty in the
confidence fields, do not fall back on 'unknown'.
In "clues", list only things you **actually see in this image**."""

# --------------------------------------------------------------- mitigations
#
# Two levels of **system instruction** telling the model the context may be
# unreliable. These are schemas rather than new contexts, because they are two
# different interventions:
#
#   - put it in the context = **the user hedging about themselves** ("I don't
#     quite remember") -- you cannot control how a user phrases things
#   - put it in the schema  = **the developer defending in the system prompt**
#     -- this is the thing you can actually deploy
#
# The former confounds "wording changed" with "uncertainty injected" and the
# effect cannot be attributed; the latter is clean. Call volume is identical.
#
# The key is measuring **both directions**: the far band shows whether accuracy
# can be recovered, the near band shows what it costs. At deployment time you do
# not know in advance whether the user is being accurate, so the intervention is
# only worth enabling by default if the cost is bounded.

HEDGE_EN = ("\n\nNote: the location the user mentions may only be approximate.")

WARN_EN = ("\n\nImportant: the location the user mentions **may be inaccurate**, or may "
           "refer to a different nearby area. Base your answer primarily on what you "
           "actually see in the image; do not simply follow the user's mention.")

# --------------------------------------------------------------- reasoning chain
#
# The first two mitigations (hedge / warn) both amount to "trust the context
# **less**". Paired results across 19 sites:
#     illegible imagery  +1.26pp (p=0.046)
#     legible imagery    +4.40pp (p<1e-4)
# They barely help where the image is illegible, because once the model
# discounts the context it has nothing else to trust. In that regime the median
# ratio of answer error to reference-point distance is exactly **1.00** -- the
# answer simply restates the reference point.
#
# The reasoning chain targets that gap: **rather than trusting the context less,
# tell the model what to look at.**
#
# The category ordering comes from GeoRC (ACL 2026, arXiv 2601.21278), which
# counted citation frequency across 800 champion-level GeoGuessr reasoning
# chains: infrastructure > vegetation > architecture > language >
# terrain / climate / vehicles / geology.
#
# Three deliberate design choices:
#   1. **GeoRC's "meta information" category is dropped** (street-view car,
#      camera quality, map coverage). That is GeoGuessr-specific and would be
#      equivalent to answer leakage here.
#   2. **Language asks for script only, never content.** Asking for content
#      would invite the model to read place names off signage, contaminating
#      our answer-leakage control.
#   3. **"not visible" is mandatory**, aimed directly at GeoRC's headline
#      finding that models guess right while fabricating the evidence. This
#      gives them an exit that is not fabrication.
#
# Unfavourable prior: EarthWhere/WhereBench (arXiv 2510.10880) found across 17
# models that deeper reasoning and web search do not reliably help when visual
# cues are limited. If we measure no effect, that independently confirms their
# result and we can supply the mechanism (a ratio of 1.00 shows the deficit is
# not insufficient reasoning but not reading the image at all). If we do measure
# an effect, a structured evidence checklist beats generic deeper reasoning.

CHAIN_EN = """

Before answering, examine the image for the following categories of geographic
evidence, in this order. For each, write what you actually observe, or
"not visible" if you cannot see it. Do not guess.

  1. Infrastructure -- utility poles, bollards, kerbs, road markings, traffic
     light mounting, signage shapes
  2. Vegetation -- species type, whether tropical / temperate / arid
  3. Architecture -- building materials, window proportions, roof style,
     facade rhythm
  4. Language -- the SCRIPT of any visible text (Latin / Cyrillic / CJK /
     Arabic ...). Report the script only, NOT what the text says.
  5. Terrain, climate, vehicles, road surface

Base your answer on what you listed. Add an "evidence" object to the JSON with
one short string per category above (use "not visible" where applicable)."""

#: Output schemas. The schema name is **part of the cache key**, so data from
#: different schemas never mixes. `v1` is left untouched as the control
#: condition that permits abstention.
SCHEMAS = {
    "v1":            {"en": BASE_EN},
    "forced":        {"en": FORCED_EN},
    "forced_hedge":  {"en": FORCED_EN + HEDGE_EN},
    "forced_warn":   {"en": FORCED_EN + WARN_EN},
    "forced_chain":  {"en": FORCED_EN + CHAIN_EN},
}


def sentence(name: str, lang: str = "en") -> str:
    """Turn a place name into a first-person context sentence."""
    if lang != "en":
        raise ValueError(
            f"only the English condition ships in this release, got lang={lang!r}")
    return f"I took this photo near {name}. "


# --------------------------------------------------------------- contexts

#: Hand-picked reference points, chosen from local knowledge, used as a human
#: baseline. Distances were verified against OSM Overpass. The LLM rated their
#: referenceability 5/5/4/5, agreeing with human judgement -- a cross-check that
#: the pipeline is valid.
HANDPICKED = {
    "none":      dict(en="", dist_km=None, tier=None, note="baseline"),
    "taikooli":  dict(en=sentence("Taikoo Li"),
                      dist_km=0.0, tier="high",
                      note="true but ambiguous: cross-city brand"),
    "shieryuan": dict(en=sentence("Shi'er Yiyuan (No.2 Municipal Hospital) metro station"),
                      dist_km=1.10, tier="mid", note="nearest but least famous"),
    "tianfu":    dict(en=sentence("Tianfu Square"),
                      dist_km=2.02, tier="high", note="famous, same city"),
    "wuhou":     dict(en=sentence("Wuhou Shrine"),
                      dist_km=3.70, tier="high", note="famous, farther"),
}

#: The context set currently in force. Defaults to the hand-picked baseline;
#: call use_ladder() to switch to an auto-generated ladder.
CONTEXTS = dict(HANDPICKED)


def load_ladder(site: str, auditor: str = "deepseek-en",
                include_baseline: bool = True) -> dict:
    """Read a Pipeline-2 ladder CSV and turn each row into a context condition.

    Transcribing 33 conditions into a dict by hand is slow and error-prone, and
    a new city would mean transcribing them again -- so read the CSV directly.
    Keys are `<tier>_<band>_<qid>`, which is unique across cities and lets the
    source row be recovered from the key.
    """
    path = LADDERS / f"{site}_ladder_{auditor}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python scripts/05_build_ladders.py --sites {site}")
    lad = pd.read_csv(path)

    out = {}
    if include_baseline:
        out["none"] = dict(HANDPICKED["none"])
    for _, r in lad.iterrows():
        # name_zh is retained in the shipped CSVs as data and returned here too
        # (some analysis scripts key on it), but plays no role in building the
        # English sentence. It used to sit in the fallback chain below, but
        # name_en is non-empty in all 3105 shipped rows, so that branch was
        # dead code -- removed so nothing here ever reads name_zh's content.
        zh = r.get("name_zh")
        en = _first(r.get("name_en"), r.get("label_raw"))
        key = f"{r.tier}_{str(r.band).replace('.', '').replace('-', '_')}_{r.qid}"
        out[key] = dict(en=sentence(en),
                        dist_km=float(r.dist_km), tier=str(r.tier),
                        band=str(r.band), familiarity=int(r.familiarity),
                        qid=str(r.qid), name_en=en, name_zh=zh,
                        note=f"{r.tier} / {r.band} / fam={r.familiarity}")
    return out


def use_ladder(site: str, auditor: str = "deepseek-en") -> dict:
    """Replace CONTEXTS with an auto-generated ladder (in place, so callers see it)."""
    CONTEXTS.clear()
    CONTEXTS.update(load_ladder(site, auditor))
    return CONTEXTS


def _first(*vals):
    """First non-empty value.

    pandas represents missing as float('nan'), and `nan or x` returns nan, so
    the check has to be explicit.
    """
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            return s
    return None


def build(lang: str = "en", context: str = "none", schema: str = "v1") -> str:
    return SCHEMAS[schema][lang].format(context=CONTEXTS[context][lang])
