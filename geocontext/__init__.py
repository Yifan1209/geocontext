"""geocontext -- library code for the GeoContext benchmark.

Data flow:

    streetview.py  Mapillary retrieval  --> data/streetview/ + streetview_meta_*.csv
    audit.py       image / place-name auditing --> data/audit_decisions.jsonl
    ladder.py      Wikidata reference-point mining + referenceability rating
                                        --> results/ladders/*_ladder_*.csv
                                               |
    prompts.py     assembles prompts for the five schemas
    providers.py   per-vendor API adapters
    runner.py      calls + caching      --> results/e1_raw.jsonl (one line per call)
    geocode.py     place name -> coordinate + scoring --> results/answers_geocoded.pkl
    viz.py         HTML for manual inspection

config.py supplies path constants and .env key loading, and every module above
depends on it.

NOTE: this file deliberately does NOT eagerly `from . import ...`.
`data.py` / `llm_filter.py` / `scoring.py` are leftovers from the early
MMS-VPR stage; the current pipeline does not use them and they are not shipped.
An earlier version hard-coded `from . import data, scoring` here, which made
*every* module in the released package raise ImportError, while looking fine
locally because those files still existed. Import what you need:

    from geocontext import config, geocode, ladder
"""

__all__ = ["config", "providers", "prompts", "runner", "geocode", "ladder",
           "audit", "streetview", "panorama", "viz"]
