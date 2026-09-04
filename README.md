# GeoContext

**When telling a vision–language model where you are makes it worse.**

A tourist photographs a building, asks a VLM "where is this?", and adds a rough
whereabouts: *"I took this near Bastille."* The sentence is **true**. This
benchmark measures whether models use it in a way that scales with how
informative it actually is.

They do not.

109 sites in 30 cities, 5 models, 21 933 scored GeoHint responses and 6 270
GeoVerify responses. English only.

---

## The four findings

**1. Reliance on the context does not scale with the context's informativeness.**

How often a model echoes the reference point back in its answer is nearly flat
across referenceability:

| Referenceability tier | Echo rate | n |
|---|---|---|
| High (everyone navigates by it) | 24.4 % | 3209 |
| Mid (locals use it) | 25.8 % | 3193 |
| Low (almost nobody says it) | 22.9 % | 1940 |

It responds weakly, and non-monotonically, to distance: 27.4 % → 22.1 % → 24.6 %
from 0.5–1.5 km to 3–6 km — the drop happens in the first step and then
reverses, so a twelvefold change in distance moves the rate by 2.8 points. A
calibrated model would discount a distant, obscure reference point roughly in
proportion to how little it constrains the answer. Meanwhile the *error* that
reliance produces grows steadily over the same range — the cost scales with
distance even though the reliance itself barely does.

Fitting that cost gives a rate: regressing `log10(error_km)` on the
reference-point distance gives β = +0.0820 (p = 1.5·10⁻²⁰ with standard
errors clustered by site, n = 8274 responses over 109 sites). The scale
has two interpretable ends — 0 means the hint is ignored, 0.176 means the answer
*is* the hint — so the models sit 47 % of the way toward reciting it.

**2. The consequence is conditional on whether the image is readable at all.**

Split the sites by *baseline readability* — the model's own sub-kilometre hit
rate with **no** context, `forced` schema. Below 50 % we call the imagery
illegible, at or above it legible; 65 sites fall below and 44 above. The same
context is a large net gain in one regime and a net loss in the other.

The 19 hand-picked sites and the 90 algorithmically selected ones share no step
of their selection procedure, and show the same signature:

| Hand-picked (19 sites) | Sites | No context | 0.5–1.5 km | 1.5–3 km | 3–6 km |
|---|---|---|---|---|---|
| Illegible | 10 | 28.0 % | **42.9 %** | 22.0 % | 20.0 % |
| Legible | 9 | 70.4 % | 68.7 % | 49.8 % | 46.4 % |

| Algorithmic (90 sites) | Sites | No context | 0.5–1.5 km | 1.5–3 km | 3–6 km |
|---|---|---|---|---|---|
| Illegible | 55 | 13.7 % | **29.2 %** | 12.8 % | 10.6 % |
| Legible | 35 | 69.1 % | 67.0 % | 53.9 % | 50.8 % |

Hit rate (`<1 km` and not resolved at city level), `forced` schema. In the
illegible regime the median ratio of answer error to reference-point distance is
exactly 1.00 — the answer *is* the reference point. In the legible regime the
image does the work instead.

> **The "No context" column needs a correction before it is compared with the
> others.** Each site's regime is assigned from that same column, so a site
> that lands in the legible group partly because its baseline draw came out
> high carries that upward noise into the number the context conditions are
> measured against — regression to the mean, no true effect required. Splitting
> each site's baseline responses at random, assigning the regime from one half
> and reading the column off the other, leaves **one** model less accurate with
> a nearby hint than without one: gemini-flash, at −8.4 points. claude-opus-5
> and qwen3-vl-235b move to within one standard deviation of zero. See
> `analysis/regime_split_half.py` and Table 3 of the paper. The correction is
> needed for this comparison only — everywhere else the regime appears, the
> quantities compared are context conditions, which never enter the
> assignment.

**3. Warning the model that the user may be wrong is obeyed, but only pays where
there is something else to trust.**

Adding *"the location the user mentions may be inaccurate"* cuts the echo rate
by 8.4 points (illegible: 29.4 % → 21.0 %) and 6.1 points (legible: 18.7 % →
12.6 %) — clearly followed in both regimes. On strictly paired data (same
site/image/model/language/reference point, 8215 pairs across 109 sites) it buys
**+3.94 points** where the image is legible (p = 0.0008) and **+0.46 points**
where it is illegible, indistinguishable from zero (p = 0.49). Both p-values
resample whole sites, not individual responses.

Obedience is therefore not what separates the two regimes; what separates them
is whether obeying helps. The warning's only action is *trust the text less*,
which is worth something only when there is something else to trust. Nothing
that merely discounts the context repairs the illegible regime.

A structured evidence checklist — enumerate infrastructure, vegetation,
architecture, script and terrain before answering — is null on every
pre-registered criterion. The models comply (an `evidence` object appears in
≥ 99.7 % of responses), so the null is about the intervention rather than about
compliance with it.

**4. No model can yet verify a location claim at the tolerance a deployment
needs.** GeoVerify asks whether a photo was taken within 150 m of a *claimed*
location — the check a platform runs when a driver reports a drop-off, or a
dockless bike is reported parked in its bay. Existing defences against falsified
positions reason over the very positioning signals an attacker manipulates; a
photograph is attractive because it is a different channel.

Controls pass: swapping in a photo from another continent drops false
acceptances to 5.0 %, replacing it with a uniform grey field drops them to 0.0 %
(main arm: 37.7 %) — the models are reading the image. But sensitivity in the
band that matters is not there. Scored with signal-detection theory, which
separates discrimination `d′` from a mere disposition to answer no:

| Model | H | 0.15–0.3 km | 0.3–0.7 | 0.7–1.5 | 1.5–3 | 3–6 | δ₁ |
|---|---|---|---|---|---|---|---|
| gemini-flash | 93.5 % | 0.68 | **1.61** | 2.16 | 2.34 | 2.36 | 0.3–0.7 km |
| claude-opus-5 | 47.2 % | 0.32 | 0.75 | **1.24** | 1.44 | 1.95 | 0.7–1.5 km |
| claude-haiku-4-5 | 78.7 % | 0.22 | 0.40 | 0.44 | 0.55 | 0.82 | **never** |
| qwen3-vl-235b | 70.4 % | 0.24 | 0.46 | 0.68 | 0.87 | **1.07** | 3–6 km |
| qwen3-vl-8b | 44.4 % | 0.51 | 0.55 | 0.66 | 0.89 | **1.13** | 3–6 km |

**No model reaches `d′ = 1` in the leftmost band — the tolerance itself.**
"Dropped off one street away" is exactly the discrimination a deployment needs
and exactly the one models fail. False acceptances are not hedged either: median
self-reported confidence 0.85, with 83.8 % of them stated at 0.8 or above, so
the gap cannot be closed by thresholding on confidence.

Reporting `d′` and the criterion separately is not fastidiousness — it is
load-bearing here. `gemini-flash` is strongly yes-biased (highest hit rate, but
79 % false acceptances in the nearest band) while `claude-opus-5` is no-biased
throughout (second-lowest hit rate, fewest false acceptances, yet the
second-best sensitivity). Accuracy alone would rank these two the wrong way
round.

---

## Why the metric is not a coordinate distance

Binary scoring ("does the answer contain *Shibuya*?") depends on whether a
neighbourhood happens to have a usable name, so it is not comparable across
cities. We geocode the answer and measure haversine distance, trying
`building → area → city` and recording which level resolved.

**Four measurement faults reversed the sign of our own headline result before we
found them.** Each is invisible in aggregate statistics, so each is documented
in full in `geocontext/geocode.py`:

1. Parenthetical aliases (`Shibuya (Miyamasuzaka)`) fail exact-label search and
   fall back to city level — which is not an error message but a
   plausible-looking 3.42 km.
2. Linear features cannot be scored as points. `Meiji-dōri + Shibuya` is correct
   and `Meiji-dōri + Harajuku` is wrong, with identical `place` fields; the
   information lives in `area`.
3. A global fallback when no candidate lies near the model's own answered city
   turned "we cannot verify this" into "wrong by 16 609 km".
4. **The city-level fallback distance is a constant, and at one site it sits
   below the hit threshold.** Paris's city centroid is 0.761 km from the site,
   so every answer that resolved no finer than "Paris" counted as a
   sub-kilometre hit — inflating that site by 20.7 points and masking the
   accuracy drop in finding 2.

`geocontext.geocode.hit()` therefore treats city-level resolution as
**censored**: it states only that the answer is no finer than the city. Across
the release the four corrections cut city-level resolution from 51.0 % to 9.0 %
of responses.

**General rule this leaves behind:** any error value that recurs as a constant
is a signal of a resolution failure, not a measurement.

---

## Scope

**Dense, walkable urban cores only.** We make the inclusion criterion
measurable: `L_district`, the median nearest-neighbour distance among OSM
`place=neighbourhood|suburb|quarter` point nodes within 6 km. The 19
hand-picked sites all fall in 0.33–0.65 km — a spread of 1.98× — so fixed
absolute distance bands are used. The 90 algorithmically selected sites were
screened at city level against the same range (0.32–0.65 km); per-site
verification for those is still outstanding.

San Jose, measured and **excluded**, gives 0.99 km with 21 neighbourhood nodes
in the same radius. Car-oriented low-density areas are out of scope: there
"nearby" is set by drive time, not neighbourhood granularity.

> One caveat worth stating loudly: `L_district` measures granularity, not
> motorised accessibility. A city can score fine-grained by this metric and
> still feel spread out to a visitor if getting between two adjacent
> neighbourhoods normally means a taxi rather than a walk. Those are different
> constructs, and future work should consider bands defined by travel time
> instead of a fixed radius.

Excluding addressing units matters. Of 800 "neighbourhoods" within 6 km of
Shibuya, 573 (72 %) are *X-chōme* street-numbering blocks. Including them yields
0.24 km for Tokyo vs 0.51 km for Paris and supports a conclusion — "Tokyo's
neighbourhoods are twice as fine" — that is purely an artefact of Japanese OSM
tagging convention.

---

## Layout

```
geocontext/        library
scripts/           the pipeline, in execution order
  00_select_sites.py     city screening + automatic site selection
  01_survey_sites.py     Mapillary coverage check before committing to a site
  02_fetch_images.py     download + brightness filter
  03_audit_images.py     model audit: answer leakage, scene type, quality
  04_select_images.py    apply human picks + explicit exclusion list
  05_build_ladders.py    Wikidata → LLM audit → distance × referenceability grid
  06_run_eval.py         the evaluation itself
  07_geocode_answers.py  answers → continuous error_km
  08_analyze.py          headline tables (GeoHint)
  09_build_verify_trials.py  signal/decoy trial table (GeoVerify)
  10_run_verify.py       the GeoVerify evaluation
tools/             human-in-the-loop viewers (image picker, response browser, audit viz)
analysis/          the analyses behind specific paper claims
data/              ladders, audit decisions, image metadata, responses
paper/             LaTeX source
```

Every script in `analysis/` runs with no arguments and prints the numbers behind
a specific claim in the paper. Two are worth calling out:
`regime_split_half.py` measures how much the regime comparison is inflated by
using the same responses to assign and to compare, and `cluster_inference.py`
recomputes every p-value with the site as the unit of resampling.

## Data

We release **image IDs and coordinates, not Mapillary pixels** (CC-BY-SA, and
large). `scripts/02_fetch_images.py` re-fetches them. GeoHint answers are scored
against the site's ground-truth coordinate in `data/sites.csv`, one per site;
`data/verify_trials.csv` additionally carries per-image camera coordinates for
the GeoVerify subset.

| File | What |
|---|---|
| `data/answers_judged.csv` | every scored GeoHint response: model, context condition, parsed place, `error_km`, resolution level, hit. Raw response text for the other schemas is available on request |
| `data/sites.csv` | all 109 sites, with `selection_method` (hand-picked vs algorithmic) |
| `data/ladders/*.csv` | the 109 context ladders |
| `data/streetview_meta_selected.csv` | the 159 images used in evaluation, with Mapillary image ID |
| `data/verify_trials.csv` | GeoVerify signal/decoy/control trial table |
| `data/verify_raw.jsonl` | every raw GeoVerify model response |
| `data/forced_chain_raw.jsonl` | raw responses for the evidence-checklist schema, whose `evidence` field the judged table does not carry |
| `data/audit_decisions.jsonl` | every reference-point audit decision |
| `data/audit_sensitive.jsonl` | contentious-name screening decisions |
| `data/candidate_referenceability.csv` | `qid`, Wikidata sitelinks, audited referenceability — for checking whether sitelinks could stand in for the audit |
| `data/image_audit.csv` | unified per-image audit table: AI verdict and human verdict as columns on the same row |
| `data/district_scale.csv` | `L_district` per site |
| `data/city_pool.csv` | the 58 candidate cities with street-view counts, `L_district` and the screening outcome |

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env      # add MAPILLARY_TOKEN, and the model API keys you need
python scripts/05_build_ladders.py --sites <site> ...
python scripts/06_run_eval.py --source streetview --site <site> --ladder <site> ...
python scripts/07_geocode_answers.py
python scripts/08_analyze.py
```

Every stage caches to disk and skips completed work, so an interrupted run is
resumed by re-issuing the same command.

## Two notes on method

**The auditor must not be an evaluated model**, or the benchmark sets its own
exam. Ours is text-only, so it cannot see the images.

**Automated image QA does not replace human review.** Across 429 audited images
the human and the model agree 80.0 % of the time, but Cohen's κ = 0.185: the
high raw agreement comes from the large majority of images that are
unproblematic for both, and the two rubrics are close to independent where it
matters. Each direction of disagreement has a cause. The human excludes 65
images the model rates usable, almost all under a criterion the model is never
asked about — "nothing in this frame identifies a location". The model rejects
21 the human keeps, 17 of them shot from inside a vehicle, which the human
accepts whenever the street is legible through the windscreen.

Nine images carry GPS coordinates burned in by a vehicle-mounted camera, and
neither reviewer catches all nine: the model's overlay flag fires on 6, the
human excludes 8, one of those exclusions rests on the model's flag alone, and
one image slips past both — that one was never sampled into the evaluation, and
no image carrying coordinates appears in it. The model's flag also fires twice
on overlays naming no place at all (an `Uber` interface label, a `BLACKVUE`
device string). Exclusions therefore run off the human verdict with the model's
leakage flags reviewed case by case, and both columns ship so the agreement can
be recomputed.

## Licence

Code MIT. Derived data CC-BY-SA 4.0, following Mapillary. Wikidata-derived
fields CC0.
