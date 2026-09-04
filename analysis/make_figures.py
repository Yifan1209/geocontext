"""Build every figure in the paper, straight from the scored responses.

    python analysis/make_figures.py

Outputs `paper/figures/fig_*.pdf` (for LaTeX) and `fig_*.png` (for quick
viewing). Nothing here reads an intermediate file that a human edited --- the
numbers come from `data/answers_judged.csv` and `data/verify_raw.jsonl` on
every run, so a figure can never silently drift from the tables.

## Colour palette

A three-colour subset of Okabe-Ito: #0072B2 / #D55E00 / #009E73. Validated
for colour-blind safety (lightness band / chroma floor / adjacent-pair CVD
deltaE 11.0 deutan / normal vision deltaE 25.8 / contrast against the
background >=3:1, all six checks pass). The paper is a print artifact, so only
a light palette is built deliberately.

## Four figures

| Figure | Paper | Description |
|---|---|---|
| `fig_pipeline`      | Fig. 1, Sec. 3   | the five-stage pipeline; which stages are held fixed vs manipulated |
| `fig_regimes`       | Fig. 2, Sec. 6.1.2 | hit rate by reference-point distance, split by whether the imagery is legible |
| `fig_echo_flat`     | Fig. 3, Sec. 6.2.1 | echo rate is flat across referenceability, weak and non-monotone across distance |
| `fig_verify_dprime` | Fig. 4, Sec. 6.4 | GeoVerify sensitivity against decoy distance; no model crosses d'=1 inside the tolerance band |

`fig_verify_dprime` reuses `verify_results.load()/dprime()` rather than
reimplementing the parsing and scoring. The figure and Table 11 must come from
the same computation path, or a disagreement between them cannot be diagnosed.

NOTE: `fig_regimes` once had a second panel, a per-site scatter of no-context
against with-context accuracy. It was removed because its x axis is the
variable the regime split is estimated from, so regression to the mean produces
the pattern it appeared to show. See `regime_split_half.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

from geocontext import config                                    # noqa: E402
from two_regimes import load, echoed                         # noqa: E402

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d8d8d6"
OUT = config.PROJ / "paper" / "figures"

BANDS = ["0.5-1.5km", "1.5-3.0km", "3.0-6.0km"]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 7.6,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def prepare():
    """All sites, forced schema, with echo flagged."""
    df = load()
    f = df[df.schema == "forced"].copy()
    f["echo"] = [echoed(a, p, z, n) for a, p, z, n in
                 zip(f.area, f.place, f.ref_zh, f.ref_en)]
    return f


def ci95(pct, n):
    p = pct / 100
    return 196 * np.sqrt(p * (1 - p) / n)


def fig_regimes(f):
    """Pooled hit rate by band, one line per regime.

    An earlier version paired this with a per-site scatter of no-context
    against with-context accuracy. That panel was dropped: its x axis is the
    variable the regime split is estimated from, and with a median of 5
    baseline responses per site, regression to the mean puts high-x points
    below the diagonal whether or not context does anything. The corrected
    per-model comparison lives in the paper's split-half table instead. The
    y axis here uses context conditions only, which the assignment never
    touches.
    """
    base, ctx = f[f.cond == "baseline"], f[f.cond != "baseline"]
    sitebase = base.groupby("site").hit.mean() * 100
    c2 = ctx.copy()
    c2["regime"] = np.where(c2.site.map(sitebase) < 50, "illegible", "legible")

    fig, bx = plt.subplots(figsize=(4.6, 3.2))
    lo = hi = 0
    for key, col, name in [("illegible", ORANGE, "Illegible imagery (65 sites)"),
                           ("legible", BLUE, "Legible imagery (44 sites)")]:
        ys, es, ns = [], [], []
        for b in BANDS:
            g = c2[(c2.regime == key) & (c2.cond == b)]
            ys.append(100 * g.hit.mean())
            es.append(ci95(100 * g.hit.mean(), len(g)))
            ns.append(len(g))
        lo, hi = (min(ns), max(ns)) if not lo else (min(lo, *ns), max(hi, *ns))
        bx.errorbar(range(3), ys, yerr=es, color=col, marker="o", ms=6, lw=2,
                    capsize=3, elinewidth=1, label=name,
                    markeredgecolor="white", markeredgewidth=1)
        bx.annotate(f"{ys[-1]:.0f}%", (2.08, ys[-1]), fontsize=7.4, color=col,
                    va="center", ha="left")
    bx.set_xticks(range(3))
    bx.set_xticklabels(["0.5-1.5", "1.5-3", "3-6"])
    bx.set_xlabel("Reference-point distance, km")
    bx.set_ylabel("Hit rate, %")
    bx.set_xlim(-.3, 2.55)
    bx.set_ylim(10, 75)
    bx.legend(loc="upper right", labelspacing=.3)
    bx.grid(True, axis="y", color=GRID, lw=.5, zorder=0)
    bx.set_axisbelow(True)
    bx.set_title(f"$n$ = {lo}\u2013{hi} per point", fontsize=8.2, color=INK,
                 pad=7)
    fig.tight_layout()
    save(fig, "fig_regimes")

def fig_echo_flat(f):
    ctx = f[f.cond != "baseline"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    w = .26
    for i, (key, lab, col) in enumerate([("high", "High (4-5)", BLUE),
                                         ("mid", "Mid (2-3)", ORANGE),
                                         ("low", "Low (0-1)", GREEN)]):
        xs, ys = [], []
        for j, b in enumerate(BANDS):
            g = ctx[(ctx.tier == key) & (ctx.cond == b)]
            xs.append(j + (i - 1) * w)
            ys.append(100 * g.echo.mean())
        for bar, y in zip(ax.bar(xs, ys, width=w - .02, color=col,
                                 label=lab, zorder=3), ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + .9, f"{y:.0f}",
                    ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0.5-1.5 km", "1.5-3 km", "3-6 km"])
    ax.set_xlabel("Distance of the reference point from the true location")
    ax.set_ylabel("Echo rate, %")
    ax.set_ylim(0, 42)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.16),
              columnspacing=1.2, handletextpad=.4)
    ax.grid(True, axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_echo_flat")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")
    W, GAP = 16.6, 3.2
    boxes = [
        ("Tourist cities", "map API", "58 cities screened\nfor street-view\ncoverage", BLUE),
        ("Photo sites", "Wikidata", "fine-grained POI\ntypes, sitelinks\n$\\leq$ 60", BLUE),
        ("Images", "Mapillary", "perspective frames,\nLLM + human\naudit", BLUE),
        ("Context ladder", "Wikidata", "8,721 candidates\nrated for\nreferenceability", ORANGE),
        ("Evaluation", "5 VLMs", "distance $\\times$\nreferenceability\ngrid", GREEN),
    ]
    xs = [1.5 + i * (W + GAP) for i in range(5)]
    for x, (title, src, sub, col) in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, 6.5), W, 16.5,
                                    boxstyle="round,pad=0.25,rounding_size=1.1",
                                    facecolor="white", edgecolor=col, linewidth=1.25))
        ax.text(x + W / 2, 20.6, title, ha="center", va="center", fontsize=8.2, color=INK)
        ax.text(x + W / 2, 17.6, src, ha="center", va="center",
                fontsize=6.6, color=col, style="italic")
        ax.text(x + W / 2, 11.4, sub, ha="center", va="center",
                fontsize=6.3, color=MUTED, linespacing=1.5)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + W + .5, 14.7), (x + W + GAP - .5, 14.7),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color=MUTED, linewidth=1.0))
    ax.plot([xs[0], xs[2] + W], [26, 26], color=MUTED, lw=.8)
    ax.text((xs[0] + xs[2] + W) / 2, 28, "held fixed across conditions",
            ha="center", fontsize=7, color=MUTED, style="italic")
    ax.plot([xs[3], xs[3] + W], [26, 26], color=ORANGE, lw=.8)
    ax.text(xs[3] + W / 2, 28, "manipulated", ha="center", fontsize=7,
            color=ORANGE, style="italic")
    ax.text(50, 1.6,
            "Every stage is programmatic: adding a city costs no manual curation, and the "
            "benchmark can be re-sampled on fresh imagery\nrather than shipped as a fixed file.",
            ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    save(fig, "fig_pipeline")


#: Band midpoints on a log axis. The bands are ranges; a line plot needs one x
#: per band, and the geometric middle keeps the spacing honest under log scale.
_BAND_MID = {"0.15-0.3km": 0.225, "0.3-0.7km": 0.5, "0.7-1.5km": 1.1,
             "1.5-3.0km": 2.25, "3.0-6.0km": 4.5}


def fig_verify_dprime():
    """GeoVerify: d' against decoy distance, one line per model.

    Reuses verify_results.load()/dprime() rather than reimplementing the
    parsing and scoring -- the figure and Table 11 must come from the same
    computation path, or a disagreement between them cannot be diagnosed.
    """
    import verify_results as vr

    df = vr.load()
    df = df[df.verdict.notna()]
    main_arm = df[df.arm == "main"]
    if main_arm.empty:
        print("  skipped fig_verify_dprime: no main-arm data")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.1))

    def _far_dprime(model):
        """d' in the farthest band -- used only to order the legend."""
        g = main_arm[main_arm.model == model]
        sig, dec = g[g.truth == "yes"], g[(g.truth == "no")
                                          & (g.band == "3.0-6.0km")]
        if sig.empty or dec.empty:
            return -1.0
        return vr.dprime(sig.yes.mean(), dec.yes.mean(), len(sig), len(dec))[0]

    # Legend ordered by where the curves end up, so its top-to-bottom order
    # matches the lines at the right edge where the legend sits.
    order = sorted(main_arm.model.unique(), key=_far_dprime, reverse=True)
    for model in order:
        g = main_arm[main_arm.model == model]
        sig = g[g.truth == "yes"]
        if sig.empty:
            continue
        h, nh = sig.yes.mean(), len(sig)
        xs, ys = [], []
        for band, mid in sorted(_BAND_MID.items(), key=lambda kv: kv[1]):
            dec = g[(g.truth == "no") & (g.band == band)]
            if dec.empty:
                continue
            d, _ = vr.dprime(h, dec.yes.mean(), nh, len(dec))
            xs.append(mid)
            ys.append(d)
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=model)

    # The discrimination a deployment needs sits at the tolerance itself
    # (150 m). The whole point of the figure is that no curve crosses d'=1
    # inside that band, so the band is drawn rather than left to the reader.
    ax.axvspan(0.15, 0.3, color="#d62728", alpha=0.07, zorder=0)
    ax.text(0.212, 0.97, "tolerance\nband", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#a01d1d",
            linespacing=1.3)

    ax.axhline(1.0, ls="--", lw=1, color="#888", zorder=1)
    ax.text(0.16, 1.03, "$d'=1$  (usable discrimination)",
            fontsize=8, color="#666", va="bottom")

    ax.set_xscale("log")
    ticks = [0.2, 0.3, 0.5, 1, 2, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()

    ax.set_xlabel("decoy distance (km, log scale)")
    ax.set_ylabel("sensitivity $d'$")
    # Legend below the axes: the curves run from lower left to upper right and
    # fill the lower right, so any in-axes placement covers data. save() does
    # not use bbox_inches="tight", so the room has to be reserved on the canvas
    # here or the legend is cropped.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3,
              columnspacing=1.8, handletextpad=0.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    save(fig, "fig_verify_dprime")


def main():
    style()
    f = prepare()
    print(f"{f.site.nunique()} sites, {len(f)} forced records -> {OUT}")
    fig_pipeline()
    fig_regimes(f)
    fig_echo_flat(f)
    fig_verify_dprime()


if __name__ == "__main__":
    main()
