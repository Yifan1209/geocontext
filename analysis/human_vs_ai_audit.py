"""Agreement between the human image review and the model audit.

    python analysis/human_vs_ai_audit.py

The human review (tools/serve_review.py) checked all candidate images and
recorded explicit exclusions; the model independently audited the same
images. Both live as columns on the same row of the unified table,
data/image_audit.csv (`human_*` / `ai_*` prefixes). Both are judgements of
"is this image usable", but **the criteria are not identical**:

    the human looks at "does the frame carry information usable for
    localisation" (a blank wall, two rubbish bins -> excluded)
    the model looks at four fields: answer_overlay / scene_type / quality / usable

So a disagreement does not mean one side is wrong -- the disagreement itself is
the thing worth reporting. Every one is listed for manual inspection.

Reported as a confusion matrix plus Cohen's kappa, and **the two kinds of
disagreement are kept separate** -- "human excluded it, AI kept it" and "human
kept it, AI excluded it" mean very different things.

NOTE ON DATA COMPLETENESS: `human_verdict` currently only carries the
**explicit exclusion list**; an empty verdict means "not flagged for
exclusion", not "positively reviewed and approved". Until every image has been
given an explicit keep/exclude verdict, `hum_ok` below measures "was not
excluded", which is a real judgement for the images that ARE in the exclusion
list, but is not yet a fully independent second opinion for every image.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 58)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    p1, q1 = sum(a) / n, sum(b) / n
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    d = pd.read_csv(config.DATA / "image_audit.csv")

    d["ai_ok"] = (d.ai_usable == True) & (d.ai_answer_overlay != True)  # noqa: E712
    d["hum_ok"] = d.human_verdict != "exclude"

    n = len(d)
    print(f"{n} images, {int(d.hum_ok.sum())} kept by the human review, "
          f"{int(d.ai_ok.sum())} judged usable by the AI\n")

    print("=== confusion matrix ===")
    ct = pd.crosstab(d.hum_ok.map({True: "human_kept", False: "human_excluded"}),
                     d.ai_ok.map({True: "ai_usable", False: "ai_unusable"}),
                     margins=True)
    print(ct.to_string())
    agree = int((d.hum_ok == d.ai_ok).sum())
    print(f"\nagreement {agree}/{n} = {agree/n:.1%}"
          f"   Cohen's kappa = {kappa(list(d.hum_ok), list(d.ai_ok)):.3f}")

    print("\n\n=== Disagreement A: human excluded it, AI judged it usable ===")
    print("(the human criterion 'no locatable information in the frame' is not "
          "among the model's four criteria -- this is a design difference)")
    a = d[(~d.hum_ok) & d.ai_ok]
    print(f"{len(a)} images")
    print(a[["site", "image", "ai_scene_type", "ai_quality", "ai_why"]].to_string(index=False))

    print("\n\n=== Disagreement B: human kept it, AI judged it unusable ===")
    print("(more worth looking at: the model may have caught something the human missed)")
    b = d[d.hum_ok & (~d.ai_ok)]
    print(f"{len(b)} images")
    print(b[["site", "image", "ai_answer_overlay", "ai_scene_type", "ai_quality",
             "ai_why"]].to_string(index=False))

    print("\n\n=== Answer leakage: the one category only the model reliably catches ===")
    lk = d[d.ai_answer_overlay == True]                                # noqa: E712
    print(f"{len(lk)} images, {int(lk.hum_ok.sum())} of which the human review had kept")
    for r in lk.itertuples():
        mark = "human also excluded it" if not r.hum_ok else "**human missed it**"
        print(f"  [{mark}] {r.site}  {r.ai_overlay_text}")

    print("\n\n=== quality score vs human verdict ===")
    print("(if the model's quality score correlates with the human judgement, "
          "it is capturing the same dimension)")
    q = pd.to_numeric(d.ai_quality, errors="coerce")
    print(pd.crosstab(q, d.hum_ok.map({True: "human_kept", False: "human_excluded"}),
                      margins=True).to_string())


if __name__ == "__main__":
    main()
