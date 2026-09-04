"""Visualise the image-audit results so a human can check the model's judgement.

    python tools/make_audit_viz.py

Produces results/audit_viz.html -- every image alongside the model's
verdict, sorted by quality score, with filters (human verdict / AI verdict /
site / quality score).

Reads the single unified audit table, `data/image_audit.csv` -- one row per
image, AI and human verdicts as columns on the same row (batches 2+3 only;
batch 1's 11 images are tracked separately, see notes/paper_outline TODO).

## NOTE ON DATA COMPLETENESS

`human_verdict` currently only carries the **explicit exclusion list**: an
empty verdict means "not flagged for exclusion", not "reviewed and
positively approved". So the human-side filter below only distinguishes
**excluded** from **not excluded** -- it cannot yet show a third "never
reviewed" state, because the data does not carry that distinction until a full
manual pass gives every image an explicit verdict.

## Verdict fields (from scripts/03_audit_images.py's PROMPT), prefixed `ai_`

    ai_answer_overlay   text burned into the image naming the place / GPS --
                        **hard exclusion**, the model would just be reading text
    ai_scene_type       street / indoor / vehicle_interior / ...
    ai_through_glass    shot through glass
    ai_quality          0-5 overall usability
    ai_usable           the AI-only verdict
    usable              combined verdict (ai_usable, forced False if human excluded)
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config  # noqa: E402

OUT = config.RESULTS / "audit_viz.html"
AUDIT_FILE = config.DATA / "image_audit.csv"

TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>AI image audit results</title>
<style>
:root{--bd:#d8dce3;--mut:#6b7280;--ok:#16a34a;--warn:#d97706;--bad:#dc2626}
*{box-sizing:border-box}
body{font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;margin:0;
     padding:0 16px 40px;background:#fafbfc;color:#111}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--bd);
     padding:10px 0;margin-bottom:14px;z-index:10}
h1{font-size:16px;margin:0 0 6px}
.row1{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
select{font:12px inherit;padding:4px 7px;border:1px solid var(--bd);border-radius:5px}
.sum{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut);font-size:12px}
.sum b{color:#111;font-size:14px}
#cnt{margin-left:auto;color:var(--mut);font-variant-numeric:tabular-nums}
.grid{display:flex;gap:12px;flex-wrap:wrap}
figure{margin:0;width:264px;background:#fff;border:1px solid var(--bd);
       border-radius:6px;overflow:hidden}
figure.bad{border-color:var(--bad);border-width:2px}
figure.warn{border-color:var(--warn);border-width:2px}
img{width:100%;height:190px;object-fit:cover;display:block;cursor:zoom-in;background:#eee}
.miss{height:190px;display:flex;align-items:center;justify-content:center;
      background:#fef2f2;color:#b91c1c;font-size:11px;padding:8px;text-align:center;
      word-break:break-all}
.body{padding:8px 9px}
.top{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.q{font-weight:700;font-size:15px;width:26px;height:26px;border-radius:50%;
   color:#fff;text-align:center;line-height:26px;flex:0 0 auto}
.q5{background:#15803d}.q4{background:var(--ok)}.q3{background:var(--warn)}
.q2{background:#ea580c}.q1{background:var(--bad)}.q0{background:#7f1d1d}
.site{font-weight:600;font-size:12px}
.tags{display:flex;gap:4px;flex-wrap:wrap;margin:4px 0}
.tag{font-size:10.5px;padding:1px 6px;border-radius:3px;background:#eef0f3;color:#374151}
.tag.bad{background:#fee2e2;color:#991b1b;font-weight:600}
.tag.warn{background:#fef3c7;color:#92400e}
.tag.hum{background:#dbeafe;color:#1e40af}
.tag.no{background:#f3f4f6;color:#6b7280}
.why{color:var(--mut);font-size:11.5px;margin-top:3px}
.ov{font-family:ui-monospace,monospace;font-size:10.5px;background:#fee2e2;
    padding:3px 5px;border-radius:3px;color:#991b1b;word-break:break-all;margin-top:4px}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;
      align-items:center;justify-content:center;z-index:100;cursor:zoom-out}
.zoom img{max-width:96vw;max-height:96vh;width:auto;height:auto;object-fit:contain}
</style>
<div id="bar">
  <h1>AI image audit results · sorted by quality score ascending (worst first)</h1>
  <div class="row1">
    <select id="fh">
      <option value="">human verdict (all)</option>
      <option value="keep">not excluded</option>
      <option value="reject">excluded</option>
    </select>
    <select id="fa">
      <option value="">AI verdict (all)</option>
      <option value="ok">AI: usable</option>
      <option value="bad">AI: not usable</option>
      <option value="unaudited">AI: not yet audited</option>
      <option value="leak">answer leak</option>
    </select>
    <select id="fd">
      <option value="">agreement (all)</option>
      <option value="1">disagreement only</option>
      <option value="0">agreement only</option>
    </select>
    <select id="fs"><option value="">site (all)</option>__SITES__</select>
    <select id="fq"><option value="">quality score (all)</option>__QS__</select>
    <span id="cnt"></span>
  </div>
  <div class="sum">__SUMMARY__</div>
</div>
<div class="grid" id="g">__BODY__</div>
<div class="zoom" id="z" onclick="this.style.display='none'"><img id="zi"></div>
<script>
const F = ["fh","fa","fd","fs","fq"];
const all = [...document.querySelectorAll("#g figure")];
function render(){
  const v = Object.fromEntries(F.map(k => [k, document.getElementById(k).value]));
  let n = 0;
  for (const f of all){
    const d = f.dataset;
    const show = (!v.fh || d.hum === v.fh)
              && (!v.fa || d.ai === v.fa || (v.fa === "bad" && d.leak === "1"))
              && (!v.fd || d.dis === v.fd)
              && (!v.fs || d.site === v.fs)
              && (!v.fq || d.q === v.fq);
    f.style.display = show ? "" : "none";
    if (show) n++;
  }
  document.getElementById("cnt").textContent = `showing ${n} / ${all.length}`;
}
F.forEach(k => document.getElementById(k).onchange = render);
document.querySelectorAll("#g img").forEach(i => i.onclick = () => {
  document.getElementById("zi").src = i.src;
  document.getElementById("z").style.display = "flex";
});
render();
</script>
"""


def load_audit() -> pd.DataFrame:
    """Read the unified audit table."""
    if not AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"{AUDIT_FILE} not found -- run scripts/03_audit_images.py first")
    return pd.read_csv(AUDIT_FILE)


def img_src(image_rel: str) -> str:
    """`image` is already a path relative to data/ (e.g.
    streetview/paris_marais/12345.jpg); the viz lives under results/, so the
    src just needs one `../data/` prefix."""
    return "../data/" + str(image_rel).replace("\\", "/")


def main():
    d = load_audit()
    d["quality"] = pd.to_numeric(d.ai_quality, errors="coerce")
    d["leak"] = d.ai_answer_overlay == True                    # noqa: E712
    d["usable_b"] = d.ai_usable == True                        # noqa: E712
    d["src"] = d.image.map(img_src)
    d["exists"] = [(config.DATA / str(im)).exists() for im in d.image]
    # NOTE: human_verdict currently only records exclusions -- an empty
    # verdict means "not flagged", not "reviewed and approved". See the module
    # docstring.
    d["hum"] = d.human_verdict.map(lambda v: "reject" if v == "exclude" else "keep")
    d["unaudited"] = d.ai_usable.isna() & d.ai_answer_overlay.isna()

    disagree = 0
    figs = []
    for r in d.sort_values(["unaudited", "quality", "site"], na_position="first").itertuples():
        ai = ("unaudited" if r.unaudited
              else "ok" if (r.usable_b and not r.leak) else "bad")
        dis = "" if r.unaudited else str(int((r.hum == "keep") != (ai == "ok")))
        if dis == "1":
            disagree += 1
        q = r.quality
        qs = f"q{int(q)}" if pd.notna(q) else "q0"
        cls = "bad" if r.leak else ("warn" if not r.usable_b else "")

        tags = []
        if r.unaudited:
            tags.append('<span class="tag no">not yet AI-audited</span>')
        if r.leak:
            tags.append('<span class="tag bad">answer leak</span>')
        if not r.unaudited and not r.usable_b:
            tags.append('<span class="tag warn">AI: not usable</span>')
        if not r.unaudited and str(r.ai_scene_type) != "street":
            tags.append(f'<span class="tag warn">{r.ai_scene_type}</span>')
        if not r.unaudited and r.ai_through_glass == True:       # noqa: E712
            tags.append('<span class="tag warn">through glass</span>')
        tags.append('<span class="tag hum">not excluded</span>' if r.hum == "keep"
                    else '<span class="tag warn">human-excluded</span>')
        if dis == "1":
            tags.append('<span class="tag bad">human/AI disagree</span>')

        img = (f'<img loading="lazy" src="{r.src}">' if r.exists and r.src
               else f'<div class="miss">image not found<br>{r.image}</div>')
        ov = (f'<div class="ov">{r.ai_overlay_text}</div>'
              if r.leak and isinstance(r.ai_overlay_text, str) and r.ai_overlay_text.strip()
              else "")
        figs.append(
            f'<figure class="{cls}" data-hum="{r.hum}" data-ai="{ai}" data-dis="{dis}"'
            f' data-site="{r.site}" data-q="{"" if pd.isna(q) else int(q)}"'
            f' data-leak="{int(bool(r.leak))}">{img}'
            f'<div class="body"><div class="top">'
            f'<span class="q {qs}">{"" if pd.isna(q) else int(q)}</span>'
            f'<span class="site">{r.site}</span></div>'
            f'<div class="tags">{"".join(tags)}</div>'
            f'<div class="why">{r.ai_why}</div>{ov}</div></figure>')

    n = len(d)
    n_aud = int((~d.unaudited).sum())
    n_excluded = int((d.hum == "reject").sum())
    summary = (
        f'<span><b>{n}</b> images total (<b>{n_aud}</b> audited)</span>'
        f'<span>AI usable <b>{int((d.usable_b & ~d.leak).sum())}</b></span>'
        f'<span>answer leak <b style="color:#dc2626">{int(d.leak.sum())}</b></span>'
        f'<span>image missing <b>{int((~d.exists).sum())}</b></span>'
        f'<span>human-excluded <b>{n_excluded}</b></span>'
        f'<span>human/AI disagreements <b>{disagree}</b></span>')

    OUT.write_text(
        TEMPLATE.replace("__BODY__", "\n".join(figs))
                .replace("__SUMMARY__", summary)
                .replace("__SITES__", "".join(
                    f"<option>{x}</option>" for x in sorted(d.site.dropna().unique())))
                .replace("__QS__", "".join(
                    f"<option>{int(x)}</option>"
                    for x in sorted(d.quality.dropna().unique()))),
        encoding="utf-8")

    print(f"{n} images ({n_aud} audited) -> {OUT}")
    print(f"  loadable {int(d.exists.sum())}, missing {int((~d.exists).sum())}")
    print(f"  human-excluded {n_excluded}, human/AI disagreements {disagree}")
    print("\n=== quality score ===")
    print(d.quality.value_counts().sort_index().to_string())
    print("\n=== scene type ===")
    print(d.ai_scene_type.value_counts().to_string())


if __name__ == "__main__":
    main()
