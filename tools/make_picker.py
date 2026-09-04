"""Build an image picker: show every downloaded street-view image, let the user tick keepers.

    python tools/make_picker.py                    # candidate sites (the 16 new ones)
    python tools/make_picker.py --all              # show existing sites too

Produces results/picker.html -- double-click to open, tick the images to keep,
click "Export selection" to download a JSON, then run
`python scripts/04_select_images.py <downloaded json>` to build the final sample table.

Why a manual pick is needed: Mapillary is crowdsourced dashcam/handheld data,
and any one site mixes in shots half-blocked by a car bonnet, blurred,
backlit, facing a wall, inside a tunnel, and so on. Automatic brightness
filtering does not catch these. And since **each site only uses 3 images**, one
bad shot already costs a third of the sample.

Selections are stored in localStorage, so closing the page does not lose them.
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geocontext.sites import CANDIDATES  # noqa: E402

OUT = config.RESULTS / "picker.html"
#: Existing sites (not re-picked this round unless --all).
EXISTING = {"nyc_soho", "paris_marais", "tokyo_shibuya",
            }

TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>Street view image picker</title>
<style>
:root{--bd:#d8dce3;--mut:#6b7280;--ok:#16a34a}
*{box-sizing:border-box}
body{font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;margin:0;padding:0 16px 40px;
     background:#fafbfc;color:#111}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--bd);
     padding:10px 0;margin-bottom:14px;z-index:10;display:flex;gap:10px;align-items:center;
     flex-wrap:wrap}
h1{font-size:16px;margin:0}
button{font:12px inherit;padding:5px 11px;border:1px solid var(--bd);border-radius:5px;
       background:#fff;cursor:pointer}
button:hover{background:#f3f4f6}
button.primary{background:#111;color:#fff;border-color:#111}
#count{color:var(--mut);margin-left:auto;font-variant-numeric:tabular-nums}
.site{background:#fff;border:1px solid var(--bd);border-radius:6px;padding:12px;margin-bottom:14px}
.site h2{font-size:14px;margin:0 0 2px}
.meta{color:var(--mut);font-size:12px;margin-bottom:8px}
.meta b{color:#111}
.row{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0;position:relative;width:250px}
img{width:250px;height:188px;object-fit:cover;border:3px solid transparent;border-radius:5px;
    display:block;cursor:pointer;background:#eee}
figure.on img{border-color:var(--ok)}
figure.on::after{content:"✓";position:absolute;top:6px;right:6px;background:var(--ok);color:#fff;
     width:22px;height:22px;border-radius:50%;text-align:center;line-height:22px;font-size:14px}
figcaption{font-size:11px;color:var(--mut);margin-top:3px;word-break:break-all}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;
      justify-content:center;z-index:100;cursor:zoom-out}
.zoom img{max-width:94vw;max-height:94vh;width:auto;height:auto;object-fit:contain;border:0}
</style>

<div id="bar">
  <h1>Street view image picker</h1>
  <button onclick="setAll(true)">Select all</button>
  <button onclick="setAll(false)">Deselect all</button>
  <button onclick="localStorage.removeItem(KEY);location.reload()">Reset</button>
  <button class="primary" onclick="exportSel()">Export selection</button>
  <span id="count"></span>
</div>
<div class="meta" style="margin:-6px 0 12px">
  Click an image to toggle it (green border = keep). Use a site's "select
  all/none" buttons to act on the whole site.
  <b>Shift+click</b> to zoom into the full image. Selections are saved to the
  browser automatically.
</div>
__BODY__
<div class="zoom" id="zoom" onclick="this.style.display='none'"><img id="zoomimg"></div>

<script>
const KEY = "geocontext_picker_v1";
let sel = JSON.parse(localStorage.getItem(KEY) || "null");
const all = [...document.querySelectorAll("figure")];
if (!sel) sel = Object.fromEntries(all.map(f => [f.dataset.p, true]));   // select all by default

function paint(){
  all.forEach(f => f.classList.toggle("on", !!sel[f.dataset.p]));
  document.querySelectorAll(".site").forEach(s => {
    const fs = [...s.querySelectorAll("figure")];
    s.querySelector(".n").textContent =
      fs.filter(f => sel[f.dataset.p]).length + " / " + fs.length;
  });
  const n = all.filter(f => sel[f.dataset.p]).length;
  document.getElementById("count").textContent = `${n} / ${all.length} selected`;
  localStorage.setItem(KEY, JSON.stringify(sel));
}
all.forEach(f => f.querySelector("img").onclick = e => {
  if (e.shiftKey){
    document.getElementById("zoomimg").src = f.querySelector("img").src;
    document.getElementById("zoom").style.display = "flex";
    return;
  }
  sel[f.dataset.p] = !sel[f.dataset.p];
  paint();
});
function setAll(v){ all.forEach(f => sel[f.dataset.p] = v); paint(); }
function setSite(btn, v){
  btn.closest(".site").querySelectorAll("figure").forEach(f => sel[f.dataset.p] = v);
  paint();
}
function exportSel(){
  const keep = all.filter(f => sel[f.dataset.p]).map(f => f.dataset.p);
  const blob = new Blob([JSON.stringify({keep: keep}, null, 1)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "image_selection.json";
  a.click();
}
paint();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="show existing sites too")
    args = ap.parse_args()

    root = config.DATA / "streetview"
    sites = sorted(d for d in root.iterdir() if d.is_dir())
    if not args.all:
        sites = [d for d in sites if d.name not in EXISTING]

    # which images are already in a sample table (shown as a reference marker)
    sampled = set()
    for f in ("streetview_meta.csv", "streetview_meta_candidates.csv"):
        p = config.DATA / f
        if p.exists():
            sampled |= set(pd.read_csv(p).path.astype(str).str.replace("\\", "/"))

    body, total = [], 0
    for d in sites:
        imgs = sorted(d.glob("*.jpg"))
        if not imgs:
            continue
        total += len(imgs)
        note = CANDIDATES.get(d.name, (0, 0, "existing site"))[2]
        figs = []
        for p in imgs:
            rel = f"streetview/{d.name}/{p.name}"
            star = " *sampled" if rel in sampled else ""
            figs.append(
                f'<figure data-p="{rel}">'
                f'<img loading="lazy" src="../data/{rel}">'
                f'<figcaption>{p.name[:30]}{star}</figcaption></figure>')
        body.append(
            f'<div class="site"><h2>{d.name}</h2>'
            f'<div class="meta">{note} &nbsp;·&nbsp; {len(imgs)} images &nbsp;·&nbsp; '
            f'selected <span class="n"></span> &nbsp; '
            f'<button onclick="setSite(this,true)">select all</button> '
            f'<button onclick="setSite(this,false)">select none</button></div>'
            f'<div class="row">{"".join(figs)}</div></div>')

    OUT.write_text(TEMPLATE.replace("__BODY__", "\n".join(body)), encoding="utf-8")
    print(f"{len(body)} sites, {total} images total")
    print(f"written to {OUT}")
    print("double-click to open -> untick unusable images -> click 'Export selection' -> download image_selection.json")


if __name__ == "__main__":
    main()
