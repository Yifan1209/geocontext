"""Download images for the candidate expansion sites, then build a contact sheet.

    python scripts/02_fetch_images.py --n 3

Download images before deciding which sites to keep -- coverage counts alone
(script 01) can only tell you "there are images", not "what they look like".
The pilot site taught this: all 141 images existed, but reprojected they
were unrecognisable.

Produces:
  data/streetview/<site>/*.jpg      the images
  results/candidates.html           a contact sheet (thumbnail wall), double-click to open

NOTE: this only downloads images, it does not run any model.
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, streetview  # noqa: E402
from geocontext.sites import CANDIDATES  # noqa: E402

OUT = config.RESULTS / "candidates.html"

TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>Candidate site street view</title>
<style>
body{font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;margin:0;padding:16px;
     background:#fafbfc;color:#111}
h1{font-size:17px;margin:0 0 4px}
.sub{color:#6b7280;margin-bottom:16px}
.site{background:#fff;border:1px solid #d8dce3;border-radius:6px;padding:12px;margin-bottom:14px}
.site h2{font-size:14px;margin:0 0 2px}
.meta{color:#6b7280;font-size:12px;margin-bottom:8px}
.row{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0}
img{height:230px;border:1px solid #d8dce3;border-radius:4px;display:block;cursor:zoom-in}
img.big{height:auto;max-width:96vw;cursor:zoom-out}
figcaption{font-size:11px;color:#6b7280;margin-top:3px}
</style>
<h1>Candidate site street view (__N__ per site, randomly sampled)</h1>
<div class="sub">Click an image to zoom. Check: can a person tell where this is?
Any printed place names / GPS overlay? Image quality?</div>
__BODY__
<script>
document.querySelectorAll("img").forEach(i=>i.onclick=()=>i.classList.toggle("big"));
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="images to download per site")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--radius", type=int, default=300)
    ap.add_argument("--min-brightness", type=float, default=90,
                    help="brightness floor, filters out night/tunnel shots")
    args = ap.parse_args()

    sites = {k: (v[0], v[1]) for k, v in CANDIDATES.items()
             if not args.only or k in args.only}
    print(f"downloading {len(sites)} sites x {args.n} images\n", flush=True)

    meta = streetview.collect(sites, n=args.n, radius_m=args.radius,
                              min_brightness=args.min_brightness, verbose=True)
    out_csv = config.DATA / "streetview_meta_candidates.csv"
    meta.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nmetadata -> {out_csv}")

    # contact sheet
    body = []
    for site, g in meta.groupby("site"):
        note = CANDIDATES.get(site, (0, 0, ""))[2]
        imgs = "".join(
            f'<figure><img src="../data/{r.path.replace(chr(92), "/")}" loading="lazy">'
            f'<figcaption>{Path(str(r.path)).name[:26]}</figcaption></figure>'
            for r in g.itertuples())
        body.append(
            f'<div class="site"><h2>{site}</h2>'
            f'<div class="meta">{note} &nbsp;·&nbsp; {len(g)} images</div>'
            f'<div class="row">{imgs}</div></div>')
    OUT.write_text(
        TEMPLATE.replace("__BODY__", "\n".join(body)).replace("__N__", str(args.n)),
        encoding="utf-8")
    print(f"contact sheet -> {OUT} (double-click to open, click an image to zoom)")
    print("\nimages per site:")
    print(meta.groupby("site").size().to_string())


if __name__ == "__main__":
    main()
