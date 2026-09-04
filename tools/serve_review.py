"""Human image review tool: a small local server that reads and writes
the unified audit table, data/image_audit.csv, directly.

    python tools/serve_review.py
    python tools/serve_review.py --port 8765 --no-browser

Opens a review page in the browser: filter by city/status, mark each image
keep/exclude, and **every click writes straight into the `human_*` columns of
data/image_audit.csv** -- no export step, no merge step, nothing lost when the
page is closed. The `ai_*` columns already in that file are read-only here and
are never touched by a save (see "Why the AI audit result is not shown"
below).

## Why a local server rather than a static HTML file

A static page opened over file:// can neither read the CSV (fetch on file://
is blocked by CORS) nor write one (it can only trigger a download). That
forces a three-step "mark in localStorage -> export a delta CSV -> run a
merge script" flow, with the working state living in one browser profile.
A stdlib http.server instead gives:

    open -> read the current CSV (not a browser cache)
    click -> write the CSV back (atomic replace)

so there is exactly one source of truth: the file on disk.

## Why the AI audit result is not shown

An earlier version displayed the AI auditor's scene/quality/overlay call
under each image "for reference". That is wrong: if the reviewer sees the
model's judgement before making their own, the two audits are no longer
independent, and the agreement statistic (Cohen's kappa) computed in
analysis/human_vs_ai_audit.py stops meaning anything. Shipping both audits
so a reader can compute agreement themselves only works if each was made
independently. So this page shows the image and nothing else.

Standard library only (http.server / csv / webbrowser) -- no Flask.
"""
import re
import sys
import csv
import json
import argparse
import threading
import webbrowser
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geocontext import config  # noqa: E402

AUDIT_FILE = config.DATA / "image_audit.csv"
SITES_CSV = config.DATA / "sites.csv"
STREETVIEW = config.DATA / "streetview"
# Every column of the unified table. This tool only ever writes the human_*
# ones -- ai_* columns are carried through unchanged so a save here can never
# clobber the model's independent judgement.
COLS = ["image", "site", "batch", "usable",
        "ai_answer_overlay", "ai_overlay_text", "ai_scene_type", "ai_through_glass",
        "ai_tourist_infrastructure", "ai_quality", "ai_usable", "ai_why",
        "human_verdict", "human_reason", "human_verified_by", "human_note"]

_lock = threading.Lock()


def _load_landmarks() -> dict:
    """site -> landmark, for display only. Missing file (or a site not in
    it, e.g. batch1's hand-picked sites) just means no landmark caption."""
    if not SITES_CSV.exists():
        return {}
    with SITES_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {r["site"]: r.get("landmark", "") for r in csv.DictReader(f)}


LANDMARKS = _load_landmarks()


# ----------------------------------------------------------------- csv i/o
def load_rows() -> dict:
    """image -> {column: value}. Missing file yields an empty table; the disk
    scan below then supplies every image with a blank verdict."""
    if not AUDIT_FILE.exists():
        return {}
    with AUDIT_FILE.open(encoding="utf-8-sig", newline="") as f:
        return {r["image"]: {c: (r.get(c) or "") for c in COLS}
                for r in csv.DictReader(f)}


def save_rows(rows: dict) -> None:
    """Write to a temp file and replace. Overwriting in place would destroy
    the whole table if interrupted mid-write, and this table is pure manual
    labour -- far more expensive to redo than an extra temp file."""
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUDIT_FILE.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for image in sorted(rows):
            w.writerow({c: rows[image].get(c, "") for c in COLS})
    tmp.replace(AUDIT_FILE)


def scan_images(rows: dict) -> list:
    """The unified table's candidate images (NOT a disk scan).

    2026-09-03: this used to walk data/streetview/ directly, which shows
    every jpg physically on disk -- including raw download candidates that
    never made it into any streetview_meta_*.csv (a site can fetch 12, keep 3;
    the other 9 stay on disk). That inflated one review session from the
    real 429 candidates to 1033 images, most of them not real candidates at
    all. The unified table (data/image_audit.csv) is the actual candidate
    list, built from the streetview_meta_*.csv files -- iterate that instead,
    and only fall back to "file missing" if something is out of sync.
    """
    items = []
    for image, r in sorted(rows.items()):
        site = r.get("site", "")
        items.append(dict(image=image, site=site, city=site.split("_")[0],
                          landmark=LANDMARKS.get(site, ""),
                          exists=(config.DATA / image).exists(),
                          verdict=r.get("human_verdict", ""),
                          reason=r.get("human_reason", ""),
                          verified_by=r.get("human_verified_by", "")))
    return items


PAGE = r"""<!doctype html><meta charset="utf-8">
<title>Image review</title>
<style>
:root{--bd:#d8dce3;--mut:#6b7280;--ok:#16a34a;--bad:#dc2626}
*{box-sizing:border-box}
body{font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;margin:0;padding:0 16px 60px;
     background:#fafbfc;color:#111}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--bd);
     padding:10px 0;margin-bottom:14px;z-index:10;display:flex;gap:10px;align-items:center;
     flex-wrap:wrap}
h1{font-size:16px;margin:0}
select,input[type=text]{font:12px inherit;padding:5px 8px;border:1px solid var(--bd);
     border-radius:5px;background:#fff}
button{font:12px inherit;padding:5px 11px;border:1px solid var(--bd);border-radius:5px;
       background:#fff;cursor:pointer}
button:hover{background:#f3f4f6}
#count{color:var(--mut);margin-left:auto;font-variant-numeric:tabular-nums}
#saved{color:var(--ok);font-size:12px;min-width:70px}
#saved.err{color:var(--bad)}
.site{background:#fff;border:1px solid var(--bd);border-radius:6px;padding:12px;margin-bottom:14px}
.site h2{font-size:14px;margin:0 0 2px}
.meta{color:var(--mut);font-size:12px;margin-bottom:8px}
.row{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0;position:relative;width:230px;border:1px solid var(--bd);border-radius:6px;
       padding:6px;background:#fcfcfd}
figure.keep{border-color:var(--ok);background:#f0fdf4}
figure.exclude{border-color:var(--bad);background:#fef2f2}
img{width:100%;height:172px;object-fit:cover;border-radius:4px;display:block;cursor:zoom-in;
    background:#eee}
.miss{width:100%;height:172px;display:flex;align-items:center;justify-content:center;
      background:#fef2f2;color:#b91c1c;font-size:11px;padding:8px;text-align:center;
      word-break:break-all;border-radius:4px}
figcaption{font-size:11px;color:var(--mut);margin-top:5px;word-break:break-all}
.btns{display:flex;gap:5px;margin-top:6px}
.btns button{flex:1;padding:4px}
.btns button.on-keep{background:var(--ok);color:#fff;border-color:var(--ok)}
.btns button.on-exclude{background:var(--bad);color:#fff;border-color:var(--bad)}
.reason{width:100%;margin-top:5px;font-size:11px;padding:3px 5px}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;
      justify-content:center;z-index:100;cursor:zoom-out}
.zoom img{max-width:94vw;max-height:94vh;width:auto;height:auto;object-fit:contain;border:0}
#more{display:block;margin:20px auto;padding:10px 30px}
</style>

<div id="bar">
  <h1>Image review</h1>
  <select id="fCity"></select>
  <select id="fStatus">
    <option value="unreviewed">Unreviewed</option>
    <option value="all">All</option>
    <option value="keep">Kept</option>
    <option value="exclude">Excluded</option>
  </select>
  <select id="fPageSize">
    <option value="20">20 / page</option>
    <option value="40" selected>40 / page</option>
    <option value="100">100 / page</option>
  </select>
  <input id="verifiedBy" type="text" placeholder="Reviewer (goes into verified_by)">
  <button onclick="reload()">Reload</button>
  <span id="saved"></span>
  <span id="count"></span>
</div>
<div class="meta" style="margin:-6px 0 12px">
  Click an image to zoom. Each keep/exclude click is <b>written to
  human_audit.csv immediately</b> -- no export, no manual save. The reviewer
  name is stored alongside every mark made after it is filled in.
</div>
<div id="body"></div>
<button id="more" onclick="renderMore()">Load more</button>
<div class="zoom" id="zoom" onclick="this.style.display='none'"><img id="zoomimg"></div>

<script>
let DATA = [];
let shown = 0;

function flash(msg, isErr){
  const el = document.getElementById("saved");
  el.textContent = msg;
  el.className = isErr ? "err" : "";
  if (!isErr) setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 1500);
}

async function reload(){
  const r = await fetch("/api/images");
  DATA = await r.json();
  populateCityFilter();
  render();
}

function populateCityFilter(){
  const sel = document.getElementById("fCity");
  const cur = sel.value;
  const cities = [...new Set(DATA.map(d => d.city))].sort();
  sel.innerHTML = '<option value="">All cities</option>' +
    cities.map(c => `<option value="${c}">${c}</option>`).join("");
  if (cur) sel.value = cur;
}

function filtered(){
  const city = document.getElementById("fCity").value;
  const status = document.getElementById("fStatus").value;
  return DATA.filter(d => {
    if (city && d.city !== city) return false;
    if (status === "unreviewed") return !d.verdict;
    if (status === "keep") return d.verdict === "keep";
    if (status === "exclude") return d.verdict === "exclude";
    return true;
  });
}

function render(){
  shown = 0;
  document.getElementById("body").innerHTML = "";
  renderMore();
}

function renderMore(){
  const pageSize = parseInt(document.getElementById("fPageSize").value);
  const items = filtered();
  // Extend the page boundary so it never splits one site's images across
  // two pages -- items of the same site are contiguous (scan_images sorts
  // by image path, which shares the "streetview/<site>/" prefix), so if the
  // raw cut point lands mid-site, push it forward to that site's end.
  let end = Math.min(shown + pageSize, items.length);
  while (end < items.length && items[end].site === items[end - 1].site) end++;
  const batch = items.slice(shown, end);
  shown = end;

  const bySite = {};
  batch.forEach(d => { (bySite[d.site] = bySite[d.site] || []).push(d); });

  const frag = document.createDocumentFragment();
  Object.entries(bySite).forEach(([site, imgs]) => {
    const div = document.createElement("div");
    div.className = "site";
    div.innerHTML = `<h2>${site}</h2><div class="meta">${imgs.length} on this page</div>
      <div class="row">${imgs.map(figFor).join("")}</div>`;
    frag.appendChild(div);
  });
  document.getElementById("body").appendChild(frag);
  wireUp();
  paintCount();
  document.getElementById("more").style.display =
    shown >= items.length ? "none" : "block";
}

function esc(s){ return String(s == null ? "" : s).replace(/"/g, "&quot;"); }

function figFor(d){
  const img = d.exists
    ? `<img loading="lazy" src="/img/${d.image}">`
    : `<div class="miss">image missing on disk<br>${esc(d.image)}</div>`;
  return `<figure data-img="${esc(d.image)}" class="${d.verdict}">
    ${img}
    <figcaption>${d.landmark ? esc(d.landmark) + " &middot; " : ""}${d.image.split("/").pop()}</figcaption>
    <div class="btns">
      <button data-v="keep" class="${d.verdict==='keep'?'on-keep':''}">Keep</button>
      <button data-v="exclude" class="${d.verdict==='exclude'?'on-exclude':''}">Exclude</button>
    </div>
    <input class="reason" type="text" placeholder="Reason for excluding (optional)" value="${esc(d.reason)}">
  </figure>`;
}

async function save(item){
  const body = {image: item.image, site: item.site, verdict: item.verdict,
                reason: item.reason, verified_by: document.getElementById("verifiedBy").value || ""};
  try {
    const r = await fetch("/api/mark", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (!r.ok) throw new Error(await r.text());
    flash("saved");
  } catch (e) {
    flash("save failed!", true);
    console.error(e);
  }
}

function wireUp(){
  document.querySelectorAll("figure").forEach(f => {
    const image = f.dataset.img;
    const item = DATA.find(x => x.image === image);
    const imgEl = f.querySelector("img");
    if (imgEl) imgEl.onclick = () => {
      document.getElementById("zoomimg").src = imgEl.src;
      document.getElementById("zoom").style.display = "flex";
    };
    f.querySelectorAll(".btns button").forEach(b => {
      b.onclick = () => {
        item.verdict = item.verdict === b.dataset.v ? "" : b.dataset.v;
        item.reason = f.querySelector(".reason").value;
        f.className = item.verdict;
        f.querySelectorAll(".btns button").forEach(x =>
          x.className = x.dataset.v === item.verdict ? "on-" + item.verdict : "");
        paintCount();
        save(item);
      };
    });
    f.querySelector(".reason").onchange = e => {
      item.reason = e.target.value;
      save(item);
    };
  });
}

function paintCount(){
  const total = DATA.length;
  const reviewed = DATA.filter(d => d.verdict).length;
  const kept = DATA.filter(d => d.verdict === "keep").length;
  const excluded = DATA.filter(d => d.verdict === "exclude").length;
  document.getElementById("count").textContent =
    `${reviewed}/${total} reviewed (kept ${kept} · excluded ${excluded})`;
}

document.getElementById("fCity").onchange = render;
document.getElementById("fStatus").onchange = render;
document.getElementById("fPageSize").onchange = render;
reload();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/images":
            with _lock:
                items = scan_images(load_rows())
            return self._send(200, json.dumps(items, ensure_ascii=False).encode("utf-8"))
        if self.path.startswith("/img/"):
            rel = self.path[len("/img/"):].split("?")[0]
            # Only jpgs under data/streetview. The server binds to 127.0.0.1
            # only, but path joining still needs a guard against ../ escapes.
            if not re.fullmatch(r"streetview/[\w.-]+/[\w.-]+\.jpg", rel):
                return self._send(400, b"bad path", "text/plain")
            p = config.DATA / rel
            if not p.exists():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, p.read_bytes(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/mark":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        image = body.get("image", "")
        if not image:
            return self._send(400, b'{"error":"no image"}')
        with _lock:
            rows = load_rows()
            row = rows.get(image, {c: "" for c in COLS})
            row["image"] = image
            row["site"] = body.get("site", "") or row.get("site", "")
            row["human_verdict"] = body.get("verdict", "")
            row["human_reason"] = body.get("reason", "")
            row["human_verified_by"] = (body.get("verified_by", "")
                                        or row.get("human_verified_by", ""))
            rows[image] = row
            save_rows(rows)
            n_done = sum(1 for r in rows.values() if r.get("human_verdict"))
        self._send(200, json.dumps({"ok": True, "reviewed": n_done}).encode("utf-8"))

    def log_message(self, *a):     # don't echo every image GET to the terminal
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    items = scan_images(rows)
    if not items:
        print(f"{AUDIT_FILE} is empty or missing -- run scripts/03_audit_images.py "
              "first to build the candidate list")
        return
    n_missing = sum(1 for i in items if not i["exists"])
    if n_missing:
        print(f"warning: {n_missing} candidate images are missing on disk "
              f"under {STREETVIEW} -- run scripts/02_fetch_images.py")
    n_done = sum(1 for i in items if i["verdict"])
    print(f"{len(items)} images, {len({i['site'] for i in items})} sites, "
          f"{len({i['city'] for i in items})} cities; {n_done} already reviewed")
    print(f"verdicts read from and written to {AUDIT_FILE}")

    url = f"http://127.0.0.1:{args.port}/"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nopen {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
