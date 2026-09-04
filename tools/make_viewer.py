"""Build a self-contained HTML browser for the raw evaluation records.

    python tools/make_viewer.py                 # every street-view site
    python tools/make_viewer.py --site nyc_soho
    python tools/make_viewer.py --include-mmsvpr

Produces results/viewer.html -- double-click to open, no install, no server.
Images are referenced by relative path, so the HTML must stay under results/
(it looks for images relative to ../data/).

What it does: filter by site/model/language/schema/context, full-text search,
click a column header to sort, click a row to expand and see the **full
prompt**, **raw response**, structured fields, and the image.

Why not Streamlit / Gradio: those need a package install and a running server,
with ports and firewalls that can go wrong, for something whose entire purpose
is casually browsing data. A self-contained HTML has zero dependencies and can
be handed to someone else directly.
"""
import sys
import json
import re
import html
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from geocontext import config, geocode, prompts, runner  # noqa: E402

OUT = config.RESULTS / "viewer.html"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)
_JSON = re.compile(r"\{.*\}", re.S)

#: Every shipped ladder carries the same auditor tag, so the per-site
#: override table that used to live here is gone.
LADDER_AUDITOR_TAG = "deepseek-en"


def parse(raw):
    if not raw:
        return {}
    s = _FENCE.sub("", str(raw).strip())
    m = _JSON.search(s)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def build_rows(site=None, include_mmsvpr=False):
    # ladder metadata: reference-point name, distance, referenceability
    ladders = {}
    for s, aud in LADDER_AUDITOR.items():
        try:
            ladders[s] = prompts.load_ladder(s, aud, include_baseline=True)
        except FileNotFoundError:
            pass

    # error (if already computed)
    err = {}
    pkl = config.RESULTS / "answers_geocoded.pkl"
    if pkl.exists():
        g = pd.read_pickle(pkl)
        for r in g.itertuples():
            err[(r.path, r.model, r.lang, r.context, r.schema)] = (
                r.error_km, r.resolved_level, r.resolved_label)

    rows = []
    for line in runner.RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p = str(rec.get("path", ""))
        st = geocode.site_of(p)
        if st is None:
            if not include_mmsvpr:
                continue
            st = "mmsvpr:" + str(rec.get("location", "?"))
        if site and st != site:
            continue

        sch = rec.get("schema", "v1")
        ctx = rec.get("context", "none")
        d = parse(rec.get("raw"))
        fine = d.get("building") if sch.startswith("forced") else d.get("place")

        # Reconstruct the prompt actually sent. CONTEXTS is global, so it has
        # to be reloaded whenever the site changes.
        full_prompt = ""
        lad = ladders.get(st)
        if lad and ctx in lad:
            prompts.CONTEXTS.clear()
            prompts.CONTEXTS.update(lad)
            try:
                full_prompt = prompts.build(rec["lang"], ctx, sch)
            except Exception:
                pass
        meta = (lad or {}).get(ctx, {})

        e = err.get((p, rec["model"], rec["lang"], ctx, sch), (None, None, None))
        rows.append(dict(
            site=st, schema=sch, model=rec["model"], lang=rec["lang"],
            context=ctx,
            ref=meta.get("name_en") or meta.get("name_zh") or ("(no context)" if ctx == "none" else ""),
            ref_km=meta.get("dist_km"), tier=meta.get("tier") or "",
            city=d.get("city"), area=d.get("area"), fine=fine,
            conf=d.get("confidence_building", d.get("confidence_area", d.get("confidence"))),
            error_km=e[0], level=e[1] or "",
            img="../data/" + p.replace("\\", "/") if st and not st.startswith("mmsvpr") else
                "../data/mmsvpr/" + p.replace("\\", "/"),
            prompt=full_prompt, raw=rec.get("raw") or "",
            clues=" | ".join(map(str, d.get("clues") or []))[:600],
            err=rec.get("error") or ""))
    return rows


TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>GeoContext · raw record browser</title>
<style>
:root{--bd:#d8dce3;--mut:#6b7280;--hi:#eff6ff}
*{box-sizing:border-box}
body{font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;margin:0;padding:12px;color:#111}
h1{font-size:16px;margin:0 0 4px}
.sub{color:var(--mut);margin-bottom:10px}
#bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:8px;
     position:sticky;top:0;background:#fff;padding:6px 0;z-index:5;border-bottom:1px solid var(--bd)}
select,input{font:12px inherit;padding:4px 6px;border:1px solid var(--bd);border-radius:4px}
input#q{width:260px}
table{border-collapse:collapse;width:100%;font-size:12px}
th{background:#f3f4f6;position:sticky;top:46px;cursor:pointer;user-select:none;
   text-align:left;padding:5px 7px;border-bottom:2px solid var(--bd);white-space:nowrap}
th:hover{background:#e5e7eb}
td{padding:4px 7px;border-bottom:1px solid #eef0f3;vertical-align:top}
tr.r:hover{background:var(--hi);cursor:pointer}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 5px;border-radius:3px;background:#eef0f3;font-size:11px}
.big{background:#fee2e2}.mid{background:#fef3c7}.small{background:#dcfce7}
tr.det>td{background:#fafbfc;padding:10px 14px}
.det-grid{display:grid;grid-template-columns:300px 1fr;gap:14px}
.det img{max-width:300px;border:1px solid var(--bd);border-radius:4px}
pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid var(--bd);
    border-radius:4px;padding:8px;margin:4px 0 10px;max-height:320px;overflow:auto;font-size:11.5px}
.lbl{font-weight:600;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
#count{color:var(--mut);margin-left:auto}
</style>
<h1>GeoContext · raw record browser</h1>
<div class="sub">Click any row to expand: full prompt, raw response, structured fields, image. Click a column header to sort.</div>
<div id="bar"></div>
<table><thead><tr id="hd"></tr></thead><tbody id="tb"></tbody></table>
<script>
const DATA = __DATA__;
const COLS = [
 ["site","site"],["schema","schema"],["model","model"],["lang","lang"],
 ["ref","reference"],["ref_km","dist_km"],["tier","tier"],
 ["city","city"],["area","area"],["fine","building/place"],
 ["conf","confidence"],["error_km","error_km"],["level","resolved_level"]];
const FILTERS = ["site","schema","model","lang","tier","level"];
let sortKey=null, sortDir=1, open=new Set();

const bar=document.getElementById("bar");
const q=document.createElement("input"); q.id="q"; q.placeholder="full-text search (reference/answer/clues...)";
bar.appendChild(q);
const sels={};
for(const f of FILTERS){
  const s=document.createElement("select");
  const vals=[...new Set(DATA.map(r=>r[f]).filter(v=>v!==null&&v!==""))].sort();
  s.innerHTML=`<option value="">${f} (all)</option>`+vals.map(v=>`<option>${v}</option>`).join("");
  s.onchange=render; sels[f]=s; bar.appendChild(s);
}
const cnt=document.createElement("span"); cnt.id="count"; bar.appendChild(cnt);
q.oninput=render;

document.getElementById("hd").innerHTML=COLS.map(([k,t])=>`<th data-k="${k}">${t}</th>`).join("");
document.querySelectorAll("th").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; sortDir = (sortKey===k)? -sortDir : 1; sortKey=k; render();});

function esc(s){return (s??"").toString().replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function errCls(v){ if(v==null) return ""; return v<1?"small":v<10?"mid":"big"; }

function render(){
  const term=q.value.trim().toLowerCase();
  let rows=DATA.filter(r=>{
    for(const f of FILTERS){ const v=sels[f].value; if(v && String(r[f])!==v) return false; }
    if(!term) return true;
    return ["ref","city","area","fine","clues","raw","context"]
      .some(k=>String(r[k]??"").toLowerCase().includes(term));
  });
  if(sortKey) rows.sort((a,b)=>{
    let x=a[sortKey], y=b[sortKey];
    if(x==null)return 1; if(y==null)return -1;
    if(typeof x==="number"&&typeof y==="number") return (x-y)*sortDir;
    return String(x).localeCompare(String(y))*sortDir;});
  cnt.textContent=`${rows.length} / ${DATA.length} records`;
  const tb=document.getElementById("tb"); tb.innerHTML="";
  rows.slice(0,1200).forEach((r,i)=>{
    const tr=document.createElement("tr"); tr.className="r";
    tr.innerHTML=COLS.map(([k])=>{
      let v=r[k];
      if(k==="error_km"&&v!=null) return `<td class="num"><span class="tag ${errCls(v)}">${v.toFixed(2)}</span></td>`;
      if(k==="ref_km"&&v!=null) return `<td class="num">${v.toFixed(2)}</td>`;
      if(k==="conf"&&v!=null) return `<td class="num">${v}</td>`;
      return `<td>${esc(String(v??"").slice(0,46))}</td>`;}).join("");
    const det=document.createElement("tr"); det.className="det"; det.style.display="none";
    det.innerHTML=`<td colspan="${COLS.length}"><div class="det-grid">
      <div><img src="${esc(r.img)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{textContent:'image not found: '+this.src,style:'color:#b91c1c'}))">
        <div style="font-size:11px;color:#6b7280;margin-top:4px">${esc(r.img)}</div></div>
      <div>
        <div class="lbl">prompt sent</div><pre>${esc(r.prompt)||"(could not be reconstructed)"}</pre>
        <div class="lbl">raw response</div><pre>${esc(r.raw)||esc(r.err)}</pre>
        <div class="lbl">clues</div><pre>${esc(r.clues)||"--"}</pre>
      </div></div></td>`;
    tr.onclick=()=>{det.style.display = det.style.display==="none"?"":"none";};
    tb.appendChild(tr); tb.appendChild(det);
  });
}
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    ap.add_argument("--include-mmsvpr", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.site, args.include_mmsvpr)
    print(f"{len(rows)} records")
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False)),
                   encoding="utf-8")
    print(f"written to {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print("double-click to open. Images are referenced by relative path, do not move the HTML out of results/")


if __name__ == "__main__":
    main()
