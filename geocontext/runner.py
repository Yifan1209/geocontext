"""Evaluation execution: API calls, caching, resumable runs.

Results are appended line by line to results/e1_raw.jsonl. A rerun automatically
skips combinations that already succeeded, so an interrupted run can be resumed
by reissuing the same command without paying for the same calls twice.

The `repeat` dimension is a deliberate control: asking the same image N times
distinguishes "the model is reciting a stable prior" from "the model is guessing
among several candidates".
"""
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from . import providers
from .config import DATA, MMSVPR, RESULTS

RAW = RESULTS / "e1_raw.jsonl"
CACHE_KEY = ("path", "model", "lang", "repeat", "context", "schema")


COLS = list(CACHE_KEY) + ["raw", "error", "location", "condition", "hour"]
# Historical records have no context field -- those are exactly the
# "no context" baseline runs, so backfill them as "none".
DEFAULTS = {"repeat": 0, "lang": "en", "error": None, "raw": None,
            "context": "none", "schema": "v1"}


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Read the results jsonl, filling in missing columns.

    Historical records may have been written under an older schema (for
    instance without `repeat`).
    """
    recs = ([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if path.exists() else [])
    df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = DEFAULTS.get(c, pd.NA)
    df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce").fillna(0).astype(int)
    df["context"] = df["context"].fillna("none")
    df["schema"] = df["schema"].fillna("v1")
    return df


def _done_keys(path: Path = RAW) -> set:
    df = load_raw(path)
    if df.empty:
        return set()
    ok = df[df["error"].isna()] if "error" in df else df
    return set(map(tuple, ok[list(CACHE_KEY)].values.tolist()))


def plan(sample: pd.DataFrame, models, langs=("zh",), repeats=1, path: Path = RAW,
         contexts=("none",), schema="v1"):
    """Tasks still to run, excluding combinations already cached as successful."""
    done = _done_keys(path)
    todo = [(row, m, lg, rep, ctx)
            for _, row in sample.iterrows()
            for m in models for lg in langs for rep in range(repeats) for ctx in contexts
            if (row["path"], m, lg, rep, ctx, schema) not in done]
    return todo, done


#: Per-vendor concurrency caps. Calls are almost entirely network wait
#: (measured 1.7-10s each, CPU essentially idle), so parallelism turns hours
#: into tens of minutes. But it cannot simply be opened wide -- we have already
#: hit Gemini 503s and
#: OpenRouter rate-limits, so concurrency is capped per vendor.
PROVIDER_LIMITS = {"anthropic": 6, "gemini": 4, "openrouter": 8}


def _provider_of(model_name: str) -> str:
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gemini"):
        return "gemini"
    return "openrouter"


def run_parallel(sample: pd.DataFrame, models, langs=("zh",), repeats=1,
                 path: Path = RAW, verbose=True, contexts=("none",),
                 limits=None, schema="v1"):
    """Parallel version. Same semantics as run(), just concurrent requests.

    Two things need thread safety: jsonl writes take a lock, and each provider
    client is constructed once and reused.
    The SDK clients are themselves thread-safe.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    limits = {**PROVIDER_LIMITS, **(limits or {})}
    todo, done = plan(sample, models, langs, repeats, path, contexts, schema)
    total = len(sample) * len(models) * len(langs) * repeats * len(contexts)
    if verbose:
        print(f"{total} combinations (schema={schema}); {len(done)} done; "
              f"{len(todo)} to run", flush=True)
    if not todo:
        return {"new": 0, "errors": 0, "tok_in": 0, "tok_out": 0}

    clients, client_lock, write_lock = {}, threading.Lock(), threading.Lock()
    stats = {"errors": 0, "tok_in": 0, "tok_out": 0, "n": 0}
    t0 = time.time()
    fout = open(path, "a", encoding="utf-8")

    def get_client(name):
        with client_lock:
            if name not in clients:
                clients[name] = providers.REGISTRY[name]()
            return clients[name]

    def work(item):
        row, mname, lang, rep, ctx = item
        rec = {"path": row["path"], "location": row["location"],
               "condition": row["condition"], "hour": int(row["hour"]),
               "model": mname, "lang": lang, "repeat": rep, "context": ctx,
               "schema": schema, "ts": time.time()}
        img = DATA / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
        if not img.exists():
            img = MMSVPR / row["path"]
        # 503 / 429 / 529 are transient (Gemini often returns 503 "high
        # demand" at peak times).
        # Back off and retry. An earlier version recorded these as failures
        # outright, throwing away usable samples for nothing.
        delay = 2.0
        for attempt in range(4):
            try:
                out = get_client(mname).query(img, lang=lang, context=ctx,
                                              schema=schema)
                rec.update(raw=out["raw"], usage=out["usage"], error=None)
                break
            except Exception as e:
                transient = any(c in str(e) for c in ("503", "429", "529", "overloaded",
                                                      "high demand", "Timeout"))
                if transient and attempt < 3:
                    time.sleep(delay); delay *= 2
                    continue
                rec.update(raw=None, usage=None, error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc()[-800:])
                break
        with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            stats["n"] += 1
            if rec["error"]:
                stats["errors"] += 1
            else:
                stats["tok_in"] += (rec["usage"] or {}).get("input_tokens") or 0
                stats["tok_out"] += (rec["usage"] or {}).get("output_tokens") or 0
            if verbose and stats["n"] % 25 == 0:
                el = time.time() - t0
                print(f"  {stats['n']}/{len(todo)} err={stats['errors']} "
                      f"{el:.0f}s ({el/stats['n']:.2f}s/call, "
                      f"~{(len(todo)-stats['n'])*el/stats['n']/60:.0f}min left)", flush=True)

    # **One thread pool per vendor**; a single shared pool plus a semaphore
    # does not work.
    # What happened when an earlier version did that: a dozen of the 18 workers
    # sat blocked on Gemini's 4-slot semaphore while Gemini retried 503s, and
    # every other model was starved.
    # Two of the models stalled at exactly 48/144 and stayed there.
    # A slow vendor should only slow itself down, never starve the others.
    by_provider = {}
    for it in todo:
        by_provider.setdefault(_provider_of(it[1]), []).append(it)
    if verbose:
        print("  pools by vendor: " + ", ".join(
            f"{p}x{limits.get(p, 4)}({len(v)} tasks)" for p, v in by_provider.items()), flush=True)

    pools = []
    try:
        for prov, items in by_provider.items():
            ex = ThreadPoolExecutor(max_workers=limits.get(prov, 4),
                                    thread_name_prefix=prov)
            pools.append((ex, [ex.submit(work, it) for it in items]))
        for ex, futs in pools:
            list(as_completed(futs))
    finally:
        for ex, _ in pools:
            ex.shutdown(wait=True)
        fout.close()

    el = time.time() - t0
    if verbose:
        print(f"\ndone: {stats['n']} calls in {el/60:.1f} min "
              f"({el/max(stats['n'],1):.2f}s/call), {stats['errors']} failed", flush=True)
    return {"new": stats["n"], "errors": stats["errors"],
            "tok_in": stats["tok_in"], "tok_out": stats["tok_out"]}


def run(sample: pd.DataFrame, models, langs=("zh",), repeats=1,
        sleep=0.0, path: Path = RAW, verbose=True, contexts=("none",)):
    """Run the evaluation. Returns records added this call plus token totals."""
    todo, done = plan(sample, models, langs, repeats, path, contexts)
    total = len(sample) * len(models) * len(langs) * repeats * len(contexts)
    if verbose:
        print(f"{total} combinations; {len(done)} done; {len(todo)} to run")
    if not todo:
        return {"new": 0, "errors": 0, "tok_in": 0, "tok_out": 0}

    clients, tok_in, tok_out, n_err = {}, 0, 0, 0
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fout:
        for i, (row, mname, lang, rep, ctx) in enumerate(todo, 1):
            if mname not in clients:                 # lazily constructed, so a missing key
                                                     # does not break every model
                clients[mname] = providers.REGISTRY[mname]()
            img = MMSVPR / row["path"]
            rec = {"path": row["path"], "location": row["location"],
                   "condition": row["condition"], "hour": int(row["hour"]),
                   "model": mname, "lang": lang, "repeat": rep, "context": ctx,
                   "ts": time.time()}
            try:
                if not img.exists():
                    raise FileNotFoundError(img)
                out = clients[mname].query(img, lang=lang, context=ctx)
                rec.update(raw=out["raw"], usage=out["usage"], error=None)
                tok_in += out["usage"].get("input_tokens") or 0
                tok_out += out["usage"].get("output_tokens") or 0
            except Exception as e:
                rec.update(raw=None, usage=None, error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc()[-800:])
                n_err += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()                              # never lose data on interruption
            if verbose and (i % 10 == 0 or i == len(todo)):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} err={n_err} tok={tok_in}/{tok_out} "
                      f"{el:.0f}s ({el/i:.1f}s/call)", flush=True)
            if sleep:
                time.sleep(sleep)
    return {"new": len(todo), "errors": n_err, "tok_in": tok_in, "tok_out": tok_out}


def smoke_test(model: str, image_path=None, lang="zh"):
    """One image, one model, one call: checks the API key and SDK call
    signature. Does not write to the cache."""
    if image_path is None:
        s = pd.read_csv(DATA / "e1_sample.csv")
        image_path = DATA / s.iloc[0]["path"]
    p = providers.REGISTRY[model]()
    out = p.query(Path(image_path), lang=lang)
    print(f"[{model}] usage={out['usage']}")
    print(out["raw"][:600])
    return out
