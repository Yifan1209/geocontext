"""Paths and API keys.

Keys live in a `.env` at the project root (gitignored), so a fresh terminal or
notebook picks them up automatically instead of needing an export every time.
The loader below is deliberately minimal, to avoid a python-dotenv dependency.
"""
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
MMSVPR = DATA / "mmsvpr"          # MMS-VPR source data (tarball, manifest, extracted images)
RESULTS = PROJ / "results"
# Context ladders live under data/, not results/: in this repository they are
# *shipped data* rather than something you regenerate, and newly built ladders
# belong beside them. (They sat in results/ in the internal project, which meant
# load_ladder() looked in a directory the release never populated.)
LADDERS = DATA / "ladders"
NOTES = PROJ / "notes"
RESULTS.mkdir(exist_ok=True)
LADDERS.mkdir(parents=True, exist_ok=True)

TRASH = PROJ / "trash"

ENV_FILE = PROJ / ".env"
KEYS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "MAPILLARY_TOKEN")

# Straight quotes plus the curly variants. Editors and IME input methods turn
# " into “ ” very easily, and a key carrying a curly quote raises
# UnicodeEncodeError when it goes out in an HTTP header -- with an error
# message that gives no hint about the real cause.
QUOTES = "'\"‘’“”«»`"


def load_env(path: Path = ENV_FILE, override: bool = False) -> list[str]:
    """Load .env into os.environ. Returns the names set on this call.

    Variables already present in the environment are not overwritten by default.
    """
    if not path.exists():
        return []
    setted = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip(QUOTES).strip()
        if v and (override or not os.environ.get(k)):
            os.environ[k] = v
            setted.append(k)
    return setted


def check_keys() -> None:
    """Health check: report each key's status and flag non-ASCII characters
    (curly quotes, full-width spaces and similar)."""
    load_env(override=True)
    for k in KEYS:
        v = os.environ.get(k)
        if not v:
            print(f"  {k:22} not set")
            continue
        bad = {c for c in v if ord(c) > 127}
        if bad:
            names = ", ".join(f"U+{ord(c):04X}({c})" for c in sorted(bad))
            print(f"  {k:22} WARNING non-ASCII characters: {names} "
                  "-- usually an input method turning quotes curly")
        else:
            print(f"  {k:22} OK (len={len(v)}, prefix {v[:6]}...)")


def key_status() -> dict[str, str]:
    """Status of each key. Reports length only, never the value."""
    load_env()
    return {k: (f"set (len={len(v)})" if (v := os.environ.get(k)) else "not set")
            for k in KEYS}


def require(*keys: str) -> None:
    load_env()
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing keys: {', '.join(missing)}.\n"
            f"Add them to {ENV_FILE}, one per line:\n"
            + "\n".join(f"  {k}=your_key" for k in missing)
        )


load_env()


def trash(path, reason: str = "") -> Path | None:
    """Move a file or directory into trash/ instead of deleting it.

    In a research project "obviously regenerable" is often wrong: a cache entry,
    an error log, a superseded CSV may be the only surviving record of some run.
    An rm cannot be undone, while looking through trash/ costs nothing.
    Deletion is left to a human.

    Returns the new path, or None if the source does not exist.
    """
    import shutil
    from datetime import datetime
    p = Path(path)
    if not p.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        rel = p.relative_to(PROJ)
    except ValueError:
        rel = Path(p.name)
    dest = TRASH / f"{rel.as_posix().replace('/', '__')}.{stamp}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dest))
    if reason:
        with (TRASH / "_reasons.log").open("a", encoding="utf-8") as f:
            f.write(f"{stamp}\t{rel}\t{reason}\n")
    return dest
