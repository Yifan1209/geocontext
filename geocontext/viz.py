"""Plotting configuration, mainly so matplotlib can render CJK labels.

Matplotlib's default font carries no CJK glyphs, so Chinese text renders as
boxes and floods the log with UserWarnings. This picks a CJK font that is
actually installed and falls back to English labels when none is available.

Kept in the released package because the inspection HTML and some exploratory
notebooks still emit bilingual labels; the paper figures themselves are
English-only and produced by `analysis/make_figures.py`.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Ordered by preference. YaHei looks best; SimSun ships with almost every
# Windows install.
PREFERRED = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
             "SimHei", "SimSun", "Microsoft JhengHei", "PingFang SC", "Heiti SC"]

CJK_FONT = None


def setup(verbose: bool = True) -> str | None:
    """Install a CJK font. Returns the font actually used, or None if none found."""
    global CJK_FONT
    available = {f.name for f in fm.fontManager.ttflist}
    for name in PREFERRED:
        if name in available:
            CJK_FONT = name
            break
    if CJK_FONT:
        matplotlib.rcParams["font.sans-serif"] = [CJK_FONT] + \
            list(matplotlib.rcParams["font.sans-serif"])
        # CJK fonts frequently lack a minus glyph, which turns negative signs
        # into boxes.
        matplotlib.rcParams["axes.unicode_minus"] = False
        if verbose:
            print(f"matplotlib CJK font: {CJK_FONT}")
    elif verbose:
        print("no CJK font found; use English labels in figures")
    matplotlib.rcParams["figure.dpi"] = 110
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.3
    return CJK_FONT


def has_cjk() -> bool:
    return CJK_FONT is not None


def label(zh: str, en: str) -> str:
    """Use the CJK label when a CJK font is available, else the English one,
    so figures never render boxes on someone else's machine."""
    return zh if CJK_FONT else en


setup(verbose=False)
