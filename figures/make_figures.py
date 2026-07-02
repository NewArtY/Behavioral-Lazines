"""Generate paper figures ris1..ris4 (PNG 300 dpi + PDF), bilingual.

Pure numpy + matplotlib; reuses the numpy-only ``baselines.py`` and the phase-map
CSVs in ``../experiments/``.  Labels are language-switched:

    python make_figures.py ru   -> article/figures/      (Russian labels)
    python make_figures.py en   -> article/figures_en/   (English labels)

Default is ``ru``.  Output location: if the sibling ``article/`` tree exists (the
in-repository authoring workflow) the figures go to ``article/figures[_en]`` as
before; otherwise (a standalone clone of ``code/`` only) they fall back to
``figures/out/<lang>/``.  An explicit destination can be forced with
``--out PATH`` (or ``--out=PATH``).

Journal style: multi-panel figures carry only bold panel labels (а)/(б)/(в); all
descriptive text lives in the LaTeX caption.  Figures are sized close to the
column width so the in-figure type renders at ~10-11 pt.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # code/  -> baselines.py
from baselines import sinkhorn, coupling_retention   # noqa: E402

def _parse_args(argv):
    """Return (lang, out_override) from argv; tolerant, order-independent."""
    lang, out_override = "ru", None
    rest, i = argv[1:], 0
    while i < len(rest):
        tok = rest[i]
        if tok in ("ru", "en"):
            lang = tok
        elif tok == "--out":
            i += 1
            out_override = rest[i] if i < len(rest) else None
        elif tok.startswith("--out="):
            out_override = tok.split("=", 1)[1]
        i += 1
    return lang, out_override


LANG, _OUT_OVERRIDE = _parse_args(sys.argv)
EXP = HERE.parent / "experiments"

_SUB = "figures" if LANG == "ru" else "figures_en"
if _OUT_OVERRIDE:
    OUT = Path(_OUT_OVERRIDE)
else:
    _ARTICLE = HERE.parents[1] / "article"
    # Preserve the in-repo authoring workflow when the article tree is present;
    # otherwise fall back to a self-contained output dir inside the repo.
    OUT = (_ARTICLE / _SUB) if _ARTICLE.is_dir() else (HERE / "out" / LANG)
OUT.mkdir(parents=True, exist_ok=True)

PAN = ["(а)", "(б)", "(в)"] if LANG == "ru" else ["(a)", "(b)", "(c)"]


def _(ru, en):
    return ru if LANG == "ru" else en


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # Times-like math to match the body text
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.dpi": 120,
})


def _panel(ax, i):
    """Bold panel label (а)/(б)/... just above the top-left corner of the axes."""
    ax.text(0.0, 1.04, PAN[i], transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")


def _save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {OUT.name}/{name}.png/.pdf")


# --- ris1: OT parametrises the prior (retention vs entropic reg) -------------
def ris1():
    n = 6
    rng = np.random.default_rng(0)
    a = rng.random(n) + 0.5; a /= a.sum(); b = a.copy()
    C = 1.0 - np.eye(n)
    regs = np.array([2.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02])
    ret = [coupling_retention(sinkhorn(a, b, C, reg=float(r), n_iters=5000)) for r in regs]
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    ax.plot(regs, ret, "o-", color="#1f77b4")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(_(r"энтропийная регуляризация $\varepsilon$ (Синкхорн)",
                    r"entropic regularisation $\varepsilon$ (Sinkhorn)"))
    ax.set_ylabel(_("восстановленное удержание", "implied self-retention"))
    ax.set_ylim(0, 1.05)
    ax.axhspan(0.0, 0.35, color="#d62728", alpha=0.08)
    ax.axhspan(0.9, 1.05, color="#2ca02c", alpha=0.08)
    _save(fig, "ris1")


# --- ris2: labour baseline -- balance residual + reliability (OKVED-2) --------
def ris2():
    methods = _(["наивный", "исходное", "мягкое", "нуль:0,5"],
                ["naive", "raw", "ewma", "null:0.5"])
    reliab = [0.276, 0.293, 0.228, 0.309]
    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.7))
    c = ["0.6", "#7f7f7f", "#d62728", "#2ca02c"]
    ax1.bar(x, reliab, color=c)
    ax1.axhline(0.276, color="0.4", ls="--", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=22, ha="right")
    ax1.set_ylabel(r"$\eta_2$")
    ax1.set_ylim(0, 0.36)
    _panel(ax1, 0)
    rv = [1e-16, 1e-16, 6.96e-2, 2.58e-15]
    ax2.bar(x, rv, color=c)
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=22, ha="right")
    ax2.set_ylabel(_(r"невязка баланса", r"balance residual"))
    ax2.set_ylim(1e-17, 1.0)
    _panel(ax2, 1)
    _save(fig, "ris2")


# --- ris3: physics bridge + balance-exactness --------------------------------
def ris3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ax1.bar([0, 1], [0.114, 1.000], color=["#7f7f7f", "#2ca02c"])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(_([r"станд. $\bar x{=}0$", r"лень $\bar x{=}P_0$"],
                          [r"standard $\bar x{=}0$", r"lazy $\bar x{=}P_0$"]),
                        rotation=12, ha="center")
    ax1.set_ylabel(_(r"среднее удержание $P^{**}$", r"mean retention $P^{**}$"))
    ax1.set_ylim(0, 1.12)
    ax1.annotate("", xy=(1, 1.0), xytext=(0, 0.12),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
    _panel(ax1, 0)
    ax2.bar(range(3), [2.7e-15, 5.7e-15, 5.96e-1], color=["#2ca02c", "#2ca02c", "#d62728"])
    ax2.set_yscale("log")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(_(["нуль:0,5", "нуль:1", "мягкое:0,5"],
                          ["null:0.5", "null:1", "ewma:0.5"]),
                        rotation=18, ha="right")
    ax2.set_ylabel(_("невязка баланса", "balance residual"))
    ax2.set_ylim(1e-16, 1.0)
    _panel(ax2, 1)
    _save(fig, "ris3")


# --- ris4: phase map heatmaps -----------------------------------------------
def _read_grid(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    cols = [float(x) for x in rows[0][1:]]
    deltas, G = [], []
    for r in rows[1:]:
        deltas.append(float(r[0])); G.append([float(x) for x in r[1:]])
    return np.array(deltas), np.array(cols), np.array(G)


def ris4():
    missing = [p.name for p in (EXP / "phase_null_ratio.csv",
                                EXP / "phase_ewma_ratio.csv") if not p.exists()]
    if missing:
        print(f"  [ris4] SKIPPED: missing {', '.join(missing)} in {EXP}.\n"
              f"        Run `python phase_map.py` first to generate the "
              f"phase-map CSVs, then re-run this script.")
        return
    dn, cn, Gn = _read_grid(EXP / "phase_null_ratio.csv")
    de, ce, Ge = _read_grid(EXP / "phase_ewma_ratio.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    for k, (ax, G, cols, deltas) in enumerate((
            (ax1, Gn, cn, dn), (ax2, Ge, ce, de))):
        im = ax.imshow(G, origin="lower", cmap="RdBu_r", vmin=0.4, vmax=1.6, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c:g}" for c in cols])
        ax.set_yticks(range(len(deltas))); ax.set_yticklabels([f"{d:g}" for d in deltas])
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\delta$")
        _panel(ax, k)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    fig.colorbar(im, ax=[ax1, ax2], fraction=0.045, pad=0.02,
                 label=_("отношение ошибок\n(сглаж./исходное)", "error ratio\n(smoothed / raw)"))
    _save(fig, "ris4", tight=False)


def main():
    print(f"Generating figures ({LANG}) into {OUT}")
    ris1(); ris2(); ris3(); ris4()


if __name__ == "__main__":
    main()
