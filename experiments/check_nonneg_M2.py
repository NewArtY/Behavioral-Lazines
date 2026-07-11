# -*- coding: utf-8 -*-
"""M2 (reviewer): does null-space smoothing push recovered operator components
below zero on the REAL panels?  The smoother moves P~(t) inside null(A) with no
projection back onto the nonnegative cone, so this is an empirical audit.

For each domain we build the operator series exactly as the paper does
(segment_series + build_A_phys), apply the null-space smoother (lam=0.5, 1.0) and
ewma(0.5), and report how negative the components get, both on the full series and
on every rolling-origin base window (the ones actually used to forecast).

Run:  python check_nonneg_M2.py     (needs FindProbability/.venv)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FINDPROB = HERE.parents[2] / "FindProbability"
sys.path.insert(0, str(FINDPROB))
sys.path.insert(0, str(HERE))

from python_forecast import _bootstrap  # noqa: F401
from python_forecast.scripts.explore_laziness import load_periods, SKW
from python_forecast.core.series import segment_series
from python_forecast.core.matrices import build_A_phys
from python_forecast.core.laziness import smooth_series_nullspace, smooth_series_ewma


def neg_stats(X):
    """Return (min, n_neg, total, frac_neg, worst_rel) for a stack of vectors."""
    X = np.asarray(X, float)
    total = X.size
    n_neg = int((X < -1e-12).sum())
    mn = float(X.min())
    scale = float(np.abs(X).max()) + 1e-12
    # count how many rows (intervals) contain at least one negative entry
    rows_neg = int((X < -1e-12).any(axis=1).sum()) if X.ndim == 2 else 0
    return mn, n_neg, total, n_neg / total, mn / scale, rows_neg


def audit_series(tag, X, A_list):
    variants = {
        "raw":        X,
        "null:0.5":   smooth_series_nullspace(X, A_list, 0.5),
        "null:1.0":   smooth_series_nullspace(X, A_list, 1.0),
        "ewma:0.5":   smooth_series_ewma(X, 0.5),
    }
    print(f"\n  {tag}: {X.shape[0]} operators x {X.shape[1]} components")
    print(f"    {'variant':>9} {'min':>12} {'#neg':>7} {'frac_neg':>9} "
          f"{'min/scale':>10} {'rows_with_neg':>14}")
    for v, Xs in variants.items():
        mn, nneg, tot, frac, rel, rows = neg_stats(Xs)
        print(f"    {v:>9} {mn:>12.4e} {nneg:>7d} {frac:>9.4f} {rel:>10.4f} "
              f"{rows:>10d}/{Xs.shape[0]}")


def main():
    for domain in ("okved1", "okved2curated"):
        n, periods = load_periods(domain)
        print("=" * 78)
        print(f"DOMAIN {domain}: n={n}, {len(periods)} snapshots")
        print("=" * 78)
        # (A) full series (as energy_diagnostic)
        series = segment_series(n, periods, **SKW)
        A_list = [build_A_phys(n, periods[i]) for i in range(series.X.shape[0])]
        audit_series("FULL series", series.X, A_list)
        # (B) every rolling-origin base window (what forecasting actually smooths)
        worst = {"null:0.5": 0.0, "null:1.0": 0.0}
        any_neg = {"null:0.5": 0, "null:1.0": 0}
        nwin = 0
        for h in range(3, len(periods)):
            base = periods[:h]
            s = segment_series(n, base, **SKW)
            Al = [build_A_phys(n, base[i]) for i in range(s.X.shape[0])]
            nwin += 1
            for lam, key in ((0.5, "null:0.5"), (1.0, "null:1.0")):
                Xs = smooth_series_nullspace(s.X, Al, lam)
                mn = float(Xs.min())
                worst[key] = min(worst[key], mn)
                if mn < -1e-12:
                    any_neg[key] += 1
        print(f"\n  ROLLING base windows ({nwin}): worst min over all windows / "
              f"#windows with any negative")
        for key in ("null:0.5", "null:1.0"):
            print(f"    {key:>9}: worst_min={worst[key]:+.4e}   "
                  f"windows_with_neg={any_neg[key]}/{nwin}")


if __name__ == "__main__":
    main()
