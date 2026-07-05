# -*- coding: utf-8 -*-
"""CAUSAL lambda cross-validation on the REAL curated OKVED-2 panel (2017-2024).

With 8 snapshots the inner expanding-window CV becomes non-degenerate (the paper
previously had to fix lambda=0.5 a priori on the 6-point panel).  Protocol,
strictly causal (no outer-target peeking):

  outer holdout h (base = periods[:h], target = periods[h]), h = 3..7:
    inner steps j = 3..h-1: forecast periods[j] from periods[:j] for every
    lambda in the grid (one segmentation per j, variants applied post hoc);
    lambda*(h) = argmin over the grid of the mean inner MAE (or max inner
    reliability -- both selectors reported); ties -> smaller lambda.
    h = 3 has no inner step -> fall back to the a-priori default 0.5.
  oracle(h) = the outer-best lambda (upper bound, NOT causal).

Run:  python lambda_cv_real.py     (writes lambda_cv_real.out)
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINDPROB = HERE.parents[2] / "FindProbability"
sys.path.insert(0, str(FINDPROB))

from python_forecast import _bootstrap  # noqa: E402,F401
from python_forecast.scripts.explore_laziness import load_periods, SKW  # noqa: E402
from python_forecast.core.series import segment_series                  # noqa: E402
from python_forecast.core.matrices import build_A_phys                  # noqa: E402
from python_forecast.core.laziness import (                             # noqa: E402
    smooth_series_nullspace, smooth_series_ewma,
)
from python_forecast.core.pipeline import forecast, as_forecast_dict    # noqa: E402
from python_forecast.core import direct, metrics                        # noqa: E402

DOMAIN = "okved2curated"
SCHEME = "common"
MIN_BASE = 3
DEFAULT_LAM = 0.5
GRIDS = {"null": [0.0, 0.25, 0.5, 0.75, 1.0],
         "ewma": [0.0, 0.3, 0.5, 0.7, 0.9]}


def _smooth(kind, X, A_list, lam):
    if lam == 0.0:
        return X
    if kind == "null":
        return smooth_series_nullspace(X, A_list, lam)
    return smooth_series_ewma(X, lam)


def main():
    t0 = time.perf_counter()
    n, periods = load_periods(DOMAIN)
    H = len(periods)

    # One segmentation per base length j (reused as inner AND outer bases).
    seg, mats = {}, {}
    for j in range(MIN_BASE, H):
        base = periods[:j]
        seg[j] = segment_series(n, base, **SKW)
        mats[j] = [build_A_phys(n, base[i]) for i in range(seg[j].X.shape[0])]

    def step_metrics(j, kind, lam):
        """Forecast periods[j] from periods[:j] with (kind, lam); (mae, reliab)."""
        base, actual = periods[:j], periods[j]
        Xs = _smooth(kind, seg[j].X, mats[j], lam)
        res = forecast(n, base, scheme=SCHEME, series=replace(seg[j], X=Xs),
                       solver_kwargs=SKW)
        rep = metrics.error_report(res.n_forecast, actual, n, prev=base[-1])
        return rep["mae"], rep["reliab_share"]

    cache: dict[tuple, tuple] = {}

    def cached(j, kind, lam):
        key = (j, kind, float(lam))
        if key not in cache:
            cache[key] = step_metrics(j, kind, lam)
        return cache[key]

    out = io.StringIO()

    def p(s=""):
        print(s)
        out.write(s + "\n")

    p(f"### causal lambda-CV on {DOMAIN}: {H} snapshots "
      f"({periods[0].label}..{periods[-1].label}), scheme={SCHEME}")
    naive_mae = []
    for h in range(MIN_BASE, H):
        nv = direct.naive(periods[:h])
        rep = metrics.error_report(as_forecast_dict(nv), periods[h], n,
                                   prev=periods[h - 1])
        naive_mae.append(rep["mae"])
    p(f"    naive anchor: MAE={np.mean(naive_mae):.2f}")

    for kind, grid in GRIDS.items():
        rows = {sel: [] for sel in
                ("cv_mae", "cv_rel", "fixed", "oracle", "raw")}
        lam_log = {"cv_mae": [], "cv_rel": [], "oracle": []}
        for h in range(MIN_BASE, H):
            inner_j = list(range(MIN_BASE, h))
            if inner_j:
                inner = {lam: [cached(j, kind, lam) for j in inner_j]
                         for lam in grid}
                mean_mae = {lam: np.mean([m for m, _ in inner[lam]]) for lam in grid}
                mean_rel = {lam: np.mean([r for _, r in inner[lam]]) for lam in grid}
                lam_mae = min(grid, key=lambda l: (round(mean_mae[l], 10), l))
                lam_rel = min(grid, key=lambda l: (round(-mean_rel[l], 10), l))
            else:
                lam_mae = lam_rel = DEFAULT_LAM
            outer = {lam: cached(h, kind, lam) for lam in grid}
            lam_orc = min(grid, key=lambda l: (round(outer[l][0], 10), l))
            rows["cv_mae"].append(outer[lam_mae])
            rows["cv_rel"].append(outer[lam_rel])
            rows["fixed"].append(outer[DEFAULT_LAM])
            rows["oracle"].append(outer[lam_orc])
            rows["raw"].append(outer[0.0])
            lam_log["cv_mae"].append(lam_mae)
            lam_log["cv_rel"].append(lam_rel)
            lam_log["oracle"].append(lam_orc)

        p(f"\n  == {kind}: rolling-origin MEAN over {H - MIN_BASE} holdouts ==")
        p(f"    {'selector':>10} {'MAE':>7} {'reliab':>7}   lambdas per holdout")
        for sel in ("raw", "fixed", "cv_mae", "cv_rel", "oracle"):
            a = np.array(rows[sel], float)
            lam_s = ("-" if sel in ("raw", "fixed")
                     else " ".join(f"{l:g}" for l in lam_log[sel]))
            p(f"    {sel:>10} {a[:, 0].mean():>7.2f} {a[:, 1].mean():>7.3f}   {lam_s}")
        orc = np.array(rows["oracle"], float)[:, 0].mean()
        for sel in ("fixed", "cv_mae"):
            m = np.array(rows[sel], float)[:, 0].mean()
            p(f"    ratio-to-oracle MAE ({sel}): {m / orc:.3f}")

    p(f"\n[time] {time.perf_counter() - t0:.1f} s")
    (HERE / "lambda_cv_real.out").write_text(out.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
