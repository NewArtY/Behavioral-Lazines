# -*- coding: utf-8 -*-
"""M7 (reviewer): the only Holm-surviving result -- directional accuracy of the
random-walk-with-drift baseline -- uses the Pesaran-Timmermann test on the pooled
41 x 5 = 205 directional calls AS IF independent, ignoring within-year (cross-
indicator) dependence.  Effective #independent units ~ 5 (years), not 205.

This script (a) reproduces the pooled PT p-value, then (b) redoes the directional
significance CLUSTERED BY YEAR: per-year hit rates, a year-cluster bootstrap CI of
the pooled hit rate, a year-block permutation p-value, and the 5-years-as-units
test -- the honest small-sample picture.

Run:  python clustered_directional_M7.py   (needs FindProbability/.venv)
"""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
FINDPROB = HERE.parents[2] / "FindProbability"
sys.path.insert(0, str(FINDPROB))
sys.path.insert(0, str(HERE))

from python_forecast import _bootstrap  # noqa: F401
from python_forecast.scripts.explore_laziness import load_periods, SKW
from python_forecast.core.series import segment_series
from python_forecast.core.matrices import build_A_phys
from python_forecast.core.laziness import smooth_series_nullspace, smooth_series_ewma
from python_forecast.core.pipeline import forecast
from python_forecast.core import direct, metrics
import econ_significance as ES

DOMAIN = "okved2curated"
SCHEME = "common"


def collect_directional():
    """Per-year directional hits for each method (year-indexed)."""
    n, periods = load_periods(DOMAIN)
    methods = ["naive", "raw", "ewma", "null", "rw_drift", "ar1", "ets", "shift_share"]
    # per-year: hits (bool list), pred signs, actual signs
    per_year = {m: [] for m in methods}   # list over years of dict(hits, pred, act)
    labels = []
    for h in range(3, len(periods)):
        base, actual = periods[:h], periods[h]
        prev = base[-1]
        labels.append(actual.label)
        series = segment_series(n, base, **SKW)
        X = series.X
        A_list = [build_A_phys(n, base[i]) for i in range(X.shape[0])]
        variants = {"raw": X, "ewma": smooth_series_ewma(X, 0.5),
                    "null": smooth_series_nullspace(X, A_list, 0.5)}
        preds = {"naive": direct.naive(base)}
        for vname, Xs in variants.items():
            preds[vname] = forecast(n, base, scheme=SCHEME,
                                    series=replace(series, X=Xs), solver_kwargs=SKW).n_forecast
        for bname in ("rw_drift", "ar1", "ets", "shift_share"):
            preds[bname] = ES.baseline_forecast(bname, base, n)
        f_act = metrics.indicator_vector(actual)
        p_prev = metrics.indicator_vector(prev)
        obs_sign = np.sign(f_act - p_prev)
        moved = np.abs(f_act - p_prev) > 1e-9
        for m, fc in preds.items():
            f_vec = metrics.indicator_vector(fc)
            pred_sign = np.sign(f_vec - p_prev)
            hits = (pred_sign[moved] == obs_sign[moved]).astype(float)
            per_year[m].append(dict(hits=hits,
                                    pred=pred_sign[moved], act=obs_sign[moved]))
    return n, labels, methods, per_year


def year_cluster_bootstrap(per_year_m, B=20000, seed=0):
    """Resample whole years; pooled hit-rate distribution -> 95% CI."""
    rng = np.random.default_rng(seed)
    G = len(per_year_m)
    rates = []
    for _ in range(B):
        pick = rng.integers(0, G, size=G)
        h = np.concatenate([per_year_m[i]["hits"] for i in pick])
        rates.append(h.mean())
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def year_block_permutation(per_year_m, B=20000, seed=0):
    """H0: no directional skill.  Flip the predicted-sign of a WHOLE year jointly
    (respecting cross-indicator dependence); recompute pooled hit rate.  p = frac
    of permutations with hit-rate >= observed.  (One-sided, skill = high hits.)"""
    rng = np.random.default_rng(seed)
    obs = np.concatenate([d["hits"] for d in per_year_m]).mean()
    G = len(per_year_m)
    cnt = 0
    for _ in range(B):
        flips = rng.choice([True, False], size=G)
        hh = []
        for i, d in enumerate(per_year_m):
            # flipping the prediction sign flips each hit to a miss and vice versa
            hh.append(1.0 - d["hits"] if flips[i] else d["hits"])
        if np.concatenate(hh).mean() >= obs - 1e-12:
            cnt += 1
    return float(obs), cnt / B


def main():
    n, labels, methods, PY = collect_directional()
    print("=" * 78)
    print(f"  M7 clustered directional test, {DOMAIN}: years {labels} (G={len(labels)}), "
          f"{2*n+1} indicators")
    print("=" * 78)

    for m in methods:
        py = PY[m]
        pooled = np.concatenate([d["hits"] for d in py])
        k, N = int(pooled.sum()), pooled.size
        # (a) pooled PT + binomial (paper's method, treats N as independent)
        pt, ppt, _, P = ES.pesaran_timmermann(
            pooled,
            np.concatenate([d["pred"] for d in py]),
            np.concatenate([d["act"] for d in py]))
        pb = stats.binomtest(k, N, 0.5, alternative="greater").pvalue
        # (b) per-year hit rates
        hr = [float(d["hits"].mean()) for d in py]
        # (c) year-cluster bootstrap CI of pooled hit rate
        lo, hi = year_cluster_bootstrap(py)
        # (d) year-block permutation p
        _, pperm = year_block_permutation(py)
        # (e) 5-years-as-units one-sided test vs 0.5 (t and sign)
        hr_arr = np.array(hr)
        t_p = float(stats.ttest_1samp(hr_arr, 0.5, alternative="greater").pvalue) \
            if hr_arr.std() > 0 else float("nan")
        print(f"\n  {m}:  pooled hit={pooled.mean():.3f} (k={k}/{N})")
        print(f"    POOLED (independence): PT p={ppt:.2e}   binomial p={pb:.2e}")
        print(f"    per-year hit rates: {['%.3f'%x for x in hr]}")
        print(f"    CLUSTERED: year-bootstrap 95% CI of hit rate = [{lo:.3f}, {hi:.3f}] "
              f"({'excludes' if lo>0.5 else 'INCLUDES'} 0.5)")
        print(f"    CLUSTERED: year-block permutation p(hit>chance) = {pperm:.3f}")
        print(f"    5-years-as-units: mean={hr_arr.mean():.3f}, one-sided t-test p={t_p:.3f}")


if __name__ == "__main__":
    main()
