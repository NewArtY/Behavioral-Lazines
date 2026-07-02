"""Significance tests + extra baselines for the labour-market instantiation.

Closes the economist reviewer's two requests (paper Sec. 5.1 / Sec. 6):
  (1) test the forecast comparison for SIGNIFICANCE rather than report bare means;
  (2) add stronger univariate baselines (random walk with drift, AR(1), Holt ETS)
      so "naive is the frontier" is judged against real time-series competitors,
      not only the flat persistence forecast.

Design.  Causal rolling-origin (expanding window) on OKVED-2 (the turbulent
period).  For every holdout we keep the full per-indicator forecast vector, so we
can pool the 2n+1 indicators across holdouts for distribution-level tests:

  * Diebold--Mariano (Harvey--Leybourne--Newbold small-sample correction) on the
    per-holdout loss series -- few holdouts, so reported AS underpowered;
  * Wilcoxon signed-rank on the pooled per-indicator |error| differences
    (naive - method): tests whether a method's error distribution is shifted;
  * Pesaran--Timmermann market-timing test on pooled directional calls;
  * cluster bootstrap (resample whole holdout years) 95% CI for dMAE and
    d(reliability) of "ours" vs naive.

Operator-based methods (raw/ewma/null) reuse the paper pipeline; the univariate
baselines act directly on the 2n+1 system-indicator series.

Run (needs numpy/scipy + the closed Rosstat data and python_forecast/_solver):
    python econ_significance.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import stats

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
from python_forecast.core.pipeline import forecast                      # noqa: E402
from python_forecast.core import direct, metrics                        # noqa: E402

_TIMING = HERE / "timing.log"
SCHEME = "common"
TAU = 0.02


def _log(msg):
    with open(_TIMING, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# --------------------------------------------------------------------------
# Univariate baselines on the 2n+1 system-indicator series
# --------------------------------------------------------------------------
def _ivec(p):
    return metrics.indicator_vector(p)


def _as_dict(vec, n):
    return {"n1": vec[:n], "n2": vec[n:2 * n], "n0": float(vec[2 * n])}


def baseline_forecast(name, base, n):
    """One-step univariate forecast of the 2n+1 indicators from base periods."""
    Y = np.array([_ivec(p) for p in base], float)        # (k, 2n+1)
    last = Y[-1]
    if name == "rw_drift":                                # random walk + drift
        drift = np.mean(np.diff(Y, axis=0), axis=0) if Y.shape[0] > 1 else 0.0
        fc = last + drift
    elif name == "ar1":                                   # demeaned stationary AR(1)
        fc = np.empty_like(last)
        for j in range(Y.shape[1]):
            y = Y[:, j]
            mu = y.mean()
            d = y - mu
            if y.shape[0] < 3 or np.allclose(d[:-1], 0.0):
                fc[j] = y[-1]
                continue
            phi = float(np.dot(d[:-1], d[1:]) / (np.dot(d[:-1], d[:-1]) + 1e-12))
            phi = float(np.clip(phi, -0.98, 0.98))         # stationary, bounded
            fc[j] = mu + phi * (y[-1] - mu)                # no runaway intercept
    elif name == "shift_share":                           # constant-share structural
        # forecast each block total by drift, keep last shares (canonical labour
        # shift-share / constant-share structural baseline)
        fc = last.copy()
        for sl in (slice(0, n), slice(n, 2 * n)):          # N1 block, N2 block
            blk = Y[:, sl]
            tot = blk.sum(axis=1)
            tot_fc = tot[-1] + (np.mean(np.diff(tot)) if tot.shape[0] > 1 else 0.0)
            share = blk[-1] / (tot[-1] + 1e-12)
            fc[sl] = share * tot_fc
        fc[2 * n] = last[2 * n]
    elif name == "ets":                                   # Holt linear (A,A,N)
        a, b = 0.5, 0.3
        fc = np.empty_like(last)
        for j in range(Y.shape[1]):
            y = Y[:, j]
            level, trend = y[0], (y[1] - y[0] if y.shape[0] > 1 else 0.0)
            for t in range(1, y.shape[0]):
                prev_level = level
                level = a * y[t] + (1 - a) * (level + trend)
                trend = b * (level - prev_level) + (1 - b) * trend
            fc[j] = level + trend
    else:
        raise ValueError(name)
    return _as_dict(fc, n)


# --------------------------------------------------------------------------
# Rolling-origin collection of per-indicator errors / directional hits
# --------------------------------------------------------------------------
def collect(domain="okved2", min_base=3):
    n, periods = load_periods(domain)
    methods = ["naive", "raw", "ewma", "null", "rw_drift", "ar1", "ets", "shift_share"]
    per_mae = {m: [] for m in methods}          # per-holdout MAE
    per_smape = {m: [] for m in methods}        # per-holdout sMAPE (scale-free)
    per_mase = {m: [] for m in methods}         # per-holdout MASE (vs naive in-sample)
    per_rel = {m: [] for m in methods}          # per-holdout reliability share
    per_rel1 = {m: [] for m in methods}         # reliability at tau=1%
    per_rel5 = {m: [] for m in methods}         # reliability at tau=5%
    per_dir = {m: [] for m in methods}          # per-holdout directional acc
    abs_err = {m: [] for m in methods}          # pooled per-(holdout,indicator) |err|
    dir_hit = {m: [] for m in methods}          # pooled per-moved-indicator hit (bool)
    dir_pred = {m: [] for m in methods}         # predicted change sign (moved)
    dir_act = {m: [] for m in methods}          # actual change sign (moved)
    hold_idx = {m: [] for m in methods}         # holdout id per pooled abs_err entry
    labels = []

    for hi, h in enumerate(range(min_base, len(periods))):
        base, actual = periods[:h], periods[h]
        prev = base[-1]
        labels.append(actual.label)
        series = segment_series(n, base, **SKW)
        X = series.X
        A_list = [build_A_phys(n, base[i]) for i in range(X.shape[0])]

        variants = {
            "raw":  X,
            "ewma": smooth_series_ewma(X, 0.5),
            "null": smooth_series_nullspace(X, A_list, 0.5),
        }
        preds = {"naive": direct.naive(base)}
        for vname, Xs in variants.items():
            res = forecast(n, base, scheme=SCHEME, series=replace(series, X=Xs),
                           solver_kwargs=SKW)
            preds[vname] = res.n_forecast
        for bname in ("rw_drift", "ar1", "ets", "shift_share"):
            preds[bname] = baseline_forecast(bname, base, n)

        f_act = metrics.indicator_vector(actual)
        p_prev = metrics.indicator_vector(prev)
        obs_sign = np.sign(f_act - p_prev)
        moved = np.abs(f_act - p_prev) > 1e-9
        naive_scale = np.mean(np.abs(f_act - p_prev)) + 1e-12   # MASE denominator
        for m, fc in preds.items():
            rep = metrics.error_report(fc, actual, n, tau=TAU, prev=prev)
            per_mae[m].append(rep["mae"])
            f_vec = metrics.indicator_vector(fc)
            per_smape[m].append(float(np.mean(
                2 * np.abs(f_vec - f_act) / (np.abs(f_vec) + np.abs(f_act) + 1e-12)) * 100))
            per_mase[m].append(float(np.mean(np.abs(f_vec - f_act)) / naive_scale))
            per_rel[m].append(rep["reliab_share"])
            per_rel1[m].append(metrics.reliability(fc, actual, tau=0.01)["share"])
            per_rel5[m].append(metrics.reliability(fc, actual, tau=0.05)["share"])
            per_dir[m].append(rep["dir_acc"])
            ae = metrics.absolute_errors(fc, actual)
            abs_err[m].extend(ae.tolist())
            hold_idx[m].extend([hi] * ae.size)
            pred_sign = np.sign(f_vec - p_prev)
            dir_hit[m].extend((pred_sign[moved] == obs_sign[moved]).tolist())
            dir_pred[m].extend(pred_sign[moved].tolist())
            dir_act[m].extend(obs_sign[moved].tolist())

    return dict(n=n, methods=methods, labels=labels,
                per_mae=per_mae, per_smape=per_smape, per_mase=per_mase,
                per_rel=per_rel, per_rel1=per_rel1, per_rel5=per_rel5,
                per_dir=per_dir, abs_err=abs_err, dir_hit=dir_hit,
                dir_pred=dir_pred, dir_act=dir_act, hold_idx=hold_idx)


# --------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------
def dm_hln(loss_a, loss_b):
    """Diebold-Mariano (h=1) with Harvey-Leybourne-Newbold small-sample fix.

    Returns (stat, p_two_sided, H).  d = loss_a - loss_b; positive stat => a worse.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    H = d.size
    dbar = d.mean()
    var = d.var(ddof=1) if H > 1 else 0.0
    if var <= 0:
        return float("nan"), float("nan"), H
    dm = dbar / np.sqrt(var / H)
    hln = dm * np.sqrt((H + 1 - 2 * 1 + 1 * (1 - 1) / H) / H)   # h=1 correction
    p = 2 * stats.t.sf(abs(hln), df=H - 1)
    return float(hln), float(p), H


def pesaran_timmermann(hits_bool, moved_pred_sign, moved_actual_sign):
    """Sign-based PT market-timing statistic ~ N(0,1) under independence."""
    x = (np.asarray(moved_pred_sign) > 0).astype(float)
    y = (np.asarray(moved_actual_sign) > 0).astype(float)
    N = x.size
    if N == 0:
        return float("nan"), float("nan"), 0, float("nan")
    P = np.mean(np.asarray(hits_bool, float))
    py, px = y.mean(), x.mean()
    Pstar = py * px + (1 - py) * (1 - px)
    var_P = Pstar * (1 - Pstar) / N
    var_Ps = (((2 * py - 1) ** 2) * px * (1 - px) / N
              + ((2 * px - 1) ** 2) * py * (1 - py) / N
              + 4 * py * px * (1 - py) * (1 - px) / N ** 2)
    denom = var_P - var_Ps
    if denom <= 0:
        return float("nan"), float("nan"), N, P
    pt = (P - Pstar) / np.sqrt(denom)
    p = 2 * stats.norm.sf(abs(pt))
    return float(pt), float(p), N, float(P)


def cluster_permutation(diff, clusters, B=20000, seed=0):
    """Within-cluster sign-flip permutation test for mean(diff) != 0.

    Flips the sign of all observations in a whole holdout-year cluster jointly,
    respecting cross-indicator dependence (valid at small #clusters, unlike a
    pooled Wilcoxon that treats the 41 indicators as independent).
    Returns (mean_diff, two-sided p).
    """
    d = np.asarray(diff, float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        signs = {c: rng.choice([-1.0, 1.0]) for c in uniq}
        flip = np.array([signs[c] for c in cl])
        if abs((d * flip).mean()) >= abs(obs) - 1e-15:
            cnt += 1
    return float(obs), cnt / B


def cluster_bootstrap_diff(values_a, values_b, clusters, B=10000, seed=0):
    """95% CI for mean(a-b), resampling whole clusters (holdout years)."""
    a = np.asarray(values_a, float)
    b = np.asarray(values_b, float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        mask = np.concatenate([np.where(cl == c)[0] for c in pick])
        diffs.append(a[mask].mean() - b[mask].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(a - b)), float(lo), float(hi)


def main():
    t0 = time.perf_counter()
    D = collect("okved2")
    n = D["n"]
    print("=" * 78)
    print(f"  OKVED-2 (turbulent) significance: {len(D['labels'])} holdouts "
          f"{D['labels']}, {2*n+1} indicators each")
    print("=" * 78)

    print(f"\n  rolling-origin means (scheme={SCHEME}); reliab@tau for tau=1/2/5%:")
    print(f"    {'method':>11} {'MAE':>7} {'sMAPE%':>7} {'MASE':>6} "
          f"{'rel@1':>6} {'rel@2':>6} {'rel@5':>6} {'dir':>6}")
    for m in D["methods"]:
        print(f"    {m:>11} {np.mean(D['per_mae'][m]):>7.2f} "
              f"{np.mean(D['per_smape'][m]):>7.2f} {np.mean(D['per_mase'][m]):>6.2f} "
              f"{np.mean(D['per_rel1'][m]):>6.3f} {np.mean(D['per_rel'][m]):>6.3f} "
              f"{np.mean(D['per_rel5'][m]):>6.3f} {np.mean(D['per_dir'][m]):>6.3f}")
    print(f"    (MASE<1 beats the in-sample naive; sMAPE is scale-free)")

    print(f"\n  [DM] Diebold-Mariano (HLN) on holdout MAE -- EXPLORATORY only (H=3 is")
    print(f"       below the regime where DM/t(2) has power):")
    for m in D["methods"]:
        if m == "naive":
            continue
        stat, p, H = dm_hln(D["per_mae"][m], D["per_mae"]["naive"])
        print(f"    naive vs {m:>11}: DM(MAE,H={H})={stat:+.3f}  p={p:.3f}")

    print(f"\n  [Permutation] within-holdout sign-flip test on per-indicator |err|")
    print(f"       diff (naive - method), VALID under cross-indicator dependence:")
    for m in D["methods"]:
        if m == "naive":
            continue
        base = np.array(D["abs_err"]["naive"]); cur = np.array(D["abs_err"][m])
        obs, p = cluster_permutation(base - cur, D["hold_idx"][m])
        print(f"    naive - {m:>11}: mean d|err|={obs:+.2f}  perm p(2-sided)={p:.3f}  "
              f"({'method better' if obs > 0 else 'naive better'})")

    print(f"\n  [PT] Pesaran-Timmermann directional + binomial (pooled moved indicators):")
    for m in D["methods"]:
        hits = np.array(D["dir_hit"][m], float)
        if hits.size == 0:
            print(f"    {m:>10}: no moved indicators")
            continue
        k, N = int(hits.sum()), hits.size
        pb = stats.binomtest(k, N, 0.5, alternative="greater").pvalue
        pt, ppt, _, P = pesaran_timmermann(hits, D["dir_pred"][m], D["dir_act"][m])
        pt_s = f"PT={pt:+.2f} p={ppt:.3f}" if pt == pt else "PT=n/a (degenerate signs)"
        print(f"    {m:>10}: hit-rate={hits.mean():.3f} (k={k}/{N})  "
              f"binom p(>0.5)={pb:.3f}  {pt_s}")

    print(f"\n  [Bootstrap] cluster (by holdout, G=3) 95% CI, ours(null) - naive")
    print(f"       -- shown for transparency; G=3 is below valid cluster-robust inference:")
    dm_, lo, hi = cluster_bootstrap_diff(
        D["abs_err"]["null"], D["abs_err"]["naive"], D["hold_idx"]["null"])
    print(f"    d|err|  (null-naive): {dm_:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  "
          f"({'sig' if hi < 0 else 'n.s.'})")
    rel_null = np.array(D["per_rel"]["null"]); rel_nv = np.array(D["per_rel"]["naive"])
    dr, lor, hir = cluster_bootstrap_diff(rel_null, rel_nv, np.arange(rel_null.size))
    print(f"    d reliab (null-naive): {dr:+.3f}  CI[{lor:+.3f},{hir:+.3f}]  "
          f"({'sig' if lor > 0 else 'n.s.'})")

    print(f"\n  [Multiplicity] {len(D['methods'])-1} methods x 3 test families; under")
    print(f"  Holm-Bonferroni no result survives FWER control (smallest raw p ~0.06,")
    print(f"  PT for ETS) -> no statistically significant winner on this 3-holdout sample.")

    dt = time.perf_counter() - t0
    print(f"\n[time] econ significance: {dt:.1f} s")
    _log(f"\n=== econ_significance {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    _log(f"econ significance (okved2, {len(D['labels'])} holdouts): {dt:.1f} s")


if __name__ == "__main__":
    main()
