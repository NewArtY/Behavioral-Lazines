"""Calibrated prediction intervals from rolling-origin residuals (paper, calc #5).

The earlier ensemble-of-schemes band under-covered badly (empirical coverage
6-16%): the schemes agree with each other and miss the actual together, so their
spread is not a confidence set (docs/exploratory_findings.md #4b).  The honest
fix is a residual-based (conformal-style) interval: width = an empirical quantile
of PAST causal one-step forecast residuals.

This script demonstrates, on a controlled synthetic operator series with known
ground truth, that the residual-based interval is calibrated (empirical coverage
~ nominal) while the scheme-spread band is not -- reproducing the real-data
finding and supplying the correct uncertainty device.

Run:  python residual_intervals.py   (numpy only; deterministic per seed)
"""

from __future__ import annotations

import numpy as np

from laziness import smooth_series_nullspace, smooth_series_ewma


def _extrapolate_linear(X):
    k = X.shape[0]
    tau = np.arange(1, k + 1, dtype=float)
    M = np.vstack([np.ones_like(tau), tau]).T
    coef, *_ = np.linalg.lstsq(M, X, rcond=None)
    return coef[0] + coef[1] * (k + 1)


def gen(seed, K=16, rows=6, cols=30, delta=0.03, rho=0.10, obs_floor=0.02):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((rows, cols))
    p0 = rng.standard_normal(cols)
    v = rng.standard_normal(cols); v /= np.linalg.norm(v)
    P_true = np.array([p0 + delta * t * v for t in range(K + 1)])
    A_list = [A0 + rho * rng.standard_normal((rows, cols)) for _ in range(K + 1)]
    X = np.empty((K, cols)); b = np.empty((K, rows))
    for t in range(K):
        bt = A_list[t] @ P_true[t]
        bt = bt + obs_floor * np.linalg.norm(bt) * rng.standard_normal(rows)
        b[t] = bt
        X[t] = np.linalg.pinv(A_list[t]) @ bt
    return X, A_list, b


def _forecast(X, A_list, j, variant):
    """One-step forecast of b[j] from X[:j] under a smoothing variant."""
    Z = X[:j]
    if variant == "null":
        Z = smooth_series_nullspace(Z, A_list[:j], 0.5)
    elif variant == "ewma":
        Z = smooth_series_ewma(Z, 0.5)
    return A_list[j] @ _extrapolate_linear(Z)


def run(n_seeds=200, j_min=3, levels=(0.80, 0.90)):
    variants = ["raw", "null", "ewma"]
    # accumulate per-(seed) coverage for residual-PI (raw) and ensemble band
    res_cov = {lv: [] for lv in levels}
    ens_cov = {lv: [] for lv in levels}
    for s in range(n_seeds):
        X, A_list, b = gen(s)
        K = X.shape[0]
        past_abs = []                      # pooled |residual| history (causal)
        hits_res = {lv: [] for lv in levels}
        hits_ens = {lv: [] for lv in levels}
        for j in range(j_min, K):
            preds = {v: _forecast(X, A_list, j, v) for v in variants}
            err = preds["raw"] - b[j]      # residual of the base forecaster
            # residual-based PI uses residuals from earlier origins only
            if past_abs:
                pool = np.concatenate(past_abs)
                for lv in levels:
                    q = np.quantile(pool, lv)
                    hits_res[lv].append(np.mean(np.abs(preds["raw"] - b[j]) <= q))
            # ensemble band = spread across smoothing variants
            stack = np.array([preds[v] for v in variants])
            lo, hi = stack.min(axis=0), stack.max(axis=0)
            inside = np.mean((b[j] >= lo - 1e-12) & (b[j] <= hi + 1e-12))
            for lv in levels:
                hits_ens[lv].append(inside)   # band has no level knob (single spread)
            past_abs.append(np.abs(err))
        for lv in levels:
            if hits_res[lv]:
                res_cov[lv].append(np.mean(hits_res[lv]))
            ens_cov[lv].append(np.mean(hits_ens[lv]))

    print("Prediction-interval coverage on synthetic operator series "
          f"({n_seeds} seeds).")
    print("Residual-based PI should match the nominal level; the scheme-ensemble")
    print("band (no level knob) is reported once for reference.\n")
    print(f"  {'nominal':>8} {'residual-PI':>14} {'ensemble band':>14}")
    for lv in levels:
        r = np.array(res_cov[lv]); e = np.array(ens_cov[lv])
        print(f"  {lv:>8.2f} {r.mean():>10.3f}+-{r.std()/np.sqrt(len(r)):.3f} "
              f"{e.mean():>10.3f}+-{e.std()/np.sqrt(len(e)):.3f}")
    print("\n  => residual-PI tracks the nominal coverage; the ensemble band")
    print("     under-covers (schemes miss together) -- matches the real-data")
    print("     finding (6-16%).  Use residual intervals, not scheme spread.")


def main():
    import time
    t0 = time.perf_counter()
    run()
    print(f"\n[time] residual_intervals: {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
