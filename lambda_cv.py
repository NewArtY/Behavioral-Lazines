"""Causal selection of the laziness strength lambda (paper Section 4.3).

A reviewer will ask: how is lambda chosen WITHOUT peeking at the target?  Here we
show that a causal inner expanding-window cross-validation (the discrete analogue
of the discrepancy / L-curve / GCV parameter choice for Tikhonov regularisation)
recovers a lambda close to the oracle and beats the unsmoothed (raw) forecast --
using only past data.

Because the real labour-market series has only ~6 yearly points (too few for a
reliable inner CV -- only the largest outer holdout has any inner room), the
demonstration is on a longer controlled synthetic series with KNOWN ground truth.
The protocol is exactly the rolling-origin one used on real data, just with enough
points to make CV meaningful.  On the 6-point real data we instead report the
phase-map-justified safe default (null, lambda~0.5).

Protocol (strictly causal): to choose lambda we only ever forecast an inner
holdout step j from the operators recovered up to j-1; the outer target (step K)
is never touched during selection.

Run:  python lambda_cv.py   (numpy only; deterministic per seed; writes timing)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from laziness import smooth_series_nullspace, smooth_series_ewma

OUT = Path(__file__).resolve().parent / "experiments"

NULL_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EWMA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]


def _extrapolate_linear(X):
    k = X.shape[0]
    tau = np.arange(1, k + 1, dtype=float)
    M = np.vstack([np.ones_like(tau), tau]).T
    coef, *_ = np.linalg.lstsq(M, X, rcond=None)
    return coef[0] + coef[1] * (k + 1)


def gen_system(seed, delta=0.03, rho=0.12, K=12, rows=6, cols=30, obs_floor=0.01):
    """Recovered operator series X (K x cols), A_list (K+1), observed b, truth."""
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
    truth = A_list[K] @ P_true[K]
    return X, A_list, b, truth


def _smooth(kind, X, A_list, lam):
    if lam == 0.0:
        return X
    if kind == "null":
        return smooth_series_nullspace(X, A_list, lam)
    return smooth_series_ewma(X, lam)


def inner_cv_select(kind, X, A_list, b, grid, j_min=3):
    """Pick lambda by mean causal one-step inner-holdout error (uses only past)."""
    best_lam, best_err = grid[0], np.inf
    for lam in grid:
        errs = []
        for j in range(j_min, X.shape[0]):
            Xs = _smooth(kind, X[:j], A_list[:j], lam)
            pred = A_list[j] @ _extrapolate_linear(Xs)
            errs.append(np.linalg.norm(pred - b[j]))
        e = float(np.mean(errs))
        if e < best_err - 1e-12:
            best_err, best_lam = e, lam
    return best_lam


def evaluate(seed, kind, grid, **kw):
    X, A_list, b, truth = gen_system(seed, **kw)
    A_fore = A_list[-1]

    def outer_err(lam):
        Xs = _smooth(kind, X, A_list[:X.shape[0]], lam)
        return float(np.linalg.norm(A_fore @ _extrapolate_linear(Xs) - truth))

    raw = outer_err(0.0)
    lam_cv = inner_cv_select(kind, X, A_list, b, grid)
    cv = outer_err(lam_cv)
    oracle = min(outer_err(l) for l in grid)
    return raw, cv, oracle, lam_cv


def run(kind, grid, n_seeds=200, **kw):
    rows = [evaluate(s, kind, grid, **kw) for s in range(n_seeds)]
    raw = np.array([r[0] for r in rows])
    cv = np.array([r[1] for r in rows])
    oracle = np.array([r[2] for r in rows])
    lams = np.array([r[3] for r in rows])
    cv_ratio = cv / raw
    print(f"\n  {kind}-laziness, causal inner-CV ({n_seeds} seeds):")
    print(f"    mean error ratio  CV/raw    = {cv_ratio.mean():.3f}  "
          f"(win rate {np.mean(cv < raw - 1e-9):.2f})")
    print(f"    mean error ratio  CV/oracle = {(cv/oracle).mean():.3f}  "
          f"(1.0 = matches oracle)")
    print(f"    selected lambda: mean={lams.mean():.2f}  "
          f"median={np.median(lams):.2f}  (grid {grid})")


def main():
    t0 = time.perf_counter()
    print("Causal CV selection of lambda on a long synthetic series (K=12).")
    print("delta=0.03 (mild trend), rho=0.12 (moderate identification jitter).")
    run("null", NULL_GRID)
    run("ewma", EWMA_GRID)
    dt = time.perf_counter() - t0
    print(f"\n[time] lambda_cv: {dt:.1f} s")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "timing.log", "a", encoding="utf-8") as fh:
        fh.write(f"lambda_cv: {dt:.1f} s\n")


if __name__ == "__main__":
    main()
