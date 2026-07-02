"""Self-contained synthetic demonstration for the paper (Section 5.3).

Controlled setup with a KNOWN ground truth: a stable true transfer operator with
a mild linear drift, observed through per-interval balance matrices ``A(t)`` that
slowly rotate (so the null space turns each step).  The minimum-norm recovery
jitters in the rotating null space even though the truth is smooth; causal
least-action smoothing removes the jitter.

Reproduces the qualitative result reported in the paper: the hard,
increment-preserving null-space variant lowers the forecast error of the
extrapolated operator, while soft EWMA -- which also distorts the data-consistent
part -- hurts when the drift is genuine.

Run:  python synthetic_demo.py   (needs only numpy; deterministic, seed=0)
"""

from __future__ import annotations

import numpy as np

from laziness import (
    smooth_series_nullspace, smooth_series_ewma, trajectory_energy,
)


def _extrapolate_linear(X: np.ndarray) -> np.ndarray:
    """Per-component linear trend on tau=1..k, predicted at tau=k+1."""
    k = X.shape[0]
    tau = np.arange(1, k + 1, dtype=float)
    M = np.vstack([np.ones_like(tau), tau]).T
    coef, *_ = np.linalg.lstsq(M, X, rcond=None)
    return coef[0] + coef[1] * (k + 1)


def run(seed: int = 0, rows: int = 6, cols: int = 30, k: int = 5,
        drift: float = 0.04, rot: float = 0.05) -> None:
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((rows, cols))
    p0 = rng.standard_normal(cols)
    v = rng.standard_normal(cols)
    v = v / np.linalg.norm(v)
    # true operator: mild, stable linear drift
    P_true = np.array([p0 + drift * t * v for t in range(k + 1)])
    # per-interval matrices: A0 slowly rotated (null space turns each step)
    A_list = [A0 + rot * rng.standard_normal((rows, cols)) for _ in range(k + 1)]
    # min-norm recovery of each interval's operator from its own increment
    X_rec = np.array([np.linalg.pinv(A_list[t]) @ (A_list[t] @ P_true[t])
                      for t in range(k)])

    A_fore = A_list[k]
    truth_incr = A_fore @ P_true[k]

    def fore_err(Xseries: np.ndarray) -> float:
        return float(np.linalg.norm(A_fore @ _extrapolate_linear(Xseries) - truth_incr))

    raw_err = fore_err(X_rec)
    print("Synthetic: stable true operator, rotating A(t).")
    print("Forecast error ||A_fore @ p_pr - truth|| (lower = better).\n")
    print(f"  {'variant':>12} {'traj_energy':>12} {'fore_err':>10} {'/ raw':>7}")
    print(f"  {'raw(minnorm)':>12} {trajectory_energy(X_rec):>12.4g} "
          f"{raw_err:>10.4f} {1.0:>7.3f}")
    for lam in (0.5, 1.0):
        Xs = smooth_series_nullspace(X_rec, A_list[:k], lam)
        e = fore_err(Xs)
        print(f"  {'null:'+format(lam,'g'):>12} {trajectory_energy(Xs):>12.4g} "
              f"{e:>10.4f} {e/raw_err:>7.3f}")
    for lam in (0.3, 0.5, 0.7):
        Xs = smooth_series_ewma(X_rec, lam)
        e = fore_err(Xs)
        print(f"  {'ewma:'+format(lam,'g'):>12} {trajectory_energy(Xs):>12.4g} "
              f"{e:>10.4f} {e/raw_err:>7.3f}")


if __name__ == "__main__":
    run()
