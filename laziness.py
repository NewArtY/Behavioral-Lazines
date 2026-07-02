"""Behavioural laziness: causal regularisation of a transfer-operator trajectory.

Standalone, numpy-only core (synced from
``FindProbability/python_forecast/core/laziness.py``).  Intended for the public
GitHub artifact accompanying the paper.

The recovered transition operator ``P(t)`` of a multicluster system is only
identifiable up to the null space of the per-interval balance matrix ``A(t)``:
many micro-structures reproduce the same observed macro increment.  Classic
Tikhonov regularisation with reference 0 resolves the ambiguity by *minimum
norm*, which on a per-source simplex prefers the maximally-spread (high-churn)
operator -- the wrong default for an inertial system.  This module supplies the
opposite, physically-motivated default: **least action on the operator
trajectory** (the system changes its behaviour, not just its state, as little as
the data allow), in two strictly causal variants.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "null_space_project",
    "smooth_series_nullspace",
    "smooth_series_ewma",
    "trajectory_energy",
]


def null_space_project(A: np.ndarray, v: np.ndarray,
                       AAt_inv: np.ndarray | None = None) -> np.ndarray:
    """Project ``v`` onto the null space of ``A``: ``v - A^T (A A^T)^+ A v``.

    The result ``w`` satisfies ``A @ w = 0`` (to numerical precision), so adding
    any multiple of it to a point leaves ``A @ point`` unchanged.
    """
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if AAt_inv is None:
        AAt_inv = np.linalg.pinv(A @ A.T)
    return v - A.T @ (AAt_inv @ (A @ v))


def smooth_series_nullspace(X: np.ndarray, A_list, lam: float,
                            anchor: np.ndarray | None = None,
                            lam_anchor: float | None = None) -> np.ndarray:
    """Causal, increment-preserving smoothing of the operator trajectory.

    For each interval ``i``, ``X[i]`` is moved a fraction ``lam`` toward the
    previous *smoothed* operator, but only along ``null(A_list[i])`` -- so
    ``A_list[i] @ X[i]`` (the observed macro increment) is preserved exactly.
    Row 0 is optionally pulled toward a static ``anchor`` with strength
    ``lam_anchor`` (defaults to ``lam``).  ``lam`` in [0, 1]; returns a new array.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0,1], got {lam}")
    X = np.asarray(X, dtype=np.float64)
    k = X.shape[0]
    if len(A_list) != k:
        raise ValueError(f"A_list has {len(A_list)} matrices, need k={k}")
    Xs = X.copy()
    AAt_inv = [np.linalg.pinv(np.asarray(A, float) @ np.asarray(A, float).T)
               for A in A_list]
    if anchor is not None and k >= 1:
        la = lam if lam_anchor is None else lam_anchor
        d = np.asarray(anchor, float) - Xs[0]
        Xs[0] = Xs[0] + la * null_space_project(A_list[0], d, AAt_inv[0])
    for i in range(1, k):
        d = Xs[i - 1] - Xs[i]
        Xs[i] = Xs[i] + lam * null_space_project(A_list[i], d, AAt_inv[i])
    return Xs


def smooth_series_ewma(X: np.ndarray, lam: float) -> np.ndarray:
    """Causal exponential smoothing of the operator trajectory (soft inertia).

    ``Xs[0] = X[0]``; ``Xs[i] = (1-lam) X[i] + lam Xs[i-1]``.  Does NOT preserve
    the per-interval macro increment -- a forecasting regulariser that lowers the
    variance of the trajectory the trend layer extrapolates.  ``lam`` in [0, 1].
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0,1], got {lam}")
    X = np.asarray(X, dtype=np.float64)
    Xs = X.copy()
    for i in range(1, X.shape[0]):
        Xs[i] = (1.0 - lam) * X[i] + lam * Xs[i - 1]
    return Xs


def trajectory_energy(X: np.ndarray) -> float:
    """Discrete behavioural action ``sum_i ||X[i] - X[i-1]||^2`` (lower = lazier)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return 0.0
    d = np.diff(X, axis=0)
    return float(np.sum(d * d))
