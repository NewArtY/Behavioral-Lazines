"""Optimal-transport / entropic baselines for the behavioural-laziness comparison.

Standalone, numpy-only (synced from
``FindProbability/python_forecast/core/baselines.py``).

Positions the least-action (laziness) prior against the two closest families in
the literature (see ../LITERATURE.md):

  * entropic optimal transport (Sinkhorn) -- the Waddington-OT / Schrodinger-bridge
    way of recovering a coupling between two cluster marginals.  The didactic demo
    (``sinkhorn_demo.py``) shows the entropic coupling's churn is a *tunable choice*
    (the regularisation ``reg``): small ``reg`` -> minimal-transport, concentrated
    on retention (the lazy / least-action end); large ``reg`` -> maximally spread
    (the maximum-entropy default the plain min-norm Tikhonov solution sits at).
    So OT does not escape the prior choice -- it parametrises it.

  * temporal smoothing of the operator trajectory (brand-share style) -- realised
    by ``laziness.smooth_series_ewma`` (an L2 smoother).  Unlike the null-space
    variant it does not preserve the observed balance ``A.P~``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["logsumexp", "sinkhorn", "coupling_retention", "transport_cost"]


def logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis``."""
    M = np.asarray(M, dtype=np.float64)
    mx = np.max(M, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(M - mx), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float,
             n_iters: int = 2000, tol: float = 1e-9) -> np.ndarray:
    """Entropic-OT (Sinkhorn) coupling between marginals ``a``, ``b`` with cost ``C``.

    Returns the coupling ``P`` (``len(a) x len(b)``) minimising
    ``<P, C> - reg * H(P)`` with row sums ``a`` and column sums ``b``.  Log-domain
    iteration, so small ``reg`` (near-pure optimal transport) is numerically safe.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    f = np.zeros(a.shape[0])
    g = np.zeros(b.shape[0])
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    for _ in range(int(n_iters)):
        f_prev = f
        f = reg * (la - logsumexp((-C + g[None, :]) / reg, axis=1))
        g = reg * (lb - logsumexp((-C + f[:, None]) / reg, axis=0))
        if np.max(np.abs(f - f_prev)) < tol:
            break
    return np.exp((f[:, None] + g[None, :] - C) / reg)


def coupling_retention(P: np.ndarray) -> float:
    """Mean diagonal of the row-normalised coupling = implied self-retention."""
    P = np.asarray(P, dtype=np.float64)
    rs = P.sum(axis=1, keepdims=True)
    R = P / (rs + 1e-300)
    return float(np.mean(np.diag(R)))


def transport_cost(P: np.ndarray, C: np.ndarray) -> float:
    """Total transport cost ``<P, C>`` of a coupling."""
    return float(np.sum(np.asarray(P, float) * np.asarray(C, float)))
