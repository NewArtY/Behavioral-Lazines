"""Unit tests for ``baselines`` (entropic-OT / Sinkhorn).

Encodes the paper's Fig.1 (``ris1``) mechanism: the entropic coupling's churn is
a tunable function of ``reg`` -- small ``reg`` -> minimal transport, concentrated
on retention (the lazy end); large ``reg`` -> maximum-entropy spread.  Also
checks the log-domain marginal constraints and the log-sum-exp helper.
"""
from __future__ import annotations

import numpy as np
import pytest

from baselines import sinkhorn, coupling_retention, transport_cost, logsumexp


def _simplex(rng, n):
    a = rng.random(n) + 0.5
    return a / a.sum()


# --------------------------------------------------------------------------- #
# logsumexp
# --------------------------------------------------------------------------- #
def test_logsumexp_matches_naive():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 5))
    assert np.allclose(logsumexp(M, axis=1), np.log(np.sum(np.exp(M), axis=1)))
    assert np.allclose(logsumexp(M, axis=0), np.log(np.sum(np.exp(M), axis=0)))


def test_logsumexp_stable_on_large_input():
    # naive exp overflows; the stable version must not.
    M = np.array([[1000.0, 1001.0, 1002.0]])
    got = logsumexp(M, axis=1)
    ref = 1000.0 + np.log(np.sum(np.exp(M - 1000.0), axis=1))
    assert np.allclose(got, ref)
    assert np.isfinite(got).all()


# --------------------------------------------------------------------------- #
# sinkhorn: marginal constraints
# --------------------------------------------------------------------------- #
def test_sinkhorn_matches_marginals():
    rng = np.random.default_rng(1)
    a = _simplex(rng, 5)
    b = _simplex(rng, 5)
    C = rng.random((5, 5))
    P = sinkhorn(a, b, C, reg=0.1, n_iters=5000, tol=1e-12)
    assert np.allclose(P.sum(axis=1), a, atol=1e-6)
    assert np.allclose(P.sum(axis=0), b, atol=1e-6)


def test_sinkhorn_low_reg_lower_transport_cost():
    # smaller reg => closer to true optimal transport => lower <P,C>.
    rng = np.random.default_rng(2)
    a = _simplex(rng, 6)
    b = a.copy()
    C = 1.0 - np.eye(6)
    cost_hi = transport_cost(sinkhorn(a, b, C, reg=2.0, n_iters=5000), C)
    cost_lo = transport_cost(sinkhorn(a, b, C, reg=0.05, n_iters=5000), C)
    assert cost_lo < cost_hi


def test_sinkhorn_cost_monotone_decreasing_in_reg():
    # transport cost decreases monotonically as reg -> 0 across a grid.
    rng = np.random.default_rng(3)
    a = _simplex(rng, 6)
    b = a.copy()
    C = 1.0 - np.eye(6)
    regs = [2.0, 1.0, 0.5, 0.2, 0.05]                     # descending reg
    costs = [transport_cost(sinkhorn(a, b, C, reg=r, n_iters=6000), C)
             for r in regs]
    assert all(costs[i + 1] < costs[i] for i in range(len(costs) - 1))


# --------------------------------------------------------------------------- #
# coupling_retention: monotone in reg (Fig.1 / ris1)
# --------------------------------------------------------------------------- #
def test_retention_increases_as_reg_drops():
    rng = np.random.default_rng(4)
    a = _simplex(rng, 6)
    b = a.copy()
    C = 1.0 - np.eye(6)
    r_hi = coupling_retention(sinkhorn(a, b, C, reg=2.0, n_iters=5000))
    r_lo = coupling_retention(sinkhorn(a, b, C, reg=0.05, n_iters=5000))
    assert r_lo > r_hi


def test_retention_monotone_over_reg_grid():
    # increasing reg strictly lowers implied self-retention (inertia).
    rng = np.random.default_rng(5)
    a = _simplex(rng, 6)
    b = a.copy()
    C = 1.0 - np.eye(6)
    regs = [0.05, 0.2, 0.5, 1.0, 2.0]                     # ascending reg
    ret = [coupling_retention(sinkhorn(a, b, C, reg=r, n_iters=6000))
           for r in regs]
    assert all(ret[i + 1] < ret[i] for i in range(len(ret) - 1))


# --------------------------------------------------------------------------- #
# transport_cost helper
# --------------------------------------------------------------------------- #
def test_transport_cost_frobenius_inner_product():
    P = np.array([[0.1, 0.2], [0.3, 0.4]])
    C = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert transport_cost(P, C) == pytest.approx(np.sum(P * C))


def test_coupling_retention_identity_is_one():
    # a purely diagonal coupling implies full self-retention.
    P = np.diag([0.2, 0.3, 0.5])
    assert coupling_retention(P) == pytest.approx(1.0)
