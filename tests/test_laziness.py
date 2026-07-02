"""Unit tests for ``laziness`` (causal operator-trajectory smoothing).

Encodes the mathematical invariants stated in the paper (``article/main.tex``):
null-space projection is an orthogonal projector that preserves the observed
macro balance unconditionally (even for rank-deficient A, thanks to the
pseudoinverse); ``smooth_series_nullspace`` is increment-preserving, causal and
non-energy-increasing; ``smooth_series_ewma`` corrupts the balance (the ~7% the
paper contrasts against null-space's ~1e-15).
"""
from __future__ import annotations

import numpy as np
import pytest

from laziness import (
    null_space_project, smooth_series_nullspace, smooth_series_ewma,
    trajectory_energy,
)


def _random_underdetermined(rng, rows=5, cols=20, k=4):
    """A wide matrix (null space nonempty) and a feasible operator series."""
    A = rng.standard_normal((rows, cols))
    X = rng.standard_normal((k, cols))
    A_list = [A.copy() for _ in range(k)]   # same A per interval (simplest)
    return A, X, A_list


def _random_drifting(rng, rows=4, cols=16, k=5):
    """A per-interval balance matrix that drifts, plus a feasible series."""
    A_list = [rng.standard_normal((rows, cols)) for _ in range(k)]
    X = rng.standard_normal((k, cols))
    return X, A_list


# --------------------------------------------------------------------------- #
# null_space_project: orthogonal projector onto ker(A)
# --------------------------------------------------------------------------- #
def test_null_space_project_lands_in_kernel():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 20))
    v = rng.standard_normal(20)
    w = null_space_project(A, v)
    assert np.linalg.norm(A @ w) < 1e-12          # A . Pi(v) = 0 exactly


def test_null_space_project_idempotent():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 20))
    v = rng.standard_normal(20)
    w = null_space_project(A, v)
    w2 = null_space_project(A, w)                  # Pi(Pi(v)) = Pi(v)
    assert np.allclose(w, w2, atol=1e-10)


def test_null_space_project_symmetric_operator():
    # Build the operator matrix column by column and check it is symmetric,
    # i.e. Pi is an *orthogonal* (self-adjoint) projector.
    rng = np.random.default_rng(2)
    m = 12
    A = rng.standard_normal((4, m))
    I = np.eye(m)
    M = np.column_stack([null_space_project(A, I[:, j]) for j in range(m)])
    assert np.allclose(M, M.T, atol=1e-10)


def test_null_space_project_rank_deficient_A_preserves_balance():
    # An all-zero row (e.g. an industry with zero population) makes A A^T
    # singular; the pseudoinverse still yields A . Pi(v) = 0 unconditionally.
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 20))
    A[2, :] = 0.0                                  # rank-deficient row
    assert np.linalg.matrix_rank(A) < A.shape[0]
    v = rng.standard_normal(20)
    w = null_space_project(A, v)
    assert np.linalg.norm(A @ w) < 1e-12


def test_null_space_project_precomputed_inv_matches():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((5, 20))
    v = rng.standard_normal(20)
    AAt_inv = np.linalg.pinv(A @ A.T)
    assert np.allclose(null_space_project(A, v),
                       null_space_project(A, v, AAt_inv), atol=1e-12)


# --------------------------------------------------------------------------- #
# smooth_series_nullspace: increment-preserving, causal, lazy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_nullspace_smoothing_preserves_increment(lam):
    rng = np.random.default_rng(5)
    _, X, A_list = _random_underdetermined(rng)
    b = [A_list[i] @ X[i] for i in range(X.shape[0])]
    Xs = smooth_series_nullspace(X, A_list, lam)
    for i in range(X.shape[0]):
        assert np.linalg.norm(A_list[i] @ Xs[i] - b[i]) < 1e-13


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_nullspace_preserves_increment_drifting_A(lam):
    rng = np.random.default_rng(6)
    X, A_list = _random_drifting(rng)
    b = [A_list[i] @ X[i] for i in range(X.shape[0])]
    Xs = smooth_series_nullspace(X, A_list, lam)
    for i in range(X.shape[0]):
        assert np.linalg.norm(A_list[i] @ Xs[i] - b[i]) < 1e-13


def test_nullspace_lam_zero_is_identity():
    rng = np.random.default_rng(7)
    _, X, A_list = _random_underdetermined(rng)
    Xs = smooth_series_nullspace(X, A_list, 0.0)
    assert np.allclose(Xs, X)


def test_nullspace_lam_one_diff_lives_in_row_space():
    # With lam=1, row i is the feasible projection of the previous smoothed row
    # onto {A x = A X[i]}; the residual Xs[i]-Xs[i-1] then lies in row(A), i.e.
    # its null-space projection is 0.
    rng = np.random.default_rng(8)
    _, X, A_list = _random_underdetermined(rng)
    Xs = smooth_series_nullspace(X, A_list, 1.0)
    for i in range(1, X.shape[0]):
        diff = Xs[i] - Xs[i - 1]
        assert np.linalg.norm(null_space_project(A_list[i], diff)) < 1e-7


def test_nullspace_is_causal():
    # Perturbing a future row must not change earlier smoothed rows.
    rng = np.random.default_rng(9)
    _, X, A_list = _random_underdetermined(rng, k=5)
    Xs = smooth_series_nullspace(X, A_list, 0.6)
    X2 = X.copy()
    X2[4] += 10.0 * rng.standard_normal(X.shape[1])   # change last row only
    Xs2 = smooth_series_nullspace(X2, A_list, 0.6)
    assert np.allclose(Xs[:4], Xs2[:4], atol=1e-12)


def test_nullspace_does_not_increase_energy():
    rng = np.random.default_rng(10)
    _, X, A_list = _random_underdetermined(rng, k=6)
    e0 = trajectory_energy(X)
    for lam in (0.25, 0.5, 1.0):
        assert trajectory_energy(smooth_series_nullspace(X, A_list, lam)) <= e0 + 1e-9


def test_nullspace_anchor_pulls_first_row_within_kernel():
    rng = np.random.default_rng(11)
    _, X, A_list = _random_underdetermined(rng)
    anchor = rng.standard_normal(X.shape[1])
    b0 = A_list[0] @ X[0]
    Xs = smooth_series_nullspace(X, A_list, 0.5, anchor=anchor, lam_anchor=0.3)
    # first row moved but macro balance of row 0 preserved
    assert not np.allclose(Xs[0], X[0])
    assert np.linalg.norm(A_list[0] @ Xs[0] - b0) < 1e-13


# --------------------------------------------------------------------------- #
# smooth_series_ewma: soft inertia, does NOT preserve balance
# --------------------------------------------------------------------------- #
def test_ewma_endpoints():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((4, 7))
    assert np.allclose(smooth_series_ewma(X, 0.0), X)         # raw
    frozen = smooth_series_ewma(X, 1.0)
    for i in range(X.shape[0]):
        assert np.allclose(frozen[i], X[0])                  # all == first


def test_ewma_corrupts_balance_vs_nullspace():
    # The article's central contrast: null-space keeps A.P~ to ~1e-15 while the
    # soft L2 smoother introduces a strictly positive balance residual.
    rng = np.random.default_rng(13)
    X, A_list = _random_drifting(rng)
    lam = 0.5
    Xs_null = smooth_series_nullspace(X, A_list, lam)
    Xs_ewma = smooth_series_ewma(X, lam)

    def balance_residual(Xs):
        return sum(np.linalg.norm(A_list[i] @ Xs[i] - A_list[i] @ X[i])
                   for i in range(X.shape[0]))

    res_null = balance_residual(Xs_null)
    res_ewma = balance_residual(Xs_ewma)
    assert res_null < 1e-12                       # machine-precision balance
    assert res_ewma > 1e-3                        # ewma corrupts the balance
    assert res_ewma > 1e6 * res_null              # qualitative gap


# --------------------------------------------------------------------------- #
# trajectory_energy
# --------------------------------------------------------------------------- #
def test_trajectory_energy_short_series_is_zero():
    assert trajectory_energy(np.zeros((0, 3))) == 0.0
    assert trajectory_energy(np.ones((1, 3))) == 0.0


def test_trajectory_energy_hand_computed():
    # rows: [0,0], [3,4], [3,4] -> ||(3,4)||^2 + ||(0,0)||^2 = 25 + 0 = 25
    X = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    assert trajectory_energy(X) == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", [-0.1, 1.1, 2.0])
def test_nullspace_rejects_bad_lam(lam):
    rng = np.random.default_rng(14)
    _, X, A_list = _random_underdetermined(rng)
    with pytest.raises(ValueError):
        smooth_series_nullspace(X, A_list, lam)


@pytest.mark.parametrize("lam", [-0.1, 1.1])
def test_ewma_rejects_bad_lam(lam):
    rng = np.random.default_rng(15)
    X = rng.standard_normal((4, 7))
    with pytest.raises(ValueError):
        smooth_series_ewma(X, lam)


def test_nullspace_rejects_length_mismatch():
    rng = np.random.default_rng(16)
    _, X, A_list = _random_underdetermined(rng, k=4)
    with pytest.raises(ValueError):
        smooth_series_nullspace(X, A_list[:2], 0.5)   # wrong number of matrices
