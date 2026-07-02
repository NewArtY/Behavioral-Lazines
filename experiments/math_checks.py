"""Numerical verification of the non-identifiability dimension claims (paper Sec. 3).

Checks, for several n, the assertions of Lemma (rank) and Lemma (kernel structure):
  * the system matrix has full row rank p,
  * dim ker A = 2n^2 + n  (the balanced-cross-flow / cycle-space dimension),
  * the smallest non-zero singular value sigma_min^+ (conditioning on the manifold,
    Sec. 6) is reported.
Pure linear algebra on the dissertation's build_A; deterministic per seed.

Run (needs numpy + python_solver):  python math_checks.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "FindProbability"))

import python_solver as ps                                            # noqa: E402

_TIMING = HERE / "timing.log"


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("  Non-identifiability dimension checks (build_A, dissertation)")
    print("=" * 70)
    print(f"  {'n':>3} {'A shape':>14} {'rank':>6} {'dim ker':>8} "
          f"{'2n^2+n':>8} {'ok':>4} {'sigma_min+':>11}")
    all_ok = True
    for n in (2, 3, 5, 12, 20):
        n1 = rng.uniform(1.0, 10.0, n)
        n2 = rng.uniform(1.0, 10.0, n)
        A = ps.build_A(n, n1, n2, float(rng.uniform(0.1, 1.0)), float(rng.uniform(0.0, 0.5)))
        rank = int(np.linalg.matrix_rank(A, tol=1e-9))
        m = A.shape[1]
        dim_ker = m - rank
        expected = 2 * n * n + n
        sv = np.linalg.svd(A, compute_uv=False)
        smin = float(sv[rank - 1])                  # smallest non-zero singular value
        ok = (rank == A.shape[0]) and (dim_ker == expected)
        all_ok &= ok
        print(f"  {n:>3} {str(A.shape):>14} {rank:>6} {dim_ker:>8} {expected:>8} "
              f"{str(ok):>4} {smin:>11.3e}")
    print(f"\n  full row rank and dim ker A = 2n^2+n verified for all n: {all_ok}")
    dt = time.perf_counter() - t0
    print(f"\n[time] math checks: {dt:.2f} s")
    with open(_TIMING, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== math_checks {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        fh.write(f"math checks (rank, dim ker, sigma_min+): {dt:.2f} s; all_ok={all_ok}\n")


if __name__ == "__main__":
    main()
