"""Didactic Sinkhorn demo (paper Section 2.3 / 3.4): OT parametrises the prior.

Entropic optimal transport (Waddington-OT / Schrodinger-bridge style) recovers a
coupling between two cluster marginals.  Its churn level is NOT intrinsic to the
data -- it is set by the entropic regularisation ``reg``:

    small reg  -> minimal-transport coupling, concentrated on the diagonal
                  (high retention) = the lazy / least-action end;
    large reg  -> maximally spread coupling (low retention) = the maximum-entropy
                  default that the plain min-norm Tikhonov solution also sits at.

So optimal transport does not escape the prior decision that drives the recovered
micro-structure -- it merely parametrises it.  The least-action prior of this
paper picks the small-reg (minimal-transport) end, and -- crucially -- imposes it
*fit-exactly within the identifiability manifold* (see laziness.smooth_series_nullspace),
which OT/Sinkhorn does not.

Run:  python sinkhorn_demo.py   (numpy only; deterministic, seed=0)
"""

from __future__ import annotations

import numpy as np

from baselines import sinkhorn, coupling_retention, transport_cost


def run(n: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.5
    a = a / a.sum()
    b = a.copy()                       # near-closed system (matching marginals)
    C = 1.0 - np.eye(n)                # uniform off-diagonal cost; 0 to stay

    print(f"Entropic-OT coupling between matching cluster marginals (n={n}).")
    print("Retention = mean diagonal of the row-normalised coupling (inertia).\n")
    print(f"  {'reg':>8} {'retention':>10} {'transport_cost':>15}")
    for reg in (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        P = sinkhorn(a, b, C, reg=reg, n_iters=5000)
        print(f"  {reg:>8.2f} {coupling_retention(P):>10.3f} {transport_cost(P, C):>15.4f}")
    print("\n  small reg = minimal-transport (lazy, high retention);")
    print("  large reg = maximum entropy (spread, low retention = min-norm default).")
    print("  => OT parametrises the prior; it does not remove the choice.")


if __name__ == "__main__":
    run()
