"""Phase map of applicability (paper Section 5.3, Fig. 4).

Controlled synthetic study with KNOWN ground truth that delimits *where* the
least-action (laziness) prior helps and where it hurts -- i.e. the class of
systems the paper claims (inertial, weakly-forced).  Two axes:

  delta  -- systematic operator TREND per step (how fast the true RULE changes in
            a directed way).  Low delta = behaviourally inertial / weakly forced;
            high delta = strongly trending behaviour.
  rho    -- identification jitter: per-step random rotation of the balance matrix
            A(t), which turns the null space so the min-norm representative jitters
            in the UNOBSERVABLE subspace -- exactly the noise null-space laziness is
            built to remove (the real-data mechanism, A(t) changing year to year).
            A small fixed observation floor conditions the error ratio.

For each cell we average over ``n_seeds`` random systems the relative forecast
error of a smoother vs the raw min-norm recovery,
    ratio = ||A_fore @ p_pr(smoothed) - truth|| / ||A_fore @ p_pr(raw) - truth||,
plus the win rate (fraction of seeds with ratio < 1).  ratio < 1 / win rate > 0.5
means the prior helps.

Two distinct, defensible regimes emerge:
  * NULL-space laziness is ~delta-independent and never exceeds ratio 1 -- by
    construction it only touches the unobservable null(A) component, so it cannot
    bias the trend (carried in row(A)); it can only denoise.  SAFE everywhere,
    helps as sigma grows.
  * EWMA smooths the full space, so it helps more in absolute terms at low delta
    but BIASES the trend at high delta (win rate collapses).  RISKY: needs CV.

Run:  python phase_map.py   (numpy only; deterministic per seed; writes CSVs)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from laziness import smooth_series_nullspace, smooth_series_ewma

OUT = Path(__file__).resolve().parent / "experiments"


def _extrapolate_linear(X: np.ndarray) -> np.ndarray:
    k = X.shape[0]
    tau = np.arange(1, k + 1, dtype=float)
    M = np.vstack([np.ones_like(tau), tau]).T
    coef, *_ = np.linalg.lstsq(M, X, rcond=None)
    return coef[0] + coef[1] * (k + 1)


def one_run(seed, delta, rho, k=5, rows=6, cols=30, obs_floor=0.01):
    """Return (raw_err, null_err, ewma_err) for one synthetic system.

    True operator has a systematic linear trend of strength ``delta``.  The
    balance matrix A(t) is rotated by ``rho`` each step (turning the null space:
    the source of null-space jitter that laziness removes).  A small fixed
    ``obs_floor`` observation noise conditions the error ratio (no degenerate
    near-zero denominators) without masking the null-space mechanism.
    """
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((rows, cols))
    p0 = rng.standard_normal(cols)
    v = rng.standard_normal(cols); v /= np.linalg.norm(v)
    P_true = np.array([p0 + delta * t * v for t in range(k + 1)])
    A_list = [A0 + rho * rng.standard_normal((rows, cols)) for _ in range(k + 1)]
    X_rec = np.empty((k, cols))
    for t in range(k):
        b = A_list[t] @ P_true[t]
        b = b + obs_floor * np.linalg.norm(b) * rng.standard_normal(b.shape[0])
        X_rec[t] = np.linalg.pinv(A_list[t]) @ b
    A_fore = A_list[k]
    truth = A_fore @ P_true[k]

    def err(Xs):
        return float(np.linalg.norm(A_fore @ _extrapolate_linear(Xs) - truth))

    raw = err(X_rec)
    null = err(smooth_series_nullspace(X_rec, A_list[:k], 1.0))
    ewma = err(smooth_series_ewma(X_rec, 0.5))
    return raw, null, ewma


def phase_map(deltas, sigmas, n_seeds=60):
    """Return dict of grids: mean ratio + win rate for null and ewma."""
    nd, nr = len(deltas), len(sigmas)
    grids = {k: np.zeros((nd, nr)) for k in
             ("null_ratio", "null_win", "ewma_ratio", "ewma_win")}
    for i, d in enumerate(deltas):
        for j, r in enumerate(sigmas):
            n_r, e_r = [], []
            for s in range(n_seeds):
                raw, null, ewma = one_run(s, d, r)
                n_r.append(null / (raw + 1e-12))
                e_r.append(ewma / (raw + 1e-12))
            n_r, e_r = np.array(n_r), np.array(e_r)
            grids["null_ratio"][i, j] = n_r.mean()
            grids["null_win"][i, j] = float(np.mean(n_r < 1.0 - 1e-9))
            grids["ewma_ratio"][i, j] = e_r.mean()
            grids["ewma_win"][i, j] = float(np.mean(e_r < 1.0 - 1e-9))
    return grids


def _print_grid(name, G, deltas, rhos):
    print(f"\n  {name}  (rows: delta trend; cols: rho jitter)")
    print("    delta\\rho " + " ".join(f"{r:>7.3f}" for r in rhos))
    for i, d in enumerate(deltas):
        print(f"    {d:>9.3f} " + " ".join(f"{G[i,j]:>7.3f}" for j in range(len(rhos))))


def _save_csv(name, G, deltas, rhos):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"phase_{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["delta\\rho"] + [f"{r:g}" for r in rhos])
        for i, d in enumerate(deltas):
            w.writerow([f"{d:g}"] + [f"{G[i,j]:.4f}" for j in range(len(rhos))])


def main(n_seeds=60):
    import time
    t0 = time.perf_counter()
    deltas = [0.0, 0.02, 0.05, 0.10, 0.20]
    sigmas = [0.0, 0.02, 0.05, 0.10, 0.20]   # rho: A(t) rotation magnitude
    grids = phase_map(deltas, sigmas, n_seeds=n_seeds)
    print(f"Phase map of applicability ({n_seeds} seeds/cell).")
    print("ratio<1 or win>0.5 => the prior helps; >1 => it hurts.")
    _print_grid("NULL-space laziness: mean error ratio (null/raw)", grids["null_ratio"], deltas, sigmas)
    _print_grid("NULL-space laziness: win rate (frac seeds ratio<1)", grids["null_win"], deltas, sigmas)
    _print_grid("EWMA smoothing: mean error ratio (ewma/raw)", grids["ewma_ratio"], deltas, sigmas)
    _print_grid("EWMA smoothing: win rate", grids["ewma_win"], deltas, sigmas)
    for name in grids:
        _save_csv(name, grids[name], deltas, sigmas)
    dt = time.perf_counter() - t0
    print(f"\n[time] phase_map: {dt:.1f} s   (CSVs in experiments/)")
    with open(OUT / "timing.log", "a", encoding="utf-8") as fh:
        fh.write(f"phase_map ({n_seeds} seeds): {dt:.1f} s\n")


if __name__ == "__main__":
    main()
