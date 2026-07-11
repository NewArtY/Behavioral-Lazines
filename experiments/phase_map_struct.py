# -*- coding: utf-8 -*-
"""Structure-preserving applicability map (reviewer/author request).

The published `phase_map.py` uses an ABSTRACT random forward matrix A0 and
rotates it directly (A(t)=A0+rho*noise) -- which leaves the class of valid balance
matrices.  Here rho is instead OBSERVATION NOISE on the levels N(t): every A(t) is
rebuilt from noisy levels via `build_A_phys` and therefore stays a *valid balance
matrix*.  Pipeline (author's proposal):

  known P_true(t) (operator, with linear drift delta) and a level trajectory N(t);
  perturb the OBSERVED levels by rho -> N_obs(t); build the valid balance block
  A_obs(t)=build_A_phys(N_obs(t)); RHS b=A_obs@P_true (+small floor); recover the
  min-norm operator X_rec(t)=pinv(A_obs)@b.  As N jitters, ker(A_obs) rotates and
  the min-norm representative random-walks in the null space even at delta=0.

Then: extrapolate the operator series, forecast the increment with the clean
forecast matrix, compare raw / null-space-smoothed / ewma.  Grid over (delta,rho),
averaged over seeds.  Question: do the map's qualitative conclusions survive?
  - null-space smoothing safe everywhere (ratio<=1), delta-independent;
  - ewma has a harmful zone (ratio>1) at strong delta / weak rho.

Run:  python phase_map_struct.py            (needs FindProbability/.venv)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FINDPROB = HERE.parents[2] / "FindProbability"
sys.path.insert(0, str(FINDPROB))

from python_forecast import _bootstrap  # noqa: F401
from python_forecast.core.matrices import build_A_phys, m_phys
from python_forecast.core.laziness import smooth_series_nullspace, smooth_series_ewma
from python_solver import PeriodData


def _extrapolate_linear(X):
    k = X.shape[0]
    tau = np.arange(1, k + 1, dtype=float)
    M = np.vstack([np.ones_like(tau), tau]).T
    coef, *_ = np.linalg.lstsq(M, X, rcond=None)
    return coef[0] + coef[1] * (k + 1)


def _period(n1, n2, n0):
    return PeriodData(n1=n1, n2=n2, n0=float(n0), delta_n0=0.0, label="")


def one_run(seed, delta, rho, n=4, k=5, obs_floor=0.01):
    """Structure-preserving analogue of phase_map.one_run (valid balance A)."""
    rng = np.random.default_rng(seed)
    m = m_phys(n)
    # true operator series with a systematic linear drift of strength delta
    P0 = rng.random(m)
    v = rng.standard_normal(m); v /= np.linalg.norm(v)
    P_true = [np.clip(P0 + delta * t * v, 0.0, None) for t in range(k + 1)]
    # a smooth positive level trajectory (clean); A(t) is built from these levels
    b1 = rng.random(n) + 0.5
    b2 = rng.random(n) + 0.5

    def clean_levels(t):
        g = 1.0 + 0.05 * t                       # mild secular level drift
        return b1 * g, b2 * g, 1.0

    X_rec = np.empty((k, m))
    A_list = []
    for t in range(k):
        c1, c2, c0 = clean_levels(t)
        # OBSERVATION NOISE rho on the levels -> still a valid balance matrix
        n1o = np.clip(c1 * (1.0 + rho * rng.standard_normal(n)), 1e-6, None)
        n2o = np.clip(c2 * (1.0 + rho * rng.standard_normal(n)), 1e-6, None)
        n0o = max(c0 * (1.0 + rho * rng.standard_normal()), 1e-6)
        A = build_A_phys(n, _period(n1o, n2o, n0o))
        b = A @ P_true[t]
        b = b + obs_floor * np.linalg.norm(b) * rng.standard_normal(b.shape[0])
        X_rec[t] = np.linalg.pinv(A) @ b
        A_list.append(A)
    c1, c2, c0 = clean_levels(k)
    A_fore = build_A_phys(n, _period(c1, c2, c0))     # clean forecast matrix
    truth = A_fore @ P_true[k]

    def err(Xs):
        return float(np.linalg.norm(A_fore @ _extrapolate_linear(Xs) - truth))

    raw = err(X_rec)
    null = err(smooth_series_nullspace(X_rec, A_list, 1.0))
    ewma = err(smooth_series_ewma(X_rec, 0.5))
    return raw, null, ewma


def phase(deltas, rhos, n_seeds=60, n=4):
    nd, nr = len(deltas), len(rhos)
    g = {k: np.zeros((nd, nr)) for k in ("null_ratio", "null_win", "ewma_ratio", "ewma_win")}
    for i, d in enumerate(deltas):
        for j, r in enumerate(rhos):
            nr_, ew_ = [], []
            for s in range(n_seeds):
                raw, nu, ew = one_run(s, d, r, n=n)
                nr_.append(nu / (raw + 1e-12)); ew_.append(ew / (raw + 1e-12))
            nr_ = np.array(nr_); ew_ = np.array(ew_)
            g["null_ratio"][i, j] = nr_.mean(); g["null_win"][i, j] = np.mean(nr_ < 1 - 1e-9)
            g["ewma_ratio"][i, j] = ew_.mean(); g["ewma_win"][i, j] = np.mean(ew_ < 1 - 1e-9)
    return g


def _grid(name, G, deltas, rhos):
    print(f"\n  {name}  (rows: delta trend; cols: rho jitter)")
    print("    delta\\rho " + " ".join(f"{r:>7.3f}" for r in rhos))
    for i, d in enumerate(deltas):
        print(f"    {d:>9.3f} " + " ".join(f"{G[i, j]:>7.3f}" for j in range(len(rhos))))


def _save_csv(name, G, deltas, rhos):
    """Write a grid CSV in the format read by make_figures.ris4 (canonical figure
    data: this structure-preserving map now supersedes phase_map.py for Fig. 4)."""
    import csv
    with open(HERE / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["delta\\rho"] + [f"{r:g}" for r in rhos])
        for i, d in enumerate(deltas):
            w.writerow([f"{d:g}"] + [f"{G[i, j]:.4f}" for j in range(len(rhos))])


def main():
    deltas = [0.0, 0.02, 0.05, 0.10, 0.20]
    rhos = [0.0, 0.02, 0.05, 0.10, 0.20]
    print("=" * 74)
    print("  STRUCTURE-PRESERVING applicability map (valid balance A, n=4);")
    print("  rho = observation noise on levels N (not abstract matrix rotation)")
    print("=" * 74)
    g = phase(deltas, rhos, n_seeds=60, n=4)
    _grid("NULL-space: mean error ratio (null/raw)  [<1 = helps]", g["null_ratio"], deltas, rhos)
    _grid("NULL-space: win rate (frac seeds ratio<1)", g["null_win"], deltas, rhos)
    _grid("EWMA: mean error ratio (ewma/raw)  [>1 = HARMS]", g["ewma_ratio"], deltas, rhos)
    _grid("EWMA: win rate", g["ewma_win"], deltas, rhos)
    for nm in ("null_ratio", "null_win", "ewma_ratio", "ewma_win"):
        _save_csv(f"phase_{nm}", g[nm], deltas, rhos)
    print("\n  wrote phase_{null,ewma}_{ratio,win}.csv (Fig.4 source)")
    print(f"  ewma worst (max ratio) = {g['ewma_ratio'].max():.3f}")
    print("  Check: null<=1 everywhere & ~flat in delta?  ewma>1 at high delta/low rho?")


if __name__ == "__main__":
    main()
