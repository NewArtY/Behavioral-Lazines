"""Baseline comparison: least-action laziness vs temporal smoothing vs OT.

Answers the reviewer question "why is this better than optimal transport /
temporal smoothing?" (LITERATURE.md competitors: Waddington-OT, brand-share).

Part A -- forecast comparison (rolling-origin, both domains): the SAME recovered
operator series, regularised four ways before extrapolation:
    raw         min-norm (max-entropy default)
    ewma:0.5    temporal smoothing (brand-share style, L2)         -- residual > 0
    null:0.5    least-action, fit-exact in null(A)  (ours)         -- residual ~ 0
    null:1      least-action, full pull                            -- residual ~ 0
For each we report MAE / reliab / dir_acc AND the balance residual
||A.Xs - A.X|| / ||A.X|| (how much the smoother corrupts the observed increment)
and the trajectory-energy ratio.  Headline: only the null-space variant buys the
smoothing benefit at zero balance cost.

Part B -- Sinkhorn didactic demo (see code/sinkhorn_demo.py for the standalone
version): the entropic-OT coupling's churn is a tunable choice (reg), so OT does
not escape the prior decision.

ASCII output (Windows console).  Run:
    python -m python_forecast.scripts.explore_baselines
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import _bootstrap  # noqa: F401

from .explore_laziness import load_periods, SKW, SCHEMES
from ..core.series import segment_series
from ..core.matrices import build_A_phys
from ..core.laziness import smooth_series_nullspace, smooth_series_ewma, trajectory_energy
from ..core.baselines import sinkhorn, coupling_retention, transport_cost
from ..core.pipeline import forecast
from ..core import direct, metrics

_TIMING = Path(__file__).resolve().parents[3] / "2026.06.20_МультиКластер" / "code" / "experiments" / "timing.log"

VARIANTS = [
    ("raw", None),
    ("ewma:0.5", lambda X, A: smooth_series_ewma(X, 0.5)),
    ("null:0.5", lambda X, A: smooth_series_nullspace(X, A, 0.5)),
    ("null:1", lambda X, A: smooth_series_nullspace(X, A, 1.0)),
]


def _log_timing(msg):
    _TIMING.parent.mkdir(parents=True, exist_ok=True)
    with open(_TIMING, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def _balance_residual(Xs, X, A_list):
    """Mean over intervals of ||A_i Xs[i] - A_i X[i]|| / ||A_i X[i]||."""
    vals = []
    for i in range(X.shape[0]):
        ref = A_list[i] @ X[i]
        vals.append(float(np.linalg.norm(A_list[i] @ Xs[i] - ref)
                          / (np.linalg.norm(ref) + 1e-12)))
    return float(np.mean(vals))


def rolling_baselines(domain, min_base=3):
    n, periods = load_periods(domain)
    skill = {(s, v): [] for s in SCHEMES for v, _ in VARIANTS}
    resid = {v: [] for v, _ in VARIANTS}
    energy = {v: [] for v, _ in VARIANTS}
    naive_acc = []
    n_hold = 0

    for h in range(min_base, len(periods)):
        base, actual = periods[:h], periods[h]
        prev = base[-1]
        n_hold += 1
        series = segment_series(n, base, **SKW)
        X = series.X
        A_list = [build_A_phys(n, base[i]) for i in range(X.shape[0])]
        e0 = trajectory_energy(X)

        nv = direct.naive(base)
        rep = metrics.error_report(nv, actual, n, prev=prev)
        naive_acc.append((rep["mae"], rep["reliab_share"], rep["dir_acc"]))

        for vname, vfn in VARIANTS:
            Xs = X if vfn is None else vfn(X, A_list)
            resid[vname].append(_balance_residual(Xs, X, A_list))
            energy[vname].append(trajectory_energy(Xs) / (e0 + 1e-12))
            series_s = replace(series, X=Xs)
            for s in SCHEMES:
                res = forecast(n, base, scheme=s, series=series_s, solver_kwargs=SKW)
                rep = metrics.error_report(res.n_forecast, actual, n, prev=prev)
                skill[(s, vname)].append((rep["mae"], rep["reliab_share"], rep["dir_acc"]))

    nv = np.array(naive_acc).mean(axis=0)
    print(f"\n{'='*78}\nDOMAIN {domain}: baselines, rolling-origin MEAN over {n_hold} holdouts"
          f"\n{'='*78}")
    print(f"  naive (anchor): MAE={nv[0]:.2f}  reliab={nv[1]:.3f}  dir_acc={nv[2]:.3f}")
    print(f"\n  trajectory regulariser -- balance corruption & energy (scheme-independent):")
    print(f"    {'variant':>10} {'balance_resid':>14} {'energy/e0':>10}")
    for vname, _ in VARIANTS:
        print(f"    {vname:>10} {np.mean(resid[vname]):>14.2e} {np.mean(energy[vname]):>10.3f}")
    for s in SCHEMES:
        print(f"\n  scheme: {s}")
        print(f"    {'variant':>10} {'MAE':>8} {'reliab':>8} {'dir_acc':>8}")
        for vname, _ in VARIANTS:
            m = np.array(skill[(s, vname)]).mean(axis=0)
            print(f"    {vname:>10} {m[0]:>8.2f} {m[1]:>8.3f} {m[2]:>8.3f}")


def sinkhorn_demo(n=6, seed=0):
    """Entropic-OT coupling churn vs regularisation: OT parametrises the prior."""
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.5
    a = a / a.sum()
    b = a.copy()                       # near-closed (matching marginals)
    C = 1.0 - np.eye(n)                # uniform off-diagonal cost; 0 to stay
    print(f"\n[Sinkhorn] entropic-OT coupling between matching cluster marginals (n={n})")
    print(f"  {'reg':>8} {'retention':>10} {'transport_cost':>15}")
    for reg in (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        P = sinkhorn(a, b, C, reg=reg, n_iters=5000)
        print(f"  {reg:>8.2f} {coupling_retention(P):>10.3f} {transport_cost(P, C):>15.4f}")
    print("  => small reg = minimal-transport (lazy, high retention); large reg =")
    print("     maximum entropy (spread, low retention = the min-norm default).")


def main():
    t_all = time.perf_counter()
    _log_timing(f"\n=== explore_baselines run {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    for domain in ("okved1", "okved2"):
        t0 = time.perf_counter()
        rolling_baselines(domain)
        dt = time.perf_counter() - t0
        print(f"\n  [time] {domain}: {dt:.1f} s")
        _log_timing(f"rolling_baselines {domain}: {dt:.1f} s")
    sinkhorn_demo()
    dt = time.perf_counter() - t_all
    print(f"\n[time] total: {dt:.1f} s")
    _log_timing(f"total: {dt:.1f} s")


if __name__ == "__main__":
    main()
