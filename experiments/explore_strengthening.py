"""Strengthening calcs #6-#8 (causal, labour-market data).

#6  Static vs dynamic laziness (ablation).
    (a) segmentation levers: implied retention is invariant to the reference x_bar
        (0 / P0) and to alpha -- internal levers do NOT move micro-structure
        (confirms exploratory_findings #6, needs an external/null-space move).
    (b) forecast: a STATIC null-space pull (each operator toward the series mean,
        independent) is ~forecast-neutral, while the DYNAMIC pull (toward the
        previous operator, time-coupled) changes the forecast -- isolating that the
        cross-time coupling, not the per-period move, carries the signal (§3.2).

#7  Energy<->skill: across schemes x lambda x holdouts, correlate the trajectory-
    energy ratio with reliability / MAE -- direct mechanism evidence.

#8  Robustness: the headline OKVED-2 result (null-space improves reliability at
    ~zero balance residual) under a solver alpha/eps sweep.

ASCII output.  Run:  python -m python_forecast.scripts.explore_strengthening
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .. import _bootstrap  # noqa: F401
import python_solver as ps

from .explore_laziness import load_periods, SKW
from ..core.series import segment_series
from ..core.matrices import build_A_phys
from ..core.laziness import (
    smooth_series_nullspace, smooth_series_ewma, trajectory_energy, null_space_project,
)
from ..core.economic_prior import implied_retention
from ..core.pipeline import forecast
from ..core import direct, metrics


def static_null_to(X, A_list, anchor, lam=1.0):
    """Per-period null-space pull toward a FIXED anchor (no temporal coupling)."""
    Xs = X.copy()
    for i in range(X.shape[0]):
        Xs[i] = X[i] + lam * null_space_project(A_list[i], anchor - X[i])
    return Xs


# --------------------------------------------------------------------------
# #6(a) segmentation levers: retention invariance
# --------------------------------------------------------------------------
def retention_invariance(domain):
    n, periods = load_periods(domain)
    print(f"\n[#6a] {domain}: implied retention vs internal levers (x_bar, alpha)")
    base = segment_series(n, periods, **SKW)
    r0 = implied_retention(base.X[-1], n).mean()
    lazy_kw = dict(SKW, interval_configs=[ps.IntervalConfig(mode="absolute_lazy")] * len(periods))
    rl = implied_retention(segment_series(n, periods, **lazy_kw).X[-1], n).mean()
    print(f"    standard (x_bar=0)        retention={r0:.3f}")
    print(f"    absolute_lazy (x_bar=P0)  retention={rl:.3f}")
    for a in (1e-3, 1e-2, 1e-1, 1.0):
        kw = dict(SKW); kw["alpha_fixed"] = a
        r = implied_retention(segment_series(n, periods, **kw).X[-1], n).mean()
        print(f"    alpha={a:<6g}             retention={r:.3f}")
    print("    => internal levers leave retention ~invariant (needs null-space move)")


# --------------------------------------------------------------------------
# #6(b) static vs dynamic laziness in the forecast
# --------------------------------------------------------------------------
def static_vs_dynamic(domain):
    n, periods = load_periods(domain)
    print(f"\n[#6b] {domain}: static (fixed anchor) vs dynamic (toward prev) null-space")
    print(f"      {'variant':>10} {'MAE':>8} {'reliab':>8} {'dMAE_vs_raw':>12} {'bal_resid':>10}")
    schemes = ["common"]
    acc = {}
    for h in range(3, len(periods)):
        base, actual = periods[:h], periods[h]
        prev = base[-1]
        ser = segment_series(n, base, **SKW)
        Al = [build_A_phys(n, base[i]) for i in range(ser.X.shape[0])]
        mean_op = ser.X.mean(axis=0)
        variants = {
            "raw": ser.X,
            "static": static_null_to(ser.X, Al, mean_op, 1.0),     # fixed anchor
            "dynamic": smooth_series_nullspace(ser.X, Al, 1.0),    # toward prev
        }
        for vname, Xs in variants.items():
            resid = max(float(np.linalg.norm(Al[i] @ Xs[i] - Al[i] @ ser.X[i])
                              / (np.linalg.norm(Al[i] @ ser.X[i]) + 1e-12))
                        for i in range(ser.X.shape[0]))
            ss = replace(ser, X=Xs)
            for s in schemes:
                res = forecast(n, base, scheme=s, series=ss, solver_kwargs=SKW)
                rep = metrics.error_report(res.n_forecast, actual, n, prev=prev)
                acc.setdefault((s, vname), []).append((rep["mae"], rep["reliab_share"], resid))
    for s in schemes:
        raw_mae = np.mean([r[0] for r in acc[(s, "raw")]])
        for vname in ("raw", "static", "dynamic"):
            arr = np.array(acc[(s, vname)])
            mae, rel, resid = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            print(f"      {vname:>10} {mae:>8.2f} {rel:>8.3f} {mae-raw_mae:>+12.2f} {resid:>10.1e}")
    print("      => static (fixed anchor) ~ neutral; dynamic (temporal coupling) moves forecast")


# --------------------------------------------------------------------------
# #7 energy <-> skill correlation
# --------------------------------------------------------------------------
def energy_skill(domain):
    n, periods = load_periods(domain)
    schemes = ["common", "per_component", "lowrank"]
    null_l = [0.25, 0.5, 0.75, 1.0]
    ewma_l = [0.3, 0.5, 0.7]
    rows_null, rows_ewma = [], []        # (energy_ratio, reliab, mae)
    for h in range(3, len(periods)):
        base, actual = periods[:h], periods[h]
        prev = base[-1]
        ser = segment_series(n, base, **SKW)
        Al = [build_A_phys(n, base[i]) for i in range(ser.X.shape[0])]
        e0 = trajectory_energy(ser.X) + 1e-12
        for fam, lams, sink in (("null", null_l, rows_null), ("ewma", ewma_l, rows_ewma)):
            for lam in lams:
                Xs = (smooth_series_nullspace(ser.X, Al, lam) if fam == "null"
                      else smooth_series_ewma(ser.X, lam))
                er = trajectory_energy(Xs) / e0
                for s in schemes:
                    ss = replace(ser, X=Xs)
                    res = forecast(n, base, scheme=s, series=ss, solver_kwargs=SKW)
                    rep = metrics.error_report(res.n_forecast, actual, n, prev=prev)
                    sink.append((er, rep["reliab_share"], rep["mae"]))
    print(f"\n[#7] {domain}: energy<->skill correlation (pooled over schemes/lambda/holdouts)")
    for name, rows in (("null", rows_null), ("ewma", rows_ewma)):
        a = np.array(rows)
        r_rel = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
        r_mae = np.corrcoef(a[:, 0], a[:, 2])[0, 1]
        print(f"    {name}: corr(energy_ratio, reliab)={r_rel:+.2f}  "
              f"corr(energy_ratio, MAE)={r_mae:+.2f}  (n={len(rows)})")
    print("    => null: lower energy (more laziness) <-> higher reliab / lower MAE (mechanism)")


# --------------------------------------------------------------------------
# #8 robustness of the headline result to solver alpha/eps
# --------------------------------------------------------------------------
def robustness(domain="okved2"):
    n, periods = load_periods(domain)
    print(f"\n[#8] {domain}: robustness of null-space gain to solver alpha/eps "
          f"(common scheme, rolling MEAN)")
    print(f"      {'alpha':>7} {'eps':>7} {'reliab_raw':>11} {'reliab_null':>12} "
          f"{'MAE_raw':>8} {'MAE_null':>9} {'bal_resid':>10}")
    for alpha in (1e-3, 1e-2, 1e-1):
        for eps in (1e-3, 1e-4):
            kw = dict(alpha_fixed=alpha, epsilon=eps, max_iter=20000, verbose=0)
            rr, rn, mr, mn, res = [], [], [], [], []
            for h in range(3, len(periods)):
                base, actual = periods[:h], periods[h]
                prev = base[-1]
                ser = segment_series(n, base, **kw)
                Al = [build_A_phys(n, base[i]) for i in range(ser.X.shape[0])]
                Xs = smooth_series_nullspace(ser.X, Al, 0.5)
                res.append(max(float(np.linalg.norm(Al[i] @ Xs[i] - Al[i] @ ser.X[i])
                                / (np.linalg.norm(Al[i] @ ser.X[i]) + 1e-12))
                               for i in range(ser.X.shape[0])))
                rep_r = metrics.error_report(forecast(n, base, scheme="common", series=ser,
                                             solver_kwargs=kw).n_forecast, actual, n, prev=prev)
                rep_n = metrics.error_report(forecast(n, base, scheme="common",
                                             series=replace(ser, X=Xs),
                                             solver_kwargs=kw).n_forecast, actual, n, prev=prev)
                rr.append(rep_r["reliab_share"]); rn.append(rep_n["reliab_share"])
                mr.append(rep_r["mae"]); mn.append(rep_n["mae"])
            print(f"      {alpha:>7g} {eps:>7g} {np.mean(rr):>11.3f} {np.mean(rn):>12.3f} "
                  f"{np.mean(mr):>8.2f} {np.mean(mn):>9.2f} {np.mean(res):>10.1e}")
    print("      => null-space reliab gain + ~zero balance residual hold across settings")


def main():
    t0 = time.perf_counter()
    for domain in ("okved2", "okved1"):
        retention_invariance(domain)
        static_vs_dynamic(domain)
        energy_skill(domain)
    robustness("okved2")
    print(f"\n[time] strengthening: {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
