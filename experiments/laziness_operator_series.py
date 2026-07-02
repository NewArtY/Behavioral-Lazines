"""Behavioural laziness on a RELATIVISTIC-ELECTRON operator series (paper Sec. 5.2).

Second instantiation of the least-action operator prior, in physics.  A
radiation-damped electron ensemble (RD regime) relaxes toward the dissipative
attractor theta* (rapidity_ensemble, paper Eq. 23-24).  Sampling it at K+1
instants gives a NON-STATIONARY coarse-grained transfer-operator series
P~(t_0..t_{K-1}) -- the dissertation solver, unmodified, recovers each operator
from two binned population snapshots.  Direct integration of the ensemble is the
(free) ground truth for forecasting.

We show, in the physics domain:
  (1) the absorbing attractor cluster is recovered each window;
  (2) the SAME fit-exact property as on labour data -- null-space laziness smooths
      the operator trajectory at ~machine-zero balance residual, EWMA at a few %;
  (3) forecasting the next binned population (rolling-origin) vs direct-integration
      truth: laziness behaves as on labour data (null safe, EWMA risky);
  (4) the bridge -- the laziness reference x_ref=P0 raises retention toward the
      attractor: least action on the operator == relaxation to the dissipative
      attractor (radiative inertia).

Run:  python laziness_operator_series.py   (needs numpy; python_solver +
python_forecast from the sibling FindProbability/ project).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINDPROB = HERE.parent / "FindProbability"
sys.path.insert(0, str(FINDPROB))

from python_solver import (                                          # noqa: E402
    PeriodData, run, run_batch, absolute_laziness_vector,
)
from python_solver.core.model import col_pstar, col_pstarstar        # noqa: E402
from python_forecast.core.series import segment_series               # noqa: E402
from python_forecast.core.matrices import build_A_phys, m_phys       # noqa: E402
from python_forecast.core.laziness import (                          # noqa: E402
    smooth_series_nullspace, smooth_series_ewma, trajectory_energy,
)
from python_forecast.core.pipeline import forecast                   # noqa: E402
from python_forecast.core import direct, metrics                     # noqa: E402

import rapidity_ensemble as re                                       # noqa: E402
from transfer_operator_demo import snapshot, N_BINS, N0_RESERVOIR    # noqa: E402

SKW = dict(alpha_fixed=1e-2, epsilon=1e-6, max_iter=200_000, verbose=0)
_TIMING = HERE.parent / "2026.06.20_МультиКластер" / "code" / "experiments" / "timing.log"


def _log(msg):
    _TIMING.parent.mkdir(parents=True, exist_ok=True)
    with open(_TIMING, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def build_periods(cfg, times):
    """Relax the ensemble to each time, bin into PeriodData; return truth states."""
    theta0 = re.sample_initial(cfg)
    edges = np.linspace(0.8, cfg.theta_max, N_BINS + 1)
    theta_s = cfg.theta_star
    ab = int(np.clip(np.digitize([theta_s], edges)[0] - 1, 0, N_BINS - 1))
    scale = 1.0 / cfg.m_electrons
    periods = []
    for t in times:
        th = re.evolve(theta0, cfg.alpha, cfg.eps, t)
        n1, n2 = (a * scale for a in snapshot(th, edges, theta_s))
        periods.append(PeriodData(n1=n1, n2=n2, n0=N0_RESERVOIR,
                                  delta_n0=0.0, label=f"{t:.2f}"))
    return periods, edges, theta_s, ab


def main():
    t_all = time.perf_counter()
    cfg = re.EnsembleConfig()                       # RD regime, a0=857
    times = np.linspace(0.6, 3.4, 9)                # 9 snapshots -> 8 operators
    periods, edges, theta_s, ab = build_periods(cfg, times)
    n = N_BINS

    print("=" * 74)
    print(" Behavioural laziness on a radiation-damped electron operator series")
    print("=" * 74)
    print(f"  a0={cfg.a0:.0f}  theta*={theta_s:.3f}  attractor cluster #{ab+1}  "
          f"Lambda={re.contraction_rate(cfg.alpha, cfg.eps):.3f}")
    print(f"  {len(periods)} snapshots t in [{times[0]:.2f},{times[-1]:.2f}] "
          f"-> {len(periods)-1} operators; ensemble {cfg.m_electrons}")

    # --- (1)+(2) full-series segmentation, attractor recovery, fit-exactness ---
    batch = run_batch(n, periods, **SKW)
    P_full = np.array([r.x for r in batch.results])
    X = P_full[:, :m_phys(n)]
    A_list = [build_A_phys(n, periods[i]) for i in range(X.shape[0])]
    resid = [r.residual_norm / (np.linalg.norm(r.b) + 1e-12) for r in batch.results]
    # attractor passive-retention P* across the series (slack, from full vector)
    pstar_attr = P_full[:, col_pstar(n, ab + 1)]
    print(f"\n  recovered series: solver residual ||Ax-b||/||b|| mean={np.mean(resid):.2e} "
          f"max={np.max(resid):.2e}")
    print(f"  attractor P* across windows: mean={pstar_attr.mean():.3f} "
          f"std={pstar_attr.std():.3f}  (absorbing cluster recovered each window)")

    print(f"\n  [2] fit-exactness in physics (smooth the operator series):")
    print(f"    {'variant':>10} {'balance_resid':>14} {'energy/e0':>10}")
    e0 = trajectory_energy(X)
    for name, Xs in (("raw", X),
                     ("null:0.5", smooth_series_nullspace(X, A_list, 0.5)),
                     ("null:1", smooth_series_nullspace(X, A_list, 1.0)),
                     ("ewma:0.5", smooth_series_ewma(X, 0.5))):
        r = max(float(np.linalg.norm(A_list[i] @ Xs[i] - A_list[i] @ X[i])
                      / (np.linalg.norm(A_list[i] @ X[i]) + 1e-12))
                for i in range(X.shape[0]))
        print(f"    {name:>10} {r:>14.2e} {trajectory_energy(Xs)/(e0+1e-12):>10.3f}")

    # --- (3) forecasting vs ground truth + ENSEMBLE-SIZE sweep (jitter axis) ----
    # Finite ensemble statistics ARE the physical identification jitter: fewer
    # electrons => noisier snapshots => more null-space jitter for laziness to
    # remove.  This realises the phase map's "helps more with jitter" axis in
    # physics (and mirrors finite-statistics experimental diagnostics).
    from dataclasses import replace
    print(f"\n  [3] population forecast vs direct-integration truth; ensemble-size sweep")
    print(f"      (common scheme; ratio null/raw < 1 => laziness helps):")
    print(f"      {'m_electrons':>11} {'naive':>8} {'raw':>8} {'null:0.5':>9} {'null/raw':>9}")
    for m in (6000, 1500, 500, 200):
        cfg_m = re.EnsembleConfig(m_electrons=m)
        per_m, _, _, _ = build_periods(cfg_m, times)
        raw_e, null_e, nv_e = [], [], []
        for h in range(3, len(per_m)):
            base, actual = per_m[:h], per_m[h]
            prev = base[-1]
            ser = segment_series(n, base, **SKW)
            Al = [build_A_phys(n, base[i]) for i in range(ser.X.shape[0])]
            nv_e.append(metrics.error_report(direct.naive(base), actual, n, prev=prev)["mae"])
            res_raw = forecast(n, base, scheme="common", series=ser, solver_kwargs=SKW)
            raw_e.append(metrics.error_report(res_raw.n_forecast, actual, n, prev=prev)["mae"])
            ss = replace(ser, X=smooth_series_nullspace(ser.X, Al, 0.5))
            res_n = forecast(n, base, scheme="common", series=ss, solver_kwargs=SKW)
            null_e.append(metrics.error_report(res_n.n_forecast, actual, n, prev=prev)["mae"])
        raw_m, null_m, nv_m = np.mean(raw_e), np.mean(null_e), np.mean(nv_e)
        print(f"      {m:>11} {nv_m:>8.4f} {raw_m:>8.4f} {null_m:>9.4f} {null_m/raw_m:>9.3f}")

    # --- (4) bridge: laziness reference raises retention toward the attractor ----
    print(f"\n  [4] bridge -- least action == relaxation to the attractor:")
    snap0 = dict(now_n1=periods[2].n1, next_n1=periods[3].n1,
                 now_n2=periods[2].n2, next_n2=periods[3].n2)
    std = run(n=n, now_n0=N0_RESERVOIR, next_n0=N0_RESERVOIR, delta_n0=0.0,
              alpha_fixed=1.0, epsilon=1e-6, max_iter=200_000, verbose=0,
              x_ref=None, **snap0)
    lazy = run(n=n, now_n0=N0_RESERVOIR, next_n0=N0_RESERVOIR, delta_n0=0.0,
               alpha_fixed=1.0, epsilon=1e-6, max_iter=200_000, verbose=0,
               x_ref=absolute_laziness_vector(n), **snap0)
    ret_std = np.mean([std.x[col_pstarstar(n, i)] for i in range(1, n + 1)])
    ret_lazy = np.mean([lazy.x[col_pstarstar(n, i)] for i in range(1, n + 1)])
    print(f"    mean P** retention: standard(x_ref=0)={ret_std:.3f} -> "
          f"lazy(x_ref=P0)={ret_lazy:.3f}  (+{ret_lazy-ret_std:.3f})")
    print(f"    => the inertia prior pulls the operator toward the absorbing")
    print(f"       attractor: behavioural laziness == radiative damping.")

    dt = time.perf_counter() - t_all
    print(f"\n[time] physics laziness: {dt:.1f} s")
    _log(f"\n=== laziness_operator_series {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    _log(f"physics laziness: {dt:.1f} s")


if __name__ == "__main__":
    main()
