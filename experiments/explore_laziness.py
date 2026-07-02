"""Causal verification of BEHAVIOURAL LAZINESS as an operator-trajectory prior.

Hypothesis (discussion): the segmentation stabiliser is not neutral -- its
geometry encodes a hypothesis about the system.  The default min-norm Tikhonov
prefers the maximally-spread (high-churn) operator, which is wrong for an
inertial system.  Replacing it with a *least-action on the operator trajectory*
prior ("laziness by behaviour, not by state") should, where the rule is stable,
lower the variance of the extrapolated operator and help out-of-sample.

Two causal operationalisations (core.laziness):
  * nullspace -- pull P~(t) toward the previous operator INSIDE null(A(t));
    preserves the observed increment A(t)@P~(t) EXACTLY.  Isolates whether the
    non-identifiable component carries cross-time signal.
  * ewma      -- causal exponential smoothing of the operator trajectory (soft
    behavioural inertia); trades a little per-interval fit for a smoother
    trajectory the trend layer extrapolates.

Everything is rolling-origin (expanding window), strictly past-only -- no
leakage.  ASCII-only output (Windows console).  Run:
    python -m python_forecast.scripts.explore_laziness
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .. import _bootstrap  # noqa: F401

from .reproduce_labour_market import load_sources, _DD, _CALC, _BZ1
from ..adapters.labour_market import build_periods_audited
from ..core.series import segment_series
from ..core.matrices import build_A_phys
from ..core.laziness import (
    smooth_series_nullspace, smooth_series_ewma, trajectory_energy,
)
from ..core.pipeline import forecast, as_forecast_dict
from ..core import direct
from ..core import metrics

SKW = dict(alpha_fixed=1e-3, epsilon=1e-3, max_iter=20000, verbose=0)

# Schemes to probe: the overfitter, the frontier model scheme, and low-rank.
SCHEMES = ["per_component", "common", "lowrank"]

# Smoothing variants: (name, fn(X, A_list) -> X_smooth).  raw = lam 0.
def _variants():
    out = [("raw", None)]
    for lam in (0.5, 1.0):
        out.append((f"null:{lam:g}",
                    lambda X, A, l=lam: smooth_series_nullspace(X, A, l)))
    for lam in (0.3, 0.5, 0.7, 0.9):
        out.append((f"ewma:{lam:g}",
                    lambda X, A, l=lam: smooth_series_ewma(X, l)))
    return out


def load_periods(domain):
    emp, une, dis, exo, n = load_sources(domain, _DD, _CALC, _BZ1)
    periods, _, _ = build_periods_audited(emp, une, dis, exo)
    return n, periods


# --------------------------------------------------------------------------
# Rolling-origin sweep over smoothing variants
# --------------------------------------------------------------------------
def rolling_sweep(domain, min_base=3):
    n, periods = load_periods(domain)
    variants = _variants()
    # acc[(scheme, variant)] -> list of (mae, reliab, dir_acc) over holdouts
    acc = {(s, v): [] for s in SCHEMES for v, _ in variants}
    naive_acc = []
    n_hold = 0

    for h in range(min_base, len(periods)):
        base, actual = periods[:h], periods[h]
        prev = base[-1]
        n_hold += 1
        # one segmentation per holdout, reused across variants/schemes
        series = segment_series(n, base, **SKW)
        A_list = [build_A_phys(n, base[i]) for i in range(series.X.shape[0])]

        nv = direct.naive(base)
        rep = metrics.error_report(as_forecast_dict(nv), actual, n, prev=prev)
        naive_acc.append((rep["mae"], rep["reliab_share"], rep["dir_acc"]))

        for vname, vfn in variants:
            Xs = series.X if vfn is None else vfn(series.X, A_list)
            series_s = replace(series, X=Xs)
            for s in SCHEMES:
                res = forecast(n, base, scheme=s, series=series_s, solver_kwargs=SKW)
                rep = metrics.error_report(res.n_forecast, actual, n, prev=prev)
                acc[(s, vname)].append(
                    (rep["mae"], rep["reliab_share"], rep["dir_acc"]))

    print(f"\n{'='*74}\nDOMAIN {domain}: rolling-origin MEAN over {n_hold} holdouts "
          f"(periods {periods[min_base].label}..{periods[-1].label})\n{'='*74}")
    nv = np.array(naive_acc).mean(axis=0)
    print(f"  {'naive (anchor)':>22}   MAE={nv[0]:7.2f}  reliab={nv[1]:.3f}  dir_acc={nv[2]:.3f}")
    for s in SCHEMES:
        print(f"  -- scheme: {s}")
        raw = np.array(acc[(s, 'raw')]).mean(axis=0)
        for vname, _ in variants:
            m = np.array(acc[(s, vname)]).mean(axis=0)
            d_mae = m[0] - raw[0]
            mark = ""
            if vname != "raw":
                better_dir = m[2] > raw[2] + 1e-9
                better_rel = m[1] > raw[1] + 1e-9
                better_mae = m[0] < raw[0] - 1e-9
                flags = "".join(c for c, b in
                                [("M", better_mae), ("R", better_rel), ("D", better_dir)] if b)
                mark = f"  <{flags}>" if flags else ""
            print(f"     {vname:>10}   MAE={m[0]:7.2f} (d{d_mae:+6.2f})  "
                  f"reliab={m[1]:.3f}  dir_acc={m[2]:.3f}{mark}")


# --------------------------------------------------------------------------
# Behavioural-action diagnostic: how much do the smoothers move / calm P~(t)?
# --------------------------------------------------------------------------
def energy_diagnostic(domain):
    n, periods = load_periods(domain)
    series = segment_series(n, periods, **SKW)
    A_list = [build_A_phys(n, periods[i]) for i in range(series.X.shape[0])]
    X = series.X
    e0 = trajectory_energy(X)
    print(f"\n[energy] {domain}: behavioural action sum||dP~||^2 and operator perturbation")
    print(f"  {'variant':>10} {'energy':>12} {'e/e0':>7} {'||Xs-X||/||X||':>16} {'max|A@Xs-A@X|/|A@X|':>20}")
    print(f"  {'raw':>10} {e0:>12.4g} {1.0:>7.3f} {0.0:>16.4f} {0.0:>20.4e}")
    rows = [("null:1", smooth_series_nullspace(X, A_list, 1.0)),
            ("ewma:0.5", smooth_series_ewma(X, 0.5)),
            ("ewma:0.9", smooth_series_ewma(X, 0.9))]
    for name, Xs in rows:
        e = trajectory_energy(Xs)
        pert = float(np.linalg.norm(Xs - X) / (np.linalg.norm(X) + 1e-12))
        incr = max(float(np.linalg.norm(A_list[i] @ Xs[i] - A_list[i] @ X[i])
                         / (np.linalg.norm(A_list[i] @ X[i]) + 1e-12))
                   for i in range(X.shape[0]))
        print(f"  {name:>10} {e:>12.4g} {e/e0:>7.3f} {pert:>16.4f} {incr:>20.4e}")


# --------------------------------------------------------------------------
# Synthetic controlled demo: stable true operator, per-step rotating A(t).
# Min-norm recovery jitters in the (rotating) null space even though the truth
# is constant; causal laziness removes the jitter -> better extrapolation.
# --------------------------------------------------------------------------
def synthetic_demo(seed=0, rows=6, cols=30, k=5, drift=0.04, rot=0.05):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((rows, cols))
    p0 = rng.standard_normal(cols)
    v = rng.standard_normal(cols)
    v = v / np.linalg.norm(v)
    # true operator with a mild, stable linear drift
    P_true = np.array([p0 + drift * t * v for t in range(k + 1)])
    # per-interval matrices: A0 slowly rotated (null space turns each step)
    A_list = [A0 + rot * rng.standard_normal((rows, cols)) for _ in range(k + 1)]
    # min-norm recovery of each interval's operator from its own increment
    X_rec = np.array([np.linalg.pinv(A_list[t]) @ (A_list[t] @ P_true[t])
                      for t in range(k)])

    def extrapolate_linear(X):
        # fit a per-component linear trend on tau=1..k, predict tau=k+1
        kk = X.shape[0]
        tau = np.arange(1, kk + 1, dtype=float)
        out = np.empty(X.shape[1])
        M = np.vstack([np.ones_like(tau), tau]).T
        coef, *_ = np.linalg.lstsq(M, X, rcond=None)
        out = coef[0] + coef[1] * (kk + 1)
        return out

    A_fore = A_list[k]
    truth_incr = A_fore @ P_true[k]

    def fore_err(Xseries):
        p_pr = extrapolate_linear(Xseries)
        return float(np.linalg.norm(A_fore @ p_pr - truth_incr))

    raw_err = fore_err(X_rec)
    print(f"\n[synthetic] stable true operator, rotating A(t); forecast error of "
          f"A_fore@p_pr vs truth (lower=better)")
    print(f"  {'variant':>12} {'traj_energy':>12} {'fore_err':>10} {'vs raw':>8}")
    print(f"  {'raw(minnorm)':>12} {trajectory_energy(X_rec):>12.4g} {raw_err:>10.4f} {1.0:>8.3f}")
    for lam in (0.5, 1.0):
        Xs = smooth_series_nullspace(X_rec, A_list[:k], lam)
        e = fore_err(Xs)
        print(f"  {'null:'+format(lam,'g'):>12} {trajectory_energy(Xs):>12.4g} "
              f"{e:>10.4f} {e/raw_err:>8.3f}")
    for lam in (0.3, 0.5, 0.7):
        Xs = smooth_series_ewma(X_rec, lam)
        e = fore_err(Xs)
        print(f"  {'ewma:'+format(lam,'g'):>12} {trajectory_energy(Xs):>12.4g} "
              f"{e:>10.4f} {e/raw_err:>8.3f}")


def main():
    synthetic_demo()
    for domain in ("okved1", "okved2"):
        energy_diagnostic(domain)
        rolling_sweep(domain)


if __name__ == "__main__":
    main()
