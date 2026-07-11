"""Can the AGGREGATE-snapshot recovery quantitatively return the LL rate Lambda?

Context (paper Sec. 5.2).  The headline number Lambda_rec=0.68 (ratio 0.95 to the
analytic Lambda=2(alpha-eps_rad)=0.72) is currently produced by ULAM pair-counting
(same electrons tracked cell->cell) -- micro-trajectory data that the paper's own
setting (two aggregated density snapshots) does not provide.  The paper's solver
(min-norm, n=8, dt=0.6, one window) gives Lambda ~ 1.99 (ratio 2.77), unreported.

This script tests, honestly and with stability sweeps, whether the aggregate
method itself can recover Lambda:

  base   -- reproduce both baselines (Ulam 0.68; solver min-norm 1.99) + validate
            a fast lsq_linear proxy for the solver.
  ident  -- identifiability diagnosis: the spectral gap along the null-space
            segment between two EXACTLY-fitting representatives (min-norm vs
            laziness-projection).  If the gap sweeps a wide range at identical
            residual, it is NOT identifiable from one window.
  res    -- hypothesis 1: single-window min-norm gap vs (n_bins, dt) incl. the
            Ulam-resolved window (n=24, dt=1.2), averaged over start times.
  alpha  -- hypothesis 2: gap vs prior (x_ref=0 vs x_ref=P0) and regularisation
            alpha, incl. the solver's own quasi-optimal alpha.
  series -- hypothesis 4: gap of the null-smoothed (paper's method) operator
            series vs raw min-norm series, lam in {0.5, 1.0}, EWMA contrast.
  stack  -- stationarity-stacked AGGREGATE recovery: one operator consistent
            with K consecutive window pairs (the flow is autonomous, so the
            time-dt transfer operator is stationary; K binned snapshots are
            still aggregate-only data).  Gap vs K, n, dt, prior.
  robust -- anti-circularity: the fixed winning recipe swept over the TRUE
            Lambda (drive alpha_d in {0.18,0.36,0.54}), a0 in {400,857,2000},
            seeds and ensemble sizes.  A recipe tuned to 0.72 would fail this.

Run:  python recover_physics_gap.py <stage>   (stage in base|ident|res|alpha|
      series|stack|robust|all)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
PHYS = HERE.parents[2] / "MultiClusterPhysics"
FINDPROB = HERE.parents[2] / "FindProbability"
sys.path.insert(0, str(PHYS))
sys.path.insert(0, str(FINDPROB))

import rapidity_ensemble as re                                        # noqa: E402
from transfer_operator_demo import extract_operator, N0_RESERVOIR     # noqa: E402
from python_solver import run, absolute_laziness_vector, m_total      # noqa: E402
from python_solver.core.matrix import build_matrix_and_rhs            # noqa: E402
from python_forecast.core.laziness import (                           # noqa: E402
    smooth_series_nullspace, smooth_series_ewma,
)
from python_forecast.core.matrices import build_A_phys, m_phys        # noqa: E402
from python_solver import PeriodData, run_batch                       # noqa: E402
from python_solver.core.model import col_p11, col_p12                 # noqa: E402

SETTLE = 0.45
SKW = dict(alpha_fixed=1e-2, epsilon=1e-6, max_iter=200_000, verbose=0)


# ---------------------------------------------------------------- helpers ---
def slem(T):
    ev = np.linalg.eigvals(T)
    mod = np.sort(np.abs(ev))[::-1]
    return float(mod[1]) if mod.size > 1 else float("nan")


def active_block_rate(op, n, dt):
    """Paper's construction: off-diag P11 (dest 1..n), diag P**, row-normalised."""
    M = op["P11"][:, :n].copy()
    for i in range(n):
        M[i, i] = op["Pstarstar"][i]
    rs = M.sum(axis=1, keepdims=True)
    M = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
    r = slem(M)
    return -np.log(max(r, 1e-12)) / dt, r


def full_chain_rate(op, n, dt):
    """SLEM of the full (2n+1)-state coarse chain: act 1..n, pas 1..n, res."""
    S = np.zeros((2 * n + 1, 2 * n + 1))
    for i in range(n):
        S[i, :n] = op["P11"][i, :n]
        S[i, i] = op["Pstarstar"][i]
        S[i, n + i] = op["P12"][i]
        S[i, 2 * n] = op["P11"][i, n]                     # leak to dest n+1 -> res
    for i in range(n):                                     # passive i = P21 row i+1
        S[n + i, :n] = op["P21"][i + 1, :n]
        S[n + i, n + i] = op["Pstar"][i + 1]
        S[n + i, 2 * n] = op["P21"][i + 1, n]
    S[2 * n, :n] = op["P21"][0, :n]
    S[2 * n, 2 * n] = op["Pstar"][0] + op["P21"][0, n]
    rs = S.sum(axis=1, keepdims=True)
    S = np.divide(S, rs, out=np.zeros_like(S), where=rs > 0)
    r = slem(S)
    return -np.log(max(r, 1e-12)) / dt, r


def op_from_phys(xp, n):
    """Operator blocks from the PHYSICAL part only (P** recomputed from rows)."""
    x = np.zeros(m_total(n))
    x[:m_phys(n)] = xp[:m_phys(n)]
    op = extract_operator(x, n)
    stay = 1.0 - op["P12"] - op["P11"].sum(axis=1)
    op["Pstarstar"] = np.clip(stay, 0.0, 1.0)
    return op


def snapshot(theta, edges, theta_s, n):
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n - 1)
    passive = np.abs(theta - theta_s) < SETTLE
    n1 = np.array([np.count_nonzero((idx == k) & ~passive) for k in range(n)], float)
    n2 = np.array([np.count_nonzero((idx == k) & passive) for k in range(n)], float)
    return n1, n2


class Ensemble:
    """Snapshot factory with caching of evolved states."""

    def __init__(self, cfg, alpha_d=None):
        self.cfg = cfg
        self.alpha_d = cfg.alpha if alpha_d is None else alpha_d
        self.theta0 = re.sample_initial(cfg)
        self.theta_s = re.theta_star(self.alpha_d, cfg.eps)
        self.Lambda = re.contraction_rate(self.alpha_d, cfg.eps)
        self._cache = {}

    def state(self, t):
        key = round(t, 6)
        if key not in self._cache:
            self._cache[key] = re.evolve(self.theta0, self.alpha_d, self.cfg.eps,
                                         t, theta_max=self.cfg.theta_max)
        return self._cache[key]

    def frac_snapshot(self, t, edges, n):
        n1, n2 = snapshot(self.state(t), edges, self.theta_s, n)
        s = 1.0 / self.cfg.m_electrons
        return n1 * s, n2 * s


def window_system(ens, t0, dt, edges, n):
    n1a, n2a = ens.frac_snapshot(t0, edges, n)
    n1b, n2b = ens.frac_snapshot(t0 + dt, edges, n)
    A, b = build_matrix_and_rhs(n=n, now_n1=n1a, now_n2=n2a,
                                now_n0=N0_RESERVOIR, delta_n0=0.0,
                                next_n1=n1b, next_n2=n2b, next_n0=N0_RESERVOIR)
    return A, b, (n1a, n2a, n1b, n2b)


def tik_solve(A, b, alpha, x_ref=None):
    """min ||Ax-b||^2 + alpha ||x - x_ref||^2, 0<=x<=1 (fast lsq_linear proxy)."""
    m = A.shape[1]
    if x_ref is None:
        x_ref = np.zeros(m)
    Aa = np.vstack([A, np.sqrt(alpha) * np.eye(m)])
    ba = np.concatenate([b, np.sqrt(alpha) * x_ref])
    res = lsq_linear(Aa, ba, bounds=(0.0, 1.0), method="trf",
                     tol=1e-12, max_iter=400)
    return res.x


def solver_solve(ens, t0, dt, edges, n, *, x_ref=None, alpha_fixed=1e-2,
                 quasi=False):
    n1a, n2a = ens.frac_snapshot(t0, edges, n)
    n1b, n2b = ens.frac_snapshot(t0 + dt, edges, n)
    kw = dict(n=n, now_n1=n1a, next_n1=n1b, now_n2=n2a, next_n2=n2b,
              now_n0=N0_RESERVOIR, next_n0=N0_RESERVOIR, delta_n0=0.0,
              epsilon=1e-6, max_iter=200_000, verbose=0, x_ref=x_ref)
    if quasi:
        kw.update(alpha_fixed=None, alpha_init=1.0, alpha_ratio=0.1,
                  alpha_max_steps=20)
    else:
        kw.update(alpha_fixed=alpha_fixed)
    return run(**kw)


def rates_from_x(x, n, dt):
    op = extract_operator(x, n)
    lam_a, rho_a = active_block_rate(op, n, dt)
    lam_f, rho_f = full_chain_rate(op, n, dt)
    return lam_a, rho_a, lam_f, rho_f


def rel_resid(A, b, x):
    return float(np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-300))


def ulam_operator(ens, t0, dt, edges, n):
    src, dst = ens.state(t0), ens.state(t0 + dt)
    si = np.clip(np.digitize(src, edges) - 1, 0, n - 1)
    di = np.clip(np.digitize(dst, edges) - 1, 0, n - 1)
    T = np.zeros((n, n))
    for a_, b_ in zip(si, di):
        T[a_, b_] += 1.0
    rs = T.sum(axis=1, keepdims=True)
    keep = rs.ravel() > 0
    T[keep] /= rs[keep]
    return T[np.ix_(keep, keep)]


def edges_for(cfg, n):
    return np.linspace(0.8, cfg.theta_max, n + 1)


# ------------------------------------------------------------------ stages --
def stage_base():
    print("=" * 74)
    print("BASE: reproduce both baselines + validate lsq proxy")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    ens = Ensemble(cfg)
    Lam = ens.Lambda
    print(f"analytic Lambda = {Lam:.4f}, theta* = {ens.theta_s:.3f}")

    # Ulam reference (micro pair-counting), resolved window
    n_u, dt_u = 24, 1.2
    e_u = edges_for(cfg, n_u)
    lams = []
    for t0 in (0.4, 0.6, 0.8, 1.0, 1.2):
        r = slem(ulam_operator(ens, t0, dt_u, e_u, n_u))
        lams.append(-np.log(max(r, 1e-12)) / dt_u)
    print(f"[Ulam micro] n=24 dt=1.2: Lambda_rec = {np.mean(lams):.4f} "
          f"+/- {np.std(lams):.4f}  ratio {np.mean(lams)/Lam:.2f}")

    # paper's solver window: n=8, dt=0.6, t0=1.0, min-norm alpha=1e-2
    n, dt, t0 = 8, 0.6, 1.0
    e = edges_for(cfg, n)
    res = solver_solve(ens, t0, dt, e, n, x_ref=None, alpha_fixed=1e-2)
    la, ra, lf, rf = rates_from_x(res.x, n, dt)
    rr = res.residual_norm / np.linalg.norm(res.b)
    print(f"[solver min-norm] n=8 dt=0.6 t0=1.0: resid={rr:.2e}")
    print(f"  active-block: rho*={ra:.4f} Lambda={la:.4f} ratio={la/Lam:.2f}")
    print(f"  full-chain  : rho*={rf:.4f} Lambda={lf:.4f} ratio={lf/Lam:.2f}")

    # proxy validation
    A, b, _ = window_system(ens, t0, dt, e, n)
    xp = tik_solve(A, b, 1e-2, None)
    la2, _, lf2, _ = rates_from_x(xp, n, dt)
    print(f"[lsq proxy same problem] resid={rel_resid(A, b, xp):.2e} "
          f"Lambda_act={la2:.4f} Lambda_full={lf2:.4f} "
          f"||x_ps - x_lsq||={np.linalg.norm(res.x - xp):.3e}")


def stage_ident():
    print("=" * 74)
    print("IDENT: gap along the exact-fit null-space segment (one window)")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    ens = Ensemble(cfg)
    Lam = ens.Lambda
    n, dt, t0 = 8, 0.6, 1.0
    e = edges_for(cfg, n)
    A, b, _ = window_system(ens, t0, dt, e, n)
    rank = np.linalg.matrix_rank(A)
    print(f"A: {A.shape}, rank {rank}, null dim {A.shape[1]-rank}")
    x_min = tik_solve(A, b, 1e-4, None)                       # ~min-norm rep
    x_lzy = tik_solve(A, b, 1e-4, absolute_laziness_vector(n))  # ~lazy projection
    print(f"resid: min-norm {rel_resid(A,b,x_min):.2e}, lazy {rel_resid(A,b,x_lzy):.2e}")
    print(f"{'t':>5} {'resid':>9} {'rho*_act':>9} {'Lam_act':>8} {'ratio':>6}")
    for t in np.linspace(0, 1, 11):
        x = (1 - t) * x_min + t * x_lzy
        la, ra, lf, rf = rates_from_x(x, n, dt)
        print(f"{t:>5.2f} {rel_resid(A,b,x):>9.2e} {ra:>9.4f} {la:>8.3f} "
              f"{la/Lam:>6.2f}")
    print("=> every point fits the SAME two snapshots; the gap is a property of")
    print("   the representative (prior), not of the aggregate data.")


def stage_res():
    print("=" * 74)
    print("RES (H1): single-window min-norm gap vs (n, dt), avg over t0")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    ens = Ensemble(cfg)
    Lam = ens.Lambda
    t0s = (0.4, 0.8, 1.2)
    print(f"{'n':>4} {'dt':>5} {'Lam_act':>8} {'ratio':>6} {'Lam_full':>9} "
          f"{'ratio':>6} {'resid':>9}")
    for n in (8, 16, 24):
        e = edges_for(cfg, n)
        for dt in (0.6, 1.2, 2.4):
            las, lfs, rrs = [], [], []
            for t0 in t0s:
                A, b, _ = window_system(ens, t0, dt, e, n)
                x = tik_solve(A, b, 1e-4, None)
                la, _, lf, _ = rates_from_x(x, n, dt)
                las.append(la); lfs.append(lf); rrs.append(rel_resid(A, b, x))
            print(f"{n:>4} {dt:>5.1f} {np.mean(las):>8.3f} "
                  f"{np.mean(las)/Lam:>6.2f} {np.mean(lfs):>9.3f} "
                  f"{np.mean(lfs)/Lam:>6.2f} {max(rrs):>9.2e}")


def stage_alpha():
    print("=" * 74)
    print("ALPHA (H2): gap vs prior and regularisation weight (n=8, dt=0.6)")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    ens = Ensemble(cfg)
    Lam = ens.Lambda
    n, dt = 8, 0.6
    e = edges_for(cfg, n)
    x_lzy_ref = absolute_laziness_vector(n)
    t0s = (0.6, 1.0, 1.4)
    print(f"{'prior':>8} {'alpha':>8} {'Lam_act(mean/std)':>18} {'ratio':>6} "
          f"{'resid(max)':>11}")
    for prior, xr in (("min", None), ("lazy", x_lzy_ref)):
        for al in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            las, rrs = [], []
            for t0 in t0s:
                r = solver_solve(ens, t0, dt, e, n, x_ref=xr, alpha_fixed=al)
                la, _, _, _ = rates_from_x(r.x, n, dt)
                las.append(la)
                rrs.append(r.residual_norm / np.linalg.norm(r.b))
            print(f"{prior:>8} {al:>8.0e} {np.mean(las):>10.3f}/{np.std(las):<7.3f} "
                  f"{np.mean(las)/Lam:>6.2f} {max(rrs):>11.2e}")
    # solver's own quasi-optimal alpha
    for prior, xr in (("min", None), ("lazy", x_lzy_ref)):
        r = solver_solve(ens, 1.0, dt, e, n, x_ref=xr, quasi=True)
        la, _, _, _ = rates_from_x(r.x, n, dt)
        print(f"[quasi-opt] prior={prior}: alpha={r.alpha:.2e} "
              f"Lambda_act={la:.3f} ratio={la/Lam:.2f} "
              f"resid={r.residual_norm/np.linalg.norm(r.b):.2e}")


def stage_series():
    print("=" * 74)
    print("SERIES (H4): null-smoothed operator series gap (paper's smoother)")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    ens = Ensemble(cfg)
    Lam = ens.Lambda
    n = 8
    e = edges_for(cfg, n)
    for dt, t_first in ((0.35, 0.6), (0.6, 0.4)):
        times = t_first + dt * np.arange(9)
        periods = []
        for t in times:
            n1, n2 = ens.frac_snapshot(t, e, n)
            periods.append(PeriodData(n1=n1, n2=n2, n0=N0_RESERVOIR,
                                      delta_n0=0.0, label=f"{t:.2f}"))
        batch = run_batch(n, periods, **SKW)
        X = np.array([r.x for r in batch.results])[:, :m_phys(n)]
        A_list = [build_A_phys(n, periods[i]) for i in range(X.shape[0])]
        print(f"\n-- dt={dt}, windows t0={times[0]:.2f}..{times[-2]:.2f} "
              f"(K={X.shape[0]}) --")
        print(f"{'variant':>10} {'Lam per window':>52} {'mean':>6} {'last':>6} "
              f"{'ratio_last':>10}")
        for name, Xs in (("raw", X),
                         ("null:0.5", smooth_series_nullspace(X, A_list, 0.5)),
                         ("null:1.0", smooth_series_nullspace(X, A_list, 1.0)),
                         ("ewma:0.5", smooth_series_ewma(X, 0.5))):
            lams = []
            for i in range(Xs.shape[0]):
                op = op_from_phys(Xs[i], n)
                la, _ = active_block_rate(op, n, dt)
                lams.append(la)
            per = " ".join(f"{v:5.2f}" for v in lams)
            print(f"{name:>10} {per:>52} {np.mean(lams):>6.2f} "
                  f"{lams[-1]:>6.2f} {lams[-1]/Lam:>10.2f}")


def stack_solve(ens, n, dt, K, t_first, *, alpha=1e-3, x_ref=None,
                edges=None):
    """One stationary operator consistent with K consecutive window pairs."""
    e = edges_for(ens.cfg, n) if edges is None else edges
    rows, rhs = [], []
    A0 = None
    for k in range(K):
        A, b, _ = window_system(ens, t_first + k * dt, dt, e, n)
        rows.append(A[:2 * n + 1])          # balance rows (window-specific)
        rhs.append(b[:2 * n + 1])
        A0 = A
    rows.append(A0[2 * n + 1:])             # normalisation rows (shared)
    rhs.append(A0[2 * n + 1:] @ np.zeros(A0.shape[1]) + 1.0)
    As = np.vstack(rows)
    bs = np.concatenate(rhs)
    x = tik_solve(As, bs, alpha, x_ref)
    return x, As, bs


def stage_stack():
    print("=" * 74)
    print("STACK: stationary aggregate recovery -- gap vs K windows, prior, n, dt")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    ens = Ensemble(cfg)
    Lam = ens.Lambda
    print(f"analytic Lambda = {Lam:.4f}")
    print(f"{'n':>4} {'dt':>5} {'K':>3} {'prior':>5} {'Lam_act':>8} {'ratio':>6} "
          f"{'Lam_full':>9} {'resid':>9}")
    for n in (6, 8, 12):
        for dt in (0.35, 0.6, 1.2):
            for K in (1, 2, 4, 8):
                for prior, xr in (("min", None),
                                  ("lazy", absolute_laziness_vector(n))):
                    x, As, bs = stack_solve(ens, n, dt, K, 0.4,
                                            alpha=1e-3, x_ref=xr)
                    la, _, lf, _ = rates_from_x(x, n, dt)
                    print(f"{n:>4} {dt:>5.2f} {K:>3} {prior:>5} {la:>8.3f} "
                          f"{la/Lam:>6.2f} {lf:>9.3f} "
                          f"{rel_resid(As, bs, x):>9.2e}")


def stage_robust():
    print("=" * 74)
    print("ROBUST: fixed recipe vs TRUE Lambda sweep / a0 / seed / ensemble size")
    print("=" * 74)
    # recipe fixed in dimensionless time Lambda*t: windows (0.4+0.6k)*s, s=0.72/Lam
    base_dt, base_t0, K, n = 0.6, 0.4, 8, 8
    print("recipe: stationary stack, n=8, K=8 windows, dt=0.6*(0.72/Lambda), "
          "t0=0.4*(0.72/Lambda), alpha=1e-3, priors min & lazy")
    print(f"{'case':>28} {'Lam_true':>8} {'prior':>5} {'Lam_act':>8} {'ratio':>6} "
          f"{'resid':>9}")

    def one(tag, cfg, alpha_d=None):
        ens = Ensemble(cfg, alpha_d=alpha_d)
        Lam = ens.Lambda
        s = 0.72 / Lam
        for prior, xr in (("min", None), ("lazy", absolute_laziness_vector(n))):
            x, As, bs = stack_solve(ens, n, base_dt * s, K, base_t0 * s,
                                    alpha=1e-3, x_ref=xr)
            la, _, _, _ = rates_from_x(x, n, base_dt * s)
            print(f"{tag:>28} {Lam:>8.3f} {prior:>5} {la:>8.3f} {la/Lam:>6.2f} "
                  f"{rel_resid(As, bs, x):>9.2e}")

    # true-Lambda sweep via drive alpha_d (a0=857)
    for ad in (0.18, 0.36, 0.54):
        one(f"drive alpha_d={ad}", re.EnsembleConfig(m_electrons=40000), ad)
    # a0 sweep (RD-regime validity theta*<theta_max needs a0 >~ 320)
    for a0 in (400.0, 857.0, 2000.0):
        one(f"a0={a0:.0f}", re.EnsembleConfig(a0=a0, m_electrons=40000))
    # seeds and ensemble sizes at the base point
    for seed in (7, 11, 23):
        one(f"seed={seed}", re.EnsembleConfig(m_electrons=40000, seed=seed))
    for m in (6000, 40000):
        one(f"m={m}", re.EnsembleConfig(m_electrons=m))


# ---------------------------------------------------------- multi-shot -----
def shot_system(cfg, alpha_d, rng, t0, dt, edges, n):
    """One 'shot': random preparation -> two aggregated binned spectra -> A,b."""
    lo = rng.uniform(1.0, cfg.theta_max - 1.2)
    width = rng.uniform(1.0, cfg.theta_max - lo)
    th0 = rng.uniform(lo, lo + width, size=cfg.m_electrons)
    theta_s = re.theta_star(alpha_d, cfg.eps)
    th1 = re.evolve(th0, alpha_d, cfg.eps, t0, theta_max=cfg.theta_max)
    th2 = re.evolve(th0, alpha_d, cfg.eps, t0 + dt, theta_max=cfg.theta_max)
    s = 1.0 / cfg.m_electrons
    n1a, n2a = (a * s for a in snapshot(th1, edges, theta_s, n))
    n1b, n2b = (a * s for a in snapshot(th2, edges, theta_s, n))
    A, b = build_matrix_and_rhs(n=n, now_n1=n1a, now_n2=n2a,
                                now_n0=N0_RESERVOIR, delta_n0=0.0,
                                next_n1=n1b, next_n2=n2b, next_n0=N0_RESERVOIR)
    return A, b


def shots_solve(cfg, alpha_d, M, n, dt, *, t0=0.3, seed0=0, alpha=1e-4,
                x_ref=None):
    """Stack M independent-preparation windows (aggregate-only data)."""
    edges = edges_for(cfg, n)
    rng = np.random.default_rng(1000 + seed0)
    rows, rhs = [], []
    A0 = None
    for _ in range(M):
        A, b = shot_system(cfg, alpha_d, rng, t0, dt, edges, n)
        rows.append(A[:2 * n + 1])
        rhs.append(b[:2 * n + 1])
        A0 = A
    rows.append(A0[2 * n + 1:])
    rhs.append(np.ones(A0.shape[0] - (2 * n + 1)))
    As = np.vstack(rows)
    bs = np.concatenate(rhs)
    x = tik_solve(As, bs, alpha, x_ref)
    return x, As, bs


def stage_shots():
    print("=" * 74)
    print("SHOTS: multi-preparation aggregate recovery -- gap vs #shots M, prior")
    print("(each shot contributes ONLY two binned spectra; no pair-counting)")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    Lam = re.contraction_rate(cfg.alpha, cfg.eps)
    print(f"analytic Lambda = {Lam:.4f}")
    print(f"{'n':>4} {'dt':>5} {'M':>3} {'prior':>5} {'Lam_act':>8} {'ratio':>6} "
          f"{'resid':>9}")
    for n, dt in ((8, 0.6), (8, 1.2), (24, 1.2)):
        for M in (1, 2, 4, 8, 16, 32):
            for prior, xr in (("min", None), ("lazy", absolute_laziness_vector(n))):
                x, As, bs = shots_solve(cfg, cfg.alpha, M, n, dt, x_ref=xr)
                la, _, lf, _ = rates_from_x(x, n, dt)
                print(f"{n:>4} {dt:>5.1f} {M:>3} {prior:>5} {la:>8.3f} "
                      f"{la/Lam:>6.2f} {rel_resid(As, bs, x):>9.2e}")


def stage_shotsrobust():
    print("=" * 74)
    print("SHOTSROBUST: fixed multi-shot recipe vs TRUE Lambda / seeds / a0")
    print("recipe: M=16 shots, n=8, dt=0.6*(0.72/Lambda), t0=0.3*(0.72/Lambda),")
    print("        alpha=1e-4, both priors  (recipe fixed in units of 1/Lambda)")
    print("=" * 74)
    print(f"{'case':>24} {'Lam_true':>8} {'prior':>5} {'Lam_act':>8} {'ratio':>6} "
          f"{'resid':>9}")

    def one(tag, cfg, alpha_d, seed0=0):
        Lam = re.contraction_rate(alpha_d, cfg.eps)
        s = 0.72 / Lam
        for prior, xr in (("min", None), ("lazy", absolute_laziness_vector(8))):
            x, As, bs = shots_solve(cfg, alpha_d, 16, 8, 0.6 * s,
                                    t0=0.3 * s, seed0=seed0, x_ref=xr)
            la, _, _, _ = rates_from_x(x, 8, 0.6 * s)
            print(f"{tag:>24} {Lam:>8.3f} {prior:>5} {la:>8.3f} {la/Lam:>6.2f} "
                  f"{rel_resid(As, bs, x):>9.2e}")

    for ad in (0.18, 0.36, 0.54):
        one(f"drive alpha_d={ad}", re.EnsembleConfig(m_electrons=40000), ad)
    for a0 in (400.0, 2000.0):
        one(f"a0={a0:.0f}", re.EnsembleConfig(a0=a0, m_electrons=40000), 0.36)
    for sd in (1, 2, 3):
        one(f"shot-seed={sd}", re.EnsembleConfig(m_electrons=40000), 0.36,
            seed0=sd * 100)
    one("m=6000", re.EnsembleConfig(m_electrons=6000), 0.36)


# ------------------------------------------- multi-shot, narrow + restricted
def narrow_shot_states(cfg, alpha_d, rng, t0, dt, width_rng=(0.3, 1.0)):
    """Random narrow energy-selected preparation, evolved to t0 and t0+dt."""
    c = rng.uniform(1.0, cfg.theta_max)
    w = rng.uniform(*width_rng)
    th0 = np.clip(rng.uniform(c - w / 2, c + w / 2, size=cfg.m_electrons),
                  0.05, None)
    th1 = re.evolve(th0, alpha_d, cfg.eps, t0, theta_max=cfg.theta_max)
    th2 = re.evolve(th0, alpha_d, cfg.eps, t0 + dt, theta_max=cfg.theta_max)
    return th1, th2


def gap_restricted(x, n, dt, excited):
    """Active-block rate restricted to data-excited cells (Ulam's `keep`)."""
    op = extract_operator(x, n)
    M = op["P11"][:, :n].copy()
    for i in range(n):
        M[i, i] = op["Pstarstar"][i]
    idx = np.flatnonzero(excited)
    M = M[np.ix_(idx, idx)]
    rs = M.sum(axis=1, keepdims=True)
    M = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
    r = slem(M)
    return -np.log(max(r, 1e-12)) / dt, r


def stage_shots2():
    print("=" * 74)
    print("SHOTS2: narrow-preparation multi-shot aggregates, excited-cell gap")
    print("(benchmark = Ulam at the SAME discretisation; target = Lambda)")
    print("=" * 74)
    cfg = re.EnsembleConfig(m_electrons=40000)
    alpha_d = cfg.alpha
    Lam = re.contraction_rate(alpha_d, cfg.eps)
    theta_s = re.theta_star(alpha_d, cfg.eps)
    t0 = 0.3
    for n, dt in ((8, 0.6), (24, 1.2)):
        edges = edges_for(cfg, n)
        # Ulam benchmark at the same (n, dt): broad ensemble, several t0
        ens = Ensemble(cfg)
        lam_u = np.mean([-np.log(max(slem(ulam_operator(ens, tu, dt, edges, n)),
                                     1e-12)) / dt
                         for tu in (0.4, 0.8, 1.2)])
        print(f"\n-- n={n}, dt={dt}:  Ulam(same grid)={lam_u:.3f} "
              f"(ratio {lam_u/Lam:.2f}), analytic Lambda={Lam:.3f} --")
        print(f"{'M':>4} {'cells':>6} {'prior':>5} {'Lam_act':>8} {'ratio':>6} "
              f"{'vs_Ulam':>8} {'resid':>9}")
        rng = np.random.default_rng(4242)
        rows, rhs = [], []
        A_last = None
        excited = np.zeros(n, bool)
        s = 1.0 / cfg.m_electrons
        M_done = 0
        for M_tgt in (8, 16, 32, 64):
            while M_done < M_tgt:
                th1, th2 = narrow_shot_states(cfg, alpha_d, rng, t0, dt)
                n1a, n2a = (a * s for a in snapshot(th1, edges, theta_s, n))
                n1b, n2b = (a * s for a in snapshot(th2, edges, theta_s, n))
                excited |= (n1a > 2e-3)
                A, b = build_matrix_and_rhs(
                    n=n, now_n1=n1a, now_n2=n2a, now_n0=N0_RESERVOIR,
                    delta_n0=0.0, next_n1=n1b, next_n2=n2b,
                    next_n0=N0_RESERVOIR)
                rows.append(A[:2 * n + 1]); rhs.append(b[:2 * n + 1])
                A_last = A
                M_done += 1
            As = np.vstack(rows + [A_last[2 * n + 1:]])
            bs = np.concatenate(rhs + [np.ones(A_last.shape[0] - (2 * n + 1))])
            for prior, xr in (("min", None), ("lazy", absolute_laziness_vector(n))):
                x = tik_solve(As, bs, 1e-4, xr)
                la, _ = gap_restricted(x, n, dt, excited)
                print(f"{M_tgt:>4} {int(excited.sum()):>6} {prior:>5} "
                      f"{la:>8.3f} {la/Lam:>6.2f} {la/lam_u:>8.2f} "
                      f"{rel_resid(As, bs, x):>9.2e}")


STAGES = dict(base=stage_base, ident=stage_ident, res=stage_res,
              alpha=stage_alpha, series=stage_series, stack=stage_stack,
              robust=stage_robust, shots=stage_shots,
              shotsrobust=stage_shotsrobust, shots2=stage_shots2)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(STAGES) if which == "all" else [which]
    for nm in names:
        t0 = time.perf_counter()
        STAGES[nm]()
        print(f"[time] {nm}: {time.perf_counter() - t0:.1f} s\n")


if __name__ == "__main__":
    main()
