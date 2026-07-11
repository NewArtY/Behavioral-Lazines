"""Probe: near-attractor (linear-response) preparation -- does the AGGREGATE
binned-density deviation decay at the analytic Lambda?

Ensemble prepared within ~1 rapidity unit of theta* (physically: a
pre-accelerated ~100 MeV bunch injected into the strong field), binned on a
local grid around theta*.  If ||v(t)-v_inf|| ~ e^{-Lambda t}, the aggregate
data itself contains Lambda and the recovered operator can be tested
out-of-sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "MultiClusterPhysics"))
sys.path.insert(0, str(HERE.parents[2] / "FindProbability"))

import rapidity_ensemble as re                                        # noqa: E402

cfg = re.EnsembleConfig(m_electrons=40000)
Lam = re.contraction_rate(cfg.alpha, cfg.eps)
ts = re.theta_star(cfg.alpha, cfg.eps)
print(f"Lambda={Lam:.4f}  theta*={ts:.4f}")

rng = np.random.default_rng(7)

for tag, lo, hi in (("below", ts - 1.4, ts - 0.2),
                    ("sym",   ts - 1.2, ts + 0.8),
                    ("tight", ts - 0.7, ts - 0.1)):
    theta0 = rng.uniform(lo, hi, size=cfg.m_electrons)
    for n, half in ((12, 1.6), (24, 1.6)):
        edges = np.linspace(ts - half, ts + 1.0, n + 1)
        def binned(t):
            th = re.evolve(theta0, cfg.alpha, cfg.eps, t,
                           theta_max=cfg.theta_max) if t > 0 else theta0
            idx = np.clip(np.digitize(th, edges) - 1, 0, n - 1)
            return np.bincount(idx, minlength=n) / cfg.m_electrons
        vinf = binned(30.0)
        dt = 0.6
        times = dt * np.arange(8)
        vs = [binned(t) for t in times]
        dev = [np.linalg.norm(v - vinf) for v in vs]
        rates = []
        for k in range(len(times) - 1):
            if dev[k + 1] > 1e-12:
                rates.append(-np.log(dev[k + 1] / dev[k]) / dt)
            else:
                rates.append(float("nan"))
        rr = " ".join(f"{r:5.2f}" for r in rates)
        print(f"  prep={tag:>5} n={n:>2}: |d0|={dev[0]:.2e} rates per window: {rr}")
        print(f"        ratios: " + " ".join(f"{r/Lam:5.2f}" for r in rates))
