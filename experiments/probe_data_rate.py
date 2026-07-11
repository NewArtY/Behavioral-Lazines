"""Probe: what relaxation rate does the AGGREGATE data itself contain?

Per-window contraction of the binned-density deviation from the invariant
density:  Lambda_data(k) = -ln(||v_{k+1}-v_inf|| / ||v_k-v_inf||)/dt.
This is the identifiable functional: any exactly-fitting recovered operator
reproduces it.  Question: does it approach Lambda=0.72 in any honest window?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "MultiClusterPhysics"))
sys.path.insert(0, str(HERE.parents[2] / "FindProbability"))

import rapidity_ensemble as re                                        # noqa: E402
from recover_physics_gap import Ensemble, edges_for                   # noqa: E402

cfg = re.EnsembleConfig(m_electrons=40000)
ens = Ensemble(cfg)
Lam = ens.Lambda
print(f"Lambda analytic = {Lam:.4f}")

for n in (8, 24):
    e = edges_for(cfg, n)
    for dt in (0.6, 1.2):
        times = 0.4 + dt * np.arange(int(14.0 / dt))
        vinf = np.concatenate(ens.frac_snapshot(20.0, e, n))
        vs = [np.concatenate(ens.frac_snapshot(t, e, n)) for t in times]
        dev = [np.linalg.norm(v - vinf) for v in vs]
        print(f"\n n={n} dt={dt}: per-window Lambda_data (t0 : ||dev|| -> rate)")
        for k in range(len(times) - 1):
            if dev[k] < 1e-12 or dev[k + 1] < 1e-12:
                r = float("nan")
            else:
                r = -np.log(dev[k + 1] / dev[k]) / dt
            print(f"   t0={times[k]:5.2f}  |d|={dev[k]:.3e}  "
                  f"Lam={r:7.3f}  ratio={r/Lam:6.2f}")
