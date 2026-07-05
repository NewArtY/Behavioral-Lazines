# -*- coding: utf-8 -*-
"""Full rerun of the labour-market experiments on the CURATED OKVED-2 data
(2017-2024, 8 snapshots -> 5 holdouts 2020-2024; author's workbook ranges,
convention note of 2026-07-02).

Produces every number the paper's Tables 1/2/4 and Sec. 5-6 text cite:
  A. rolling-origin regulariser table + balance residuals (Table 1);
  B. baselines + scale-free metrics + the full significance battery with an
     explicit Holm correction (Table 2 + text);
  C. retention invariance, static-vs-dynamic ablation (Table 4),
     energy<->skill correlations, solver-grid robustness (Sec. 6 text).

Run (from code/experiments/):  python rerun_curated_2024.py
Output is duplicated into rerun_curated_2024.out (UTF-8).
"""
from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINDPROB = HERE.parents[2] / "FindProbability"
sys.path.insert(0, str(FINDPROB))
sys.path.insert(0, str(HERE))

from python_forecast import _bootstrap  # noqa: E402,F401
from python_forecast.scripts.explore_laziness import (  # noqa: E402
    rolling_sweep, energy_diagnostic,
)
from python_forecast.scripts.explore_baselines import rolling_baselines  # noqa: E402
from python_forecast.scripts.explore_strengthening import (  # noqa: E402
    retention_invariance, static_vs_dynamic, energy_skill, robustness,
)

import econ_significance as ES  # noqa: E402  (article battery, sibling module)

DOMAIN = "okved2curated"


def significance_battery(domain=DOMAIN):
    """econ_significance's battery, domain-parameterised, with explicit Holm."""
    from scipy import stats
    D = ES.collect(domain)
    n = D["n"]
    G = len(D["labels"])
    print("=" * 78)
    print(f"  {domain} significance: {G} holdouts {D['labels']}, "
          f"{2 * n + 1} indicators each")
    print("=" * 78)

    print(f"\n  rolling-origin means (scheme={ES.SCHEME}); reliab tau=1/2/5%:")
    print(f"    {'method':>11} {'MAE':>7} {'sMAPE%':>7} {'MASE':>6} "
          f"{'rel@1':>6} {'rel@2':>6} {'rel@5':>6} {'dir':>6}")
    for m in D["methods"]:
        print(f"    {m:>11} {np.mean(D['per_mae'][m]):>7.2f} "
              f"{np.mean(D['per_smape'][m]):>7.2f} {np.mean(D['per_mase'][m]):>6.2f} "
              f"{np.mean(D['per_rel1'][m]):>6.3f} {np.mean(D['per_rel'][m]):>6.3f} "
              f"{np.mean(D['per_rel5'][m]):>6.3f} {np.mean(D['per_dir'][m]):>6.3f}")

    pvals = {}   # (family, method) -> raw two-sided p

    print(f"\n  [DM] Diebold-Mariano (HLN) on holdout MAE, H={G} (exploratory):")
    for m in D["methods"]:
        if m == "naive":
            continue
        stat, p, H = ES.dm_hln(D["per_mae"][m], D["per_mae"]["naive"])
        print(f"    naive vs {m:>11}: DM={stat:+.3f}  p={p:.3f}")

    print(f"\n  [Permutation] within-holdout sign-flip on |err| diff (naive-method):")
    for m in D["methods"]:
        if m == "naive":
            continue
        base = np.array(D["abs_err"]["naive"]); cur = np.array(D["abs_err"][m])
        obs, p = ES.cluster_permutation(base - cur, D["hold_idx"][m])
        pvals[("perm", m)] = p
        print(f"    naive - {m:>11}: d|err|={obs:+.2f}  p={p:.3f}  "
              f"({'method better' if obs > 0 else 'naive better'})")

    print(f"\n  [PT] Pesaran-Timmermann + binomial (pooled moved indicators):")
    for m in D["methods"]:
        hits = np.array(D["dir_hit"][m], float)
        if hits.size == 0:
            continue
        k, N = int(hits.sum()), hits.size
        pb = stats.binomtest(k, N, 0.5, alternative="greater").pvalue
        pt, ppt, _, P = ES.pesaran_timmermann(hits, D["dir_pred"][m], D["dir_act"][m])
        if m != "naive" and pt == pt:
            pvals[("PT", m)] = ppt
        pt_s = f"PT={pt:+.2f} p={ppt:.3f}" if pt == pt else "PT=n/a"
        print(f"    {m:>11}: hit={hits.mean():.3f} (k={k}/{N})  "
              f"binom p={pb:.3f}  {pt_s}")

    print(f"\n  [Bootstrap] cluster (by holdout, G={G}) 95% CI, null - naive:")
    dm_, lo, hi = ES.cluster_bootstrap_diff(
        D["abs_err"]["null"], D["abs_err"]["naive"], D["hold_idx"]["null"])
    print(f"    d|err|  (null-naive): {dm_:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  "
          f"({'sig' if hi < 0 else 'n.s.'})")
    rel_null = np.array(D["per_rel"]["null"]); rel_nv = np.array(D["per_rel"]["naive"])
    dr, lor, hir = ES.cluster_bootstrap_diff(rel_null, rel_nv, np.arange(rel_null.size))
    print(f"    d reliab (null-naive): {dr:+.3f}  CI[{lor:+.3f},{hir:+.3f}]  "
          f"({'sig' if lor > 0 else 'n.s.'})")

    # Holm-Bonferroni over the confirmatory families (perm + PT), all methods.
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    M = len(items)
    print(f"\n  [Holm] {M} raw p-values (perm + PT families):")
    any_sig = False
    for i, ((fam, m), p) in enumerate(items):
        thr = 0.05 / (M - i)
        sig = p <= thr
        if not sig:
            print(f"    {fam:>5}/{m:<11} p={p:.4f}  thr={thr:.4f}  -> n.s. "
                  f"(and all below stop)")
            break
        any_sig = True
        print(f"    {fam:>5}/{m:<11} p={p:.4f}  thr={thr:.4f}  -> SIGNIFICANT")
    print(f"    smallest raw p = {items[0][1]:.4f} ({items[0][0]})  "
          f"any significant after Holm: {any_sig}")
    return D


def main():
    t0 = time.perf_counter()
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        print(f"### rerun on {DOMAIN}: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "#" * 78 + "\n# A. regulariser sweep + balance residuals\n" + "#" * 78)
        energy_diagnostic(DOMAIN)
        rolling_sweep(DOMAIN)
        rolling_baselines(DOMAIN)
        print("\n" + "#" * 78 + "\n# B. baselines + significance battery\n" + "#" * 78)
        significance_battery(DOMAIN)
        print("\n" + "#" * 78 + "\n# C. levers / ablation / correlation / robustness\n" + "#" * 78)
        retention_invariance(DOMAIN)
        static_vs_dynamic(DOMAIN)
        energy_skill(DOMAIN)
        robustness(DOMAIN)
        print(f"\n[time] total: {time.perf_counter() - t0:.1f} s")

    (HERE / "rerun_curated_2024.out").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
