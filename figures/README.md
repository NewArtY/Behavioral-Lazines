# Figures (`ris1`–`ris4`)

Regenerates the four figures used in the article, bilingual (Russian / English),
as PNG (300 dpi) + PDF. Pure `numpy` + `matplotlib`; no other dependencies.

| File | Article | What it shows | Data source |
|------|---------|---------------|-------------|
| `ris1` | Fig. 1 | Optimal transport parametrises the prior: implied self-retention vs. entropic regularisation ε (Sinkhorn) — small ε ⇒ high retention (least-action end). | `../baselines.py` (computed live) |
| `ris2` | Fig. 2 | Labour market (ОКВЭД-2): (а) reliability η₂ by regulariser, (б) balance residual (null ≈ 10⁻¹⁵ vs. EWMA ≈ 1.2·10⁻¹). | hard-coded from Table `okved2` (`main.tex`) |
| `ris3` | Fig. 3 | Physics bridge: (а) mean target retention P** 0.114 → 1.000 (lazy prior = attractor), (б) balance exactness (null vs. EWMA). | hard-coded from Table `phys` (`main.tex`) |
| `ris4` | Fig. 4 | Applicability map: error-ratio heatmaps over drift δ × jitter ρ — (а) null-space smoothing is safe everywhere, (б) EWMA has a harm zone. | `../experiments/phase_null_ratio.csv`, `phase_ewma_ratio.csv` (structure-preserving map, `phase_map_struct.py`) |

## Regenerate

```bash
python make_figures.py ru     # Russian labels (default)
python make_figures.py en     # English labels
python make_figures.py ru --out /some/dir   # force an explicit destination
```

## Dependencies

- `numpy`, `matplotlib` (see `../requirements-dev.txt`; matplotlib is only needed
  for figure generation — the numerical core is numpy-only).
- `ris4` additionally needs the phase-map CSVs
  `../experiments/phase_null_ratio.csv` and `phase_ewma_ratio.csv` (the canonical,
  structure-preserving map). They are committed; if missing, regenerate them with
  `python ../experiments/phase_map_struct.py` (needs the `python_forecast`/
  `python_solver` package). The self-contained `../phase_map.py` instead writes the
  *abstract* illustration to `*.abstract.csv`. `make_figures` prints a clear message
  and skips `ris4` rather than crashing.

## Output location

- If the sibling `article/` tree exists (the in-repository authoring workflow),
  figures are written to `article/figures/` (ru) or `article/figures_en/` (en).
- In a standalone clone of `code/` only (no `article/`), they fall back to
  `figures/out/ru/` or `figures/out/en/`.
- `--out PATH` (or `--out=PATH`) overrides both.
